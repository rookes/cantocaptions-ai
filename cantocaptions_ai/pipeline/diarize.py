"""Speaker diarization via pyannote's speaker-diarization-community-1 pipeline.

This stage answers only "who spoke when". Turning those turns into segment labels is a
separate, model-free concern and lives in ``pipeline/speaker_assign.py``.

Runs after alignment, so the speaker turns can be mapped onto the over-split subsegments that
cue assembly is about to glue back together -- see ``pipeline/segmentation.py``, which refuses
to merge across a confident speaker change.

Two scopes, as sibling strategies over a shared base (the same shape as the VAD and ASR
backends):

- ``SegmentDiarization`` (default) diarizes each VAD segment independently. Speaker identity
  is then only meaningful *within* a segment, so labels are namespaced with the segment id
  (``S0007/SPEAKER_00``) and the merge gate ignores comparisons across that boundary. This
  matches what the gate actually asks -- "is the voice in these two adjacent clauses the same
  one?" -- which is a local question that global clustering can only get wrong.
- ``FileDiarization`` diarizes the whole file in one pass, giving globally consistent speaker
  identities at the cost of every clustering error being spread across the episode.
"""

import logging
from typing import List, Optional

import numpy as np
import torch

from cantocaptions_ai.pipeline.speaker_assign import scoped_speaker, segment_scope_id
from cantocaptions_ai.utils.audio import SAMPLE_RATE, load_audio, resolve_device
from cantocaptions_ai.utils.debug import load_diarization_debug, write_diarization_debug
from cantocaptions_ai.utils.log_utils import get_logger
from cantocaptions_ai.utils.model_utils import (
    MemoryPolicy,
    PipelineStage,
    flush_vram,
    partition_by_cache,
    vram_stats,
)
from cantocaptions_ai.utils.schema import (
    DiarizationResult,
    ProgressCallback,
    SpeakerTurn,
    VadAudioSegment,
)

logger = get_logger(__name__)

DEFAULT_DIARIZE_MODEL = "pyannote/speaker-diarization-community-1"

DIARIZE_SCOPES = ("segment", "file")

# Below this a VAD segment carries too little speech for the segmentation model's sliding
# window to say anything useful, and asking anyway invites a spurious second speaker. Such
# segments are left undiarized, which reads downstream as "unknown" (merge-permissive).
MIN_SEGMENT_DURATION = 0.5


def _to_turns(annotation, *, offset: float = 0.0, scope: Optional[str] = None) -> List[SpeakerTurn]:
    """Flatten a ``pyannote.core.Annotation`` into sorted plain-dict speaker turns.

    ``offset`` shifts segment-relative times back onto the file timeline; ``scope`` namespaces
    the speaker labels so identities from different segments can never be confused for each
    other (see ``speaker_assign.speaker_scope``).
    """
    turns: List[SpeakerTurn] = [
        {
            "start": round(segment.start + offset, 3),
            "end": round(segment.end + offset, 3),
            "speaker": scoped_speaker(scope, speaker) if scope else speaker,
        }
        for segment, _, speaker in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda turn: (turn["start"], turn["end"]))
    return turns


class _BaseDiarization(PipelineStage):
    """Shared plumbing for the two diarization scopes: the model, the checkpoint, the carrier.

    Subclasses supply ``_extract`` (what slice of the item they need) and ``process`` (how
    they drive the model over it).
    """

    scope: str

    def __init__(
        self,
        pipeline,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        return_embeddings: bool = False,
        device_index: Optional[int] = None,
    ):
        self.pipeline = pipeline
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.return_embeddings = return_embeddings
        self.device_index = device_index

    @staticmethod
    def read_debug(audio_path, debug_dir): return load_diarization_debug(audio_path, debug_dir)

    @staticmethod
    def write_debug(audio_path, result, debug_dir): write_diarization_debug(audio_path, result, debug_dir)

    @staticmethod
    def _pack(item, result): return {**item, 'diarization': result}

    def _annotate(self, waveform: np.ndarray):
        """Run the pipeline over one waveform and return pyannote's ``DiarizeOutput``.

        Audio is handed over in memory rather than by path so the caller controls exactly
        what is diarized -- the chosen audio track, or a single vocal-isolated VAD segment.
        """
        file = {
            "waveform": torch.from_numpy(waveform).unsqueeze(0),
            "sample_rate": SAMPLE_RATE,
        }
        return self.pipeline(
            file, min_speakers=self.min_speakers, max_speakers=self.max_speakers
        )


class FileDiarization(_BaseDiarization):
    """Diarize a whole file in one pass: globally consistent speakers, global failure modes."""

    scope = "file"

    @staticmethod
    def _extract(item): return load_audio(item['audio_path'], audio_track=item.get('audio_track', 0))

    def process(
        self, input: np.ndarray, *, progress_callback: ProgressCallback = None
    ) -> DiarizationResult:
        """Diarize one file's waveform.

        Progress stays at the base class's one-unit-per-file granularity: pyannote's per-batch
        hook only learns its batch count once the model is already running, and
        ``StageTimer._start_determinate`` replaces the bar on every ``set_total`` (see the same
        note in ``vocal_isolation.run``), so a per-file total would reset the bar each file.
        """
        output = self._annotate(input)

        # Exclusive diarization drops overlapping speech, which is what segment attribution
        # wants: a subsegment should be credited to whoever actually carries it, not to both
        # parties of a moment of crosstalk. The overlapping version is kept for debugging.
        speakers = sorted(output.speaker_diarization.labels())
        result: DiarizationResult = {
            "scope": self.scope,
            "speakers": speakers,
            "turns": _to_turns(output.exclusive_speaker_diarization),
            "overlap_turns": _to_turns(output.speaker_diarization),
        }

        if self.return_embeddings and output.speaker_embeddings is not None:
            # speaker_embeddings rows follow speaker_diarization.labels() order.
            result["embeddings"] = {
                speaker: output.speaker_embeddings[index].tolist()
                for index, speaker in enumerate(output.speaker_diarization.labels())
            }

        logger.info(f"Diarization found {len(speakers)} speaker(s): {', '.join(speakers)}")
        return result


class SegmentDiarization(_BaseDiarization):
    """Diarize each VAD segment independently, namespacing speakers by segment id.

    The merge gate only ever compares *adjacent* cues, so it only needs to know whether two
    neighbouring clauses share a voice. That is a local question; answering it locally avoids
    inheriting the errors of clustering a whole episode into a fixed speaker set. The cost is
    that identities are not comparable between segments, which the ``S0007/`` prefix makes
    explicit and ``segmentation._same_speaker`` honours by declining to compare across it.
    """

    scope = "segment"

    def __init__(self, *args, flush_every: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.flush_every = flush_every

    def _release(self, position: int, index: int) -> None:
        """Trace VRAM at debug level, and optionally return cached blocks to the driver.

        ``flush_every`` defaults to 0 (off), matching BatchExecutor. Measurement showed live
        allocation flat at 42 MB across segments -- diarization does not accumulate between
        calls, so there is nothing for a flush to reclaim. What breaks it is the *in-call*
        peak, which is governed by --diarize_batch_size, not by flushing.
        """
        if self.flush_every and position % self.flush_every == 0:
            flush_vram()
        if logger.isEnabledFor(logging.DEBUG):
            stats = vram_stats(self.device_index)
            if stats is not None:
                # peak is the number that matters: a segment fails not by leaking but by its
                # in-call high-water mark exceeding the card. free/total come from
                # torch.cuda.mem_get_info, which on Windows/WDDM does NOT agree with Task
                # Manager's dedicated-memory figure -- the driver can page an oversubscribed
                # allocation into host RAM while CUDA still reports memory free -- so read
                # peak, not free, when judging whether a segment fits.
                logger.debug(
                    "Diarization segment %d: VRAM live %.0f MB, peak %.0f MB, reserved %.0f MB "
                    "(mem_get_info says %.0f MB free of %.0f MB; unreliable on Windows)",
                    index, stats['allocated_mb'], torch.cuda.max_memory_allocated(
                        self.device_index if self.device_index is not None else 0
                    ) / 1e6,
                    stats['reserved_mb'], stats['free_mb'], stats['total_mb'],
                )
            torch.cuda.reset_peak_memory_stats(
                self.device_index if self.device_index is not None else 0
            )

    @staticmethod
    def _extract(item) -> List[VadAudioSegment]:
        segments = item.get('vad_segments')
        if segments is None:
            raise RuntimeError(
                "--diarize_scope segment needs the VAD segments, which are not available for "
                f"'{item['audio_path']}' (this happens under --retime). "
                "Use --diarize_scope file instead."
            )
        return segments

    def process(
        self, input: List[VadAudioSegment], *, progress_callback: ProgressCallback = None
    ) -> DiarizationResult:
        """Diarize each VAD segment and merge the results onto the file timeline."""
        turns: List[SpeakerTurn] = []
        overlap_turns: List[SpeakerTurn] = []
        speakers: List[str] = []
        breakdown = []
        skipped = 0

        # Longest segment first. Every segment is a separate model invocation over a
        # differently shaped tensor, so in input order the caching allocator meets a longer
        # segment than it has ever seen again and again, and its reserved pool ratchets up
        # instead of being reused. Front-loading the biggest one sizes the pool once, up
        # front (the same reasoning as BatchExecutor.order_key). Results are keyed by index
        # and re-sorted below, so processing order does not affect output.
        order = sorted(range(len(input)), key=lambda i: len(input[i]["audio"]), reverse=True)

        for position, index in enumerate(order):
            segment = input[index]
            duration = segment["end"] - segment["start"]
            if duration < MIN_SEGMENT_DURATION:
                skipped += 1
            else:
                scope = segment_scope_id(index)
                output = self._annotate(segment["audio"])
                self._release(position, index)
                local = sorted(output.speaker_diarization.labels())
                speakers.extend(scoped_speaker(scope, label) for label in local)
                # Segment audio starts at t=0; shift back onto the file timeline.
                turns.extend(_to_turns(
                    output.exclusive_speaker_diarization, offset=segment["start"], scope=scope
                ))
                overlap_turns.extend(_to_turns(
                    output.speaker_diarization, offset=segment["start"], scope=scope
                ))
                breakdown.append({
                    "index": index,
                    "start": round(segment["start"], 3),
                    "end": round(segment["end"], 3),
                    "scope": scope,
                    "speakers": local,
                })
            if progress_callback is not None:
                progress_callback.advance(1)

        turns.sort(key=lambda turn: (turn["start"], turn["end"]))
        overlap_turns.sort(key=lambda turn: (turn["start"], turn["end"]))
        breakdown.sort(key=lambda entry: entry["index"])

        multi = sum(1 for entry in breakdown if len(entry["speakers"]) > 1)
        logger.info(
            "Diarization: %d/%d VAD segments diarized (%d skipped as shorter than %.2gs), "
            "%d with more than one speaker",
            len(breakdown), len(input), skipped, MIN_SEGMENT_DURATION, multi,
        )
        return {
            "scope": self.scope,
            "speakers": sorted(speakers),
            "turns": turns,
            "overlap_turns": overlap_turns,
            "segment_speakers": breakdown,
        }

    def run(self, items, *, debug_dir=None, load_debug_dir=None, progress_callback=None):
        """Override the per-file bar with a per-segment one.

        Unlike whole-file diarization, the unit count is knowable before the model runs, so a
        single ``set_total`` up front spans every file (``StageTimer._start_determinate``
        closes and replaces the bar on each call, so it must only be called once).
        """
        cached, to_compute = partition_by_cache(items, self.read_debug, load_debug_dir)

        if progress_callback is not None:
            progress_callback.set_total(
                sum(len(self._extract(item)) for _, item in to_compute), unit="seg"
            )

        results = dict(cached)
        for index, item in to_compute:
            result = self.process(self._extract(item), progress_callback=progress_callback)
            if debug_dir:
                self.write_debug(item['audio_path'], result, debug_dir)
            results[index] = result

        return [self._pack(item, results[i]) for i, item in enumerate(items)]


_SCOPES = {cls.scope: cls for cls in (SegmentDiarization, FileDiarization)}


def load_diarization_cache(items: List[dict], debug_dir: str) -> List[dict]:
    """Attach every item's cached diarization without loading a model.

    Scope-independent: the checkpoint records which scope produced it, and reading it back
    needs neither the model nor the scope's own logic.
    """
    return _BaseDiarization.load_cache(items, debug_dir)


def load_diarization(
    device: str = "cpu",
    device_index: int = 0,
    model_name: str = DEFAULT_DIARIZE_MODEL,
    token: Optional[str] = None,
    model_dir: Optional[str] = None,
    scope: str = "segment",
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    return_embeddings: bool = False,
    batch_size: Optional[int] = None,
    vram_checks: bool = True,
    vram_headroom_mb: int = 512,
) -> _BaseDiarization:
    """Load the diarization pipeline and return the processor for the requested scope."""
    try:
        processor_cls = _SCOPES[scope]
    except KeyError:
        raise ValueError(
            f"Unknown diarize_scope {scope!r}; choose from {sorted(_SCOPES)}"
        ) from None

    # Import from pyannote.audio.core.pipeline rather than pyannote.audio.pipelines: the
    # package __init__ eagerly imports speaker_verification, which probes for NeMo,
    # speechbrain and onnxruntime. Same reasoning as vads/pyannote.py.
    from pyannote.audio.core.pipeline import Pipeline

    logger.info(f"Loading diarization model: {model_name} (scope: {scope})")
    pipeline = Pipeline.from_pretrained(model_name, token=token, cache_dir=model_dir)
    if pipeline is None:
        raise RuntimeError(
            f"Could not load diarization pipeline '{model_name}'. It is a gated model: accept "
            "its terms on huggingface.co and pass a token via --hf_token."
        )
    # The checkpoint's own config sets both batch sizes (32 for community-1), which is
    # tuned for one pass over a whole file. Under segment scope the model runs once per VAD
    # segment, and it is that per-batch peak -- not the total audio -- that has to fit; lower
    # it here when it does not.
    if batch_size is not None:
        pipeline.segmentation_batch_size = batch_size
        pipeline.embedding_batch_size = batch_size
        logger.info(f"Diarization batch size set to {batch_size}")

    pipeline.to(torch.device(resolve_device(device, device_index)))

    # Cap the allocator once the weights are resident, exactly as the ASR backend does. This
    # stage runs the model once per VAD segment, so without a cap a near-OOM does not raise
    # on Windows -- the driver pages into host RAM instead and the stage crawls.
    if device == "cuda":
        MemoryPolicy(vram_checks, vram_headroom_mb).cap_after_load(device_index)

    return processor_cls(
        pipeline=pipeline,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        return_embeddings=return_embeddings,
        device_index=device_index if device == "cuda" else None,
    )

"""The pipeline's stages: what runs between reading the audio and writing the subtitles.

``transcribe._execute_pipeline`` drives these in order. Each helper here is one stage's work
(or one step of it), kept apart from the CLI and config plumbing in ``transcribe.py``, which
re-exports the names other code imports from there.
"""
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from cantocaptions_ai.errors import ConfigError
from cantocaptions_ai.text_profiles import DEFAULT_PUNCTUATION
from cantocaptions_ai.utils.debug import write_speaker_assignment_debug
from cantocaptions_ai.utils.log_utils import ProgressSink, StageTimer, TranscriptionSummary, get_logger
from cantocaptions_ai.utils.model_utils import flush_vram, load_with_offline_fallback
from cantocaptions_ai.utils.schema import (
    AlignedTranscriptionResult,
    ProcessingItem,
    ProgressCallback,
    VadItem,
    item_name,
)

logger = get_logger(__name__)


_VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.ts', '.m2ts'}


def _select_audio_track(path: str, language: Optional[str] = "yue",
                        override: Optional[int] = None) -> int:
    """Return the 0-based audio stream index to use for *path*.

    ``override`` (--audio_track) wins outright. Otherwise video files are probed with
    ffprobe and the stream most likely to carry *language* is chosen (see
    utils.audio.select_track); audio-only files, or a probe that finds nothing, get 0
    (ffmpeg's default).
    """
    if override is not None:
        return override
    ext = os.path.splitext(path)[1].lower()
    if ext not in _VIDEO_EXTENSIONS:
        return 0
    from cantocaptions_ai.languages import get_language_pack
    from cantocaptions_ai.utils.audio import probe_audio_tracks
    streams = probe_audio_tracks(path)
    if not streams:
        logger.warning("No audio streams found via ffprobe for '%s'; using default track", path)
        return 0
    track = get_language_pack(language).choose_track(streams)
    if track != 0:
        logger.info("Selected audio track index %d (%s) for '%s'", track, language, path)
    return track


def _substitution_overrides(cfg):
    """Hand-curated character substitutions for the align model, or None."""
    if not getattr(cfg, "align_substitutions", None):
        return None
    from cantocaptions_ai.pipeline.align_vocab import load_substitution_overrides
    return load_substitution_overrides(cfg.align_substitutions)


def _run_alignment(
    items: List[ProcessingItem],
    align_model,
    align_metadata,
    device: str,
    align_padding: float,
    align_release: float,
    interpolate_method: str,
    return_char_alignments: bool,
    print_progress: bool,
    batch_size: int,
    progress_callback: ProgressCallback = None,
    vram_checks: bool = True,
    spotchecks=None,
    punctuation=DEFAULT_PUNCTUATION,
    split_gap: Optional[float] = None,
    script=None,
) -> List[ProcessingItem]:
    from cantocaptions_ai.pipeline.alignment import align
    if progress_callback is not None:
        progress_callback.set_total(sum(len(it['result']['segments']) for it in items), unit="seg")
    aligned_items = []
    for item in items:
        result = item['result']
        # Popped, not read: the file's emissions are ~0.5 GB an hour of audio, and the caller's
        # list still holds this item, so leaving the timeline on it would keep every file's
        # alive through the rest of the batch and every later stage.
        timeline = item.pop('emission_timeline', None)
        if align_model is not None and len(result["segments"]) > 0:
            logger.info("Performing alignment...")
            aligned_result: AlignedTranscriptionResult = align(
                result["segments"],
                align_model,
                align_metadata,
                item['vad_segments'],
                device,
                align_padding=align_padding,
                align_release=align_release,
                interpolate_method=interpolate_method,
                return_char_alignments=return_char_alignments,
                print_progress=print_progress,
                batch_size=batch_size,
                progress_callback=progress_callback,
                vram_checks=vram_checks,
                spotchecks=spotchecks,
                punctuation=punctuation,
                timeline=timeline,
                split_gap=split_gap, script=script,
            )
            aligned_result['language'] = result['language']
        else:
            aligned_result = result
        # Keep the rest of the carrier (notably audio_track, chosen by _select_audio_track):
        # diarization loads the audio again downstream and must not fall back to track 0.
        aligned_items.append({**item, 'result': aligned_result})
    return aligned_items


def _extract_timestamps(items: list) -> List[ProcessingItem]:
    """Segment timings without alignment (used when no_align=True).

    Uses the ASR's per-character timestamps where a backend provides them, and otherwise
    the segment's own span -- the VAD chunk it was decoded from. Neither Qwen backend emits
    ``time_stamps``, so the fallback is the normal path; requiring them made --no_align
    fail with a KeyError on every run.
    """
    extracted_items = []
    for item in items:
        result = item['result']
        segments = []
        for segment in result['segments']:
            stamps = segment.get('time_stamps')
            s_start = stamps[0]['start'] if stamps else segment['start']
            s_end = stamps[-1]['end'] if stamps else segment['end']
            # Cue assembly merges on `words`; an unaligned segment simply has none.
            segments.append({'words': [], **segment, 'start': s_start, 'end': s_end})
        # Preserve the rest of the carrier (audio_track, vad_segments): diarization
        # needs both, and --no_align routes through here instead of _run_alignment.
        extracted_items.append({**item, 'result': {**result, 'segments': segments}})
    return extracted_items


def _run_diarization(
    items: List[ProcessingItem],
    cfg,
    summary: TranscriptionSummary,
    progress: Optional[ProgressSink],
    need_diarize: bool,
) -> List[ProcessingItem]:
    """Stage 5: attach ``diarization`` to every item, from the model or the debug cache.

    Held in its own helper so the stage's load/run/free cycle reads as one unit and
    ``_execute_pipeline`` stays an outline of the pipeline rather than an implementation.
    """
    from cantocaptions_ai.pipeline.diarize import load_diarization, load_diarization_cache

    if not need_diarize:
        return load_diarization_cache(items, cfg.load_debug_dir)

    if cfg.hf_token is None:
        logger.info(
            "No --hf_token provided; %s is a gated model, so its terms must already be "
            "accepted and a token cached (huggingface-cli login) or this load will fail.",
            cfg.diarize_model,
        )
    with StageTimer("Diarization", summary, progress=progress) as stage:
        diarizer = load_with_offline_fallback(
            load_diarization,
            device=cfg.device,
            device_index=cfg.device_index,
            model_name=cfg.diarize_model,
            token=cfg.hf_token,
            model_dir=cfg.model_dir,
            scope=cfg.diarize_scope,
            min_speakers=cfg.min_speakers,
            max_speakers=cfg.max_speakers,
            return_embeddings=cfg.speaker_embeddings,
            batch_size=cfg.diarize_batch_size,
            vram_checks=cfg.vram_checks,
            vram_headroom_mb=cfg.vram_headroom_mb,
        )
        stage.mark_inference_start()
        items = diarizer.run(
            items,
            debug_dir=cfg.debug_dir,
            load_debug_dir=cfg.load_debug_dir,
            progress_callback=stage.reporter,
        )
    del diarizer
    flush_vram()
    return items


def _assign_speakers(
    items: List[ProcessingItem],
    cfg,
    debug_dir: Optional[str] = None,
) -> List[ProcessingItem]:
    """Label each aligned subsegment with a speaker, where diarization is confident enough.

    Runs outside the diarization StageTimer: it is pure arithmetic over already-computed
    turns, and it deliberately re-runs on a --load_debug_dir replay so threshold changes
    take effect without re-diarizing.
    """
    from cantocaptions_ai.pipeline.speaker_assign import (
        SpeakerAssignmentConfig,
        assign_speakers,
        format_stats,
    )

    config = SpeakerAssignmentConfig(
        min_dominant_share=cfg.speaker_confidence,
        conflict_share=cfg.speaker_conflict_share,
        flag_conflicts=cfg.flag_speaker_conflicts,
    )
    for item in items:
        diarization = item.get('diarization')
        if diarization is None:
            continue
        segments = item['result']["segments"]
        stats = assign_speakers(segments, diarization["turns"], config)
        logger.info(f"Speaker assignment: {format_stats(stats)}")
        if debug_dir is not None:
            write_speaker_assignment_debug(item_name(item), segments, debug_dir)
    return items


def _load_realign_checkpoint(item: dict, realign_path: str, load_debug_dir: Optional[str]):
    """Cached line placements for *item*, or None if absent or made under other settings."""
    from cantocaptions_ai.utils.checkpoints import checkpoint_is_current
    from cantocaptions_ai.utils.debug import load_realign_debug

    if not load_debug_dir:
        return None
    name = item_name(item)
    settings = (item.get("checkpoints") or {}).get("realign")
    if not checkpoint_is_current(load_debug_dir, name, "realign", settings):
        return None
    return load_realign_debug(name, realign_path, load_debug_dir)


def _write_realign_checkpoint(item: dict, realign_path: str, timings, lines, debug_dir) -> None:
    from cantocaptions_ai.utils.checkpoints import write_checkpoint_meta
    from cantocaptions_ai.utils.debug import write_realign_debug

    if debug_dir is None:
        return
    name = item_name(item)
    write_realign_debug(name, realign_path, timings, lines, debug_dir)
    settings = (item.get("checkpoints") or {}).get("realign")
    if settings is not None:
        write_checkpoint_meta(debug_dir, name, "realign", settings)


def _run_realign(
    items: List[VadItem],
    realign_path: str,
    align_model,
    align_metadata,
    device: str,
    *,
    chunk_size: float,
    window_seconds: float,
    commit_margin: float,
    min_score: float,
    mode: str = "transcript",
    cut_policy: str = "drop",
    max_scale: float = 0.25,
    adjust_tolerance: float = 2.0,
    align_padding: float = 0.04,
    sync_anchor_density: float = 3.0,
    normalize: bool = True,
    batch_size: int = 4,
    vram_checks: bool = True,
    debug_dir: Optional[str] = None,
    load_debug_dir: Optional[str] = None,
    punctuation=None,
) -> List[ProcessingItem]:
    """Put a transcript on the timeline and build the next stage's input from it.

    In ``transcript`` and ``adjust`` modes, returns items whose ``vad_segments`` have been
    *replaced* by chunks re-cut onto the gaps between placed lines, and whose ``result`` holds
    one segment per chunk carrying that chunk's lines and their cue_spans. Dropping the coarse
    chunks here also frees their audio, which for a feature-length file is a few hundred MB.

    In ``sync`` mode the cues are already finished -- the transform placed them and nothing
    downstream is allowed to move them -- so the item carries them directly and is marked
    ``realign_sync`` so the caller skips alignment entirely.
    """
    from cantocaptions_ai.pipeline.align_profiles import DEFAULT_ALIGN_PROFILE
    from cantocaptions_ai.pipeline.alignment import compute_vad_emissions
    from cantocaptions_ai.pipeline.realign import (
        EmissionTimeline, assign_lines, assign_lines_adjust, assign_lines_sync,
        build_align_input, load_transcript_lines, segments_from_timings, warn_low_confidence,
    )
    from cantocaptions_ai.utils.debug import write_realign_transform

    timed = mode in ("sync", "adjust")
    lines = load_transcript_lines(realign_path, keep_timings=timed, normalize=normalize)
    # The language's realign punctuation (realign.realign_punctuation); the searches
    # default to the CJK one when none is given.
    punct = {"punctuation": punctuation} if punctuation is not None else {}
    if not normalize:
        logger.info(
            "realign: punctuation normalization is off, so the text reaches the subtitle "
            "exactly as written. Halfwidth marks are not in the align vocabulary and will be "
            "dropped, so the pauses they stand for go unmodelled."
        )
    logger.info(f"Loaded {len(lines)} transcript line(s) from: {realign_path}")
    if not lines:
        raise ConfigError(f"realign transcript is empty: {realign_path}")
    if len(items) > 1:
        logger.warning(
            "One transcript is being realigned against %d audio files; --realign takes a "
            "transcript of a single recording.", len(items),
        )

    # Give the dictionary a token for the transcript's unknown characters *before* the
    # coarse search reads it, not after: a line whose every character is invisible to the
    # trellis can only be placed from its neighbours. Alignment runs the same call later and
    # finds nothing left to do, which is what keeps the two passes agreeing.
    repair = align_metadata.get("vocab_repair")
    if repair is not None:
        repair.augment(line.text for line in lines)

    profile = align_metadata.get("profile") or DEFAULT_ALIGN_PROFILE
    result_items: List[ProcessingItem] = []
    for item in items:
        vad_segments = item["vad_segments"]

        def compute(segments, _model=align_model):
            return compute_vad_emissions(
                segments, _model, align_metadata["type"], align_metadata["processor"], device,
                batch_size, vram_checks=vram_checks, primer=profile.primer,
                min_samples=profile.min_samples, dtype=EmissionTimeline.dtype,
            )

        # One timeline for the whole file, built here and handed to the alignment stage
        # below: the placements and the final alignment read the same emissions, so the
        # encoder runs once over the file instead of once per pass.
        # TODO: run realign and alignment per file. Every file's timeline is built here before
        # any is aligned, so a batch peaks at the sum of them (~0.5 GB an hour of audio each);
        # _run_alignment frees each one only after the whole batch has been realigned.
        timeline = EmissionTimeline(
            vad_segments, compute, frame_rate=align_metadata.get("frame_rate"))

        common = dict(
            window_seconds=window_seconds, commit_margin=commit_margin,
            max_scale_dev=max_scale, cut_policy=cut_policy, **punct,
        )
        dropped: frozenset = frozenset()
        transform = None
        if mode == "sync":
            timings, dropped, transform, report = assign_lines_sync(
                lines, vad_segments, timeline,
                align_metadata["dictionary"], align_metadata["language"],
                align_padding=align_padding, target_anchors_per_minute=sync_anchor_density,
                **common,
            )
        elif mode == "adjust":
            timings, dropped, transform, report = assign_lines_adjust(
                lines, vad_segments, timeline,
                align_metadata["dictionary"], align_metadata["language"],
                tolerance=adjust_tolerance, **common,
            )
        else:
            # The checkpoint holds coarse placements indexed by line number, so it is only
            # reusable for the mode that produced it. sync and adjust re-fit instead, which
            # is cheap next to the encoder pass they both still need.
            timings = _load_realign_checkpoint(item, realign_path, load_debug_dir)
            if timings is None:
                timings = assign_lines(
                    lines, vad_segments, timeline,
                    align_metadata["dictionary"], align_metadata["language"],
                    window_seconds=window_seconds, commit_margin=commit_margin, **punct,
                )
                _write_realign_checkpoint(item, realign_path, timings, lines, debug_dir)

        warn_low_confidence(timings, lines, min_score)
        if transform is not None and debug_dir is not None:
            write_realign_transform(
                item_name(item), realign_path, mode, lines, timings, transform, report,
                dropped, debug_dir,
            )

        if mode == "sync":
            # Nothing downstream may re-time these: the whole contract of sync is that the
            # subtitle's own proportions survive, and forced alignment would undo that.
            segments = segments_from_timings(lines, timings, dropped)
            result = {"segments": segments, "language": align_metadata["language"]}
            result_items.append({**item, "result": result, "realign_sync": True})
            continue

        kept = [line for line in lines if line.index not in dropped]
        kept_timings = [t for t in timings if t.index not in dropped]
        chunks, transcript = build_align_input(
            kept, kept_timings, vad_segments, chunk_size,
        )
        result = {"segments": transcript, "language": align_metadata["language"]}
        result_items.append({
            **item, "vad_segments": chunks, "result": result,
            "emission_timeline": timeline,
        })
    return result_items


def _run_realign_asr(
    items: List[ProcessingItem],
    realign_path: str,
    *,
    chunk_size: float,
    normalize: bool = True,
    debug_dir: Optional[str] = None,
    load_debug_dir: Optional[str] = None,
    punctuation=None,
) -> List[ProcessingItem]:
    """Time an untimed transcript against the ASR hypothesis (--realign_anchor asr).

    Same output shape as _run_realign, so stage 4 alignment onwards is identical; only the
    way each line's approximate position was found differs.
    """
    from cantocaptions_ai.pipeline.realign import (
        assign_lines_via_asr, build_align_input, load_transcript_lines,
    )

    lines = load_transcript_lines(realign_path, normalize=normalize)
    logger.info(f"Loaded {len(lines)} transcript line(s) from: {realign_path}")
    if not lines:
        raise ConfigError(f"realign transcript is empty: {realign_path}")

    out: List[ProcessingItem] = []
    for item in items:
        timings = _load_realign_checkpoint(item, realign_path, load_debug_dir)
        if timings is None:
            timings = assign_lines_via_asr(
                lines, item["result"]["segments"], vad_segments=item["vad_segments"],
                **({"punctuation": punctuation} if punctuation is not None else {}),
            )
            _write_realign_checkpoint(item, realign_path, timings, lines, debug_dir)
        chunks, transcript = build_align_input(
            lines, timings, item["vad_segments"], chunk_size,
        )
        result = {**item["result"], "segments": transcript}
        out.append({**item, "vad_segments": chunks, "result": result})
    return out


def _pre_align_clean(items: List[ProcessingItem], cleaner) -> List[ProcessingItem]:
    """Apply the cleaning manifest's ``pre_align`` steps to each ASR segment's text.

    Runs after the ASR (and any correction) cache loads rather than inside the ASR
    backend, so a --load_debug_dir replay picks up rule edits without re-running ASR.
    Punctuation added here is what alignment splits sentences on, so a clause comma
    such as 嘅話， can end one cue and start the next instead of trailing on a line.
    """
    changed = 0
    for item in items:
        segments = []
        for seg in item['result']['segments']:
            text = cleaner.pre_align(seg['text'])
            if text != seg['text']:
                changed += 1
                seg = {**seg, 'text': text}
            segments.append(seg)
        item['result']['segments'] = segments
    if changed:
        logger.info("Pre-alignment cleaning: updated %d segment(s)", changed)
    return items


# =========================================================================================
# The stage list
# =========================================================================================
#
# ``_execute_pipeline`` builds a RunContext, asks ``build_stages`` for this run's stages, and
# runs them in order over the item list; cue assembly and writing (``_merge_and_write``) follow.
# A stage reads everything it needs from the context. A model stage that has a debug checkpoint
# is a CachedStage: it computes when any input lacks a current checkpoint under
# --load_debug_dir, and otherwise loads them all from there. That check is made just before the
# stage runs; a stage's checkpoint is only ever written by the stage itself, so it is the same
# answer the whole run would have given up front.


@dataclass
class RunContext:
    """What one run's stages share: the config, the language, and the per-run state that
    ``_execute_pipeline`` sets up before the first stage."""

    cfg: Any
    audio_paths: List[str]
    name_of: Dict[str, str]
    checkpoints: Dict[str, Any]
    pack: Any
    profile: Any
    align_model_name: Optional[str]
    summary: TranscriptionSummary
    progress: Optional[ProgressSink] = None
    display_paths: Optional[dict] = None
    # A preloaded VAD model to reuse (PipelineService's resident mode).
    vad_model: Any = None
    qwen_threads: int = 1
    reference_cues: Optional[list] = None
    realign_mode: Optional[str] = None
    realign_punct: Any = None
    cleaner: Any = None
    layout: Any = None

    @property
    def align_language(self) -> str:
        return self.cfg.language

    @property
    def realign_acoustic(self) -> bool:
        """--realign with the acoustic anchor: the align model places the transcript, and the
        ASR-to-alignment run of stages is replaced by one."""
        return bool(self.cfg.realign) and self.cfg.realign_anchor == "acoustic"

    @property
    def vocal_isolation_active(self) -> bool:
        method = self.cfg.vocal_isolation_method
        return bool(method) and method.lower() != "none"

    def cached(self, path: str, stage: str) -> bool:
        """Whether *path* has a current *stage* checkpoint under --load_debug_dir."""
        from cantocaptions_ai.utils.checkpoints import checkpoint_is_current
        from cantocaptions_ai.utils.debug import _debug_stage_exists

        name = self.name_of[path]
        return (
            _debug_stage_exists(name, stage, self.cfg.load_debug_dir)
            and checkpoint_is_current(self.cfg.load_debug_dir, name, stage, self.checkpoints[stage])
        )

    def all_cached(self, stage: str, paths: Optional[Sequence[str]] = None) -> bool:
        """Whether every input (or every one of *paths*) can be read back from the cache."""
        paths = self.audio_paths if paths is None else paths
        return bool(self.cfg.load_debug_dir) and all(self.cached(p, stage) for p in paths)

    def isolation_cached(self) -> List[bool]:
        """Per input: a current vocal isolation checkpoint, which makes its VAD dead weight."""
        return [
            self.vocal_isolation_active and bool(self.cfg.load_debug_dir)
            and self.cached(p, "vocal_isolation")
            for p in self.audio_paths
        ]


class Stage:
    """One step of the pipeline. ``run`` takes and returns the item list.

    ``key`` is the stage's stable, language-neutral id (what a UI translates by and a
    progress record stores); ``name`` is its English label, the one its StageTimer and so
    ``ProgressSink.stage_start`` use. ``timed`` says the stage runs under a StageTimer, and
    so reports progress, rather than finishing in an instant.
    """

    name: str = ""
    key: str = ""
    timed: bool = False

    def active(self, ctx: RunContext) -> bool:
        """Whether this stage is part of the run at all."""
        return True

    def run(self, ctx: RunContext, items: List[dict]) -> List[dict]:
        raise NotImplementedError

    def status(self, ctx: RunContext) -> Optional[str]:
        """``"compute"`` or ``"cached"`` for a stage with a checkpoint; None otherwise."""
        return None

    def timed_in(self, ctx: RunContext) -> bool:
        """Whether this run of the stage will show progress (a cached load shows none)."""
        return self.timed

    def describe(self, ctx: RunContext) -> str:
        """The stage as the run's plan line shows it."""
        status = self.status(ctx)
        return f"{self.name} [{status}]" if status else self.name


class CachedStage(Stage):
    """A model stage with a debug checkpoint: compute under a StageTimer, or load the cache."""

    checkpoint: str = ""
    timed = True

    def needs_compute(self, ctx: RunContext) -> bool:
        return not ctx.all_cached(self.checkpoint)

    def status(self, ctx: RunContext) -> Optional[str]:
        return "compute" if self.needs_compute(ctx) else "cached"

    def timed_in(self, ctx: RunContext) -> bool:
        return self.needs_compute(ctx)

    def run(self, ctx: RunContext, items: List[dict]) -> List[dict]:
        if self.needs_compute(ctx):
            with StageTimer(self.name, ctx.summary, progress=ctx.progress) as timer:
                return self.compute(ctx, items, timer)
        return self.from_cache(ctx, items)

    def compute(self, ctx: RunContext, items: List[dict], timer) -> List[dict]:
        raise NotImplementedError

    def from_cache(self, ctx: RunContext, items: List[dict]) -> List[dict]:
        raise NotImplementedError


def _stage_run(processor, ctx: RunContext, items, timer):
    """A PipelineStage's run(), with the run's debug dirs and the timer's progress reporter."""
    cfg = ctx.cfg
    return processor.run(items, debug_dir=cfg.debug_dir, load_debug_dir=cfg.load_debug_dir,
                         progress_callback=timer.reporter)


# --- Stage 1: VAD -------------------------------------------------------------------------

class VadStage(Stage):
    """Files covered by a cached vocal isolation checkpoint are held back entirely: they
    enter stage 2 as bare carriers and get their vad_segments from the isolation cache."""

    name = "VAD"
    key = "vad"

    def _vad_paths(self, ctx: RunContext) -> List[str]:
        return [p for p, cached in zip(ctx.audio_paths, ctx.isolation_cached()) if not cached]

    def status(self, ctx: RunContext) -> Optional[str]:
        paths = self._vad_paths(ctx)
        return "compute" if paths and not ctx.all_cached("vad", paths) else "cached"

    def timed_in(self, ctx: RunContext) -> bool:
        # Its timer also covers loading VAD's own cache, for every file isolation did not.
        return bool(self._vad_paths(ctx))

    def run(self, ctx: RunContext, items: List[dict]) -> List[dict]:
        from cantocaptions_ai.pipeline.vad import VadProcessor

        cfg = ctx.cfg
        audio_paths = ctx.audio_paths
        isolation_cached = ctx.isolation_cached()
        vad_indices = [i for i, cached in enumerate(isolation_cached) if not cached]
        need_vad = any(
            not cfg.load_debug_dir
            or not ctx.cached(audio_paths[i], "vad")
            for i in vad_indices
        )
        if len(vad_indices) < len(audio_paths):
            logger.info(
                "Skipping VAD for %d of %d file(s) already covered by cached vocal isolation",
                len(audio_paths) - len(vad_indices), len(audio_paths),
            )
        if vad_indices:
            with StageTimer("VAD", ctx.summary, progress=ctx.progress) as stage:
                vad_items = [
                    {
                        'audio_path': audio_paths[i],
                        'name': ctx.name_of[audio_paths[i]],
                        'checkpoints': ctx.checkpoints,
                        # A clip's temp WAV holds only the track already chosen for it.
                        'audio_track': _select_audio_track(
                            audio_paths[i], cfg.language,
                            None if audio_paths[i] in (ctx.display_paths or {}) else cfg.audio_track,
                        ),
                        'audio_downmix': cfg.audio_downmix,
                        'audio_normalize': cfg.audio_normalize,
                    }
                    for i in vad_indices
                ]
                if need_vad:
                    from cantocaptions_ai.pipeline.vad import load_vad
                    # A caller (e.g. PipelineService in resident mode) may pass a preloaded
                    # VAD model to reuse across jobs — load_vad reuses it and ignores
                    # vad_method. It stays alive via the caller's reference after the
                    # processor wrapper is dropped below, skipping the ~20-30s reload.
                    vad_processor = load_vad(
                        vad_method=cfg.vad_method,
                        device=cfg.device,
                        device_index=cfg.device_index,
                        vad_onset=cfg.vad_onset,
                        vad_offset=cfg.vad_offset,
                        vad_pad_onset=cfg.vad_pad_onset,
                        vad_pad_offset=cfg.vad_pad_offset,
                        vad_min_duration_off=cfg.vad_min_duration_off,
                        chunk_size=cfg.chunk_size,
                        vad_model=ctx.vad_model,
                        use_auth_token=cfg.hf_token,
                        reference_cues=(
                            ctx.reference_cues
                            if cfg.asr_context and cfg.asr_context_vad_expand
                            else None
                        ),
                        reference_padding=cfg.asr_context_padding,
                        # --realign holds a transcript line for every utterance, including ones
                        # VAD scores below threshold, so segmentation may only choose cut points
                        # -- it may not decide what to keep. See Vad.cover_chunks. This applies
                        # under both anchors: the 'asr' anchor pays for transcribing the whole
                        # file rather than just its speech, but in exchange the chunks it hands
                        # to alignment cover the audio with no gaps (and carry vocal isolation
                        # throughout), which is what the final chunk re-cut assumes.
                        cover_all=bool(cfg.realign),
                    )
                    stage.mark_inference_start()
                    vad_out = vad_processor.run(vad_items, debug_dir=cfg.debug_dir, load_debug_dir=cfg.load_debug_dir, progress_callback=stage.reporter)
                    del vad_processor
                else:
                    vad_out = VadProcessor.load_cache(vad_items, cfg.load_debug_dir)
            for i, out in zip(vad_indices, vad_out):
                items[i] = out
            flush_vram()
        return items


# --- Stage 2: vocal isolation -------------------------------------------------------------

class VocalIsolationStage(CachedStage):
    name = "Vocal isolation"
    key = "vocal_isolation"
    checkpoint = "vocal_isolation"

    def active(self, ctx):
        return ctx.vocal_isolation_active

    def run(self, ctx, items):
        items = super().run(ctx, items)
        flush_vram()
        return items

    def compute(self, ctx, items, timer):
        from cantocaptions_ai.pipeline.vocal_isolation import load_vocal_isolation

        cfg = ctx.cfg
        processor = load_with_offline_fallback(
            load_vocal_isolation,
            model_name=cfg.vocal_isolation_method,
            device=cfg.device,
            device_index=cfg.device_index,
            batch_size=cfg.vocal_isolation_batch_size,
            compute_type=cfg.vocal_isolation_compute_type,
            vram_checks=cfg.vram_checks,
            model_dir=cfg.model_dir,
            local_files_only=cfg.model_cache_only,
            segment_mode=cfg.vocal_isolation_segment_mode,
        )
        timer.mark_inference_start()
        items = _stage_run(processor, ctx, items, timer)
        del processor
        return items

    def from_cache(self, ctx, items):
        # All files' isolated audio is cached: load it so downstream ASR sees the
        # isolated (not raw) audio even if ASR itself is being recomputed.
        from cantocaptions_ai.pipeline.vocal_isolation import MbRoformerProcessor
        return MbRoformerProcessor.load_cache(items, ctx.cfg.load_debug_dir)


# --- Realign, acoustic anchor: replaces ASR through alignment ------------------------------

class RealignAcousticStage(Stage):
    """The transcript is known and complete, only its timings are missing. The alignment
    model does both jobs -- a coarse sliding search for where each line sits, then forced
    alignment for the timings within a line. The 'asr' anchor takes the ordinary ASR path
    instead and rejoins at alignment."""

    name = "Transcript realignment"
    key = "realign"
    timed = True

    def active(self, ctx):
        return ctx.realign_acoustic

    def run(self, ctx, items):
        from cantocaptions_ai.pipeline.alignment import load_align_model
        from cantocaptions_ai.pipeline.realign import (
            ensure_visible_cues, strip_sentinels, tighten_cue_spans,
        )

        cfg = ctx.cfg
        with StageTimer(self.name, ctx.summary, progress=ctx.progress) as stage:
            align_model, align_metadata = load_with_offline_fallback(
                load_align_model,
                ctx.align_language, cfg.device, cfg.device_index,
                model_name=ctx.align_model_name, model_dir=cfg.model_dir, model_cache_only=cfg.model_cache_only,
                compute_type=cfg.align_compute_type,
                vram_checks=cfg.vram_checks,
                char_substitution=cfg.align_char_substitution,
                substitution_overrides=_substitution_overrides(cfg),
            )
            stage.mark_inference_start()
            items = _run_realign(
                items, cfg.realign, align_model, align_metadata, cfg.device,
                chunk_size=cfg.chunk_size,
                window_seconds=cfg.realign_window,
                commit_margin=cfg.realign_commit_margin,
                min_score=cfg.realign_min_score,
                mode=ctx.realign_mode,
                cut_policy=cfg.realign_cut_policy,
                max_scale=cfg.realign_max_scale,
                adjust_tolerance=cfg.realign_adjust_tolerance,
                align_padding=cfg.align_padding,
                sync_anchor_density=cfg.realign_sync_anchor_density,
                normalize=cfg.realign_normalize,
                batch_size=cfg.align_batch_size,
                vram_checks=cfg.vram_checks,
                debug_dir=cfg.debug_dir,
                load_debug_dir=cfg.load_debug_dir,
                punctuation=ctx.realign_punct,
            )
            # Mode sync's cues are finished: the transform placed them, and the guarantee it
            # makes -- that the subtitle's own proportions survive exactly -- is only true if
            # nothing re-times them afterwards. So alignment and its fixups are skipped
            # rather than run and then overridden.
            pending = [item for item in items if not item.get("realign_sync")]
            if pending:
                aligned = _run_alignment(
                    pending, align_model, align_metadata, cfg.device,
                    cfg.align_padding, cfg.align_release, cfg.interpolate_method,
                    cfg.return_char_alignments, cfg.print_progress, cfg.align_batch_size,
                    progress_callback=stage.reporter,
                    vram_checks=cfg.vram_checks,
                    spotchecks=ctx.profile.spotchecks,
                    # 0, not cfg.align_split_gap: under --realign the transcript's line
                    # breaks are the cue boundaries and a cue is one whole line by contract,
                    # so nothing here may break one in two -- not even an align profile that
                    # asked for it on the ASR path.
                    split_gap=0,
                    script=ctx.profile.script,
                    # Not profile.punctuation: realign needs the space, the newline and the
                    # line sentinel to be pause tokens, and declares its cue boundaries
                    # through cue_spans rather than letting punctuation derive them.
                    punctuation=ctx.realign_punct,
                )
                by_path = {item["audio_path"]: item for item in aligned}
                items = [by_path.get(item["audio_path"], item) for item in items]
            for item in items:
                if item.get("realign_sync"):
                    # No words and no sentinels to tidy; only the two validity guarantees.
                    ensure_visible_cues(item["result"]["segments"])
                    continue
                segments = item["result"]["segments"]
                tighten_cue_spans(segments)
                ensure_visible_cues(segments)
                strip_sentinels(segments)
        del align_model, align_metadata
        flush_vram()
        return items


class _AsrPath(Stage):
    """A stage of the ordinary ASR path, which the acoustic realign anchor replaces."""

    def active(self, ctx):
        return not ctx.realign_acoustic


class _CachedAsrPath(CachedStage):
    def active(self, ctx):
        return not ctx.realign_acoustic


# --- ASR context ---------------------------------------------------------------------------

class AsrContextStage(_AsrPath):
    """Attach each VAD segment's reference-subtitle context prompt.

    Here rather than at VAD time: both the VAD and vocal isolation debug round-trips rebuild
    segment dicts from scratch and would drop the key, and this point is downstream of both
    cache loads. Contexts are cheap and always re-derived, so --asr_context_template edits
    take effect on replay.
    """

    name = "ASR context"
    key = "asr_context"

    def active(self, ctx):
        return super().active(ctx) and bool(ctx.cfg.asr_context and ctx.reference_cues)

    def run(self, ctx, items):
        from cantocaptions_ai.pipeline.reference_context import build_segment_contexts

        cfg = ctx.cfg
        for item in items:
            spans = None
            if cfg.asr_context_scope == "expanded":
                # Provenance recorded at VAD time and carried through the debug
                # manifests; absent means this segment is entirely VAD's own find.
                spans = [
                    sp for seg in item['vad_segments'] for sp in seg.get('expanded', ())
                ]
            contexts = build_segment_contexts(
                item['vad_segments'],
                ctx.reference_cues,
                neighbours=cfg.asr_context_neighbours,
                template=cfg.asr_context_template,
                max_chars=cfg.asr_context_max_chars,
                restrict_to_spans=spans,
            )
            item['vad_segments'] = [
                {**seg, 'context': context}
                for seg, context in zip(item['vad_segments'], contexts)
            ]
            if cfg.asr_context_scope == "expanded" and cfg.asr_context_template != "none":
                logger.info(
                    "ASR context: scope 'expanded' -- %d of %d segment(s) carry a "
                    "context over reference-recovered audio; the rest decode bare",
                    sum(1 for c in contexts if c), len(contexts),
                )
            elif cfg.asr_context_template == "none":
                logger.info(
                    "ASR context: template 'none' -- %d segment(s) decode without a "
                    "context prompt; the reference subtitle affected the VAD "
                    "timeline only", len(contexts),
                )
            else:
                logger.info(
                    "ASR context: %d of %d segment(s) biased by the reference subtitle",
                    sum(1 for c in contexts if c), len(contexts),
                )
        return items


# --- Stage 3: transcription, and the optional second opinions ------------------------------

class TranscriptionStage(_CachedAsrPath):
    name = "Transcription"
    key = "transcription"
    checkpoint = "transcription"

    def compute(self, ctx, items, timer):
        from cantocaptions_ai.pipeline.asr import load_model
        from cantocaptions_ai.utils.model_utils import model_scope

        cfg = ctx.cfg
        with model_scope(
            load_model,
            ctx.profile.model,
            normalization=ctx.profile.normalization,
            device=cfg.device,
            device_index=cfg.device_index,
            download_root=cfg.model_dir,
            compute_type=cfg.asr_compute_type,
            attn_implementation=cfg.attn_implementation,
            language=cfg.language,
            local_files_only=cfg.model_cache_only,
            threads=ctx.qwen_threads,
            use_auth_token=cfg.hf_token,
            batch_size=cfg.batch_size,
            compile_enabled=cfg.compile,
            print_progress=cfg.print_progress,
            verbose=cfg.verbose,
            vram_checks=cfg.vram_checks,
            vram_headroom_mb=cfg.vram_headroom_mb,
        ) as model:
            timer.mark_inference_start()
            items = _stage_run(model, ctx, items, timer)
            # Drop this frame's reference so model_scope's exit frees the model; a
            # name bound by `as` outlives the block and held the ASR model on the
            # GPU through alignment and diarization.
            del model
        return items

    def from_cache(self, ctx, items):
        from cantocaptions_ai.pipeline.asr import AsrStage
        return AsrStage.load_cache(items, ctx.cfg.load_debug_dir)


class EnsembleStage(_CachedAsrPath):
    name = "Ensemble ASR (faster-whisper)"
    key = "ensemble"
    checkpoint = "ensemble"

    def active(self, ctx):
        return super().active(ctx) and ctx.cfg.ensemble_model != "none"

    def compute(self, ctx, items, timer):
        from cantocaptions_ai.pipeline.ensemble import load_faster_whisper
        from cantocaptions_ai.utils.model_utils import model_scope

        cfg = ctx.cfg
        with model_scope(
            load_faster_whisper,
            device=cfg.device,
            device_index=cfg.device_index,
            model_dir=cfg.model_dir,
            local_files_only=cfg.model_cache_only,
            language=cfg.language,
        ) as ensemble:
            timer.mark_inference_start()
            items = _stage_run(ensemble, ctx, items, timer)
            del ensemble  # see TranscriptionStage
        return items

    def from_cache(self, ctx, items):
        from cantocaptions_ai.pipeline.ensemble import FasterWhisperEnsemble
        return FasterWhisperEnsemble.load_cache(items, ctx.cfg.load_debug_dir)


class ReferenceMatchStage(_AsrPath):
    """Pair each ASR segment with the reference subtitle's text, for LLM correction."""

    name = "Reference subtitle matching"
    key = "reference_match"
    timed = True

    def active(self, ctx):
        return super().active(ctx) and bool(ctx.cfg.llm_correction and ctx.reference_cues)

    def run(self, ctx, items):
        from cantocaptions_ai.pipeline.llm_correction import match_reference_to_segments

        with StageTimer(self.name, ctx.summary, progress=ctx.progress):
            for item in items:
                item['reference_texts'] = match_reference_to_segments(
                    item['result']['segments'], ctx.reference_cues
                )
        return items


class LlmCorrectionStage(_CachedAsrPath):
    name = "LLM correction"
    key = "llm_correction"
    checkpoint = "llm_correction"

    def active(self, ctx):
        return super().active(ctx) and bool(ctx.cfg.llm_correction)

    def run(self, ctx, items):
        if self.needs_compute(ctx) and ctx.cfg.vram_checks:
            from cantocaptions_ai.utils.model_utils import vram_stats
            stats = vram_stats()
            if stats:
                logger.info(
                    f"VRAM before LLM load: allocated={stats['allocated_mb']:.0f} MB, "
                    f"reserved={stats['reserved_mb']:.0f} MB, "
                    f"free={stats['free_mb']:.0f} MB / {stats['total_mb']:.0f} MB"
                )
        return super().run(ctx, items)

    def compute(self, ctx, items, timer):
        from cantocaptions_ai.pipeline.llm_correction import load_llm
        from cantocaptions_ai.utils.model_utils import model_scope

        cfg = ctx.cfg
        with model_scope(
            load_llm,
            model_id=cfg.llm_model,
            model_dir=cfg.llm_model_dir,
            device=cfg.device,
            local_files_only=cfg.model_cache_only,
            semantic_mode=cfg.reference_correction_semantic,
            attn_implementation=cfg.attn_implementation,
            vram_checks=cfg.vram_checks,
            language=cfg.language,
        ) as corrector:
            timer.mark_inference_start()
            items = _stage_run(corrector, ctx, items, timer)
            del corrector  # see TranscriptionStage
        return items

    def from_cache(self, ctx, items):
        from cantocaptions_ai.pipeline.llm_correction import LLMCorrector
        return LLMCorrector.load_cache(items, ctx.cfg.load_debug_dir)


# --- Realign, ASR anchor ------------------------------------------------------------------

class RealignAsrStage(_AsrPath):
    """The transcript replaces the ASR hypothesis here: ASR ran only to say *where* each line
    is, and from this point on the pipeline is identical to the acoustic anchor."""

    name = "Transcript matching"
    key = "realign_match"
    timed = True

    def active(self, ctx):
        return super().active(ctx) and bool(ctx.cfg.realign)

    def run(self, ctx, items):
        cfg = ctx.cfg
        with StageTimer(self.name, ctx.summary, progress=ctx.progress):
            return _run_realign_asr(
                items, cfg.realign, chunk_size=cfg.chunk_size,
                normalize=cfg.realign_normalize,
                debug_dir=cfg.debug_dir, load_debug_dir=cfg.load_debug_dir,
                punctuation=ctx.realign_punct,
            )


class PreAlignCleanStage(_AsrPath):
    """The cleaning manifest's pre_align steps. Not under --realign: the transcript's
    cue_spans index its text, so inserting a character would shift every span after it --
    and that text is the user's anyway."""

    name = "Pre-alignment cleaning"
    key = "pre_align_clean"

    def active(self, ctx):
        return super().active(ctx) and ctx.cleaner is not None and not ctx.cfg.realign

    def run(self, ctx, items):
        return _pre_align_clean(items, ctx.cleaner)


# --- Stage 4: alignment (or, under --no_align, the ASR's own timings) ----------------------

class AlignmentStage(_AsrPath):
    name = "Alignment"
    key = "alignment"
    timed = True

    def active(self, ctx):
        return super().active(ctx) and not ctx.cfg.no_align

    def run(self, ctx, items):
        from cantocaptions_ai.pipeline.alignment import load_align_model

        cfg = ctx.cfg
        with StageTimer(self.name, ctx.summary, progress=ctx.progress) as stage:
            align_model, align_metadata = load_with_offline_fallback(
                load_align_model,
                ctx.align_language, cfg.device, cfg.device_index,
                model_name=ctx.align_model_name, model_dir=cfg.model_dir, model_cache_only=cfg.model_cache_only,
                compute_type=cfg.align_compute_type,
                vram_checks=cfg.vram_checks,
                char_substitution=cfg.align_char_substitution,
                substitution_overrides=_substitution_overrides(cfg),
            )
            stage.mark_inference_start()
            items = _run_alignment(
                items, align_model, align_metadata, cfg.device,
                cfg.align_padding, cfg.align_release, cfg.interpolate_method,
                cfg.return_char_alignments, cfg.print_progress, cfg.align_batch_size,
                progress_callback=stage.reporter,
                vram_checks=cfg.vram_checks,
                spotchecks=ctx.profile.spotchecks,
                punctuation=(
                    ctx.realign_punct if cfg.realign else ctx.profile.punctuation
                ),
                # See RealignAcousticStage for why this is forced off under --realign.
                split_gap=0 if cfg.realign else cfg.align_split_gap,
                script=ctx.profile.script,
            )
            if cfg.realign:
                from cantocaptions_ai.pipeline.realign import (
                    enforce_cue_order, ensure_visible_cues, strip_sentinels,
                    tighten_cue_spans, warn_on_implausible_cues,
                )
                for item in items:
                    segments = item["result"]["segments"]
                    tighten_cue_spans(segments)
                    # Order first: ensure_visible_cues reads the previous cue's end as
                    # its floor, which is only meaningful once the cues are in order.
                    enforce_cue_order(segments)
                    ensure_visible_cues(segments)
                    strip_sentinels(segments)
                    # Last: the cue text has to be final before its span can be judged
                    # against what that text could have been spoken in.
                    warn_on_implausible_cues(segments)
        del align_model, align_metadata
        flush_vram()
        return items


class TimestampsStage(_AsrPath):
    name = "ASR timestamps (no alignment)"
    key = "timestamps"

    def active(self, ctx):
        return super().active(ctx) and bool(ctx.cfg.no_align)

    def run(self, ctx, items):
        return _extract_timestamps(items)


# --- Stage 5: diarization -----------------------------------------------------------------

class DiarizationStage(CachedStage):
    """After alignment, so speaker turns land on the over-split subsegments that cue
    assembly is about to merge back together, which is what lets segmentation._same_speaker
    veto a merge across a speaker change."""

    name = "Diarization"
    key = "diarization"
    checkpoint = "diarization"

    def active(self, ctx):
        return bool(ctx.cfg.diarize)

    def run(self, ctx, items):
        return _run_diarization(items, ctx.cfg, ctx.summary, ctx.progress,
                                self.needs_compute(ctx))


class SpeakerAssignStage(Stage):
    name = "Speaker assignment"
    key = "speaker_assign"

    def active(self, ctx):
        return bool(ctx.cfg.diarize)

    def run(self, ctx, items):
        return _assign_speakers(items, ctx.cfg, debug_dir=ctx.cfg.debug_dir)


# --- The list -----------------------------------------------------------------------------

DEFAULT_STAGES = (
    VadStage, VocalIsolationStage, RealignAcousticStage, AsrContextStage, TranscriptionStage,
    EnsembleStage, ReferenceMatchStage, LlmCorrectionStage, RealignAsrStage,
    PreAlignCleanStage, AlignmentStage, TimestampsStage, DiarizationStage, SpeakerAssignStage,
)


def build_stages(ctx: RunContext) -> List[Stage]:
    """This run's stages, in order: every default stage that is active for its config."""
    return [stage for stage in (cls() for cls in DEFAULT_STAGES) if stage.active(ctx)]


def plan_entries(ctx: RunContext, stages: Sequence[Stage]) -> List[Dict[str, Any]]:
    """The run's plan as data, for ``ProgressSink.plan``: one entry per stage, in order.

    ``key`` and ``label`` as on Stage (a stage of the caller's own with no key gets its
    class name); ``timed`` whether it will report progress this run; ``cached`` True/False
    for a stage with a checkpoint (read back, or computed), None for one without.
    """
    out = []
    for stage in stages:
        status = stage.status(ctx)
        out.append({
            "key": stage.key or type(stage).__name__,
            "label": stage.name,
            "timed": stage.timed_in(ctx),
            "cached": None if status is None else status == "cached",
        })
    return out


def describe_plan(ctx: RunContext, stages: Sequence[Stage]) -> str:
    """One line saying what the run will do, e.g. ``VAD [cached] → Transcription [compute]``."""
    return " → ".join(stage.describe(ctx) for stage in stages)


def run_stages(ctx: RunContext, stages: Sequence[Stage], items: List[dict]) -> List[dict]:
    for stage in stages:
        items = stage.run(ctx, items)
    return items

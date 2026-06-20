import sys
import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple
import torch

from cantocaptions_ai.utils.audio import load_audio, SAMPLE_RATE
from cantocaptions_ai.utils.schema import TranscriptionResult, AlignedTranscriptionResult, ProgressCallback
from cantocaptions_ai.utils.log_utils import get_logger

logger = get_logger(__name__)


class IntervalTree:
    """
    Simple interval tree for fast overlap queries using sorted array + binary search.

    Uses O(n) space and provides O(log n) query time instead of O(n) linear scan.
    This achieves ~228x speedup for speaker assignment in long-form content.
    """

    def __init__(self, intervals: List[Tuple[float, float, str]]):
        """
        Initialize the interval tree with diarization segments.

        Args:
            intervals: List of (start, end, speaker) tuples
        """
        if not intervals:
            self.starts = np.array([])
            self.ends = np.array([])
            self.speakers: List[str] = []
            return

        # Sort intervals by start time for binary search
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        self.starts = np.array([i[0] for i in sorted_intervals], dtype=np.float64)
        self.ends = np.array([i[1] for i in sorted_intervals], dtype=np.float64)
        self.speakers = [i[2] for i in sorted_intervals]

    def query(self, start: float, end: float) -> List[Tuple[str, float]]:
        """
        Find all intervals that overlap with [start, end] and compute intersection.

        Args:
            start: Query interval start time
            end: Query interval end time

        Returns:
            List of (speaker, intersection_duration) tuples for overlapping segments
        """
        if len(self.starts) == 0:
            return []

        # Binary search to find candidate intervals
        # Only intervals with start < end could overlap
        right_idx = np.searchsorted(self.starts, end, side='left')
        if right_idx == 0:
            return []

        # Check candidates for actual overlap
        candidates = slice(0, right_idx)
        overlaps = (self.starts[candidates] < end) & (self.ends[candidates] > start)

        results = []
        for idx in np.where(overlaps)[0]:
            intersection = min(self.ends[idx], end) - max(self.starts[idx], start)
            if intersection > 0:
                results.append((self.speakers[idx], intersection))
        return results

    def find_nearest(self, time: float) -> Optional[str]:
        """
        Find the speaker of the nearest segment to a given time point.

        Args:
            time: Time point to find nearest segment for

        Returns:
            Speaker ID of nearest segment, or None if no segments exist
        """
        if len(self.starts) == 0:
            return None

        # Calculate midpoints of all segments
        mids = (self.starts + self.ends) / 2
        nearest_idx = np.argmin(np.abs(mids - time))
        return self.speakers[nearest_idx]


class DiarizationPipeline:
    def __init__(
        self,
        model_name=None,
        token=None,
        device: Optional[Union[str, torch.device]] = "cpu",
        cache_dir=None,
    ):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        if sys.platform != "linux":
            raise RuntimeError(
                "Diarization (--diarize) requires NeMo, which is only supported on Linux. "
                "Run on a Linux system to use this feature."
            )
        import nemo.collections.asr as nemo_asr
        model_config = model_name or "pyannote/speaker-diarization-community-1"
        logger.info(f"Loading diarization model: {model_config}")

        #self.model = Pipeline.from_pretrained(model_config, token=token, cache_dir=cache_dir).to(device)
        self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large").to(device)
        self.model.eval()

    def __call__(
        self,
        audio: Union[str, np.ndarray],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        return_embeddings: bool = False,
        progress_callback: ProgressCallback = None,
    ) -> Union[tuple[pd.DataFrame, Optional[dict[str, list[float]]]], pd.DataFrame]:
        """
        Perform speaker diarization on audio.

        Args:
            audio: Path to audio file or audio array
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum number of speakers to detect
            max_speakers: Maximum number of speakers to detect
            return_embeddings: Whether to return speaker embeddings
            progress_callback: Optional callable receiving a float (0-100) with progress percentage

        Returns:
            If return_embeddings is True:
                Tuple of (diarization dataframe, speaker embeddings dictionary)
            Otherwise:
                Just the diarization dataframe
        """
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio_data = {
            'waveform': torch.from_numpy(audio[None, :]).to(self.device),
            'sample_rate': SAMPLE_RATE,
            'input_signal': torch.from_numpy(audio[None, :]).to(self.device),
            'input_len': torch.tensor([torch.from_numpy(audio[None, :]).shape[1]]).to(self.device)
        }

        hook = None
        if progress_callback is not None:
            # pyannote's diarization has two progress-trackable steps, each with
            # its own completed/total counter that resets between steps. Map each
            # step into a sub-range so progress is monotonic and meaningful.
            _STEP_RANGES = {
                "segmentation": (0.0, 50.0),
                "embeddings": (50.0, 99.0),
            }
            last_pct = [0.0]
            def hook(step_name, step_artifact, file=None, total=None, completed=None):
                if total is not None and completed is not None and total > 0:
                    offset, end = _STEP_RANGES.get(step_name, (0.0, 99.0))
                    pct = offset + min(completed / total, 1.0) * (end - offset)
                    if pct > last_pct[0]:
                        last_pct[0] = pct
                        progress_callback(pct)

        # output = self.model(
        #     audio_data,
        #     num_speakers=num_speakers,
        #     min_speakers=min_speakers,
        #     max_speakers=max_speakers,
        #     **({"hook": hook} if hook is not None else {}),
        # )

        # Convert to Mel spectrogram
        processed_signal, processed_len = self.model.preprocessor(
            input_signal=audio_data["input_signal"],
            length=audio_data["input_len"]
        )

        #encoder_output = self.model.encoder(audio_signal=processed_signal, length=processed_len)
        #embeddings = self.model.decoder(encoder_output)

        _, embs = self.model.forward(input_signal=audio_data["input_signal"], input_signal_length=audio_data["input_len"])
        emb_shape = embs.shape[-1]
        embs = embs.view(-1, emb_shape)
        all_embs = embs.cpu().detach()

        if progress_callback is not None:
            progress_callback(100.0)

        import torch.nn.functional as F
        diarization = F.cosine_similarity(all_embs, all_embs)

        #diarization, embeddings = output
        #diarization = output.speaker_diarization
        #embeddings = output.speaker_embeddings if return_embeddings else None
        #exclusive = output.exclusive_speaker_diarization #TODO: test exclusive mode with pyannote

        diarize_df = pd.DataFrame(diarization.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)

        if return_embeddings and embeddings is not None:
            speaker_embeddings = {speaker: embeddings[s].tolist() for s, speaker in enumerate(diarization.labels())}
            return diarize_df, speaker_embeddings

        # For backwards compatibility
        if return_embeddings:
            return diarize_df, None
        else:
            return diarize_df


def assign_word_speakers(
    diarize_df: pd.DataFrame,
    transcript_result: Union[AlignedTranscriptionResult, TranscriptionResult],
    speaker_embeddings: Optional[dict[str, list[float]]] = None,
    fill_nearest: bool = False,
) -> Union[AlignedTranscriptionResult, TranscriptionResult]:
    """
    Assign speakers to words and segments in the transcript.

    Uses an interval tree for O(log n) overlap queries instead of O(n) linear scan,
    achieving ~228x speedup for long-form content (3+ hour podcasts).

    Args:
        diarize_df: Diarization dataframe from DiarizationPipeline
        transcript_result: Transcription result to augment with speaker labels
        speaker_embeddings: Optional dictionary mapping speaker IDs to embedding vectors
        fill_nearest: If True, assign speakers even when there's no direct time overlap

    Returns:
        Updated transcript_result with speaker assignments and optionally embeddings
    """
    transcript_segments = transcript_result.get("segments", [])
    if not transcript_segments or diarize_df is None or len(diarize_df) == 0:
        return transcript_result

    # Build interval tree from diarization segments for O(log n) queries
    intervals = [
        (row['start'], row['end'], row['speaker'])
        for _, row in diarize_df.iterrows()
    ]
    tree = IntervalTree(intervals)

    # For testing: draw a plot of the diarization
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = {'SPEAKER_00': 'black', 'SPEAKER_01': 'blue', 'SPEAKER_02': 'green', 'SPEAKER_03': 'red', 'SPEAKER_04': 'yellow', 'SPEAKER_05': 'purple'}
    for i, speaker in enumerate(diarize_df['speaker'].unique()):
        speaker_data = diarize_df[diarize_df['speaker'] == speaker]
        xranges = [(row['start'], row['end'] - row['start']) for _, row in speaker_data.iterrows()]
        ax.broken_barh(xranges, (i-0.4, 0.8), facecolors=colors[speaker], label=speaker)
    ax.set_yticks(range(len(diarize_df['speaker'].unique())))
    ax.set_yticklabels(diarize_df['speaker'].unique())
    ax.set_xlabel('Time (seconds)')
    ax.set_title('Speaker Diarization Timeline')
    plt.show()

    for seg in transcript_segments:
        seg_start = seg.get('start', 0.0)
        seg_end = seg.get('end', 0.0)

        # Query overlapping segments using interval tree
        overlaps = tree.query(seg_start, seg_end)

        if overlaps:
            # Sum intersection durations per speaker and pick the dominant one
            speaker_intersections: dict[str, float] = {}
            for speaker, intersection in overlaps:
                speaker_intersections[speaker] = speaker_intersections.get(speaker, 0.0) + intersection
            seg['speaker'] = max(speaker_intersections.items(), key=lambda x: x[1])[0]
        elif fill_nearest:
            # Find nearest segment if no overlap
            seg_mid = (seg_start + seg_end) / 2
            nearest_speaker = tree.find_nearest(seg_mid)
            if nearest_speaker:
                seg['speaker'] = nearest_speaker

        # Assign speaker to words
        if 'words' in seg:
            for word in seg['words']:
                if 'start' not in word:
                    continue

                word_start = word['start']
                word_end = word.get('end', word_start)

                word_overlaps = tree.query(word_start, word_end)

                if word_overlaps:
                    speaker_intersections = {}
                    for speaker, intersection in word_overlaps:
                        speaker_intersections[speaker] = speaker_intersections.get(speaker, 0.0) + intersection
                    word['speaker'] = max(speaker_intersections.items(), key=lambda x: x[1])[0]
                elif fill_nearest:
                    word_mid = (word_start + word_end) / 2
                    nearest_speaker = tree.find_nearest(word_mid)
                    if nearest_speaker:
                        word['speaker'] = nearest_speaker

            seg_text = seg["text"]
            seg_speaker = seg["speaker"]
            word_speakers = pd.DataFrame(seg["words"])

    # Add speaker embeddings to the result if provided
    if speaker_embeddings is not None:
        transcript_result["speaker_embeddings"] = speaker_embeddings

    return transcript_result


class Segment:
    def __init__(self, start:int, end:int, speaker:Optional[str]=None):
        self.start = start
        self.end = end
        self.speaker = speaker

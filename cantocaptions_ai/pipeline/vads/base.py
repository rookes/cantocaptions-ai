from typing import Optional

class Vad:
    def __init__(self, vad_onset):
        if not (0 < vad_onset < 1):
            raise ValueError(
                "vad_onset is a decimal value between 0 and 1."
            )

    @staticmethod
    def preprocess_audio(audio):
        pass

    @staticmethod
    def cover_chunks(segments, chunk_size, duration):
        """Partition ``[0, duration]`` into contiguous chunks of at most ``chunk_size``.

        The opposite contract to ``merge_chunks``: VAD is used only to choose *where* to cut,
        never to decide what to keep, so the emitted chunks tile the whole file with no gaps.
        Required by --realign, where a transcript line exists for speech VAD may well score
        below threshold (sung, shouted, music-bedded), and dropping that audio would leave
        forced alignment nothing to align the line against.
        """
        raise NotImplementedError(
            f"{type(segments).__name__} VAD backend does not implement cover_chunks; "
            "contiguous chunking is required by --realign."
        )

    @staticmethod
    def speech_regions(segments, **kwargs):
        """The binarized speech turns, before any grouping into chunks.

        ``merge_chunks`` groups these and ``cover_chunks`` ignores them; this returns them
        raw. --realign runs in split-only mode, so its chunks say nothing about where inside
        them anyone is speaking, and the ASR anchor needs exactly that to place a character
        stream in time. See ``realign._hypothesis_stream``.
        """
        raise NotImplementedError(
            f"{type(segments).__name__} VAD backend does not implement speech_regions."
        )

    # keep merge_chunks as static so it can be also used by manually assigned vad_model (see 'load_model')
    @staticmethod
    def merge_chunks(segments, chunk_size):
        """Group speech regions into chunks of at most ``chunk_size`` seconds.

        Each emitted chunk spans contiguously from the first grouped region's start to the
        last one's end, so silence *between* grouped regions is kept; only the silence at a
        chunk boundary is left out. Thresholds are not used here — they were already applied
        when the regions were binarized.
        """
        curr_end = 0
        merged_segments = []
        seg_idxs: list[tuple]= []

        curr_start = segments[0].start
        for seg in segments:
            if seg.end - curr_start > chunk_size and curr_end - curr_start > 0:
                merged_segments.append({
                    "start": curr_start,
                    "end": curr_end,
                    "segments": seg_idxs,
                })
                curr_start = seg.start
                seg_idxs = []
            curr_end = seg.end
            seg_idxs.append((seg.start, seg.end))

        # add final
        merged_segments.append({
            "start": curr_start,
            "end": curr_end,
            "segments": seg_idxs,
        })

        return merged_segments

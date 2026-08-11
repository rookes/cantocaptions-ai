"""Tests for the diarization stage (pipeline/diarize.py), driven by a stub pipeline.

No model is loaded: pyannote is replaced by a callable returning a hand-built Annotation, so
these cover the parts this project owns -- scoping, timeline offsetting, short-segment
skipping, processing order, and the progress contract.
"""

import types
import unittest

import numpy as np
from pyannote.core import Annotation, Segment

from cantocaptions_ai.pipeline.diarize import (
    MIN_SEGMENT_DURATION,
    FileDiarization,
    SegmentDiarization,
    load_diarization,
)

SR = 16000


def vad(start, end):
    """A VAD segment of the right length; the audio itself is never inspected by the stub."""
    return {"start": start, "end": end,
            "audio": np.zeros(int(round((end - start) * SR)), dtype=np.float32)}


def stub_pipeline(seen=None, speakers=("SPEAKER_00",)):
    """A pyannote stand-in that splits whatever it is given evenly between *speakers*."""
    def call(file, **kwargs):
        duration = tuple(file["waveform"].shape)[1] / file["sample_rate"]
        if seen is not None:
            seen.append(round(duration, 3))
        annotation = Annotation()
        step = duration / len(speakers)
        for i, speaker in enumerate(speakers):
            annotation[Segment(i * step, (i + 1) * step)] = speaker
        return types.SimpleNamespace(
            speaker_diarization=annotation,
            exclusive_speaker_diarization=annotation,
            speaker_embeddings=None,
        )
    return call


class TestSegmentScope(unittest.TestCase):
    def test_labels_are_namespaced_and_times_are_absolute(self):
        result = SegmentDiarization(stub_pipeline()).process([vad(0, 4), vad(10, 14)])
        self.assertEqual(result["scope"], "segment")
        self.assertEqual(
            [(t["start"], t["speaker"]) for t in result["turns"]],
            [(0.0, "S0000/SPEAKER_00"), (10.0, "S0001/SPEAKER_00")],
        )

    def test_short_segments_are_skipped_rather_than_guessed_at(self):
        seen = []
        short = MIN_SEGMENT_DURATION / 2
        result = SegmentDiarization(stub_pipeline(seen)).process([vad(0, short), vad(10, 14)])
        self.assertEqual(seen, [4.0], "the short segment must never reach the model")
        self.assertEqual(len(result["segment_speakers"]), 1)

    def test_processed_longest_first_without_disturbing_output(self):
        # Increasing input order is the worst case for the caching allocator: each segment is
        # larger than any seen before, so its reserved pool ratchets up instead of being
        # reused. Processing must therefore be longest-first, and output must not notice.
        seen = []
        segments = [vad(0, 2), vad(10, 16), vad(20, 24)]
        result = SegmentDiarization(stub_pipeline(seen)).process(segments)
        self.assertEqual(seen, [6.0, 4.0, 2.0], "segments must run longest-first")
        self.assertEqual([b["index"] for b in result["segment_speakers"]], [0, 1, 2])
        self.assertEqual([t["start"] for t in result["turns"]], [0.0, 10.0, 20.0])
        self.assertEqual(result["turns"][1]["speaker"], "S0001/SPEAKER_00")

    def test_multiple_speakers_within_one_segment_share_its_scope(self):
        result = SegmentDiarization(
            stub_pipeline(speakers=("SPEAKER_00", "SPEAKER_01"))
        ).process([vad(10, 14)])
        self.assertEqual(
            [t["speaker"] for t in result["turns"]],
            ["S0000/SPEAKER_00", "S0000/SPEAKER_01"],
        )
        self.assertEqual(result["segment_speakers"][0]["speakers"], ["SPEAKER_00", "SPEAKER_01"])

    def test_extract_explains_itself_when_vad_segments_are_missing(self):
        with self.assertRaises(RuntimeError) as raised:
            SegmentDiarization._extract({"audio_path": "clip.mkv"})
        self.assertIn("--diarize_scope file", str(raised.exception))


class TestSegmentScopeProgress(unittest.TestCase):
    class Reporter:
        def __init__(self): self.totals = []; self.advanced = 0
        def set_total(self, total, unit="it"): self.totals.append((total, unit))
        def advance(self, n=1): self.advanced += n

    def test_total_is_set_once_across_every_file(self):
        # StageTimer._start_determinate closes and replaces the bar on each set_total, so a
        # per-file call would restart the bar partway through the stage.
        reporter = self.Reporter()
        items = [
            {"audio_path": "a.wav", "vad_segments": [vad(0, 4), vad(5, 9)]},
            {"audio_path": "b.wav", "vad_segments": [vad(0, 4)]},
        ]
        packed = SegmentDiarization(stub_pipeline()).run(items, progress_callback=reporter)
        self.assertEqual(reporter.totals, [(3, "seg")])
        self.assertEqual(reporter.advanced, 3)
        self.assertEqual([item["audio_path"] for item in packed], ["a.wav", "b.wav"])
        self.assertTrue(all("diarization" in item for item in packed))


class TestFileScope(unittest.TestCase):
    def test_labels_are_not_namespaced(self):
        result = FileDiarization(stub_pipeline()).process(np.zeros(4 * SR, dtype=np.float32))
        self.assertEqual(result["scope"], "file")
        self.assertEqual(result["speakers"], ["SPEAKER_00"])
        self.assertEqual(result["turns"][0]["speaker"], "SPEAKER_00")


class TestScopeSelection(unittest.TestCase):
    def test_unknown_scope_names_the_valid_ones(self):
        with self.assertRaises(ValueError) as raised:
            load_diarization(scope="whole")
        self.assertIn("segment", str(raised.exception))


if __name__ == "__main__":
    unittest.main()

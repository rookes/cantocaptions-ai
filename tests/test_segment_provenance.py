"""The 'expanded' provenance key must survive both debug round-trips.

--asr_context_scope expanded reads VadAudioSegment['expanded'], which records the
sub-spans a reference cue added to the VAD timeline. Unlike 'context' it cannot be
re-derived downstream -- by then the pre-expansion timeline is gone -- so it has to
persist through the VAD and vocal-isolation manifests, both of which otherwise rebuild
segment dicts as exactly {start, end, audio}.
"""
import os
import shutil
import tempfile
import unittest

import numpy as np

from cantocaptions_ai.utils.debug import (_PERSISTED_SEGMENT_KEYS, load_isolation_debug,
                                          load_vad_debug, write_isolation_debug,
                                          write_vad_debug)


class TestProvenanceRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.audio = os.path.join(self.tmp, "clip.wav")
        self.segments = [
            {"start": 0.0, "end": 1.0, "audio": np.zeros(16000, dtype=np.float32),
             "expanded": [[0.4, 0.8]]},
            {"start": 1.0, "end": 2.0, "audio": np.zeros(16000, dtype=np.float32)},
        ]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_vad_round_trip_preserves_expanded(self):
        write_vad_debug(self.audio, self.segments, self.tmp)
        loaded = load_vad_debug(self.audio, self.tmp)
        self.assertEqual(loaded[0]["expanded"], [[0.4, 0.8]])

    def test_isolation_round_trip_preserves_expanded(self):
        write_isolation_debug(self.audio, self.segments, self.tmp)
        loaded = load_isolation_debug(self.audio, self.tmp)
        self.assertEqual(loaded[0]["expanded"], [[0.4, 0.8]])

    def test_segments_without_provenance_stay_clean(self):
        # A segment VAD found on its own must not gain an empty key, which would read
        # as "recovered nothing" rather than "was never expanded".
        write_vad_debug(self.audio, self.segments, self.tmp)
        loaded = load_vad_debug(self.audio, self.tmp)
        self.assertNotIn("expanded", loaded[1])

    def test_timings_and_audio_are_unaffected(self):
        write_vad_debug(self.audio, self.segments, self.tmp)
        loaded = load_vad_debug(self.audio, self.tmp)
        self.assertEqual([s["start"] for s in loaded], [0.0, 1.0])
        self.assertEqual([s["end"] for s in loaded], [1.0, 2.0])
        self.assertEqual(len(loaded[0]["audio"]), 16000)

    def test_manifests_written_before_this_key_existed_still_load(self):
        write_vad_debug(self.audio, [self.segments[1]], self.tmp)
        self.assertIsNotNone(load_vad_debug(self.audio, self.tmp))

    def test_allowlist_is_not_empty(self):
        # A blanket passthrough would break json.dump on the numpy audio; an empty
        # allowlist would silently disable the feature. Both are worth catching.
        self.assertIn("expanded", _PERSISTED_SEGMENT_KEYS)


if __name__ == "__main__":
    unittest.main()

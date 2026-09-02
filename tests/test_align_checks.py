"""Tests for the model-agnostic alignment output check.

Pure functions over timings and audio arrays — no model, no network. The real-model
behaviour these encode (a first character pinned to frame 0 by
alvanlii/wav2vec2-BERT-cantonese) is exercised end-to-end by
scripts/bench_align_primer.py.
"""

import unittest

import numpy as np

from cantocaptions_ai.pipeline.align_checks import (
    FRAME_SECONDS,
    MIN_GAP_FRAMES,
    find_gapped_cues,
    find_silent_starts,
    frame_dbfs,
    whole_file_region,
)

SR = 16000


def _tone(seconds, amplitude=0.3, sample_rate=SR):
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _silence(seconds, amplitude=0.0, sample_rate=SR):
    n = int(seconds * sample_rate)
    if amplitude == 0.0:
        return np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(0)
    return (rng.normal(0, amplitude, n)).astype(np.float32)


def _region(audio, start=0.0, sample_rate=SR):
    return {"start": start, "end": start + len(audio) / sample_rate, "audio": audio}


class TestFrameDbfs(unittest.TestCase):
    def test_silence_floors_and_tone_is_loud(self):
        db = frame_dbfs(np.concatenate([_silence(0.4), _tone(0.4)]))
        self.assertEqual(len(db), 20)  # 0.8s / 40ms
        self.assertLess(db[:10].max(), -200.0)
        self.assertGreater(db[10:].min(), -20.0)

    def test_frame_index_maps_by_plain_division(self):
        # A tone starting exactly 0.4s in must land on frame 10, not 9 or 11.
        db = frame_dbfs(np.concatenate([_silence(0.4), _tone(0.2)]))
        self.assertLess(db[9], -200.0)
        self.assertGreater(db[10], -20.0)

    def test_empty_audio(self):
        self.assertEqual(len(frame_dbfs(np.zeros(0, dtype=np.float32))), 0)


class TestFindSilentStarts(unittest.TestCase):
    def test_start_on_silence_far_from_sound_is_flagged(self):
        # 2s of silence then speech; a cue claims to start at t=0.
        audio = np.concatenate([_silence(2.0), _tone(1.0)])
        hits = find_silent_starts([{"start": 0.0, "text": "早晨"}], [_region(audio)])
        self.assertEqual(len(hits), 1)
        self.assertAlmostEqual(hits[0].time, 0.0)
        self.assertEqual(hits[0].text, "早晨")
        self.assertGreater(hits[0].gap, 1.9)

    def test_start_on_speech_is_not_flagged(self):
        audio = np.concatenate([_silence(2.0), _tone(1.0)])
        hits = find_silent_starts([{"start": 2.0, "text": "早晨"}], [_region(audio)])
        self.assertEqual(hits, [])

    def test_small_lead_on_the_onset_is_not_flagged(self):
        """CTC leads the acoustic onset slightly, and energy under-reads unvoiced onsets."""
        for lead in range(1, MIN_GAP_FRAMES):
            with self.subTest(lead=lead):
                self.assertEqual(self._hits_for_lead(lead), [])

    def test_lead_past_the_threshold_is_flagged(self):
        self.assertEqual(len(self._hits_for_lead(MIN_GAP_FRAMES)), 1)

    def _hits_for_lead(self, lead, onset_frame=50):
        """A cue starting `lead` frames before the tone. Times are taken at frame centres:
        a start sitting exactly on a frame boundary rounds either way under float error,
        which says nothing about the threshold under test."""
        audio = np.concatenate([_silence(onset_frame * FRAME_SECONDS), _tone(1.0)])
        start = (onset_frame - lead + 0.5) * FRAME_SECONDS
        return find_silent_starts([{"start": start, "text": "x"}], [_region(audio)])

    def test_music_bed_region_is_skipped(self):
        """No usable silence floor => skip rather than guess.

        A bed only a few dB under the speech is exactly the case that produced false
        positives during development.
        """
        bed = _silence(2.0, amplitude=0.05)
        audio = np.concatenate([bed, _tone(1.0, amplitude=0.09)])
        hits = find_silent_starts([{"start": 0.0, "text": "x"}], [_region(audio)])
        self.assertEqual(hits, [])

    def test_floor_is_relative_not_absolute(self):
        """A quiet-but-nonzero room tone still counts as silence against loud speech."""
        audio = np.concatenate([_silence(2.0, amplitude=1e-4), _tone(1.0)])
        hits = find_silent_starts([{"start": 0.0, "text": "x"}], [_region(audio)])
        self.assertEqual(len(hits), 1)

    def test_each_region_uses_its_own_floor(self):
        loud = _region(np.concatenate([_silence(2.0), _tone(1.0)]), start=0.0)
        quiet = _region(np.concatenate([_silence(2.0), _tone(1.0, amplitude=0.02)]), start=10.0)
        segs = [{"start": 0.0, "text": "a"}, {"start": 10.0, "text": "b"}]
        hits = find_silent_starts(segs, [loud, quiet])
        self.assertEqual([h.text for h in hits], ["a", "b"])

    def test_results_sorted_by_time(self):
        r1 = _region(np.concatenate([_silence(2.0), _tone(1.0)]), start=10.0)
        r2 = _region(np.concatenate([_silence(2.0), _tone(1.0)]), start=0.0)
        segs = [{"start": 10.0, "text": "later"}, {"start": 0.0, "text": "earlier"}]
        hits = find_silent_starts(segs, [r1, r2])
        self.assertEqual([h.text for h in hits], ["earlier", "later"])

    def test_segment_outside_every_region_is_ignored(self):
        audio = np.concatenate([_silence(2.0), _tone(1.0)])
        hits = find_silent_starts([{"start": 99.0, "text": "x"}], [_region(audio)])
        self.assertEqual(hits, [])

    def test_missing_start_is_ignored(self):
        audio = np.concatenate([_silence(2.0), _tone(1.0)])
        hits = find_silent_starts([{"text": "x"}], [_region(audio)])
        self.assertEqual(hits, [])

    def test_region_without_audio_is_ignored(self):
        hits = find_silent_starts([{"start": 0.0, "text": "x"}], [{"start": 0.0, "end": 1.0}])
        self.assertEqual(hits, [])


class TestWholeFileRegion(unittest.TestCase):
    def test_accepts_2d_and_torch(self):
        import torch

        audio = np.concatenate([_silence(2.0), _tone(1.0)])
        for candidate in (audio, audio.reshape(1, -1), torch.from_numpy(audio).unsqueeze(0)):
            with self.subTest(kind=type(candidate).__name__):
                region = whole_file_region(candidate, 3.0)
                hits = find_silent_starts([{"start": 0.0, "text": "x"}], region)
                self.assertEqual(len(hits), 1)


class TestGappedCues(unittest.TestCase):
    """A cue holding a silence between two of its own adjacent characters.

    CTC must place every token it is given, so a cue whose text contains something not said
    where the cue sits gets those characters put wherever scores least badly. On test/bluey,
    ASR emitted 爸爸， once for what the reference has as two separate calls: the first 爸
    landed 3.35 s before the second, and the subtitle appeared that far ahead of the speech.
    """

    def _cue(self, *spans):
        return {"start": spans[0][1], "end": spans[-1][2], "text": "".join(s[0] for s in spans),
                "words": [{"word": w, "start": a, "end": b, "score": 0.9} for w, a, b in spans]}

    def test_a_hole_between_two_characters_is_found(self):
        cue = self._cue(("爸", 56.44, 56.60), ("爸", 59.80, 59.84))
        hits = find_gapped_cues([cue])
        self.assertEqual(len(hits), 1)
        self.assertAlmostEqual(hits[0].gap, 3.20, places=2)
        self.assertEqual((hits[0].before, hits[0].after), ("爸", "爸"))

    def test_a_continuous_cue_is_not(self):
        cue = self._cue(("你", 1.0, 1.2), ("好", 1.2, 1.4), ("嗎", 1.4, 1.7))
        self.assertEqual(find_gapped_cues([cue]), [])

    def test_a_pause_a_punctuation_mark_holds_is_not_a_hole(self):
        # Punctuation is mapped to blank precisely so it can absorb a pause, and it holds it
        # by *spanning* it -- so the mark is contiguous with both neighbours and there is no
        # gap to find. The check is immune to a clause break by construction, with or
        # without split_chars; this is why it can be left on for every model.
        cue = self._cue(("好", 1.0, 1.2), ("，", 1.2, 4.0), ("係", 4.0, 4.2))
        self.assertEqual(find_gapped_cues([cue], split_chars="，。？！"), [])
        self.assertEqual(find_gapped_cues([cue]), [])

    def test_split_chars_covers_a_mark_that_sits_apart_from_its_neighbours(self):
        # The residual case: the mark did not span the pause, it sits inside it.
        cue = self._cue(("好", 1.0, 1.2), ("，", 2.5, 2.6), ("係", 4.0, 4.2))
        self.assertEqual(len(find_gapped_cues([cue])), 1)
        self.assertEqual(find_gapped_cues([cue], split_chars="，。？！"), [],
                         "with the mark discounted, 好 and 係 are not adjacent characters")

    def test_only_the_worst_gap_in_a_cue_is_reported(self):
        cue = self._cue(("a", 0.0, 0.1), ("b", 1.5, 1.6), ("c", 4.0, 4.1))
        hits = find_gapped_cues([cue])
        self.assertEqual(len(hits), 1)
        self.assertAlmostEqual(hits[0].gap, 2.4, places=2)

    def test_an_untimed_character_is_skipped_not_treated_as_zero(self):
        cue = {"start": 0.0, "end": 1.0, "text": "ab",
               "words": [{"word": "a"}, {"word": "b", "start": 0.9, "end": 1.0}]}
        self.assertEqual(find_gapped_cues([cue]), [])

    def test_the_hit_names_its_cue(self):
        cues = [self._cue(("你", 1.0, 1.2), ("好", 1.2, 1.4)),
                self._cue(("爸", 56.44, 56.60), ("爸", 59.80, 59.84))]
        self.assertEqual([h.index for h in find_gapped_cues(cues)], [1])


if __name__ == "__main__":
    unittest.main()

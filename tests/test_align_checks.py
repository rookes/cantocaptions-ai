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
    SPLIT_INTERNAL_GAP,
    find_gapped_cues,
    find_silent_starts,
    frame_dbfs,
    split_gapped_cues,
    whole_file_region,
)
from cantocaptions_ai.pipeline.align_profiles import (
    ALIGN_PROFILES,
    DEFAULT_ALIGN_PROFILE,
    get_align_profile,
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


class TestSplitGappedCues(unittest.TestCase):
    """Acting on that finding: break the cue in two rather than only distrusting it.

    Opt-in, unlike every other check in the module, and at a threshold well above the one
    find_gapped_cues reports at -- see split_gapped_cues' docstring for why.
    """

    def _cue(self, *spans, **extra):
        cue = {"start": spans[0][1], "end": spans[-1][2],
               "text": "".join(s[0] for s in spans),
               "words": [{"word": w, "start": a, "end": b, "score": 0.9}
                         for w, a, b in spans]}
        cue.update(extra)
        return cue

    def test_the_bluey_case_becomes_one_cue_per_call(self):
        # ASR emitted 爸爸， once for what the reference has as two separate calls
        # 3.35s apart, so the subtitle went up long before the second one was spoken.
        cue = self._cue(("爸", 56.44, 56.60), ("爸", 59.80, 59.84),
                        ("，", 59.84, 59.90))
        out, breaks = split_gapped_cues([cue], 1.5, "，。？！")
        self.assertEqual(breaks, 1)
        self.assertEqual([c["text"] for c in out], ["爸", "爸，"])
        self.assertEqual([(c["start"], c["end"]) for c in out],
                         [(56.44, 56.60), (59.80, 59.90)])

    def test_the_outer_edges_are_the_cues_own_and_the_cut_is_the_silence(self):
        # Each piece keeps the edge the caller's release/trim pass gave it and takes its
        # inner edge from its own characters, so nothing outside the split can move.
        cue = self._cue(("a", 1.0, 1.1), ("b", 5.0, 5.1))
        cue["start"], cue["end"] = 0.5, 6.0  # as a release/trim pass would leave them
        (head, tail), _ = split_gapped_cues([cue], 1.5)
        self.assertEqual((head["start"], head["end"]), (0.5, 1.1))
        self.assertEqual((tail["start"], tail["end"]), (5.0, 6.0))
        self.assertGreater(tail["start"], head["end"], "the pieces must not overlap")

    def test_a_continuous_cue_is_untouched_and_the_input_is_not_mutated(self):
        cue = self._cue(("你", 1.0, 1.2), ("好", 1.2, 1.4), ("嗎", 1.4, 1.7))
        out, breaks = split_gapped_cues([cue], 1.5)
        self.assertEqual(breaks, 0)
        self.assertEqual([c["text"] for c in out], ["你好嗎"])
        out[0]["text"] = "changed"
        self.assertEqual(cue["text"], "你好嗎",
                         "segments must be copied, never mutated")

    def test_a_gap_under_the_threshold_does_not_split(self):
        # 1.2s is over find_gapped_cues' 1.0s reporting threshold and under this one:
        # reporting and breaking are deliberately different bets.
        cue = self._cue(("a", 1.0, 1.1), ("b", 2.3, 2.4))
        self.assertEqual(len(find_gapped_cues([cue])), 1)
        self.assertEqual(split_gapped_cues([cue], SPLIT_INTERNAL_GAP)[1], 0)

    def test_a_pause_a_punctuation_mark_holds_is_not_a_split(self):
        # Same construction that makes find_gapped_cues immune to an ordinary clause break:
        # the mark spans the pause, so no two adjacent characters are apart.
        cue = self._cue(("好", 1.0, 1.2), ("，", 1.2, 4.0), ("係", 4.0, 4.2))
        self.assertEqual(split_gapped_cues([cue], 1.5, "，。？！")[1], 0)

    def test_two_gaps_give_three_cues(self):
        cue = self._cue(("a", 0.0, 0.1), ("b", 2.0, 2.1), ("c", 5.0, 5.1))
        out, breaks = split_gapped_cues([cue], 1.5)
        self.assertEqual(breaks, 2)
        self.assertEqual([c["text"] for c in out], ["a", "b", "c"])
        self.assertEqual([(c["start"], c["end"]) for c in out],
                         [(0.0, 0.1), (2.0, 2.1), (5.0, 5.1)])

    def test_an_untimed_character_goes_with_the_head(self):
        # A character the align model has no token for was never placed, so there is no
        # evidence it belongs to the clause *after* the silence.
        cue = {"start": 0.0, "end": 5.1, "text": "a喿b",
               "words": [{"word": "a", "start": 0.0, "end": 0.1},
                         {"word": "喿"},
                         {"word": "b", "start": 5.0, "end": 5.1}]}
        out, breaks = split_gapped_cues([cue], 1.5)
        self.assertEqual(breaks, 1)
        self.assertEqual([c["text"] for c in out], ["a喿", "b"])
        self.assertEqual([[w["word"] for w in c["words"]] for c in out],
                         [["a", "喿"], ["b"]])

    def test_char_alignments_are_cut_at_the_same_offsets_as_the_text(self):
        cue = self._cue(("a", 0.0, 0.1), ("b", 5.0, 5.1))
        cue["chars"] = [{"char": "a", "start": 0.0, "end": 0.1},
                        {"char": "b", "start": 5.0, "end": 5.1}]
        out, _ = split_gapped_cues([cue], 1.5)
        self.assertEqual([[c["char"] for c in piece["chars"]] for piece in out],
                         [["a"], ["b"]])

    def test_every_piece_records_why_it_was_cut(self):
        cue = self._cue(("爸", 56.44, 56.60), ("爸", 59.80, 59.84))
        out, _ = split_gapped_cues([cue], 1.5)
        self.assertEqual(out[0]["notes"], ["split_gap:3.2s after 爸"])
        self.assertEqual(out[1]["notes"], ["split_gap:3.2s before 爸"])

    def test_a_cue_whose_words_do_not_match_its_text_is_passed_through_whole(self):
        # Refusing is the only safe answer: a wrong offset would cut the text in one place
        # and the characters in another.
        cue = self._cue(("a", 0.0, 0.1), ("b", 5.0, 5.1))
        cue["text"] = "something else entirely"
        out, breaks = split_gapped_cues([cue], 1.5)
        self.assertEqual(breaks, 0)
        self.assertEqual(len(out), 1)

    def test_keys_later_stages_attach_survive_the_split(self):
        cue = self._cue(("a", 0.0, 0.1), ("b", 5.0, 5.1),
                        avg_logprob=-0.3, realign_reason="isolated")
        out, _ = split_gapped_cues([cue], 1.5)
        for piece in out:
            self.assertEqual(piece["avg_logprob"], -0.3)
            self.assertEqual(piece["realign_reason"], "isolated")

    def test_notes_are_not_shared_between_the_pieces(self):
        cue = self._cue(("a", 0.0, 0.1), ("b", 5.0, 5.1), notes=["homophone:x->y"])
        out, _ = split_gapped_cues([cue], 1.5)
        self.assertIn("homophone:x->y", out[0]["notes"])
        self.assertIn("homophone:x->y", out[1]["notes"])
        self.assertIsNot(out[0]["notes"], out[1]["notes"])
        self.assertEqual(cue["notes"], ["homophone:x->y"])

    def test_cue_order_is_preserved_across_a_file(self):
        cues = [self._cue(("x", 0.0, 0.2)),
                self._cue(("a", 1.0, 1.1), ("b", 5.0, 5.1)),
                self._cue(("y", 9.0, 9.2))]
        out, breaks = split_gapped_cues(cues, 1.5)
        self.assertEqual(breaks, 1)
        self.assertEqual([c["text"] for c in out], ["x", "a", "b", "y"])
        starts = [c["start"] for c in out]
        self.assertEqual(starts, sorted(starts))


class TestSplitIsOptIn(unittest.TestCase):
    """The threshold is per align model, and no shipped model sets one."""

    def test_no_profile_splits_by_default(self):
        self.assertIsNone(DEFAULT_ALIGN_PROFILE.split_gap)
        for name, profile in ALIGN_PROFILES.items():
            with self.subTest(model=name):
                self.assertIsNone(profile.split_gap)

    def test_a_model_with_no_entry_gets_the_no_op_default(self):
        self.assertIsNone(get_align_profile("some/model-added-later").split_gap)


if __name__ == "__main__":
    unittest.main()

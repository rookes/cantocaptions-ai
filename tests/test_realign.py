"""Tests for --realign: placing an untimed transcript on the audio timeline.

Everything here runs on synthetic emissions with a five-entry vocabulary, so the CTC search
is exercised without a model. That is enough to pin the parts that are easy to get subtly
wrong -- the free end of the Viterbi, the window commit rule, and the index arithmetic
linking a transcript line to its characters in a joined chunk -- while the real-model
behaviour is measured by scripts/eval_realign.py against a ground-truth SRT.
"""

import math
import unittest

import numpy as np
import torch

from cantocaptions_ai.pipeline.alignment import (
    MAX_CHAR_DWELL_SECONDS,
    Segment,
    _preprocess_segment,
    _reseat_dwelling_chars,
    backtrack,
    get_trellis,
)
from cantocaptions_ai.pipeline.realign import (
    MAX_INTERNAL_GAP,
    REALIGN_PUNCTUATION,
    REALIGN_SENTINEL,
    EmissionTimeline,
    REASON_ISOLATED,
    REASON_NO_AUDIO,
    REASON_NO_VOCABULARY,
    LineTiming,
    TranscriptLine,
    MAX_CHARS_PER_SECOND,
    MAX_DETACHED_RUN,
    MIN_CHUNK_DURATION,
    MIN_CUE_SCORE,
    MIN_LINE_SPAN,
    MIN_VISIBLE_DURATION,
    NOMINAL_CHAR_RATE,
    _bounded_timing,
    _char_times,
    acquire,
    _core_token_run,
    _sanitize,
    assign_lines,
    assign_lines_via_asr,
    build_align_input,
    enforce_cue_order,
    ensure_visible_cues,
    sanitise_anchors,
    verify_anchors,
    find_implausible_cues,
    line_tokens,
    load_transcript_lines,
    normalize_transcript_text,
    strip_sentinels,
    tighten_cue_spans,
)

SR = 16000
FPS = 25.0
VOCAB = {"[pad]": 0, "a": 1, "b": 2, "c": 3, "d": 4}
BLANK = 0


def _emission(labels, confident=-0.05, other=-12.0):
    """Log-prob emission of one frame per entry in *labels* (None means blank)."""
    data = np.full((len(labels), len(VOCAB)), other, dtype=np.float32)
    for i, label in enumerate(labels):
        data[i, BLANK if label is None else VOCAB[label]] = confident
    return torch.from_numpy(data)


def _labels(spans, total):
    """Frame labels for {start_frame: text} placed on a blank canvas of *total* frames."""
    out = [None] * total
    for start, text in spans.items():
        for k, ch in enumerate(text):
            out[start + k] = ch
    return out


def _chunk(start, num_frames):
    end = start + num_frames / FPS
    return {
        "start": start,
        "end": end,
        "audio": np.zeros(int((end - start) * SR), dtype=np.float32),
    }


def _timeline(chunks, emissions):
    by_start = {c["start"]: (e, FPS) for c, e in zip(chunks, emissions)}
    return EmissionTimeline(chunks, lambda segs: [by_start[s["start"]] for s in segs])


class TestNormalization(unittest.TestCase):
    def test_halfwidth_marks_convert_only_after_cjk(self):
        self.assertEqual(normalize_transcript_text("喂,家駒"), "喂，家駒")
        self.assertEqual(normalize_transcript_text("你好?"), "你好？")
        # An English clause is the subtitle as written; leave its punctuation alone.
        self.assertEqual(normalize_transcript_text("Thank you, sir."), "Thank you, sir.")
        self.assertEqual(normalize_transcript_text("x1.5y"), "x1.5y")

    def test_midline_ellipsis_becomes_the_split_char(self):
        # The source file uses U+22EF, which is neither in the vocab nor a split char.
        self.assertEqual(normalize_transcript_text("你呀⋯⋯"), "你呀…")
        self.assertEqual(normalize_transcript_text("你呀……"), "你呀…")

    def test_space_runs_collapse_but_a_single_space_survives(self):
        self.assertEqual(normalize_transcript_text("喂  有三個人"), "喂 有三個人")
        self.assertEqual(normalize_transcript_text("  喂 有  "), "喂 有")

    def test_a_stray_sentinel_in_the_source_is_removed(self):
        self.assertEqual(normalize_transcript_text("你" + REALIGN_SENTINEL + "好"), "你好")


class TestLoader(unittest.TestCase):
    def _write(self, data):
        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        self.addCleanup(os.unlink, path)
        return path

    def test_bom_and_mixed_line_endings(self):
        path = self._write("﻿你好\r\n世界\n\r\n再見\r\n".encode("utf-8"))
        lines = load_transcript_lines(path)
        self.assertEqual([l.text for l in lines], ["你好", "世界", "再見"])
        self.assertEqual([l.index for l in lines], [0, 1, 2])

    def test_blank_lines_are_separators_not_cues(self):
        path = self._write("a\n\n\n   \nb\n".encode("utf-8"))
        self.assertEqual([l.text for l in load_transcript_lines(path)], ["a", "b"])


class TestTokenization(unittest.TestCase):
    def test_split_chars_become_blank_and_unknown_chars_are_dropped(self):
        # 'z' is out of vocabulary; the space and the comma are pause tokens.
        # 'z' is dropped entirely (neither vocab nor split char); the space and the comma
        # each survive as a blank, so the pause they mark can be spent by the aligner.
        tokens = line_tokens("ab z，c", "yue", VOCAB, BLANK)
        self.assertEqual(tokens, [VOCAB["a"], VOCAB["b"], BLANK, BLANK, VOCAB["c"]])

    def test_cue_spans_override_punctuation_derived_spans(self):
        text = "ab，cd"
        derived = _preprocess_segment(text, "yue", VOCAB, REALIGN_PUNCTUATION)
        self.assertEqual(derived["sentence_spans"], [(0, 2), (3, 5)])
        declared = _preprocess_segment(
            text, "yue", VOCAB, REALIGN_PUNCTUATION, spans=[(0, 5)]
        )
        self.assertEqual(declared["sentence_spans"], [(0, 5)])


class TestFreeEndBacktrack(unittest.TestCase):
    def test_free_end_consumes_only_what_the_audio_explains(self):
        emission = _emission(_labels({2: "ab"}, 12))
        tokens = [VOCAB[c] for c in "abcd"]

        forced = backtrack(get_trellis(emission, tokens, BLANK), emission, tokens, BLANK)
        free = backtrack(
            get_trellis(emission, tokens, BLANK, free_end=True), emission, tokens, BLANK,
            free_end=True,
        )
        # Forced alignment must place all four tokens somewhere in the blank tail.
        self.assertEqual(forced[-1].token_index, 3)
        # The free end stops after the two the audio actually contains.
        self.assertEqual(free[-1].token_index, 1)

    def test_free_end_survives_where_the_forced_walk_cannot(self):
        # CTC needs a frame per token, so a forced walk over more tokens than frames has no
        # path at all. The free end simply stops at the furthest column it could reach --
        # which is the entire point of it for a window that runs out of audio mid-transcript.
        emission = _emission([None] * 3)
        tokens = [VOCAB[c] for c in "abcdabcdab"]
        self.assertIsNone(backtrack(get_trellis(emission, tokens, BLANK), emission, tokens, BLANK))
        path = backtrack(
            get_trellis(emission, tokens, BLANK, free_end=True), emission, tokens, BLANK,
            free_end=True,
        )
        self.assertIsNotNone(path)
        self.assertLess(path[-1].token_index, len(tokens))


class TestAssignLines(unittest.TestCase):
    def setUp(self):
        # One 8 s file as two contiguous 4 s chunks, three utterances with silence between.
        self.chunks = [_chunk(0.0, 100), _chunk(4.0, 100)]
        self.emissions = [
            _emission(_labels({10: "ab", 60: "cd"}, 100)),
            _emission(_labels({30: "ac"}, 100)),
        ]
        self.lines = [
            TranscriptLine(0, "ab"),
            TranscriptLine(1, "cd"),
            TranscriptLine(2, "ac"),
        ]

    def _assign(self, **kwargs):
        return assign_lines(
            self.lines, self.chunks, _timeline(self.chunks, self.emissions),
            VOCAB, "yue", blank_id=BLANK,
            **{"window_seconds": 6.0, "commit_margin": 1.0, **kwargs},
        )

    def test_every_line_is_placed_once_in_order(self):
        timings = self._assign()
        self.assertEqual([t.index for t in timings], [0, 1, 2])
        self.assertTrue(all(t.placed for t in timings))

    def test_placements_land_on_the_frames_that_carry_the_text(self):
        timings = self._assign()
        expected = [10 / FPS, 60 / FPS, 4.0 + 30 / FPS]
        for timing, want in zip(timings, expected):
            self.assertAlmostEqual(timing.start, want, delta=0.12)
            self.assertGreater(timing.end, timing.start)

    def test_timings_are_monotonic_and_non_overlapping(self):
        timings = self._assign()
        for a, b in zip(timings, timings[1:]):
            self.assertLessEqual(a.start, b.start)
            self.assertLessEqual(a.end, b.start + 1e-6)

    def test_a_narrow_window_still_covers_the_whole_file(self):
        # Forces several windows, so the commit-and-resync loop is what does the work.
        timings = self._assign(window_seconds=2.5, commit_margin=0.5)
        self.assertEqual([t.index for t in timings], [0, 1, 2])

    def test_more_lines_than_audio_still_terminates(self):
        # A transcript claiming utterances the recording does not contain must not spin.
        self.lines = self.lines + [TranscriptLine(3 + i, "bd") for i in range(6)]
        timings = self._assign()
        self.assertEqual([t.index for t in timings], list(range(9)))


class TestDetachedTokens(unittest.TestCase):
    """A cue is displayed all at once, so a character seconds from its line is misplaced."""

    def _run(self, frames_per_token, max_gap=MAX_INTERNAL_GAP):
        body = list(range(len(frames_per_token)))
        first = {j: f for j, f in enumerate(frames_per_token)}
        last = dict(first)
        times = np.arange(max(frames_per_token) + 2) / FPS
        return _core_token_run(body, first, last, times, max_gap)

    def test_a_contiguous_line_is_kept_whole(self):
        self.assertEqual(self._run([10, 11, 12, 13]), [0, 1, 2, 3])

    def test_a_stranded_leading_interjection_is_dropped(self):
        # 噢 parked 8 s early; the body of the line is where the cue actually belongs.
        self.assertEqual(self._run([2, 202, 203, 204, 205]), [1, 2, 3, 4])

    def test_a_stranded_trailing_token_is_dropped(self):
        self.assertEqual(self._run([10, 11, 12, 13, 300]), [0, 1, 2, 3])

    def test_an_even_split_keeps_the_opening(self):
        # No majority body, so the cue starts where the line starts rather than jumping.
        self.assertEqual(self._run([10, 11, 300, 301]), [0, 1])

    def test_a_single_token_line_is_untouched(self):
        self.assertEqual(self._run([42]), [0])


class TestAsrAnchor(unittest.TestCase):
    """--realign_anchor asr: time the transcript by matching what ASR actually heard."""

    def _asr(self, *segments):
        return [{"start": s, "end": e, "text": t} for s, e, t in segments]

    def test_a_clean_hypothesis_places_every_line(self):
        lines = [TranscriptLine(0, "你好嗎"), TranscriptLine(1, "幾好呀")]
        timings = assign_lines_via_asr(lines, self._asr((10.0, 13.0, "你好嗎幾好呀")))
        self.assertTrue(all(t.placed for t in timings))
        self.assertLess(timings[0].start, timings[1].start)
        self.assertAlmostEqual(timings[0].start, 10.0, delta=0.3)
        self.assertAlmostEqual(timings[1].end, 13.0, delta=0.3)

    def test_asr_errors_do_not_derail_the_match(self):
        # The point of matching streams rather than requiring equality.
        lines = [TranscriptLine(0, "你好嗎"), TranscriptLine(1, "幾好呀")]
        timings = assign_lines_via_asr(lines, self._asr((10.0, 13.0, "你好媽幾好呀")))
        self.assertTrue(all(t.placed for t in timings))
        self.assertLess(timings[0].start, timings[1].start)

    def test_a_line_the_recording_never_says_is_flagged_not_hidden(self):
        # The reason this anchor exists: forced alignment would have to put it somewhere.
        lines = [
            TranscriptLine(0, "你好嗎"),
            TranscriptLine(1, "呢句戲入面冇人講過"),
            TranscriptLine(2, "幾好呀"),
        ]
        timings = assign_lines_via_asr(lines, self._asr((10.0, 13.0, "你好嗎幾好呀")))
        self.assertEqual([t.placed for t in timings], [True, False, True])
        self.assertEqual(len(timings), 3, "an unmatched line must survive, not be dropped")

    def test_timings_stay_ordered_even_with_unmatched_lines(self):
        lines = [TranscriptLine(i, t) for i, t in enumerate(
            ["你好嗎", "無關嘅嘢", "幾好呀", "另一句冇嘅", "多謝晒"]
        )]
        timings = assign_lines_via_asr(lines, self._asr((5.0, 11.0, "你好嗎幾好呀多謝晒")))
        for a, b in zip(timings, timings[1:]):
            self.assertLessEqual(a.start, b.start)

    def test_empty_asr_output_still_returns_one_timing_per_line(self):
        lines = [TranscriptLine(0, "你好嗎"), TranscriptLine(1, "幾好呀")]
        timings = assign_lines_via_asr(lines, self._asr((0.0, 5.0, "")))
        self.assertEqual([t.index for t in timings], [0, 1])
        self.assertTrue(all(not t.placed for t in timings))


class TestSpanBound(unittest.TestCase):
    """No placement may claim more audio than its text could be spoken in.

    This is what keeps a stretch the model cannot read (singing, a foreign language) from
    marching the pointer off the end of the file. On the Doraemon fixture the bound was
    present but the "nothing committed in the safe zone" fallback built its LineTiming
    directly and bypassed it, so single lines of 7-19 characters claimed 90-120 s windows
    and the search drifted 547 s. Every path must go through _bounded_timing.
    """

    def test_a_plausible_span_is_left_alone(self):
        line = TranscriptLine(0, "你好嗎幾好呀")
        timing = _bounded_timing(line, 10.0, 12.0, floor=0.0)
        self.assertTrue(timing.placed)
        self.assertAlmostEqual(timing.end, 12.0, places=3)

    def test_an_implausible_span_falls_back_to_the_nominal_rate(self):
        line = TranscriptLine(0, "你好嗎幾好呀")  # 6 chars
        timing = _bounded_timing(line, 10.0, 130.0, floor=0.0)
        self.assertFalse(timing.placed, "an unreadable stretch must be flagged")
        self.assertAlmostEqual(
            timing.end - timing.start, len(line.text) / NOMINAL_CHAR_RATE, places=2,
            msg="falling back to the detection limit would still run ~5x too fast",
        )

    def test_a_short_line_may_still_be_drawn_out(self):
        line = TranscriptLine(0, "吓")
        timing = _bounded_timing(line, 10.0, 10.0 + MIN_LINE_SPAN - 0.1, floor=0.0)
        self.assertTrue(timing.placed)

    def test_the_floor_is_never_crossed(self):
        timing = _bounded_timing(TranscriptLine(0, "你好"), 5.0, 6.0, floor=8.0)
        self.assertGreaterEqual(timing.start, 8.0)

    def test_a_window_that_commits_nothing_still_bounds_its_fallback(self):
        # The regression, end to end: audio the model cannot read at all. Every committed
        # line must stay within its plausible span rather than swallowing the window.
        chunks = [_chunk(0.0, 250), _chunk(10.0, 250)]
        emissions = [_emission([None] * 250), _emission([None] * 250)]
        lines = [TranscriptLine(i, "你好嗎") for i in range(6)]
        timings = assign_lines(
            lines, chunks, _timeline(chunks, emissions), VOCAB, "yue", blank_id=BLANK,
            window_seconds=10.0, commit_margin=2.0,
        )
        self.assertEqual(len(timings), 6)
        for t in timings:
            self.assertLessEqual(
                t.end - t.start, max(MIN_LINE_SPAN, 3 / 1.0) + 1e-6,
                f"line {t.index} claimed {t.end - t.start:.1f}s of unreadable audio",
            )


class TestBuildAlignInput(unittest.TestCase):
    def setUp(self):
        self.chunks = [_chunk(0.0, 250), _chunk(10.0, 250)]
        self.lines = [TranscriptLine(i, text) for i, text in enumerate(["ab", "cd", "ac"])]
        self.timings = [
            LineTiming(0, 1.0, 2.0),
            LineTiming(1, 5.0, 6.0),
            LineTiming(2, 12.0, 13.0),
        ]

    def test_chunks_are_sorted_disjoint_and_within_budget(self):
        chunks, _ = build_align_input(self.lines, self.timings, self.chunks, chunk_size=8.0)
        self.assertGreater(len(chunks), 1)
        for a, b in zip(chunks, chunks[1:]):
            self.assertLessEqual(a["end"], b["start"] + 1e-6)
        for chunk in chunks:
            self.assertLessEqual(chunk["end"] - chunk["start"], 8.0 + 1e-6)
            self.assertEqual(
                len(chunk["audio"]),
                int(round((chunk["end"] - chunk["start"]) * SR)),
                "sliced audio must match the chunk it claims to cover",
            )

    def test_every_line_lies_wholly_inside_its_own_chunk(self):
        chunks, transcript = build_align_input(
            self.lines, self.timings, self.chunks, chunk_size=8.0
        )
        placed = {t.index: t for t in self.timings}
        cursor = 0
        for chunk, segment in zip(chunks, transcript):
            self.assertEqual((segment["start"], segment["end"]), (chunk["start"], chunk["end"]))
            for _span in segment["cue_spans"]:
                timing = placed[self.lines[cursor].index]
                self.assertGreaterEqual(timing.start, chunk["start"] - 1e-6)
                self.assertLessEqual(timing.end, chunk["end"] + 1e-6)
                cursor += 1
        self.assertEqual(cursor, len(self.lines))

    def test_cue_spans_index_the_joined_text_back_to_each_line(self):
        _, transcript = build_align_input(
            self.lines, self.timings, self.chunks, chunk_size=30.0
        )
        self.assertEqual(len(transcript), 1, "30s budget should hold all three lines")
        segment = transcript[0]
        recovered = [
            segment["text"][start:end + 1] for start, end in segment["cue_spans"]
        ]
        self.assertEqual(recovered, [line.text + REALIGN_SENTINEL for line in self.lines])


class TestTightenCueSpans(unittest.TestCase):
    """The bluey regression: a cue inheriting its start from a blank token's dwell."""

    def _cue(self, start, end, words):
        return {"start": start, "end": end, "text": "".join(w for w, _, _ in words),
                "words": [
                    {"word": w} if s is None else {"word": w, "start": s, "end": e}
                    for w, s, e in words
                ]}

    def test_a_blank_dwell_no_longer_sets_the_cue_start(self):
        # 噢 is out of vocabulary so it has no timing, and the comma after it was mapped to
        # the blank token, whose CTC path dwelt across ten seconds of preceding silence.
        cue = self._cue(72.23, 84.0, [
            ("噢", None, None), ("，", 72.23, 82.11),
            ("係", 82.11, 82.27), ("島", 82.27, 82.59), ("嶼", 82.59, 82.75),
        ])
        self.assertEqual(tighten_cue_spans([cue]), 1)
        self.assertAlmostEqual(cue["start"], 82.11, places=2)
        self.assertAlmostEqual(cue["end"], 84.0, places=2, msg="the release must survive")

    def test_a_well_formed_cue_is_left_alone(self):
        cue = self._cue(10.0, 12.0, [
            ("你", 10.0, 10.3), ("好", 10.3, 10.6), ("嗎", 10.6, 10.9),
        ])
        self.assertEqual(tighten_cue_spans([cue]), 0)
        self.assertEqual((cue["start"], cue["end"]), (10.0, 12.0))

    def test_a_stranded_trailing_character_pulls_the_end_in(self):
        cue = self._cue(10.0, 30.0, [
            ("你", 10.0, 10.3), ("好", 10.3, 10.6), ("嗎", 10.6, 10.9),
            ("啊", 29.0, 29.4),
        ])
        self.assertEqual(tighten_cue_spans([cue]), 1)
        self.assertAlmostEqual(cue["end"], 10.9, places=2)

    def test_a_cue_with_no_timed_characters_is_skipped(self):
        cue = self._cue(5.0, 6.0, [("嘿", None, None), ("嘿", None, None)])
        self.assertEqual(tighten_cue_spans([cue]), 0)
        self.assertEqual((cue["start"], cue["end"]), (5.0, 6.0))

    def test_the_span_only_ever_shrinks(self):
        # Guarantees cue ordering cannot be disturbed, whatever the word timings say.
        cue = self._cue(20.0, 21.0, [("你", 5.0, 5.3), ("好", 5.3, 5.6)])
        tighten_cue_spans([cue])
        self.assertGreaterEqual(cue["start"], 20.0)
        self.assertLessEqual(cue["end"], 21.0)


class TestDegenerateTimings(unittest.TestCase):
    """The Doraemon regression: a transcript longer than its recording.

    The coarse search stranded 295 lines on the last frame of the audio with inverted spans.
    Non-monotonic boundaries meant the grouping pass never split, so one chunk covered the
    whole 1450 s file and the encoder forward pass over it consumed 17 GB before being
    killed. build_align_input has to survive any placement it is handed.
    """

    def setUp(self):
        self.source = [_chunk(0.0, 2500)]  # 100 s of contiguous audio
        self.file_end = self.source[0]["end"]

    def _lines(self, n):
        return [TranscriptLine(i, "你好嗎") for i in range(n)]

    def test_lines_stranded_on_the_final_frame_still_chunk_within_budget(self):
        lines = self._lines(60)
        timings = [LineTiming(i, 1.0 + i, 2.0 + i) for i in range(10)]
        # ...and the other 50 jammed at the end, ends before starts, as observed.
        timings += [
            LineTiming(i, self.file_end + 0.33, self.file_end, reason="unreadable")
            for i in range(10, 60)
        ]
        chunks, transcript = build_align_input(lines, timings, self.source, chunk_size=30.0)
        self.assertEqual(len(chunks), len(transcript))
        for chunk in chunks:
            self.assertLessEqual(chunk["end"] - chunk["start"], 30.0 + 1e-6)
            self.assertGreater(chunk["end"], chunk["start"])

    def test_chunks_stay_sorted_and_disjoint_under_degenerate_input(self):
        lines = self._lines(30)
        timings = [LineTiming(i, self.file_end, self.file_end - 1.0) for i in range(30)]
        chunks, _ = build_align_input(lines, timings, self.source, chunk_size=30.0)
        for a, b in zip(chunks, chunks[1:]):
            self.assertLessEqual(a["end"], b["start"] + 1e-6)

    def test_no_line_is_lost_however_bad_the_placements(self):
        lines = self._lines(40)
        timings = [LineTiming(i, 500.0, -3.0) for i in range(40)]  # wholly nonsensical
        _, transcript = build_align_input(lines, timings, self.source, chunk_size=30.0)
        self.assertEqual(sum(len(s["cue_spans"]) for s in transcript), 40)

    def test_a_stranded_run_is_split_by_the_character_budget(self):
        """The Police Story 2 regression: 1015 lines on one timestamp became one subtitle.

        CTC needs a frame per token, so a chunk holding far more characters than its audio has
        frames cannot align at all -- backtrack fails and _align_segment falls back to a single
        cue carrying every line in the group. Grouping has to bound characters, not just time.
        """
        lines = self._lines(400)
        end = self.file_end
        timings = [LineTiming(i, end, end, reason="unreadable") for i in range(400)]
        chunks, transcript = build_align_input(lines, timings, self.source, chunk_size=30.0)
        budget = 30.0 * MAX_CHARS_PER_SECOND
        for segment in transcript:
            self.assertLessEqual(len(segment["text"]), budget + 1)
        self.assertGreater(len(chunks), 1, "one chunk for 400 stranded lines is the bug")
        self.assertEqual(sum(len(s["cue_spans"]) for s in transcript), 400)

    def test_a_dense_but_valid_stretch_is_also_split(self):
        # Not just a degenerate-input guard: real dialogue can outrun the frame budget too.
        lines = [TranscriptLine(i, "你" * 40) for i in range(30)]
        timings = [LineTiming(i, 1.0 + i * 0.5, 1.4 + i * 0.5) for i in range(30)]
        _, transcript = build_align_input(lines, timings, self.source, chunk_size=30.0)
        for segment in transcript:
            self.assertLessEqual(len(segment["text"]), 30.0 * MAX_CHARS_PER_SECOND + 1)

    def test_an_implausibly_long_span_is_flagged_not_believed(self):
        # One line cannot occupy 90 s of a 100 s file; the search smeared it.
        chunks, _ = build_align_input(
            self._lines(2), [LineTiming(0, 1.0, 91.0), LineTiming(1, 92.0, 93.0)],
            self.source, chunk_size=30.0,
        )
        for chunk in chunks:
            self.assertLessEqual(chunk["end"] - chunk["start"], 30.0 + 1e-6)


class TestEnsureVisibleCues(unittest.TestCase):
    """A zero-length cue never displays and makes strict SRT parsers reject the whole file."""

    def test_a_zero_length_cue_borrows_from_the_silence_before_it(self):
        cues = [{"start": 5.0, "end": 5.0, "text": "欸"}]
        self.assertEqual(ensure_visible_cues(cues), 1)
        self.assertGreaterEqual(cues[0]["end"] - cues[0]["start"], MIN_VISIBLE_DURATION - 1e-9)
        self.assertEqual(cues[0]["end"], 5.0, "borrowing backwards must not move the end")

    def test_it_never_reaches_back_into_the_previous_cue(self):
        cues = [
            {"start": 1.0, "end": 4.99, "text": "你好"},
            {"start": 5.0, "end": 5.0, "text": "欸"},
        ]
        ensure_visible_cues(cues)
        self.assertGreaterEqual(cues[1]["start"], 4.99 - 1e-9)
        self.assertGreater(cues[1]["end"], cues[1]["start"])

    def test_a_healthy_cue_is_untouched(self):
        cues = [{"start": 1.0, "end": 3.0, "text": "你好嗎"}]
        self.assertEqual(ensure_visible_cues(cues), 0)
        self.assertEqual((cues[0]["start"], cues[0]["end"]), (1.0, 3.0))

    def test_every_cue_ends_up_displayable(self):
        cues = [{"start": 2.0, "end": 2.0, "text": "a"} for _ in range(5)]
        ensure_visible_cues(cues)
        for cue in cues:
            self.assertGreaterEqual(
                cue["end"] - cue["start"], MIN_VISIBLE_DURATION - 1e-9,
                "a cue that never displays is invalid output, not merely short",
            )


class TestStripSentinels(unittest.TestCase):
    def test_sentinels_are_removed_and_fusion_is_counted(self):
        segments = [
            {"text": "你好" + REALIGN_SENTINEL},
            {"text": "早晨" + REALIGN_SENTINEL + "食咗飯未" + REALIGN_SENTINEL},
        ]
        fused = strip_sentinels(segments)
        self.assertEqual([s["text"] for s in segments], ["你好", "早晨食咗飯未"])
        self.assertEqual(fused, 1)

    def test_clean_output_reports_no_fusion(self):
        segments = [{"text": "你好" + REALIGN_SENTINEL}]
        self.assertEqual(strip_sentinels(segments), 0)


class TestSanitizeKeepsTranscriptOrder(unittest.TestCase):
    """_sanitize enforces bounds; it must never permute the transcript.

    Sorting by (start, end) looked harmless while the search behaved and was wrong the moment
    it did not: a run of lines the ASR text match could not place all land on the same instant,
    and the tie was then broken by their *ends*. On Police Story 2 that reversed four
    consecutive cues.
    """

    def test_lines_sharing_an_instant_keep_their_order(self):
        timings = [
            LineTiming(0, 100.0, 101.2, reason="unreadable"),
            LineTiming(1, 100.0, 100.6, reason="unreadable"),
            LineTiming(2, 100.0, 100.4, reason="unreadable"),
            LineTiming(3, 100.0, 101.2, reason="unreadable"),
        ]
        out = _sanitize(timings, 0.0, 200.0)
        self.assertEqual([t.index for t in out], [0, 1, 2, 3])

    def test_a_backwards_placement_is_clamped_not_reordered(self):
        timings = [LineTiming(0, 50.0, 51.0), LineTiming(1, 10.0, 11.0), LineTiming(2, 60.0, 61.0)]
        out = _sanitize(timings, 0.0, 200.0)
        self.assertEqual([t.index for t in out], [0, 1, 2])
        self.assertGreaterEqual(out[1].start, out[0].start)

    def test_output_is_still_monotonic_and_in_bounds(self):
        timings = [LineTiming(i, 300.0 - 10 * i, 301.0 - 10 * i) for i in range(8)]
        out = _sanitize(timings, 5.0, 250.0)
        self.assertEqual([t.index for t in out], list(range(8)))
        for a, b in zip(out, out[1:]):
            self.assertLessEqual(a.start, b.start)
        for t in out:
            self.assertGreaterEqual(t.start, 5.0)
            self.assertLessEqual(t.start, 250.0)


class TestSmearedSpanAnchoring(unittest.TestCase):
    """A smeared line keeps its *start*. Hanging it on the end was tried and is worse.

    The theory for the end was that it is the edge the evidence supports: a line whose audio
    sits later in the window dwells on blanks and then fires its characters when the speech
    arrives. That holds for one measured case and is false of the general smear, where the
    line belongs at the start of the window and the dwell runs forward. Backing out to the
    end reported every stranded line about a window late -- Doraemon p90 start error
    0.209s -> 425.7s, bluey worst cue 1.07s -> 10.71s. See _bounded_timing's docstring.
    """

    def test_the_start_is_kept_and_the_end_pulled_in(self):
        line = TranscriptLine(0, "May呀")
        timing = _bounded_timing(line, 405.04, 463.64, floor=399.0)
        self.assertFalse(timing.placed, "58s for a two-character line is not slow speech")
        self.assertAlmostEqual(timing.start, 405.04, places=2)
        self.assertLess(timing.end, 410.0)

    def test_the_floor_still_wins(self):
        line = TranscriptLine(0, "你好嗎" * 6)
        timing = _bounded_timing(line, 10.0, 40.0, floor=39.0)
        self.assertGreaterEqual(timing.start, 39.0)
        self.assertGreater(timing.end, timing.start)


class TestUnmatchedRuns(unittest.TestCase):
    """A run of unmatched lines is spread across the gap, not stacked on one instant.

    A text match fails in runs by its nature -- what defeats it is a stretch of repeated or
    near-identical short lines -- so placing each unmatched line immediately after the last
    matched one is the common case, not the rare one. On Police Story 2 four consecutive cues
    ended up sharing a timestamp that way.
    """

    def _asr(self, *segments):
        return [{"start": s, "end": e, "text": t} for s, e, t in segments]

    def test_a_run_comes_out_ordered_and_separated(self):
        lines = [TranscriptLine(0, "你好嗎")]
        lines += [TranscriptLine(i, "唔該") for i in range(1, 5)]
        lines.append(TranscriptLine(5, "幾好呀"))
        timings = assign_lines_via_asr(
            lines, self._asr((0.0, 2.0, "你好嗎"), (40.0, 42.0, "幾好呀")),
        )
        run = timings[1:5]
        self.assertTrue(all(not t.placed for t in run))
        for a, b in zip(run, run[1:]):
            self.assertGreater(b.start, a.start, "stacked lines are indistinguishable cues")

    def test_the_run_stays_inside_the_gap(self):
        lines = [TranscriptLine(0, "你好嗎")]
        lines += [TranscriptLine(i, "唔該") for i in range(1, 4)]
        lines.append(TranscriptLine(4, "幾好呀"))
        timings = assign_lines_via_asr(
            lines, self._asr((0.0, 2.0, "你好嗎"), (40.0, 42.0, "幾好呀")),
        )
        for t in timings[1:4]:
            self.assertGreaterEqual(t.start, timings[0].end - 1e-6)
            self.assertLessEqual(t.end, timings[4].start + 1e-6)

    def test_a_lone_unmatched_line_is_centred_in_its_gap(self):
        lines = [TranscriptLine(0, "你好嗎"), TranscriptLine(1, "May呀"),
                 TranscriptLine(2, "幾好呀")]
        timings = assign_lines_via_asr(
            lines, self._asr((0.0, 2.0, "你好嗎"), (60.0, 62.0, "幾好呀")),
        )
        self.assertGreater(timings[1].start, 20.0,
                           "jamming it against the near edge makes the error the whole gap")
        self.assertLess(timings[1].end, 45.0)

    def test_a_gap_too_small_to_share_still_produces_ordered_cues(self):
        lines = [TranscriptLine(0, "你好嗎")]
        lines += [TranscriptLine(i, "唔該警察叔叔") for i in range(1, 6)]
        lines.append(TranscriptLine(6, "幾好呀"))
        timings = assign_lines_via_asr(
            lines, self._asr((0.0, 2.0, "你好嗎"), (3.0, 5.0, "幾好呀")),
        )
        for a, b in zip(timings, timings[1:]):
            self.assertGreaterEqual(b.start, a.start - 1e-6)


class TestDetachedRunSize(unittest.TestCase):
    """Only a *short* detached run is discarded -- a real clause after a pause is not.

    Police Story 2 cue 18: the emission puts the line's first character at 296.120s (hand
    measured 296.114) and the comma three characters later then dwells 1.400s. Keeping
    whichever run was longer threw the correctly-aligned opening away and put the subtitle
    on screen 1.77s late.
    """

    def _cue(self, words):
        return {"start": words[0][1], "end": words[-1][2],
                "text": "".join(w for w, _, _ in words),
                "words": [{"word": w, "start": s, "end": e} for w, s, e in words]}

    def test_a_clause_before_a_real_pause_survives(self):
        cue = self._cue([
            ("無", 296.12, 296.24), ("錯", 296.24, 296.44), ("喇", 296.44, 296.48),
            ("我", 297.88, 297.96), ("哋", 297.96, 298.08), ("係", 298.08, 298.20),
            ("唔", 298.20, 298.28), ("需", 298.28, 298.40), ("要", 298.40, 298.52),
        ])
        self.assertEqual(tighten_cue_spans([cue]), 0)
        self.assertAlmostEqual(cue["start"], 296.12, places=2)

    def test_a_one_character_interjection_is_still_dropped(self):
        cue = self._cue([
            ("噢", 72.23, 72.40),
            ("係", 82.11, 82.27), ("島", 82.27, 82.59), ("嶼", 82.59, 82.75),
        ])
        self.assertEqual(tighten_cue_spans([cue]), 1)
        self.assertAlmostEqual(cue["start"], 82.11, places=2)

    def test_one_character_over_the_limit_is_a_clause(self):
        head = [("字", 10.0 + 0.1 * i, 10.1 + 0.1 * i) for i in range(MAX_DETACHED_RUN + 1)]
        tail = [("尾", 40.0 + 0.1 * i, 40.1 + 0.1 * i) for i in range(6)]
        cue = self._cue(head + tail)
        self.assertEqual(tighten_cue_spans([cue]), 0)


class TestCharTimesOverSpeech(unittest.TestCase):
    """Characters are laid over the speech in a chunk, not over its silence.

    Split-only chunking hands the ASR anchor a contiguous block of film, so spreading a
    segment's characters evenly across it puts them wherever the silence happens to be. On
    Police Story 2 the chunk at 382.5-409.8s held 37 characters over 27.3s and timed a line
    spoken at 391.4s to 382.5s -- nine seconds, which is a whole alignment chunk out.
    """

    def test_without_speech_spans_it_spreads_evenly(self):
        out = _char_times(0.0, 10.0, 5, None)
        self.assertEqual(len(out), 5)
        self.assertAlmostEqual(out[0][0], 0.0, places=3)
        self.assertAlmostEqual(out[-1][1], 10.0, places=3)

    def test_characters_land_inside_the_speech(self):
        out = _char_times(0.0, 30.0, 8, [[20.0, 28.0]])
        self.assertEqual(len(out), 8)
        for a, b in out:
            self.assertGreaterEqual(a, 20.0 - 1e-6)
            self.assertLessEqual(b, 28.0 + 1e-6)

    def test_two_regions_split_the_characters_by_duration(self):
        out = _char_times(0.0, 40.0, 12, [[0.0, 3.0], [30.0, 39.0]])
        self.assertEqual(len(out), 12)
        self.assertEqual(sum(1 for a, _ in out if a < 10.0), 3, "a quarter of the speech")
        self.assertTrue(all(a >= 30.0 - 1e-6 for a, _ in out[3:]))

    def test_speech_outside_the_segment_is_clipped_away(self):
        out = _char_times(10.0, 20.0, 4, [[0.0, 5.0]])
        self.assertEqual(len(out), 4)
        self.assertAlmostEqual(out[0][0], 10.0, places=3)

    def test_the_anchor_uses_them_end_to_end(self):
        lines = [TranscriptLine(0, "你好嗎"), TranscriptLine(1, "幾好呀")]
        asr = [{"start": 0.0, "end": 30.0, "text": "你好嗎幾好呀"}]
        chunks = [{"start": 0.0, "end": 30.0, "speech": [[24.0, 30.0]]}]
        timings = assign_lines_via_asr(lines, asr, vad_segments=chunks)
        self.assertGreater(timings[0].start, 23.0,
                           "without the speech spans this lands at 0.0s")


class TestImplausibleCues(unittest.TestCase):
    """The mirror of _bounded_timing, applied to the final timings.

    When the coarse pass puts a line in a chunk that does not hold its speech, alignment still
    places every token somewhere. Police Story 2: an eight-character line came out as an
    18.5s cue *at score 0.983*, so confidence cannot see this -- but the geometry can.
    """

    def _cue(self, start, end, text, score=0.9):
        return {"start": start, "end": end, "text": text,
                "words": [{"word": c, "start": start, "end": end, "score": score}
                          for c in text]}

    def test_a_cue_too_long_for_its_text_is_flagged(self):
        cues = [self._cue(372.9, 391.44, "捉賊又唔見你品叻", score=0.983)]
        found = find_implausible_cues(cues)
        self.assertEqual(len(found), 1)
        self.assertIn("per character", found[0]["realign_detail"])

    def test_a_cue_that_aligned_on_nothing_is_flagged(self):
        cues = [self._cue(350.92, 351.12, "由今日開始", score=0.0)]
        found = find_implausible_cues(cues)
        self.assertEqual(len(found), 1)
        self.assertIn("score", found[0]["realign_detail"])

    def test_ordinary_cues_are_left_alone(self):
        cues = [
            self._cue(10.0, 11.5, "你好嗎幾好呀"),
            self._cue(11.5, 12.0, "係啦"),
            self._cue(12.0, 14.5, "邊到需要人幫手"),
        ]
        self.assertEqual(find_implausible_cues(cues), [])

    def test_a_short_cue_may_still_be_drawn_out(self):
        self.assertEqual(find_implausible_cues([self._cue(10.0, 12.5, "吓")]), [])

    def test_the_score_floor_is_where_MIN_CUE_SCORE_says(self):
        cues = [self._cue(10.0, 11.0, "你好嗎", score=MIN_CUE_SCORE + 0.01)]
        self.assertEqual(find_implausible_cues(cues), [])


class TestChunkAudioFloor(unittest.TestCase):
    """Every chunk must carry enough audio for the encoder to accept it.

    The feature extractor needs one whole analysis window and reports a shortfall as
    "negative dimensions are not allowed" from inside numpy. A chunk that short is reachable
    whenever the search runs out of audio and strands the remaining lines on the final frame:
    the group's bounds collapse onto file_end and the slice comes back empty. On the Doraemon
    fixture, where the acoustic anchor genuinely does run out of film, that aborted the run.
    """

    def _chunks(self, timings, duration=20.0):
        lines = [TranscriptLine(t.index, "你好嗎") for t in timings]
        segs = [{"start": 0.0, "end": duration,
                 "audio": np.zeros(int(duration * SR), dtype=np.float32)}]
        return build_align_input(lines, timings, segs, 28.0)[0]

    def test_lines_stranded_on_the_last_frame_still_get_audio(self):
        timings = [LineTiming(0, 5.0, 6.0)] + [
            LineTiming(i, 20.0, 20.04, reason="unreadable") for i in range(1, 5)
        ]
        for chunk in self._chunks(timings):
            self.assertGreaterEqual(
                len(chunk["audio"]), int(MIN_CHUNK_DURATION * SR),
                f"chunk at {chunk['start']}s carries {len(chunk['audio'])} samples",
            )

    def test_the_declared_span_still_matches_the_audio(self):
        timings = [LineTiming(0, 5.0, 6.0), LineTiming(1, 20.0, 20.04, reason="unreadable")]
        for chunk in self._chunks(timings):
            self.assertAlmostEqual(
                len(chunk["audio"]) / SR, chunk["end"] - chunk["start"], places=2,
                msg="a padded chunk whose timestamps still claim the old span would "
                    "misplace every character in it",
            )

    def test_ordinary_chunks_are_untouched(self):
        timings = [LineTiming(i, 2.0 * i, 2.0 * i + 1.5) for i in range(5)]
        chunks = self._chunks(timings)
        self.assertTrue(all(c["end"] - c["start"] > MIN_CHUNK_DURATION for c in chunks))


class TestCueOrder(unittest.TestCase):
    """An SRT whose cues are not in start order is rejected outright, so this is validity.

    _sanitize orders the coarse placements, but the final timings come from forced alignment
    inside each chunk and it has no such constraint. Where the search ran out of audio and
    stranded the tail of the transcript, consecutive chunks collapse onto the last fraction of
    a second and their cues can come back interleaved -- one pair did on the Doraemon fixture.
    """

    def test_a_backwards_cue_is_clamped_forward(self):
        segs = [{"start": 10.0, "end": 11.0}, {"start": 9.5, "end": 10.2}]
        self.assertEqual(enforce_cue_order(segs), 1)
        self.assertEqual(segs[0]["start"], 10.0)
        self.assertGreaterEqual(segs[1]["start"], segs[0]["start"])

    def test_cues_are_never_reordered(self):
        segs = [{"start": 10.0, "end": 11.0, "text": "a"},
                {"start": 9.5, "end": 10.2, "text": "b"},
                {"start": 12.0, "end": 13.0, "text": "c"}]
        enforce_cue_order(segs)
        self.assertEqual([s["text"] for s in segs], ["a", "b", "c"])

    def test_an_ordered_list_is_untouched(self):
        segs = [{"start": 1.0, "end": 2.0}, {"start": 2.0, "end": 3.0}]
        self.assertEqual(enforce_cue_order(segs), 0)

    def test_the_result_is_always_non_decreasing(self):
        segs = [{"start": s, "end": s + 0.1} for s in (5.0, 4.0, 4.5, 3.0, 9.0)]
        enforce_cue_order(segs)
        for a, b in zip(segs, segs[1:]):
            self.assertGreaterEqual(b["start"], a["start"])
            self.assertGreaterEqual(b["end"], b["start"])


class TestEmissionTimeline(unittest.TestCase):
    """One timeline over the file, sliceable across the joins between chunks.

    The joins are an encoder limit, not a property of the audio, and 22-44% of the spans the
    placement search wants to align in one piece cross one. A slice has to be exact anyway:
    chunks can differ in frame rate by a couple of percent, so stretching one rate across a
    join would smear every timestamp after it.
    """

    def _timeline(self, spec):
        """spec: [(start, end, n_frames)] -- deliberately uneven frame rates."""
        chunks, ems = [], []
        for start, end, n in spec:
            chunks.append({"start": start, "end": end, "audio": np.zeros(4, dtype=np.float32)})
            ems.append((torch.zeros((n, len(VOCAB))), n / (end - start)))
        by = {c["start"]: e for c, e in zip(chunks, ems)}
        return EmissionTimeline(chunks, lambda segs: [by[s["start"]] for s in segs])

    def test_a_slice_inside_one_chunk_is_exact(self):
        tl = self._timeline([(0.0, 10.0, 250)])
        emission, times = tl.slice(2.0, 4.0)
        self.assertEqual(emission.size(0), len(times))
        self.assertAlmostEqual(times[0], 2.0, places=2)
        self.assertLess(times[-1], 4.041)

    def test_a_slice_across_a_join_is_continuous(self):
        # 25 fps then 20 fps: a single global rate would drift after the join.
        tl = self._timeline([(0.0, 10.0, 250), (10.0, 20.0, 200)])
        emission, times = tl.slice(8.0, 12.0)
        self.assertEqual(emission.size(0), len(times))
        self.assertTrue(all(b > a for a, b in zip(times, times[1:])), "times must increase")
        self.assertAlmostEqual(times[0], 8.0, places=2)
        # The frame starting exactly at t1 is included, as the old per-chunk slice was
        # also generous by up to a frame; what matters is that it is one frame, not a rate
        # error accumulated across the join.
        self.assertAlmostEqual(times[-1], 12.0, places=2)
        # ...and each side keeps its own rate rather than an averaged one.
        self.assertAlmostEqual(float(times[1] - times[0]), 0.04, places=3)
        self.assertAlmostEqual(float(times[-1] - times[-2]), 0.05, places=3)

    def test_a_chunk_is_encoded_at_most_once(self):
        tl = self._timeline([(0.0, 10.0, 250), (10.0, 20.0, 250), (20.0, 30.0, 250)])
        tl.slice(0.0, 30.0)
        tl.slice(5.0, 25.0)
        tl.slice(0.0, 30.0)
        self.assertEqual(tl.computed, 3, "the second pass must reuse the first pass's work")

    def test_only_the_chunks_a_slice_needs_are_encoded(self):
        tl = self._timeline([(0.0, 10.0, 250), (10.0, 20.0, 250), (20.0, 30.0, 250)])
        tl.slice(0.5, 1.0)
        self.assertEqual(tl.computed, 1)

    def test_a_request_narrower_than_a_frame_still_returns_one(self):
        tl = self._timeline([(0.0, 10.0, 250)])
        emission, times = tl.slice(3.001, 3.002)
        self.assertEqual(emission.size(0), 1)
        self.assertEqual(len(times), 1)

    def test_emissions_are_held_as_float16_but_handed_out_as_float32(self):
        tl = self._timeline([(0.0, 10.0, 250)])
        emission, _ = tl.slice(0.0, 10.0)
        self.assertEqual(emission.dtype, torch.float32)
        self.assertEqual(tl._emissions[0].dtype, np.float16)


class TestAcquireProgress(unittest.TestCase):
    """The search must always move, whatever the audio does.

    The old sweep had no lower bound on progress in *time*: a window that consumed almost
    nothing advanced the pointer by almost nothing, and the loop could grind. The overlap is
    also allowed to be larger than a caller's window, which would step time backwards.
    """

    def _timeline(self, chunks, emissions):
        by = {c["start"]: (e, FPS) for c, e in zip(chunks, emissions)}
        return EmissionTimeline(chunks, lambda segs: [by[s["start"]] for s in segs])

    def test_it_terminates_on_audio_that_explains_nothing(self):
        chunks = [_chunk(0.0, 250), _chunk(10.0, 250)]
        emissions = [_emission([None] * 250), _emission([None] * 250)]
        lines = [TranscriptLine(i, "abcd") for i in range(8)]
        timings = assign_lines(
            lines, chunks, self._timeline(chunks, emissions), VOCAB, "yue", blank_id=BLANK,
            window_seconds=10.0, commit_margin=2.0,
        )
        self.assertEqual(len(timings), 8)
        self.assertEqual([t.index for t in timings], list(range(8)))

    def test_a_window_smaller_than_the_overlap_still_advances(self):
        # ANCHOR_WINDOW_OVERLAP is 20s; a 6s window must not step time backwards.
        chunks = [_chunk(0.0, 250), _chunk(10.0, 250)]
        emissions = [_emission([None] * 250), _emission([None] * 250)]
        lines = [TranscriptLine(i, "abcd") for i in range(4)]
        timings = assign_lines(
            lines, chunks, self._timeline(chunks, emissions), VOCAB, "yue", blank_id=BLANK,
            window_seconds=6.0, commit_margin=1.0,
        )
        self.assertEqual(len(timings), 4)


class TestAnchorSanitising(unittest.TestCase):
    """Anchors that cannot all be true at once: the chain keeps the heaviest set that can."""

    def _lines(self, n, chars=8):
        return [TranscriptLine(i, "a" * chars) for i in range(n)]

    def test_an_anchor_that_goes_backwards_in_time_is_dropped(self):
        lines = self._lines(4)
        anchors = [(0, 1.0, 2.0, 0.9), (1, 60.0, 61.0, 0.9), (2, 5.0, 6.0, 0.9),
                   (3, 90.0, 91.0, 0.9)]
        chain = sanitise_anchors(anchors, lines)
        starts = [a[1] for a in chain]
        self.assertEqual(starts, sorted(starts))
        self.assertEqual([a[0] for a in chain], sorted(a[0] for a in chain))

    def test_a_gap_too_small_for_the_text_between_is_rejected(self):
        # 200 characters cannot be spoken in a tenth of a second, so these two cannot both
        # be right; the chain must not contain both.
        lines = [TranscriptLine(0, "a" * 8), TranscriptLine(1, "b" * 200),
                 TranscriptLine(2, "c" * 8)]
        anchors = [(0, 1.0, 2.0, 0.9), (2, 2.1, 3.0, 0.9)]
        self.assertEqual(len(sanitise_anchors(anchors, lines)), 1)

    def test_a_long_wordless_gap_is_allowed(self):
        # The documented trap: only the lower bound is a real constraint. Films are silent
        # for minutes at a time and clamping that cascades.
        lines = self._lines(2)
        anchors = [(0, 1.0, 2.0, 0.9), (1, 600.0, 601.0, 0.9)]
        self.assertEqual(len(sanitise_anchors(anchors, lines)), 2)

    def test_no_anchors_is_not_an_error(self):
        self.assertEqual(sanitise_anchors([], self._lines(3)), [])


class TestNoVocabularyLines(unittest.TestCase):
    """A line the align model has no character for is reported as such, not as a failure."""

    def _run(self, texts):
        chunks = [_chunk(0.0, 250)]
        labels = _labels({20: "abc", 60: "abcd", 120: "abc"}, 250)
        emissions = [_emission(labels)]
        by = {c["start"]: (e, FPS) for c, e in zip(chunks, emissions)}
        timeline = EmissionTimeline(chunks, lambda segs: [by[s["start"]] for s in segs])
        lines = [TranscriptLine(i, t) for i, t in enumerate(texts)]
        return assign_lines(lines, chunks, timeline, VOCAB, "yue", blank_id=BLANK,
                            window_seconds=10.0, commit_margin=1.0)

    def test_it_is_named_no_vocabulary_rather_than_isolated(self):
        # "zz" holds nothing in the vocabulary; the full stop is a pause token, not evidence.
        timings = self._run(["abc", "zz。", "abcd"])
        self.assertEqual(timings[1].reason, REASON_NO_VOCABULARY)

    def test_it_is_still_placed_between_its_neighbours(self):
        timings = self._run(["abc", "zz。", "abcd"])
        self.assertLessEqual(timings[0].start, timings[1].start)
        self.assertLessEqual(timings[1].start, timings[2].start + 1e-6)

    def test_every_line_comes_back_exactly_once_and_in_order(self):
        timings = self._run(["abc", "zz。", "abcd", "zzzz"])
        self.assertEqual([t.index for t in timings], [0, 1, 2, 3])


class TestDwellingCharacters(unittest.TestCase):
    """A character that merely waited must not keep the wait as its span.

    CTC fires a character on a frame or two and then sits on it, emitting blank, until the
    next thing arrives; merge_repeats reports that whole wait as the character's span. On
    Police Story 2 one character held 13.9s (the first character of a line ran 330.26-344.19 while the next ran 344.19-344.35),
    which put the subtitle on screen fourteen seconds before anybody spoke. No gap-between-
    characters test can see it, and neither can an energy envelope -- the pause there is room
    tone a dozen dB under the dialogue, not silence.
    """

    def _emission(self, n, peaks):
        """Log-probs over VOCAB: blank everywhere, a named token peaking on given frames."""
        data = np.full((n, len(VOCAB)), math.log(0.02 / len(VOCAB)), dtype=np.float32)
        data[:, BLANK] = math.log(0.95)
        for frame, ch in peaks.items():
            data[frame, :] = math.log(0.02 / len(VOCAB))
            data[frame, VOCAB[ch]] = math.log(0.95)
        return torch.from_numpy(data)

    def test_a_dwelling_character_moves_to_its_peak(self):
        # 'a' holds 350 frames (14s at 25fps) but only fires at frame 340.
        chars = [Segment("a", 0, 350, 0.9), Segment("b", 350, 354, 0.9)]
        emission = self._emission(360, {340: "a", 351: "b"})
        moved = _reseat_dwelling_chars(chars, emission, [VOCAB["a"], VOCAB["b"]], BLANK, 0.04)
        self.assertEqual(moved, 1)
        self.assertEqual(chars[0].start, 340)
        self.assertLessEqual(chars[0].end, 350, "it must not run into the next character")

    def test_a_dwelling_character_at_the_end_is_pulled_back(self):
        chars = [Segment("a", 0, 4, 0.9), Segment("b", 4, 300, 0.9)]
        emission = self._emission(310, {1: "a", 6: "b"})
        self.assertEqual(
            _reseat_dwelling_chars(chars, emission, [VOCAB["a"], VOCAB["b"]], BLANK, 0.04), 1)
        self.assertEqual(chars[1].start, 6)
        self.assertLess(chars[1].end, 300, "the trailing wait must not stay in the cue")

    def test_ordinary_characters_are_untouched(self):
        chars = [Segment("a", 0, 4, 0.9), Segment("b", 4, 9, 0.9), Segment("c", 9, 13, 0.9)]
        before = [(c.start, c.end) for c in chars]
        emission = self._emission(20, {1: "a", 5: "b", 10: "c"})
        tokens = [VOCAB["a"], VOCAB["b"], VOCAB["c"]]
        self.assertEqual(_reseat_dwelling_chars(chars, emission, tokens, BLANK, 0.04), 0)
        self.assertEqual([(c.start, c.end) for c in chars], before)

    def test_a_drawn_out_syllable_under_the_limit_is_kept(self):
        # Just under MAX_CHAR_DWELL_SECONDS: a held particle is a character, not a wait.
        held = int(MAX_CHAR_DWELL_SECONDS / 0.04) - 2
        chars = [Segment("a", 0, held, 0.9), Segment("b", held, held + 4, 0.9)]
        emission = self._emission(held + 10, {2: "a", held + 1: "b"})
        self.assertEqual(
            _reseat_dwelling_chars(chars, emission, [VOCAB["a"], VOCAB["b"]], BLANK, 0.04), 0)

    def test_characters_stay_ordered_and_non_empty(self):
        chars = [Segment("a", 0, 300, 0.9), Segment("b", 300, 600, 0.9),
                 Segment("c", 600, 604, 0.9)]
        emission = self._emission(610, {290: "a", 590: "b", 601: "c"})
        tokens = [VOCAB["a"], VOCAB["b"], VOCAB["c"]]
        _reseat_dwelling_chars(chars, emission, tokens, BLANK, 0.04)
        for a, b in zip(chars, chars[1:]):
            self.assertLessEqual(a.start, b.start)
        for c in chars:
            self.assertGreater(c.end, c.start)

    def test_a_blank_token_is_left_alone(self):
        # Punctuation maps to blank and is *meant* to hold the pause; that is what
        # release_from spends. Only real characters are re-seated.
        chars = [Segment("a", 0, 4, 0.9), Segment("，", 4, 300, 0.9)]
        emission = self._emission(310, {1: "a"})
        self.assertEqual(
            _reseat_dwelling_chars(chars, emission, [VOCAB["a"], BLANK], BLANK, 0.04), 0)


if __name__ == "__main__":
    unittest.main()

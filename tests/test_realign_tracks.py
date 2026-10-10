"""--realign: chunk reach, line-break splits and the top-of-screen track.

All three come from one ReZero broadcast episode run in mode adjust (hand-checked to 05:20):

* a line after the 93 s opening came out 46 s early, inside the theme song, because its
  alignment chunk began in the middle of the gap and was capped 19 s short of the line;
* two-line cues held two sentences on screen across a real pause, which the checker split
  by hand every time;
* ``{\\an8}`` cues lost their tag and were aligned in sequence with the line they overlap.
"""

import unittest

import numpy as np
import torch

from cantocaptions_ai.pipeline import timefit
from cantocaptions_ai.pipeline.realign import (
    LINE_SPLIT_GAP,
    SHORT_PIECE_EXTRA,
    MAX_INTERNAL_GAP,
    REALIGN_SENTINEL,
    REASON_ISOLATED,
    EmissionTimeline,
    LineTiming,
    TranscriptLine,
    _overlay_transform,
    _with_overlay,
    build_align_input,
    load_transcript_lines,
    place_overlay,
    segments_from_timings,
    split_at_pauses,
    split_mode,
    split_tracks,
)

SR = 16000
FPS = 25.0
VOCAB = {"[pad]": 0, "a": 1, "b": 2, "c": 3, "d": 4}
BLANK = 0
TOP = r"{\an8}"


def _chunk(start, num_frames):
    end = start + num_frames / FPS
    return {"start": start, "end": end,
            "audio": np.zeros(int(round((end - start) * SR)), dtype=np.float32)}


def _emission(labels, confident=-0.05, other=-12.0):
    data = np.full((len(labels), len(VOCAB)), other, dtype=np.float32)
    for i, label in enumerate(labels):
        data[i, BLANK if label is None else VOCAB[label]] = confident
    return torch.from_numpy(data)


def _timeline(chunks, emissions):
    by_start = {c["start"]: (e, FPS) for c, e in zip(chunks, emissions)}
    return EmissionTimeline(chunks, lambda segs: [by_start[s["start"]] for s in segs])


# --- Chunk reach ----------------------------------------------------------------------

class TestChunkReach(unittest.TestCase):
    """The ReZero geometry: a line, a 93 s opening with nothing in it, the next line."""

    def setUp(self):
        self.source = [_chunk(0.0, 200 * 25)]   # 200 s of contiguous audio
        self.lines = [TranscriptLine(i, t) for i, t in enumerate(["激氣啊", "點解呢個世界", "到底點解啊"])]
        self.timings = [LineTiming(0, 61.4, 62.3), LineTiming(1, 155.95, 159.95),
                        LineTiming(2, 159.96, 163.7)]

    def _chunk_of(self, chunks, transcript, line):
        k = 0
        for chunk, segment in zip(chunks, transcript):
            for _span in segment["cue_spans"]:
                if k == line:
                    return chunk
                k += 1
        raise AssertionError(f"line {line} is in no chunk")

    def test_a_line_after_a_wide_gap_lies_inside_its_own_chunk(self):
        """The cap used to come off the end: [mid-gap, mid-gap + 28 s] ended 19 s early."""
        chunks, transcript = build_align_input(self.lines, self.timings, self.source, 28.0)
        chunk = self._chunk_of(chunks, transcript, 1)
        self.assertLessEqual(chunk["start"], 155.95)
        self.assertGreaterEqual(chunk["end"], 159.95)
        self.assertLessEqual(chunk["end"] - chunk["start"], 28.0 + 1e-6)

    def test_reach_keeps_the_gap_out_of_both_neighbouring_chunks(self):
        chunks, transcript = build_align_input(
            self.lines, self.timings, self.source, 28.0, reach=2.0)
        before, after = self._chunk_of(chunks, transcript, 0), self._chunk_of(chunks, transcript, 1)
        self.assertLessEqual(before["end"], 62.3 + 2.0 + 1e-6)
        self.assertGreaterEqual(after["start"], 155.95 - 2.0 - 1e-6)

    def test_reach_changes_nothing_across_a_narrow_gap(self):
        # A gap under twice the reach already ends mid-gap, so the boundary is shared.
        timings = [LineTiming(0, 10.0, 11.0), LineTiming(1, 12.0, 13.0), LineTiming(2, 14.0, 15.0)]
        plain = build_align_input(self.lines, timings, self.source, 28.0)
        reached = build_align_input(self.lines, timings, self.source, 28.0, reach=2.0)
        self.assertEqual([(c["start"], c["end"]) for c in plain[0]],
                         [(c["start"], c["end"]) for c in reached[0]])

    def test_reach_never_leaves_a_wide_gap_inside_one_chunk(self):
        # Two lines either side of an 8 s gap fit one 28 s chunk easily. With the gap inside
        # it, forced alignment pulled the later line 8.9 s back into the background speech
        # an {\an8} cue had left there once it moved to its own track.
        timings = [LineTiming(0, 95.5, 98.0), LineTiming(1, 106.1, 107.0),
                   LineTiming(2, 107.0, 107.9)]
        chunks, transcript = build_align_input(
            self.lines, timings, self.source, 28.0, reach=2.0)
        self.assertEqual([len(t["cue_spans"]) for t in transcript], [1, 2])
        self.assertLessEqual(chunks[0]["end"], 98.0 + 2.0 + 1e-6)
        self.assertGreaterEqual(chunks[1]["start"], 106.1 - 2.0 - 1e-6)
        # Without a reach nothing changes: one chunk, as before.
        _, plain = build_align_input(self.lines, timings, self.source, 28.0)
        self.assertEqual([len(t["cue_spans"]) for t in plain], [3])

    def test_chunks_stay_sorted_and_disjoint_with_a_reach(self):
        chunks, _ = build_align_input(self.lines, self.timings, self.source, 28.0, reach=2.0)
        for a, b in zip(chunks, chunks[1:]):
            self.assertLessEqual(a["end"], b["start"] + 1e-6)


# --- Splitting at a line break ---------------------------------------------------------

def _word(ch, start, end, peak=0.99):
    return {"word": ch, "start": start, "end": end, "score": peak, "peak": peak}


def _two_line_cue(head, tail, gap, *, head_start=10.0, rate=0.15, peak=0.99, sentinel=True):
    """A cue of two lines whose characters run at *rate* s each, *gap* seconds apart."""
    words, t = [], head_start
    for ch in head:
        words.append(_word(ch, t, t + rate, peak))
        t += rate
    t += gap
    for ch in tail:
        words.append(_word(ch, t, t + rate, peak))
        t += rate
    text = f"{head}\n{tail}"
    if sentinel:
        words.append({"word": REALIGN_SENTINEL, "start": t, "end": t + 0.6, "score": 1.0,
                      "peak": 1.0})
        text += REALIGN_SENTINEL
    return {"start": head_start, "end": t + 0.6, "text": text, "words": words}


class TestSplitLineBreaks(unittest.TestCase):
    def test_a_pause_at_the_break_splits_the_cue(self):
        cue = _two_line_cue("雖然真係好難相信啊", "但係唔可以否認", 0.44)
        out, cuts = split_at_pauses([cue], align_padding=0.04, align_release=0.64)
        self.assertEqual(cuts, 1)
        self.assertEqual([s["text"] for s in out],
                         ["雖然真係好難相信啊", "但係唔可以否認" + REALIGN_SENTINEL])
        head, tail = out
        tail_start = cue["words"][9]["start"]
        self.assertAlmostEqual(tail["start"], tail_start, places=3)
        # Released into the pause, stopping align_padding short of the tail -- the way the
        # checker timed every one of these by hand.
        self.assertAlmostEqual(head["end"], tail_start - 0.04, places=3)
        self.assertAlmostEqual(head["start"], cue["start"], places=3)
        self.assertAlmostEqual(tail["end"], cue["end"], places=3)
        self.assertEqual(len(head["words"]) + len(tail["words"]), len(cue["words"]))
        self.assertTrue(any(n.startswith("split_line:") for n in head["notes"]))

    def test_a_short_release_does_not_reach_the_tail(self):
        cue = _two_line_cue("既然我已經返返過去", "即係話我而家喺度", 0.8)
        (head, tail), _ = split_at_pauses([cue], align_padding=0.04, align_release=0.4)
        self.assertAlmostEqual(head["end"], cue["words"][8]["end"] + 0.4, places=3)
        self.assertLess(head["end"], tail["start"])

    def test_a_wrapped_sentence_is_left_whole(self):
        # 好啲 / 嘅理由啊: one sentence wrapped onto two lines, no pause at the break.
        cue = _two_line_cue("有心要講大話都做個好啲", "嘅理由啊", 0.08)
        out, cuts = split_at_pauses([cue])
        self.assertEqual((cuts, len(out)), (0, 1))
        self.assertIs(out[0], cue)

    def test_the_threshold_is_the_gap(self):
        just_under = _two_line_cue("今日天氣真係幾好", "不如出去行下街", LINE_SPLIT_GAP - 0.05)
        just_over = _two_line_cue("今日天氣真係幾好", "不如出去行下街", LINE_SPLIT_GAP + 0.05)
        self.assertEqual(split_at_pauses([just_under])[1], 0)
        self.assertEqual(split_at_pauses([just_over])[1], 1)

    def test_a_short_line_needs_a_longer_pause(self):
        # 我唔知呀 / 唔講喇 at 0.44 s: a short exchange the finished file kept as one cue.
        # The same pause splits two full lines, and a longer one splits the short pair too.
        short = lambda gap: _two_line_cue("我唔知呀", "唔講喇", gap)
        self.assertEqual(split_at_pauses([short(0.44)])[1], 0)
        self.assertEqual(split_at_pauses([short(LINE_SPLIT_GAP + SHORT_PIECE_EXTRA + 0.05)])[1], 1)
        full = _two_line_cue("雖然真係好難相信啊", "但係唔可以否認", 0.44)
        self.assertEqual(split_at_pauses([full])[1], 1)

    def test_a_dashed_two_speaker_pair_is_never_split(self):
        cue = _two_line_cue("-講真嘠？", "-講真㗎！", 1.0)
        self.assertEqual(split_at_pauses([cue])[1], 0)

    def test_a_half_the_model_did_not_hear_is_not_split_off(self):
        # A fabricated gap -- one half aligned onto audio that is not it -- looks exactly
        # like a real pause on the timings alone.
        cue = _two_line_cue("我發咗瘋啫", "好多謝你啊阿叔", 2.56, peak=0.3)
        self.assertEqual(split_at_pauses([cue])[1], 0)

    def test_a_short_half_far_from_the_rest_is_a_misplacement_not_a_cue(self):
        far = _two_line_cue("喂", "你喺度做乜嘢啊", MAX_INTERNAL_GAP + 0.5)
        near = _two_line_cue("喂", "你喺度做乜嘢啊", MAX_INTERNAL_GAP - 0.1)
        self.assertEqual(split_at_pauses([far])[1], 0)
        self.assertEqual(split_at_pauses([near])[1], 1)

    def test_a_cue_already_in_doubt_is_not_split(self):
        cue = _two_line_cue("今日天氣", "幾好喎", 1.0)
        cue["realign_reason"] = REASON_ISOLATED
        self.assertEqual(split_at_pauses([cue])[1], 0)

    def test_three_lines_split_at_each_qualifying_break(self):
        a = _two_line_cue("今日天氣", "幾好喎", 0.6)
        # Append a third line after another pause onto the same cue.
        t = a["words"][-2]["end"] + 0.6
        a["words"] = a["words"][:-1] + [_word(c, t + 0.15 * k, t + 0.15 * (k + 1))
                                        for k, c in enumerate("不如出去行下")]
        a["text"] = "今日天氣\n幾好喎\n不如出去行下"
        a["end"] = a["words"][-1]["end"] + 0.5
        out, cuts = split_at_pauses([a])
        self.assertEqual(cuts, 2)
        self.assertEqual([s["text"] for s in out], ["今日天氣", "幾好喎", "不如出去行下"])
        for x, y in zip(out, out[1:]):
            self.assertLessEqual(x["end"], y["start"])

    def test_other_fields_survive_onto_both_pieces(self):
        cue = _two_line_cue("今日天氣", "幾好喎", 0.6)
        cue.update(style_tags=TOP, speaker="S0")
        out, _ = split_at_pauses([cue])
        self.assertTrue(all(s["style_tags"] == TOP and s["speaker"] == "S0" for s in out))

    def test_auto_means_pauses_in_adjust_only(self):
        self.assertEqual(split_mode("auto", "adjust"), "pauses")
        self.assertEqual(split_mode("auto", "transcript"), "off")
        self.assertEqual(split_mode("lines", "transcript"), "lines")
        self.assertEqual(split_mode("off", "adjust"), "off")


def _run_on(text, onsets, *, peak=0.99, untimed=()):
    """A one-line cue whose characters start at *onsets*, each span running to the next onset.

    That is how the trellis lays out characters written together: any silence between two of
    them is folded into the first one's span, so the spans touch and only the onsets move.
    Characters at the positions in *untimed* have no timing (out of vocabulary).
    """
    words = []
    for k, (ch, start) in enumerate(zip(text, onsets)):
        end = onsets[k + 1] if k + 1 < len(onsets) else start + 0.15
        words.append({"word": ch} if k in untimed else _word(ch, start, end, peak))
    return {"start": onsets[0], "end": onsets[-1] + 0.6, "text": text, "words": words}


def _onsets(n, pause_after=None, pause=0.0, start=10.0, rate=0.16):
    out, t = [], start
    for k in range(n):
        out.append(round(t, 3))
        t += rate + (pause if k == pause_after else 0.0)
    return out


class TestSplitAtPauses(unittest.TestCase):
    """A pause between two words written together, with no line break to say so."""

    def test_a_pause_between_two_words_splits_the_cue(self):
        # 喺呢個異世界享受人生 / 諗起都傻歪歪嘅: the checker's cut, at 0.76 s onset to onset.
        text = "喺呢個異世界享受人生諗起都傻歪歪嘅"
        onsets = _onsets(len(text), pause_after=9, pause=0.6)
        out, cuts = split_at_pauses([_run_on(text, onsets)], align_padding=0.04)
        self.assertEqual(cuts, 1)
        self.assertEqual([s["text"] for s in out], ["喺呢個異世界享受人生", "諗起都傻歪歪嘅"])
        self.assertAlmostEqual(out[1]["start"], onsets[10], places=3)
        self.assertAlmostEqual(out[0]["end"], onsets[10] - 0.04, places=3)
        self.assertTrue(any(n.startswith("split_pause:") for n in out[0]["notes"]))

    def test_a_phrase_pause_is_kept_by_default_and_cut_on_request(self):
        # 0.5 s onset to onset: one hand-finished episode split these run-ons and the other
        # kept them whole, so the default leaves them and --realign_pause_gap 0.4 cuts them.
        text = "你哋三個而家投降仲嚟得切"
        onsets = _onsets(len(text), pause_after=5, pause=0.34)     # 你哋三個而家 | 投降…
        self.assertEqual(split_at_pauses([_run_on(text, onsets)])[1], 0)
        self.assertEqual(split_at_pauses([_run_on(text, onsets)], pause_gap=0.4)[1], 1)

    def test_ordinary_speech_is_left_whole(self):
        text = "喺呢個異世界享受人生諗起都傻歪歪嘅"
        self.assertEqual(split_at_pauses([_run_on(text, _onsets(len(text)))])[1], 0)

    def test_a_drawn_out_syllable_before_a_particle_is_not_a_pause(self):
        # 唔好講笑喇好痛|啊: one character on the far side is a held syllable, not a cut.
        text = "唔好講笑喇好痛啊"
        onsets = _onsets(len(text), pause_after=6, pause=0.8)
        self.assertEqual(split_at_pauses([_run_on(text, onsets)])[1], 0)

    def test_each_piece_keeps_three_characters(self):
        text = "就算莎緹拉唔識我都好"
        short = _onsets(len(text), pause_after=1, pause=0.7)    # 就算 | 莎緹拉…
        long = _onsets(len(text), pause_after=2, pause=0.8)     # 就算莎 | 緹拉…
        self.assertEqual(split_at_pauses([_run_on(text, short)])[1], 0)
        self.assertEqual(split_at_pauses([_run_on(text, long)])[1], 1)

    def test_an_untimed_character_between_is_not_counted_as_pause(self):
        # The onset gap across 僵 includes 僵 itself, which the align model could not time.
        text = "問斷你都幾夠僵㗎喎好"
        onsets = _onsets(len(text), pause_after=5, pause=0.8)
        self.assertEqual(split_at_pauses([_run_on(text, onsets, untimed={6})])[1], 0)

    def test_punctuation_is_not_a_bare_boundary(self):
        # A comma is a pause token of its own; a subtitler keeps most such clauses together.
        text = "冇咩啊，下次我一定會嚟"
        onsets = _onsets(len(text), pause_after=3, pause=0.8)
        self.assertEqual(split_at_pauses([_run_on(text, onsets)])[1], 0)

    def test_lines_mode_ignores_bare_pauses(self):
        text = "喺呢個異世界享受人生諗起都傻歪歪嘅"
        onsets = _onsets(len(text), pause_after=9, pause=0.6)
        self.assertEqual(split_at_pauses([_run_on(text, onsets)], mode="lines")[1], 0)
        self.assertEqual(split_at_pauses([_run_on(text, onsets)], mode="off")[1], 0)

    def test_the_strongest_pause_is_cut_first_and_each_piece_tested_again(self):
        text = "頭先有個同你一樣嘅死窮鬼幫我揾返走失咗個女"
        onsets = _onsets(len(text))
        # 0.8 s after 鬼 and 0.9 s after 揾返: both qualify (幫我揾返 is short, so it needs the
        # longer of the two thresholds), and both get cut.
        for k, extra in ((11, 0.8), (15, 0.9)):
            onsets = onsets[:k + 1] + [t + extra for t in onsets[k + 1:]]
        out, cuts = split_at_pauses([_run_on(text, onsets)])
        self.assertEqual(cuts, 2)
        self.assertEqual([s["text"] for s in out], ["頭先有個同你一樣嘅死窮鬼", "幫我揾返", "走失咗個女"])
        for x, y in zip(out, out[1:]):
            self.assertLessEqual(x["end"], y["start"])

    def test_a_line_break_is_preferred_to_a_bare_pause(self):
        # The line break's 0.5 s beats a 0.9 s bare pause: the input said where to cut.
        head, tail = "雖然真係好難相信啊", "但係唔可以否認咗佢"
        cue = _two_line_cue(head, tail, 0.5)
        # A bare pause in the tail, too short a piece to cut once the break has been taken.
        for w in cue["words"][len(head) + 7:]:
            w["start"] += 0.9
            w["end"] += 0.9
        cue["end"] += 0.9
        out, cuts = split_at_pauses([cue])
        self.assertEqual(out[0]["text"], head)
        self.assertTrue(any(n.startswith("split_line:") for n in out[0]["notes"]))


# --- The top-of-screen track -----------------------------------------------------------

def _line(i, text, start, end, style=""):
    return TranscriptLine(i, text, source_start=start, source_end=end, style=style)


class TestTracks(unittest.TestCase):
    def test_loading_keeps_the_position_tag_off_the_text(self):
        import os
        import tempfile
        srt = ("1\n00:00:01,000 --> 00:00:03,000\n主線\n\n"
               "2\n00:00:01,500 --> 00:00:02,500\n" + TOP + "背景對白\n\n")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.srt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(srt)
            lines = load_transcript_lines(path, keep_timings=True)
        self.assertEqual([(l.text, l.style, l.top) for l in lines],
                         [("主線", "", False), ("背景對白", TOP, True)])

    def test_split_tracks_keeps_each_track_in_order(self):
        lines = [_line(0, "a", 0, 1), _line(1, "b", 0.5, 1.5, TOP), _line(2, "c", 2, 3)]
        main, overlay = split_tracks(lines)
        self.assertEqual([l.index for l in main], [0, 2])
        self.assertEqual([l.index for l in overlay], [1])

    def test_the_main_track_is_fitted_without_the_overlay(self):
        lines = [_line(0, "a", 0, 1), _line(1, "b", 0.5, 1.5, TOP), _line(2, "c", 2, 3)]
        seen = {}

        def run_main(main):
            seen["main"] = [(l.index, l.text) for l in main]
            return ([LineTiming(l.index, l.source_start + 5, l.source_end + 5) for l in main],
                    frozenset({1}), "transform", "report")

        def place(overlay, transform, report):
            seen["overlay"] = [(l.index, l.text) for l in overlay]
            self.assertEqual((transform, report), ("transform", "report"))
            return [LineTiming(0, 5.5, 6.5)], frozenset()

        timings, dropped, _t, _r = _with_overlay(lines, run_main, place)
        # Both tracks are numbered from 0 for the machinery, and handed back on the
        # original line numbers.
        self.assertEqual(seen["main"], [(0, "a"), (1, "c")])
        self.assertEqual(seen["overlay"], [(0, "b")])
        self.assertEqual([t.index for t in timings], [0, 1, 2])
        self.assertEqual(dropped, frozenset({2}))
        self.assertAlmostEqual(timings[1].start, 5.5)

    def test_no_overlay_leaves_the_main_path_untouched(self):
        lines = [_line(0, "a", 0, 1), _line(1, "c", 2, 3)]
        marker = object()
        self.assertIs(_with_overlay(lines, lambda main: marker, None), marker)

    def test_the_overlay_never_matches_a_main_track_anchor_by_index(self):
        """Transform.ids and each cut's cue_indices number the *main* track's lines."""
        pairs = [(x, x + 5.0, 1.0) for x in np.arange(0.0, 200.0, 10.0)]
        transform = timefit.fit_transform(pairs, ids=list(range(len(pairs))))
        cut = timefit.Break(50.0, 60.0, -10.0, "cut", cue_indices=(0, 1))
        transform = timefit.Transform(**{**transform.__dict__, "breaks": (cut,)})
        detached = _overlay_transform(transform, [(5.0, 6.0), (55.0, 56.0)])
        self.assertEqual(detached.ids, ())
        self.assertEqual(detached.breaks[0].cue_indices, (1,))

    def test_sync_maps_the_overlay_through_the_main_fit(self):
        pairs = [(x, x + 5.0, 1.0) for x in np.arange(0.0, 200.0, 10.0)]
        # Anchor 0 sits far off the line: an overlay cue numbered 0 must not inherit it.
        pairs[0] = (0.0, 9.0, 1.0)
        transform = timefit.locate_breaks(
            timefit.fit_transform(pairs, ids=list(range(len(pairs)))), [])
        chunks = [_chunk(0.0, 250 * 25)]
        timeline = _timeline(chunks, [_emission([None] * (250 * 25))])
        lines = [_line(0, "ab", 101.0, 102.0, TOP)]
        timings, dropped = place_overlay(
            lines, transform, chunks, timeline, VOCAB, "yue", leash=None)
        self.assertEqual(dropped, frozenset())
        self.assertAlmostEqual(timings[0].start, 106.0, delta=0.1)

    def test_adjust_aligns_an_overlay_line_on_its_own_even_over_the_main_line(self):
        # "cd" is the main line at 4.0 s; the top cue "ab" is spoken over it at 4.4 s. In one
        # trellis they would have to be put one after the other; alone, "ab" lands where it is.
        labels = [None] * 500
        labels[100:102] = ["c", "d"]
        labels[110:112] = ["a", "b"]
        chunks = [_chunk(0.0, 500)]
        timeline = _timeline(chunks, [_emission(labels)])
        lines = [_line(0, "ab", 4.0, 5.0, TOP)]
        timings, _ = place_overlay(lines, timefit.identity_transform(), chunks, timeline,
                                   VOCAB, "yue", leash=2.0, blank_id=BLANK)
        self.assertAlmostEqual(timings[0].start, 4.4, delta=0.05)
        self.assertIsNone(timings[0].reason)

    def test_without_filler_the_main_line_pushes_an_overlay_to_the_window_edge(self):
        # Why place_overlay aligns concurrently: as plain forced alignment the main line's
        # frames are charged as blank, and finishing "ab" before them is the cheaper path.
        from cantocaptions_ai.pipeline.realign import _Aligner
        labels = [None] * 500
        labels[100:103] = ["c", "d", "c"]
        labels[110:112] = ["a", "b"]
        chunks = [_chunk(0.0, 500)]
        timeline = _timeline(chunks, [_emission(labels)])
        plain = _Aligner(timeline, [[VOCAB["a"], VOCAB["b"]]], BLANK)
        concurrent = _Aligner(timeline, [[VOCAB["a"], VOCAB["b"]]], BLANK, concurrent=True)
        self.assertAlmostEqual(plain.run(0, 1, 2.0, 7.0, free=False)[0][1], 2.0, delta=0.05)
        self.assertAlmostEqual(concurrent.run(0, 1, 2.0, 7.0, free=False)[0][1], 4.4, delta=0.05)

    def test_adjust_never_lets_an_overlay_line_leave_its_leash(self):
        # The only "ab" in the file is 6 s from where the input put the cue.
        labels = [None] * 500
        labels[100:102] = ["a", "b"]
        chunks = [_chunk(0.0, 500)]
        timeline = _timeline(chunks, [_emission(labels)])
        lines = [_line(0, "ab", 10.0, 11.0, TOP)]
        timings, _ = place_overlay(lines, timefit.identity_transform(), chunks, timeline,
                                   VOCAB, "yue", leash=2.0, blank_id=BLANK)
        self.assertGreaterEqual(timings[0].start, 8.0)
        self.assertLessEqual(timings[0].start, 12.0)

    def test_style_travels_into_the_align_input_and_sync_cues(self):
        lines = [_line(0, "ab", 1.0, 2.0), _line(1, "cd", 3.0, 4.0, TOP)]
        timings = [LineTiming(0, 1.0, 2.0), LineTiming(1, 3.0, 4.0)]
        _, transcript = build_align_input(lines, timings, [_chunk(0.0, 250)], 28.0)
        self.assertEqual(transcript[0]["cue_styles"], ["", TOP])
        segments = segments_from_timings(lines, timings)
        self.assertEqual([s.get("style_tags") for s in segments], [None, TOP])

    def test_no_styles_means_no_cue_styles_key(self):
        lines = [_line(0, "ab", 1.0, 2.0)]
        _, transcript = build_align_input(lines, [LineTiming(0, 1.0, 2.0)],
                                          [_chunk(0.0, 250)], 28.0)
        self.assertNotIn("cue_styles", transcript[0])


class TestOverlayOutput(unittest.TestCase):
    def test_the_overlay_is_written_over_the_main_line_untouched(self):
        import tempfile
        from pathlib import Path

        from cantocaptions_ai.pipeline.transcribe import _merge_and_write
        from cantocaptions_ai.utils.output import get_writer

        item = {
            "audio_path": "/m/ep.mkv", "name": "ep",
            "result": {"language": "yue", "segments": [
                {"start": 1.0, "end": 4.0, "text": "主線對白", "words": []},
                {"start": 5.0, "end": 6.0, "text": "下一句", "words": []}]},
            "overlay_segments": [
                {"start": 2.0, "end": 3.0, "text": "背景", "words": [], "style_tags": TOP}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            results = _merge_and_write([item], get_writer("srt", tmp), "yue", 0.12, 0.04, {},
                                       merge=False, order_cues=True, collect=True)
            out = Path(tmp, "ep.srt").read_text(encoding="utf-8")
        segments = results[0]["result"]["segments"]
        self.assertEqual([s["text"] for s in segments], ["主線對白", "背景", "下一句"])
        self.assertEqual((segments[0]["start"], segments[0]["end"]), (1.0, 4.0),
                         "the main cue must not be cut short where the top one begins")
        self.assertIn("\n" + TOP + "背景\n", out)


if __name__ == "__main__":
    unittest.main()


class TestHoldToPlacement(unittest.TestCase):
    """Mode adjust's leash, enforced on the final alignment as well as the placement."""

    def _cue(self, start, end, placement, **extra):
        return {"start": start, "end": end, "text": "x", "words": [],
                "realign_placement": placement, **extra}

    def test_a_cue_pulled_past_the_leash_takes_its_placement_back(self):
        from cantocaptions_ai.pipeline.realign import REASON_OFF_PRIOR, hold_to_placement
        cues = [self._cue(58.58, 59.4, (61.40, 62.29)), self._cue(70.0, 71.0, (70.0, 71.0))]
        self.assertEqual(hold_to_placement(cues, 2.0), 1)
        self.assertEqual((cues[0]["start"], cues[0]["end"]), (61.4, 62.29))
        self.assertEqual(cues[0]["realign_reason"], REASON_OFF_PRIOR)
        self.assertNotIn("realign_placement", cues[1], "bookkeeping must not reach the output")

    def test_a_move_inside_the_leash_is_a_correction_and_stays(self):
        from cantocaptions_ai.pipeline.realign import hold_to_placement
        cue = self._cue(211.25, 213.0, (209.61, 213.8))
        self.assertEqual(hold_to_placement([cue], 2.0), 0)
        self.assertEqual(cue["start"], 211.25)

    def test_a_held_cue_never_overlaps_the_next(self):
        from cantocaptions_ai.pipeline.realign import hold_to_placement
        cues = [self._cue(1.0, 2.0, (5.0, 9.0)), self._cue(6.0, 7.0, (6.0, 7.0))]
        hold_to_placement(cues, 2.0, align_padding=0.04)
        self.assertLessEqual(cues[0]["end"], cues[1]["start"] - 0.04 + 1e-9)

    def test_no_leash_only_clears_the_bookkeeping(self):
        from cantocaptions_ai.pipeline.realign import hold_to_placement
        cue = self._cue(1.0, 2.0, (50.0, 51.0))
        self.assertEqual(hold_to_placement([cue], None), 0)
        self.assertEqual(cue["start"], 1.0)
        self.assertNotIn("realign_placement", cue)

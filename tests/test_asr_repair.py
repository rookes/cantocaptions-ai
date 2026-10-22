"""ASR repair (pipeline/asr_repair.py): loop detection and trimming, the language checks,
where a chunk is cut, what each piece ends up saying, and how a repaired chunk is written back.

All CPU and model-free: the two model calls the repair makes are passed in as functions, so
the ladder runs here against fakes. The thresholds themselves were chosen on real
checkpoints with scripts/eval_hallucination.py; these tests pin the behaviour around them.
"""
import numpy as np
import pytest

import torch

from cantocaptions_ai.pipeline import asr_repair as ar
from cantocaptions_ai.pipeline import word_align as wa
from cantocaptions_ai.pipeline.asr_repair import (
    ChunkRepair,
    LoopRule,
    Piece,
    RepairSettings,
    combine,
    find_loop,
    loop_verdict,
    trim_loop,
)
from cantocaptions_ai.text_profiles import LATIN_WORD
from cantocaptions_ai.utils.schema import attach_timed_notes

NATIVE = ("Cantonese", "Chinese", "English")
SR = 16000
LOOP_96 = "一、二、三…" + "我哋揀咗喇，" * 16 + "我哦"


class _Text(str):
    """What QwenPipelineNative.transcribe returns: a str with generation flags."""

    def __new__(cls, text, language=None, budget_hit=False, retry_budget_hit=False, raw=None):
        out = super().__new__(cls, text)
        out.language, out.budget_hit, out.retry_budget_hit, out.raw = (
            language, budget_hit, retry_budget_hit, raw)
        return out


# --- loops ----------------------------------------------------------------------------------

class TestFindLoop:
    def test_a_budget_loop_is_found_with_its_cut_off_copy(self):
        loop = find_loop(LOOP_96)
        assert (loop.phrase, loop.copies, loop.truncated) == ("我哋揀咗喇", 16, True)

    @pytest.mark.parametrize("text", ["拜拜，拜拜，拜拜", "多謝晒，唔該晒，唔該晒，唔該晒，", "早晨早晨早晨早晨",
                                      "喂，我啊…喂，我啊…喂，我啊…喂，我啊…佢媽咪走咗"])
    def test_real_speech_repeats_are_not_loops(self, text):
        assert loop_verdict({"text": text})[0] is None

    def test_the_phase_ends_each_copy_on_its_separator(self):
        # Read from 晒，唔該 the end of the text would look like a cut-off copy.
        loop = find_loop("多謝晒，唔該晒，唔該晒，唔該晒，")
        assert (loop.unit, loop.copies, loop.truncated) == ("唔該晒，", 3, False)

    def test_a_sung_syllable_is_a_loop(self):
        assert loop_verdict({"text": "噔" * 30})[0] == "repeats"

    def test_evidence_from_how_generation_ended(self):
        assert loop_verdict({"text": "放咗手先，放�"})[0] == "broken_char"
        assert loop_verdict({"text": "好", "raw_text": "好" * 40})[0] == "folded"
        assert loop_verdict({"text": "好", "retry_budget_hit": True})[0] == "retry_looped"
        assert loop_verdict({"text": "好", "asr_looped": True})[0] == "looped"
        # A budget hit the retry recovered is not a loop: the retry's text stands.
        assert loop_verdict({"text": "好", "budget_hit": True, "retried": True})[0] is None

    def test_the_rule_is_adjustable(self):
        text = "唔好走，" * 4
        assert loop_verdict({"text": text})[0] is None
        assert loop_verdict({"text": text}, LoopRule(min_copies=4, min_chars=12))[0] == "repeats"

    def test_compression_is_off_by_default(self):
        text = "火神爺爺顯靈啊，火神爺爺…" * 3
        assert loop_verdict({"text": text}, LoopRule(min_copies=99))[0] is None
        assert loop_verdict({"text": text}, LoopRule(min_copies=99, max_compression=1.5))[0] == "compresses"


class TestTrimLoop:
    def test_the_loop_is_written_once(self):
        assert trim_loop(LOOP_96) == "一、二、三…我哋揀咗喇…"

    def test_speech_after_the_loop_is_kept(self):
        text = "燒死佢…" * 8 + "天師，恭喜你"
        assert trim_loop(text) == "燒死佢…天師，恭喜你"

    def test_broken_characters_go_even_without_a_loop(self):
        assert trim_loop("放咗手先，放�") == "放咗手先，放"


# --- language -------------------------------------------------------------------------------

class TestLanguage:
    @pytest.mark.parametrize("text,lang", [
        ("Scusa, la tua ragazza? State insieme?", "Italian"),
        ("Rosé, até. Il nasce sulle gambe blu, la ragazza più rossa", "Italian"),
        ("Want to go for a drink after work? Sure, just five of me", "English"),
        ("Hey, hey, hey, I just need more choice", "English"),
        ("OK", None),
    ])
    def test_latin_language(self, text, lang):
        assert ar.latin_language(text) == lang

    def test_foreign_text_needs_a_run_of_words_not_in_a_native_language(self):
        assert ar.foreign_text("哦，佢以為我係你女朋友。Que peccado pensavo che era tua ragazza", NATIVE) == "Italian"
        assert ar.foreign_text("我嚟到cafe喇，Sam哥", NATIVE) is None
        assert ar.foreign_text("Where you from? I'm from Hong Kong", NATIVE) is None

    def test_native_matches_any_chinese(self):
        assert ar.is_native("Chinese (Sichuanese)", NATIVE)
        assert ar.is_native("cantonese", NATIVE)
        assert not ar.is_native("Italian", NATIVE)

    def test_a_latin_reading_corrects_its_label(self):
        assert ar.heard_language("English", "Non mi va, non mi va.") == "Italian"
        assert ar.heard_language("English", "That's it. That's it.") == "English"
        # Largely Chinese text keeps its label: half Cantonese is not relabelled by its other half.
        assert ar.heard_language("Cantonese", "佢以為我係你女朋友。Che peccato") == "Cantonese"
        assert ar.heard_language("None", "") == "None"


# --- splitting ------------------------------------------------------------------------------

def _bursts(*spans, total=20.0):
    """Tone where *spans* say, near-silence elsewhere."""
    t = np.arange(int(total * SR)) / SR
    audio = 1e-4 * np.random.default_rng(0).standard_normal(len(t))
    for a, b in spans:
        mask = (t >= a) & (t < b)
        audio[mask] += 0.3 * np.sin(2 * np.pi * 220 * t[mask])
    return audio.astype(np.float32)


class TestSplitPoint:
    def test_the_longest_speech_gap_wins(self):
        speech = [(100.0, 106.0), (106.5, 110.0), (112.0, 120.0)]
        assert ar.split_point(100.0, np.zeros(20 * SR), speech) == pytest.approx(111.0)

    def test_a_gap_too_near_an_end_is_passed_over(self):
        speech = [(100.0, 101.0), (103.0, 120.0)]
        cut = ar.split_point(100.0, _bursts((0, 1), (3, 9), (9.5, 20)), speech, min_piece=4.0)
        assert 104.0 <= cut <= 116.0
        assert cut == pytest.approx(109.25, abs=0.2)  # the quiet audio, not the VAD gap at 2 s

    def test_without_speech_regions_the_quietest_stretch(self):
        cut = ar.split_point(50.0, _bursts((0, 8), (10, 20)))
        assert cut == pytest.approx(59.0, abs=0.2)

    def test_too_short_to_cut(self):
        assert ar.split_point(0.0, np.zeros(7 * SR), min_piece=4.0) is None


# --- what a piece says ----------------------------------------------------------------------

def _piece(text="", alt=None, label=None, looped=False, hinted=None):
    return Piece(0.0, 5.0, text, looped=looped, alt_text=alt, alt_language=label,
                 hinted_text=hinted)


class TestCombine:
    def test_empty_own_text_takes_the_foreign_reading(self):
        p = combine(_piece("", "Si può scattare una foto?", "Italian"), mode="transcribe", native=NATIVE)
        assert (p.text, p.source, p.note) == ("Si può scattare una foto?", "foreign", "asr_lang:Italian")

    def test_drop_says_nothing_and_keeps_what_it_replaced(self):
        p = combine(_piece("", "Si può scattare una foto?", "Italian"), mode="drop", native=NATIVE)
        assert (p.text, p.source, p.note) == ("", "dropped", "asr_lang_dropped:Italian")

    def test_note_only_marks_it(self):
        p = combine(_piece("嘿，艾爾", "Hey, vieni", "Italian", looped=True), mode="note", native=NATIVE)
        assert (p.text, p.note) == ("嘿，艾爾", "asr_lang:Italian")

    def test_own_text_already_in_that_language_is_kept_and_names_it(self):
        own = "Cucineros？Andata verso di là。Sì, verso di là"
        p = combine(_piece(own, "Andar para cima. Sim.", "Portuguese"), mode="transcribe", native=NATIVE)
        assert (p.text, p.source, p.note) == (own, "asr", "asr_lang:Italian")

    def test_a_looping_own_text_is_replaced(self):
        p = combine(_piece("我哋唔理佢，" * 3, "che non vi va basta", "Italian", looped=True),
                    mode="transcribe", native=NATIVE)
        assert p.source == "foreign"

    def test_clean_cantonese_is_kept_on_a_conflict(self):
        # The multilingual model, told to expect Cantonese, heard Cantonese: a translation.
        p = combine(_piece("佢覺得我哋好登對", "Ha detto che eravamo molto d'accordo", "Italian",
                           hinted="佢覺得我哋好登對"), mode="drop", native=NATIVE)
        assert (p.text, p.source, p.note) == ("佢覺得我哋好登對", "asr", "asr_lang_conflict:Italian")

    def test_clean_cantonese_goes_when_the_hinted_reading_hears_none(self):
        p = combine(_piece("我哋唔理佢，快啲走啊", "che non vi va basta", "Italian",
                           hinted="Non mi va, basta."), mode="transcribe", native=NATIVE)
        assert (p.text, p.replaced) == ("che non vi va basta", "我哋唔理佢，快啲走啊")

    def test_a_mostly_chinese_reading_is_only_noted(self):
        p = combine(_piece("哦，佢以為我係你女朋友", "哦，佢以為我係你女朋友。Oh, che peccato", "Italian"),
                    mode="drop", native=NATIVE)
        assert (p.source, p.note) == ("asr", "asr_lang_mixed:Italian")

    @pytest.mark.parametrize("label", ["Cantonese", "Chinese", "English"])
    def test_native_readings_never_replace_or_drop(self, label):
        p = combine(_piece("", "嗯，啊，这", label), mode="drop", native=NATIVE)
        assert (p.text, p.source) == ("", "asr")

    def test_english_over_chinese_text_is_noted(self):
        p = combine(_piece("係，得喇。四！得喇，四！", "That's it. That's it.", "English"),
                    mode="transcribe", native=NATIVE)
        assert (p.text, p.note) == ("係，得喇。四！得喇，四！", "asr_lang_mismatch:English")

    def test_no_speech_heard_over_a_loop_is_noted(self):
        p = combine(_piece("唔得喇…", "", "None", looped=True), mode="transcribe", native=NATIVE)
        assert (p.text, p.note) == ("唔得喇…", "asr_unheard:a second model heard no speech")


# --- the ladder, end to end with fakes ------------------------------------------------------

def _files(texts, durations, extra=None):
    """One file: a transcription result and its vad_segments, chunks back to back."""
    segs, vads, t = [], [], 0.0
    for text, dur in zip(texts, durations):
        segs.append({"text": text, "start": t, "end": t + dur, **(extra or {}).get(len(segs), {})})
        vads.append({"start": t, "end": t + dur, "audio": _bursts((0, dur / 2 - 0.5), (dur / 2 + 0.5, dur), total=dur)})
        t += dur + 1.0
    return {"segments": segs, "language": "Cantonese"}, vads


def _own_model(loop_over=10.0):
    """The run's own model: loops on audio longer than *loop_over* seconds."""
    calls = []

    def transcribe(wavs):
        calls.append([len(w) / SR for w in wavs])
        return [_Text("好" * 8 if len(w) / SR > loop_over else "講完喇",
                      retry_budget_hit=len(w) / SR > loop_over) for w in wavs]
    return transcribe, calls


def _join(texts):
    return "，".join(texts)


class TestRepair:
    def test_plan_reasons_and_splits(self):
        result, vads = _files([LOOP_96, "", "Scusa, la tua ragazza? State insieme?", "你好，你好嗎？"],
                              [21.0, 10.0, 7.0, 5.0])
        chunks = ar.plan_chunks([(result, vads)], RepairSettings())
        assert [(c.index, c.reasons, c.split) for c in chunks] == [
            (0, ["loop:repeats"], True), (1, ["empty"], False),
            (2, ["foreign_text:Italian"], False)]

    def test_empty_chunks_are_checked_only_where_another_language_shows(self):
        result, vads = _files([LOOP_96, "", "你好，你好嗎？"], [21.0, 10.0, 5.0])
        chunks = ar.plan_chunks([(result, vads)], RepairSettings())
        assert [(c.index, c.reasons) for c in chunks] == [(0, ["loop:repeats"])]
        chunks = ar.plan_chunks([(result, vads)], RepairSettings(empty_needs_foreign=False))
        assert [(c.index, c.reasons) for c in chunks] == [(0, ["loop:repeats"]), (1, ["empty"])]

    def test_a_model_that_labels_its_language_is_believed_enough_to_check(self):
        result, vads = _files(["Ciao", "你好", "好"], [5.0, 5.0, 5.0],
                              {0: {"asr_language": "Italian"}, 1: {"asr_language": "Chinese"},
                               2: {"asr_language": "none"}})
        chunks = ar.plan_chunks([(result, vads)], RepairSettings())
        assert [(c.index, c.reasons) for c in chunks] == [(0, ["foreign_label:Italian"])]

    def test_language_check_off_plans_loops_only(self):
        result, vads = _files([LOOP_96, ""], [21.0, 10.0])
        chunks = ar.plan_chunks([(result, vads)], RepairSettings(foreign="none"))
        assert [c.index for c in chunks] == [0]

    def test_a_loop_is_cut_until_its_pieces_stop_looping(self):
        result, vads = _files([LOOP_96], [24.0])
        settings = RepairSettings(split_depth=3)
        chunks = ar.plan_chunks([(result, vads)], settings)
        transcribe, calls = _own_model(loop_over=10.0)
        ar.run_ladder(chunks, transcribe, settings)
        assert [round(sum(c), 1) for c in calls][0] == 24.0
        assert all(not p.looped for p in chunks[0].pieces)
        rows = ar.finish(chunks, [result], settings, _join)
        seg = result["segments"][0]
        assert seg["text"] == _join(["講完喇"] * len(chunks[0].pieces))
        assert (seg["start"], seg["end"]) == (0.0, 24.0)  # one segment per VAD chunk
        assert rows[0][2].startswith("repetition loop: split into")

    def test_the_depth_limit_holds_and_the_loop_is_flagged(self):
        result, vads = _files([LOOP_96], [24.0], {0: {"budget_hit": True, "retried": True}})
        settings = RepairSettings(split_depth=1, foreign="none")
        chunks = ar.plan_chunks([(result, vads)], settings)
        transcribe, calls = _own_model(loop_over=1.0)
        ar.run_ladder(chunks, transcribe, settings)
        assert len(calls) == 1 and len(chunks[0].pieces) == 2
        ar.finish(chunks, [result], settings, _join)
        seg = result["segments"][0]
        assert seg["asr_looped"] and "budget_hit" not in seg and "retried" not in seg

    def test_foreign_pieces_in_a_mixed_chunk(self):
        result, vads = _files(["", ""], [10.0, 10.0])
        settings = RepairSettings(foreign="drop", empty_needs_foreign=False)
        chunks = ar.plan_chunks([(result, vads)], settings)

        def foreign(wavs):
            return [_Text("Si può scattare una foto?", language="Italian"),
                    _Text("嗯，啊", language="Chinese")]

        ar.run_language_check(chunks, foreign, NATIVE)
        rows = ar.finish(chunks, [result], settings, _join)
        assert [r[2] for r in rows] == ["no text: Italian speech dropped", "no text: unchanged"]
        assert result["segments"][0]["text"] == ""
        assert result["segments"][0]["notes_at"] == [(0.0, 10.0, "asr_lang_dropped:Italian")]

    def test_conflicts_are_asked_again_with_the_hint(self):
        result, vads = _files(["Andiamo andiamo andiamo andiamo", "你好"], [8.0, 8.0])
        chunks = ar.plan_chunks([(result, vads)], RepairSettings())
        chunks[0].pieces[0].text = "我哋唔理佢"  # as if re-transcribed into clean Cantonese
        hinted_calls = []

        def hinted(wavs):
            hinted_calls.append(len(wavs))
            return [_Text("Andiamo")]

        ar.run_language_check(chunks, lambda wavs: [_Text("andiamo andiamo", language="Italian")],
                              NATIVE, hinted)
        assert hinted_calls == [1]
        assert chunks[0].pieces[0].hinted_text == "Andiamo"


# --- tidy-up and notes ----------------------------------------------------------------------

class TestSanitize:
    def test_a_visible_loop_is_trimmed_and_noted(self):
        seg, loop = ar.sanitize_segment({"text": LOOP_96, "start": 10.0, "end": 30.0})
        assert seg["text"] == "一、二、三…我哋揀咗喇…"
        assert loop.copies == 16
        assert seg["notes_at"] == [(29.0, 30.0, "asr_loop:written once, 16x 我哋揀咗喇")]

    def test_trimming_can_be_turned_off(self):
        seg, loop = ar.sanitize_segment({"text": LOOP_96, "start": 0.0, "end": 9.0}, trim=False)
        assert loop is None and seg["text"] == LOOP_96

    def test_a_recovered_retry_is_noted_not_changed(self):
        seg, _ = ar.sanitize_segment({"text": "原來佢叫做Vincent", "start": 5.0, "end": 9.0,
                                      "budget_hit": True, "retried": True})
        assert seg["text"] == "原來佢叫做Vincent"
        assert seg["notes_at"] == [(5.0, 5.5, "asr_loop:recovered by a retry")]

    def test_broken_characters_are_removed(self):
        seg, _ = ar.sanitize_segment({"text": "放咗手先，放�", "start": 0.0, "end": 1.0})
        assert seg["text"] == "放咗手先，放"

    def test_ordinary_segments_are_untouched(self):
        seg = {"text": "拜拜，拜拜，拜拜", "start": 0.0, "end": 2.0}
        assert ar.sanitize_segment(seg) == (seg, None)


class TestTimedNotes:
    def test_a_note_lands_on_the_cues_it_covers(self):
        cues = [{"start": 0.0, "end": 2.0, "text": "a"}, {"start": 2.1, "end": 4.0, "text": "b"},
                {"start": 4.1, "end": 6.0, "text": "c"}]
        attach_timed_notes({"notes_at": [(2.0, 6.0, "asr_lang:Italian")]}, cues)
        assert [c.get("notes") for c in cues] == [None, ["asr_lang:Italian"], ["asr_lang:Italian"]]

    def test_a_cue_that_only_grazes_the_range_is_left_out(self):
        cues = [{"start": 0.0, "end": 2.08, "text": "OK"}, {"start": 2.1, "end": 4.0, "text": "vieni"}]
        attach_timed_notes({"notes_at": [(2.0, 6.0, "asr_lang:Italian")]}, cues)
        assert "notes" not in cues[0]

    def test_with_no_overlap_the_nearest_cue(self):
        cues = [{"start": 0.0, "end": 1.0, "text": "a"}, {"start": 5.0, "end": 6.0, "text": "b"}]
        attach_timed_notes({"notes_at": [(6.5, 7.0, "asr_loop:written once")]}, cues)
        assert cues[1]["notes"] == ["asr_loop:written once"]


# --- timing other-language text -------------------------------------------------------------

def _aligned(text, words, start=0.0, end=10.0):
    """A cue as _align_segment leaves it: *words* (token, start, end) in order."""
    return {"text": text, "start": start, "end": end,
            "words": [{"word": w, "start": a, "end": b, "score": 0.5, "peak": 0.9}
                      for w, a, b in words]}


def _unspaced_words(text, step=0.5):
    """Words the way alignment makes them in unspaced text: a Latin run whole, every other
    character (space, mark, Chinese character) on its own; times one *step* apart."""
    tokens, pos = [], 0
    for m in LATIN_WORD.finditer(text):
        tokens += list(text[pos:m.start()]) + [m.group(0)]
        pos = m.end()
    tokens += list(text[pos:])
    return [(t, i * step, i * step + step) for i, t in enumerate(tokens)]


class TestPlaceForeign:
    TEXT = "佢以為我係你女朋友，Che peccato, pensavo che era tua ragazza. Mi vedevano"
    SPAN = "Che peccato, pensavo che era tua ragazza. Mi vedevano"

    def _seg(self, times):
        return {"text": self.TEXT, "foreign_spans": [
            {"text": self.SPAN, "start": 5.0, "end": 15.0, "language": "Italian", "words": times}]}

    def test_the_aligners_times_replace_the_align_models_and_sentences_are_cut(self):
        cue = _aligned(self.TEXT, _unspaced_words(self.TEXT), end=40.0)
        times = [[10.0 + i, 10.5 + i] for i in range(9)]  # Che .. vedevano
        out = ar.place_foreign(self._seg(times), [cue])
        # Too wide for one line, the first sentence is also cut at its comma.
        assert [p["text"] for p in out] == [
            "佢以為我係你女朋友，Che peccato,", "pensavo che era tua ragazza.", "Mi vedevano"]
        italian = [w for p in out for w in p["words"] if w["word"].isalpha() and w["word"].isascii()]
        assert [(w["start"], w["end"]) for w in italian] == [tuple(t) for t in times]
        assert all("peak" not in w for w in italian)
        # The Cantonese keeps its own times.
        assert out[0]["words"][0] == cue["words"][0] and out[0]["start"] == 0.0
        assert out[1]["start"] == 12.0 and out[1]["end"] == 16.5
        assert [p["release_from"] for p in out] == [p["end"] for p in out]

    def test_a_word_the_aligner_could_not_time_keeps_its_own(self):
        cue = _aligned(self.TEXT, _unspaced_words(self.TEXT), end=40.0)
        times = [[10.0 + i, 10.5 + i] for i in range(9)]
        times[1] = None
        out = ar.place_foreign(self._seg(times), [cue])
        peccato = next(w for p in out for w in p["words"] if w["word"] == "peccato")
        assert "peak" in peccato

    @pytest.mark.parametrize("times", [None, [[1.0, 2.0]] * 3])
    def test_no_times_or_a_mismatch_leave_the_cues_alone(self, times):
        cue = _aligned(self.TEXT, _unspaced_words(self.TEXT), end=40.0)
        before = [dict(w) for w in cue["words"]]
        out = ar.place_foreign(self._seg(times), [cue])
        assert out == [cue] and cue["words"] == before

    def test_repair_records_each_span_with_its_language_and_time(self):
        result, vads = _files(["", "你好"], [12.0, 5.0])
        chunks = ar.plan_chunks([(result, vads)], RepairSettings(empty_needs_foreign=False))
        ar.run_language_check(chunks, lambda wavs: [_Text("Si può. Basta, dai.", language="Italian")],
                              NATIVE)
        ar.finish(chunks, [result], RepairSettings(), _join)
        assert result["segments"][0]["foreign_spans"] == [
            {"text": "Si può. Basta, dai.", "start": 0.0, "end": 12.0, "language": "Italian"}]


class TestSpokenTokens:
    def test_latin_runs_whole_everything_else_one_letter_at_a_time(self):
        text = "Si può, 你好！OK? café"
        assert [text[a:b] for a, b in wa.spoken_tokens(text)] == ["Si", "può", "你", "好", "OK", "café"]


class _Inputs(dict):
    def to(self, *args, **kwargs):
        return self


class _FakeAlignerProcessor:
    """Records the prompts; decodes word k of each sample to (k, k + 0.5) seconds."""

    def __init__(self):
        self.conversations = []

    def apply_chat_template(self, conversations, **kwargs):
        self.conversations.extend(conversations)
        return _Inputs(input_ids=torch.zeros((len(conversations), 3), dtype=torch.long))

    def decode_forced_alignment(self, logits, input_ids, word_lists, timestamp_token_id):
        return [[{"text": w, "start_time": k, "end_time": k + 0.5} for k, w in enumerate(words)]
                for words in word_lists]


class _FakeAlignerModel:
    device, dtype = "cpu", torch.float32
    config = type("C", (), {"timestamp_token_id": 7})()

    def __call__(self, **kwargs):
        return type("Out", (), {"logits": torch.zeros(1)})()


class TestQwen3ForcedAligner:
    def _aligner(self):
        return wa.Qwen3ForcedAligner(_FakeAlignerModel(), _FakeAlignerProcessor(), batch_size=2)

    def test_times_come_back_per_word_with_unreadable_words_untimed(self):
        aligner = self._aligner()
        audio = np.zeros(SR, dtype=np.float32)
        out = aligner.align([wa.WordAlignJob(audio, ["Si", "—", "può"], "Italian"),
                             wa.WordAlignJob(audio, ["Bonjour"], "French")])
        assert out == [[(0, 0.5), None, (1, 1.5)], [(0, 0.5)]]
        texts = [[c["text"] for c in conv[0]["content"][1:]] for conv in aligner.processor.conversations]
        assert sorted(texts) == [["Bonjour"], ["Si", "può"]]

    def test_unsupported_languages_and_overlong_audio_are_skipped(self):
        aligner = self._aligner()
        out = aligner.align([wa.WordAlignJob(np.zeros(SR, dtype=np.float32), ["Cześć"], "Polish"),
                             wa.WordAlignJob(np.zeros(301 * SR, dtype=np.float32), ["Ciao"], "Italian")])
        assert out == [[None], [None]] and not aligner.processor.conversations

    def test_supported_languages_ignore_case(self):
        aligner = self._aligner()
        assert aligner.supports("italian") and aligner.supports("Cantonese")
        assert not aligner.supports(None) and not aligner.supports("Polish")


class TestUnpile:
    def test_words_piled_on_one_instant_share_the_gap_around_them(self):
        times = [(1.0, 1.32), (1.32, 1.32), (1.32, 1.32), (1.32, 1.32), (1.64, 2.0)]
        assert wa.unpile(times, 5.0) == [(1.0, 1.32), (1.32, 1.427), (1.427, 1.533),
                                         (1.533, 1.64), (1.64, 2.0)]

    def test_a_pile_spreads_no_faster_than_a_word_in_four_tenths(self):
        # Seven words piled at 9.0 s with nothing timed after them: fast speech, not slow.
        times = [(1.0, 1.4)] + [(9.0, 9.0)] * 7
        out = wa.unpile(times, 25.0)
        assert out[1] == (9.0, 9.4) and out[-1] == (11.4, 11.8)

    def test_ordinary_times_and_gaps_are_untouched(self):
        times = [(0.5, 0.9), None, (1.0, 1.0), (1.4, 1.8)]
        assert wa.unpile(times, 5.0) == times


class TestTimeForeignSpans:
    def test_spans_get_file_timeline_word_times_from_the_aligner(self, monkeypatch):
        from types import SimpleNamespace

        from cantocaptions_ai.pipeline import stages, word_align

        seen = {}

        class Aligner:
            def supports(self, language):
                return language == "Italian"

            def align(self, jobs):
                seen["jobs"] = jobs
                return [[(0.5 * k, 0.5 * k + 0.4) for k in range(len(j.words))] for j in jobs]

        monkeypatch.setattr(word_align, "load_word_aligner", lambda *a, **k: Aligner())
        span = {"text": "Si può.", "start": 12.0, "end": 14.0, "language": "Italian"}
        items = [{"vad_segments": [{"start": 10.0, "end": 20.0, "audio": np.zeros(10 * SR)}],
                  "result": {"segments": [{"text": "Si può.", "start": 10.0, "end": 20.0,
                                           "foreign_spans": [span]}]}}]
        cfg = SimpleNamespace(align_foreign_model="x", device="cpu", device_index=0,
                              align_compute_type="float32", model_dir=None, model_cache_only=True,
                              align_batch_size=2)
        stages._time_foreign_spans(SimpleNamespace(cfg=cfg), items,
                                   SimpleNamespace(status=lambda *a: None))
        (job,) = seen["jobs"]
        assert job.words == ["Si", "può"] and len(job.audio) == 2 * SR
        assert span["words"] == [[12.0, 12.4], [12.5, 12.9]]

    def test_none_turns_it_off(self, monkeypatch):
        from types import SimpleNamespace

        from cantocaptions_ai.pipeline import stages, word_align

        monkeypatch.setattr(word_align, "load_word_aligner",
                            lambda *a, **k: pytest.fail("the word aligner must not load"))
        items = [{"vad_segments": [], "result": {"segments": [
            {"text": "x", "foreign_spans": [{"text": "x", "start": 0.0, "end": 1.0}]}]}}]
        stages._time_foreign_spans(SimpleNamespace(cfg=SimpleNamespace(align_foreign_model="none")),
                                   items, None)

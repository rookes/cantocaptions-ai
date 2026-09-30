"""``alignment.align`` on emissions whose correct answer is known in advance.

The fakes in ``_pipeline_fakes`` make the audio encode its own transcript, one peaky CTC
frame per character, so every character's true time is known and forced alignment has a
single right answer. These tests pin the aligner's contract directly: character timings,
clause subsegments, particle spot-checks, and what happens to text it cannot align.
"""
import pytest

from _pipeline_fakes import FRAME_S, ScriptedAudio, StubVad, fake_align_model
from cantocaptions_ai.languages.yue.text import SpotCheck
from cantocaptions_ai.pipeline.alignment import align


def _align(scripted, texts=None, **kwargs):
    """Align each VAD segment of *scripted* against its text (or the override *texts*)."""
    segments = StubVad().process(scripted.samples)
    if texts is None:
        texts = [scripted.text_at(s["start"], s["end"]) for s in segments]
    transcript = [{"start": s["start"], "end": s["end"], "text": t} for s, t in zip(segments, texts)]
    model, meta = fake_align_model(scripted)
    return align(transcript, model, meta, segments, "cpu", vram_checks=False, **kwargs)


def _chars(result):
    return [(c["char"], c.get("start")) for seg in result["segments"] for c in seg["chars"]
            if c["char"].strip() and c["char"] not in "，。？！"]


def test_every_character_lands_on_its_own_frame():
    scripted = ScriptedAudio()
    result = _align(scripted, return_char_alignments=True)
    expected = [(c, start) for utt in scripted.char_times for c, start, _ in utt]
    got = _chars(result)
    assert [c for c, _ in got] == [c for c, _ in expected]
    for (_, t), (_, want) in zip(got, expected):
        assert t == pytest.approx(want, abs=FRAME_S / 2)


def test_clauses_become_their_own_subsegments():
    # Alignment over-splits at punctuation on purpose; cue assembly joins them later.
    result = _align(ScriptedAudio())
    assert [s["text"] for s in result["segments"]] == [
        "你好，", "今日天氣幾好。", "我哋去公園行下啦。", "好呀，", "我帶埋隻狗。"]
    starts = [round(s["start"], 2) for s in result["segments"]]
    assert starts == [1.0, 1.52, 4.0, 7.0, 7.52]


def test_spot_check_swaps_a_particle_for_the_one_actually_spoken():
    # The audio says 喇; the "ASR" wrote 啦. Both are in the align vocabulary.
    scripted = ScriptedAudio(script=((1.0, "好喇。"),), duration=3.0, extra_vocab=("啦",))
    checks = {"啦": SpotCheck(("喇", "啦"))}
    result = _align(scripted, texts=["好啦。"], spotchecks=checks)
    assert result["segments"][0]["text"] == "好喇。"


def test_spot_check_weight_can_overrule_the_acoustics():
    scripted = ScriptedAudio(script=((1.0, "好喇。"),), duration=3.0, extra_vocab=("啦",))
    checks = {"啦": SpotCheck(("喇", "啦"), weights={"啦": 100.0})}
    result = _align(scripted, texts=["好啦。"], spotchecks=checks)
    assert result["segments"][0]["text"] == "好啦。"


def test_without_spot_checks_the_text_is_never_changed():
    scripted = ScriptedAudio(script=((1.0, "好喇。"),), duration=3.0, extra_vocab=("啦",))
    result = _align(scripted, texts=["好啦。"])
    assert result["segments"][0]["text"] == "好啦。"


def test_out_of_vocabulary_characters_stay_in_the_text():
    # 佢 is not in the align vocabulary: it cannot be timed, but it must not be dropped.
    scripted = ScriptedAudio()
    result = _align(scripted, texts=["你好，今日天氣幾好。", "我哋佢去公園行下啦。", "好呀，我帶埋隻狗。"])
    assert "我哋佢去公園行下啦。" in [s["text"] for s in result["segments"]]
    assert round(result["segments"][2]["start"], 2) == 4.0


def test_a_segment_with_nothing_alignable_is_kept_with_its_own_span():
    scripted = ScriptedAudio()
    result = _align(scripted, texts=["你好，今日天氣幾好。", "……", "好呀，我帶埋隻狗。"])
    texts = [s["text"] for s in result["segments"]]
    assert "你好，" in texts and "我帶埋隻狗。" in texts

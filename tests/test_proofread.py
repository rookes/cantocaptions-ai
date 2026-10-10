"""Tests for the optional LLM proofreading stage (pipeline/proofread).

No network, no keys and no SDK are needed: the provider is replaced by a scripted fake
everywhere except the request-shape tests at the bottom, which build real SDK request
objects when the SDK is installed and skip otherwise.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cantocaptions_ai.languages import get_language_pack
from cantocaptions_ai.languages.base import ProofreadStandard
from cantocaptions_ai.pipeline.config import PipelineConfig
from cantocaptions_ai.pipeline.proofread import Proofreader, ProofreadSettings, load_proofreader
from cantocaptions_ai.pipeline.proofread import providers
from cantocaptions_ai.pipeline.proofread.apply import (
    apply_answer, apply_names, introduced_foreign, moves_text, name_table, protect_particles,
)
from cantocaptions_ai.pipeline.proofread.render import (
    OUTPUT_SCHEMA, Cue, render_stream, system_prompt, user_message,
)
from cantocaptions_ai.text_profiles import CJK_SCRIPT, SPACED_SCRIPT

YUE = get_language_pack("yue").standard_for(None)
PLAIN = ProofreadStandard(name="plain", language_name="English")


def segments():
    return [
        {"start": 0.0, "end": 1.0, "text": "你返嚟喇"},
        {"start": 1.0, "end": 2.0, "text": "我哋去食飯啦"},
        {"start": 2.0, "end": 3.0, "text": "我個名叫做柏克"},
        {"start": 5.0, "end": 6.0, "text": "柏克唔喺度"},
    ]


REFERENCE = [{"start": 0.1, "end": 1.0, "text": "你回來了？"},
             {"start": 2.1, "end": 3.0, "text": "我叫帕克"}]


def reply(answer: dict, cost=0.01) -> providers.Reply:
    return providers.Reply(raw=json.dumps(answer, ensure_ascii=False), answer=answer,
                           usage={"input_tokens": 1000, "output_tokens": 500, "cost_usd": cost})


ANSWER = {
    "names": [{"use": "帕克", "draft_forms": ["柏克"], "reference": "帕克"}],
    "edits": [{"id": 1, "text": "佢返嚟喇？", "type": "mishearing", "confidence": "high",
               "reason": "pronoun"}],
    "flags": [{"id": 2, "note": "check the particle"}],
}


def settings(**kw) -> ProofreadSettings:
    base = dict(provider="gemini", model="gemini-3.7-flash", standard=YUE, script=CJK_SCRIPT,
                max_cost=None)
    base.update(kw)
    return ProofreadSettings(**base)


class TestOffByDefault(unittest.TestCase):
    """The library's contract is offline; proofreading must never turn itself on."""

    def test_default_config_is_off(self):
        self.assertEqual(PipelineConfig.defaults()["proofread"], "none")

    def test_off_builds_no_proofreader_and_loads_no_sdk(self):
        before = {m for m in sys.modules if m.startswith(("google.genai", "anthropic"))}
        cfg = PipelineConfig()
        pack = get_language_pack("yue")
        self.assertIsNone(load_proofreader(cfg, pack, pack.resolve(None)))
        after = {m for m in sys.modules if m.startswith(("google.genai", "anthropic"))}
        self.assertEqual(after, before)

    def test_a_cfg_file_can_turn_it_on(self):
        from cantocaptions_ai.__main__ import build_parser
        from cantocaptions_ai.pipeline.cli_config import load_cfg_file
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "user.cfg"
            path.write_text("[proofreading]\nproofread = gemini\nproofread_max_cost = None\n",
                            encoding="utf-8")
            loaded = load_cfg_file(path, build_parser())
        self.assertEqual(loaded["proofread"], "gemini")
        self.assertIsNone(loaded["proofread_max_cost"])


class TestValidation(unittest.TestCase):
    def _cfg(self, **kw):
        cfg = PipelineConfig(device="cpu", proofread="gemini", **kw)
        return cfg

    def test_missing_sdk_or_key_fails_at_startup(self):
        from cantocaptions_ai.errors import ConfigError
        from cantocaptions_ai.pipeline.transcribe import _validate_proofreading
        with mock.patch.object(providers, "preflight", return_value="no API key"):
            with self.assertRaisesRegex(ConfigError, "no API key"):
                _validate_proofreading(self._cfg())

    def test_dry_run_needs_neither(self):
        from cantocaptions_ai.pipeline.transcribe import _validate_proofreading
        with mock.patch.object(providers, "preflight", return_value="no API key"):
            _validate_proofreading(self._cfg(proofread_dry_run=True))

    def test_unknown_standard_and_missing_files(self):
        from cantocaptions_ai.errors import ConfigError
        from cantocaptions_ai.pipeline.transcribe import _validate_proofreading
        with mock.patch.object(providers, "preflight", return_value=None):
            with self.assertRaisesRegex(ConfigError, "no proofreading standard 'nope'"):
                _validate_proofreading(self._cfg(proofread_standard="nope"))
            with self.assertRaisesRegex(ConfigError, "not found"):
                _validate_proofreading(self._cfg(proofread_conventions="missing.md"))

    def test_reference_subtitle_is_allowed_for_proofreading_alone(self):
        from cantocaptions_ai.pipeline.transcribe import validate_config
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "ref.srt"
            ref.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
            with mock.patch.object(providers, "preflight", return_value=None):
                validate_config(self._cfg(reference_subtitle=str(ref)))


class TestPrompt(unittest.TestCase):
    def test_standard_is_spliced_in(self):
        text = system_prompt(YUE)
        self.assertIn("written Cantonese", text)
        self.assertIn("CantoCaptions", text)
        self.assertIn("## Sentence-final particles", text)       # the conventions file
        self.assertIn("陳大文、大文、阿文", text)
        self.assertNotIn("{", text.replace("{\"", ""))             # no placeholder left

    def test_a_language_without_a_standard_gets_a_clean_generic_prompt(self):
        text = system_prompt(get_language_pack("fr").standard_for(None))
        self.assertIn("proofreader of fr subtitles", text)
        for placeholder in ("{LANGUAGE_NAME}", "{DESCRIPTION}", "{ERROR_EXAMPLES}",
                            "{NAME_EXAMPLE}", "{CONVENTIONS}", "{NOT_ERRORS}"):
            self.assertNotIn(placeholder, text)

    def test_overrides(self):
        self.assertIn("MY RULES", system_prompt(YUE, conventions="MY RULES"))
        self.assertNotIn("## Sentence-final particles", system_prompt(YUE, conventions="MY RULES"))
        self.assertEqual(system_prompt(YUE, template="only {LANGUAGE_NAME}"),
                         "only written Cantonese (粵文)")

    def test_names_come_first_in_the_schema(self):
        self.assertEqual(list(OUTPUT_SCHEMA["properties"])[0], "names")


class TestStream(unittest.TestCase):
    def test_reference_follows_the_cue_it_overlaps_and_gaps_are_marked(self):
        cues = [Cue(i + 1, s["start"], s["end"], s["text"]) for i, s in enumerate(segments())]
        lines = render_stream(cues, REFERENCE).splitlines()
        self.assertEqual(lines[lines.index("1|你返嚟喇") + 1], "R 你回來了？")
        self.assertEqual(lines[lines.index("3|我個名叫做柏克") + 1], "R 我叫帕克")
        self.assertEqual(lines[lines.index("4|柏克唔喺度") - 1], "--")      # a 2 s pause

    def test_context_cues_and_second_lines(self):
        text = render_stream([Cue(1, 0, 1, "上句", editable=False), Cue(2, 1, 2, "-甲\n-乙")])
        self.assertEqual(text.splitlines(), ["~1|上句", "2|-甲", "  -乙"])

    def test_reference_sentence_only_when_there_is_one(self):
        cues = [Cue(1, 0, 1, "x")]
        self.assertNotIn("Lines starting `R `", user_message(cues, []))
        self.assertIn("Lines starting `R `", user_message(cues, REFERENCE))


class TestApply(unittest.TestCase):
    def test_validation(self):
        current = {1: "a", 2: "b"}
        out = apply_answer(current, {"edits": [{"id": 9, "text": "x"}, {"id": 1, "text": "a"},
                                              {"id": 2, "text": "c"}, {"id": 2, "text": "d"}]},
                           PLAIN, SPACED_SCRIPT)
        self.assertEqual(out.texts, {2: "c"})
        self.assertEqual(sorted(e["problem"] for e in out.invalid),
                         ["duplicate id", "no change", "not an editable cue id"])

    def test_low_confidence_becomes_a_flag(self):
        out = apply_answer({1: "a"}, {"edits": [{"id": 1, "text": "b", "confidence": "low"}]},
                           PLAIN, SPACED_SCRIPT, min_confidence="medium")
        self.assertEqual(out.texts, {})
        self.assertIn("low confidence", out.flags[0]["note"])

    def test_foreign_register_guard(self):
        self.assertEqual(introduced_foreign("唔知武家會有冇技術", "既冇加護又冇技術", YUE, CJK_SCRIPT), ["既"])
        self.assertEqual(introduced_foreign("但是好難", "但是好煩", YUE, CJK_SCRIPT), [])
        out = apply_answer({1: "唔知武家會有冇技術"},
                           {"edits": [{"id": 1, "text": "既冇加護又冇技術", "confidence": "high"}]},
                           YUE, CJK_SCRIPT)
        self.assertEqual(out.texts, {})
        self.assertIn("Standard Written Chinese 既", out.flags[0]["note"])
        spaced = ProofreadStandard("s", "x", foreign_register=("thou",), foreign_register_name="archaic")
        self.assertEqual(introduced_foreign("you go", "thou go", spaced, SPACED_SCRIPT), ["thou"])
        self.assertEqual(introduced_foreign("thou go", "thou went", spaced, SPACED_SCRIPT), [])

    def test_name_table_safeguards(self):
        table = name_table([{"use": "帕克", "draft_forms": ["柏克", "克", "帕克"]},
                            {"use": "阿明", "draft_forms": ["明", "阿明仔"]}])
        self.assertEqual(table, {"阿明仔": "阿明", "柏克": "帕克"})

    def test_names_respect_word_boundaries_in_a_spaced_script(self):
        self.assertEqual(apply_names("Anne and Annual", {"Anne": "Ann"}, SPACED_SCRIPT)[0],
                         "Ann and Annual")
        self.assertEqual(apply_names("柏克唔喺度", {"柏克": "帕克"}, CJK_SCRIPT)[0], "帕克唔喺度")


class TestMoves(unittest.TestCase):
    """Words are never carried from one cue into its neighbour."""

    def test_detects_moves_both_ways_and_with_a_fix(self):
        self.assertTrue(moves_text("冇用啦不", "冇用啦", "如等聽日", "不如等聽日"))
        self.assertTrue(moves_text("今次弊喇", "今次弊喇，放心啦", "放心啦，佢冇事", "佢冇事"))
        self.assertTrue(moves_text("條腸嘅顏", "條腸嘅顏色", "色好靚", "好靚"))
        # moved and fixed at once: still a move
        self.assertTrue(moves_text("全部為咗我，所以", "全部為咗我", "你要報恩", "所以你要報恩啊"))

    def test_independent_fixes_are_not_moves(self):
        self.assertFalse(moves_text("你噉講嘅係喎", "你噉講又係喎", "唔該姐姐", "唔該借借"))
        self.assertFalse(moves_text("走嚟", "走嚟？", "做架兩", "做架梁"))

    def test_a_move_is_refused_whole_and_flagged(self):
        out = apply_answer({1: "冇用啦不", 2: "如等聽日", 3: "唔該姐姐"},
                           {"edits": [{"id": 1, "text": "冇用啦"}, {"id": 2, "text": "不如等聽日"},
                                      {"id": 3, "text": "唔該借借"}]}, YUE, CJK_SCRIPT)
        self.assertEqual(out.texts, {3: "唔該借借"})
        self.assertEqual(len(out.flags), 1)
        self.assertIn("moving text", out.flags[0]["note"])

    def test_the_prompt_no_longer_offers_moves(self):
        text = system_prompt(YUE)
        self.assertIn("never move", text)
        self.assertNotIn("`moved`", text)
        self.assertNotIn("moved", OUTPUT_SCHEMA["properties"]["edits"]["items"]["properties"]["type"]["enum"])


class TestParticles(unittest.TestCase):
    """Sentence-final particles are protected by default (``proofread_particles``)."""

    def protect(self, before, after, std=YUE, script=CJK_SCRIPT):
        return protect_particles(before, after, std, script)

    def test_swaps_additions_and_drops_are_undone(self):
        self.assertEqual(self.protect("走啦", "走喇"), ("走啦", [("啦", "喇")]))
        self.assertEqual(self.protect("睇下", "睇下吖")[0], "睇下")
        self.assertEqual(self.protect("好痛㗎", "好痛")[0], "好痛㗎")
        self.assertEqual(self.protect("而家喺邊㗎吓？", "而家喺邊𠿪？")[0], "而家喺邊㗎吓？")

    def test_the_rest_of_the_edit_survives(self):
        # the word fix applies; the particle swap before the comma does not
        self.assertEqual(self.protect("唔係緊要事啦，我就擰握你個頭啊", "唔係緊要事呢，我就擰甩你個頭啊")[0],
                         "唔係緊要事啦，我就擰甩你個頭啊")
        # punctuation the answer added is kept
        self.assertEqual(self.protect("你返嚟喇", "你返嚟嗱？")[0], "你返嚟喇？")
        self.assertEqual(self.protect("唔通係夢嚟𠸏", "唔通係夢嚟𠸏？"), ("唔通係夢嚟𠸏？", []))

    def test_only_clause_final_runs_count(self):
        # 呢 as "this" and 嘅 as a possessive are words, not particles
        self.assertEqual(self.protect("呢度好靚", "嗰度好靚"), ("嗰度好靚", []))
        self.assertEqual(self.protect("一個10円個舊幣", "一個10円嘅舊幣")[1], [])

    def test_allow_and_standards_without_particles(self):
        out = apply_answer({1: "走啦"}, {"edits": [{"id": 1, "text": "走喇"}]}, YUE, CJK_SCRIPT,
                           particles="allow")
        self.assertEqual(out.texts, {1: "走喇"})
        self.assertEqual(protect_particles("走啦", "走喇", PLAIN, CJK_SCRIPT), ("走喇", []))

    def test_a_particle_only_edit_becomes_a_flag(self):
        out = apply_answer({1: "走啦", 2: "佢好開心啊"},
                           {"edits": [{"id": 1, "text": "走喇", "reason": "change of state"},
                                      {"id": 2, "text": "佢好開心吖"}]}, YUE, CJK_SCRIPT)
        self.assertEqual(out.texts, {})
        self.assertEqual(len(out.flags), 2)
        self.assertIn("啦→喇", out.flags[0]["note"])

    def test_a_spaced_script_protects_whole_words(self):
        std = ProofreadStandard("v", "Vietnamese", final_particles=("nhé", "nhỉ"))
        self.assertEqual(protect_particles("đi nhé.", "đi nhỉ.", std, SPACED_SCRIPT)[0], "đi nhé.")
        self.assertEqual(protect_particles("đi nhé.", "về nhé.", std, SPACED_SCRIPT), ("về nhé.", []))

    def test_the_prompt_follows_the_setting(self):
        protect, allow = system_prompt(YUE), system_prompt(YUE, particles="allow")
        self.assertIn("Leave every sentence-final", protect)
        self.assertNotIn("Wrong sentence-final particle", protect)
        self.assertIn("Wrong sentence-final particle", allow)
        self.assertNotIn("Leave every sentence-final", allow)
        self.assertNotIn("{NOT_ERRORS}", system_prompt(PLAIN))


class FakeProvider:
    def __init__(self, *answers):
        import threading
        self.answers = list(answers)
        self.requests = []
        self._lock = threading.Lock()

    def __call__(self, provider, model, req, effort="medium", timeout_s=0, cache=None):
        with self._lock:
            self.requests.append(req)
            n = len(self.requests)
        return reply(self.answers[min(n - 1, len(self.answers) - 1)])


class TestProofreader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.debug = self.tmp.name

    def _run(self, segs, fake, save_dir=None, **kw):
        self.open_cache = mock.Mock(return_value="cache-1")
        self.close_cache = mock.Mock()
        with mock.patch.object(providers, "send", fake), \
                mock.patch.object(providers, "count_tokens", return_value=None), \
                mock.patch.object(providers, "open_cache", self.open_cache), \
                mock.patch.object(providers, "close_cache", self.close_cache):
            return Proofreader(settings(**kw)).run(
                "ep", segs, REFERENCE, debug_dir=None if save_dir else self.debug,
                load_debug_dir=kw.get("_load"), save_dir=save_dir)

    def test_edits_names_flags_notes_and_artifacts(self):
        segs = segments()
        fake = FakeProvider(ANSWER)
        done = self._run(segs, fake)
        self.assertEqual([s["text"] for s in segs],
                         ["佢返嚟喇？", "我哋去食飯啦", "我個名叫做帕克", "帕克唔喺度"])
        self.assertIn("proofread:mishearing", segs[0]["notes"])
        self.assertIn("proofread:name", segs[3]["notes"])
        self.assertEqual(len(done.flags), 1)
        stage = Path(self.debug) / "ep" / "proofread"
        for f in ("chunk_00.json", "chunk_00.request.md", "changes.srt", "flags.srt",
                  "summary.json"):
            self.assertTrue((stage / f).is_file(), f)
        changes = (stage / "changes.srt").read_text(encoding="utf-8")
        # new text over its own [was] line, the changed characters coloured in each
        self.assertIn('\n<font color="#66ff66">佢</font>返嚟喇<font color="#66ff66">？</font>\n'
                      '[was] <font color="#ff6666">你</font>返嚟喇\n', changes)

    def test_replay_does_not_pay_again_but_a_changed_draft_does(self):
        self._run(segments(), FakeProvider(ANSWER))
        with mock.patch.object(providers, "send", side_effect=AssertionError("billed twice")):
            done = Proofreader(settings()).run("ep", segments(), REFERENCE,
                                               debug_dir=self.debug, load_debug_dir=self.debug)
        self.assertEqual(done.replayed, 1)
        changed = segments()
        changed[1]["text"] = "我哋去飲茶啦"
        fake = FakeProvider(ANSWER)
        with mock.patch.object(providers, "send", fake), \
                mock.patch.object(providers, "count_tokens", return_value=None):
            Proofreader(settings()).run("ep", changed, REFERENCE, debug_dir=self.debug,
                                        load_debug_dir=self.debug)
        self.assertEqual(len(fake.requests), 1)

    def test_dry_run_sends_nothing(self):
        segs = segments()
        with mock.patch.object(providers, "send", side_effect=AssertionError("sent")):
            done = Proofreader(settings(dry_run=True)).run("ep", segs, REFERENCE,
                                                           debug_dir=self.debug)
        self.assertEqual(done.requests, 0)
        self.assertEqual(segs[0]["text"], "你返嚟喇")
        self.assertTrue((Path(self.debug) / "ep" / "proofread" / "ep.proofread.request.md").is_file())

    def test_cost_ceiling_refuses_before_sending(self):
        with mock.patch.object(providers, "send", side_effect=AssertionError("sent")), \
                mock.patch.object(providers, "count_tokens", return_value=2_000_000):
            with self.assertRaises(providers.ProviderError):
                Proofreader(settings(max_cost=0.10)).run("ep", segments(), REFERENCE)

    def test_a_failed_request_leaves_the_file_as_written(self):
        from cantocaptions_ai.pipeline.transcribe import _proofread
        result = {"segments": segments()}
        broken = mock.Mock()
        broken.run.side_effect = providers.ProviderError("network down", {"cost_usd": 0.002})
        _proofread(broken, "ep", result, REFERENCE, None, None, None, None)
        self.assertEqual([s["text"] for s in result["segments"]], [s["text"] for s in segments()])

    def test_edited_cues_are_recleaned(self):
        from cantocaptions_ai.pipeline.transcribe import _proofread
        result = {"segments": segments()}
        cleaner = mock.Mock()
        cleaner.clean.side_effect = lambda t: t + "!"
        cleaner.is_noise.return_value = False
        fake = FakeProvider(ANSWER)
        with mock.patch.object(providers, "send", fake), \
                mock.patch.object(providers, "count_tokens", return_value=None):
            _proofread(Proofreader(settings()), "ep", result, REFERENCE, cleaner, None,
                       None, None)
        texts = [s["text"] for s in result["segments"]]
        self.assertEqual(texts[0], "佢返嚟喇？!")         # edited: cleaned again
        self.assertEqual(texts[1], "我哋去食飯啦")         # untouched: not re-cleaned

    def _run_with_progress(self, segs, fake, **kw):
        progress = mock.Mock()
        with mock.patch.object(providers, "send", fake), \
                mock.patch.object(providers, "count_tokens", return_value=None), \
                mock.patch.object(providers, "open_cache", return_value=None), \
                mock.patch.object(providers, "close_cache"):
            Proofreader(settings(**kw)).run("ep", segs, REFERENCE, debug_dir=self.debug,
                                            progress=progress)
        return progress

    def test_one_request_is_a_status_naming_what_is_awaited(self):
        progress = self._run_with_progress(segments(), FakeProvider(ANSWER))
        progress.set_total.assert_not_called()
        status = progress.status.call_args_list[0].args[0]
        self.assertTrue(status.startswith("waiting for "), status)
        self.assertIn("(4 cues)", status)

    def test_chunks_are_a_bar_advanced_as_each_is_answered(self):
        segs = [{"start": float(i), "end": i + 0.9, "text": f"第{i}句"} for i in range(10)]
        progress = self._run_with_progress(
            segs, FakeProvider({"names": [], "edits": [], "flags": []}), chunk_cues=4)
        progress.set_total.assert_called_once_with(3, unit="chunk")
        self.assertEqual(sum(c.args[0] for c in progress.advance.call_args_list), 3)

    def test_chunking_sends_each_window_with_context(self):
        segs = [{"start": float(i), "end": i + 0.9, "text": f"第{i}句"} for i in range(10)]
        fake = FakeProvider({"names": [], "edits": [], "flags": []})
        self._run(segs, fake, chunk_cues=4, chunk_context=1)
        self.assertEqual(len(fake.requests), 3)
        self.assertIn("1|第0句", fake.requests[0].user)       # the first chunk goes first, alone
        second = next(r.user for r in fake.requests if "\n5|第4句" in r.user)
        self.assertIn("~4|第3句", second)       # one cue of context from the first chunk
        self.assertNotIn("~3|", second)
        self.assertIn("~9|第8句", second)       # ...and one from the next
        self.assertNotIn("~10|", second)
        self.assertNotIn("~5|", second)

    def test_chunks_share_one_prompt_cache(self):
        segs = [{"start": float(i), "end": i + 0.9, "text": f"第{i}句"} for i in range(10)]
        fake = FakeProvider({"names": [], "edits": [], "flags": []})
        self._run(segs, fake, chunk_cues=4)
        self.open_cache.assert_called_once()
        self.close_cache.assert_called_once_with("gemini", "cache-1")
        self._run(segments(), FakeProvider(ANSWER))         # one request: nothing to share
        self.open_cache.assert_not_called()

    def test_a_failed_chunk_costs_only_itself(self):
        segs = [{"start": float(i), "end": i + 0.9, "text": f"第{i}句"} for i in range(8)]
        good = {"names": [], "edits": [{"id": 1, "text": "第零句", "confidence": "high"}],
                "flags": []}

        def flaky(provider, model, req, effort="medium", timeout_s=0, cache=None):
            if "\n5|" in req.user:
                raise providers.ProviderError("boom", {"cost_usd": 0.02})
            return reply(good)
        with self.assertLogs("cantocaptions_ai.pipeline.proofread", "WARNING") as logs:
            done = self._run(segs, flaky, chunk_cues=4)
        self.assertEqual(segs[0]["text"], "第零句")
        self.assertIn("chunk 2/2 failed", "\n".join(logs.output))
        self.assertAlmostEqual(done.usage["cost_usd"], 0.03)

        def broken(*a, **k):
            raise providers.ProviderError("down")
        with self.assertRaises(providers.ProviderError):
            self._run(segments(), broken)

    def test_answers_and_review_are_saved_without_a_debug_dir(self):
        out = Path(self.tmp.name) / "out" / "ep.proofread"
        self._run(segments(), FakeProvider(ANSWER), save_dir=str(out))
        for f in ("chunk_00.json", "chunk_00.request.md", "changes.srt", "flags.srt",
                  "summary.json"):
            self.assertTrue((out / f).is_file(), f)

    def test_a_chunked_dry_run_writes_every_request(self):
        segs = [{"start": float(i), "end": i + 0.9, "text": f"第{i}句"} for i in range(10)]
        with mock.patch.object(providers, "send") as send:
            Proofreader(settings(chunk_cues=4, dry_run=True)).run("ep", segs, debug_dir=self.debug)
        send.assert_not_called()
        stage = Path(self.debug) / "ep" / "proofread"
        self.assertEqual(sorted(p.name for p in stage.glob("*.request.*.md")),
                         ["ep.proofread.request.00.md", "ep.proofread.request.01.md",
                          "ep.proofread.request.02.md"])


class TestConsole(unittest.TestCase):
    def test_redirected_output_does_not_raise(self):
        from cantocaptions_ai.utils.log_utils import make_console_safe
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding="cp1252")
        with mock.patch.object(sys, "__stdout__", stream):
            make_console_safe()
            print("字幕 → ═", file=sys.__stdout__)
            stream.flush()
        self.assertIn("字幕 → ═", raw.getvalue().decode("utf-8"))


class TestRequestShapes(unittest.TestCase):
    """Real SDK request objects, built offline. Skipped when the SDK is not installed."""

    def test_gemini_request(self):
        try:
            from google.genai import types  # noqa: F401
        except ImportError:
            self.skipTest("google-genai not installed")
        seen = {}

        class Models:
            def generate_content(self, model, contents, config):
                seen["config"] = config
                cand = mock.Mock(finish_reason="STOP")
                return mock.Mock(text=json.dumps(ANSWER), candidates=[cand],
                                 usage_metadata=mock.Mock(prompt_token_count=100,
                                                          cached_content_token_count=0,
                                                          candidates_token_count=10,
                                                          thoughts_token_count=5))

        with mock.patch.object(providers, "_gemini_client", return_value=mock.Mock(models=Models())):
            out = providers.send("gemini", "gemini-3.7-flash",
                                 providers.Request("sys", "user"), effort="medium")
        self.assertEqual(seen["config"].response_json_schema, OUTPUT_SCHEMA)
        self.assertEqual(out.answer["edits"][0]["id"], 1)
        self.assertGreater(out.usage["cost_usd"], 0)

    def test_gemini_client_has_a_timeout(self):
        try:
            import google.genai  # noqa: F401
        except ImportError:
            self.skipTest("google-genai not installed")
        with mock.patch("google.genai.Client") as client:
            providers._gemini_client(timeout_s=600)
        opts = client.call_args.kwargs["http_options"]
        self.assertEqual(opts.timeout, 600_000)


SRT = """1
00:00:00,000 --> 00:00:01,000
你返嚟喇

2
00:00:01,000 --> 00:00:02,500
- 我哋去食飯啦
- 好啊

3
00:00:02,500 --> 00:00:03,000
我個名叫做柏克
"""


class TestProofreadInput(unittest.TestCase):
    """--proofread_input: an existing subtitle through the proofreader, no audio."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.src = self.tmp / "ep.srt"
        self.src.write_text(SRT, encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def _cli(self, *argv):
        from cantocaptions_ai.__main__ import cli
        err = io.StringIO()
        with mock.patch.object(sys, "argv", ["cantocaptions", *argv]), \
                mock.patch("sys.stderr", err), self.assertRaises(SystemExit):
            cli()
        return err.getvalue()

    def test_refuses_audio_and_a_missing_provider(self):
        self.assertIn("takes no audio", self._cli("ep.mkv", "--proofread_input", str(self.src),
                                                  "--proofread", "gemini"))
        self.assertIn("needs a provider", self._cli("--proofread_input", str(self.src),
                                                    "--proofread", "none"))

    def test_refuses_what_is_not_a_timed_subtitle(self):
        txt = self.tmp / "lines.txt"
        txt.write_text("你好\n", encoding="utf-8")
        with mock.patch.object(providers, "preflight", return_value=None):
            self.assertIn("timed subtitle", self._cli("--proofread_input", str(txt),
                                                       "--proofread", "gemini"))
            self.assertIn("not found", self._cli("--proofread_input", str(self.tmp / "no.srt"),
                                                 "--proofread", "gemini"))
            other = self.tmp / "b.srt"
            other.write_text(SRT, encoding="utf-8")
            self.assertIn("one proofread_input", self._cli(
                "--proofread_input", str(self.src), str(other), "--proofread", "gemini",
                "--reference_subtitle", str(other)))

    def _run(self, **kw):
        from cantocaptions_ai.pipeline.transcribe import proofread_files
        cfg = PipelineConfig(device="cpu", proofread="gemini", output_dir=str(self.tmp / "out"),
                             debug_dir=str(self.tmp / "debug"), **kw)
        answer = {"names": [{"use": "帕克", "draft_forms": ["柏克"], "reference": "帕克"}],
                  "edits": [{"id": 1, "text": "佢返嚟喇", "type": "mishearing",
                             "confidence": "high", "reason": "pronoun"}],
                  "flags": []}
        fake = FakeProvider(answer)
        with mock.patch.object(providers, "send", fake), \
                mock.patch.object(providers, "count_tokens", return_value=None):
            proofread_files([str(self.src)], cfg)
        return fake

    def test_writes_a_proofread_copy_and_keeps_the_rest_exact(self):
        from cantocaptions_ai.utils.subtitles import read_subtitle_cues
        fake = self._run(no_clean_text=True)
        out = read_subtitle_cues(str(self.tmp / "out" / "ep.proofread.srt"))
        src = read_subtitle_cues(str(self.src))
        self.assertEqual([(c.start, c.end) for c in out], [(c.start, c.end) for c in src])
        self.assertEqual([c.text for c in out],
                         ["佢返嚟喇", "- 我哋去食飯啦\n- 好啊", "我個名叫做帕克"])
        self.assertEqual(self.src.read_text(encoding="utf-8"), SRT)      # input untouched
        self.assertIn("no reference subtitle", fake.requests[0].user)
        self.assertTrue((self.tmp / "debug" / "ep" / "proofread" / "changes.srt").is_file())

    def test_a_reference_is_interleaved(self):
        ref = self.tmp / "ref.srt"
        ref.write_text("1\n00:00:00,100 --> 00:00:00,900\n你回來了\n", encoding="utf-8")
        fake = self._run(no_clean_text=True, reference_subtitle=str(ref))
        self.assertIn("R 你回來了", fake.requests[0].user)

    def test_dry_run_writes_no_subtitle(self):
        with mock.patch.object(providers, "send") as send:
            from cantocaptions_ai.pipeline.transcribe import proofread_files
            proofread_files([str(self.src)], PipelineConfig(
                device="cpu", proofread="gemini", proofread_dry_run=True, no_clean_text=True,
                output_dir=str(self.tmp / "out"), debug_dir=str(self.tmp / "debug")))
        send.assert_not_called()
        self.assertFalse((self.tmp / "out" / "ep.proofread.srt").exists())
        self.assertTrue((self.tmp / "debug" / "ep" / "proofread" / "ep.proofread.request.md").is_file())


class TestBasicCleaning(unittest.TestCase):
    """Finished subtitles get only the language's basic cleaning before proofreading."""

    def _pc(self, **kw):
        from cantocaptions_ai.pipeline.transcribe import _build_precleaner
        pack = get_language_pack("yue")
        return _build_precleaner(PipelineConfig(device="cpu", proofread="gemini", **kw),
                                 pack, pack.resolve(None))

    def test_punctuation_and_variants_only(self):
        from cantocaptions_ai.pipeline.transcribe import _preclean_text
        pc = self._pc()
        self.assertEqual(_preclean_text(pc, "你好嗎?我好好,多謝"), "你好嗎？我好好，多謝")
        self.assertEqual(_preclean_text(pc, "爲咗佢 , 我咩都肯做..."), "為咗佢，我咩都肯做…")
        self.assertEqual(_preclean_text(pc, "咁樣都得,"), "咁樣都得")         # end comma only
        self.assertEqual(_preclean_text(pc, "八點三十分見"), "八點三十分見")    # no numerals
        # line by line: a two-speaker cue keeps its break; the dash loses its space
        self.assertEqual(_preclean_text(pc, "- 你去邊呀?\n- 返屋企"), "-你去邊呀？\n-返屋企")

    def test_off_switches(self):
        self.assertIsNone(self._pc(proofread_preclean=False))
        self.assertIsNone(self._pc(no_clean_text=True))
        from cantocaptions_ai.pipeline.transcribe import _build_precleaner
        fr = get_language_pack("fr")
        self.assertIsNone(_build_precleaner(PipelineConfig(device="cpu", proofread="gemini"),
                                            fr, fr.resolve(None)))

    def test_a_proofread_input_is_cleaned_before_it_is_sent(self):
        from cantocaptions_ai.pipeline.transcribe import proofread_files
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "ep.srt"
            src.write_text("1\n00:00:00,000 --> 00:00:01,000\n你好嗎?\n", encoding="utf-8")
            fake = FakeProvider({"names": [], "edits": [], "flags": []})
            cfg = PipelineConfig(device="cpu", proofread="gemini",
                                 output_dir=str(Path(tmp) / "out"), debug_dir="")
            with mock.patch.object(providers, "send", fake), \
                    mock.patch.object(providers, "count_tokens", return_value=None):
                proofread_files([str(src)], cfg)
            self.assertIn("1|你好嗎？", fake.requests[0].user)
            pre = (Path(tmp) / "out" / "ep.proofread" / "precleaned.srt").read_text(encoding="utf-8")
            self.assertIn("[was] 你好嗎", pre)


class TestDurationReport(unittest.TestCase):
    def test_proofreading_is_a_row_with_its_cost(self):
        from cantocaptions_ai.pipeline.transcribe import _proofread
        from cantocaptions_ai.utils.log_utils import TranscriptionSummary
        summary = TranscriptionSummary(enabled=True)
        result = {"segments": segments()}
        with mock.patch.object(providers, "send", FakeProvider(ANSWER)), \
                mock.patch.object(providers, "count_tokens", return_value=None):
            _proofread(Proofreader(settings()), "ep", result, REFERENCE, None, None, None, None,
                       summary=summary)
        (label, load, run, vram), = summary._stages
        self.assertEqual(label, "Proofreading")
        self.assertIsNone(vram)                      # a network call: no VRAM column
        self.assertGreaterEqual(run, 0.0)
        self.assertAlmostEqual(summary._amounts["Proofreading cost"][0], 0.01)


class TestRunPlan(unittest.TestCase):
    """The whole run is described before anything runs, proofreading included."""

    def _describe(self, **kw):
        from cantocaptions_ai.pipeline.transcribe import _describe_run
        stage = mock.Mock()
        stage.describe.return_value = "VAD [compute] → Transcript realignment"
        cfg = PipelineConfig(device="cpu", proofread="gemini", realign="in.srt", **kw)
        return _describe_run(cfg, None, [stage], Proofreader(settings()), None, None,
                             object(), False)

    def test_media_timed_reference_proofreads_the_output(self):
        line = self._describe(reference_subtitle="ref.srt", reference_timing="media")
        self.assertTrue(line.startswith("VAD [compute]"))
        self.assertIn("Basic cleaning → Proofread output [gemini gemini-3.7-flash, medium; "
                      "reference ref.srt, media-timed]", line)
        self.assertTrue(line.endswith("Write srt + Subtitle Edit bookmarks"))

    def test_proofread_first_comes_first(self):
        from cantocaptions_ai.pipeline.transcribe import _describe_run
        stage = mock.Mock()
        stage.describe.return_value = "VAD [compute]"
        cfg = PipelineConfig(device="cpu", proofread="gemini", realign="in.srt")
        line = _describe_run(cfg, None, [stage], Proofreader(settings()), None, None,
                             object(), True)
        self.assertTrue(line.startswith("Basic cleaning (realign input) → Proofread realign "
                                        "input in.srt [gemini gemini-3.7-flash, medium; no reference]"))
        self.assertEqual(line.count("Proofread"), 1)


class TestInterrupt(unittest.TestCase):
    """Ctrl+C must stop a run even while a request is blocked on the network."""

    def test_results_come_back_in_order(self):
        from cantocaptions_ai.pipeline.proofread import _in_background
        import time
        self.assertEqual(_in_background(lambda x: (time.sleep(0.05 * (5 - x)), x)[1],
                                        list(range(5)), 3), [0, 1, 2, 3, 4])

    def test_ctrl_c_lands_while_a_request_is_still_blocked(self):
        import _thread
        import threading
        import time
        from cantocaptions_ai.pipeline.proofread import _in_background
        release = threading.Event()
        threading.Timer(0.3, _thread.interrupt_main).start()      # what Ctrl+C does
        t0 = time.time()
        with self.assertRaises(KeyboardInterrupt):
            _in_background(lambda _: release.wait(30), [1, 2], 2)
        self.assertLess(time.time() - t0, 3)       # not the 30 s the "request" would take
        release.set()

    def test_answers_received_before_the_interrupt_are_saved(self):
        import _thread
        import threading
        segs = [{"start": float(i), "end": i + 0.9, "text": f"第{i}句"} for i in range(8)]
        release = threading.Event()

        def send(provider, model, req, effort="medium", timeout_s=0, cache=None):
            if "\n5|" in req.user:                  # the second chunk hangs
                threading.Timer(0.3, _thread.interrupt_main).start()
                release.wait(30)
            return reply({"names": [], "edits": [], "flags": []})
        out = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(out, ignore_errors=True))
        with mock.patch.object(providers, "send", send), \
                mock.patch.object(providers, "count_tokens", return_value=None), \
                mock.patch.object(providers, "open_cache", return_value=None), \
                self.assertRaises(KeyboardInterrupt):
            Proofreader(settings(chunk_cues=4)).run("ep", segs, save_dir=str(out))
        # What is on disk when the interrupt lands. Checked *before* the hung request is
        # released: the abandoned daemon thread then saves its own answer as soon as it
        # arrives, and whether that write beats an assertion made after release.set() is up
        # to the scheduler (it did on a Linux CI runner).
        self.assertTrue((out / "chunk_00.json").is_file())          # paid for, so kept
        self.assertFalse((out / "chunk_01.json").exists())
        release.set()
        for t in threading.enumerate():     # let it finish before the temp dir is removed
            if t.name.startswith("proofread-"):
                t.join(5)


class TestBookmarks(unittest.TestCase):
    """Subtitle Edit bookmarks for the proofread SRT (format of libse BookmarkPersistence)."""

    def test_the_file_is_what_subtitle_edit_writes(self):
        from cantocaptions_ai.pipeline.proofread.review import write_bookmarks
        with tempfile.TemporaryDirectory() as tmp:
            srt = os.path.join(tmp, "ep.srt")
            path = write_bookmarks(srt, [(3, '[was] 你"好"\n[flag] a\\b'), (0, "x")])
            self.assertEqual(path, srt + ".SE.bookmarks")
            raw = Path(path).read_bytes()
            self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
            self.assertEqual(raw[3:].decode("utf-8"),
                             '{"bookmarks":[\r\n{"idx":0,"txt":"x"},'
                             '{"idx":3,"txt":"[was] 你\\"好\\"<br />[flag] a\\\\b"}]}\r\n')
            self.assertIsNone(write_bookmarks(srt, []))

    def test_an_existing_file_is_never_overwritten(self):
        from cantocaptions_ai.pipeline.proofread.review import write_bookmarks
        with tempfile.TemporaryDirectory() as tmp:
            srt = os.path.join(tmp, "ep.proofread.srt")
            first = write_bookmarks(srt, [(0, "a")])
            second = write_bookmarks(srt, [(0, "b")])
            third = write_bookmarks(srt, [(0, "c")])
            self.assertEqual(Path(first).name, "ep.proofread.srt.SE.bookmarks")
            self.assertEqual(Path(second).name, "ep.proofread (1).srt.SE.bookmarks")
            self.assertEqual(Path(third).name, "ep.proofread (2).srt.SE.bookmarks")
            self.assertIn('"txt":"a"', Path(first).read_text(encoding="utf-8-sig"))

    def test_indices_follow_the_cues_actually_written(self):
        from cantocaptions_ai.pipeline.proofread.review import bookmark_marks
        a, b, c = {"text": "a"}, {"text": "b"}, {"text": "c"}
        notes = {id(a): ["[was] x"], id(c): ["[was] y", "[flag] check"]}
        self.assertEqual(bookmark_marks([a, b, c], notes),
                         [(0, "[was] x"), (2, "[was] y\n[flag] check")])
        # b dropped by re-cleaning: c moves up to index 1
        self.assertEqual(bookmark_marks([a, c], notes), [(0, "[was] x"), (1, "[was] y\n[flag] check")])

    def test_marks_follow_the_text_through_realign(self):
        from cantocaptions_ai.pipeline.proofread.review import remap_marks
        source = ["你返嚟喇", "我哋去食飯啦，好唔好？", "嗯", "我個名叫做帕克", "帕克唔喺度", "再見"]
        marks = [(0, "a"), (1, "b"), (2, "c"), (3, "d"), (5, "f")]
        # unchanged: one to one
        self.assertEqual(remap_marks(source, marks, source),
                         [(0, "a"), (1, "b"), (2, "c"), (3, "d"), (5, "f")])
        target = [
            "你返嚟喇",                     # 0
            "我哋去食飯啦，",                # 1  cue 1 split in two...
            "好唔好？",                      # 2
            # cue 2 (嗯) dropped as noise
            "我個名叫做帕克 帕克唔喺度",      # 3  cues 3 and 4 merged
            "再見！",                        # 4  cleaned
        ]
        # b: on the first half of its split; c: its cue is gone, so on the cue holding the
        # text just before where it was; d: on the merged cue; f: through the cleaning
        self.assertEqual(remap_marks(source, marks, target),
                         [(0, "a"), (1, "b"), (2, "c"), (3, "d"), (4, "f")])
        merged = ["你返嚟喇 我哋去食飯啦，好唔好？", "嗯 我個名叫做帕克 帕克唔喺度 再見"]
        self.assertEqual(remap_marks(source, marks, merged), [(0, "a\nb"), (1, "c\nd\nf")])
        self.assertEqual(remap_marks(source, marks, []), [])

    def test_a_proofread_input_gets_bookmarks_beside_its_srt(self):
        from cantocaptions_ai.pipeline.transcribe import proofread_files
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "ep.srt"
            src.write_text(SRT, encoding="utf-8")
            answer = {"names": [], "flags": [{"id": 3, "note": "name unclear"}],
                      "edits": [{"id": 2, "text": "- 我哋去食飯囉\n- 好啊", "confidence": "high"}]}
            cfg = PipelineConfig(device="cpu", proofread="gemini", proofread_particles="allow",
                                 no_clean_text=True, output_dir=str(Path(tmp) / "out"),
                                 debug_dir="")
            with mock.patch.object(providers, "send", FakeProvider(answer)), \
                    mock.patch.object(providers, "count_tokens", return_value=None):
                proofread_files([str(src)], cfg)
            marks = (Path(tmp) / "out" / "ep.proofread.srt.SE.bookmarks").read_text(encoding="utf-8-sig")
            self.assertIn('{"idx":1,"txt":"[was] - 我哋去食飯啦 / - 好啊"}', marks)
            self.assertIn('{"idx":2,"txt":"[flag] name unclear"}', marks)


def _cues(starts):
    return [Cue(i + 1, s, s + 1.5, f"第{i}句") for i, s in enumerate(starts)]


class TestReferenceTiming(unittest.TestCase):
    """The reference must share the cues' timeline, and a run says which one that is."""

    STARTS = [3.0 * i + (0.7 if i % 3 else 0.0) for i in range(60)]

    def test_agreement_and_the_shift_that_fixes_it(self):
        from cantocaptions_ai.pipeline.proofread.render import best_reference_shift, reference_agreement
        cues = _cues(self.STARTS)
        ref = [{"start": s + 0.1, "end": s + 1.0, "text": "x"} for s in self.STARTS]
        self.assertEqual(reference_agreement(cues, ref), 1.0)
        late = [dict(r, start=r["start"] + 2.3) for r in ref]
        self.assertLess(reference_agreement(cues, late), 0.6)
        shift, share = best_reference_shift(cues, late)
        self.assertAlmostEqual(shift, -2.3, delta=0.45)
        self.assertGreaterEqual(share, 0.95)

    def test_a_mismatched_reference_is_refused_before_anything_is_sent(self):
        segs = [{"start": s, "end": s + 1.5, "text": f"第{i}句"} for i, s in enumerate(self.STARTS)]
        late = [{"start": s + 2.3, "end": s + 3.0, "text": "x"} for s in self.STARTS]
        with mock.patch.object(providers, "send") as send:
            with self.assertRaisesRegex(providers.ProviderError, "reference_offset -2"):
                Proofreader(settings()).run("ep", segs, late)
        send.assert_not_called()

    def _cfg(self, tmp, **kw):
        sub = Path(tmp) / "sub.srt"
        sub.write_text(SRT, encoding="utf-8")
        ref = Path(tmp) / "ref.srt"
        ref.write_text("1\n00:00:00,100 --> 00:00:00,900\n你回來了\n", encoding="utf-8")
        base = dict(device="cpu", proofread="gemini", realign=str(sub), reference_subtitle=str(ref),
                    output_dir=str(Path(tmp) / "out"), debug_dir="")
        base.update(kw)
        return PipelineConfig(**base)

    def test_realign_with_a_reference_must_say_which_timeline(self):
        from cantocaptions_ai.errors import ConfigError
        from cantocaptions_ai.pipeline.transcribe import validate_config
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(providers, "preflight", return_value=None):
            self.assertEqual(self._cfg(tmp).reference_timing, "media")   # the default
            validate_config(self._cfg(tmp))
            with self.assertRaisesRegex(ConfigError, "needs reference_timing"):
                validate_config(self._cfg(tmp, reference_timing=None))
            validate_config(self._cfg(tmp, reference_timing="subtitle"))
            bare = Path(tmp) / "lines.txt"
            bare.write_text("你好\n", encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "bare transcript"):
                validate_config(self._cfg(tmp, reference_timing="subtitle", realign=str(bare)))

    def test_which_side_of_realign_proofreading_runs_on(self):
        from cantocaptions_ai.pipeline.transcribe import proofreads_before_realign
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(proofreads_before_realign(self._cfg(tmp, reference_timing="subtitle")))
            self.assertFalse(proofreads_before_realign(self._cfg(tmp, reference_timing="media")))
            self.assertTrue(proofreads_before_realign(self._cfg(tmp, reference_subtitle=None)))
            self.assertFalse(proofreads_before_realign(self._cfg(tmp, proofread="none")))

    def test_the_input_is_proofread_first_and_its_copy_realigned(self):
        from cantocaptions_ai.pipeline.transcribe import _proofread_realign_input
        from cantocaptions_ai.utils.subtitles import read_subtitle_cues
        fake = FakeProvider({"names": [], "edits": [{"id": 1, "text": "佢返嚟喇",
                                                     "confidence": "high"}], "flags": []})
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(providers, "send", fake), \
                mock.patch.object(providers, "count_tokens", return_value=None):
            cfg = self._cfg(tmp, reference_timing="subtitle")
            pack = get_language_pack("yue")
            new, carried = _proofread_realign_input(
                cfg, load_proofreader(cfg, pack, pack.resolve(None)))
            self.assertEqual(new.proofread, "none")                 # not proofread twice
            texts, marks = carried                                   # for the realigned output
            self.assertEqual(texts[0], "佢返嚟喇")
            self.assertEqual(marks, [(0, "[was] 你返嚟喇")])
            self.assertTrue(new.realign.endswith("sub.proofread.srt"))
            self.assertEqual(read_subtitle_cues(new.realign)[0].text, "佢返嚟喇")
            self.assertIn("R 你回來了", fake.requests[0].user)       # paired on the input's timeline
            self.assertTrue((Path(tmp) / "out" / "sub.proofread" / "changes.srt").is_file())


if __name__ == "__main__":
    unittest.main()

"""Tests for reference-subtitle-guided LLM correction.

Covers:
- _edit_distance helper
- match_reference_to_segments (time-overlap matching)
- LLMCorrector._sanitize_reference (rejection rules)
- LLMCorrector.correct_with_reference (end-to-end with mocked LLM)

The three spec examples are used as canonical inputs throughout.
"""

import unittest
from unittest.mock import MagicMock

from cantocaptions_ai.pipeline.llm_correction import (
    LLMCorrector,
    _edit_distance,
    match_reference_to_segments,
)


def _seg(start: float, end: float, text: str) -> dict:
    return {'start': start, 'end': end, 'text': text}


def _corrector(semantic: bool = False) -> LLMCorrector:
    return LLMCorrector(
        model=MagicMock(),
        tokenizer=MagicMock(),
        device='cpu',
        semantic_mode=semantic,
    )


# ---------------------------------------------------------------------------
# _edit_distance
# ---------------------------------------------------------------------------

class TestEditDistance(unittest.TestCase):

    def test_identical_strings(self):
        self.assertEqual(_edit_distance('abc', 'abc'), 0)

    def test_empty_inputs(self):
        self.assertEqual(_edit_distance('', ''), 0)
        self.assertEqual(_edit_distance('', 'abc'), 3)
        self.assertEqual(_edit_distance('abc', ''), 3)

    def test_single_substitution(self):
        self.assertEqual(_edit_distance('abc', 'aXc'), 1)

    def test_single_insertion(self):
        self.assertEqual(_edit_distance('ab', 'abc'), 1)

    def test_single_deletion(self):
        self.assertEqual(_edit_distance('abc', 'ac'), 1)

    def test_example1_edit_distance(self):
        # 魚落天 → 余樂天: 2 substitutions out of 17 chars → ratio ≈ 0.12
        primary =   '我硬係覺得唔似喎，魚落天你覺得呢？'
        corrected = '我硬係覺得唔似喎，余樂天你覺得呢？'
        self.assertEqual(_edit_distance(primary, corrected), 2)

    def test_example2_edit_distance(self):
        # idiom half has 5 substitutions out of 18 chars → ratio ≈ 0.28
        primary =   '不如我哋一齊去吖？仲掛得啩鐘鬥得豆'
        corrected = '不如我哋一齊去吖？種瓜得瓜種豆得豆'
        self.assertEqual(_edit_distance(primary, corrected), 5)

    def test_example3_edit_distance(self):
        # Insert 唔 and ， : 2 edits out of 8 chars → ratio = 0.25
        primary =   '我諗係你先諗嘅'
        corrected = '我唔諗，係你先諗嘅'
        self.assertEqual(_edit_distance(primary, corrected), 2)


# ---------------------------------------------------------------------------
# match_reference_to_segments
# ---------------------------------------------------------------------------

class TestMatchReferenceToSegments(unittest.TestCase):

    def test_exact_time_overlap(self):
        segs = [_seg(0.0, 2.0, '廣東話')]
        ref  = [_seg(0.0, 2.0, '普通話')]
        self.assertEqual(match_reference_to_segments(segs, ref), ['普通話'])

    def test_partial_overlap_picks_correct_line(self):
        segs = [_seg(1.0, 3.0, '廣東話')]
        ref  = [_seg(0.0, 2.0, 'A'), _seg(3.5, 5.0, 'B')]
        # Only A overlaps [1.0, 3.0]; B starts after 3.0
        self.assertEqual(match_reference_to_segments(segs, ref), ['A'])

    def test_multiple_overlaps_concatenated(self):
        segs = [_seg(0.0, 5.0, '廣東話')]
        ref  = [_seg(0.0, 2.0, 'A'), _seg(2.0, 4.0, 'B')]
        self.assertEqual(match_reference_to_segments(segs, ref), ['A，B'])

    def test_no_overlap_nearest_within_2s(self):
        # seg midpoint = 5.5; ref midpoint = 7.0; distance = 1.5 < 2.0 → match
        segs = [_seg(5.0, 6.0, '廣東話')]
        ref  = [_seg(6.5, 7.5, '參考')]
        self.assertEqual(match_reference_to_segments(segs, ref), ['參考'])

    def test_no_overlap_beyond_2s_returns_empty(self):
        segs = [_seg(0.0, 1.0, '廣東話')]
        ref  = [_seg(10.0, 11.0, '參考')]
        self.assertEqual(match_reference_to_segments(segs, ref), [''])

    def test_empty_reference_list(self):
        segs = [_seg(0.0, 1.0, '廣東話')]
        self.assertEqual(match_reference_to_segments(segs, []), [''])

    def test_multiple_segments(self):
        segs = [_seg(0.0, 2.0, 'seg1'), _seg(3.0, 5.0, 'seg2')]
        ref  = [_seg(0.0, 2.0, 'ref1'), _seg(3.0, 5.0, 'ref2')]
        self.assertEqual(match_reference_to_segments(segs, ref), ['ref1', 'ref2'])


# ---------------------------------------------------------------------------
# LLMCorrector._sanitize_reference
# ---------------------------------------------------------------------------

class TestSanitizeReference(unittest.TestCase):

    def test_empty_response_returns_primary(self):
        self.assertEqual(LLMCorrector._sanitize_reference('', '原文'), '原文')

    def test_response_starting_with_bracket_rejected(self):
        self.assertEqual(
            LLMCorrector._sanitize_reference('【廣東話ASR】blah', '原文'), '原文'
        )

    def test_blank_first_line_rejected(self):
        self.assertEqual(LLMCorrector._sanitize_reference('\n\n', '原文'), '原文')

    def test_response_too_long_rejected(self):
        primary  = '短文'
        response = '非常非常非常非常非常長嘅回應，遠超原文長度嘅一點五倍以上，唔應該接受'
        self.assertEqual(LLMCorrector._sanitize_reference(response, primary), primary)

    def test_edit_distance_too_large_conservative_rejected(self):
        primary  = '我硬係覺得唔似喎'   # 9 chars
        response = '完全唔同嘅字完全唔同'  # 9 chars, but ≥ 40% different
        ratio = _edit_distance(response, primary) / len(primary)
        self.assertGreater(ratio, 0.4)
        self.assertEqual(LLMCorrector._sanitize_reference(response, primary, semantic=False), primary)

    def test_cantonese_particle_removal_rejected(self):
        # LLM strips 嘅 and 喎 (sinicization)
        primary  = '我係唔想嘅喎'
        response = '我是不想的'
        self.assertEqual(LLMCorrector._sanitize_reference(response, primary), primary)

    def test_first_line_used_when_multiline(self):
        # primary and first line must be close enough to clear the edit-distance gate
        primary  = '修正前'         # 3 chars
        response = '修正後\n第二行廢話'  # first line differs by 1 char (前→後), ratio ≈ 0.33 < 0.40
        self.assertEqual(LLMCorrector._sanitize_reference(response, primary), '修正後')

    # --- spec examples pass the sanitizer ---

    def test_example1_passes_conservative(self):
        primary   = '我硬係覺得唔似喎，魚落天你覺得呢？'
        corrected = '我硬係覺得唔似喎，余樂天你覺得呢？'
        self.assertEqual(LLMCorrector._sanitize_reference(corrected, primary, semantic=False), corrected)

    def test_example2_passes_conservative(self):
        primary   = '不如我哋一齊去吖？仲掛得啩鐘鬥得豆'
        corrected = '不如我哋一齊去吖？種瓜得瓜種豆得豆'
        self.assertEqual(LLMCorrector._sanitize_reference(corrected, primary, semantic=False), corrected)

    def test_example3_passes_semantic(self):
        primary   = '我諗係你先諗嘅'
        corrected = '我唔諗，係你先諗嘅'
        self.assertEqual(LLMCorrector._sanitize_reference(corrected, primary, semantic=True), corrected)

    def test_example3_also_passes_conservative(self):
        # The edit ratio is 0.25, which is under the 0.40 conservative threshold too
        primary   = '我諗係你先諗嘅'
        corrected = '我唔諗，係你先諗嘅'
        self.assertEqual(LLMCorrector._sanitize_reference(corrected, primary, semantic=False), corrected)


# ---------------------------------------------------------------------------
# LLMCorrector.correct_with_reference — mocked LLM
# ---------------------------------------------------------------------------

class TestCorrectWithReferenceExamples(unittest.TestCase):
    """End-to-end tests using mocked _generate to simulate the three spec examples."""

    def test_example1_name_homophone(self):
        """魚落天 → 余樂天 (proper noun transcribed as homophone)."""
        asr      = '我硬係覺得唔似喎，魚落天你覺得呢？'
        ref      = '我真的覺得不像，余樂天你覺得怎麼嗎？'
        expected = '我硬係覺得唔似喎，余樂天你覺得呢？'

        c = _corrector(semantic=False)
        c._generate = MagicMock(return_value=expected)

        result = c.correct_with_reference([_seg(0, 3, asr)], [ref])
        self.assertEqual(result, [expected])

        user_prompt = c._generate.call_args.kwargs['user']
        self.assertIn(asr, user_prompt)
        self.assertIn(ref, user_prompt)

    def test_example2_idiom_garbled(self):
        """仲掛得啩鐘鬥得豆 → 種瓜得瓜種豆得豆 (garbled idiom recovered from reference)."""
        asr      = '不如我哋一齊去吖？仲掛得啩鐘鬥得豆'
        ref      = '不如我們一起去嗎？種瓜得瓜種豆得豆'
        expected = '不如我哋一齊去吖？種瓜得瓜種豆得豆'

        c = _corrector(semantic=False)
        c._generate = MagicMock(return_value=expected)

        result = c.correct_with_reference([_seg(0, 4, asr)], [ref])
        self.assertEqual(result, [expected])

    def test_example3_semantic_missing_negation(self):
        """我諗係你先諗嘅 → 我唔諗，係你先諗嘅 (missing 唔 and comma, semantic mode)."""
        asr      = '我諗係你先諗嘅'
        ref      = '我不想，是你才想的'
        expected = '我唔諗，係你先諗嘅'

        c = _corrector(semantic=True)
        c._generate = MagicMock(return_value=expected)

        result = c.correct_with_reference([_seg(0, 2, asr)], [ref])
        self.assertEqual(result, [expected])

    def test_empty_reference_skips_llm_call(self):
        """Segments with an empty reference string are returned unchanged without calling the LLM."""
        c = _corrector()
        c._generate = MagicMock()

        result = c.correct_with_reference([_seg(0, 2, '原文唔變')], [''])
        self.assertEqual(result, ['原文唔變'])
        c._generate.assert_not_called()

    def test_sinicization_rejected_by_sanitizer(self):
        """LLM response that removes Cantonese particles is rejected; original is kept."""
        asr = '我係唔想嘅喎'
        ref = '我是不想的'

        c = _corrector()
        c._generate = MagicMock(return_value='我是不想的')  # stripped 嘅 and 喎

        result = c.correct_with_reference([_seg(0, 2, asr)], [ref])
        self.assertEqual(result, [asr])

    def test_hallucination_rejected_by_sanitizer(self):
        """LLM response that is too dissimilar from the original is rejected."""
        asr = '你好'
        ref = '你好'

        c = _corrector()
        c._generate = MagicMock(return_value='完全唔相關嘅長篇大論係一個全新嘅句子同原文冇任何關係')

        result = c.correct_with_reference([_seg(0, 1, asr)], [ref])
        self.assertEqual(result, [asr])

    def test_multiple_segments_mixed_reference(self):
        """First segment has a reference match; second does not."""
        asr1 = '我硬係覺得唔似喎，魚落天你覺得呢？'
        asr2 = '今日天氣真係好正'
        expected1 = '我硬係覺得唔似喎，余樂天你覺得呢？'

        c = _corrector()
        c._generate = MagicMock(return_value=expected1)

        segs = [_seg(0, 3, asr1), _seg(4, 6, asr2)]
        refs = ['我真的覺得不像，余樂天你覺得怎麼嗎？', '']

        result = c.correct_with_reference(segs, refs)
        self.assertEqual(result[0], expected1)
        self.assertEqual(result[1], asr2)   # no reference → unchanged
        self.assertEqual(c._generate.call_count, 1)  # only called for seg 0

    def test_conservative_mode_uses_correct_prompt(self):
        """Conservative mode sends _PASS_REF_SYSTEM; semantic sends _PASS_REF_SEMANTIC_SYSTEM."""
        from cantocaptions_ai.pipeline.llm_correction import _PASS_REF_SYSTEM, _PASS_REF_SEMANTIC_SYSTEM

        c_cons = _corrector(semantic=False)
        c_cons._generate = MagicMock(return_value='unchanged')
        c_cons.correct_with_reference([_seg(0, 1, 'unchanged')], ['ref'])
        self.assertEqual(c_cons._generate.call_args.kwargs['system'], _PASS_REF_SYSTEM)

        c_sem = _corrector(semantic=True)
        c_sem._generate = MagicMock(return_value='unchanged')
        c_sem.correct_with_reference([_seg(0, 1, 'unchanged')], ['ref'])
        self.assertEqual(c_sem._generate.call_args.kwargs['system'], _PASS_REF_SEMANTIC_SYSTEM)


if __name__ == '__main__':
    unittest.main(verbosity=2)

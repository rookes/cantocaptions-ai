"""The pure parts of teacher-forced alternative scoring (asr_scoring.py); CPU only."""
from cantocaptions_ai.pipeline.asr_scoring import (
    Spot,
    branch_token,
    continuations,
    label_prefix,
)


def test_branch_token_finds_the_token_holding_the_spot():
    # 我 | 走 | 喇， (the particle welded to its comma) | 你
    offsets = [(0, 1), (1, 2), (2, 4), (4, 5)]
    assert branch_token(offsets, 2) == 2
    assert branch_token(offsets, 4) == 3
    assert branch_token(offsets, 9) is None


def test_continuations_share_head_and_tail():
    text = "我走先喇，你自己小心"
    spot = Spot(3, 4, ["啊", "啦", "喇"])
    # Branched at the start of the spot itself: no head.
    assert continuations(text, spot, 3) == [("喇", "喇，你自己"), ("啊", "啊，你自己"), ("啦", "啦，你自己")]
    # Branched inside a token that starts at 先: 先 leads every continuation.
    assert continuations(text, spot, 2, tail_chars=1)[1] == ("啊", "先啊，")


def test_label_prefix_only_for_models_that_write_labels():
    assert label_prefix("language Chinese<asr_text>飞龙", "Cantonese") == "language Cantonese<asr_text>"
    assert label_prefix("飛老大，<|im_end|>", "Cantonese") == ""

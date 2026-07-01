"""Debug script for Pass REF (reference-guided correction) LLM calls.

Usage — interactive session (recommended):
    uv run python -i scripts/debug_reference_pass.py

    This loads Qwen once, then drops you into a REPL. From there:
        >>> CANTONESE_ASR = "新嘅句子"
        >>> MANDARIN_REFERENCE = "新的句子"
        >>> run()   # re-runs generation with current variable values

Usage — one-shot:
    uv run python scripts/debug_reference_pass.py
"""

import re

import torch

from cantocaptions_ai.pipeline.llm_correction import load_llm
from cantocaptions_ai.utils.log_utils import setup_logging

# ---------------------------------------------------------------------------
# Edit these — reassignable at any time in the interactive REPL
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_V1 = (
    "你係粵語自動語音辨識（ ASR ）助理。你將會收到一份粵語 ASR 文字同一份國語參考字幕。請使用國語字幕作為參考，更正粵語文字嘅拼字錯誤同人名錯誤。\n"
    "嚴格規則：\n"
    "1. 唔好將粵語詞語改寫成普通話。禁止：將「嘅」改成「的」、「唔」改成「不」、"
    "「係」改成「是」、「佢」改成「他／她」、「喺」改成「在」、「哋」改成「們」。\n"
    "2. 只修正明顯係同音字錯誤嘅部分。如果唔確定，保留原文。\n"
    "3. 只有當你非常確定粵語文本入面嘅錯誤可以透過普通話文本解決嗰陣，先可以修改粵語文本。\n"
    "4. 只輸出更正後嘅粵語文本，其他任何內容都唔輸出。\n\n"
    "例如：\n【粵語】得，我仲未玩夠㗎，巴哥\n【國語參考】不行，我還沒玩夠，巴高\n【修正後嘅粵語】唔得，我仲未玩夠㗎，巴高"
)

SYSTEM_PROMPT_V2 = (
    "您是粵語字幕編輯專家。您的唯一任務是使用一段國語文字作為語意參考，修正粵語自動語音辨識（ASR）文字中的拼字錯誤（同音字詞/字元錯誤）。\n"
    "關鍵限制：您必須保留粵語口語詞彙。切勿將粵語文法/助詞（例如，後、唔、咗、喺、咁）替換為國語對應詞（例如，的、不、了、在、這）。僅修正明顯的拼字錯誤。\n"
    
)

SYSTEM_PROMPT = SYSTEM_PROMPT_V2

CANTONESE_ASR = "我硬係覺得唔似喎，魚落天你覺得呢？"
MANDARIN_REFERENCE = "我真的覺得不像，余樂天你覺得怎麼嗎？"
USER_INSTRUCTION = "【修正後嘅粵語】："

#Samples:
#CANTONESE_ASR = "得！喂，我部鋼琴啊，咁多位先生女士真係好對唔住啊，啱啱我部鋼琴有少少問題出現咗啊！得得得得得得得得得得得得得得得得得得！爸爸，我想試下就呢部鋼琴啊，得喎，我仲未玩夠㗎。你哋兩個要學下輪流玩啊。爸爸，但係佢已經玩咗好多次啦。"
#MANDARIN_REFERENCE = "嘿，鋼琴！先生女士們，非常抱歉，我的鋼琴出了點小狀況，爸爸，我想做你的鋼琴，不行，我還沒玩夠，有時候輪流玩的確不容易，爸爸，她都玩很多次了"


MODEL_ID = "Qwen/Qwen3-4B"
MODEL_DIR = None
DEVICE = "cuda"
SEMANTIC_MODE = False
MAX_NEW_TOKENS = None  # None = auto (max(1000, len(CANTONESE_ASR) * 3))

ENABLE_THINKING = True
# Prefill the think block. The model continues reasoning from this text.
# Set to None or "" to let the model think freely.
# Example: "The Cantonese 魚落天 sounds like 余樂天 in Mandarin, so..."
THINK_PREFILL: str | None = (
    "<think>\n"
    "1. Identify Cantonese homophone/ASR typos using the Mandarin subtitle strictly as a semantic anchor.\n"
    "2. Ensure no standard Cantonese words (e.g., 嘅, 唔, 啱啱, 咗) are mapped to Mandarin synonyms (的, 不, 刚刚, 了).\n"
    "3. Draft the corrected Cantonese text: "
)
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)

# Load model at import time so -i drops into a live REPL with Qwen ready.
setup_logging(level="debug")
print("Loading model…")
corrector = load_llm(
    model_id=MODEL_ID,
    model_dir=MODEL_DIR,
    device=DEVICE,
    semantic_mode=SEMANTIC_MODE,
    attn_implementation="flash_attention_2"
)
_model     = corrector._model
_tokenizer = corrector._tokenizer
print("Model ready. Call run() to generate.")


def run() -> str:
    """Run one Pass REF generation with the current module-level variables.

    Returns the sanitized output string (also printed to stdout).
    Reassign SYSTEM_PROMPT / CANTONESE_ASR / MANDARIN_REFERENCE / USER_INSTRUCTION
    in the REPL, then call run() again — no model reload needed.
    """
    max_new_tokens = MAX_NEW_TOKENS or max(1000, len(CANTONESE_ASR) * 3)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": (
            f"【粵語】{CANTONESE_ASR}\n"
            f"【國語參考】{MANDARIN_REFERENCE}\n"
            f"{USER_INSTRUCTION}"
        )},
    ]

    # ── 1. Build prompt ──────────────────────────────────────────────────────
    thinking_supported = True
    try:
        prompt_text = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING,
        )
    except TypeError:
        thinking_supported = False
        prompt_text = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    # Prefill the think block so the model continues from our seed text.
    # apply_chat_template with enable_thinking=True ends with "<think>\n";
    # appending here puts the model mid-thought before it generates.
    if ENABLE_THINKING and thinking_supported and THINK_PREFILL:
        prompt_text += THINK_PREFILL

    print("=" * 60)
    print(f"enable_thinking supported: {thinking_supported}  |  thinking: {'ON' if ENABLE_THINKING else 'OFF'}")
    prefill_preview = repr(THINK_PREFILL[:60] + "…") if THINK_PREFILL and len(THINK_PREFILL) > 60 else repr(THINK_PREFILL)
    print(f"think prefill: {prefill_preview}")
    print(f"max_new_tokens: {max_new_tokens}")
    print("=" * 60)
    print("FORMATTED PROMPT:")
    print(prompt_text)
    print("=" * 60)

    # ── 2. Generate ──────────────────────────────────────────────────────────
    inputs    = _tokenizer(prompt_text, return_tensors="pt").to(_model.device)
    input_len = inputs['input_ids'].shape[1]

    eos_ids = {_tokenizer.eos_token_id}
    im_end  = _tokenizer.convert_tokens_to_ids('<|im_end|>')
    if im_end != _tokenizer.unk_token_id:
        eos_ids.add(im_end)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
            eos_token_id=list(eos_ids),
        )

    new_ids    = output_ids[0][input_len:]
    output_len = len(new_ids)
    raw_full   = _tokenizer.decode(new_ids, skip_special_tokens=False)
    raw        = _tokenizer.decode(new_ids, skip_special_tokens=True)

    # ── 3. Diagnostics ───────────────────────────────────────────────────────
    print(f"Input tokens:  {input_len}")
    print(f"Output tokens: {output_len}  (max_new_tokens={max_new_tokens})")
    print(f"Output truncated at limit: {output_len >= max_new_tokens}")
    has_open  = '<think>' in raw_full
    has_close = '</think>' in raw_full
    print(f"<think>: {has_open}  |  </think>: {has_close}")
    print("=" * 60)
    print("RAW OUTPUT (with special tokens — shows role boundaries):")
    print(raw_full)
    print("-" * 60)
    print("RAW OUTPUT (special tokens stripped):")
    print(raw)
    print("=" * 60)

    stripped = _THINK_RE.sub('', raw).strip()
    print("AFTER think-strip:")
    print(stripped)
    print("=" * 60)

    sanitized = corrector._sanitize_reference(stripped, CANTONESE_ASR, semantic=SEMANTIC_MODE)
    print("AFTER _sanitize_reference:")
    print(sanitized)
    print("=" * 60)

    return sanitized


if __name__ == "__main__":
    run()

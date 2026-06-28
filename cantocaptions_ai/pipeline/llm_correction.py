import re
from typing import List, Optional, Tuple

import torch

from cantocaptions_ai.utils.schema import ProgressCallback, SingleSegment
from cantocaptions_ai.utils.log_utils import get_logger

logger = get_logger(__name__)

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)
_SUBSTITUTION_RE = re.compile(r'^(.+?)→(.+)$')

_PASS_A_SYSTEM = (
    "你係一個粵語字幕校對員。你嘅工作係修正ASR轉寫錯誤，特別係語氣助詞"
    "（譬如：喇/囉/喎/㗎/啩/呢/啦）嘅誤用，以及明顯嘅錯別字。\n"
    "規則：\n"
    "1. 只輸出修正後嘅粵語文字，唔好加任何解釋或標點嘅字元。\n"
    "2. 如果兩個版本相同或差距極少，保留主要ASR版本。\n"
    "3. 唔好改動內容意思，唔好翻譯成普通話。\n"
    "4. 保持繁體中文香港標準。"
)

_PASS_B_SYSTEM = (
    "你係一個粵語字幕校對員。任務：審視整份字幕，找出所有人名、地名、品牌名等專有名詞，"
    "確保全文用法一致。只修改明顯不一致嘅專有名詞，唔好改其他內容。\n"
    "輸出格式：每行一個替換指令，格式為「錯誤寫法→正確寫法」。如果唔需要修正，輸出「無需修正」。"
)


def _detect_quantization() -> Tuple[bool, str]:
    """Returns (use_4bit, reason)."""
    try:
        import bitsandbytes  # noqa: F401
        if torch.cuda.is_available():
            return True, "4-bit NF4 via bitsandbytes"
    except ImportError:
        pass
    return False, "fp16 (bitsandbytes not installed or no CUDA)"


class LLMCorrector:
    """LLM-based transcript corrector: per-segment particle fix + full-doc name normalization."""

    def __init__(self, model, tokenizer, device: str) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device

    def correct_segments(
        self,
        segments: List[SingleSegment],
        ensemble_texts: Optional[List[str]] = None,
        progress_callback: ProgressCallback = None,
    ) -> List[str]:
        """Pass A: per-segment particle and error correction."""
        corrected = []
        n = len(segments)
        for i, seg in enumerate(segments):
            primary = seg.get('text', '')
            alt = ensemble_texts[i] if ensemble_texts and i < len(ensemble_texts) else None

            prev_text = segments[i - 1].get('text', '') if i > 0 else ''
            next_text = segments[i + 1].get('text', '') if i < n - 1 else ''

            user_lines = []
            if prev_text:
                user_lines.append(f"【前文】{prev_text}")
            user_lines.append(f"【主要ASR】{primary}")
            if alt:
                user_lines.append(f"【備選ASR】{alt}")
            if next_text:
                user_lines.append(f"【後文】{next_text}")
            user_lines.append("\n請輸出修正後嘅文字：")

            response = self._generate(
                system=_PASS_A_SYSTEM,
                user="\n".join(user_lines),
                max_new_tokens=max(128, len(primary) * 3),
            )
            corrected.append(self._sanitize_pass_a(response, primary))

            if progress_callback is not None:
                progress_callback((i + 1) / n)

        return corrected

    def normalize_names(self, texts: List[str]) -> List[str]:
        """Pass B: full-document proper noun normalization."""
        if not texts:
            return texts

        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
        response = self._generate(
            system=_PASS_B_SYSTEM,
            user=f"以下係完整字幕文字（每行一段）：\n{numbered}\n\n請列出需要統一嘅專有名詞替換：",
            max_new_tokens=512,
        )

        substitutions = self._parse_substitutions(response)
        if not substitutions:
            return texts

        result = list(texts)
        for wrong, correct in substitutions:
            result = [t.replace(wrong, correct) for t in result]
        return result

    def _generate(self, system: str, user: str, max_new_tokens: int = 128) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_ids = output_ids[0][inputs['input_ids'].shape[1]:]
        raw = self._tokenizer.decode(new_ids, skip_special_tokens=True)
        return _THINK_RE.sub('', raw).strip()

    @staticmethod
    def _sanitize_pass_a(response: str, primary: str) -> str:
        if not response:
            return primary
        first_line = next((l.strip() for l in response.splitlines() if l.strip()), '')
        if not first_line or first_line.startswith('【'):
            return primary
        if len(first_line) > len(primary) * 2.5:
            return primary
        return first_line

    @staticmethod
    def _parse_substitutions(response: str) -> List[Tuple[str, str]]:
        """Parse 'X→Y' lines; only accept len(X) >= 2 to avoid over-broad replacements."""
        result = []
        for line in response.splitlines():
            line = line.strip()
            if not line or line == '無需修正':
                continue
            m = _SUBSTITUTION_RE.match(line)
            if m:
                wrong, correct = m.group(1).strip(), m.group(2).strip()
                if len(wrong) >= 2 and wrong != correct:
                    result.append((wrong, correct))
        return result


def load_llm(
    model_id: str = "Qwen/Qwen3-4B",
    model_dir: Optional[str] = None,
    device: str = "cuda",
    local_files_only: bool = False,
) -> LLMCorrector:
    """Load a causal LM for transcript correction.

    Uses 4-bit NF4 quantization via bitsandbytes when available; fp16 otherwise.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_4bit, reason = _detect_quantization()
    logger.info(f"Loading LLM ({model_id}), quantization: {reason}")

    model_path = model_dir if model_dir else model_id
    load_kwargs: dict = dict(
        device_map="auto",
        local_files_only=local_files_only,
    )

    if use_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)

    return LLMCorrector(model=model, tokenizer=tokenizer, device=device)

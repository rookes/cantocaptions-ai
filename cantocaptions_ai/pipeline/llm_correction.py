import re
from typing import List, Optional, Tuple

import torch

from cantocaptions_ai.utils.schema import ProgressCallback, SingleSegment, TranscriptionResult
from cantocaptions_ai.utils.model_utils import PipelineStage
from cantocaptions_ai.utils.debug import load_llm_correction_debug, write_llm_correction_debug
from cantocaptions_ai.utils.log_utils import get_logger

logger = get_logger(__name__)

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)
_SUBSTITUTION_RE = re.compile(r'^(.+?)→(.+)$')

_CANTONESE_PARTICLES = frozenset('嘅喎囉啦㗎呀喇吖咋咩乜')

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

_PASS_REF_SYSTEM = (
    "你係一個粵語字幕校對員。你會收到一段粵語ASR字幕同埋對應嘅普通話參考字幕。\n"
    "任務：只修正因粵語同音字而造成嘅ASR錯誤，例如人名、地名、成語入面嘅錯別字。\n"
    "嚴格規則：\n"
    "1. 只輸出修正後嘅粵語文字，唔好加任何解釋。\n"
    "2. 唔好將粵語詞語改寫成普通話。禁止：將「嘅」改成「的」、「唔」改成「不」、"
    "「係」改成「是」、「佢」改成「他／她」、「喺」改成「在」、「哋」改成「們」。\n"
    "3. 只修正明顯係同音字錯誤嘅部分。如果唔確定，保留原文。\n"
    "4. 唔好增加原文冇嘅內容，唔好改動原文意思。"
)

_PASS_REF_SEMANTIC_SYSTEM = (
    "你係一個粵語字幕校對員。你會收到一段粵語ASR字幕同埋對應嘅普通話參考字幕。\n"
    "任務：修正ASR字幕入面嘅錯誤，包括同音字錯誤同埋缺漏嘅關鍵字（例如否定詞「唔」、標點）。\n"
    "嚴格規則：\n"
    "1. 只輸出修正後嘅粵語文字，唔好加任何解釋。\n"
    "2. 唔好將粵語詞語改寫成普通話。禁止：將「嘅」改成「的」、「唔」改成「不」、"
    "「係」改成「是」、「佢」改成「他／她」、「喺」改成「在」、「哋」改成「們」。\n"
    "3. 可以修正：同音字錯誤；如果普通話參考清晰顯示缺漏嘅否定詞或關鍵標點，可以補回。\n"
    "4. 如果普通話參考同ASR意思差異過大（例如係唔同版本），保留原文。\n"
    "5. 最多只能補充少量缺漏字，唔好大幅改寫句子。"
)


def _edit_distance(a: str, b: str) -> int:
    """Character-level Levenshtein distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for ca in a:
        curr = [prev[0] + 1] + [0] * len(b)
        for j, cb in enumerate(b):
            curr[j + 1] = min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb))
        prev = curr
    return prev[-1]


def match_reference_to_segments(
    segments: List[SingleSegment],
    reference: List[SingleSegment],
) -> List[str]:
    """Return one reference string per ASR segment, matched by time overlap.

    Falls back to nearest midpoint within 2 s when no overlap exists.
    Returns empty string for segments with no usable reference match.
    """
    result = []
    for seg in segments:
        seg_start, seg_end = seg['start'], seg['end']
        overlapping = [r for r in reference if r['start'] < seg_end and r['end'] > seg_start]
        if overlapping:
            result.append('，'.join(r['text'] for r in overlapping))
        else:
            if not reference:
                result.append('')
                continue
            seg_mid = (seg_start + seg_end) / 2
            nearest = min(reference, key=lambda r: abs((r['start'] + r['end']) / 2 - seg_mid))
            ref_mid = (nearest['start'] + nearest['end']) / 2
            result.append(nearest['text'] if abs(ref_mid - seg_mid) <= 2.0 else '')
    return result


def _detect_quantization() -> Tuple[bool, str]:
    """Returns (use_4bit, reason)."""
    try:
        import bitsandbytes  # noqa: F401
        if torch.cuda.is_available():
            return True, "4-bit NF4 via bitsandbytes"
    except ImportError:
        pass
    return False, "fp16 (bitsandbytes not installed or no CUDA)"


class LLMCorrector(PipelineStage["dict", "TranscriptionResult"]):
    """LLM-based transcript corrector: per-segment particle fix + full-doc name normalization."""

    def __init__(self, model, tokenizer, device: str, semantic_mode: bool = False) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._semantic_mode = semantic_mode

    @staticmethod
    def read_debug(audio_path, debug_dir): return load_llm_correction_debug(audio_path, debug_dir)

    @staticmethod
    def write_debug(audio_path, result, debug_dir): write_llm_correction_debug(audio_path, result, debug_dir)

    @staticmethod
    def _extract(item): return {'result': item['result'], 'ensemble_texts': item.get('ensemble_texts'), 'reference_texts': item.get('reference_texts')}

    @staticmethod
    def _pack(item, result): return {**item, 'result': result}

    def process(self, input: dict, *, progress_callback: ProgressCallback = None) -> TranscriptionResult:
        """input = {'result': TranscriptionResult, 'ensemble_texts': Optional[List[str]], 'reference_texts': Optional[List[str]]}"""
        logger.info("Running LLM correction...")
        segments = input['result']['segments']
        ensemble_texts = input.get('ensemble_texts')
        reference_texts = input.get('reference_texts')

        if ensemble_texts:
            corrected = self.correct_segments(segments, ensemble_texts=ensemble_texts)
        else:
            corrected = [seg.get('text', '') for seg in segments]

        if reference_texts:
            pass_a_segs = [{**seg, 'text': corrected[i]} for i, seg in enumerate(segments)]
            corrected = self.correct_with_reference(pass_a_segs, reference_texts)

        corrected = self.normalize_names(corrected)
        new_segs = [{**seg, 'text': corrected[i]} for i, seg in enumerate(segments)]
        return {**input['result'], 'segments': new_segs}

    def correct_segments(
        self,
        segments: List[SingleSegment],
        ensemble_texts: Optional[List[str]] = None,
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

        return corrected

    def correct_with_reference(
        self,
        segments: List[SingleSegment],
        reference_texts: List[str],
    ) -> List[str]:
        """Pass REF: per-segment correction using a standard Chinese subtitle as reference."""
        system = _PASS_REF_SEMANTIC_SYSTEM if self._semantic_mode else _PASS_REF_SYSTEM
        corrected = []
        for i, seg in enumerate(segments):
            primary = seg.get('text', '')
            ref = reference_texts[i] if i < len(reference_texts) else ''

            if not ref:
                corrected.append(primary)
            else:
                user = f"【廣東話ASR】{primary}\n【普通話參考】{ref}\n請輸出修正後嘅廣東話字幕："
                response = self._generate(
                    system=system,
                    user=user,
                    max_new_tokens=max(128, len(primary) * 3),
                )
                result = self._sanitize_reference(response, primary, self._semantic_mode)
                if result != primary:
                    logger.debug(f"Reference correction [{i}]: {primary!r} → {result!r}")
                corrected.append(result)

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

        # Stop at <|im_end|> so generation halts after the first assistant turn.
        # Without this, Qwen chat models continue generating synthetic user/assistant
        # turns after the first <|im_end|>, producing garbled multi-turn output.
        eos_ids = {self._tokenizer.eos_token_id}
        im_end = self._tokenizer.convert_tokens_to_ids('<|im_end|>')
        if im_end != self._tokenizer.unk_token_id:
            eos_ids.add(im_end)

        _cuda = torch.cuda.is_available()
        _before_mb = torch.cuda.memory_allocated() / 1e6 if _cuda else None
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=list(eos_ids),
            )
        if _cuda:
            _after_mb = torch.cuda.memory_allocated() / 1e6
            _peak_mb  = torch.cuda.max_memory_allocated() / 1e6
            logger.debug(f"_generate: VRAM {_before_mb:.0f}→{_after_mb:.0f} MB, peak={_peak_mb:.0f} MB")

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
    def _sanitize_reference(response: str, primary: str, semantic: bool = False) -> str:
        if not response:
            return primary
        first_line = next((l.strip() for l in response.splitlines() if l.strip()), '')
        if not first_line or first_line.startswith('【'):
            return primary
        if len(first_line) > len(primary) * 1.5:
            return primary
        if primary and _edit_distance(first_line, primary) / len(primary) > (0.6 if semantic else 0.4):
            return primary
        orig_particles = _CANTONESE_PARTICLES & set(primary)
        if orig_particles - (set(first_line) & _CANTONESE_PARTICLES):
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
    semantic_mode: bool = False,
    attn_implementation: str = "sdpa",
) -> LLMCorrector:
    """Load a causal LM for transcript correction.

    Uses 4-bit NF4 quantization via bitsandbytes when available; fp16 otherwise.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_4bit, reason = _detect_quantization()
    logger.info(f"Loading LLM ({model_id}), quantization: {reason}, attn_implementation={attn_implementation}")

    model_path = model_dir if model_dir else model_id
    load_kwargs: dict = dict(
        device_map="auto",
        local_files_only=local_files_only,
        attn_implementation=attn_implementation,
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

    if torch.cuda.is_available():
        from cantocaptions_ai.utils.model_utils import vram_stats
        stats = vram_stats()
        if stats:
            logger.info(
                f"LLM loaded: {stats['allocated_mb']:.0f} MB allocated "
                f"({stats['free_mb']:.0f} MB free / {stats['total_mb']:.0f} MB total)"
            )

    return LLMCorrector(model=model, tokenizer=tokenizer, device=device, semantic_mode=semantic_mode)

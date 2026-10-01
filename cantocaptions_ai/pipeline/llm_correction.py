import re
from collections.abc import Mapping
from typing import Dict, List, Optional, Tuple

import torch

from cantocaptions_ai.languages.base import CorrectionPrompts
from cantocaptions_ai.utils.schema import ProgressCallback, SingleSegment, TranscriptionResult
from cantocaptions_ai.utils.model_utils import PipelineStage, ensure_hf_model_downloaded, guard_model_load
from cantocaptions_ai.utils.debug import load_llm_correction_debug, write_llm_correction_debug
from cantocaptions_ai.utils.log_utils import get_logger

logger = get_logger(__name__)

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)
_SUBSTITUTION_RE = re.compile(r'^(.+?)→(.+)$')

_CANTONESE_PARTICLES = frozenset('嘅喎囉啦㗎呀喇吖咋咩乜')

# The system prompts are the language pack's (languages/<code>/prompts.py). Correction is
# written for one language at a time: the prompts, and the particle-aware sanitising of the
# model's answers below, are Cantonese, and validate_config refuses llm_correction for a
# language whose pack has none rather than feed Cantonese instructions another language.
def correction_prompts(language: str) -> Optional[CorrectionPrompts]:
    from cantocaptions_ai.languages import get_language_pack
    return get_language_pack(language).correction_prompts


class _PromptsByLanguage(Mapping):
    """``CORRECTION_PROMPTS[code]``: a read-only view of the registered packs' prompts."""

    def _table(self) -> Dict[str, CorrectionPrompts]:
        from cantocaptions_ai.languages import LANGUAGE_PACKS
        return {code: p.correction_prompts for code, p in LANGUAGE_PACKS.items()
                if p.correction_prompts is not None}

    def __getitem__(self, code):
        return self._table()[code]

    def __iter__(self):
        return iter(self._table())

    def __len__(self):
        return len(self._table())


CORRECTION_PROMPTS: Mapping[str, CorrectionPrompts] = _PromptsByLanguage()


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
    joiner: str = '，',
    fallback_window: float = 2.0,
) -> List[str]:
    """Return one reference string per ASR segment, matched by time overlap.

    Falls back to the nearest cue by midpoint within *fallback_window* seconds when no
    overlap exists. Returns an empty string for segments with no usable match.

    Thin wrapper over ``reference_context.overlapping_reference_indices`` so ASR
    context biasing and LLM reference correction share one time-matcher; the defaults
    preserve this stage's original behaviour.
    """
    from cantocaptions_ai.pipeline.reference_context import overlapping_reference_indices

    matches = overlapping_reference_indices(segments, reference, fallback_window=fallback_window)
    return [joiner.join(reference[i]['text'] for i in idxs) for idxs in matches]


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

    debug_stage = "llm_correction"

    def __init__(self, model, tokenizer, device: str, semantic_mode: bool = False,
                 language: str = "yue") -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._semantic_mode = semantic_mode
        prompts = correction_prompts(language)
        if prompts is None:
            raise ValueError(f"no LLM correction prompts for language '{language}'")
        self._prompts = prompts

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
                system=self._prompts.particles,
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
        system = self._prompts.reference_semantic if self._semantic_mode else self._prompts.reference
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
            system=self._prompts.names,
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
    vram_checks: bool = True,
    language: str = "yue",
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

    try:
        ensure_hf_model_downloaded(model_path, cache_dir=None, local_files_only=local_files_only)
    except Exception as e:
        logger.warning("Could not download %r: %s — using cached version if available.", model_path, e)

    model = guard_model_load(
        "LLM correction",
        "consider --llm_model with fewer parameters, or disable --llm_correction",
        lambda: AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)

    if vram_checks and torch.cuda.is_available():
        from cantocaptions_ai.utils.model_utils import vram_stats
        stats = vram_stats()
        if stats:
            logger.info(
                f"LLM loaded: {stats['allocated_mb']:.0f} MB allocated "
                f"({stats['free_mb']:.0f} MB free / {stats['total_mb']:.0f} MB total)"
            )

    return LLMCorrector(model=model, tokenizer=tokenizer, device=device, semantic_mode=semantic_mode,
                        language=language)

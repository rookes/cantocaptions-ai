"""Native ASR backend: uses AutoModelForMultimodalLM, transformers' official Qwen3-ASR support.

Loaded lazily by asr.load_model() when native qwen3_asr support is detected
(transformers>=5.13.0, a base dependency).
"""
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from cantocaptions_ai.pipeline.asr import BatchedAsrStage, _normalize_language
from cantocaptions_ai.utils.audio import SAMPLE_RATE, resolve_device
from cantocaptions_ai.utils.model_utils import (
    MemoryPolicy,
    ensure_hf_model_downloaded,
    guard_model_load,
    inductor_usable,
)
from cantocaptions_ai.text_profiles import DEFAULT_NORMALIZATION, TextNormalization
from cantocaptions_ai.pipeline.model_profiles import get_model_profile
from cantocaptions_ai.utils.log_utils import get_logger

logger = get_logger(__name__)

# Generation budget per second of audio, used to size max_new_tokens for segments that
# exceed the ASR chunk budget. Reference-subtitle expansion deliberately lets a segment
# run past chunk_size rather than cut mid-dialogue, and the fixed 200-token default was
# sized for <=30s chunks: measured speech density on the Doraemon fixture is ~4.3 chars/s,
# so a 78s segment needs ~336 characters and would silently truncate. 8 leaves headroom
# over the densest observed speech.
_TOKENS_PER_SECOND = 8

# A row whose last tokens are one pattern of up to _LOOP_MAX_PERIOD tokens repeated
# _LOOP_REPEATS times (over at least _LOOP_MIN_TOKENS) is stopped there: it is looping, and
# would otherwise run to the end of its budget -- holding every other row of its batch,
# finished or not, to the same number of decode steps. On Triumph in the Skies E01 (98 chunks,
# 5 loops) this took the pass from 182 s to 114 s, every non-looping chunk's text identical.
# Five repeats, like asr_repair's text rule: real speech repeats a phrase three or four times.
_LOOP_REPEATS = 5
_LOOP_MIN_TOKENS = 24
_LOOP_MAX_PERIOD = 24


class _LoopStop:
    """A per-row ``StoppingCriteria`` for ``generate``: True for a row whose generated tail
    repeats itself (see _LOOP_REPEATS). Rows that already emitted an EOS are left alone --
    their padding repeats too. ``stopped`` collects which rows it ended."""

    def __init__(self, n_input: int, eos_ids, repeats: int = _LOOP_REPEATS,
                 min_tokens: int = _LOOP_MIN_TOKENS, max_period: int = _LOOP_MAX_PERIOD):
        self.n_input = n_input
        self.eos_ids = sorted(eos_ids)
        self.repeats, self.min_tokens, self.max_period = repeats, min_tokens, max_period
        self.stopped: Optional[torch.Tensor] = None

    def __call__(self, input_ids: torch.Tensor, scores=None, **kwargs) -> torch.Tensor:
        gen = input_ids[:, self.n_input:]
        n = gen.shape[1]
        done = torch.zeros(gen.shape[0], dtype=torch.bool, device=gen.device)
        for period in range(1, self.max_period + 1):
            span = max(period * self.repeats, self.min_tokens)
            if span > n:
                break
            tail = gen[:, n - span:]
            done |= (tail[:, period:] == tail[:, :-period]).all(dim=1)
        if self.eos_ids and n:
            done &= ~torch.isin(gen, torch.tensor(self.eos_ids, device=gen.device)).any(dim=1)
        self.stopped = done if self.stopped is None else (self.stopped | done)
        return done


class AsrText(str):
    """A decoded transcription that also says how generation ended.

    ``budget_hit``: generation never finished -- it spent every allowed token without an
    EOS, or was stopped looping (``_LoopStop``). On this model that is a repetition loop, not
    a long line (a sung "da da da" decoding to 噔 over and over, which the processor's
    repetition pass then folds back to a single 噔). ``retried``: the row was re-generated
    with a repetition penalty because of it, and ``retry_budget_hit`` says the retry looped
    as well.

    ``language`` is the label the model wrote ahead of the text (``language Italian<asr_text>``):
    None when it wrote none (the context path prefills it), ``"None"`` for what it took to be
    no speech. Without a context the requested language is only a system-prompt hint, so the
    label is the model's own call. ``raw`` is the text before the processor's repetition
    pass, kept only when that pass changed it -- i.e. when it folded a loop away.
    A plain ``str`` everywhere else, so callers that only want text see no change.
    """

    budget_hit: bool = False
    retried: bool = False
    retry_budget_hit: bool = False
    language: Optional[str] = None
    raw: Optional[str] = None

    @classmethod
    def of(cls, text: str, *, budget_hit: bool = False, retried: bool = False,
           retry_budget_hit: bool = False, language: Optional[str] = None,
           raw: Optional[str] = None) -> "AsrText":
        out = cls(text)
        out.budget_hit = budget_hit
        out.retried = retried
        out.retry_budget_hit = retry_budget_hit
        out.language = language
        out.raw = raw
        return out


_ASR_TEXT_MARKER = "<asr_text>"


def _split_label(raw: str) -> Tuple[Optional[str], Optional[str]]:
    """(language label, transcription) from one undecorated decode, with no repetition pass.

    Mirrors the processor's own parse (``language <Name><asr_text>text``) except that the
    label of an empty-audio reply is kept as ``"None"`` rather than dropped, and the text is
    left as generated. With no label at all (the context path prefills it) the label is None
    and the whole reply is the text.
    """
    raw = (raw or "").strip()
    if "assistant\n" in raw:
        raw = raw.split("assistant\n", 1)[-1]
    if _ASR_TEXT_MARKER not in raw:
        return None, raw
    prefix, text = raw.split(_ASR_TEXT_MARKER, 1)
    label = None
    for line in prefix.splitlines():
        line = line.strip()
        if line:
            label = line[len("language "):].strip() if line.lower().startswith("language ") else line
            break
    return label or None, text.strip()


# torch.compile for the decode step (--compile). generate() runs a prefill over the prompt,
# audio tower included, then one forward per new token. Compiling model.forward whole was a
# net loss (scripts/bench_asr_compile.py): the audio tower's get_audio_cu_seqlens breaks the
# graph on .item(), and with the default growing KV cache every decode step is a new shape.
# transformers can instead keep the prefill eager and compile only the decode step, whose
# shape is fixed once the cache is static (cache_implementation="static"), and run it as a
# CUDA graph (mode="reduce-overhead"). Measured on a 3080 10 GB over 324 chunks
# (scripts/bench_asr_static_cache.py): at batch 12, decode 12.6 -> 49 steps/s and the stage
# 290 s -> 95 s; GT CER 8.91% -> 8.95%, inside the 8.90-8.97% that eager spans between batch
# sizes 8, 12 and 16, which change more chunks' text than compiling does; output identical
# run to run. The static cache holds batch x _cache_len of KV from the start (~0.8 GB more
# at batch 12), yet compiled batch 8 (6.1 GB peak) ran 2.5x faster than eager batch 12 (6.4).
#
# Every batch size is its own graph (~7 s warm, ~40-60 s the first time on a machine), so
# only full batches of the configured size take the compiled path; a loop retry, a short
# last batch, a batch halved after an OOM and a repair call's odd-sized batches run as they
# always have. One cache length serves the process (sized for a chunk_size chunk), so later
# file groups and ASR repair's full batches reuse the graph rather than build another.
#
# "auto" compiles when a run has _COMPILE_MIN_BATCHES full batches: each saves ~6.6 s at
# batch 12 against ~10 s for a warm compile (a first-ever one, ~50 s, takes ~8 to repay).
_COMPILE_MIN_BATCHES = 3
_COMPILE_SETTINGS = ("auto", "on", "off")
# Set when a compiled generate fails other than by running out of memory (e.g. a broken
# Triton), so the process stops trying.
_COMPILE_FAILED = False


def _compile_config():
    from transformers import CompileConfig
    # dynamic=False: one batch size is all the compiled path ever sees, so a shape-generic
    # graph would only be slower.
    return CompileConfig(mode="reduce-overhead", dynamic=False)


def _warn_vram(inputs, batch_size: int, model, max_new_tokens: int, device, policy: MemoryPolicy,
               cache_len: Optional[int] = None) -> None:
    """Estimate one batch's peak VRAM (input tensors + generation KV-cache high-water)
    and log it against real headroom via *policy*. Callers guard on ``policy.enabled``
    so this whole function — including the estimate math below — is skipped when VRAM
    checks are off (measurably faster on the hot path; see the diagnosis findings).
    ``cache_len`` is a static cache's length, which it holds in full from the start.
    """
    dtype_bytes = next(model.parameters()).element_size()
    seq_len = inputs["input_ids"].shape[1]
    input_bytes = sum(t.numel() * t.element_size() for t in inputs.values() if isinstance(t, torch.Tensor))
    try:
        text_cfg = model.config.text_config
        kv_bytes = (
            batch_size * (cache_len or seq_len + max_new_tokens)
            * text_cfg.num_hidden_layers * 2
            * text_cfg.num_key_value_heads * text_cfg.head_dim
            * dtype_bytes
        )
    except AttributeError:
        kv_bytes = 0
    estimated = input_bytes + kv_bytes
    policy.warn(
        f"ASR batch (batch_size={batch_size}, seq_len={seq_len})",
        device,
        estimated / 1e6,
        "consider reducing --batch_size or using --asr_compute_type int8",
    )


class QwenPipelineNative(BatchedAsrStage):
    """Native backend: uses AutoModelForMultimodalLM (Qwen3ASRForConditionalGeneration).

    Loads a Qwen3-ASR checkpoint (repo id / local path resolved from the model profile)
    via transformers' official qwen3_asr support, and applies the language pack's post-ASR
    ``normalization`` to each decoded segment. Batching and checkpointing are
    BatchedAsrStage's; this class supplies the Qwen prompt, generation and decode.
    """

    backend_label = "native"
    loop_stop = True

    def __init__(
        self,
        model,
        processor,
        device: Union[int, str, "torch.device"],
        language: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_new_tokens: int = 256,
        print_progress: bool = False,
        verbose: bool = False,
        vram_checks: bool = True,
        normalization: TextNormalization = DEFAULT_NORMALIZATION,
        loop_retry_penalty: float = 1.1,
        loop_stop: bool = True,
        compile: str = "off",
        chunk_size: Optional[float] = None,
    ):
        super().__init__(
            device=device, language=language, batch_size=batch_size,
            print_progress=print_progress, verbose=verbose, vram_checks=vram_checks,
            normalization=normalization,
        )
        if compile not in _COMPILE_SETTINGS:
            raise ValueError(f"unknown ASR compile setting {compile!r} (expected auto, on or off)")
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.loop_retry_penalty = loop_retry_penalty
        self.loop_stop = loop_stop
        self._compile = compile
        # The longest chunk a run is expected to hold, which sizes the static KV cache.
        self._chunk_size = chunk_size
        # The batch size whose decode step runs compiled (None: none does), the static
        # cache length every compiled call shares, and whether that graph has run yet.
        self._compiled_rows: Optional[int] = None
        self._cache_len: Optional[int] = None
        self._compiled_ready = False
        eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        self._eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}

    def _backend_language(self, language: Optional[str]) -> str:
        # Qwen3-ASR names languages in English ("Cantonese"), and that is also what its
        # transcription checkpoints have always recorded.
        return _normalize_language(language)

    def _build_context_prompts(self, contexts: List[Optional[str]], language: str) -> List[str]:
        """Render the SDK's prompt shape for a batch carrying context biasing text.

        ``apply_transcription_request`` cannot be reused here. It puts the *language
        name* in the system slot (processing_qwen3_asr.py:495-540), while the chat
        template concatenates every system text item with no separator -- so appending
        context to it would emit "<context>Cantonese" as one run-on string.

        The official SDK (qwen_asr/inference/qwen3_asr.py:448-465) and vLLM both do it
        the other way round: system carries the context *alone*, and the language is
        forced by prefilling the assistant turn with "language <Lang><asr_text>".
        Because that prefill sits in the input, generated tokens are the bare
        transcription -- ``decode(return_format="transcription_only")`` still handles
        it (no marker => the whole string is the transcription) and still applies the
        repetition-collapsing pass.
        """
        conversations = [
            [
                {"role": "system", "content": ctx or ""},
                {"role": "user", "content": [{"type": "audio", "audio": ""}]},
            ]
            for ctx in contexts
        ]
        prompts = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )
        return [f"{p}language {language}<asr_text>" for p in prompts]

    def _infer_batch(
        self,
        wavs: List,
        language: str,
        contexts: Optional[List[Optional[str]]] = None,
    ) -> List[str]:
        """Run one batch of audio arrays through the model, returning parsed texts.

        With no context supplied this takes the plain ``apply_transcription_request``
        path unchanged, so runs without --asr_context are bit-identical to before.

        A row that spends the whole token budget without an EOS is a repetition loop.
        It is re-generated once with ``repetition_penalty`` = ``loop_retry_penalty``
        (0 disables the retry). On the Bluey fixture's sung opening the loop decodes to a
        lone 噔, and the retry recovers the whole chunk. Every text comes back as an
        :class:`AsrText` carrying how generation ended and the language label it wrote.

        Raises RuntimeError on CUDA OOM (caught and retried at a smaller batch size by
        BatchExecutor); no shared state is mutated before the model call.
        """
        return self._transcribe_batch(wavs, language, contexts)

    def transcribe(self, wavs: List, language: Optional[str],
                   contexts: Optional[List[Optional[str]]] = None, *,
                   retry_penalty: Optional[float] = None,
                   repetition_penalty: Optional[float] = None,
                   token_floor: Optional[int] = None,
                   progress_callback=None) -> List[AsrText]:
        """Transcribe audio arrays of the caller's choosing, batched like the stage itself.

        For ASR repair, which re-runs chunks and pieces of chunks: ``language`` None lets the
        model identify the language itself (no system hint; incompatible with contexts),
        ``repetition_penalty`` applies to the first generation, and ``retry_penalty``
        overrides ``loop_retry_penalty`` for the loop retry (0 disables it), and
        ``token_floor`` replaces the generation budget's floor (``max_new_tokens``) -- a
        few seconds of audio needs far fewer than a whole chunk, and a loop spends all it
        is given. Batches run longest-first and halve on CUDA OOM.
        """
        from cantocaptions_ai.utils.model_utils import BatchExecutor

        self._plan_batches(len(wavs))
        out: List[Optional[AsrText]] = [None] * len(wavs)

        def infer_fn(batch):
            texts = self._transcribe_batch(
                [wavs[i] for i in batch], language,
                [contexts[i] for i in batch] if contexts is not None else None,
                retry_penalty=retry_penalty, repetition_penalty=repetition_penalty,
                token_floor=token_floor)
            for i, text in zip(batch, texts):
                out[i] = text

        BatchExecutor(self._batch_size, order_key=lambda i: len(wavs[i])).run(
            list(range(len(wavs))), infer_fn, reporter=progress_callback)
        return out

    @property
    def compiled(self) -> bool:
        """Whether this model's full batches run the compiled decode step."""
        return self._compiled_rows is not None

    def _plan_batches(self, n_jobs: int) -> None:
        """Before *n_jobs* chunks are batched: whether full batches take the compiled decode
        step (see _COMPILE_MIN_BATCHES). Once this model has compiled it, it keeps it."""
        if self._compiled_rows is not None or self._compile == "off" or not n_jobs:
            return
        reason = self._compile_blocker()
        full = n_jobs // self._batch_size if self._batch_size else 0
        if reason is None and self._compile == "auto" and full < _COMPILE_MIN_BATCHES:
            reason = (f"auto: {n_jobs} chunks make {full} full batch(es), under the "
                      f"{_COMPILE_MIN_BATCHES} that repay compiling")
        if reason is not None:
            # Only news when compiling was asked for; otherwise it is the expected path.
            (logger.info if self._compile == "on" else logger.debug)(
                "ASR: running the decode step uncompiled (%s)", reason)
            return
        self._compiled_rows = self._batch_size
        logger.info(
            "Compiling the ASR decode step for batches of %d. The first run on this machine "
            "takes about a minute; later runs reuse the compile cache and take about 10 s. "
            "Pass --compile off to skip it.", self._batch_size)

    def _compile_blocker(self) -> Optional[str]:
        """Why the decode step can't be compiled here, or None."""
        if self.device.type != "cuda":
            return "compiling needs a CUDA device"
        if not self._batch_size:
            return "compiling needs a fixed --batch_size"
        if _COMPILE_FAILED:
            return "an earlier compile in this process failed"
        if not inductor_usable(self.device):
            return "no working Triton; on Windows, install the `compile` extra"
        return None

    def _static_cache_len(self, needed: int, language: Optional[str]) -> int:
        """The static KV cache length for a compiled call needing *needed* positions.

        Sized once, for the longest of this call and a chunk_size chunk, so every later
        call fits the same cache and reuses its graph. A call that needs more (a context
        prompt, a segment past chunk_size) grows it, costing one new cache and graph.
        """
        if self._cache_len is None:
            hint = 0
            if self._chunk_size:
                clip = np.zeros(int(self._chunk_size * SAMPLE_RATE), dtype=np.float32)
                prompt = self.processor.apply_transcription_request(audio=[clip], language=language)
                hint = int(prompt["input_ids"].shape[1]) + max(
                    self.max_new_tokens, int(self._chunk_size * _TOKENS_PER_SECOND))
            self._cache_len = max(needed, hint)
        elif needed > self._cache_len:
            logger.debug("ASR: a call needs %d cache positions, over the %d sized; growing it",
                         needed, self._cache_len)
            self._cache_len = needed
        return self._cache_len

    def _generate_compiled(self, generate, static: dict):
        """``generate(static)``, under a status while its first call compiles the graph.

        Either way a failure re-runs the batch uncompiled, at the same size, and the model
        stays uncompiled. Out of memory, that is because the static cache holds the whole
        KV cache from the start while an uncompiled batch only grows into it, so a batch
        size that fits uncompiled is kept rather than halved; if it doesn't fit either, the
        error reaches the caller, which halves. Any other failure is a broken compile, and
        the rest of the process doesn't try again.
        """
        global _COMPILE_FAILED
        from cantocaptions_ai.utils.model_utils import _stage_status

        started = time.perf_counter()
        try:
            if self._compiled_ready:
                result = generate(static)
            else:
                with _stage_status("compiling the decode step"):
                    result = generate(static)
        except Exception as e:
            self._drop_compiled()
            if "out of memory" in str(e).lower():
                logger.info("ASR: out of memory with the static KV cache at batches of %d; "
                            "running the decode step uncompiled from here", self._batch_size)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return generate({})
            logger.warning("Compiling the ASR decode step failed; running it uncompiled for "
                           "the rest of this process. %s: %s", type(e).__name__, e)
            _COMPILE_FAILED = True
            return generate({})
        if not self._compiled_ready:
            self._compiled_ready = True
            logger.info("Compiled the ASR decode step (its first batch took %.0f s)",
                        time.perf_counter() - started)
        return result

    def _drop_compiled(self) -> None:
        """Run uncompiled from here on, and free the static cache transformers keeps."""
        self._compiled_rows = None
        # The static cache, and the compiled call holding its CUDA graphs' memory pool.
        for attr in ("_cache", "_compiled_call"):
            if getattr(self.model, attr, None) is not None:
                delattr(self.model, attr)

    def _transcribe_batch(self, wavs: List, language: Optional[str],
                          contexts: Optional[List[Optional[str]]] = None, *,
                          retry_penalty: Optional[float] = None,
                          repetition_penalty: Optional[float] = None,
                          token_floor: Optional[int] = None) -> List[AsrText]:
        """One batch through the model, retrying looped rows: :meth:`_infer_batch`'s body."""
        retry_penalty = self.loop_retry_penalty if retry_penalty is None else retry_penalty
        extra = {} if repetition_penalty is None else {"repetition_penalty": repetition_penalty}
        if token_floor is not None:
            extra["token_floor"] = token_floor
        rows = self._generate(wavs, language, contexts, **extra)
        out = [AsrText.of(t, budget_hit=h, language=lab, raw=r) for t, h, lab, r in rows]
        looped = [i for i, (_, h, _, _) in enumerate(rows) if h]
        if looped and retry_penalty:
            retry_rows = self._generate(
                [wavs[i] for i in looped], language,
                [contexts[i] for i in looped] if contexts is not None else None,
                repetition_penalty=retry_penalty,
                **({"token_floor": token_floor} if token_floor is not None else {}),
            )
            for i, (text, hit, label, raw) in zip(looped, retry_rows):
                logger.debug(
                    "ASR: a %.1f s chunk spent the whole token budget (repetition loop: %r); "
                    "re-generated with repetition_penalty=%s%s",
                    len(wavs[i]) / SAMPLE_RATE, str(out[i])[:20], retry_penalty,
                    " -- it still looped" if hit else "")
                out[i] = AsrText.of(text, budget_hit=True, retried=True, retry_budget_hit=hit,
                                    language=label, raw=raw)
        return out

    def _generate(self, wavs: List, language: Optional[str],
                  contexts: Optional[List[Optional[str]]] = None,
                  token_floor: Optional[int] = None,
                  **generate_kwargs) -> List[Tuple[str, bool, Optional[str], Optional[str]]]:
        """One ``generate`` over a batch: per row (text, budget_hit, language label, raw).

        ``raw`` is the text before the processor's repetition pass, or None when that pass
        changed nothing (see :class:`AsrText`).
        """
        if contexts is not None and any(contexts):
            if language is None:
                raise ValueError("a context prompt needs a language; auto-detection has none")
            prompts = self._build_context_prompts(contexts, language)
            for prompt in prompts:
                logger.debug("[PROMPT] %s", prompt)
            # padding_side="left" comes from Qwen3ASRProcessorKwargs._defaults, which is
            # what makes batched generation correct on this path.
            inputs = self.processor(text=prompts, audio=wavs, return_tensors="pt", padding=True)
        else:
            inputs = self.processor.apply_transcription_request(audio=wavs, language=language)
        inputs = inputs.to(self.model.device, self.model.dtype)

        # Segments longer than the chunk budget need a proportionally larger generation
        # budget or their transcription is cut off mid-sentence. Raising the cap costs
        # nothing when generation stops at EOS -- but note it also lets a context
        # recitation loop run longer, which is a separate defect.
        longest_s = max(len(w) for w in wavs) / SAMPLE_RATE
        floor = self.max_new_tokens if token_floor is None else token_floor
        max_new = max(floor, int(longest_s * _TOKENS_PER_SECOND))
        if max_new > self.max_new_tokens:
            logger.debug("[BUDGET] longest=%.1fs -> max_new_tokens=%d (default %d)",
                         longest_s, max_new, self.max_new_tokens)

        n_input = inputs["input_ids"].shape[1]
        static = {}
        if self._compiled_rows == len(wavs):
            static = dict(cache_implementation="static",
                          max_cache_len=self._static_cache_len(n_input + max_new, language),
                          compile_config=_compile_config())

        if self.device.type == "cuda" and self.policy.enabled:
            _warn_vram(inputs, len(wavs), self.model, max_new, self.device, self.policy,
                       cache_len=static.get("max_cache_len"))

        def generate(extra):
            # A fresh loop stop per attempt: it records which rows it ended.
            stop = _LoopStop(n_input, self._eos_ids) if self.loop_stop else None
            kwargs = {**generate_kwargs, **extra}
            if stop is not None:
                from transformers import StoppingCriteriaList
                kwargs["stopping_criteria"] = StoppingCriteriaList([stop])
            with torch.inference_mode():
                return self.model.generate(
                    **inputs, max_new_tokens=max_new, do_sample=False, **kwargs), stop

        if not static:
            generated, stop = generate({})
        else:
            generated, stop = self._generate_compiled(generate, static)

        # generate() returns a plain tensor by default; handle both for safety
        sequences = generated if isinstance(generated, torch.Tensor) else generated.sequences
        new_tokens = sequences[:, n_input:]
        logger.debug("[TOKENS] batch=%d, new_tokens=%d, max=%d", len(wavs), new_tokens.shape[1], self.max_new_tokens)
        # A row that finished early carries its EOS (then pad, itself an EOS id on
        # Qwen3-ASR) somewhere in it; a row that never did ran the full budget. A row the
        # loop stop ended is padded the same way, so it says so itself.
        stopped = (stop.stopped.tolist() if stop is not None and stop.stopped is not None
                   else [False] * len(wavs))
        hits = [bool(was_stopped) or (new_tokens.shape[1] >= max_new
                                      and not any(int(t) in self._eos_ids for t in row))
                for row, was_stopped in zip(new_tokens, stopped)]

        # Qwen3ASRProcessor.decode(return_format="transcription_only") strips the
        # "language <LANG><asr_text>" prefix and applies the same repetition-collapsing
        # dehallucination pass the original qwen_asr package used — no extra
        # post-processing needed here.
        texts = self.processor.decode(
            new_tokens,
            return_format="transcription_only",
            clean_up_tokenization_spaces=False,
        )
        # The same tokens undecorated: the language label the model wrote, and the text
        # before the repetition pass -- which folds a loop away without a trace.
        raws = self.processor.decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        del inputs, generated

        rows = []
        for text, hit, raw in zip(texts, hits, raws):
            logger.debug("[RAW] %s", raw)
            label, unfolded = _split_label(raw) if isinstance(raw, str) else (None, None)
            rows.append((text, hit, label, unfolded if unfolded and unfolded != text else None))
        return rows


def load_model_native(
    model_name: str,
    device: str,
    device_index: int = 0,
    compute_type: str = "default",
    attn_implementation: str = "sdpa",
    language: Optional[str] = "yue",
    model=None,
    download_root: Optional[str] = None,
    local_files_only: bool = False,
    batch_size: Optional[int] = None,
    compile: str = "off",
    chunk_size: Optional[float] = None,
    print_progress: bool = False,
    verbose: bool = False,
    vram_checks: bool = True,
    vram_headroom_mb: int = 512,
    processor=None,
    normalization=None,
    loop_retry_penalty: float = 1.1,
    loop_stop: bool = True,
) -> QwenPipelineNative:
    """Build the native ASR pipeline, loading whatever the caller did not supply.

    ``model`` and ``processor`` may be passed in pre-built. That is how a caller
    with its own checkpoint layout reuses this pipeline's batching, generation
    budget and decode unchanged: a LoRA adapter directory, for instance, holds
    neither base weights nor processor files, so the caller merges the adapter
    and loads the processor from the base itself. When both are supplied nothing
    is downloaded.

    ``compile`` ("auto", "on" or "off") compiles the decode step for full batches on CUDA
    (see _COMPILE_MIN_BATCHES); ``chunk_size``, the longest chunk expected in seconds,
    sizes the static KV cache that needs.
    """
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    from cantocaptions_ai.pipeline.asr import _resolve_normalization

    model_id = get_model_profile(model_name, language).hf_id
    normalization = _resolve_normalization(model_name, language, normalization)

    if model is None or processor is None:
        try:
            ensure_hf_model_downloaded(model_id, cache_dir=download_root, local_files_only=local_files_only)
        except Exception as e:
            logger.warning("Could not download %r: %s — using cached version if available.", model_id, e)

    if compute_type == "default":
        compute_type = "float16" if device == "cuda" else "float32"
        logger.debug("Compute type defaulting to %s for device %s", compute_type, device)

    device_map = resolve_device(device, device_index)
    pipeline_device = device_index if device == "cuda" else device

    logger.info("ASR model: %s (native backend, %s attention)", model_id, attn_implementation)

    hf_model = model
    if hf_model is None:
        hf_model = guard_model_load(
            "ASR",
            "consider --asr_compute_type int8 or a lower --batch_size",
            lambda: AutoModelForMultimodalLM.from_pretrained(
                model_id,
                dtype=compute_type,
                device_map=device_map,
                attn_implementation=attn_implementation,
                local_files_only=local_files_only,
                cache_dir=download_root,
            ).eval(),
        )

    if processor is None:
        processor = AutoProcessor.from_pretrained(
            model_id,
            local_files_only=local_files_only,
            cache_dir=download_root,
        )

    # Cap the allocator now that the weights are resident, so the reading reflects
    # true post-load free VRAM. Keeps generation's growing KV-cache from silently
    # paging into host RAM; a near-OOM instead raises and the BatchExecutor halves.
    if device == "cuda":
        MemoryPolicy(vram_checks, vram_headroom_mb).cap_after_load(device_index)

    return QwenPipelineNative(
        model=hf_model,
        processor=processor,
        device=pipeline_device,
        language=language,
        batch_size=batch_size,
        max_new_tokens=200,
        print_progress=print_progress,
        verbose=verbose,
        vram_checks=vram_checks,
        normalization=normalization,
        loop_retry_penalty=loop_retry_penalty,
        loop_stop=loop_stop,
        compile=compile,
        chunk_size=chunk_size,
    )

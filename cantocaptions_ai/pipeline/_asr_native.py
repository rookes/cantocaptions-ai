"""Native ASR backend: uses AutoModelForMultimodalLM, transformers' official Qwen3-ASR support.

Loaded lazily by asr.load_model() when native qwen3_asr support is detected
(transformers>=5.13.0, a base dependency).
"""
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from cantocaptions_ai.pipeline.asr import BatchedAsrStage, _normalize_language
from cantocaptions_ai.utils.audio import SAMPLE_RATE, resolve_device
from cantocaptions_ai.utils.model_utils import (
    MemoryPolicy,
    ensure_hf_model_downloaded,
    guard_model_load,
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


def _compile_and_warmup(model, processor, language: str, batch_size: Optional[int]) -> None:
    """Compile model.forward and run a short warmup so the first real batch isn't the one paying the compile cost.

    Opt-in only (--compile) — benchmarked in scripts/bench_asr_compile.py against this
    workload's actual shape variance (VAD segments are essentially unique-length) and found
    to be a net loss in every configuration tried:
      - Default (graph break allowed): get_audio_cu_seqlens() in the audio tower does
        `.max().item()` + a Python loop over tensor-derived lengths, which dynamo can't trace.
        Each new (batch_size, audio_length) shape burns up to `recompile_limit` (8) full
        compile attempts (~20-30s each) before giving up and falling back to eager for that
        code location — tens of seconds wasted per distinct shape, for at most a ~2x speedup
        on the rare batch that repeats an exact prior shape before the budget trips.
      - dynamic=True: made the first-compile cost dramatically worse (137-408s), not better.
      - Padding every segment to a fixed shape (chunk_size worth of samples) does eliminate
        the recompiles and gives a real, stable steady-state speedup once compiled — but only
        pays off once the workload is already VRAM-healthy; at a starved batch_size, eager
        itself was 10-50x slower (near-OOM allocator overhead unrelated to compile), which
        made compile look better than it is. At a batch_size with real headroom, eager alone
        is already fast and stable, and compile's edge shrinks to ~25-40% against a ~30s+
        per-process tax.
      - That per-process tax cannot be cached away: neither torch.compiler.save_cache_artifacts/
        load_cache_artifacts (torch's "Mega-Cache", meant for cross-machine cache portability)
        nor the default on-disk Inductor cache avoid it — the dominant cost is TorchDynamo's
        own bytecode tracing/graph-break handling for this model, which happens once per fresh
        Python process regardless of what's cached on disk, not the kernel compilation itself.

    Uses a small batch of silent audio at roughly the configured batch size — torch.compile
    specializes to the shapes it first sees, so warming up near the real batch size avoids an
    extra recompilation on the first genuine inference call.
    """
    model.forward = torch.compile(model.forward)
    warmup_batch = max(1, min(batch_size or 4, 8))
    silence = [np.zeros(3 * 16000, dtype=np.float32) for _ in range(warmup_batch)]
    inputs = processor.apply_transcription_request(audio=silence, language=language)
    inputs = inputs.to(model.device, model.dtype)
    with torch.inference_mode():
        for _ in range(3):
            model.generate(**inputs, max_new_tokens=8, do_sample=False)


def _warn_vram(inputs, batch_size: int, model, max_new_tokens: int, device, policy: MemoryPolicy) -> None:
    """Estimate one batch's peak VRAM (input tensors + generation KV-cache high-water)
    and log it against real headroom via *policy*. Callers guard on ``policy.enabled``
    so this whole function — including the estimate math below — is skipped when VRAM
    checks are off (measurably faster on the hot path; see the diagnosis findings).
    """
    dtype_bytes = next(model.parameters()).element_size()
    seq_len = inputs["input_ids"].shape[1]
    input_bytes = sum(t.numel() * t.element_size() for t in inputs.values() if isinstance(t, torch.Tensor))
    try:
        text_cfg = model.config.text_config
        kv_bytes = (
            batch_size * (seq_len + max_new_tokens)
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
    ):
        super().__init__(
            device=device, language=language, batch_size=batch_size,
            print_progress=print_progress, verbose=verbose, vram_checks=vram_checks,
            normalization=normalization,
        )
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.loop_retry_penalty = loop_retry_penalty
        self.loop_stop = loop_stop
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

        if self.device.type == "cuda" and self.policy.enabled:
            _warn_vram(inputs, len(wavs), self.model, max_new, self.device, self.policy)

        n_input = inputs["input_ids"].shape[1]
        stop = _LoopStop(n_input, self._eos_ids) if self.loop_stop else None
        if stop is not None:
            from transformers import StoppingCriteriaList
            generate_kwargs["stopping_criteria"] = StoppingCriteriaList([stop])
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False, **generate_kwargs
            )

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
    compile_enabled: bool = False,
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

    if device == "cuda" and compile_enabled:
        try:
            _compile_and_warmup(hf_model, processor, _normalize_language(language), batch_size)
            logger.info("torch.compile enabled for ASR model")
        except Exception as e:
            logger.warning(
                "torch.compile failed (%s); falling back to eager mode. "
                "Install the compile extra for triton support.",
                e,
            )

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
    )

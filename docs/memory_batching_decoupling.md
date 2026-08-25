# Memory & batching decoupling — analysis and strategy

Captures (A) why "each additional batch bloats GPU memory" across the batched
stages, and (B) three strategies for pulling VRAM/batching concerns out of the
conceptual model functions (`transcribe`/`align`/vocal-isolation `infer`), with a
recommendation and an incremental path.

> **Status (Strategy A landed).** The bench (`scripts/bench_asr_native.py`)
> confirmed the levers: `--sort desc` ≈ +10% over VAD order (`asc` far worse —
> CUDA-allocator effect), larger batch (20+) much faster, and `--no-warn-vram`
> a small but real gain. Acted on:
> - **`BatchExecutor`** (`model_utils.py`) now owns ordering + fixed-size
>   batching + OOM-adaptive halving + optional flush cadence; `run_adaptive_batches`
>   is a thin back-compat shim over it.
> - **`MemoryPolicy`** (`model_utils.py`) bundles `vram_checks`/`headroom_mb`;
>   `enabled` gates the estimate math so `_warn_vram`/`_warn_alignment_vram` (and
>   their pre-guard `next(model.parameters())`/`sum(numel)` work) are skipped
>   entirely when checks are off — the `--no-warn-vram` win, now the default path.
> - **ASR** processes **longest-first** (`_asr_native.py` `run()`/`process()`),
>   the +10% win; **alignment** keeps its longest-first order via the executor's
>   `order_key`; **vocal isolation** uses the executor (no order — fixed chunks).
>
> Still open (deferred): fully collapsing the `vram_checks`/`vram_headroom_mb`
> parameter threading (config→`__main__`→`load_*`) into an injected `MemoryPolicy`
> everywhere; a `BatchExecutor` output-sink to make the off-GPU boundary a single
> owned step; and the `log_vram_delta` per-batch instrumentation helper. The
> +3.5 GB higher peak in the full program vs. the bench points at cross-stage
> allocator residue — `BatchExecutor.flush_every` is the available knob (default
> off to preserve the throughput win), but the real fix is stage-handoff hygiene
> (`model_scope`), not per-batch flushing.

Related: `scripts/bench_asr_native.py` (the diagnostic harness),
`tests/test_batch_executor.py`, `tests/test_asr_batching.py`, and the shared
helpers in `cantocaptions_ai/utils/model_utils.py`.

---

## Part A — "each batch bloats GPU memory," per stage

The recurring symptom (free VRAM declines batch after batch until it hits the
floor, then the run slows down — on Windows/WDDM without ever crashing) is, in
every current stage, the **CUDA caching allocator's `reserved` pool tracking a
rising high-water mark** — not a Python-level tensor leak. `reserved` only grows
(the allocator caches freed blocks for reuse and rarely returns them to the
driver); when a later batch needs a *larger* contiguous block than any cached
one, it reserves more. So the shape/size of the largest-yet allocation, and
*when* it first occurs, drives the curve. Three stages, three profiles:

### ASR (`_asr_native.py`)
- **Live tensors are bounded.** `_infer_batch` decodes to **Python strings**
  (`processor.decode(...)`), appends those to host-side buffers, and does `del
  inputs, generated` before returning. No GPU tensor outlives the batch.
- **What grows is `reserved`,** tracking autoregressive generation's KV-cache
  high-water mark, ~ `batch_size × (seq_len + new_tokens) × layers × kv_heads ×
  head_dim × dtype_bytes` (this is exactly what `_warn_vram` estimates). Two
  compounding factors:
  - `model.generate` runs a **batch until *all* rows hit EOS/`max_new_tokens`**,
    so one long clip stretches the whole batch's KV-cache and compute.
  - `run()` builds jobs in **VAD order** (`_asr_native.py:148`), never by length.
    The first batch that happens to contain the longest clip forces the
    largest-yet reservation, and if that lands late, `reserved` **ratchets up
    across the run** instead of being paid once at the start.
- **Lever (applied):** processing **longest-first** — as `alignment.py` already
  does — front-loads the peak reservation into batch 1 and flattens the curve
  thereafter. `bench_asr_native.py --sort desc` confirmed ≈ +10% over VAD order,
  so `run()`/`process()` now sort via `BatchExecutor(order_key=…)`. Not a leak; a
  scheduling effect.

### Alignment (`alignment.py`)
- **Already clean.** Emissions are copied to host immediately per segment —
  `results[i] = (emissions[row, :real_len, :].cpu().detach(), frame_rate)` — and
  the GPU emissions tensor is closure-local to the batch, freed when the batch
  returns.
- **Already longest-first:** jobs are `sorted(..., key=len(audio), reverse=True)`
  (~`alignment.py:429`), so the peak padded forward pass happens on batch 1 and
  `reserved` is flat afterwards. This is the template the ASR lever would copy.

### Vocal isolation (`vocal_isolation.py`)
- **GPU side never grows:** Mel-Band RoFormer runs **fixed-size chunks** (≈8 s @
  44.1 kHz), and each chunk's output is moved off-GPU immediately
  (`out.float().cpu().numpy()`); overlap-add accumulation is host-side numpy,
  freed per segment. GPU shapes are constant ⇒ `reserved` is flat.
- The growth fixed here recently was **host RAM**, not VRAM: a large batch
  pre-allocated every file's segments at once. The fix was **per-file windowing**
  (`_iter_windows`, `_MAX_SEGMENTS_PER_WINDOW`), bounding how much audio is
  resident at once. Called out so the "bloat" framing isn't misapplied to this
  stage — its knob is host memory, not the allocator.

### Reusable instrumentation (proposed helper, not yet added)
A `log_vram_delta(label, device)` context manager next to `vram_stats`
(`model_utils.py:230`), built on `vram_stats`, logging `reserved`/`allocated`
before vs after the block at debug level. Dropped into `run_adaptive_batches`'s
per-batch loop it instruments **all three stages uniformly** and makes a
monotonic `reserved` climb visible in normal runs — the same signal
`bench_asr_native.py` prints per batch (`d_reserved`), but in production behind
`--verbose`/debug gating. This is the natural first artifact of any of the
strategies below, since all three centralize the per-batch loop.

---

## Part B — decoupling batching/VRAM from the model functions

### The coupling (before this work)
VRAM/batching logic was threaded through the conceptual model code:
- `_warn_vram(...)` was called **inline inside** `_asr_native._infer_batch`, and
  its pre-guard math (`next(model.parameters())`, `sum(t.numel()...)`) ran on CUDA
  **even when `vram_checks=False`** — only the `check_vram_headroom` round-trip was
  gated. *(Fixed: the call is now guarded on `policy.enabled`, skipping the math.)*
- `_warn_alignment_vram(...)` lived inside alignment's `infer_fn`. *(Now guarded on
  `vram_checks` at the call site.)*
- `cap_cuda_memory(...)` was called inline in `load_model_native`. *(Now
  `MemoryPolicy(...).cap_after_load(...)`.)*
- `check_vram_headroom(...)` is called in every `load_*` (already internally gated).
- `vram_checks` / `vram_headroom_mb` are **parameter-threaded**
  config → `__main__` → `load_*` → `infer_fn`, touching every stage's signature.
  *(Still open — the remaining B step; see Status.)*
- `run_adaptive_batches` owned batching + OOM-halving but nothing else — flush
  cadence, ordering, and the off-GPU boundary were re-implemented per stage.
  *(Fixed: `BatchExecutor` owns ordering + batching + OOM-halving + flush;
  `run_adaptive_batches` is a shim.)*

The goal: model functions express *what to compute*; a separate layer owns
*batch scheduling, VRAM policy, and where live tensors live*.

### Strategy A — a `BatchExecutor` / runner (recommended target)
One object owns the whole batched hot loop:
- **Batching + OOM-adaptive halving** — absorbs `run_adaptive_batches`.
- **A pre-batch estimator hook** — absorbs `_warn_vram` / `_warn_alignment_vram`
  (the estimate becomes a callback the stage supplies, called only when policy
  says so — killing the "runs even when disabled" wart).
- **Ordering policy** — longest-first vs VAD order becomes one uniform knob; ASR
  inherits the alignment sort for free if the bench confirms it.
- **Flush cadence** — `empty_cache` timing in one place.
- **An output sink** — the single, explicit off-GPU boundary (`.cpu()` / decode
  to strings), so "where live tensors live" is answered in exactly one spot.

Model functions shrink to a pure `infer_fn(batch) -> outputs` plus a sink.
Best satisfies the "easily manage where we store live tensors" goal and lets the
ASR sort land as policy rather than a bespoke edit. Cost: it **replaces**
`run_adaptive_batches` and **subsumes the batching half of** `PipelineStage`
(the debug-cache half survives) — medium but mechanical blast radius across ASR,
alignment, and vocal isolation.

### Strategy B — an injected `MemoryPolicy` (recommended first step)
One config-built object exposing e.g. `warn(estimate)`, `cap_after_load()`,
`enabled`. It **collapses the `vram_checks` / `vram_headroom_mb` parameter
threading** into a single injected dependency and gives the inline call sites a
cheap `enabled` short-circuit (fixing the always-on `_warn_vram` math). Lowest
risk, purely additive, reuses `check_vram_headroom` / `cap_cuda_memory` /
`vram_stats` unchanged. It does **not** by itself remove the inline hot-loop call
or centralize batching — it tidies the plumbing that A later consumes.

### Strategy C — push introspection to the `transcribe.py` stage boundary
Make every `infer_fn` pure; have the orchestrator wrap each stage with
pre/post estimate → cap → flush. Cleanest purity for the model functions, but it
**loses the per-batch VRAM signal** (estimates once per stage, not per batch) and
**can't hoist OOM-halving** (which is inherently inside the batch loop). Good as
a conceptual endpoint for the load/cap bracketing, weak for the per-batch memory
dynamics that are the actual problem.

### Recommendation — A, reached incrementally (status)
1. **[done] ASR longest-first sort** + guard tests (`tests/test_asr_batching.py`,
   incl. `test_process_processes_longest_segment_first`). Bench-confirmed +10%.
2. **[partial] `MemoryPolicy`** — landed as an object with `enabled` / `warn()` /
   `cap_after_load()`, consumed by the ASR pipeline and used to gate the estimate
   math (the `--no-warn-vram` win) and the post-load cap. **Still open:** fully
   collapsing the `vram_checks` / `vram_headroom_mb` param threading through
   config → `__main__` → `load_*` into the injected policy everywhere; and the
   `log_vram_delta` per-batch signal.
3. **[done] `BatchExecutor` (A)** — owns ordering + batching + OOM-halving + flush;
   `run_adaptive_batches` is now a shim over it; all three stages migrated
   (**ASR → alignment → vocal isolation**). **Still open:** the executor's
   **output sink** (single owned off-GPU boundary) — today each stage still does
   its own `.cpu()`/decode.

`PipelineStage`'s debug-cache role and `model_scope` / `flush_vram` are untouched —
none of this changed the load/unload-between-stages contract.

---

## Verification
- **CPU tests:** `uv run --with pytest pytest tests/test_batch_executor.py
  tests/test_asr_batching.py tests/test_alignment_batching.py
  tests/test_vocal_isolation_batching.py -v` — executor ordering/OOM/flush,
  MemoryPolicy gating, ASR longest-first + scatter/order, and the migrated
  alignment/vocal-isolation stages all green.
- **Full suite:** `uv run --with pytest pytest tests/ -q` — green aside from the
  pre-existing, unrelated `test_cantonese_cleaner.py` numeral failure.
- **On-GPU perf:** `scripts/bench_asr_native.py --sort none` vs `desc` (ordering
  win, reserved-climb flattening) and `--warn-vram` vs `--no-warn-vram` (estimate-
  gating win), per its docstring's diagnosis ladder.

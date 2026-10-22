# torch.compile

Two stages compile their model on CUDA: vocal isolation (`--vocal_isolation_compile`,
`pipeline/vocal_isolation.py`) and the Qwen3-ASR decode step (`--compile`,
`pipeline/_asr_native.py`). Both default to `auto`, both need Triton (Linux has it; on Windows
it is the `compile` extra), and both fall back to running uncompiled when it is missing or a
compile fails. Measurements below are from an RTX 3080 10 GB, torch 2.11, transformers 5.13.

## What a compile costs, and why it was costing more

A compile in a fresh process does three things: Dynamo traces the Python bytecode into a
graph, AOTAutograd turns that into an inference graph, and Inductor generates and loads the
kernels. Inductor's FX-graph cache and the AOTAutograd cache, both on disk, skip the last two
on a later run, but **only for a graph they are willing to cache**. Dynamo's tracing is never
cached. So a "warm" compile costs the tracing plus whatever the caches refused.

Check what a compile hit before reasoning about its cost:

```python
from torch._dynamo.utils import counters, compile_times
# after the first compiled call:
counters["inductor"]       # fxgraph_cache_hit / fxgraph_cache_miss
counters["aot_autograd"]   # autograd_cache_hit / autograd_cache_bypass
print(compile_times(repr="str", aggregate=True))  # compile_attempt_N, bytecode_tracing, ...
```

Two things to look for:

- **`compile_attempt_1`** means Dynamo threw away a whole trace and started again. A graph
  break mid-forward does this: it traces to the break, then restarts to compile the part before
  it.
- **`autograd_cache_bypass`** means AOTAutograd rebuilt the graph from scratch. The debug log
  (`torch._functorch._aot_autograd.autograd_cache` at DEBUG) names the node it refused:
  "Unsupported call_function target ...". `autograd_cache.check_node_safe` run over a graph's
  nodes lists them without a GPU (compile with a custom backend on CPU).

A standalone script and the pipeline produce different cache keys for the same model, so each
pays its own first compile. A miss on the first pipeline run after a model change is expected.

## Vocal isolation

Every chunk is the same `(batch, 2, chunk_size)` shape, so one static graph serves a run once
each file's short last batch is padded up to size. Compiled, a chunk takes about half as long
(batch 2: ~184 ms eager, ~96 ms compiled; at batch 4 earlier, 152 vs 77 ms). That is not
batching paying off: eager spends most of its time on small elementwise kernels and their
launch overhead, which compiling fuses away.

Warm compiles used to take 31-54 s, which a 580-chunk episode barely repaid. The breakdown of
a warm 30 s compile: ~11.5 s Dynamo tracing, of which ~7 s was a discarded first attempt
(the eager `istft` is a graph break, see `mbroformer/model.py`), ~15 s AOTAutograd for a graph
whose cache was bypassed, ~1 s loading Inductor's kernels. The bypass had two causes in turn:
einops `pack`/`unpack`, then the `sdpa_kernel` context entered inside every attention call.
The fixes:

- compile `MelBandRoformer.masked_stft` only (the model up to its `istft`), then call
  `reconstruct` eagerly: one graph, no break, no restart;
- single-tensor `pack`/`unpack` written as reshapes;
- `Attend` skips its `sdpa_kernel` while compiling, and `_compile_model` holds the same
  restriction (`attend.sdpa_backends_for`) around the compiled call, so the same kernel is
  chosen.

Warm compile: ~13-14 s, of which ~7 s is Dynamo tracing that nothing caches. First compile on
a machine: ~3 minutes. Output is bit-identical to the old compiled path, and eager output is
unchanged. `auto` compiles at `_COMPILE_MIN_CHUNKS` (200, about 11 minutes of speech): ~88 ms
saved per chunk against ~14 s repays at ~160.

Not done: the remaining tracing could only be skipped by exporting an AOTInductor package,
which would need its own invalidation by torch version and GPU. Not worth ~7 s.

## ASR decode step

`generate()` runs a prefill over the prompt (the audio tower included), then one forward per
new token.

**Rejected: compiling `model.forward` whole** (`scripts/bench_asr_compile.py`). The audio
tower's `get_audio_cu_seqlens` calls `.item()` and breaks the graph, and with the default
growing KV cache every decode step is a new shape, so each new `(batch, length)` shape spent
the recompile limit (tens of seconds) for a speedup on the rare batch that repeated one.
`dynamic=True` made the first compile far worse (137-408 s). Padding everything to one shape
stopped the recompiles but gained only 25-40% against 30 s+ per process.

**What is done instead:** transformers keeps the prefill eager and compiles only the decode
step, whose shape is fixed once the cache is static: `cache_implementation="static"`,
`CompileConfig(mode="reduce-overhead", dynamic=False)`, which runs the step as a CUDA graph.
The rules, all in `QwenPipelineNative`:

- **Only full batches of the configured `batch_size` take the compiled path.** Every batch
  size is its own graph (~7 s warm, ~40-60 s the first time on a machine), so a loop retry
  (1-2 rows), a short last batch, a batch halved after an OOM and ASR repair's odd-sized
  batches all run as they did before.
- **One cache length for the process** (`_static_cache_len`), sized for a `chunk_size` chunk
  (prompt tokens plus the generation budget, ~613 positions at 28 s). Later file groups and
  ASR repair's full batches reuse the graph. A call that needs more grows it once.
- **Out of memory, it falls back to uncompiled at the same batch size** before anything
  halves. The static cache holds the whole KV cache from the start, while an uncompiled batch
  only grows into it, so a batch size that fits uncompiled is kept. Any other failure is a
  broken compile: that batch re-runs uncompiled, and the process stops trying.
- **`auto` compiles at `_COMPILE_MIN_BATCHES` (3) full batches** in one call: each saves
  ~6.6 s at batch 12 against ~10 s for a warm compile (a first-ever one, ~50 s, takes ~8 to
  repay). `_plan_batches` decides before batching, in `run`, `process` and `transcribe`.
- The language-check model ASR repair loads (`asr_foreign_model`) always runs uncompiled.

### Measured

`scripts/bench_asr_static_cache.py`, 324 chunks (Bluey, Doraemon 555-556, Triumph in the
Skies E01, five 10-minute `build_eval_episodes.py` shows), GT CER over the seven with ground
truth, ASR stage only, warm cache:

| mode | batch | time | peak allocated | peak reserved | GT CER | chunks same as eager b12 |
|---|---|---|---|---|---|---|
| eager | 8 | 397 s | 5.6 GB | 6.1 GB | 8.97% | 90.4% |
| eager | 12 | 290 s | 6.4 GB | 8.4 GB | 8.91% | — |
| eager | 16 | 246 s | 7.2 GB | 9.9 GB | 8.90% | 91.4% |
| static cache, uncompiled | 12 | 299 s | 7.2 GB | 9.4 GB | 8.91% | 99.4% |
| compiled | 8 | 115 s | 6.1 GB | 6.8 GB | 8.96% | 89.8% |
| compiled | 10 | 103 s | 6.7 GB | 7.9 GB | 8.94% | 90.1% |
| compiled | 12 | 95 s | 7.2 GB | 9.2 GB | 8.95% | 94.4% |
| compiled | 16 | 104 s | 8.3 GB | 10.9 GB | 8.92% | 91.4% |

- Decode at batch 12: 12.6 steps/s eager, 49 compiled. The gain is all CUDA graphs: the static
  cache alone is no faster.
- **VRAM:** the static cache costs ~70 MB per batch row (+0.8 GB at batch 12), held during the
  eager prefill, which sets the peak. At equal or lower VRAM it still wins: compiled batch 8
  (6.1 GB) ran 2.5x faster than eager batch 12 (6.4 GB), so on a tight card lower `batch_size`
  rather than turn compiling off. Batch 16 reserved more than the card and spilled into shared
  memory (WDDM); throughput levels off around batch 10-12 anyway.
- **Text:** compiling changes fewer chunks (5.6%, mostly punctuation; 6 of 324 in their
  characters) than eager does between batch sizes 8, 12 and 16 (~9-10%). GT CER spans
  8.90-8.97% over every run: numerical noise, not a quality shift. Output is identical from
  run to run at every batch size.

End to end, the same 8 inputs through the pipeline (batch 12, isolation and diarization on,
VAD and isolation reloaded): Transcription 4:49 off, 1:52 with `--compile auto` (14 s for the
first batch; 66 s the first time), the whole run 9:36 -> 6:29, peak 6.7 -> 7.5 GB. The final
SRTs had the same cue counts, 2469 of 2479 cues identical, start error and coverage unchanged,
GT CER 8.86% -> 8.90%. One cue boundary moved where a 。 became ，, which cue assembly cuts at:
expect that kind of difference about once in 250 cues.

cantocaptions-dataset's `Transcriber` (dev split, 537 clips, batch 8): 452 s off, 111 s
compiled (153 s the first time), CER 6.66% both ways, 529 of 537 clips identical in their
characters, 5.6 -> 5.9 GB.

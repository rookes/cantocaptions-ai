"""Teacher-forced scoring of alternative wordings with a Qwen3-ASR model.

Given a clip and its transcript, how likely does the model find each of a few alternatives
at given spots in the transcript (喇 or 啦 at the end of a clause)? This is the model's
own judgement, audio and context included, of a choice it was trained to make. It is used
by the proofreading tools to weigh sentence-final particles; see docs/particle-scoring.md.

The method follows cantocaptions-dataset's probe (probe.py `_continuation_logprobs`):

* one forward pass over prompt + transcript + EOS keeps its KV cache;
* for each spot, latest first, the cache is cropped to just before it, widened to one row
  per alternative, and each alternative is scored as `alternative + the next few
  transcript characters` (a shared tail, so the options explain the same following text);
  the audio encoder never runs again;
* a spot that starts inside a multi-character token is branched at that token's start,
  and the token's leading characters become part of every continuation.

Only the differences between alternatives at one spot mean anything; the raw joint
log-probabilities are returned and the caller compares them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

# Transcript characters after a spot that every alternative must also explain.
TAIL_CHARS = 4


@dataclass
class Spot:
    start: int                      # character span of the transcript's own wording
    end: int
    alts: List[str]                 # alternatives to it (its own wording is scored too)


@dataclass
class SpotScore:
    start: int
    end: int
    logprobs: dict = field(default_factory=dict)   # wording -> joint log-probability
    tokens: dict = field(default_factory=dict)     # wording -> tokens scored


def branch_token(offsets: Sequence[Tuple[int, int]], start: int) -> Optional[int]:
    """Index of the token whose span holds character *start* (where the alternatives
    branch off), or None when no token covers it."""
    for t, (a, b) in enumerate(offsets):
        if a <= start < b:
            return t
    return None


def continuations(text: str, spot: Spot, branch_char: int, tail_chars: int = TAIL_CHARS) -> List[Tuple[str, str]]:
    """[(wording, continuation text)] for the transcript's own wording and each alternative:
    from the branch point, through the wording, to the shared tail."""
    head = text[branch_char:spot.start]
    tail = text[spot.end:spot.end + tail_chars]
    own = text[spot.start:spot.end]
    out, seen = [], set()
    for w in [own, *spot.alts]:
        if w not in seen:
            seen.add(w)
            out.append((w, head + w + tail))
    return out


def label_prefix(raw: str, language: str) -> str:
    """The preamble to prefill for a model that writes `language X<asr_text>` before its
    transcription (stock Qwen3-ASR), forcing *language* as the official SDK does; "" for a
    model that writes the text straight away (the fine-tuned model)."""
    return f"language {language}<asr_text>" if "<asr_text>" in raw else ""


def score_spots(model, processor, wav, text: str, spots: Sequence[Spot], language: str,
                prefix: str = "", tail_chars: int = TAIL_CHARS) -> List[SpotScore]:
    """Joint log-probability of each wording at each spot of *text* spoken in *wav*."""
    import torch

    tok = processor.tokenizer
    inputs = processor.apply_transcription_request(audio=[wav], language=language)
    inputs = inputs.to(model.device, model.dtype)
    prompt = inputs["input_ids"]
    if prefix:
        pre = tok(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"]
        prompt = torch.cat([prompt, pre.to(prompt.device, prompt.dtype)], dim=1)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
    tgt = torch.tensor([ids + [tok.eos_token_id]], dtype=prompt.dtype, device=prompt.device)
    input_ids = torch.cat([prompt, tgt], dim=1)
    batch = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    batch.update(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    with torch.inference_mode():
        out = model(**batch, use_cache=True)
    cache = out.past_key_values
    del out
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    planned = []
    for spot in spots:
        t = branch_token(offsets, spot.start)
        if t is None:
            continue
        conts = continuations(text, spot, offsets[t][0], tail_chars)
        planned.append((prompt.shape[1] + t, spot, conts))
    results = []
    # Latest spot first: a cache can only be cropped shorter.
    for abs_pos, spot, conts in sorted(planned, key=lambda p: -p[0]):
        cont_ids = [tok(c, add_special_tokens=False)["input_ids"] for _, c in conts]
        keep = [(w, i) for (w, _), i in zip(conts, cont_ids) if i]
        lps = _continuation_logprobs(model, cache, input_ids, abs_pos, [i for _, i in keep], pad_id)
        results.append(SpotScore(spot.start, spot.end, {w: lp for (w, _), lp in zip(keep, lps)},
                                 {w: len(i) for w, i in keep}))
    results.sort(key=lambda r: r.start)
    return results


def _continuation_logprobs(model, cache, input_ids, abs_pos: int, conts: List[List[int]], pad_id: int) -> List[float]:
    """logP of several continuations at one position, in a single forward over the cropped,
    widened gold cache. The gold pass already used the logits of the position before
    *abs_pos*, so that token is fed again as context and the logits realign."""
    import torch

    past = abs_pos - 1
    ctx = input_ids[:, past:abs_pos]
    lens = [len(c) for c in conts]
    m, n = max(lens), len(conts)
    rows = [torch.cat([ctx, torch.tensor([c + [pad_id] * (m - len(c))], dtype=ctx.dtype, device=ctx.device)], dim=1)
            for c in conts]
    new = torch.cat(rows, dim=0)
    attn = torch.ones((n, past + 1 + m), dtype=torch.long, device=new.device)
    for i, length in enumerate(lens):
        attn[i, past + 1 + length:] = 0
    with torch.inference_mode():
        cache.crop(past)
        cache.batch_repeat_interleave(n)
        out = model(input_ids=new, attention_mask=attn, past_key_values=cache, use_cache=True,
                    cache_position=torch.arange(past, past + 1 + m, device=new.device))
        # Back to one row, so the next (earlier) spot can crop this cache again.
        cache.batch_select_indices(torch.zeros(1, dtype=torch.long, device=new.device))
    logprobs = torch.log_softmax(out.logits[:, :-1].float(), dim=-1)
    return [float(logprobs[i, :lens[i]].gather(-1, torch.tensor(conts[i], device=new.device)[:, None]).sum())
            for i in range(n)]

"""Hosted LLM providers for the proofreading stage.

This is the only module in the package that reaches the network, and it imports each SDK
only when that provider is used, so an offline install never loads one. Install with
``pip install 'cantocaptions_ai[proofread]'``. Keys come from the environment only
(``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``, ``ANTHROPIC_API_KEY``) and are never logged.

Both providers make a single request with structured (JSON schema) output. A tool loop that
let the model query the audio was built and measured during development and is deliberately
absent: acoustic models share the recogniser's mishearings, so the model learned to
optimise toward them.
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from cantocaptions_ai.pipeline.proofread.render import OUTPUT_SCHEMA, Request
from cantocaptions_ai.utils.log_utils import get_logger

logger = get_logger(__name__)

PROVIDERS = ("gemini", "anthropic")
DEFAULT_MODELS = {"gemini": "gemini-3.7-flash", "anthropic": "claude-opus-5-5"}

# $ per million tokens (input, output), for cost reports and --proofread_max_cost. Prices
# change; a model missing here simply has no estimate. Cached input bills at a tenth.
PRICES: Dict[str, Tuple[float, float]] = {
    "gemini-3.7-flash": (0.75, 3.75),
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

_REQS = {
    "gemini": ("google-genai", "google.genai", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
    "anthropic": ("anthropic", "anthropic", ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")),
}
_GEMINI_THINKING = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH", "xhigh": "HIGH",
                    "max": "HIGH"}


class ProviderError(RuntimeError):
    """A request that did not produce a usable answer. ``usage`` is what it still cost."""

    def __init__(self, message: str, usage: Optional[dict] = None):
        super().__init__(message)
        self.usage = usage or {}


@dataclass
class Reply:
    raw: str
    answer: dict
    usage: dict = field(default_factory=dict)
    seconds: float = 0.0


def preflight(provider: str) -> Optional[str]:
    """Why *provider* cannot run here, or None. Checked before any model loads."""
    if provider not in _REQS:
        return f"unknown proofreading provider {provider!r}; one of {', '.join(PROVIDERS)}"
    package, module, env = _REQS[provider]
    try:
        found = importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        found = False
    if not found:
        return (f"the {package} package is not installed; install the proofreading extra: "
                "pip install 'cantocaptions_ai[proofread]'")
    if not any(os.environ.get(v) for v in env):
        return f"no API key for {provider}: set {' or '.join(env)} in the environment"
    return None


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    if model not in PRICES:
        return None
    pin, pout = PRICES[model]
    return (input_tokens * pin + output_tokens * pout) / 1e6


def cost_of(model: str, usage: dict) -> Optional[float]:
    if model not in PRICES:
        return None
    pin, pout = PRICES[model]
    return round((usage.get("input_tokens", 0) * pin
                  + usage.get("cache_read_input_tokens", 0) * pin * 0.1
                  + usage.get("cache_creation_input_tokens", 0) * pin * 1.25
                  + usage.get("output_tokens", 0) * pout) / 1e6, 6)


def parse_answer(raw: str) -> dict:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("answer is not a JSON object")
    return {"names": list(data.get("names") or []), "edits": list(data.get("edits") or []),
            "flags": list(data.get("flags") or [])}


def count_tokens(provider: str, model: str, req: Request) -> Optional[int]:
    """Exact input tokens from the provider (free on both), or None if it cannot say."""
    try:
        if provider == "gemini":
            client = _gemini_client(timeout_s=60)
            return client.models.count_tokens(
                model=model, contents=req.system + "\n\n" + req.user).total_tokens
        if provider == "anthropic":
            import anthropic
            return anthropic.Anthropic().messages.count_tokens(
                model=model, system=req.system,
                messages=[{"role": "user", "content": req.user}]).input_tokens
    except Exception as e:  # an estimate is not worth failing a run over
        logger.debug("Token count unavailable (%s); using the offline estimate", e)
    return None


def send(provider: str, model: str, req: Request, effort: str = "medium",
         timeout_s: float = 1800.0, cache: Optional[str] = None) -> Reply:
    """One request. ``cache`` is an :func:`open_cache` handle holding ``req.system``."""
    if provider == "gemini":
        return _gemini(model, req, effort, timeout_s, cache)
    if provider == "anthropic":
        return _anthropic(model, req, effort, timeout_s)
    raise ValueError(f"unknown proofreading provider {provider!r}")


def open_cache(provider: str, model: str, system: str, timeout_s: float) -> Optional[str]:
    """An explicit cache of the system prompt for several requests in a row, or None.

    The system prompt (instructions plus conventions) is the same for every chunk of a file
    and is most of each chunk's input. Anthropic needs no handle: its system block carries a
    ``cache_control`` breakpoint on every request. Gemini caches a repeated prefix implicitly
    but only best-effort, and not at all for requests that start before the first finishes,
    so chunked runs make an explicit cache and delete it afterwards. Any failure (a prompt
    under the model's minimum, an SDK without the API) falls back to implicit caching.
    """
    if provider != "gemini":
        return None
    try:
        from google.genai import types
        client = _gemini_client(timeout_s=120)
        cache = client.caches.create(model=model, config=types.CreateCachedContentConfig(
            system_instruction=system, ttl=f"{int(min(3600, 2 * timeout_s + 300))}s"))
        logger.info("Proofreading: system prompt cached for this file (%s)", cache.name)
        return cache.name
    except Exception as e:
        logger.info("Proofreading: no explicit prompt cache (%s); relying on implicit caching",
                    f"{type(e).__name__}: {e}")
        return None


def close_cache(provider: str, cache: Optional[str]) -> None:
    if provider != "gemini" or not cache:
        return
    try:
        _gemini_client(timeout_s=120).caches.delete(name=cache)
    except Exception as e:   # it expires on its own; not worth failing over
        logger.debug("Could not delete prompt cache %s: %s", cache, e)


# --- Gemini ---------------------------------------------------------------------------

def _gemini_client(timeout_s: float):
    from google import genai
    from google.genai import types
    # The SDK sets no request timeout of its own, so a dropped connection otherwise waits
    # forever; HTTP retries cover rate limits and transient server errors.
    return genai.Client(http_options=types.HttpOptions(
        timeout=int(timeout_s * 1000),
        retry_options=types.HttpRetryOptions(attempts=3,
                                             http_status_codes=[429, 500, 502, 503, 504])))


def _gemini(model: str, req: Request, effort: str, timeout_s: float,
            cache: Optional[str] = None) -> Reply:
    from google.genai import types

    client = _gemini_client(timeout_s)
    # A cached system prompt replaces system_instruction; the API refuses both at once.
    prefix = {"cached_content": cache} if cache else {"system_instruction": req.system}
    # Temperature stays at the model default: Google's guidance for Gemini 3 is that lowering
    # it can cause looping on reasoning tasks.
    config = types.GenerateContentConfig(
        **prefix, max_output_tokens=65536,
        response_mime_type="application/json", response_json_schema=OUTPUT_SCHEMA,
        thinking_config=types.ThinkingConfig(
            thinking_level=getattr(types.ThinkingLevel, _GEMINI_THINKING.get(effort, "MEDIUM"))))
    t0 = time.time()
    resp = None
    for attempt in (1, 2):
        try:
            resp = client.models.generate_content(model=model, contents=req.user, config=config)
            break
        except Exception as e:
            timed_out = "timeout" in type(e).__name__.lower() or "timed out" in str(e).lower()
            if attempt == 2 or not timed_out:
                raise ProviderError(f"Gemini request failed: {type(e).__name__}: {e}") from e
            logger.warning("Proofreading request timed out after %.0fs; retrying once", timeout_s)
    usage = _gemini_usage(resp)
    usage["cost_usd"] = cost_of(model, usage)
    cand = (resp.candidates or [None])[0]
    finish = str(getattr(cand, "finish_reason", "") or "")
    if cand is None or any(finish.endswith(x) for x in ("SAFETY", "RECITATION",
                                                        "PROHIBITED_CONTENT", "BLOCKLIST")):
        raise ProviderError(f"Gemini returned no usable answer (finish_reason={finish or None})",
                            usage)
    if finish.endswith("MAX_TOKENS"):
        raise ProviderError("Gemini hit its output cap; the answer is truncated", usage)
    try:
        answer = parse_answer(resp.text)
    except (ValueError, json.JSONDecodeError, IndexError) as e:
        raise ProviderError(f"Gemini's answer is not valid JSON: {e}", usage) from e
    return Reply(raw=resp.text, answer=answer, usage=usage, seconds=round(time.time() - t0, 1))


def _gemini_usage(resp) -> dict:
    u = getattr(resp, "usage_metadata", None)
    if u is None:
        return {}
    g = lambda name: getattr(u, name, 0) or 0  # noqa: E731
    cached = g("cached_content_token_count")
    # prompt_token_count includes the implicitly cached part; split it so cost is right.
    return {"input_tokens": g("prompt_token_count") - cached,
            "cache_read_input_tokens": cached,
            "output_tokens": g("candidates_token_count") + g("thoughts_token_count"),
            "thinking_tokens": g("thoughts_token_count")}


# --- Anthropic ------------------------------------------------------------------------

def _anthropic(model: str, req: Request, effort: str, timeout_s: float) -> Reply:
    import anthropic

    client = anthropic.Anthropic(timeout=timeout_s)
    kwargs = dict(
        model=model, max_tokens=64000,
        # The system prompt (rules and conventions) is identical for every file of a run,
        # so it is the block worth caching.
        system=[{"type": "text", "text": req.system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": req.user}],
        thinking={"type": "adaptive"},
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        # Server-side refusal fallback: a decline would otherwise read as "no edits".
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
    )
    t0 = time.time()
    try:
        with client.beta.messages.stream(**kwargs) as stream:
            msg = stream.get_final_message()
    except anthropic.APIError as e:
        raise ProviderError(f"Anthropic request failed: {type(e).__name__}: {e}") from e
    usage = {k: getattr(msg.usage, k, 0) or 0 for k in (
        "input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}
    usage["cost_usd"] = cost_of(model, usage)
    if msg.stop_reason == "refusal":
        raise ProviderError("the request was refused", usage)
    if msg.stop_reason == "max_tokens":
        raise ProviderError("hit max_tokens; the answer is truncated", usage)
    raw = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        answer = parse_answer(raw)
    except (ValueError, json.JSONDecodeError, IndexError) as e:
        raise ProviderError(f"the answer is not valid JSON: {e}", usage) from e
    return Reply(raw=raw, answer=answer, usage=usage, seconds=round(time.time() - t0, 1))

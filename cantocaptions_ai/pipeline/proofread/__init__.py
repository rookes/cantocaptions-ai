"""Optional LLM proofreading of finished subtitles (stage 9).

**Off unless asked for.** This is the only part of the pipeline that uses the network and
costs money per run, against the library's contract of running offline once models are
downloaded. It runs only when ``proofread`` is set to a provider -- per run with
``--proofread gemini``, or as a personal default in ``user.cfg``::

    [proofreading]
    proofread = gemini

What it does: the finished cues of one file go to a hosted LLM in one request, with the
reference subtitle (``--reference_subtitle``) interleaved by time when there is one and the
language's proofreading standard in the system prompt. The model returns structured JSON --
proper-noun decisions first, then per-cue corrections, then flags -- and the corrections are
applied in code (``apply.py``), never by trusting the model's text wholesale. Edited cues go
back through the cleaner, so a correction cannot reintroduce a convention violation.

Everything language-specific comes from the language pack's ``ProofreadStandard``: what the
written language is called, its conventions file, examples of its typical errors, and the
register edits must not drift into. A language with no standard proofreads for meaning only;
``--proofread_conventions`` and ``--proofread_prompt`` override either half without code.

Measured on Cantonese episodes with a strict human subtitle and a Standard Chinese
reference, one request per episode with Gemini 3.7 Flash at medium thinking: roughly a
quarter of the remaining character errors removed, about nine correct edits for every wrong
one, at around ten US cents and a few minutes per 25-minute episode. Results without a
reference, and on other languages, have not been measured.

Artifacts under ``{debug_dir}/{stem}/proofread/``: each request and its answer (reused by a
``--load_debug_dir`` replay only when the request is byte-identical, so a replay never
re-bills), ``changes.srt`` (each edited cue as new text over ``[was]`` old text) and
``flags.srt`` (lines the model doubted but could not fix, for a human with the audio).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from cantocaptions_ai.languages.base import ProofreadStandard
from cantocaptions_ai.pipeline.proofread import providers
from cantocaptions_ai.pipeline.proofread.apply import Outcome, apply_answer
from cantocaptions_ai.pipeline.proofread.render import (
    AGREEMENT_TOLERANCE, MIN_AGREEMENT, Cue, Request, approx_tokens, best_reference_shift,
    reference_agreement, system_prompt, user_message,
)
from cantocaptions_ai.text_profiles import DEFAULT_SCRIPT, ScriptConfig
from cantocaptions_ai.utils.log_utils import get_logger
from cantocaptions_ai.utils.schema import add_note

logger = get_logger(__name__)

# Output is mostly the model's thinking, which no estimate can know in advance. Measured
# over whole episodes at medium effort it came to well under the input; this covers that
# with room, for --proofread_max_cost only.
OUTPUT_ESTIMATE_RATIO = 0.8
OUTPUT_ESTIMATE_FLOOR = 8000


@dataclass
class ProofreadSettings:
    provider: str
    model: str
    standard: ProofreadStandard
    script: ScriptConfig = DEFAULT_SCRIPT
    effort: str = "medium"
    template: Optional[str] = None        # replaces prompt.md
    conventions: Optional[str] = None     # replaces the standard's conventions
    context: str = ""                     # a paragraph about the show
    min_confidence: str = "low"
    particles: str = "protect"            # protect | allow
    max_cost: Optional[float] = None      # USD per request; None: no ceiling
    dry_run: bool = False
    timeout: float = 1200.0
    chunk_cues: int = 0                   # 0: the whole file in one request
    chunk_context: int = 6                # read-only cues either side of a chunk
    parallel: int = 4                     # chunk requests in flight at once

    @classmethod
    def from_config(cls, cfg, pack, profile) -> "ProofreadSettings":
        def read(path: Optional[str]) -> Optional[str]:
            return Path(path).read_text(encoding="utf-8") if path else None
        return cls(
            provider=cfg.proofread,
            model=cfg.proofread_model or providers.DEFAULT_MODELS[cfg.proofread],
            standard=pack.standard_for(cfg.proofread_standard),
            script=profile.script,
            effort=cfg.proofread_effort,
            template=read(cfg.proofread_prompt),
            conventions=read(cfg.proofread_conventions),
            context=read(cfg.proofread_context) or "",
            min_confidence=cfg.proofread_min_confidence,
            particles=cfg.proofread_particles,
            max_cost=cfg.proofread_max_cost,
            dry_run=cfg.proofread_dry_run,
            timeout=cfg.proofread_timeout,
            chunk_cues=cfg.proofread_chunk_cues,
            chunk_context=cfg.proofread_chunk_context,
            parallel=cfg.proofread_parallel,
        )


@dataclass
class FileResult:
    """What proofreading one file did, for the summary and for callers."""
    edits: List[dict] = field(default_factory=list)
    flags: List[dict] = field(default_factory=list)
    invalid: int = 0
    names: List[dict] = field(default_factory=list)
    usage: Dict[str, float] = field(default_factory=dict)
    requests: int = 0
    replayed: int = 0
    before: Dict[int, str] = field(default_factory=dict)   # changed cue id -> its old text


class Proofreader:
    def __init__(self, settings: ProofreadSettings):
        self.settings = settings
        self.system = system_prompt(settings.standard, settings.template, settings.conventions,
                                    particles=settings.particles)

    # -- requests -----------------------------------------------------------------------

    def _chunks(self, cues: List[Cue]) -> List[List[Cue]]:
        size = self.settings.chunk_cues
        if not size or len(cues) <= size:
            return [cues]
        out = []
        for lo in range(0, len(cues), size):
            hi = min(lo + size, len(cues))
            pad = max(0, self.settings.chunk_context)
            window = cues[max(lo - pad, 0):min(hi + pad, len(cues))]
            out.append([Cue(c.id, c.start, c.end, c.text, editable=lo < c.id <= hi)
                        for c in window])
        return out

    def _request(self, chunk: List[Cue], reference: Sequence[dict], title: str) -> Request:
        t0, t1 = chunk[0].start - 1.0, chunk[-1].end + 1.0
        ref = [r for r in reference if r["end"] > t0 and r["start"] < t1]
        return Request(self.system, user_message(chunk, ref, title, self.settings.context))

    def _key(self, req: Request) -> str:
        s = self.settings
        blob = json.dumps([s.provider, s.model, s.effort, req.system, req.user],
                          ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _guard_cost(self, req: Request) -> None:
        s = self.settings
        n_in = providers.count_tokens(s.provider, s.model, req) or approx_tokens(req.system + req.user)
        n_out = max(OUTPUT_ESTIMATE_FLOOR, int(n_in * OUTPUT_ESTIMATE_RATIO))
        estimate = providers.estimate_cost(s.model, n_in, n_out)
        if estimate is None:
            if s.max_cost is not None:
                logger.warning("No price known for %s, so proofread_max_cost cannot be "
                               "enforced for it", s.model)
            return
        logger.info("Proofreading request: ~%d input tokens, estimated $%.3f", n_in, estimate)
        if s.max_cost is not None and estimate > s.max_cost:
            raise providers.ProviderError(
                f"estimated cost ${estimate:.3f} is over proofread_max_cost ${s.max_cost:.2f}")

    # -- one file -------------------------------------------------------------------------

    def run(self, name: str, segments: List[dict], reference: Sequence[dict] = (),
            debug_dir: Optional[str] = None, load_debug_dir: Optional[str] = None,
            title: str = "", save_dir: Optional[str] = None) -> FileResult:
        """Proofread *segments* in place (their ``text``) and return what was done.

        ``save_dir`` is where each paid answer and the review files go; by default the debug
        stage dir, ``{debug_dir}/{stem}/proofread/``. A caller with no debug dir should pass
        one anyway (the CLI uses ``{output_dir}/{name}.proofread/``): an answer is paid for.

        A file in several chunks sends the first chunk alone and the rest
        ``proofread_parallel`` at a time, behind one explicit prompt cache where the provider
        has one. A chunk whose request fails is skipped with a warning and the others still
        apply; only when every request fails does this raise ``ProviderError``.
        """
        from cantocaptions_ai.utils import debug

        s = self.settings
        result = FileResult()
        cues = [Cue(i + 1, float(seg["start"]), float(seg["end"]), str(seg.get("text", "")))
                for i, seg in enumerate(segments)]
        if not cues:
            return result
        if save_dir is None and debug_dir:
            save_dir = debug._stage_dir(name, "proofread", debug_dir)
        if reference:
            self._check_reference_timing(name, cues, reference)
        chunks = self._chunks(cues)
        plans = []
        for k, chunk in enumerate(chunks):
            req = self._request(chunk, reference, title or name.split("/")[-1])
            key = self._key(req)
            record = debug.load_proofread_debug(name, load_debug_dir, k, key) if load_debug_dir else None
            if record is not None:
                result.replayed += 1
            plans.append([k, chunk, req, key, record])
        pending = [p for p in plans if p[4] is None]

        if s.dry_run:
            self._write_dry_run(name, pending, len(chunks), save_dir)
            pending = []
        elif pending:
            for p in pending:          # refuse before anything is sent, not halfway through
                self._guard_cost(p[2])
            self._send_all(name, pending, len(chunks), save_dir, result)

        texts: Dict[int, str] = {c.id: c.text for c in cues}
        names: List[dict] = []
        for k, chunk, req, key, record in plans:
            if record is None:
                continue
            for key_, value in record.get("usage", {}).items():
                if isinstance(value, (int, float)):
                    result.usage[key_] = round(result.usage.get(key_, 0) + value, 6)
            editable = {c.id: texts[c.id] for c in chunk if c.editable}
            answer = dict(record["answer"], names=[])     # names are applied file-wide below
            outcome = apply_answer(editable, answer, s.standard, s.script, s.min_confidence,
                                   particles=s.particles)
            texts.update(outcome.texts)
            result.edits += outcome.edits
            result.flags += outcome.flags
            result.invalid += len(outcome.invalid)
            names += record["answer"].get("names", [])

        if names:
            named = apply_answer(dict(texts), {"names": names, "edits": [], "flags": []},
                                 s.standard, s.script)
            texts.update(named.texts)
            edited = {e["id"] for e in result.edits}
            for e in result.edits:
                if e["id"] in named.texts:
                    e["text"] = named.texts[e["id"]]
            result.edits += [e for e in named.edits if e["id"] not in edited]
            result.names = names

        for c in cues:
            if texts[c.id] != c.text:
                seg = segments[c.id - 1]
                seg["text"] = texts[c.id]
                result.before[c.id] = c.text
                kinds = sorted({e.get("type", "other") for e in result.edits if e["id"] == c.id})
                add_note(seg, "proofread:" + ",".join(kinds or ["edit"]))
        if save_dir and not s.dry_run:
            self._write_review(save_dir, cues, texts, result)
        if result.invalid:
            logger.info("Proofreading: %d proposal(s) were invalid and ignored", result.invalid)
        return result

    def _check_reference_timing(self, name: str, cues: List[Cue], reference: Sequence[dict]) -> None:
        """Refuse, before anything is spent, a reference on a different timeline from the cues.

        The stream pairs each reference line with the draft cue it overlaps, so a reference
        timed to another release, or to the media when the cues are not (or the reverse,
        under ``--realign``), pairs every line with the wrong one -- at full price. A constant
        shift that would line it up is named, for ``--reference_offset``.
        """
        share = reference_agreement(cues, reference)
        logger.info("Proofreading %s: reference timing agreement %.0f%% (cue starts within %.1fs)",
                    name, 100 * share, AGREEMENT_TOLERANCE)
        if share >= MIN_AGREEMENT:
            return
        shift, best = best_reference_shift(cues, reference)
        hint = (f"; shifting it by {shift:+.1f}s would line up {best:.0%}, so try "
                f"--reference_offset {shift:+.1f} (on top of any offset already set)"
                if best >= MIN_AGREEMENT else
                "; no constant shift within 10 s fixes it, so it is probably timed to a "
                "different release (under --realign, check --reference_timing)")
        raise providers.ProviderError(
            f"the reference subtitle does not match these cues' timing: timing agreement "
            f"{share:.0%} (a matching reference scores ~83-94%){hint}. Nothing was sent.")

    def _send_all(self, name: str, pending: list, n_chunks: int, save_dir: Optional[str],
                  result: FileResult) -> None:
        """Send every pending request, filling in each plan's record (None if it failed).

        Each answer is saved the moment it arrives, so stopping the run (Ctrl+C) loses none
        that were already paid for; see :func:`_in_background` for why Ctrl+C works at all.
        """
        from cantocaptions_ai.utils import debug

        s = self.settings
        cache = None
        if len(pending) > 1:
            cache = providers.open_cache(s.provider, s.model, self.system, s.timeout)

        def one(plan):
            k, chunk, req, key, _ = plan
            label = f"chunk {k + 1}/{n_chunks}" if n_chunks > 1 else "request"
            n_edit = sum(c.editable for c in chunk)
            logger.info("Proofreading %s: %s (%d cues)", name, label, n_edit)
            try:
                reply = providers.send(s.provider, s.model, req, s.effort, s.timeout, cache=cache)
            except providers.ProviderError as e:
                return plan, None, e
            except Exception as e:      # an SDK surprise must not take the other chunks with it
                return plan, None, providers.ProviderError(f"{type(e).__name__}: {e}")
            if save_dir:
                record = {"request_hash": key, "provider": s.provider, "model": s.model,
                          "effort": s.effort, "answer": reply.answer, "raw": reply.raw,
                          "usage": reply.usage, "seconds": reply.seconds}
                debug.write_proofread_record(
                    save_dir, k, record, f"# system\n\n{req.system}\n\n# user\n\n{req.user}\n")
            return plan, reply, None

        try:
            first, rest = pending[:1], pending[1:]
            # The first request goes alone: it is what fills an implicit cache (and, for
            # Anthropic, writes the cache_control breakpoint) that the others then read.
            outcomes = _in_background(one, first, 1)
            if rest:
                outcomes += _in_background(one, rest, max(1, s.parallel))
        except KeyboardInterrupt:
            logger.warning(
                "Proofreading %s interrupted. Answers already received are saved%s; a request "
                "still in flight may be billed by the provider even though it is abandoned.",
                name, f" in {save_dir}" if save_dir else " nowhere (no output or debug dir)")
            raise
        finally:
            if cache is not None:
                providers.close_cache(s.provider, cache)

        failed = []
        for plan, reply, error in outcomes:
            k, chunk, req, key, _ = plan
            if error is not None:
                failed.append((k, error))
                for key_, value in (error.usage or {}).items():
                    if isinstance(value, (int, float)):
                        result.usage[key_] = round(result.usage.get(key_, 0) + value, 6)
                continue
            result.requests += 1
            record = {"request_hash": key, "provider": s.provider, "model": s.model,
                      "effort": s.effort, "answer": reply.answer, "raw": reply.raw,
                      "usage": reply.usage, "seconds": reply.seconds}
            plan[4] = record
            logger.info("Proofreading %s: chunk %d/%d answered in %.0fs, $%s", name, k + 1,
                        n_chunks, reply.seconds, reply.usage.get("cost_usd"))
        if failed and len(failed) == len(outcomes) and result.replayed == 0:
            k, error = failed[0]
            raise providers.ProviderError(
                f"every request failed ({len(failed)}); first: {error}", result.usage)
        for k, error in failed:
            logger.warning("Proofreading %s: chunk %d/%d failed and was skipped: %s",
                           name, k + 1, n_chunks, error)

    def _write_dry_run(self, name: str, pending: list, n_chunks: int,
                       save_dir: Optional[str]) -> None:
        s = self.settings
        where = save_dir or "."
        os.makedirs(where, exist_ok=True)
        total = 0.0
        for k, chunk, req, key, _ in pending:
            suffix = f".{k:02d}" if n_chunks > 1 else ""
            base = name.split("/")[-1]
            path = os.path.join(where, f"{base}.proofread.request{suffix}.md")
            Path(path).write_text(f"# system\n\n{req.system}\n\n# user\n\n{req.user}\n",
                                  encoding="utf-8")
            n = approx_tokens(req.system + req.user)
            est = providers.estimate_cost(s.model, n, max(OUTPUT_ESTIMATE_FLOOR,
                                                          int(n * OUTPUT_ESTIMATE_RATIO)))
            total += est or 0.0
            logger.info("Proofreading dry run: request written to %s (~%d input tokens%s); "
                        "nothing sent", path, n, f", estimated ${est:.3f}" if est is not None else "")
        if n_chunks > 1 and pending:
            logger.info("Proofreading dry run: %d request(s), estimated $%.3f in all "
                        "(before prompt caching)", len(pending), total)

    def _write_review(self, save_dir: str, cues: List[Cue], texts: Dict[int, str],
                      result: FileResult) -> None:
        from cantocaptions_ai.pipeline.proofread.review import write_changes_srt
        from cantocaptions_ai.utils.debug import write_labelled_srt
        os.makedirs(save_dir, exist_ok=True)
        write_changes_srt(os.path.join(save_dir, "changes.srt"), (
            (c.start, c.end, texts[c.id], c.text) for c in cues if texts[c.id] != c.text))
        by_id = {c.id: c for c in cues}
        write_labelled_srt(os.path.join(save_dir, "flags.srt"), (
            (by_id[f["id"]].start, by_id[f["id"]].end, [f.get("note", "")], texts[f["id"]])
            for f in sorted(result.flags, key=lambda f: f["id"]) if f["id"] in by_id))
        with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump({"edits": result.edits, "flags": result.flags, "names": result.names,
                       "invalid": result.invalid, "usage": result.usage,
                       "requests": result.requests, "replayed": result.replayed},
                      f, ensure_ascii=False, indent=1)
        logger.info("Proofreading review files (changes.srt, flags.srt, summary.json) and the "
                    "model's answers: %s", save_dir)


POLL_SECONDS = 0.2


def _in_background(fn, items: list, workers: int) -> list:
    """``[fn(item) for item in items]``, *workers* at a time on daemon threads, in order.

    Why not on the main thread, or a ThreadPoolExecutor: a request is one blocking network
    read lasting minutes, and on Windows Ctrl+C does not interrupt a blocking read -- Python
    raises KeyboardInterrupt only once the main thread runs bytecode again, i.e. after the
    answer arrives. An executor's threads are joined at interpreter exit, so even a delivered
    interrupt then waits for every request in flight. Here the main thread only ever waits in
    ``POLL_SECONDS`` slices, so Ctrl+C lands within a fraction of a second, and daemon threads
    do not hold the process open once it does. *fn* must not raise.
    """
    import threading

    results: list = [None] * len(items)
    queue = iter(enumerate(items))
    lock = threading.Lock()

    def worker():
        while True:
            with lock:
                nxt = next(queue, None)
            if nxt is None:
                return
            k, item = nxt
            results[k] = fn(item)

    threads = [threading.Thread(target=worker, daemon=True, name=f"proofread-{i}")
               for i in range(min(workers, len(items)))]
    for t in threads:
        t.start()
    for t in threads:
        while t.is_alive():
            t.join(POLL_SECONDS)
    return results


def load_proofreader(cfg, pack, profile) -> Optional[Proofreader]:
    """The run's proofreader, or None when proofreading is off (the default)."""
    if not getattr(cfg, "proofread", "none") or cfg.proofread == "none":
        return None
    settings = ProofreadSettings.from_config(cfg, pack, profile)
    if settings.standard.name == "generic" and settings.conventions is None:
        logger.warning(
            "No proofreading standard for language '%s': proofreading for meaning only. "
            "Pass --proofread_conventions FILE to give it conventions.", pack.code)
    logger.info("Proofreading enabled: %s %s (%s effort), standard '%s'%s",
                settings.provider, settings.model, settings.effort, settings.standard.name,
                " -- DRY RUN, nothing will be sent" if settings.dry_run else "")
    return Proofreader(settings)


__all__ = ["Proofreader", "ProofreadSettings", "FileResult", "load_proofreader"]

import argparse
import dataclasses
import os
import time
import warnings

import numpy as np
import torch

from cantocaptions_ai.utils.audio import load_audio, SAMPLE_RATE
from cantocaptions_ai.utils.schema import ProcessingItem, item_name
from typing import Callable, Dict, List, Optional
from cantocaptions_ai.utils.output import LANGUAGES, TO_LANGUAGE_CODE, get_writer, writer_args as build_writer_args
from cantocaptions_ai.utils.log_utils import ProgressSink, StageTimer, TranscriptionSummary, get_logger
from cantocaptions_ai.text_profiles import (
    DEFAULT_PUNCTUATION,
    DEFAULT_SCRIPT,
    DEFAULT_SEGMENTATION,
)
from cantocaptions_ai.pipeline.reference_context import CONTEXT_TEMPLATES
from cantocaptions_ai.pipeline.segmentation import assemble_cues
from cantocaptions_ai.utils.debug import (
    _stage_dir,
    write_precleaning_debug,
)

logger = get_logger(__name__)

# The stages' helpers live in pipeline/stages.py; re-exported here, where callers (the worker
# imports _select_audio_track, the tests several more) have always found them.
from cantocaptions_ai.pipeline.stages import (  # noqa: E402,F401
    _VIDEO_EXTENSIONS,
    _assign_speakers,
    _extract_timestamps,
    _load_realign_checkpoint,
    _pre_align_clean,
    _run_alignment,
    _run_diarization,
    _run_realign,
    _run_realign_asr,
    _select_audio_track,
    _substitution_overrides,
    _write_realign_checkpoint,
)

def _offset_result_times(result: dict, offset: float) -> None:
    """Shift every timestamp in *result* forward by *offset* seconds, in place.

    Used to map clip-relative times (produced when an audio_start/audio_end clip is
    applied at load time) back onto the source-media timeline. No-op when offset==0.
    """
    if not offset:
        return
    for segment in result.get("segments", []):
        for key in ("start", "end"):
            if segment.get(key) is not None:
                segment[key] += offset
        for token_key in ("words", "chars"):
            for token in segment.get(token_key, []) or []:
                for key in ("start", "end"):
                    if token.get(key) is not None:
                        token[key] += offset
    for token in result.get("word_segments", []) or []:
        for key in ("start", "end"):
            if token.get(key) is not None:
                token[key] += offset


def _merge_and_write(
    items: List[ProcessingItem],
    writer,
    align_language: str,
    align_merge_distance: float,
    align_padding: float,
    writer_args: dict,
    cleaner=None,
    layout=None,
    debug_dir: Optional[str] = None,
    punctuation=DEFAULT_PUNCTUATION,
    segmentation=DEFAULT_SEGMENTATION,
    script=DEFAULT_SCRIPT,
    min_cue_duration: float = 0.5,
    merge_gap: float = 0.25,
    max_line_width: Optional[int] = None,
    max_line_count: Optional[int] = None,
    merge: bool = True,
    max_cue_duration: float = 0.0,
    order_cues: bool = False,
    *,
    collect: bool = False,
    audio_start_offset: float = 0.0,
    display_paths: Optional[dict] = None,
    proofreader=None,
    reference_cues: Optional[list] = None,
    load_debug_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    output_format: str = "srt",
    carried_marks: Optional[tuple] = None,
    precleaner=None,
    summary=None,
    progress=None,
) -> List[ProcessingItem]:
    """Assemble cues, clean, offset, then write (unless writer is None) and/or return results.

    Output files and debug checkpoints are named by ``item['name']`` (see
    ``utils.output.output_names``). ``display_paths`` maps a (possibly clip-substituted
    temp) audio path back to the original path, so returned results reference the source
    file, not the temp clip. ``collect`` returns the final per-item results for in-memory
    (server) use; ``writer`` is None when nothing should be written to disk.

    ``layout`` (text -> text) breaks lines when there is no ``cleaner`` to do it; with a
    cleaner, line breaking is one of its own manifest steps.

    ``proofreader`` (``pipeline/proofread``, None unless the user enabled it) runs last on
    the finished cues, on the source timeline so ``reference_cues`` line up with them.
    """
    # An ordinary merge is capped at one subtitle line, but a short-cue rescue may use the
    # full multi-line budget -- the cleaner's linebreak step will split the result across
    # max_line_count lines anyway. Fall back to the single-line cap when line limits are off
    # (e.g. under --no_align, where both are required to be unset).
    rescue_max_chars = (
        max_line_width * max_line_count
        if max_line_width and max_line_count
        else script.line_width
    )
    # Reuse the cleaner as the single source of truth for "is this line pure noise": a cue
    # whose cleaned text is removable would have been dropped later anyway, so dropping it
    # before the rescue pass just stops it being glued onto a neighbour first.
    is_noise = (lambda text: cleaner.is_noise(cleaner.clean(text))) if cleaner is not None else None
    finalized: List[ProcessingItem] = []
    for item in items:
        result = item['result']
        name = item_name(item)
        audio_path = item['audio_path']
        if display_paths:
            audio_path = display_paths.get(audio_path, audio_path)
        result["language"] = align_language

        new_segments = assemble_cues(
            result["segments"],
            punctuation=punctuation,
            segmentation=segmentation,
            align_merge_distance=align_merge_distance,
            align_padding=align_padding,
            min_cue_duration=min_cue_duration,
            merge_gap=merge_gap,
            rescue_max_chars=rescue_max_chars,
            is_noise=is_noise,
            script=script,
            merge=merge,
            max_cue_duration=max_cue_duration,
        )

        if order_cues and debug_dir is not None:
            # After assembly so the timings and text are the ones that shipped, and before
            # cleaning so a cue dropped as noise does not silently take its reason with it.
            from cantocaptions_ai.utils.debug import write_realign_suspects
            write_realign_suspects(name, new_segments, debug_dir)

        if debug_dir is not None:
            # The general annotation channel, unlike suspect.srt above: not "these timings
            # are doubtful" but "something happened to this cue you may want to see". Written
            # on every run, since nothing about it is specific to --realign.
            from cantocaptions_ai.utils.debug import write_segment_notes
            write_segment_notes(name, new_segments, debug_dir)

        if order_cues:
            # An out-of-order SRT is rejected outright by strict readers, so this is the
            # last chance to guarantee validity -- and it has to be *here*, after assembly,
            # not with the other --realign fixups: the alignment stage's own output is in
            # order, and on the Doraemon fixture the one inverted pair appears somewhere
            # between there and the writer.
            from cantocaptions_ai.pipeline.realign import enforce_cue_order
            enforce_cue_order(new_segments)

        result["segments"] = new_segments  # TODO: update word_segments as well

        if debug_dir is not None:
            write_precleaning_debug(name, result, debug_dir)

        if cleaner is not None:
            # Cleaning edits segment text only; words/chars keep the original
            # alignment tokens and timings.
            cleaned_segments = []
            for segment in new_segments:
                text = cleaner.clean(segment["text"])
                if cleaner.is_noise(text):
                    continue
                segment["text"] = text
                cleaned_segments.append(segment)
            dropped = len(new_segments) - len(cleaned_segments)
            if dropped:
                logger.info(f"Text cleaning: dropped {dropped} interjection/noise subtitles")
            result["segments"] = cleaned_segments
        elif layout is not None:
            for segment in new_segments:
                segment["text"] = layout(segment["text"])

        # Map clip-relative times back onto the source-media timeline before output.
        _offset_result_times(result, audio_start_offset)

        marks = []
        if proofreader is not None and result["segments"]:
            marks = _proofread(proofreader, name, result, reference_cues, cleaner, layout,
                               debug_dir, load_debug_dir,
                               output_dir=output_dir if writer else None,
                               precleaner=precleaner, summary=summary, progress=progress)

        if carried_marks:
            # Proofread before realign: its bookmarks were on the copy's cues, which realign
            # may have dropped, split or merged. Carry them onto these cues by their text.
            from cantocaptions_ai.pipeline.proofread.review import remap_marks
            source_texts, source_marks = carried_marks
            moved = remap_marks(source_texts, source_marks,
                                [s.get("text", "") for s in result["segments"]])
            joined: dict = {}
            for idx, note in moved + marks:
                joined.setdefault(idx, []).append(note)
            marks = sorted((idx, "\n".join(notes)) for idx, notes in joined.items())
        if writer is not None:
            writer(result, name, writer_args)
            if marks and output_dir:
                _write_bookmarks(output_dir, name, output_format, marks)
        if collect:
            finalized.append({'audio_path': audio_path, 'name': name, 'result': result})
    return finalized


def _load_reference(cfg) -> Optional[list]:
    """``--reference_subtitle``'s cues with ``--reference_offset`` applied, or None."""
    if not cfg.reference_subtitle:
        return None
    from cantocaptions_ai.utils.subtitles import load_subtitle_file
    logger.info("Loading reference subtitle: %s", cfg.reference_subtitle)
    reference_cues = load_subtitle_file(cfg.reference_subtitle)
    logger.info("Loaded %d reference subtitle lines.", len(reference_cues))
    if cfg.reference_offset:
        from cantocaptions_ai.pipeline.reference_context import shift_cues
        before = len(reference_cues)
        reference_cues = shift_cues(reference_cues, cfg.reference_offset)
        logger.info(
            "Shifted reference subtitle by %+.3fs (%d cue(s), %d dropped before zero)",
            cfg.reference_offset, len(reference_cues), before - len(reference_cues),
        )
    return reference_cues


def proofreads_before_realign(cfg) -> bool:
    """Whether this run proofreads the ``--realign`` input first, rather than its output.

    First whenever the input carries timings and the reference (if any) shares them
    (``reference_timing = subtitle``): the corrected text is then what realign places, and the
    reference pairs with the cues it was timed against. A reference timed to the media
    (``media``), or a bare transcript with nothing to pair it against, waits for the
    realigned cues instead.
    """
    if cfg.proofread == "none" or not cfg.realign:
        return False
    from cantocaptions_ai.utils.subtitles import subtitle_has_timings
    if not subtitle_has_timings(cfg.realign):
        return False
    return not cfg.reference_subtitle or cfg.reference_timing == "subtitle"


def _proofread_realign_input(cfg, proofreader, precleaner=None, collect: bool = False,
                             summary=None, progress=None):
    """Proofread the ``--realign`` input on its own timeline; return *cfg* realigning the copy.

    The corrected copy is written as ``{input stem}.proofread.srt`` (to ``output_dir``, or a
    temporary folder when nothing is written to disk) beside the review files, exactly as
    ``--proofread_input`` would write it, and the returned config points ``realign`` at it
    with proofreading switched off so the realigned result is not proofread twice. No text
    cleaning here: the realign run cleans (or, for a finished subtitle, deliberately does
    not clean) the whole file afterwards -- only the basic cleaning (*precleaner*), as for any
    finished subtitle going into proofreading.
    """
    import tempfile
    from cantocaptions_ai.utils.output import WriteSRT
    from cantocaptions_ai.utils.subtitles import read_subtitle_cues

    src = cfg.realign
    stem = os.path.splitext(os.path.basename(src))[0]
    segments = [{"start": c.start, "end": c.end, "text": c.text} for c in read_subtitle_cues(src)]
    result = {"segments": segments, "language": cfg.language}
    logger.info("Proofreading the realign input before realigning it (%d cues): %s",
                len(segments), src)
    out_dir = tempfile.mkdtemp(prefix="cantocaptions-proofread-") if collect else cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)
    marks = _proofread(proofreader, stem, result, _load_reference(cfg), None, None,
                       cfg.debug_dir, cfg.load_debug_dir, output_dir=out_dir,
                       precleaner=precleaner, summary=summary, progress=progress)
    WriteSRT(out_dir)(result, f"{stem}.proofread", build_writer_args(cfg))
    _write_bookmarks(out_dir, f"{stem}.proofread", "srt", marks)
    path = os.path.join(out_dir, f"{stem}.proofread.srt")
    logger.info("Realigning the proofread copy: %s", path)
    # The copy's text and its marks, for _merge_and_write to carry onto the realigned output.
    carried = ([s["text"] for s in result["segments"]], marks) if marks else None
    return dataclasses.replace(cfg, realign=path, proofread="none"), carried


def _build_precleaner(cfg, pack, profile):
    """The language's basic cleaner for text going into proofreading, or None.

    Only for text this run does not clean in full (an existing subtitle under
    ``--proofread_input`` or ``--realign``): punctuation and the standard character variants,
    from ``CleaningSpec.basic_manifest``. Off with ``proofread_preclean = False``, with
    ``--no_clean_text``, and for a language with no basic manifest.
    """
    spec = pack.cleaning
    if (cfg.proofread == "none" or not cfg.proofread_preclean or cfg.no_clean_text
            or spec is None or not spec.basic_manifest):
        return None
    from cantocaptions_ai.cleaning import SubtitleCleaner
    return SubtitleCleaner(
        rules_dir=cfg.clean_rules_dir or spec.rules_dir, manifest=spec.basic_manifest,
        builtin_steps=spec.builtin_steps(), noise_tokens=spec.noise_tokens,
        layout=profile.script.layout, max_line_count=None,
    )


def _preclean_text(precleaner, text: str) -> str:
    """*text* through the basic cleaner one display line at a time, so a cue's line breaks
    (often a change of speaker) survive. A line it would empty is kept as it was."""
    lines = []
    for line in str(text).split("\n"):
        cleaned = precleaner.clean(line)
        lines.append(cleaned if cleaned.strip() else line)
    return "\n".join(lines)


def _describe_run(cfg, ctx, stage_list, proofreader, cleaner, layout, precleaner,
                  proofread_first: bool) -> str:
    """The whole run in one line, before anything is computed: the stages, and what happens
    around them -- proofreading on whichever side of realign it runs, cleaning, writing."""
    from cantocaptions_ai.pipeline.stages import describe_plan

    def proofread_step(what: str) -> str:
        s = proofreader.settings
        if not cfg.reference_subtitle:
            ref = "no reference"
        elif cfg.realign:
            ref = f"reference {os.path.basename(cfg.reference_subtitle)}, {cfg.reference_timing}-timed"
        else:
            ref = f"reference {os.path.basename(cfg.reference_subtitle)}"
        dry = ", DRY RUN" if s.dry_run else ""
        return f"Proofread {what} [{s.provider} {s.model}, {s.effort}; {ref}{dry}]"

    steps = []
    if proofread_first:
        if precleaner is not None:
            steps.append("Basic cleaning (realign input)")
        steps.append(proofread_step(f"realign input {os.path.basename(cfg.realign)}"))
    steps.append(describe_plan(ctx, stage_list))
    steps.append("Cue assembly")
    if cleaner is not None:
        steps.append("Text cleaning")
    elif layout is not None:
        steps.append("Line layout")
    if proofreader is not None and not proofread_first:
        if precleaner is not None:
            steps.append("Basic cleaning")
        steps.append(proofread_step("output"))
    writes = f"Write {cfg.output_format}"
    if proofreader is not None and cfg.output_format in ("srt", "vtt", "all"):
        writes += " + Subtitle Edit bookmarks"
    steps.append(writes)
    return " → ".join(steps)


def _build_cleaner(cfg, pack, profile):
    """``(cleaner, layout)`` for *cfg*: at most one is set, and both are None when neither applies.

    Constructed eagerly by callers so bad rule files fail before any model inference.
    """
    if not cfg.no_clean_text:
        from cantocaptions_ai.cleaning import SubtitleCleaner
        spec = pack.cleaning  # validate_config guarantees one when cleaning is on
        return SubtitleCleaner(
            rules_dir=cfg.clean_rules_dir or spec.rules_dir,
            line_max_length=cfg.max_line_width,
            max_line_count=cfg.max_line_count,
            # How much cleaning the text needs depends on how the model writes it, so the
            # step manifest comes from the model's conventions like every other output one.
            manifest=profile.cleaning.manifest,
            builtin_steps=spec.builtin_steps(),
            noise_tokens=spec.noise_tokens,
            layout=profile.script.layout,
        ), None
    if cfg.max_line_width:
        # --no_clean_text turns off the rewriting, not the line limits the user also set:
        # line breaking lives in the cleaner's manifest, so it is applied on its own here.
        from cantocaptions_ai.cleaning.layout import linebreak_step
        return None, linebreak_step(cfg.max_line_width, cfg.max_line_count, profile.script.layout)
    return None, None


def proofread_task(args: dict, parser: argparse.ArgumentParser, inputs: List[str]) -> None:
    """CLI adapter for ``--proofread_input``: proofread existing subtitles, no audio and no ASR.

    The same stage as at the end of a normal run, on cues read from disk instead of
    produced: each file is one request (or ``proofread_chunk_cues`` chunks), with
    ``--reference_subtitle`` interleaved when one is given. Validation errors become
    ``parser.error`` like :func:`transcribe_task`'s.
    """
    from cantocaptions_ai.errors import ConfigError
    from cantocaptions_ai.pipeline.config import PipelineConfig
    from cantocaptions_ai.utils.output import output_names
    from cantocaptions_ai.utils.subtitles import subtitle_has_timings

    cfg = PipelineConfig.from_args(args)
    try:
        validate_config(cfg)
        if cfg.proofread == "none":
            raise ConfigError("proofread_input needs a provider: --proofread gemini or "
                              "--proofread anthropic (or proofread = ... in user.cfg)")
        for path in inputs:
            if not os.path.isfile(path):
                raise ConfigError(f"proofread_input file not found: {path}")
            if not subtitle_has_timings(path):
                raise ConfigError(f"proofread_input must be a timed subtitle (SRT or WebVTT): {path}")
        if cfg.reference_subtitle and len(inputs) > 1:
            raise ConfigError("reference_subtitle is one file's reference; pass one "
                              "proofread_input with it, or leave it out")
        names = output_names(inputs)
    except ConfigError as e:
        parser.error(str(e))
    proofread_files(inputs, cfg, names)


def proofread_files(paths: List[str], cfg, names: Optional[Dict[str, str]] = None) -> List[dict]:
    """Proofread finished subtitle files and write ``{name}.proofread.{ext}`` for each.

    Cue timings and every cue the model leaves alone come out exactly as they went in;
    edited cues go back through text cleaning (unless ``no_clean_text``), as in a normal run.
    The input is never overwritten: the output name carries ``.proofread`` even when
    ``output_dir`` is the input's own folder. Returns ``{'path', 'name', 'result'}`` per file.
    Assumes *cfg* has passed :func:`validate_config`.
    """
    from cantocaptions_ai.languages import get_language_pack
    from cantocaptions_ai.pipeline.proofread import load_proofreader
    from cantocaptions_ai.utils.output import output_names
    from cantocaptions_ai.utils.subtitles import read_subtitle_cues

    pack = get_language_pack(cfg.language)
    profile = pack.resolve(cfg.model)
    cleaner, layout = _build_cleaner(cfg, pack, profile)
    precleaner = _build_precleaner(cfg, pack, profile)
    proofreader = load_proofreader(cfg, pack, profile)
    names = names or output_names(paths)
    summary = TranscriptionSummary(enabled=cfg.print_progress, title="Proofreading complete")
    started = time.perf_counter()

    reference_cues = _load_reference(cfg)
    if reference_cues is None:
        logger.info("No reference subtitle: proofreading from the text alone")

    os.makedirs(cfg.output_dir, exist_ok=True)
    writer = get_writer(cfg.output_format, cfg.output_dir)
    writer_args = build_writer_args(cfg)
    done = []
    for path in paths:
        name = names[path]
        # Line breaks kept: inside a cue one usually marks a change of speaker.
        segments = [{"start": c.start, "end": c.end, "text": c.text}
                    for c in read_subtitle_cues(path)]
        result = {"segments": segments, "language": cfg.language}
        logger.info("Proofreading %s (%d cues)", path, len(segments))
        marks = _proofread(proofreader, name, result, reference_cues, cleaner, layout,
                           cfg.debug_dir, cfg.load_debug_dir, output_dir=cfg.output_dir,
                           precleaner=precleaner, summary=summary)
        if not cfg.proofread_dry_run:
            writer(result, f"{name}.proofread", writer_args)
            _write_bookmarks(cfg.output_dir, f"{name}.proofread", cfg.output_format, marks)
        done.append({"path": path, "name": name, "result": result})
    summary.print_summary(process_elapsed=time.perf_counter() - started)
    return done


def _preclean(precleaner, name: str, segments: List[dict], review_dir: Optional[str]) -> None:
    """Basic cleaning on *segments* in place, before they are proofread (see _build_precleaner).

    What it changed is logged and, with somewhere to put it, written as ``precleaned.srt``
    (new over ``[was]`` old) beside the proofreading review files, so a rewrite of the
    user's own text is never silent.
    """
    changed = []
    for seg in segments:
        before = str(seg.get("text", ""))
        after = _preclean_text(precleaner, before)
        if after != before:
            seg["text"] = after
            changed.append((float(seg["start"]), float(seg["end"]), after, before))
    logger.info("Basic cleaning before proofreading %s: %d of %d cue(s) changed",
                name, len(changed), len(segments))
    if changed and review_dir:
        from cantocaptions_ai.pipeline.proofread.review import write_changes_srt
        os.makedirs(review_dir, exist_ok=True)
        write_changes_srt(os.path.join(review_dir, "precleaned.srt"), changed)


def _proofread(proofreader, name, result, reference_cues, cleaner, layout,
               debug_dir, load_debug_dir, output_dir: Optional[str] = None,
               precleaner=None, summary=None, progress=None) -> list:
    """Stage 9: proofread one file's finished cues in place. A failure costs only itself.

    A request that fails or would exceed ``proofread_max_cost`` leaves the file exactly as
    the rest of the pipeline wrote it, with a warning: a finished subtitle is worth more than
    a proofreading attempt. Edited cues go back through the cleaner (or the line layout), so
    a correction is held to the same rules as everything else, and one the cleaner reduces to
    noise is dropped like any other.

    The model's answers and the review files (``changes.srt``, ``flags.srt``,
    ``summary.json``) are kept whatever the settings, since they were paid for: in the debug
    stage dir when there is one, else in ``{output_dir}/{name}.proofread/``.

    Returns Subtitle Edit bookmarks for the cues as they will be written -- ``(0-based index,
    note)`` for each changed cue (``[was] <old text>``) and each flagged one -- for
    :func:`_write_bookmarks`; empty when nothing was proofread.
    """
    from cantocaptions_ai.pipeline.proofread.providers import ProviderError
    from cantocaptions_ai.pipeline.proofread.review import bookmark_marks

    segments = result["segments"]
    save_dir = None
    if not debug_dir and output_dir:
        save_dir = os.path.join(output_dir, *f"{name}.proofread".split("/"))
    from contextlib import nullcontext
    # A row in the end-of-run duration table, like any stage (no VRAM: it is a network call).
    timer = (StageTimer("Proofreading", summary, progress=progress, track_vram=False)
             if summary is not None else nullcontext())
    try:
        with timer:
            if precleaner is not None:
                _preclean(precleaner, name, segments, save_dir or (
                    _stage_dir(name, "proofread", debug_dir) if debug_dir else None))
            done = proofreader.run(name, segments, reference=reference_cues or [],
                                   debug_dir=debug_dir, load_debug_dir=load_debug_dir,
                                   save_dir=save_dir)
    except ProviderError as e:
        spent = e.usage.get("cost_usd") if getattr(e, "usage", None) else None
        if summary is not None and spent:
            summary.add_amount("Proofreading cost", spent, "${:.3f}")
        logger.warning("Proofreading skipped for %s: %s%s", name, e,
                       f" (the attempt cost ${spent:.3f})" if spent else "")
        return []
    # Notes keyed by the segment object, so they survive the re-clean dropping a cue below.
    notes: dict = {}
    for cid, old in done.before.items():
        notes.setdefault(id(segments[cid - 1]), []).append(
            "[was] " + old.replace("\n", " / "))
    for flag in done.flags:
        if 1 <= flag.get("id", 0) <= len(segments):
            notes.setdefault(id(segments[flag["id"] - 1]), []).append(
                "[flag] " + str(flag.get("note", "")))
    edited = {e["id"] for e in done.edits}
    kept = []
    for k, segment in enumerate(segments, 1):
        if k in edited:
            if cleaner is not None or layout is not None:
                text = cleaner.clean(segment["text"]) if cleaner is not None else layout(segment["text"])
                if cleaner is not None and cleaner.is_noise(text):
                    continue
                segment["text"] = text
            # An edit that empties a cue removes it: written, it would be a blank cue that
            # some editors skip, putting every later bookmark one line out.
            if not str(segment["text"]).strip():
                continue
        kept.append(segment)
    result["segments"] = kept
    cost = done.usage.get("cost_usd")
    if summary is not None and cost:
        summary.add_amount("Proofreading cost", cost, "${:.3f}")
    logger.info(
        "Proofreading %s: %d cue(s) edited, %d flagged for review%s%s", name, len(edited),
        len(done.flags), f", {done.replayed} answer(s) replayed" if done.replayed else "",
        f", ${cost:.3f}" if cost else "")
    return bookmark_marks(result["segments"], notes)


def _write_bookmarks(output_dir: str, name: str, output_format: str, marks: list) -> None:
    """Subtitle Edit bookmarks beside each SRT/VTT just written for *name* (see proofread/review.py)."""
    from cantocaptions_ai.pipeline.proofread.review import write_bookmarks
    if not marks:
        return
    exts = ("srt", "vtt") if output_format == "all" else (output_format,)
    for ext in exts:
        if ext not in ("srt", "vtt"):
            continue
        path = write_bookmarks(os.path.join(output_dir, *name.split("/")) + "." + ext, marks)
        if path:
            logger.info("Subtitle Edit bookmarks for %d proofread cue(s): %s", len(marks), path)


def _validate_language_support(cfg) -> None:
    """Refuse a configuration that would run another language's parts on this one.

    The language pack (``languages/``) says what the language has. A fully supported one
    (a default model and cleaning rules: yue) runs end to end. Any other runs as a raw
    pipeline -- an ASR model that speaks it, an align model for it, no text cleaning -- and
    each missing piece is named in the error.
    """
    from cantocaptions_ai.errors import ConfigError
    from cantocaptions_ai.languages import LANGUAGE_PACKS, get_language_pack
    from cantocaptions_ai.pipeline.model_profiles import get_model_profile

    language = cfg.language
    pack = get_language_pack(language)
    if cfg.ensemble_model != "none" and pack.ensemble_model is None:
        raise ConfigError(f"ensemble_model has no model for language '{language}' (available: "
                          f"{', '.join(sorted(c for c, p in LANGUAGE_PACKS.items() if p.ensemble_model))})")
    if cfg.llm_correction and pack.correction_prompts is None:
        raise ConfigError(f"llm_correction has no prompts for language '{language}' (available: "
                          f"{', '.join(sorted(c for c, p in LANGUAGE_PACKS.items() if p.correction_prompts))})")

    needed = []
    model = pack.model_name(cfg.model)
    if model is None:
        needed.append(f"a --model that transcribes '{language}' (it has no default)")
    else:
        languages = get_model_profile(model, language).languages
        if languages is not None and language not in languages:
            needed.append(f"a --model that transcribes '{language}' ({model} is trained for "
                          f"{', '.join(sorted(languages))} only)")
    if not cfg.no_align and cfg.align_model is None and pack.default_align_model is None:
        needed.append(f"an --align_model for '{language}' (or --no_align)")
    if pack.cleaning is None and not cfg.no_clean_text:
        needed.append(f"--no_clean_text (there are no cleaning rules for '{language}')")
    if needed:
        supported = sorted(c for c, p in LANGUAGE_PACKS.items() if p.fully_supported)
        raise ConfigError(
            f"language '{language}' is not fully supported (only {', '.join(supported)}); "
            "to run it anyway, set " + "; ".join(needed)
        )
    if not pack.fully_supported:
        warnings.warn(
            f"language '{language}' runs as a raw pipeline: the ASR model's text is timed and "
            f"laid out, but not cleaned or checked against conventions for it"
        )


def validate_config(cfg) -> None:
    """Validate and normalize a PipelineConfig, raising ConfigError on bad input.

    Extracted from the former CLI-only body so library/server callers get a
    catchable exception instead of argparse's process-killing ``sys.exit``. Mutates
    ``cfg.language`` in place (lowercase + code mapping). Advisory-only issues are
    emitted as warnings, not errors.
    """
    from cantocaptions_ai.errors import ConfigError

    for option in ("speaker_embeddings", "flag_speaker_conflicts", "speaker_labels"):
        if getattr(cfg, option) and not cfg.diarize:
            warnings.warn(f"{option} has no effect without diarize")
    if cfg.min_speakers is not None and cfg.max_speakers is not None:
        if cfg.min_speakers > cfg.max_speakers:
            raise ConfigError(
                f"min_speakers ({cfg.min_speakers}) cannot exceed max_speakers ({cfg.max_speakers})"
            )
    # Checked here rather than at the point of use: diarization is stage 5, so a bad
    # threshold would otherwise only surface after ASR and alignment have already run.
    if not 0 < cfg.speaker_confidence <= 1:
        raise ConfigError(f"speaker_confidence must be in (0, 1], got {cfg.speaker_confidence}")
    if not 0 < cfg.speaker_change_threshold < 1:
        raise ConfigError(
            f"speaker_change_threshold must be in (0, 1), got {cfg.speaker_change_threshold}"
        )
    if cfg.files_per_group < 0:
        raise ConfigError(
            f"files_per_group must be 0 (all inputs at once) or more, got {cfg.files_per_group}"
        )
    if cfg.max_cue_duration and cfg.max_cue_duration < 2 * cfg.min_cue_duration:
        raise ConfigError(
            f"max_cue_duration must be 0 (no cap) or at least twice min_cue_duration "
            f"({cfg.min_cue_duration}), got {cfg.max_cue_duration}"
        )
    if not 0 < cfg.speaker_conflict_share <= 1:
        raise ConfigError(
            f"speaker_conflict_share must be in (0, 1], got {cfg.speaker_conflict_share}"
        )
    # The ensemble's only consumer is LLM correction; without it the second ASR pass
    # would run to completion and its output would be thrown away.
    if cfg.ensemble_model != "none" and not cfg.llm_correction:
        raise ConfigError("ensemble_model requires llm_correction")
    if cfg.reference_subtitle and not (cfg.llm_correction or cfg.asr_context
                                       or cfg.proofread != "none"):
        raise ConfigError("reference_subtitle requires llm_correction, asr_context or proofread")
    if cfg.reference_correction_semantic and not cfg.reference_subtitle:
        warnings.warn("reference_correction_semantic has no effect without reference_subtitle")
    if cfg.asr_context and not cfg.reference_subtitle:
        raise ConfigError("asr_context requires reference_subtitle")
    if cfg.asr_context:
        # Context biasing is a prompt feature of the model family; only some backends have
        # one. Checked only when asked for, so a plain run never reads a model config here.
        from cantocaptions_ai.pipeline.asr import backend_for
        backend = backend_for(cfg.model, cfg.language, cache_dir=cfg.model_dir,
                              local_files_only=cfg.model_cache_only)
        if not backend.supports_context:
            raise ConfigError(
                f"asr_context is not supported by the {backend.name} ASR backend "
                "(only qwen3-asr models take a context prompt)"
            )
    if cfg.asr_context and cfg.asr_context_template not in CONTEXT_TEMPLATES:
        raise ConfigError(
            f"asr_context_template must be one of {sorted(CONTEXT_TEMPLATES)}, "
            f"got {cfg.asr_context_template!r}"
        )
    if cfg.asr_context and cfg.asr_context_scope not in ("all", "expanded"):
        raise ConfigError(
            f"asr_context_scope must be 'all' or 'expanded', got {cfg.asr_context_scope!r}"
        )
    if (
        cfg.asr_context
        and cfg.asr_context_scope == "expanded"
        and not cfg.asr_context_vad_expand
    ):
        raise ConfigError(
            "asr_context_scope 'expanded' requires asr_context_vad_expand: with no "
            "expansion there is no recovered audio to prompt over, so no segment would "
            "get a context"
        )
    if cfg.asr_context and cfg.asr_context_neighbours < 0:
        raise ConfigError(
            f"asr_context_neighbours must be >= 0, got {cfg.asr_context_neighbours}"
        )
    if cfg.reference_offset and not cfg.reference_subtitle:
        warnings.warn("reference_offset has no effect without reference_subtitle")
    if cfg.reference_timing not in (None, "subtitle", "media"):
        raise ConfigError(f"reference_timing must be 'subtitle' or 'media', got {cfg.reference_timing!r}")
    if cfg.reference_timing == "subtitle" and not (cfg.realign and cfg.reference_subtitle):
        warnings.warn("reference_timing 'subtitle' has no effect without realign and "
                      "reference_subtitle (without realign, a reference always follows the "
                      "media's timeline)")
    if cfg.realign and cfg.reference_subtitle and cfg.proofread != "none":
        from cantocaptions_ai.utils.subtitles import subtitle_has_timings
        if cfg.reference_timing is None:
            raise ConfigError(
                "realign with proofread and reference_subtitle needs reference_timing "
                "(it was set to None): "
                "'subtitle' if the reference is timed like the --realign input (proofreading then "
                "runs first, on the input), or 'media' if it is timed to this audio/video "
                "(proofreading then runs after realign, on the realigned cues)")
        if cfg.reference_timing == "subtitle" and os.path.isfile(cfg.realign) \
                and not subtitle_has_timings(cfg.realign):
            raise ConfigError(
                "reference_timing 'subtitle' needs a --realign input with timings to pair the "
                f"reference with, and {cfg.realign} is a bare transcript; use 'media'")
    if cfg.realign:
        from cantocaptions_ai.pipeline.realign import REALIGN_MODES, resolve_realign_mode
        from cantocaptions_ai.utils.subtitles import subtitle_has_timings
        if not os.path.isfile(cfg.realign):
            raise ConfigError(f"realign transcript not found: {cfg.realign}")
        if cfg.realign_anchor not in ("acoustic", "asr"):
            raise ConfigError(
                f"realign_anchor must be 'acoustic' or 'asr', got {cfg.realign_anchor!r}"
            )
        if cfg.realign_mode not in ("auto",) + REALIGN_MODES:
            raise ConfigError(
                f"realign_mode must be one of {('auto',) + REALIGN_MODES}, "
                f"got {cfg.realign_mode!r}"
            )
        if cfg.realign_cut_policy not in ("drop", "keep"):
            raise ConfigError(
                f"realign_cut_policy must be 'drop' or 'keep', got {cfg.realign_cut_policy!r}"
            )
        if not 0.0 < cfg.realign_max_scale < 1.0:
            raise ConfigError(
                f"realign_max_scale must be between 0 and 1, got {cfg.realign_max_scale}"
            )
        if cfg.realign_adjust_tolerance <= 0:
            raise ConfigError(
                f"realign_adjust_tolerance must be positive, got "
                f"{cfg.realign_adjust_tolerance}"
            )
        mode = resolve_realign_mode(cfg.realign, cfg.realign_mode)
        if mode in ("sync", "adjust") and not subtitle_has_timings(cfg.realign):
            raise ConfigError(
                f"realign_mode {mode!r} maps a subtitle's existing timings onto this "
                f"recording, but {cfg.realign} carries none. Use realign_mode 'transcript' "
                "to place every line from scratch."
            )
        # Mode sync never forced-aligns -- the transform places every cue -- so no_align is
        # simply what it already does, rather than a contradiction.
        if cfg.no_align and mode != "sync":
            raise ConfigError(
                "realign is forced alignment; with no_align there is nothing left for it "
                "to do and every cue would keep the coarse search's timing"
            )
        if cfg.realign_commit_margin >= cfg.realign_window:
            raise ConfigError(
                f"realign_commit_margin ({cfg.realign_commit_margin}) must be smaller than "
                f"realign_window ({cfg.realign_window}), or no line is ever committed"
            )
        if cfg.asr_context and cfg.realign_anchor == "acoustic":
            raise ConfigError(
                "asr_context has no effect with realign_anchor 'acoustic', which skips ASR "
                "entirely; use realign_anchor 'asr' if you want the ASR pass"
            )
    elif cfg.realign_anchor != "acoustic":
        warnings.warn("realign_anchor has no effect without realign")
    if cfg.audio_downmix not in ("mix", "center"):
        raise ConfigError(
            f"audio_downmix must be 'mix' or 'center', got {cfg.audio_downmix!r}"
        )
    # The "none" template is the VAD-expansion-only control, so it is the one template
    # that does nothing at all once expansion is also off.
    if (
        cfg.asr_context
        and cfg.asr_context_template == "none"
        and not cfg.asr_context_vad_expand
        and not cfg.llm_correction
    ):
        raise ConfigError(
            "asr_context_template 'none' with asr_context_vad_expand off uses the "
            "reference subtitle for nothing; enable asr_context_vad_expand (the "
            "control this template exists for) or pick another template"
        )

    # Required: every stage after ASR (alignment, cleaning, cue assembly) is chosen by
    # language, so there is nothing sensible to fall back to if it is left unset.
    if cfg.language is None:
        raise ConfigError("language is required (e.g. 'yue')")
    cfg.language = cfg.language.lower()
    if cfg.language not in LANGUAGES:
        if cfg.language in TO_LANGUAGE_CODE:
            cfg.language = TO_LANGUAGE_CODE[cfg.language]
        else:
            raise ConfigError(f"Unsupported language: {cfg.language}")
    _validate_language_support(cfg)

    # Settings left unset that the language decides (config.LANGUAGE_DEFAULTED). Line
    # breaking is a text step (cleaning/layout.py), so it applies under --no_align too; the
    # old rule refusing line limits there dated from writers that split on word timings.
    from cantocaptions_ai.languages import get_language_pack
    from cantocaptions_ai.pipeline.config import LANGUAGE_DEFAULTED
    profile = get_language_pack(cfg.language).resolve(cfg.model)
    for option, from_profile in LANGUAGE_DEFAULTED.items():
        if getattr(cfg, option) is None:
            setattr(cfg, option, from_profile(profile))

    _validate_proofreading(cfg)


def _validate_proofreading(cfg) -> None:
    """Fail at startup, not after an hour of ASR, if proofreading cannot run.

    Off (the default) checks nothing: the stage then never imports a provider SDK or reads
    the environment. A dry run needs neither SDK nor key, since it sends nothing.
    """
    from cantocaptions_ai.errors import ConfigError

    if cfg.proofread == "none":
        return
    from cantocaptions_ai.languages import get_language_pack
    from cantocaptions_ai.pipeline.proofread.providers import PROVIDERS, preflight
    if cfg.proofread not in PROVIDERS:
        raise ConfigError(f"proofread must be 'none' or one of {PROVIDERS}, got {cfg.proofread!r}")
    try:
        get_language_pack(cfg.language).standard_for(cfg.proofread_standard)
    except KeyError as e:
        raise ConfigError(str(e.args[0])) from None
    for option in ("proofread_conventions", "proofread_prompt", "proofread_context"):
        path = getattr(cfg, option)
        if path and not os.path.isfile(path):
            raise ConfigError(f"{option} file not found: {path}")
    if cfg.proofread_chunk_context < 0:
        raise ConfigError(f"proofread_chunk_context must be >= 0, got {cfg.proofread_chunk_context}")
    if cfg.proofread_parallel < 1:
        raise ConfigError(f"proofread_parallel must be >= 1, got {cfg.proofread_parallel}")
    if cfg.proofread_particles not in ("protect", "allow"):
        raise ConfigError("proofread_particles must be protect or allow, got "
                          f"{cfg.proofread_particles!r}")
    if cfg.proofread_min_confidence not in ("low", "medium", "high"):
        raise ConfigError("proofread_min_confidence must be low, medium or high, got "
                          f"{cfg.proofread_min_confidence!r}")
    if cfg.proofread_max_cost is not None and cfg.proofread_max_cost <= 0:
        raise ConfigError(f"proofread_max_cost must be positive or None, got {cfg.proofread_max_cost}")
    if cfg.proofread_chunk_cues < 0:
        raise ConfigError(f"proofread_chunk_cues must be 0 (whole file) or more, got "
                          f"{cfg.proofread_chunk_cues}")
    if cfg.proofread_timeout <= 0:
        raise ConfigError(f"proofread_timeout must be positive, got {cfg.proofread_timeout}")
    if not cfg.proofread_dry_run:
        problem = preflight(cfg.proofread)
        if problem:
            raise ConfigError(f"proofread {cfg.proofread}: {problem}")


def _prepare_clips(audio_paths: List[str], cfg):
    """Apply an audio_start/audio_end clip by extracting a clipped WAV per input.

    Returns ``(paths, display_paths, temp_files)``. When no clip is configured the
    inputs pass through unchanged. Otherwise each input is written to a temporary
    16 kHz mono WAV (with its Cantonese track already selected) so every downstream
    stage — including those that reload the file by path — sees the clipped audio;
    ``display_paths`` maps each temp path back to the original for output naming, and
    ``temp_files`` must be cleaned up by the caller.
    """
    if cfg.audio_start is None and cfg.audio_end is None:
        return audio_paths, {}, []
    import tempfile
    from cantocaptions_ai.utils.audio import extract_clip_to_wav
    new_paths: List[str] = []
    display: dict = {}
    temps: List[str] = []
    for p in audio_paths:
        track = _select_audio_track(p, cfg.language, cfg.audio_track)
        fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="cantoclip_")
        os.close(fd)
        extract_clip_to_wav(
            p, tmp, audio_start=cfg.audio_start, audio_end=cfg.audio_end, audio_track=track
        )
        new_paths.append(tmp)
        display[tmp] = p
        temps.append(tmp)
    return new_paths, display, temps


def _cleanup_temp_files(paths: List[str]) -> None:
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            logger.warning("Could not remove temp clip file: %s", p)


def transcribe_task(args: dict, parser: argparse.ArgumentParser, input_dir: Optional[str] = None):
    """CLI adapter: build a PipelineConfig from parsed args and run the pipeline.

    Thin wrapper over :func:`_execute_pipeline` that preserves the CLI contract —
    validation errors become ``parser.error(...)`` (exit 2) and results are written
    to ``cfg.output_dir``. Library/server callers should use
    ``cantocaptions_ai.service.run_pipeline`` instead.

    ``input_dir`` is the --input_dir the paths were discovered under, if any; outputs
    then mirror its subfolders (see ``utils.output.output_names``).
    """
    from cantocaptions_ai.pipeline.config import PipelineConfig
    from cantocaptions_ai.errors import ConfigError
    from cantocaptions_ai.utils.output import output_names

    audio_paths = args.pop("audio")
    cfg = PipelineConfig.from_args(args)
    try:
        validate_config(cfg)
        names = output_names(audio_paths, input_dir)
    except ConfigError as e:
        parser.error(str(e))

    paths, display_paths, temp_files = _prepare_clips(audio_paths, cfg)
    try:
        _execute_pipeline(
            paths, cfg, collect=False,
            audio_start_offset=cfg.audio_start or 0.0,
            display_paths=display_paths,
            names=names,
        )
    finally:
        _cleanup_temp_files(temp_files)


def _execute_pipeline(
    audio_paths: List[str],
    cfg,
    *,
    progress: "Optional[ProgressSink]" = None,
    collect: bool = False,
    audio_start_offset: float = 0.0,
    display_paths: Optional[dict] = None,
    vad_model=None,
    names: Optional[Dict[str, str]] = None,
    stages: Optional[Callable] = None,
) -> List[ProcessingItem]:
    """Run all pipeline stages for *audio_paths* under *cfg*.

    The stages are ``pipeline.stages.build_stages``'s list for this config, run in order,
    then cue assembly and writing. ``stages``, a ``(ctx, default_stages) -> stages`` callable,
    lets a caller insert, replace or drop a stage without editing this function.

    Assumes *cfg* has already passed :func:`validate_config`, and that any audio clip
    has already been applied (paths point at clipped temp files; see
    :func:`_prepare_clips`). With ``collect=True`` the final per-item results are
    returned and nothing is written to disk; otherwise results are written to
    ``cfg.output_dir``. ``progress`` receives stage/progress events out-of-band.

    ``names`` maps each *original* input path (before any clip substitution) to the name
    its outputs and debug checkpoints use; by default, each file's stem. Raises
    ConfigError if two inputs would share a name.
    """
    from huggingface_hub.utils.tqdm import disable_progress_bars

    # HF Hub's own tqdm download bars race StageTimer's spinner over the same
    # terminal line (both are \r-driven redraw loops); silencing them means our
    # explicit "Downloading %r..." log lines are the only download-progress signal,
    # so a slow first-run download never looks like a stalled/hung stage.
    disable_progress_bars()

    # One name per input, fixed before any stage runs: it keys every debug checkpoint and
    # output file, so a clip's temp path must resolve to its original's name.
    from cantocaptions_ai.utils.output import output_names
    originals = [(display_paths or {}).get(p, p) for p in audio_paths]
    if names is None:
        names = output_names(originals)
    name_of = {p: names[orig] for p, orig in zip(audio_paths, originals)}

    # The settings each stage's debug checkpoint must have been made with to be replayed;
    # carried on every item so the stages can check what they read (utils/checkpoints.py).
    from cantocaptions_ai.utils.checkpoints import checkpoint_settings
    checkpoints = checkpoint_settings(cfg)

    align_language = cfg.language

    # The language pack, resolved for the ASR model, drives the downstream path: post-ASR
    # text normalization (applied inside the ASR backend), the alignment particle
    # spot-checks, the punctuation and script used for sentence splitting / cue assembly /
    # line layout, and the cleaning rules. A model the pack has no conventions for gets
    # all-default (no-op) ones. See languages/base.py.
    from cantocaptions_ai.languages import get_language_pack
    pack = get_language_pack(cfg.language)
    profile = pack.resolve(cfg.model)
    align_model_name = cfg.align_model or pack.default_align_model

    qwen_threads = torch.get_num_threads()
    if cfg.threads > 0:
        torch.set_num_threads(cfg.threads)
        qwen_threads = cfg.threads

    if collect:
        writer = None
    else:
        os.makedirs(cfg.output_dir, exist_ok=True)
        writer = get_writer(cfg.output_format, cfg.output_dir)
    writer_args = build_writer_args(cfg)

    realign_mode = None
    if cfg.realign:
        from cantocaptions_ai.pipeline.realign import resolve_realign_mode
        realign_mode = resolve_realign_mode(cfg.realign, cfg.realign_mode)
        if cfg.realign_mode == "auto":
            logger.info(
                "realign_mode auto resolved to %r for: %s", realign_mode, cfg.realign,
            )

    # Text cleaning runs on the final merged segments just before writing, apart from the
    # manifest's pre_align steps, which run on the ASR text before alignment (see
    # _pre_align_clean). Constructed eagerly so bad rule files fail before any model inference.
    #
    # What decides it is what the input *is*, not which feature is running. A bare transcript
    # wants the rule files as much as ASR output does, so realign_mode 'transcript' cleans; a
    # subtitle arriving with timings is already a finished subtitle whose text is the user's,
    # so 'sync' and 'adjust' leave it alone. Cleaning still only ever edits text, never the
    # timings.
    cleaner = None
    layout = None
    if realign_mode in ("sync", "adjust"):
        if not cfg.no_clean_text:
            logger.info(
                "Text cleaning skipped: --realign_mode %s preserves the subtitle's own text "
                "(it is already a finished subtitle, not a raw transcript)", realign_mode,
            )
    else:
        cleaner, layout = _build_cleaner(cfg, pack, profile)

    if cfg.load_debug_dir:
        missing = [
            ap for ap in audio_paths
            if not os.path.isdir(os.path.join(cfg.load_debug_dir, name_of[ap]))
        ]
        if missing:
            # Not fatal: files without cached data are simply (re)computed from scratch,
            # which matters for --input_dir runs where only some files were cached before.
            logger.warning(
                "No debug data under '%s' for %d of %d file(s); they will be computed from "
                "scratch: %s",
                cfg.load_debug_dir, len(missing), len(audio_paths),
                ", ".join(name_of[ap] for ap in missing),
            )

    realign_punct = None
    if cfg.realign:
        from cantocaptions_ai.pipeline.realign import realign_punctuation
        realign_punct = realign_punctuation(profile.punctuation, profile.script)

    # None unless the user turned proofreading on (the default is off): it is the one stage
    # that needs the network. Built up front so a bad standard or override file fails here.
    from cantocaptions_ai.pipeline.proofread import load_proofreader
    proofreader = load_proofreader(cfg, pack, profile)

    summary = TranscriptionSummary(enabled=cfg.print_progress)
    process_start = time.perf_counter()

    # Loaded once here because two stages want it: VAD expansion (stage 1) and ASR
    # context (stage 3), plus LLM reference correction (stage 3c) further down.
    reference_cues = _load_reference(cfg)
    if reference_cues is not None:
        if cfg.asr_context and len(audio_paths) > 1:
            warnings.warn(
                f"asr_context is using one reference subtitle for {len(audio_paths)} audio "
                "files; the cues can only be correct for one of them"
            )

    from cantocaptions_ai.pipeline.stages import (
        RunContext, build_stages, describe_plan, plan_entries, run_stages,
    )
    ctx = RunContext(
        cfg=cfg, audio_paths=list(audio_paths), name_of=name_of, checkpoints=checkpoints,
        pack=pack, profile=profile, align_model_name=align_model_name, summary=summary,
        progress=progress, display_paths=display_paths, vad_model=vad_model,
        qwen_threads=qwen_threads, reference_cues=reference_cues, realign_mode=realign_mode,
        realign_punct=realign_punct, cleaner=cleaner, layout=layout,
    )
    stage_list = build_stages(ctx)
    if stages is not None:
        stage_list = list(stages(ctx, stage_list))
    # --realign with proofreading, on an input with timings and no media-timed reference:
    # proofread the input on its own timeline first (see proofreads_before_realign). Text
    # this run will not clean in full -- that input, or a sync/adjust realign's output --
    # gets the language's basic cleaning before it is proofread.
    proofread_first = proofreader is not None and proofreads_before_realign(cfg)
    precleaner = (_build_precleaner(cfg, pack, profile)
                  if proofreader is not None and (proofread_first or cleaner is None) else None)
    logger.info("Pipeline: %s", _describe_run(cfg, ctx, stage_list, proofreader, cleaner,
                                              layout, precleaner, proofread_first))
    # A sink that wants the whole plan up front (a UI's stage list) says so with plan().
    send_plan = getattr(progress, "plan", None)
    if callable(send_plan):
        send_plan(plan_entries(ctx, stage_list))

    # Only now, once the plan is out: every stage then reads the corrected copy, and the
    # end-of-run proofreading is off so nothing is proofread twice. No checkpoint setting
    # depends on the realign path, so the swap invalidates nothing.
    carried_marks = None
    if proofread_first:
        cfg, carried_marks = _proofread_realign_input(cfg, proofreader, precleaner,
                                                      collect=collect, summary=summary,
                                                      progress=progress)
        ctx = dataclasses.replace(ctx, cfg=cfg)
        proofreader = None

    # Every stage runs over all of a group's files before the next stage starts, and each
    # file's decoded audio is held from VAD until its subtitles are written. Taken all at
    # once, a 277-episode --input_dir run held ~25 GB of audio and crashed loading the ASR
    # model on a 16 GB Windows machine. So the inputs go through in groups, each one written
    # before the next is decoded. Each group reloads the models, which costs seconds.
    group_size = cfg.files_per_group or len(audio_paths)
    groups = [audio_paths[i:i + group_size] for i in range(0, len(audio_paths), group_size)]
    results: List[ProcessingItem] = []
    for g, group in enumerate(groups, 1):
        if len(groups) > 1:
            first = (g - 1) * group_size + 1
            logger.info(
                "File group %d of %d: inputs %d-%d of %d",
                g, len(groups), first, first + len(group) - 1, len(audio_paths),
            )
        group_ctx = dataclasses.replace(ctx, audio_paths=list(group))
        items: List[dict] = [
            {'audio_path': p, 'name': name_of[p], 'checkpoints': checkpoints} for p in group
        ]
        items = run_stages(group_ctx, stage_list, items)

        # Write and/or collect final results
        results += _merge_and_write(
            items, writer, align_language, cfg.align_merge_distance, cfg.align_padding,
            writer_args, cleaner=cleaner, layout=layout, debug_dir=cfg.debug_dir,
            punctuation=profile.punctuation, segmentation=profile.segmentation,
            script=profile.script,
            # Mode sync promises that the subtitle's own proportions survive the round trip,
            # and the duration floor (pass D) would quietly break that by stretching any cue
            # the transform made shorter than min_cue_duration. Zero turns passes B-D off,
            # which is right here for the same reason merge is: every cue's span is already
            # a decision somebody made, not an artefact of over-splitting.
            min_cue_duration=0.0 if realign_mode == "sync" else cfg.min_cue_duration,
            merge_gap=cfg.merge_gap, max_line_width=cfg.max_line_width,
            max_line_count=cfg.max_line_count,
            # Under --realign the cue boundaries came from the transcript's own line breaks
            # and are not an artifact to be undone, so the two passes that join cues are
            # off; the noise drop and the duration floor still run.
            merge=not cfg.realign,
            max_cue_duration=cfg.max_cue_duration,
            order_cues=bool(cfg.realign),
            collect=collect, audio_start_offset=audio_start_offset, display_paths=display_paths,
            proofreader=proofreader, reference_cues=reference_cues, output_dir=cfg.output_dir,
            output_format=cfg.output_format, carried_marks=carried_marks,
            precleaner=precleaner, summary=summary, progress=progress,
            load_debug_dir=cfg.load_debug_dir,
        )
        del items

    summary.print_summary(process_elapsed=time.perf_counter() - process_start)
    return results

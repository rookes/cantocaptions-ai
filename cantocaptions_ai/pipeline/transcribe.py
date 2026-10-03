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
from cantocaptions_ai.utils.log_utils import ProgressSink, TranscriptionSummary, get_logger
from cantocaptions_ai.text_profiles import (
    DEFAULT_PUNCTUATION,
    DEFAULT_SCRIPT,
    DEFAULT_SEGMENTATION,
)
from cantocaptions_ai.pipeline.reference_context import CONTEXT_TEMPLATES
from cantocaptions_ai.pipeline.segmentation import assemble_cues
from cantocaptions_ai.utils.debug import (
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
) -> List[ProcessingItem]:
    """Assemble cues, clean, offset, then write (unless writer is None) and/or return results.

    Output files and debug checkpoints are named by ``item['name']`` (see
    ``utils.output.output_names``). ``display_paths`` maps a (possibly clip-substituted
    temp) audio path back to the original path, so returned results reference the source
    file, not the temp clip. ``collect`` returns the final per-item results for in-memory
    (server) use; ``writer`` is None when nothing should be written to disk.

    ``layout`` (text -> text) breaks lines when there is no ``cleaner`` to do it; with a
    cleaner, line breaking is one of its own manifest steps.
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

        if writer is not None:
            writer(result, name, writer_args)
        if collect:
            finalized.append({'audio_path': audio_path, 'name': name, 'result': result})
    return finalized


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
    if cfg.reference_subtitle and not (cfg.llm_correction or cfg.asr_context):
        raise ConfigError("reference_subtitle requires llm_correction or asr_context")
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
    elif not cfg.no_clean_text:
        from cantocaptions_ai.cleaning import SubtitleCleaner
        spec = pack.cleaning  # validate_config guarantees one when cleaning is on
        cleaner = SubtitleCleaner(
            rules_dir=cfg.clean_rules_dir or spec.rules_dir,
            line_max_length=cfg.max_line_width,
            max_line_count=cfg.max_line_count,
            # How much cleaning the text needs depends on how the model writes it, so the
            # step manifest comes from the model's conventions like every other output one.
            manifest=profile.cleaning.manifest,
            builtin_steps=spec.builtin_steps(),
            noise_tokens=spec.noise_tokens,
            layout=profile.script.layout,
        )
    elif cfg.max_line_width:
        # --no_clean_text turns off the rewriting, not the line limits the user also set:
        # line breaking lives in the cleaner's manifest, so it is applied on its own here.
        from cantocaptions_ai.cleaning.layout import linebreak_step
        layout = linebreak_step(cfg.max_line_width, cfg.max_line_count, profile.script.layout)

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

    summary = TranscriptionSummary(enabled=cfg.print_progress)
    process_start = time.perf_counter()

    # Loaded once here because two stages want it: VAD expansion (stage 1) and ASR
    # context (stage 3), plus LLM reference correction (stage 3c) further down.
    reference_cues = None
    if cfg.reference_subtitle:
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
    logger.info("Pipeline: %s", describe_plan(ctx, stage_list))
    # A sink that wants the whole plan up front (a UI's stage list) says so with plan().
    send_plan = getattr(progress, "plan", None)
    if callable(send_plan):
        send_plan(plan_entries(ctx, stage_list))

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
        )
        del items

    summary.print_summary(process_elapsed=time.perf_counter() - process_start)
    return results

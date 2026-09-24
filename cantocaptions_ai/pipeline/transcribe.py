import argparse
import os
import time
import warnings

import numpy as np
import torch

from cantocaptions_ai.utils.audio import load_audio, SAMPLE_RATE
from cantocaptions_ai.utils.schema import AlignedTranscriptionResult, ProcessingItem, ProgressCallback, VadItem
from typing import Callable, List, Optional
from cantocaptions_ai.utils.output import LANGUAGES, TO_LANGUAGE_CODE, get_writer
from cantocaptions_ai.utils.log_utils import ProgressSink, StageTimer, TranscriptionSummary, get_logger
from cantocaptions_ai.utils.model_utils import model_scope, flush_vram, vram_stats, load_with_offline_fallback
from cantocaptions_ai.cantonese.text import (
    DEFAULT_PUNCTUATION,
    DEFAULT_SEGMENTATION,
    MAX_CHARS,
    is_removable,
)
from cantocaptions_ai.pipeline.reference_context import CONTEXT_TEMPLATES
from cantocaptions_ai.pipeline.segmentation import assemble_cues
from cantocaptions_ai.utils.debug import (
    _debug_stage_exists,
    write_precleaning_debug,
    write_speaker_assignment_debug,
)
from cantocaptions_ai.errors import ConfigError
from cantocaptions_ai.pipeline.vad import VadProcessor

logger = get_logger(__name__)

_VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.ts', '.m2ts'}


def _select_audio_track(path: str) -> int:
    """Return the 0-based audio stream index to use for *path*.

    For video files, probes with ffprobe and selects the first stream tagged
    language=yue or with a title containing "Cantonese". For audio-only files
    (or when probing returns nothing), returns 0 (ffmpeg default).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in _VIDEO_EXTENSIONS:
        return 0
    from cantocaptions_ai.utils.audio import probe_audio_tracks, select_cantonese_track
    streams = probe_audio_tracks(path)
    if not streams:
        logger.warning("No audio streams found via ffprobe for '%s'; using default track", path)
        return 0
    track = select_cantonese_track(streams)
    if track != 0:
        logger.info("Selected audio track index %d (Cantonese) for '%s'", track, path)
    return track


def _substitution_overrides(cfg):
    """Hand-curated character substitutions for the align model, or None."""
    if not getattr(cfg, "align_substitutions", None):
        return None
    from cantocaptions_ai.pipeline.align_vocab import load_substitution_overrides
    return load_substitution_overrides(cfg.align_substitutions)


def _run_alignment(
    items: List[ProcessingItem],
    align_model,
    align_metadata,
    bert_processor,
    device: str,
    align_padding: float,
    align_release: float,
    interpolate_method: str,
    return_char_alignments: bool,
    print_progress: bool,
    batch_size: int,
    progress_callback: ProgressCallback = None,
    vram_checks: bool = True,
    spotchecks=None,
    punctuation=DEFAULT_PUNCTUATION,
    split_gap: Optional[float] = None,
) -> List[ProcessingItem]:
    from cantocaptions_ai.pipeline.alignment import align
    if progress_callback is not None:
        progress_callback.set_total(sum(len(it['result']['segments']) for it in items), unit="seg")
    aligned_items = []
    for item in items:
        result = item['result']
        if align_model is not None and len(result["segments"]) > 0:
            logger.info("Performing alignment...")
            aligned_result: AlignedTranscriptionResult = align(
                result["segments"],
                align_model,
                align_metadata,
                item['vad_segments'],
                device,
                bert_processor=bert_processor,
                align_padding=align_padding,
                align_release=align_release,
                interpolate_method=interpolate_method,
                return_char_alignments=return_char_alignments,
                print_progress=print_progress,
                batch_size=batch_size,
                progress_callback=progress_callback,
                vram_checks=vram_checks,
                spotchecks=spotchecks,
                punctuation=punctuation,
                timeline=item.get('emission_timeline'),
                split_gap=split_gap,
            )
            aligned_result['language'] = result['language']
        else:
            aligned_result = result
        # Keep the rest of the carrier (notably audio_track, chosen by _select_audio_track):
        # diarization loads the audio again downstream and must not fall back to track 0.
        aligned_items.append({**item, 'result': aligned_result})
    return aligned_items


def _extract_timestamps(items: list) -> List[ProcessingItem]:
    """Build segment timings from ASR-provided per-character timestamps (used when no_align=True)."""
    extracted_items = []
    for item in items:
        result = item['result']
        segments = []
        for segment in result['segments']:
            s_start = segment['time_stamps'][0]['start']
            s_end = segment['time_stamps'][-1]['end']
            segments.append({**segment, 'start': s_start, 'end': s_end})
        # Preserve the rest of the carrier (audio_track, vad_segments): diarization
        # needs both, and --no_align routes through here instead of _run_alignment.
        extracted_items.append({**item, 'result': {**result, 'segments': segments}})
    return extracted_items


def _run_diarization(
    items: List[ProcessingItem],
    cfg,
    summary: TranscriptionSummary,
    progress: Optional[ProgressSink],
    need_diarize: bool,
) -> List[ProcessingItem]:
    """Stage 5: attach ``diarization`` to every item, from the model or the debug cache.

    Held in its own helper so the stage's load/run/free cycle reads as one unit and
    ``_execute_pipeline`` stays an outline of the pipeline rather than an implementation.
    """
    from cantocaptions_ai.pipeline.diarize import load_diarization, load_diarization_cache

    if not need_diarize:
        return load_diarization_cache(items, cfg.load_debug_dir)

    if cfg.hf_token is None:
        logger.info(
            "No --hf_token provided; %s is a gated model, so its terms must already be "
            "accepted and a token cached (huggingface-cli login) or this load will fail.",
            cfg.diarize_model,
        )
    with StageTimer("Diarization", summary, progress=progress) as stage:
        diarizer = load_with_offline_fallback(
            load_diarization,
            device=cfg.device,
            device_index=cfg.device_index,
            model_name=cfg.diarize_model,
            token=cfg.hf_token,
            model_dir=cfg.model_dir,
            scope=cfg.diarize_scope,
            min_speakers=cfg.min_speakers,
            max_speakers=cfg.max_speakers,
            return_embeddings=cfg.speaker_embeddings,
            batch_size=cfg.diarize_batch_size,
            vram_checks=cfg.vram_checks,
            vram_headroom_mb=cfg.vram_headroom_mb,
        )
        stage.mark_inference_start()
        items = diarizer.run(
            items,
            debug_dir=cfg.debug_dir,
            load_debug_dir=cfg.load_debug_dir,
            progress_callback=stage.reporter,
        )
    del diarizer
    flush_vram()
    return items


def _assign_speakers(
    items: List[ProcessingItem],
    cfg,
    debug_dir: Optional[str] = None,
) -> List[ProcessingItem]:
    """Label each aligned subsegment with a speaker, where diarization is confident enough.

    Runs outside the diarization StageTimer: it is pure arithmetic over already-computed
    turns, and it deliberately re-runs on a --load_debug_dir replay so threshold changes
    take effect without re-diarizing.
    """
    from cantocaptions_ai.pipeline.speaker_assign import (
        SpeakerAssignmentConfig,
        assign_speakers,
        format_stats,
    )

    config = SpeakerAssignmentConfig(
        min_dominant_share=cfg.speaker_confidence,
        conflict_share=cfg.speaker_conflict_share,
        flag_conflicts=cfg.flag_speaker_conflicts,
    )
    for item in items:
        diarization = item.get('diarization')
        if diarization is None:
            continue
        segments = item['result']["segments"]
        stats = assign_speakers(segments, diarization["turns"], config)
        logger.info(f"Speaker assignment: {format_stats(stats)}")
        if debug_dir is not None:
            write_speaker_assignment_debug(item['audio_path'], segments, debug_dir)
    return items


def _run_realign(
    items: List[VadItem],
    realign_path: str,
    align_model,
    align_metadata,
    bert_processor,
    device: str,
    *,
    chunk_size: float,
    window_seconds: float,
    commit_margin: float,
    min_score: float,
    mode: str = "transcript",
    cut_policy: str = "drop",
    max_scale: float = 0.25,
    adjust_tolerance: float = 2.0,
    align_padding: float = 0.04,
    sync_anchor_density: float = 3.0,
    normalize: bool = True,
    batch_size: int = 4,
    vram_checks: bool = True,
    debug_dir: Optional[str] = None,
    load_debug_dir: Optional[str] = None,
) -> List[ProcessingItem]:
    """Put a transcript on the timeline and build the next stage's input from it.

    In ``transcript`` and ``adjust`` modes, returns items whose ``vad_segments`` have been
    *replaced* by chunks re-cut onto the gaps between placed lines, and whose ``result`` holds
    one segment per chunk carrying that chunk's lines and their cue_spans. Dropping the coarse
    chunks here also frees their audio, which for a feature-length file is a few hundred MB.

    In ``sync`` mode the cues are already finished -- the transform placed them and nothing
    downstream is allowed to move them -- so the item carries them directly and is marked
    ``realign_sync`` so the caller skips alignment entirely.
    """
    from cantocaptions_ai.pipeline.align_profiles import DEFAULT_ALIGN_PROFILE
    from cantocaptions_ai.pipeline.alignment import compute_vad_emissions
    from cantocaptions_ai.pipeline.realign import (
        EmissionTimeline, assign_lines, assign_lines_adjust, assign_lines_sync,
        build_align_input, load_transcript_lines, segments_from_timings, warn_low_confidence,
    )
    from cantocaptions_ai.utils.debug import (
        load_realign_debug, write_realign_debug, write_realign_transform,
    )

    timed = mode in ("sync", "adjust")
    lines = load_transcript_lines(realign_path, keep_timings=timed, normalize=normalize)
    if not normalize:
        logger.info(
            "realign: punctuation normalization is off, so the text reaches the subtitle "
            "exactly as written. Halfwidth marks are not in the align vocabulary and will be "
            "dropped, so the pauses they stand for go unmodelled."
        )
    logger.info(f"Loaded {len(lines)} transcript line(s) from: {realign_path}")
    if not lines:
        raise ConfigError(f"realign transcript is empty: {realign_path}")
    if len(items) > 1:
        logger.warning(
            "One transcript is being realigned against %d audio files; --realign takes a "
            "transcript of a single recording.", len(items),
        )

    # Give the dictionary a token for the transcript's unknown characters *before* the
    # coarse search reads it, not after: a line whose every character is invisible to the
    # trellis can only be placed from its neighbours. Alignment runs the same call later and
    # finds nothing left to do, which is what keeps the two passes agreeing.
    repair = align_metadata.get("vocab_repair")
    if repair is not None:
        repair.augment(line.text for line in lines)

    profile = align_metadata.get("profile") or DEFAULT_ALIGN_PROFILE
    result_items: List[ProcessingItem] = []
    for item in items:
        vad_segments = item["vad_segments"]

        def compute(segments, _model=align_model):
            return compute_vad_emissions(
                segments, _model, align_metadata["type"], bert_processor, device,
                batch_size, vram_checks=vram_checks, primer=profile.primer,
            )

        # One timeline for the whole file, built here and handed to the alignment stage
        # below: the placements and the final alignment read the same emissions, so the
        # encoder runs once over the file instead of once per pass.
        timeline = EmissionTimeline(vad_segments, compute)

        common = dict(
            window_seconds=window_seconds, commit_margin=commit_margin,
            max_scale_dev=max_scale, cut_policy=cut_policy,
        )
        dropped: frozenset = frozenset()
        transform = None
        if mode == "sync":
            timings, dropped, transform, report = assign_lines_sync(
                lines, vad_segments, timeline,
                align_metadata["dictionary"], align_metadata["language"],
                align_padding=align_padding, target_anchors_per_minute=sync_anchor_density,
                **common,
            )
        elif mode == "adjust":
            timings, dropped, transform, report = assign_lines_adjust(
                lines, vad_segments, timeline,
                align_metadata["dictionary"], align_metadata["language"],
                tolerance=adjust_tolerance, **common,
            )
        else:
            # The checkpoint holds coarse placements indexed by line number, so it is only
            # reusable for the mode that produced it. sync and adjust re-fit instead, which
            # is cheap next to the encoder pass they both still need.
            timings = None
            if load_debug_dir:
                timings = load_realign_debug(
                    item["audio_path"], realign_path, load_debug_dir,
                )
            if timings is None:
                timings = assign_lines(
                    lines, vad_segments, timeline,
                    align_metadata["dictionary"], align_metadata["language"],
                    window_seconds=window_seconds, commit_margin=commit_margin,
                )
                if debug_dir is not None:
                    write_realign_debug(
                        item["audio_path"], realign_path, timings, lines, debug_dir,
                    )

        warn_low_confidence(timings, lines, min_score)
        if transform is not None and debug_dir is not None:
            write_realign_transform(
                item["audio_path"], realign_path, mode, lines, timings, transform, report,
                dropped, debug_dir,
            )

        if mode == "sync":
            # Nothing downstream may re-time these: the whole contract of sync is that the
            # subtitle's own proportions survive, and forced alignment would undo that.
            segments = segments_from_timings(lines, timings, dropped)
            result = {"segments": segments, "language": align_metadata["language"]}
            result_items.append({**item, "result": result, "realign_sync": True})
            continue

        kept = [line for line in lines if line.index not in dropped]
        kept_timings = [t for t in timings if t.index not in dropped]
        chunks, transcript = build_align_input(
            kept, kept_timings, vad_segments, chunk_size,
        )
        result = {"segments": transcript, "language": align_metadata["language"]}
        result_items.append({
            **item, "vad_segments": chunks, "result": result,
            "emission_timeline": timeline,
        })
    return result_items


def _run_realign_asr(
    items: List[ProcessingItem],
    realign_path: str,
    *,
    chunk_size: float,
    normalize: bool = True,
    debug_dir: Optional[str] = None,
    load_debug_dir: Optional[str] = None,
) -> List[ProcessingItem]:
    """Time an untimed transcript against the ASR hypothesis (--realign_anchor asr).

    Same output shape as _run_realign, so stage 4 alignment onwards is identical; only the
    way each line's approximate position was found differs.
    """
    from cantocaptions_ai.pipeline.realign import (
        assign_lines_via_asr, build_align_input, load_transcript_lines,
    )
    from cantocaptions_ai.utils.debug import load_realign_debug, write_realign_debug

    lines = load_transcript_lines(realign_path, normalize=normalize)
    logger.info(f"Loaded {len(lines)} transcript line(s) from: {realign_path}")
    if not lines:
        raise ConfigError(f"realign transcript is empty: {realign_path}")

    out: List[ProcessingItem] = []
    for item in items:
        timings = None
        if load_debug_dir:
            timings = load_realign_debug(item["audio_path"], realign_path, load_debug_dir)
        if timings is None:
            timings = assign_lines_via_asr(
                lines, item["result"]["segments"], vad_segments=item["vad_segments"],
            )
            if debug_dir is not None:
                write_realign_debug(item["audio_path"], realign_path, timings, lines, debug_dir)
        chunks, transcript = build_align_input(
            lines, timings, item["vad_segments"], chunk_size,
        )
        result = {**item["result"], "segments": transcript}
        out.append({**item, "vad_segments": chunks, "result": result})
    return out


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


def _pre_align_clean(items: List[ProcessingItem], cleaner) -> List[ProcessingItem]:
    """Apply the cleaning manifest's ``pre_align`` steps to each ASR segment's text.

    Runs after the ASR (and any correction) cache loads rather than inside the ASR
    backend, so a --load_debug_dir replay picks up rule edits without re-running ASR.
    Punctuation added here is what alignment splits sentences on, so a clause comma
    such as 嘅話， can end one cue and start the next instead of trailing on a line.
    """
    changed = 0
    for item in items:
        segments = []
        for seg in item['result']['segments']:
            text = cleaner.pre_align(seg['text'])
            if text != seg['text']:
                changed += 1
                seg = {**seg, 'text': text}
            segments.append(seg)
        item['result']['segments'] = segments
    if changed:
        logger.info("Pre-alignment cleaning: updated %d segment(s)", changed)
    return items


def _merge_and_write(
    items: List[ProcessingItem],
    writer,
    align_language: str,
    align_merge_distance: float,
    align_padding: float,
    writer_args: dict,
    cleaner=None,
    debug_dir: Optional[str] = None,
    punctuation=DEFAULT_PUNCTUATION,
    segmentation=DEFAULT_SEGMENTATION,
    min_cue_duration: float = 0.5,
    merge_gap: float = 0.25,
    max_line_width: Optional[int] = None,
    max_line_count: Optional[int] = None,
    merge: bool = True,
    order_cues: bool = False,
    *,
    collect: bool = False,
    audio_start_offset: float = 0.0,
    display_paths: Optional[dict] = None,
) -> List[ProcessingItem]:
    """Assemble cues, clean, offset, then write (unless writer is None) and/or return results.

    ``display_paths`` maps a (possibly clip-substituted temp) audio path back to the
    original path so output filenames and returned results reference the source file,
    not the temp clip. ``collect`` returns the final per-item results for in-memory
    (server) use; ``writer`` is None when nothing should be written to disk.
    """
    # An ordinary merge is capped at one subtitle line, but a short-cue rescue may use the
    # full multi-line budget -- the cleaner's linebreak step will split the result across
    # max_line_count lines anyway. Fall back to the single-line cap when line limits are off
    # (e.g. under --no_align, where both are required to be unset).
    rescue_max_chars = (
        max_line_width * max_line_count
        if max_line_width and max_line_count
        else MAX_CHARS
    )
    # Reuse the cleaner as the single source of truth for "is this line pure noise": a cue
    # whose cleaned text is removable would have been dropped later anyway, so dropping it
    # before the rescue pass just stops it being glued onto a neighbour first.
    is_noise = (lambda text: is_removable(cleaner.clean(text))) if cleaner is not None else None
    finalized: List[ProcessingItem] = []
    for item in items:
        result = item['result']
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
            merge=merge,
        )

        if order_cues and debug_dir is not None:
            # After assembly so the timings and text are the ones that shipped, and before
            # cleaning so a cue dropped as noise does not silently take its reason with it.
            from cantocaptions_ai.utils.debug import write_realign_suspects
            write_realign_suspects(audio_path, new_segments, debug_dir)

        if debug_dir is not None:
            # The general annotation channel, unlike suspect.srt above: not "these timings
            # are doubtful" but "something happened to this cue you may want to see". Written
            # on every run, since nothing about it is specific to --realign.
            from cantocaptions_ai.utils.debug import write_segment_notes
            write_segment_notes(audio_path, new_segments, debug_dir)

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
            write_precleaning_debug(audio_path, result, debug_dir)

        if cleaner is not None:
            # Cleaning edits segment text only; words/chars keep the original
            # alignment tokens and timings.
            cleaned_segments = []
            for segment in new_segments:
                text = cleaner.clean(segment["text"])
                if is_removable(text):
                    continue
                segment["text"] = text
                cleaned_segments.append(segment)
            dropped = len(new_segments) - len(cleaned_segments)
            if dropped:
                logger.info(f"Text cleaning: dropped {dropped} interjection/noise subtitles")
            result["segments"] = cleaned_segments

        # Map clip-relative times back onto the source-media timeline before output.
        _offset_result_times(result, audio_start_offset)

        if writer is not None:
            writer(result, audio_path, writer_args)
        if collect:
            finalized.append({'audio_path': audio_path, 'result': result})
    return finalized


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
    if not 0 < cfg.speaker_conflict_share <= 1:
        raise ConfigError(
            f"speaker_conflict_share must be in (0, 1], got {cfg.speaker_conflict_share}"
        )
    if cfg.reference_subtitle and not (cfg.llm_correction or cfg.asr_context):
        raise ConfigError("reference_subtitle requires llm_correction or asr_context")
    if cfg.reference_correction_semantic and not cfg.reference_subtitle:
        warnings.warn("reference_correction_semantic has no effect without reference_subtitle")
    if cfg.asr_context and not cfg.reference_subtitle:
        raise ConfigError("asr_context requires reference_subtitle")
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

    if cfg.language is not None:
        cfg.language = cfg.language.lower()
        if cfg.language not in LANGUAGES:
            if cfg.language in TO_LANGUAGE_CODE:
                cfg.language = TO_LANGUAGE_CODE[cfg.language]
            else:
                raise ConfigError(f"Unsupported language: {cfg.language}")
    if cfg.language != "yue":
        warnings.warn(
            f"Configured language '{cfg.language}' is not yue/cantonese, and may not be compatible with this framework."
        )

    if cfg.no_align:
        for option in ("highlight_words", "max_line_count", "max_line_width"):
            if getattr(cfg, option):
                raise ConfigError(f"{option} not possible with no_align")
    if cfg.max_line_count and not cfg.max_line_width:
        warnings.warn("max_line_count has no effect without max_line_width")


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
        track = _select_audio_track(p)
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


def transcribe_task(args: dict, parser: argparse.ArgumentParser):
    """CLI adapter: build a PipelineConfig from parsed args and run the pipeline.

    Thin wrapper over :func:`_execute_pipeline` that preserves the CLI contract —
    validation errors become ``parser.error(...)`` (exit 2) and results are written
    to ``cfg.output_dir``. Library/server callers should use
    ``cantocaptions_ai.service.run_pipeline`` instead.
    """
    from cantocaptions_ai.pipeline.config import PipelineConfig
    from cantocaptions_ai.errors import ConfigError

    audio_paths = args.pop("audio")
    cfg = PipelineConfig.from_args(args)
    try:
        validate_config(cfg)
    except ConfigError as e:
        parser.error(str(e))

    paths, display_paths, temp_files = _prepare_clips(audio_paths, cfg)
    try:
        _execute_pipeline(
            paths, cfg, collect=False,
            audio_start_offset=cfg.audio_start or 0.0,
            display_paths=display_paths,
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
) -> List[ProcessingItem]:
    """Run all pipeline stages for *audio_paths* under *cfg*.

    Assumes *cfg* has already passed :func:`validate_config`, and that any audio clip
    has already been applied (paths point at clipped temp files; see
    :func:`_prepare_clips`). With ``collect=True`` the final per-item results are
    returned and nothing is written to disk; otherwise results are written to
    ``cfg.output_dir``. ``progress`` receives stage/progress events out-of-band.
    """
    from cantocaptions_ai.pipeline.model_profiles import get_model_profile
    from huggingface_hub.utils.tqdm import disable_progress_bars

    # HF Hub's own tqdm download bars race StageTimer's spinner over the same
    # terminal line (both are \r-driven redraw loops); silencing them means our
    # explicit "Downloading %r..." log lines are the only download-progress signal,
    # so a slow first-run download never looks like a stalled/hung stage.
    disable_progress_bars()

    align_language = cfg.language if cfg.language is not None else "yue"
    task: str = "transcribe"

    # The ASR model's profile drives the downstream path: post-ASR text normalization
    # (applied inside the ASR backend), the alignment particle spot-checks, and the
    # punctuation set used for sentence splitting / line merging. Unregistered models get
    # an all-default (no-op) profile. See pipeline/model_profiles.py.
    profile = get_model_profile(cfg.model)

    qwen_threads = torch.get_num_threads()
    if cfg.threads > 0:
        torch.set_num_threads(cfg.threads)
        qwen_threads = cfg.threads

    asr_options = {
        "condition_on_previous_text": False,
        "initial_prompt": cfg.initial_prompt,
        "hotwords": cfg.hotwords,
        "suppress_tokens": [int(x) for x in cfg.suppress_tokens.split(",")],
        "suppress_numerals": cfg.suppress_numerals,
    }

    if collect:
        writer = None
    else:
        os.makedirs(cfg.output_dir, exist_ok=True)
        writer = get_writer(cfg.output_format, cfg.output_dir)
    writer_args = {
        "highlight_words": cfg.highlight_words,
        "max_line_count": cfg.max_line_count,
        "max_line_width": cfg.max_line_width,
        "speaker_labels": cfg.speaker_labels,
    }

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
    if realign_mode in ("sync", "adjust"):
        if not cfg.no_clean_text:
            logger.info(
                "Text cleaning skipped: --realign_mode %s preserves the subtitle's own text "
                "(it is already a finished subtitle, not a raw transcript)", realign_mode,
            )
    elif not cfg.no_clean_text:
        from cantocaptions_ai.cantonese.cleaner import SubtitleCleaner
        cleaner = SubtitleCleaner(
            rules_dir=cfg.clean_rules_dir,
            line_max_length=cfg.max_line_width or 21,
            max_line_count=cfg.max_line_count,
            # How much cleaning the text needs depends on how the model writes it, so the
            # step manifest comes from the profile like every other output convention.
            manifest=profile.cleaning.manifest,
        )
        if cfg.highlight_words:
            warnings.warn(
                "--highlight_words uses word timings that text cleaning does not update; "
                "highlighted output may not match the cleaned text"
            )

    if cfg.load_debug_dir:
        from pathlib import Path as _Path
        missing = [
            ap for ap in audio_paths
            if not os.path.isdir(os.path.join(cfg.load_debug_dir, _Path(ap).stem.strip()))
        ]
        if missing:
            # Not fatal: files without cached data are simply (re)computed from scratch,
            # which matters for --input_dir runs where only some files were cached before.
            logger.warning(
                "No debug data under '%s' for %d of %d file(s); they will be computed from "
                "scratch: %s",
                cfg.load_debug_dir, len(missing), len(audio_paths),
                ", ".join(_Path(ap).name for ap in missing),
            )

    if cfg.realign:
        from cantocaptions_ai.pipeline.realign import (
            REALIGN_PUNCTUATION, enforce_cue_order, ensure_visible_cues, strip_sentinels,
            tighten_cue_spans, warn_on_implausible_cues,
        )
    realign_acoustic = bool(cfg.realign) and cfg.realign_anchor == "acoustic"
    need_asr = not realign_acoustic and (
        not cfg.load_debug_dir or any(
            not _debug_stage_exists(ap, "transcription", cfg.load_debug_dir) for ap in audio_paths
        )
    )
    vocal_isolation_active = (
        bool(cfg.vocal_isolation_method) and cfg.vocal_isolation_method.lower() != "none"
    )
    # A cached vocal isolation checkpoint makes that file's VAD stage entirely dead
    # weight: the isolation manifest carries the same segment boundaries, its WAVs
    # carry the audio ASR actually consumes, and stage 2 replaces vad_segments
    # wholesale — so reading the VAD WAVs back would only be thrown away.
    isolation_cached = [
        vocal_isolation_active
        and bool(cfg.load_debug_dir)
        and _debug_stage_exists(ap, "vocal_isolation", cfg.load_debug_dir)
        for ap in audio_paths
    ]
    vad_indices = [i for i, cached in enumerate(isolation_cached) if not cached]
    need_vad = any(
        not cfg.load_debug_dir
        or not _debug_stage_exists(audio_paths[i], "vad", cfg.load_debug_dir)
        for i in vad_indices
    )
    need_vocal_isolation = vocal_isolation_active and not all(isolation_cached)
    need_ensemble = (
        cfg.ensemble_model != "none"
        and (not cfg.load_debug_dir or any(
            not _debug_stage_exists(ap, "ensemble", cfg.load_debug_dir) for ap in audio_paths
        ))
    )
    need_llm = (
        cfg.llm_correction
        and (not cfg.load_debug_dir or any(
            not _debug_stage_exists(ap, "llm_correction", cfg.load_debug_dir) for ap in audio_paths
        ))
    )
    need_diarize = (
        cfg.diarize
        and (not cfg.load_debug_dir or any(
            not _debug_stage_exists(ap, "diarization", cfg.load_debug_dir) for ap in audio_paths
        ))
    )

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

    # Stage 1: VAD
    # Files covered by a cached vocal isolation checkpoint are held back entirely
    # (see isolation_cached above); they enter stage 2 as bare carriers and get their
    # vad_segments from the isolation cache.
    items: List[dict] = [{'audio_path': p} for p in audio_paths]
    if len(vad_indices) < len(audio_paths):
        logger.info(
            "Skipping VAD for %d of %d file(s) already covered by cached vocal isolation",
            len(audio_paths) - len(vad_indices), len(audio_paths),
        )
    if vad_indices:
        with StageTimer("VAD", summary, progress=progress) as stage:
            vad_items = [
                {
                    'audio_path': audio_paths[i],
                    'audio_track': _select_audio_track(audio_paths[i]),
                    'audio_downmix': cfg.audio_downmix,
                    'audio_normalize': cfg.audio_normalize,
                }
                for i in vad_indices
            ]
            if need_vad:
                from cantocaptions_ai.pipeline.vad import load_vad
                # A caller (e.g. PipelineService in resident mode) may pass a preloaded
                # VAD model to reuse across jobs — load_vad reuses it and ignores
                # vad_method. It stays alive via the caller's reference after the
                # processor wrapper is dropped below, skipping the ~20-30s reload.
                vad_processor = load_vad(
                    vad_method=cfg.vad_method,
                    device=cfg.device,
                    device_index=cfg.device_index,
                    vad_onset=cfg.vad_onset,
                    vad_offset=cfg.vad_offset,
                    vad_pad_onset=cfg.vad_pad_onset,
                    vad_pad_offset=cfg.vad_pad_offset,
                    vad_min_duration_off=cfg.vad_min_duration_off,
                    chunk_size=cfg.chunk_size,
                    vad_model=vad_model,
                    use_auth_token=cfg.hf_token,
                    reference_cues=(
                        reference_cues
                        if cfg.asr_context and cfg.asr_context_vad_expand
                        else None
                    ),
                    reference_padding=cfg.asr_context_padding,
                    # --realign holds a transcript line for every utterance, including ones
                    # VAD scores below threshold, so segmentation may only choose cut points
                    # -- it may not decide what to keep. See Vad.cover_chunks. This applies
                    # under both anchors: the 'asr' anchor pays for transcribing the whole
                    # file rather than just its speech, but in exchange the chunks it hands
                    # to alignment cover the audio with no gaps (and carry vocal isolation
                    # throughout), which is what the final chunk re-cut assumes.
                    cover_all=bool(cfg.realign),
                )
                stage.mark_inference_start()
                vad_out = vad_processor.run(vad_items, debug_dir=cfg.debug_dir, load_debug_dir=cfg.load_debug_dir, progress_callback=stage.reporter)
                del vad_processor
            else:
                vad_out = VadProcessor.load_cache(vad_items, cfg.load_debug_dir)
        for i, out in zip(vad_indices, vad_out):
            items[i] = out
        flush_vram()

    # Stage 2: Vocal Isolation (conditional)
    if need_vocal_isolation:
        with StageTimer("Vocal isolation", summary, progress=progress) as stage:
            from cantocaptions_ai.pipeline.vocal_isolation import load_vocal_isolation
            vocal_isolation_processor = load_with_offline_fallback(
                load_vocal_isolation,
                model_name=cfg.vocal_isolation_method,
                device=cfg.device,
                device_index=cfg.device_index,
                batch_size=cfg.vocal_isolation_batch_size,
                compute_type=cfg.vocal_isolation_compute_type,
                vram_checks=cfg.vram_checks,
                local_files_only=cfg.model_cache_only,
                segment_mode=cfg.vocal_isolation_segment_mode,
            )
            stage.mark_inference_start()
            items = vocal_isolation_processor.run(items, debug_dir=cfg.debug_dir, load_debug_dir=cfg.load_debug_dir, progress_callback=stage.reporter)
            del vocal_isolation_processor
    elif vocal_isolation_active and cfg.load_debug_dir:
        # All files' isolated audio is cached: load it so downstream ASR sees the
        # isolated (not raw) audio even if ASR itself is being recomputed.
        from cantocaptions_ai.pipeline.vocal_isolation import MbRoformerProcessor
        items = MbRoformerProcessor.load_cache(items, cfg.load_debug_dir)

    flush_vram()

    if realign_acoustic:
        # Realign, acoustic anchor: the transcript is known and complete, only its timings
        # are missing. The alignment model does both jobs -- a coarse sliding search for
        # where each line sits, then forced alignment for the timings within a line. The
        # 'asr' anchor takes the ordinary ASR path below and rejoins at stage 4.
        with StageTimer("Transcript realignment", summary, progress=progress) as stage:
            from cantocaptions_ai.pipeline.alignment import load_align_model, load_bert_processor
            bert_processor = load_with_offline_fallback(
                load_bert_processor, model_dir=cfg.model_dir, model_cache_only=cfg.model_cache_only
            )
            align_model, align_metadata = load_with_offline_fallback(
                load_align_model,
                align_language, cfg.device, cfg.device_index,
                model_name=cfg.align_model, model_dir=cfg.model_dir, model_cache_only=cfg.model_cache_only,
                compute_type=cfg.align_compute_type,
                vram_checks=cfg.vram_checks,
                char_substitution=cfg.align_char_substitution,
                substitution_overrides=_substitution_overrides(cfg),
            )
            stage.mark_inference_start()
            items = _run_realign(
                items, cfg.realign, align_model, align_metadata, bert_processor, cfg.device,
                chunk_size=cfg.chunk_size,
                window_seconds=cfg.realign_window,
                commit_margin=cfg.realign_commit_margin,
                min_score=cfg.realign_min_score,
                mode=realign_mode,
                cut_policy=cfg.realign_cut_policy,
                max_scale=cfg.realign_max_scale,
                adjust_tolerance=cfg.realign_adjust_tolerance,
                align_padding=cfg.align_padding,
                sync_anchor_density=cfg.realign_sync_anchor_density,
                normalize=cfg.realign_normalize,
                batch_size=cfg.align_batch_size,
                vram_checks=cfg.vram_checks,
                debug_dir=cfg.debug_dir,
                load_debug_dir=cfg.load_debug_dir,
            )
            # Mode sync's cues are finished: the transform placed them, and the guarantee it
            # makes -- that the subtitle's own proportions survive exactly -- is only true if
            # nothing re-times them afterwards. So alignment and its fixups are skipped
            # rather than run and then overridden.
            pending = [item for item in items if not item.get("realign_sync")]
            if pending:
                aligned = _run_alignment(
                    pending, align_model, align_metadata, bert_processor, cfg.device,
                    cfg.align_padding, cfg.align_release, cfg.interpolate_method,
                    cfg.return_char_alignments, cfg.print_progress, cfg.align_batch_size,
                    progress_callback=stage.reporter,
                    vram_checks=cfg.vram_checks,
                    spotchecks=profile.spotchecks,
                    # 0, not cfg.align_split_gap: under --realign the transcript's line
                    # breaks are the cue boundaries and a cue is one whole line by contract,
                    # so nothing here may break one in two -- not even an align profile that
                    # asked for it on the ASR path.
                    split_gap=0,
                    # Not profile.punctuation: realign needs the space, the newline and the
                    # line sentinel to be pause tokens, and declares its cue boundaries
                    # through cue_spans rather than letting punctuation derive them.
                    punctuation=REALIGN_PUNCTUATION,
                )
                by_path = {item["audio_path"]: item for item in aligned}
                items = [by_path.get(item["audio_path"], item) for item in items]
            for item in items:
                if item.get("realign_sync"):
                    # No words and no sentinels to tidy; only the two validity guarantees.
                    ensure_visible_cues(item["result"]["segments"])
                    continue
                segments = item["result"]["segments"]
                tighten_cue_spans(segments)
                ensure_visible_cues(segments)
                strip_sentinels(segments)
        del align_model, bert_processor
        flush_vram()
    else:
        # Attach ASR context here rather than at VAD time: both the VAD and vocal
        # isolation debug round-trips rebuild segment dicts from scratch and would drop
        # the key, and this point is downstream of both cache loads. Contexts are cheap
        # and always re-derived, so --asr_context_template edits take effect on replay.
        if cfg.asr_context and reference_cues:
            from cantocaptions_ai.pipeline.reference_context import build_segment_contexts
            for item in items:
                spans = None
                if cfg.asr_context_scope == "expanded":
                    # Provenance recorded at VAD time and carried through the debug
                    # manifests; absent means this segment is entirely VAD's own find.
                    spans = [
                        sp for seg in item['vad_segments'] for sp in seg.get('expanded', ())
                    ]
                contexts = build_segment_contexts(
                    item['vad_segments'],
                    reference_cues,
                    neighbours=cfg.asr_context_neighbours,
                    template=cfg.asr_context_template,
                    max_chars=cfg.asr_context_max_chars,
                    restrict_to_spans=spans,
                )
                item['vad_segments'] = [
                    {**seg, 'context': ctx}
                    for seg, ctx in zip(item['vad_segments'], contexts)
                ]
                if cfg.asr_context_scope == "expanded" and cfg.asr_context_template != "none":
                    logger.info(
                        "ASR context: scope 'expanded' -- %d of %d segment(s) carry a "
                        "context over reference-recovered audio; the rest decode bare",
                        sum(1 for c in contexts if c), len(contexts),
                    )
                elif cfg.asr_context_template == "none":
                    logger.info(
                        "ASR context: template 'none' -- %d segment(s) decode without a "
                        "context prompt; the reference subtitle affected the VAD "
                        "timeline only", len(contexts),
                    )
                else:
                    logger.info(
                        "ASR context: %d of %d segment(s) biased by the reference subtitle",
                        sum(1 for c in contexts if c), len(contexts),
                    )

        # Stage 3: Transcription
        if need_asr:
            from cantocaptions_ai.pipeline.asr import load_model
            with StageTimer("Transcription", summary, progress=progress) as stage:
                with model_scope(
                    load_model,
                    cfg.model,
                    device=cfg.device,
                    device_index=cfg.device_index,
                    download_root=cfg.model_dir,
                    compute_type=cfg.asr_compute_type,
                    attn_implementation=cfg.attn_implementation,
                    language=cfg.language,
                    asr_options=asr_options,
                    vocal_isolation_method=cfg.vocal_isolation_method,
                    task=task,
                    local_files_only=cfg.model_cache_only,
                    threads=qwen_threads,
                    use_auth_token=cfg.hf_token,
                    batch_size=cfg.batch_size,
                    compile_enabled=cfg.compile,
                    print_progress=cfg.print_progress,
                    verbose=cfg.verbose,
                    vram_checks=cfg.vram_checks,
                    vram_headroom_mb=cfg.vram_headroom_mb,
                ) as model:
                    stage.mark_inference_start()
                    items = model.run(items, debug_dir=cfg.debug_dir, load_debug_dir=cfg.load_debug_dir, progress_callback=stage.reporter)
        else:
            from cantocaptions_ai.pipeline.asr import QwenPipeline
            items = QwenPipeline.load_cache(items, cfg.load_debug_dir)

        # Stage 3b: Ensemble ASR (optional)
        if cfg.ensemble_model != "none":
            if need_ensemble:
                from cantocaptions_ai.pipeline.ensemble import load_faster_whisper
                with StageTimer("Ensemble ASR (faster-whisper)", summary, progress=progress) as stage:
                    with model_scope(
                        load_faster_whisper,
                        device=cfg.device,
                        device_index=cfg.device_index,
                        model_dir=cfg.model_dir,
                        local_files_only=cfg.model_cache_only,
                    ) as ensemble:
                        stage.mark_inference_start()
                        items = ensemble.run(items, debug_dir=cfg.debug_dir, load_debug_dir=cfg.load_debug_dir, progress_callback=stage.reporter)
            else:
                from cantocaptions_ai.pipeline.ensemble import FasterWhisperEnsemble
                items = FasterWhisperEnsemble.load_cache(items, cfg.load_debug_dir)

        # Stage 3c: LLM correction (optional)
        if cfg.llm_correction:
            if reference_cues:
                from cantocaptions_ai.pipeline.llm_correction import match_reference_to_segments
                with StageTimer("Reference subtitle matching", summary, progress=progress):
                    for item in items:
                        item['reference_texts'] = match_reference_to_segments(
                            item['result']['segments'], reference_cues
                        )

            if need_llm:
                from cantocaptions_ai.pipeline.llm_correction import load_llm
                if cfg.vram_checks:
                    stats = vram_stats()
                    if stats:
                        logger.info(
                            f"VRAM before LLM load: allocated={stats['allocated_mb']:.0f} MB, "
                            f"reserved={stats['reserved_mb']:.0f} MB, "
                            f"free={stats['free_mb']:.0f} MB / {stats['total_mb']:.0f} MB"
                        )
                with StageTimer("LLM correction", summary, progress=progress) as stage:
                    with model_scope(
                        load_llm,
                        model_id=cfg.llm_model,
                        model_dir=cfg.llm_model_dir,
                        device=cfg.device,
                        local_files_only=cfg.model_cache_only,
                        semantic_mode=cfg.reference_correction_semantic,
                        attn_implementation=cfg.attn_implementation,
                        vram_checks=cfg.vram_checks,
                    ) as corrector:
                        stage.mark_inference_start()
                        items = corrector.run(items, debug_dir=cfg.debug_dir, load_debug_dir=cfg.load_debug_dir, progress_callback=stage.reporter)
            else:
                from cantocaptions_ai.pipeline.llm_correction import LLMCorrector
                items = LLMCorrector.load_cache(items, cfg.load_debug_dir)

        # The transcript replaces the ASR hypothesis here: ASR ran only to say *where* each
        # line is, and from this point on the pipeline is identical to the acoustic anchor.
        if cfg.realign:
            with StageTimer("Transcript matching", summary, progress=progress):
                items = _run_realign_asr(
                    items, cfg.realign, chunk_size=cfg.chunk_size,
                    normalize=cfg.realign_normalize,
                    debug_dir=cfg.debug_dir, load_debug_dir=cfg.load_debug_dir,
                )

        # Not under --realign: the transcript's cue_spans index its text, so inserting a
        # character would shift every span after it -- and that text is the user's anyway.
        if cleaner is not None and not cfg.realign:
            items = _pre_align_clean(items, cleaner)

        # Stage 4: Alignment
        if not cfg.no_align:
            with StageTimer("Alignment", summary, progress=progress) as stage:
                from cantocaptions_ai.pipeline.alignment import load_align_model, load_bert_processor
                bert_processor = load_with_offline_fallback(
                    load_bert_processor, model_dir=cfg.model_dir, model_cache_only=cfg.model_cache_only
                )
                align_model, align_metadata = load_with_offline_fallback(
                    load_align_model,
                    align_language, cfg.device, cfg.device_index,
                    model_name=cfg.align_model, model_dir=cfg.model_dir, model_cache_only=cfg.model_cache_only,
                    compute_type=cfg.align_compute_type,
                    vram_checks=cfg.vram_checks,
                    char_substitution=cfg.align_char_substitution,
                    substitution_overrides=_substitution_overrides(cfg),
                )
                stage.mark_inference_start()
                items = _run_alignment(
                    items, align_model, align_metadata, bert_processor, cfg.device,
                    cfg.align_padding, cfg.align_release, cfg.interpolate_method,
                    cfg.return_char_alignments, cfg.print_progress, cfg.align_batch_size,
                    progress_callback=stage.reporter,
                    vram_checks=cfg.vram_checks,
                    spotchecks=profile.spotchecks,
                    punctuation=(
                        REALIGN_PUNCTUATION if cfg.realign else profile.punctuation
                    ),
                    # See the realign call site above for why this is forced off there.
                    split_gap=0 if cfg.realign else cfg.align_split_gap,
                )
                if cfg.realign:
                    for item in items:
                        segments = item["result"]["segments"]
                        tighten_cue_spans(segments)
                        # Order first: ensure_visible_cues reads the previous cue's end as
                        # its floor, which is only meaningful once the cues are in order.
                        enforce_cue_order(segments)
                        ensure_visible_cues(segments)
                        strip_sentinels(segments)
                        # Last: the cue text has to be final before its span can be judged
                        # against what that text could have been spoken in.
                        warn_on_implausible_cues(segments)
            del align_model, bert_processor
            flush_vram()
        else:
            items = _extract_timestamps(items)

    # Stage 5: Diarization. Runs after alignment so speaker turns land on the over-split
    # subsegments that cue assembly is about to merge back together, which is what lets
    # segmentation._same_speaker veto a merge across a speaker change.
    if cfg.diarize:
        items = _run_diarization(items, cfg, summary, progress, need_diarize)
        items = _assign_speakers(items, cfg, debug_dir=cfg.debug_dir)

    # Write and/or collect final results
    results = _merge_and_write(
        items, writer, align_language, cfg.align_merge_distance, cfg.align_padding, writer_args,
        cleaner=cleaner, debug_dir=cfg.debug_dir, punctuation=profile.punctuation,
        segmentation=profile.segmentation,
        # Mode sync promises that the subtitle's own proportions survive the round trip, and
        # the duration floor (pass D) would quietly break that by stretching any cue the
        # transform made shorter than min_cue_duration. Zero turns passes B-D off, which is
        # right here for the same reason merge is: every cue's span is already a decision
        # somebody made, not an artefact of over-splitting.
        min_cue_duration=0.0 if realign_mode == "sync" else cfg.min_cue_duration,
        merge_gap=cfg.merge_gap, max_line_width=cfg.max_line_width,
        max_line_count=cfg.max_line_count,
        # Under --realign the cue boundaries came from the transcript's own line breaks and
        # are not an artifact to be undone, so the two passes that join cues are off; the
        # noise drop and the duration floor still run.
        merge=not cfg.realign,
        order_cues=bool(cfg.realign),
        collect=collect, audio_start_offset=audio_start_offset, display_paths=display_paths,
    )

    summary.print_summary(process_elapsed=time.perf_counter() - process_start)
    return results

"""The pipeline's stages: what runs between reading the audio and writing the subtitles.

``transcribe._execute_pipeline`` drives these in order. Each helper here is one stage's work
(or one step of it), kept apart from the CLI and config plumbing in ``transcribe.py``, which
re-exports the names other code imports from there.
"""
import os
from typing import List, Optional

from cantocaptions_ai.errors import ConfigError
from cantocaptions_ai.text_profiles import DEFAULT_PUNCTUATION
from cantocaptions_ai.utils.debug import write_speaker_assignment_debug
from cantocaptions_ai.utils.log_utils import ProgressSink, StageTimer, TranscriptionSummary, get_logger
from cantocaptions_ai.utils.model_utils import flush_vram, load_with_offline_fallback
from cantocaptions_ai.utils.schema import (
    AlignedTranscriptionResult,
    ProcessingItem,
    ProgressCallback,
    VadItem,
    item_name,
)

logger = get_logger(__name__)


_VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.ts', '.m2ts'}


def _select_audio_track(path: str, language: Optional[str] = "yue",
                        override: Optional[int] = None) -> int:
    """Return the 0-based audio stream index to use for *path*.

    ``override`` (--audio_track) wins outright. Otherwise video files are probed with
    ffprobe and the stream most likely to carry *language* is chosen (see
    utils.audio.select_track); audio-only files, or a probe that finds nothing, get 0
    (ffmpeg's default).
    """
    if override is not None:
        return override
    ext = os.path.splitext(path)[1].lower()
    if ext not in _VIDEO_EXTENSIONS:
        return 0
    from cantocaptions_ai.languages import get_language_pack
    from cantocaptions_ai.utils.audio import probe_audio_tracks
    streams = probe_audio_tracks(path)
    if not streams:
        logger.warning("No audio streams found via ffprobe for '%s'; using default track", path)
        return 0
    track = get_language_pack(language).choose_track(streams)
    if track != 0:
        logger.info("Selected audio track index %d (%s) for '%s'", track, language, path)
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
    script=None,
) -> List[ProcessingItem]:
    from cantocaptions_ai.pipeline.alignment import align
    if progress_callback is not None:
        progress_callback.set_total(sum(len(it['result']['segments']) for it in items), unit="seg")
    aligned_items = []
    for item in items:
        result = item['result']
        # Popped, not read: the file's emissions are ~0.5 GB an hour of audio, and the caller's
        # list still holds this item, so leaving the timeline on it would keep every file's
        # alive through the rest of the batch and every later stage.
        timeline = item.pop('emission_timeline', None)
        if align_model is not None and len(result["segments"]) > 0:
            logger.info("Performing alignment...")
            aligned_result: AlignedTranscriptionResult = align(
                result["segments"],
                align_model,
                align_metadata,
                item['vad_segments'],
                device,
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
                timeline=timeline,
                split_gap=split_gap, script=script,
            )
            aligned_result['language'] = result['language']
        else:
            aligned_result = result
        # Keep the rest of the carrier (notably audio_track, chosen by _select_audio_track):
        # diarization loads the audio again downstream and must not fall back to track 0.
        aligned_items.append({**item, 'result': aligned_result})
    return aligned_items


def _extract_timestamps(items: list) -> List[ProcessingItem]:
    """Segment timings without alignment (used when no_align=True).

    Uses the ASR's per-character timestamps where a backend provides them, and otherwise
    the segment's own span -- the VAD chunk it was decoded from. Neither Qwen backend emits
    ``time_stamps``, so the fallback is the normal path; requiring them made --no_align
    fail with a KeyError on every run.
    """
    extracted_items = []
    for item in items:
        result = item['result']
        segments = []
        for segment in result['segments']:
            stamps = segment.get('time_stamps')
            s_start = stamps[0]['start'] if stamps else segment['start']
            s_end = stamps[-1]['end'] if stamps else segment['end']
            # Cue assembly merges on `words`; an unaligned segment simply has none.
            segments.append({'words': [], **segment, 'start': s_start, 'end': s_end})
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
            write_speaker_assignment_debug(item_name(item), segments, debug_dir)
    return items


def _load_realign_checkpoint(item: dict, realign_path: str, load_debug_dir: Optional[str]):
    """Cached line placements for *item*, or None if absent or made under other settings."""
    from cantocaptions_ai.utils.checkpoints import checkpoint_is_current
    from cantocaptions_ai.utils.debug import load_realign_debug

    if not load_debug_dir:
        return None
    name = item_name(item)
    settings = (item.get("checkpoints") or {}).get("realign")
    if not checkpoint_is_current(load_debug_dir, name, "realign", settings):
        return None
    return load_realign_debug(name, realign_path, load_debug_dir)


def _write_realign_checkpoint(item: dict, realign_path: str, timings, lines, debug_dir) -> None:
    from cantocaptions_ai.utils.checkpoints import write_checkpoint_meta
    from cantocaptions_ai.utils.debug import write_realign_debug

    if debug_dir is None:
        return
    name = item_name(item)
    write_realign_debug(name, realign_path, timings, lines, debug_dir)
    settings = (item.get("checkpoints") or {}).get("realign")
    if settings is not None:
        write_checkpoint_meta(debug_dir, name, "realign", settings)


def _run_realign(
    items: List[VadItem],
    realign_path: str,
    align_model,
    align_metadata,
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
    punctuation=None,
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
    from cantocaptions_ai.utils.debug import write_realign_transform

    timed = mode in ("sync", "adjust")
    lines = load_transcript_lines(realign_path, keep_timings=timed, normalize=normalize)
    # The language's realign punctuation (realign.realign_punctuation); the searches
    # default to the CJK one when none is given.
    punct = {"punctuation": punctuation} if punctuation is not None else {}
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
                segments, _model, align_metadata["type"], align_metadata["processor"], device,
                batch_size, vram_checks=vram_checks, primer=profile.primer,
                min_samples=profile.min_samples, dtype=EmissionTimeline.dtype,
            )

        # One timeline for the whole file, built here and handed to the alignment stage
        # below: the placements and the final alignment read the same emissions, so the
        # encoder runs once over the file instead of once per pass.
        # TODO: run realign and alignment per file. Every file's timeline is built here before
        # any is aligned, so a batch peaks at the sum of them (~0.5 GB an hour of audio each);
        # _run_alignment frees each one only after the whole batch has been realigned.
        timeline = EmissionTimeline(
            vad_segments, compute, frame_rate=align_metadata.get("frame_rate"))

        common = dict(
            window_seconds=window_seconds, commit_margin=commit_margin,
            max_scale_dev=max_scale, cut_policy=cut_policy, **punct,
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
            timings = _load_realign_checkpoint(item, realign_path, load_debug_dir)
            if timings is None:
                timings = assign_lines(
                    lines, vad_segments, timeline,
                    align_metadata["dictionary"], align_metadata["language"],
                    window_seconds=window_seconds, commit_margin=commit_margin, **punct,
                )
                _write_realign_checkpoint(item, realign_path, timings, lines, debug_dir)

        warn_low_confidence(timings, lines, min_score)
        if transform is not None and debug_dir is not None:
            write_realign_transform(
                item_name(item), realign_path, mode, lines, timings, transform, report,
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
    punctuation=None,
) -> List[ProcessingItem]:
    """Time an untimed transcript against the ASR hypothesis (--realign_anchor asr).

    Same output shape as _run_realign, so stage 4 alignment onwards is identical; only the
    way each line's approximate position was found differs.
    """
    from cantocaptions_ai.pipeline.realign import (
        assign_lines_via_asr, build_align_input, load_transcript_lines,
    )

    lines = load_transcript_lines(realign_path, normalize=normalize)
    logger.info(f"Loaded {len(lines)} transcript line(s) from: {realign_path}")
    if not lines:
        raise ConfigError(f"realign transcript is empty: {realign_path}")

    out: List[ProcessingItem] = []
    for item in items:
        timings = _load_realign_checkpoint(item, realign_path, load_debug_dir)
        if timings is None:
            timings = assign_lines_via_asr(
                lines, item["result"]["segments"], vad_segments=item["vad_segments"],
                **({"punctuation": punctuation} if punctuation is not None else {}),
            )
            _write_realign_checkpoint(item, realign_path, timings, lines, debug_dir)
        chunks, transcript = build_align_input(
            lines, timings, item["vad_segments"], chunk_size,
        )
        result = {**item["result"], "segments": transcript}
        out.append({**item, "vad_segments": chunks, "result": result})
    return out


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

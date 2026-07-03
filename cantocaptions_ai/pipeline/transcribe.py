import argparse
import os
import time
import warnings

import numpy as np
import torch

from cantocaptions_ai.utils.audio import load_audio, SAMPLE_RATE
from cantocaptions_ai.utils.schema import AlignedTranscriptionResult, ProcessingItem, ProgressCallback, VadItem, merge_segments
from typing import Callable, List, Optional
from cantocaptions_ai.utils.output import LANGUAGES, TO_LANGUAGE_CODE, get_writer
from cantocaptions_ai.utils.log_utils import StageTimer, TranscriptionSummary, get_logger
from cantocaptions_ai.utils.model_utils import model_scope, flush_vram, vram_stats
from cantocaptions_ai.cantonese.text import is_mergeable, is_removable
from cantocaptions_ai.utils.debug import _debug_stage_exists, write_precleaning_debug

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
            )
            aligned_result['language'] = result['language']
        else:
            aligned_result = result
        aligned_items.append({'audio_path': item['audio_path'], 'result': aligned_result})
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
        extracted_items.append({'audio_path': item['audio_path'], 'result': {**result, 'segments': segments}})
    return extracted_items


def _run_diarization(
    items: List[ProcessingItem],
    diarize_model_name: str,
    hf_token: Optional[str],
    device: str,
    model_dir: Optional[str],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    return_speaker_embeddings: bool,
    progress_callback: ProgressCallback = None,
    load_complete_callback: Optional[Callable[[], None]] = None,
) -> List[ProcessingItem]:
    from cantocaptions_ai.pipeline.diarize import DiarizationPipeline, assign_word_speakers
    logger.info("Performing diarization...")
    logger.info(f"Using model: {diarize_model_name}")
    diarize_model = DiarizationPipeline(model_name=diarize_model_name, token=hf_token, device=device, cache_dir=model_dir)
    if load_complete_callback:
        load_complete_callback()

    if progress_callback is not None:
        progress_callback.set_total(len(items), unit="file")
    diarized_items = []
    for item in items:
        diarize_result = diarize_model(
            item['audio_path'],
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            return_embeddings=return_speaker_embeddings,
        )
        if return_speaker_embeddings:
            diarize_segments, speaker_embeddings = diarize_result
        else:
            diarize_segments, speaker_embeddings = diarize_result, None
        result = assign_word_speakers(diarize_segments, item['result'], speaker_embeddings)
        diarized_items.append({'audio_path': item['audio_path'], 'result': result})
        if progress_callback is not None:
            progress_callback.advance(1)
    return diarized_items


def _run_speaker_verification(
    items: List[ProcessingItem],
    device: str,
    model_dir: Optional[str],
    progress_callback: ProgressCallback = None,
    load_complete_callback: Optional[Callable[[], None]] = None,
) -> List[ProcessingItem]:
    from cantocaptions_ai.pipeline.speaker_verification import SpeakerVerificationPipeline
    logger.info("Performing speaker verification...")
    speaker_verification_model = SpeakerVerificationPipeline(device=device, cache_dir=model_dir)
    if load_complete_callback:
        load_complete_callback()

    if progress_callback is not None:
        progress_callback.set_total(len(items), unit="file")
    verified_items = []
    for item in items:
        audio = load_audio(item['audio_path'], audio_track=item.get('audio_track', 0))
        audio_segments = [
            audio[int(seg["start"] * SAMPLE_RATE):int(seg["end"] * SAMPLE_RATE)]
            for seg in item['result']["segments"]
        ]
        speaker_verification_model(transcript=item['result']["segments"], audio=audio_segments)
        verified_items.append(item)
        if progress_callback is not None:
            progress_callback.advance(1)
    return verified_items


def _run_retime(
    items: List[VadItem],
    retime_path: str,
    align_model,
    align_metadata,
    bert_processor,
    device: str,
    score_threshold: float = -5.0,
    search_window: float = 120.0,
) -> List[ProcessingItem]:
    from cantocaptions_ai.pipeline.retime import load_subtitle_file, retime_subtitles
    logger.info(f"Loading subtitles from: {retime_path}")
    subtitles = load_subtitle_file(retime_path)
    logger.info(f"Loaded {len(subtitles)} subtitle lines.")
    result_items = []
    for item in items:
        logger.info("Retiming subtitles against VAD segments...")
        coarse_segments = retime_subtitles(
            subtitles,
            item["vad_segments"],
            align_model,
            align_metadata,
            bert_processor,
            device,
            score_threshold=score_threshold,
            search_window=search_window,
        )
        result = {"segments": coarse_segments, "language": align_metadata["language"]}
        result_items.append({**item, "result": result})
    return result_items


def _merge_and_write(
    items: List[ProcessingItem],
    writer,
    align_language: str,
    align_merge_distance: float,
    align_padding: float,
    writer_args: dict,
    cleaner=None,
    debug_dir: Optional[str] = None,
) -> None:
    for item in items:
        result = item['result']
        audio_path = item['audio_path']
        result["language"] = align_language

        new_segments = []
        prev_segment = None

        for segment in result["segments"]:
            text = segment["text"].strip()
            start = segment["start"]

            if prev_segment is None:
                new_segments.append(segment)
                prev_segment = segment
                continue

            prev_text = prev_segment["text"]
            prev_end = prev_segment["end"]

            if start - prev_end <= align_merge_distance - align_padding and is_mergeable(prev_text, text):
                new_seg = merge_segments(prev_segment, segment)
                new_segments.pop()
                new_segments.append(new_seg)
                prev_segment = new_seg
            else:
                new_segments.append(segment)
                prev_segment = segment

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

        writer(result, audio_path, writer_args)


def transcribe_task(args: dict, parser: argparse.ArgumentParser):
    """Transcription task to be called from CLI.

    Args:
        args: Dictionary of command-line arguments.
        parser: argparse.ArgumentParser object.
    """
    from cantocaptions_ai.pipeline.config import PipelineConfig

    audio_paths = args.pop("audio")
    cfg = PipelineConfig.from_args(args)
    os.makedirs(cfg.output_dir, exist_ok=True)

    if cfg.speaker_embeddings and not cfg.diarize:
        warnings.warn("--speaker_embeddings has no effect without --diarize")

    if cfg.reference_subtitle and not cfg.llm_correction:
        parser.error("--reference_subtitle requires --llm_correction")
    if cfg.reference_correction_semantic and not cfg.reference_subtitle:
        warnings.warn("--reference_correction_semantic has no effect without --reference_subtitle")

    if cfg.language is not None:
        cfg.language = cfg.language.lower()
        if cfg.language not in LANGUAGES:
            if cfg.language in TO_LANGUAGE_CODE:
                cfg.language = TO_LANGUAGE_CODE[cfg.language]
            else:
                raise ValueError(f"Unsupported language: {cfg.language}")

    if cfg.language != "yue":
        warnings.warn(
            f"Configured language '{cfg.language}' is not yue/cantonese, and may not be compatible with this framework."
        )

    align_language = cfg.language if cfg.language is not None else "yue"
    task: str = "transcribe"

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

    writer = get_writer(cfg.output_format, cfg.output_dir)
    word_options = ["highlight_words", "max_line_count", "max_line_width"]
    if cfg.no_align:
        for option in word_options:
            if getattr(cfg, option):
                parser.error(f"--{option} not possible with --no_align")
    if cfg.max_line_count and not cfg.max_line_width:
        warnings.warn("--max_line_count has no effect without --max_line_width")
    writer_args = {
        "highlight_words": cfg.highlight_words,
        "max_line_count": cfg.max_line_count,
        "max_line_width": cfg.max_line_width,
    }

    # Text cleaning runs on the final merged segments just before writing.
    # Constructed eagerly so bad rule files fail before any model inference.
    cleaner = None
    if cfg.retime:
        if not cfg.no_clean_text:
            logger.info("Text cleaning skipped: --retime preserves subtitle text")
    elif not cfg.no_clean_text:
        from cantocaptions_ai.cantonese.cleaner import SubtitleCleaner
        cleaner = SubtitleCleaner(
            rules_dir=cfg.clean_rules_dir,
            line_max_length=cfg.max_line_width or 21,
            max_line_count=cfg.max_line_count,
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

    need_asr = not cfg.retime and (
        not cfg.load_debug_dir or any(
            not _debug_stage_exists(ap, "transcription", cfg.load_debug_dir) for ap in audio_paths
        )
    )
    need_vad = not cfg.load_debug_dir or any(
        not _debug_stage_exists(ap, "vad", cfg.load_debug_dir) for ap in audio_paths
    )
    need_vocal_isolation = (
        bool(cfg.vocal_isolation_method) and cfg.vocal_isolation_method.lower() != "none"
        and (not cfg.load_debug_dir or any(
            not _debug_stage_exists(ap, "vocal_isolation", cfg.load_debug_dir) for ap in audio_paths
        ))
    )
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

    summary = TranscriptionSummary(enabled=cfg.print_progress)
    process_start = time.perf_counter()

    # Stage 1: VAD
    with StageTimer("VAD", summary) as stage:
        stub_items = [{'audio_path': p, 'audio_track': _select_audio_track(p)} for p in audio_paths]
        if need_vad:
            from cantocaptions_ai.pipeline.vad import load_vad, VadProcessor
            vad_processor = load_vad(
                vad_method=cfg.vad_method,
                device=cfg.device,
                device_index=cfg.device_index,
                vad_onset=cfg.vad_onset,
                vad_offset=cfg.vad_offset,
                chunk_size=cfg.chunk_size,
                use_auth_token=cfg.hf_token,
            )
            stage.mark_inference_start()
            items = vad_processor.run(stub_items, debug_dir=cfg.debug_dir, load_debug_dir=cfg.load_debug_dir, progress_callback=stage.reporter)
            del vad_processor
        else:
            from cantocaptions_ai.pipeline.vad import VadProcessor
            items = VadProcessor.load_cache(stub_items, cfg.load_debug_dir)

    # Stage 2: Vocal Isolation (conditional)
    vocal_isolation_active = (
        bool(cfg.vocal_isolation_method) and cfg.vocal_isolation_method.lower() != "none"
    )
    if need_vocal_isolation:
        with StageTimer("Vocal isolation", summary) as stage:
            from cantocaptions_ai.pipeline.vocal_isolation import load_vocal_isolation
            vocal_isolation_processor = load_vocal_isolation(
                model_name=cfg.vocal_isolation_method,
                device=cfg.device,
                device_index=cfg.device_index,
                batch_size=cfg.vocal_isolation_batch_size,
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

    if cfg.retime:
        # Retime mode: skip ASR entirely; use the alignment model for both search and fine alignment.
        with StageTimer("Subtitle retiming + alignment", summary) as stage:
            from transformers import Wav2Vec2BertProcessor
            from cantocaptions_ai.pipeline.alignment import load_align_model
            bert_processor = Wav2Vec2BertProcessor.from_pretrained("alvanlii/wav2vec2-BERT-cantonese")
            align_model, align_metadata = load_align_model(
                align_language, cfg.device, cfg.device_index,
                model_name=cfg.align_model, model_dir=cfg.model_dir, model_cache_only=cfg.model_cache_only,
            )
            stage.mark_inference_start()
            items = _run_retime(items, cfg.retime, align_model, align_metadata, bert_processor, cfg.device)
            if not cfg.no_align:
                items = _run_alignment(
                    items, align_model, align_metadata, bert_processor, cfg.device,
                    cfg.align_padding, cfg.align_release, cfg.interpolate_method,
                    cfg.return_char_alignments, cfg.print_progress, cfg.batch_size,
                    progress_callback=stage.reporter,
                )
            else:
                items = _extract_timestamps(items)
        del align_model, bert_processor
        flush_vram()
    else:
        # Stage 3: Transcription
        if need_asr:
            from cantocaptions_ai.pipeline.asr import load_model
            with StageTimer("Transcription", summary) as stage:
                with model_scope(
                    load_model,
                    cfg.model,
                    device=cfg.device,
                    device_index=cfg.device_index,
                    download_root=cfg.model_dir,
                    compute_type=cfg.compute_type,
                    attn_implementation=cfg.attn_implementation,
                    language=cfg.language,
                    asr_options=asr_options,
                    vocal_isolation_method=cfg.vocal_isolation_method,
                    task=task,
                    local_files_only=cfg.model_cache_only,
                    threads=qwen_threads,
                    use_auth_token=cfg.hf_token,
                    batch_size=cfg.batch_size,
                    print_progress=cfg.print_progress,
                    verbose=cfg.verbose,
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
                with StageTimer("Ensemble ASR (faster-whisper)", summary) as stage:
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
            if cfg.reference_subtitle:
                from cantocaptions_ai.pipeline.retime import load_subtitle_file
                from cantocaptions_ai.pipeline.llm_correction import match_reference_to_segments
                with StageTimer("Reference subtitle matching", summary):
                    logger.info(f"Loading reference subtitle: {cfg.reference_subtitle}")
                    reference_subs = load_subtitle_file(cfg.reference_subtitle)
                    logger.info(f"Loaded {len(reference_subs)} reference subtitle lines.")
                    for item in items:
                        item['reference_texts'] = match_reference_to_segments(
                            item['result']['segments'], reference_subs
                        )

            if need_llm:
                from cantocaptions_ai.pipeline.llm_correction import load_llm
                stats = vram_stats()
                if stats:
                    logger.info(
                        f"VRAM before LLM load: allocated={stats['allocated_mb']:.0f} MB, "
                        f"reserved={stats['reserved_mb']:.0f} MB, "
                        f"free={stats['free_mb']:.0f} MB / {stats['total_mb']:.0f} MB"
                    )
                with StageTimer("LLM correction", summary) as stage:
                    with model_scope(
                        load_llm,
                        model_id=cfg.llm_model,
                        model_dir=cfg.llm_model_dir,
                        device=cfg.device,
                        local_files_only=cfg.model_cache_only,
                        semantic_mode=cfg.reference_correction_semantic,
                        attn_implementation=cfg.attn_implementation,
                    ) as corrector:
                        stage.mark_inference_start()
                        items = corrector.run(items, debug_dir=cfg.debug_dir, load_debug_dir=cfg.load_debug_dir, progress_callback=stage.reporter)
            else:
                from cantocaptions_ai.pipeline.llm_correction import LLMCorrector
                items = LLMCorrector.load_cache(items, cfg.load_debug_dir)

        # Stage 4: Alignment
        if not cfg.no_align:
            with StageTimer("Alignment", summary) as stage:
                from transformers import Wav2Vec2BertProcessor
                from cantocaptions_ai.pipeline.alignment import load_align_model
                bert_processor = Wav2Vec2BertProcessor.from_pretrained("alvanlii/wav2vec2-BERT-cantonese")
                align_model, align_metadata = load_align_model(
                    align_language, cfg.device, cfg.device_index,
                    model_name=cfg.align_model, model_dir=cfg.model_dir, model_cache_only=cfg.model_cache_only,
                )
                stage.mark_inference_start()
                items = _run_alignment(
                    items, align_model, align_metadata, bert_processor, cfg.device,
                    cfg.align_padding, cfg.align_release, cfg.interpolate_method,
                    cfg.return_char_alignments, cfg.print_progress, cfg.batch_size,
                    progress_callback=stage.reporter,
                )
            del align_model, bert_processor
            flush_vram()
        else:
            items = _extract_timestamps(items)

    # Stage 5: Diarization
    if cfg.diarize:
        if cfg.hf_token is None:
            logger.warning(
                "No --hf_token provided, needs to be saved in environment variable, otherwise will throw error loading diarization model"
            )
        with StageTimer("Diarization", summary) as stage:
            items = _run_diarization(
                items, cfg.diarize_model, cfg.hf_token, cfg.device, cfg.model_dir,
                cfg.min_speakers, cfg.max_speakers, cfg.speaker_embeddings,
                progress_callback=stage.reporter,
                load_complete_callback=stage.mark_inference_start,
            )

    # Stage 6: Speaker Verification
    if cfg.verify_speakers:
        with StageTimer("Speaker verification", summary) as stage:
            items = _run_speaker_verification(
                items, cfg.device, cfg.model_dir,
                progress_callback=stage.reporter,
                load_complete_callback=stage.mark_inference_start,
            )

    # Write
    _merge_and_write(
        items, writer, align_language, cfg.align_merge_distance, cfg.align_padding, writer_args,
        cleaner=cleaner, debug_dir=cfg.debug_dir,
    )

    summary.print_summary(process_elapsed=time.perf_counter() - process_start)

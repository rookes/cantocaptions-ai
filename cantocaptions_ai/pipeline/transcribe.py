import argparse
import os
import warnings

import numpy as np
import torch

from cantocaptions_ai.utils.audio import load_audio, SAMPLE_RATE
from cantocaptions_ai.utils.schema import AlignedTranscriptionResult, ProcessingItem, ProgressCallback, TranscriptionResult, VadItem, merge_segments
from typing import Callable, List, Optional
from cantocaptions_ai.utils.output import LANGUAGES, TO_LANGUAGE_CODE, get_writer
from cantocaptions_ai.utils.log_utils import StageTimer, TranscriptionSummary, get_logger
from cantocaptions_ai.utils.model_utils import model_scope, flush_vram
from cantocaptions_ai.cantonese.text import is_mergeable, normalize_segment_text
from cantocaptions_ai.utils.debug import (
    write_vad_debug, write_isolation_debug, write_transcription_debug,
    write_ensemble_debug, write_llm_correction_debug,
    load_vad_debug, load_isolation_debug, load_transcription_debug,
    load_ensemble_debug, load_llm_correction_debug,
    _debug_stage_exists,
)

logger = get_logger(__name__)


def _load_or_compute(audio_path, load_debug_dir, debug_dir, load_fn, write_fn, compute_fn):
    """Load a stage result from debug cache, or compute and optionally save it."""
    if load_debug_dir:
        cached = load_fn(audio_path, load_debug_dir)
        if cached is not None:
            return cached
    result = compute_fn()
    if debug_dir:
        write_fn(audio_path, result, debug_dir)
    return result


def _run_vad(
    audio_paths: List[str],
    vad_processor,
    debug_dir: Optional[str] = None,
    load_debug_dir: Optional[str] = None,
) -> List[VadItem]:
    items: List[VadItem] = []
    for audio_path in audio_paths:
        def _compute():
            audio = load_audio(audio_path)
            logger.info("Performing voice activity detection...")
            return vad_processor.process(audio)
        vad_segments = _load_or_compute(
            audio_path, load_debug_dir, debug_dir,
            load_vad_debug, write_vad_debug, _compute,
        )
        items.append({'audio_path': audio_path, 'vad_segments': vad_segments})
    return items


def _run_isolation(
    items: List[VadItem],
    vocal_isolation_processor,
    debug_dir: Optional[str] = None,
    load_debug_dir: Optional[str] = None,
    progress_callback: ProgressCallback = None,
) -> List[VadItem]:
    result_items: List[VadItem] = []
    for item in items:
        audio_path = item['audio_path']
        def _compute():
            logger.info("Performing vocal isolation...")
            return vocal_isolation_processor.process(item['vad_segments'], progress_callback=progress_callback)
        vad_segments = _load_or_compute(
            audio_path, load_debug_dir, debug_dir,
            load_isolation_debug, write_isolation_debug, _compute,
        )
        result_items.append({'audio_path': audio_path, 'vad_segments': vad_segments})
    return result_items


def _run_transcription(
    items: List[VadItem],
    model,
    debug_dir: Optional[str] = None,
    load_debug_dir: Optional[str] = None,
    progress_callback: ProgressCallback = None,
) -> List[ProcessingItem]:
    result_items: List[ProcessingItem] = []
    for item in items:
        audio_path = item['audio_path']
        def _compute():
            logger.info("Performing transcription...")
            result: TranscriptionResult = model.process(
                item['vad_segments'],
                progress_callback=progress_callback,
            )
            return {**result, 'segments': [normalize_segment_text(seg) for seg in result['segments']]}
        result = _load_or_compute(
            audio_path, load_debug_dir, debug_dir,
            load_transcription_debug, write_transcription_debug, _compute,
        )
        result_items.append({'audio_path': audio_path, 'result': result, 'vad_segments': item['vad_segments']})
    return result_items


def _run_ensemble(
    items: List[ProcessingItem],
    ensemble_model,
    debug_dir: Optional[str] = None,
    load_debug_dir: Optional[str] = None,
    progress_callback: ProgressCallback = None,
) -> List[ProcessingItem]:
    result_items: List[ProcessingItem] = []
    for item in items:
        audio_path = item['audio_path']
        def _compute():
            logger.info("Running ensemble ASR (faster-whisper)...")
            return ensemble_model.process(
                item['vad_segments'],
                progress_callback=progress_callback,
            )
        alt_texts = _load_or_compute(
            audio_path, load_debug_dir, debug_dir,
            load_ensemble_debug, write_ensemble_debug, _compute,
        )
        result_items.append({**item, 'ensemble_texts': alt_texts})
    return result_items


def _run_llm_correction(
    items: List[ProcessingItem],
    corrector,
    debug_dir: Optional[str] = None,
    load_debug_dir: Optional[str] = None,
    progress_callback: ProgressCallback = None,
) -> List[ProcessingItem]:
    result_items: List[ProcessingItem] = []
    for item in items:
        audio_path = item['audio_path']
        def _compute():
            logger.info("Running LLM correction...")
            segments = item['result']['segments']
            ensemble_texts = item.get('ensemble_texts')
            corrected_texts = corrector.correct_segments(
                segments,
                ensemble_texts=ensemble_texts,
                progress_callback=progress_callback,
            )
            corrected_texts = corrector.normalize_names(corrected_texts)
            new_segments = [{**seg, 'text': corrected_texts[i]} for i, seg in enumerate(segments)]
            return {**item['result'], 'segments': new_segments}
        result = _load_or_compute(
            audio_path, load_debug_dir, debug_dir,
            load_llm_correction_debug, write_llm_correction_debug, _compute,
        )
        result_items.append({**item, 'result': result})
    return result_items


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
) -> List[ProcessingItem]:
    from cantocaptions_ai.pipeline.diarize import DiarizationPipeline, assign_word_speakers
    logger.info("Performing diarization...")
    logger.info(f"Using model: {diarize_model_name}")
    diarize_model = DiarizationPipeline(model_name=diarize_model_name, token=hf_token, device=device, cache_dir=model_dir)

    diarized_items = []
    for item in items:
        diarize_result = diarize_model(
            item['audio_path'],
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            return_embeddings=return_speaker_embeddings,
            progress_callback=progress_callback,
        )
        if return_speaker_embeddings:
            diarize_segments, speaker_embeddings = diarize_result
        else:
            diarize_segments, speaker_embeddings = diarize_result, None
        result = assign_word_speakers(diarize_segments, item['result'], speaker_embeddings)
        diarized_items.append({'audio_path': item['audio_path'], 'result': result})
    return diarized_items


def _run_speaker_verification(
    items: List[ProcessingItem],
    device: str,
    model_dir: Optional[str],
    progress_callback: ProgressCallback = None,
) -> List[ProcessingItem]:
    from cantocaptions_ai.pipeline.speaker_verification import SpeakerVerificationPipeline
    logger.info("Performing speaker verification...")
    speaker_verification_model = SpeakerVerificationPipeline(device=device, cache_dir=model_dir)

    verified_items = []
    for item in items:
        audio = load_audio(item['audio_path'])
        audio_segments = [
            audio[int(seg["start"] * SAMPLE_RATE):int(seg["end"] * SAMPLE_RATE)]
            for seg in item['result']["segments"]
        ]
        speaker_verification_model(transcript=item['result']["segments"], audio=audio_segments, progress_callback=progress_callback)
        verified_items.append(item)
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

    if cfg.load_debug_dir:
        from pathlib import Path as _Path
        for ap in audio_paths:
            stem_dir = os.path.join(cfg.load_debug_dir, _Path(ap).stem)
            if not os.path.isdir(stem_dir):
                parser.error(f"No debug data found for '{ap}': directory '{stem_dir}' does not exist")

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

    # Stage 1: VAD loading & processing
    with StageTimer("VAD loading & processing", summary):
        if need_vad:
            from cantocaptions_ai.pipeline.vad import load_vad
            vad_processor = load_vad(
                vad_method=cfg.vad_method,
                device=cfg.device,
                device_index=cfg.device_index,
                vad_onset=cfg.vad_onset,
                vad_offset=cfg.vad_offset,
                chunk_size=cfg.chunk_size,
                use_auth_token=cfg.hf_token,
            )
        else:
            vad_processor = None
        items = _run_vad(audio_paths, vad_processor, debug_dir=cfg.debug_dir, load_debug_dir=cfg.load_debug_dir)

    # Stage 2: Vocal Isolation (conditional)
    vocal_isolation_processor = None
    if need_vocal_isolation:
        with StageTimer("Vocal isolation", summary) as stage:
            from cantocaptions_ai.pipeline.vocal_isolation import load_vocal_isolation
            vocal_isolation_processor = load_vocal_isolation(
                model_name=cfg.vocal_isolation_method,
                device=cfg.device,
                device_index=cfg.device_index,
            )
            items = _run_isolation(items, vocal_isolation_processor, debug_dir=cfg.debug_dir, load_debug_dir=cfg.load_debug_dir, progress_callback=stage.callback)

    # Free VAD and isolation models before loading ASR to maximise available GPU memory.
    if vad_processor is not None:
        del vad_processor
    if vocal_isolation_processor is not None:
        del vocal_isolation_processor
    flush_vram()

    if cfg.retime:
        # Retime mode: skip ASR entirely; use the alignment model for both search and fine alignment.
        from transformers import Wav2Vec2BertProcessor
        from cantocaptions_ai.pipeline.alignment import load_align_model
        bert_processor = Wav2Vec2BertProcessor.from_pretrained("alvanlii/wav2vec2-BERT-cantonese")
        align_model, align_metadata = load_align_model(
            align_language, cfg.device, cfg.device_index,
            model_name=cfg.align_model, model_dir=cfg.model_dir, model_cache_only=cfg.model_cache_only,
        )
        with StageTimer("Subtitle retiming + alignment", summary) as stage:
            items = _run_retime(items, cfg.retime, align_model, align_metadata, bert_processor, cfg.device)
            if not cfg.no_align:
                items = _run_alignment(
                    items, align_model, align_metadata, bert_processor, cfg.device,
                    cfg.align_padding, cfg.align_release, cfg.interpolate_method,
                    cfg.return_char_alignments, cfg.print_progress, cfg.batch_size,
                    progress_callback=stage.callback,
                )
            else:
                items = _extract_timestamps(items)
        del align_model, bert_processor
        flush_vram()
    else:
        # Stage 3: Transcription
        load_model = None
        if need_asr:
            from cantocaptions_ai.pipeline.asr import load_model
        with model_scope(
            load_model,
            cfg.model,
            device=cfg.device,
            device_index=cfg.device_index,
            download_root=cfg.model_dir,
            compute_type=cfg.compute_type,
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
            with StageTimer("Transcription", summary) as stage:
                items = _run_transcription(
                    items, model,
                    debug_dir=cfg.debug_dir, load_debug_dir=cfg.load_debug_dir,
                    progress_callback=stage.callback,
                )

        # Stage 3b: Ensemble ASR (optional)
        if cfg.ensemble_model != "none":
            load_faster_whisper = None
            if need_ensemble:
                from cantocaptions_ai.pipeline.ensemble import load_faster_whisper
            with model_scope(
                load_faster_whisper,
                device=cfg.device,
                device_index=cfg.device_index,
                model_dir=cfg.model_dir,
                local_files_only=cfg.model_cache_only,
            ) as ensemble:
                with StageTimer("Ensemble ASR (faster-whisper)", summary) as stage:
                    items = _run_ensemble(items, ensemble, debug_dir=cfg.debug_dir, load_debug_dir=cfg.load_debug_dir, progress_callback=stage.callback)

        # Stage 3c: LLM correction (optional)
        if cfg.llm_correction:
            load_llm = None
            if need_llm:
                from cantocaptions_ai.pipeline.llm_correction import load_llm
            with model_scope(
                load_llm,
                model_id=cfg.llm_model,
                model_dir=cfg.llm_model_dir,
                device=cfg.device,
                local_files_only=cfg.model_cache_only,
            ) as corrector:
                with StageTimer("LLM correction", summary) as stage:
                    items = _run_llm_correction(items, corrector, debug_dir=cfg.debug_dir, load_debug_dir=cfg.load_debug_dir, progress_callback=stage.callback)

        # Stage 4: Alignment
        if not cfg.no_align:
            from transformers import Wav2Vec2BertProcessor
            from cantocaptions_ai.pipeline.alignment import load_align_model
            bert_processor = Wav2Vec2BertProcessor.from_pretrained("alvanlii/wav2vec2-BERT-cantonese")
            align_model, align_metadata = load_align_model(
                align_language, cfg.device, cfg.device_index,
                model_name=cfg.align_model, model_dir=cfg.model_dir, model_cache_only=cfg.model_cache_only,
            )
            with StageTimer("Alignment", summary) as stage:
                items = _run_alignment(
                    items, align_model, align_metadata, bert_processor, cfg.device,
                    cfg.align_padding, cfg.align_release, cfg.interpolate_method,
                    cfg.return_char_alignments, cfg.print_progress, cfg.batch_size,
                    progress_callback=stage.callback,
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
                progress_callback=stage.callback,
            )

    # Stage 6: Speaker Verification
    if cfg.verify_speakers:
        with StageTimer("Speaker verification", summary) as stage:
            items = _run_speaker_verification(items, cfg.device, cfg.model_dir, progress_callback=stage.callback)

    # Write
    _merge_and_write(items, writer, align_language, cfg.align_merge_distance, cfg.align_padding, writer_args)

    summary.print_summary()

import argparse
import gc
import os
import time
import warnings

import numpy as np
import torch

from cantocaptions_ai.utils.audio import load_audio, SAMPLE_RATE
from cantocaptions_ai.utils.schema import AlignedTranscriptionResult, ProgressCallback, TranscriptionResult
from cantocaptions_ai.utils.output import LANGUAGES, TO_LANGUAGE_CODE, get_writer, merge_segments
from cantocaptions_ai.utils.log_utils import StageTimer, TranscriptionSummary, get_logger
from cantocaptions_ai.cantonese.text import is_mergeable, normalize_segment_text
from cantocaptions_ai.utils.debug import (
    write_vad_debug, write_isolation_debug, write_transcription_debug,
    write_ensemble_debug, write_llm_correction_debug,
    load_vad_debug, load_isolation_debug, load_transcription_debug,
    load_ensemble_debug, load_llm_correction_debug,
    _debug_stage_exists,
)

logger = get_logger(__name__)


def _run_vad(
    audio_paths: list,
    vad_processor,
    debug_dir: str = None,
    load_debug_dir: str = None,
) -> list:
    items = []
    for audio_path in audio_paths:
        vad_loaded = load_vad_debug(audio_path, load_debug_dir) if load_debug_dir else None
        if vad_loaded is not None:
            vad_segments = vad_loaded
        else:
            audio = load_audio(audio_path)
            logger.info("Performing voice activity detection...")
            vad_segments = vad_processor.segment(audio)
            if debug_dir:
                write_vad_debug(audio_path, vad_segments, debug_dir)
        items.append({'audio_path': audio_path, 'vad_segments': vad_segments})
    return items


def _run_isolation(
    items: list,
    vocal_isolation_processor,
    debug_dir: str = None,
    load_debug_dir: str = None,
    progress_callback: ProgressCallback = None,
) -> list:
    result_items = []
    for item in items:
        audio_path = item['audio_path']
        isol_loaded = load_isolation_debug(audio_path, load_debug_dir) if load_debug_dir else None
        if isol_loaded is not None:
            vad_segments = isol_loaded
        else:
            logger.info("Performing vocal isolation...")
            vad_segments = vocal_isolation_processor.isolate(item['vad_segments'], progress_callback=progress_callback)
            if debug_dir:
                write_isolation_debug(audio_path, vad_segments, debug_dir)
        result_items.append({'audio_path': audio_path, 'vad_segments': vad_segments})
    return result_items


def _run_transcription(
    items: list,
    model,
    batch_size: int,
    print_progress: bool,
    verbose: bool,
    debug_dir: str = None,
    load_debug_dir: str = None,
    progress_callback: ProgressCallback = None,
) -> list:
    result_items = []
    for item in items:
        audio_path = item['audio_path']
        trans_loaded = load_transcription_debug(audio_path, load_debug_dir) if load_debug_dir else None
        if trans_loaded is not None:
            result = trans_loaded
        else:
            logger.info("Performing transcription...")
            result: TranscriptionResult = model.transcribe(
                item['vad_segments'],
                batch_size=batch_size,
                print_progress=print_progress,
                verbose=verbose,
                progress_callback=progress_callback,
                use_native=False,  # set True to test Qwen's own transcribe()
            )
        result_items.append({'audio_path': audio_path, 'result': result, 'vad_segments': item['vad_segments']})
    return result_items


def _run_ensemble(
    items: list,
    ensemble_model,
    debug_dir: str = None,
    load_debug_dir: str = None,
    progress_callback: ProgressCallback = None,
) -> list:
    result_items = []
    for item in items:
        audio_path = item['audio_path']
        loaded = load_ensemble_debug(audio_path, load_debug_dir) if load_debug_dir else None
        if loaded is not None:
            alt_texts = loaded
        else:
            logger.info("Running ensemble ASR (faster-whisper)...")
            alt_texts = ensemble_model.transcribe_segments(
                item['vad_segments'],
                progress_callback=progress_callback,
            )
            if debug_dir:
                write_ensemble_debug(audio_path, alt_texts, debug_dir)
        result_items.append({**item, 'ensemble_texts': alt_texts})
    return result_items


def _run_llm_correction(
    items: list,
    corrector,
    debug_dir: str = None,
    load_debug_dir: str = None,
    progress_callback: ProgressCallback = None,
) -> list:
    result_items = []
    for item in items:
        audio_path = item['audio_path']
        loaded = load_llm_correction_debug(audio_path, load_debug_dir) if load_debug_dir else None
        if loaded is not None:
            result = loaded
        else:
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
            result = {**item['result'], 'segments': new_segments}
            if debug_dir:
                write_llm_correction_debug(audio_path, result, debug_dir)
        result_items.append({**item, 'result': result})
    return result_items


def _normalize_cantonese(items: list) -> list:
    return [
        {**item, 'result': {**item['result'], 'segments': [normalize_segment_text(seg) for seg in item['result']['segments']]}}
        for item in items
    ]


def _run_alignment(
    items: list,
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
) -> list:
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


def _extract_timestamps(items: list) -> list:
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
    items: list,
    diarize_model_name: str,
    hf_token: str,
    device: str,
    model_dir: str,
    min_speakers: int,
    max_speakers: int,
    return_speaker_embeddings: bool,
    progress_callback: ProgressCallback = None,
) -> list:
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
    items: list,
    device: str,
    model_dir: str,
    progress_callback: ProgressCallback = None,
) -> list:
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


def _merge_and_write(
    items: list,
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

    audio_start: float = args.pop("audio_start")
    audio_end: float = args.pop("audio_end")

    model_name: str = args.pop("model")
    batch_size: int = args.pop("batch_size")
    model_dir: str = args.pop("model_dir")
    model_cache_only: bool = args.pop("model_cache_only")
    output_dir: str = args.pop("output_dir")
    output_format: str = args.pop("output_format")
    device: str = args.pop("device")
    device_index: int = args.pop("device_index")
    compute_type: str = args.pop("compute_type")
    verbose: bool = args.pop("verbose")

    # model_flush: bool = args.pop("model_flush")
    os.makedirs(output_dir, exist_ok=True)

    align_model_name: str = args.pop("align_model")
    interpolate_method: str = args.pop("interpolate_method")
    no_align: bool = args.pop("no_align")
    task: str = "transcribe"

    return_char_alignments: bool = args.pop("return_char_alignments")

    hf_token: str = args.pop("hf_token")
    vad_method: str = args.pop("vad_method")
    vad_onset: float = args.pop("vad_onset")
    vad_offset: float = args.pop("vad_offset")

    chunk_size: int = args.pop("chunk_size")

    vocal_isolation_method: str = args.pop("vocal_isolation_method")

    align_padding: float = args.pop("align_padding")
    align_release: float = args.pop("align_release")
    align_merge_distance: float = args.pop("align_merge_distance")
    diarize_merge: bool = args.pop("diarize_merge")

    diarize: bool = args.pop("diarize")
    min_speakers: int = args.pop("min_speakers")
    max_speakers: int = args.pop("max_speakers")
    diarize_model_name: str = args.pop("diarize_model")
    print_progress: bool = args.pop("print_progress")
    return_speaker_embeddings: bool = args.pop("speaker_embeddings")

    verify_speakers: bool = args.pop("verify_speakers")
    debug_dir: str = args.pop("debug_dir", None)
    load_debug_dir: str = args.pop("load_debug_dir", None)

    ensemble_model_name: str = args.pop("ensemble_model")
    llm_correction: bool = args.pop("llm_correction")
    llm_model: str = args.pop("llm_model")
    llm_model_dir = args.pop("llm_model_dir")

    if return_speaker_embeddings and not diarize:
        warnings.warn("--speaker_embeddings has no effect without --diarize")

    if args["language"] is not None:
        args["language"] = args["language"].lower()
        if args["language"] not in LANGUAGES:
            if args["language"] in TO_LANGUAGE_CODE:
                args["language"] = TO_LANGUAGE_CODE[args["language"]]
            else:
                raise ValueError(f"Unsupported language: {args['language']}")

    if args["language"] != "yue":
        warnings.warn(
            f"Configured language '{args['language']}' is not yue/cantonese, and may not be compatibile with this framework."
        )

    align_language = args["language"] if args["language"] is not None else "yue"

    qwen_threads = torch.get_num_threads()
    if (threads := args.pop("threads")) > 0:
        torch.set_num_threads(threads)
        qwen_threads = threads

    asr_options = {
        "condition_on_previous_text": False,
        "initial_prompt": args.pop("initial_prompt"),
        "hotwords": args.pop("hotwords"),
        "suppress_tokens": [int(x) for x in args.pop("suppress_tokens").split(",")],
        "suppress_numerals": args.pop("suppress_numerals"),
    }

    writer = get_writer(output_format, output_dir)
    word_options = ["highlight_words", "max_line_count", "max_line_width"]
    if no_align:
        for option in word_options:
            if args[option]:
                parser.error(f"--{option} not possible with --no_align")
    if args["max_line_count"] and not args["max_line_width"]:
        warnings.warn("--max_line_count has no effect without --max_line_width")
    writer_args = {arg: args.pop(arg) for arg in word_options}

    audio_paths = args.pop("audio")

    if load_debug_dir:
        from pathlib import Path as _Path
        for ap in audio_paths:
            stem_dir = os.path.join(load_debug_dir, _Path(ap).stem)
            if not os.path.isdir(stem_dir):
                parser.error(f"No debug data found for '{ap}': directory '{stem_dir}' does not exist")

    need_asr = not load_debug_dir or any(
        not _debug_stage_exists(ap, "transcription", load_debug_dir) for ap in audio_paths
    )
    need_vad = not load_debug_dir or any(
        not _debug_stage_exists(ap, "vad", load_debug_dir) for ap in audio_paths
    )
    need_vocal_isolation = (
        bool(vocal_isolation_method) and vocal_isolation_method.lower() != "none"
        and (not load_debug_dir or any(
            not _debug_stage_exists(ap, "vocal_isolation", load_debug_dir) for ap in audio_paths
        ))
    )
    need_ensemble = (
        ensemble_model_name != "none"
        and (not load_debug_dir or any(
            not _debug_stage_exists(ap, "ensemble", load_debug_dir) for ap in audio_paths
        ))
    )
    need_llm = (
        llm_correction
        and (not load_debug_dir or any(
            not _debug_stage_exists(ap, "llm_correction", load_debug_dir) for ap in audio_paths
        ))
    )

    summary = TranscriptionSummary(enabled=print_progress)

    # Stage 1: VAD loading & processing
    with StageTimer("VAD loading & processing", summary):
        if need_vad:
            from cantocaptions_ai.pipeline.vad import load_vad
            vad_processor = load_vad(
                vad_method=vad_method,
                device=device,
                device_index=device_index,
                vad_onset=vad_onset,
                vad_offset=vad_offset,
                chunk_size=chunk_size,
                use_auth_token=hf_token,
            )
        else:
            vad_processor = None
        items = _run_vad(audio_paths, vad_processor, debug_dir=debug_dir, load_debug_dir=load_debug_dir)

    # Stage 2: Vocal Isolation (conditional)
    vocal_isolation_processor = None
    if need_vocal_isolation:
        with StageTimer("Vocal isolation", summary) as stage:
            from cantocaptions_ai.pipeline.vocal_isolation import load_vocal_isolation
            vocal_isolation_processor = load_vocal_isolation(
                model_name=vocal_isolation_method,
                device=device,
                device_index=device_index,
            )
            items = _run_isolation(items, vocal_isolation_processor, debug_dir=debug_dir, load_debug_dir=load_debug_dir, progress_callback=stage.callback)

    t0 = time.time()

    # Free VAD and isolation models before loading ASR to maximise available GPU memory.
    if vad_processor is not None:
        del vad_processor
        vad_processor = None
    if vocal_isolation_processor is not None:
        del vocal_isolation_processor
        vocal_isolation_processor = None
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[TIMING] Model cleanup: {time.time() - t0:.1f}s")  # line 459

    # Stage 3: Transcription
    model = None

    t1 = time.time()
    with StageTimer("Transcription", summary) as stage:
        if need_asr:
            from cantocaptions_ai.pipeline.asr import load_model
            model = load_model(
                model_name,
                device=device,
                device_index=device_index,
                download_root=model_dir,
                compute_type=compute_type,
                language=args["language"],
                asr_options=asr_options,
                vocal_isolation_method=vocal_isolation_method,
                task=task,
                local_files_only=model_cache_only,
                threads=qwen_threads,
                use_auth_token=hf_token,
            )

            print(f"[TIMING] load_model: {time.time() - t1:.1f}s")     # line after load_model

        t2 = time.time()
        items = _run_transcription(items, model, batch_size, print_progress, verbose, debug_dir=debug_dir, load_debug_dir=load_debug_dir, progress_callback=stage.callback)
        print(f"[TIMING] _run_transcription: {time.time() - t2:.1f}s")

    items = _normalize_cantonese(items)
    if debug_dir:
        for item in items:
            write_transcription_debug(item['audio_path'], item['result'], debug_dir)

    if model is not None:
        del model
    gc.collect()
    torch.cuda.empty_cache()

    # Stage 3b: Ensemble ASR (optional)
    if ensemble_model_name != "none":
        with StageTimer("Ensemble ASR (faster-whisper)", summary) as stage:
            ensemble = None
            if need_ensemble:
                from cantocaptions_ai.pipeline.ensemble import load_faster_whisper
                ensemble = load_faster_whisper(
                    device=device,
                    device_index=device_index,
                    model_dir=model_dir,
                    local_files_only=model_cache_only,
                )
            items = _run_ensemble(items, ensemble, debug_dir=debug_dir, load_debug_dir=load_debug_dir, progress_callback=stage.callback)
        if ensemble is not None:
            del ensemble
        gc.collect()
        torch.cuda.empty_cache()

    # Stage 3c: LLM correction (optional)
    if llm_correction:
        with StageTimer("LLM correction", summary) as stage:
            corrector = None
            if need_llm:
                from cantocaptions_ai.pipeline.llm_correction import load_llm
                corrector = load_llm(
                    model_id=llm_model,
                    model_dir=llm_model_dir,
                    device=device,
                    local_files_only=model_cache_only,
                )
            items = _run_llm_correction(items, corrector, debug_dir=debug_dir, load_debug_dir=load_debug_dir, progress_callback=stage.callback)
        if corrector is not None:
            del corrector
        gc.collect()
        torch.cuda.empty_cache()

    # Stage 4: Alignment
    if not no_align:
        with StageTimer("Alignment", summary) as stage:
            from transformers import Wav2Vec2BertProcessor
            from cantocaptions_ai.pipeline.alignment import load_align_model
            bert_processor = Wav2Vec2BertProcessor.from_pretrained("alvanlii/wav2vec2-BERT-cantonese")
            align_model, align_metadata = load_align_model(
                align_language, device, model_name=align_model_name, model_dir=model_dir, model_cache_only=model_cache_only
            )
            items = _run_alignment(
                items, align_model, align_metadata, bert_processor, device,
                align_padding, align_release, interpolate_method, return_char_alignments, print_progress, batch_size,
                progress_callback=stage.callback,
            )
        del align_model, bert_processor
        gc.collect()
        torch.cuda.empty_cache()
    else:
        items = _extract_timestamps(items)

    # Stage 5: Diarization
    if diarize:
        if hf_token is None:
            logger.warning(
                "No --hf_token provided, needs to be saved in environment variable, otherwise will throw error loading diarization model"
            )
        with StageTimer("Diarization", summary) as stage:
            items = _run_diarization(items, diarize_model_name, hf_token, device, model_dir, min_speakers, max_speakers, return_speaker_embeddings, progress_callback=stage.callback)

    # Stage 6: Speaker Verification
    if verify_speakers:
        with StageTimer("Speaker verification", summary) as stage:
            items = _run_speaker_verification(items, device, model_dir, progress_callback=stage.callback)

    # Write
    _merge_and_write(items, writer, align_language, align_merge_distance, align_padding, writer_args)

    summary.print_summary()

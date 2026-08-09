"""Library/server entry point for the cantocaptions-ai pipeline.

The CLI (``cantocaptions_ai.__main__``) is optimized for a terminal: it takes an
argparse namespace, writes results to disk, and turns bad input into
``parser.error`` (which calls ``sys.exit``). That contract is hostile to a hosted
service, where a bad request must raise a catchable exception (not kill the worker)
and the result must come back in memory (not only as a file on disk).

This module is that server-friendly surface:

* :func:`run_pipeline` — run one file through the pipeline from a
  :class:`~cantocaptions_ai.pipeline.config.PipelineConfig`, returning a
  :class:`PipelineResult` (rendered subtitle text + segments) and raising
  :class:`~cantocaptions_ai.errors.ConfigError` /
  :class:`~cantocaptions_ai.errors.InputError` instead of exiting.
* :class:`PipelineService` — a long-lived holder a worker process instantiates once.
  It serializes GPU access with a lock (the pipeline has process-wide global state,
  so two concurrent jobs in one process are unsafe) and, in ``resident`` mode, keeps
  the VAD model warm across jobs to skip its ~20-30s reload.

The heavy staged logic itself lives in ``pipeline/transcribe.py``
(``_execute_pipeline``); this module only adapts its inputs/outputs.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import List, Optional

from cantocaptions_ai.errors import ConfigError, InputError
from cantocaptions_ai.pipeline.config import PipelineConfig
from cantocaptions_ai.utils.log_utils import ProgressSink, get_logger

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """The in-memory outcome of one :func:`run_pipeline` call.

    ``subtitle_text`` is the rendered document in ``output_format``; ``segments`` is
    the raw list of timed segments (for callers that want to re-render or inspect).
    ``empty`` is True when VAD found no speech — a distinct, non-error outcome the CLI
    silently turns into an empty file, surfaced here so a server can tell the user.
    """
    subtitle_text: str
    output_format: str
    language: str
    segments: List[dict] = field(default_factory=list)
    num_segments: int = 0
    empty: bool = False


def _writer_args(cfg: PipelineConfig) -> dict:
    return {
        "highlight_words": cfg.highlight_words,
        "max_line_count": cfg.max_line_count,
        "max_line_width": cfg.max_line_width,
    }


def run_pipeline(
    audio_path: str,
    cfg: PipelineConfig,
    *,
    progress: Optional[ProgressSink] = None,
    validate_input: bool = True,
    max_duration_s: Optional[float] = None,
    max_bytes: Optional[int] = None,
    _vad_model=None,
) -> PipelineResult:
    """Transcribe a single file and return the result in memory.

    Raises :class:`ConfigError` for an invalid config and :class:`InputError` for an
    unusable input file — neither exits the process. Nothing is written to disk;
    ``cfg.output_dir`` is ignored. ``progress`` (a :class:`ProgressSink`) receives
    per-stage progress events. ``_vad_model`` is an internal hook used by
    :class:`PipelineService` to inject a warm VAD model.
    """
    # Imported lazily: transcribe.py pulls in torch and the whole pipeline graph,
    # which we don't want to load just to import this module.
    from cantocaptions_ai.pipeline.transcribe import (
        _cleanup_temp_files,
        _execute_pipeline,
        _prepare_clips,
        validate_config,
    )
    from cantocaptions_ai.utils.audio import validate_input_file
    from cantocaptions_ai.utils.output import render_result

    if cfg.output_format == "all":
        raise ConfigError(
            "output_format='all' writes multiple files and is unsupported by run_pipeline; "
            "request a single format (srt, vtt, txt, tsv, json)"
        )

    validate_config(cfg)  # raises ConfigError; also normalizes cfg.language in place

    if validate_input:
        validate_input_file(audio_path, max_duration_s=max_duration_s, max_bytes=max_bytes)

    paths, display_paths, temp_files = _prepare_clips([audio_path], cfg)
    try:
        results = _execute_pipeline(
            paths, cfg,
            progress=progress,
            collect=True,
            audio_start_offset=cfg.audio_start or 0.0,
            display_paths=display_paths,
            vad_model=_vad_model,
        )
    finally:
        _cleanup_temp_files(temp_files)

    language = cfg.language or "yue"
    if not results:
        return PipelineResult(
            subtitle_text="", output_format=cfg.output_format, language=language,
            segments=[], num_segments=0, empty=True,
        )

    result = results[0]["result"]
    segments = result.get("segments", [])
    subtitle_text = render_result(result, cfg.output_format, _writer_args(cfg))
    return PipelineResult(
        subtitle_text=subtitle_text,
        output_format=cfg.output_format,
        language=result.get("language", language),
        segments=segments,
        num_segments=len(segments),
        empty=len(segments) == 0,
    )


class PipelineService:
    """Long-lived pipeline holder for a worker process.

    Instantiate **once** per worker and call :meth:`run` per job. Two things it
    provides that ``run_pipeline`` alone does not:

    * **Serialized GPU access** — the pipeline mutates process-wide state
      (``torch.set_num_threads``, CUDA per-process memory fraction, HF offline env,
      the shared logger), so concurrent jobs in one process corrupt each other. A
      lock makes ``run`` safe to call from multiple threads (it just queues them).
    * **Warm VAD model** (``resident=True``) — the VAD model load is the single
      largest per-job reload (~20-30s). In resident mode it is loaded once and
      reused across jobs via ``load_vad``'s existing ``vad_model`` hook. The other
      stage models still load/unload per job (they are sequenced to fit limited
      VRAM); extending residency to the ASR and alignment models is future work and
      needs a bigger GPU (≥16 GB) to hold them simultaneously.
    """

    def __init__(self, *, resident: bool = False) -> None:
        self.resident = resident
        self._lock = threading.Lock()
        self._vad_model = None
        self._vad_key = None

    def _get_vad_model(self, cfg: PipelineConfig):
        key = (
            cfg.vad_method, cfg.device, cfg.device_index,
            cfg.vad_onset, cfg.vad_offset, cfg.chunk_size, cfg.hf_token,
            cfg.vad_pad_onset, cfg.vad_pad_offset, cfg.vad_min_duration_off,
        )
        if self._vad_model is None or self._vad_key != key:
            from cantocaptions_ai.pipeline.vad import load_vad
            logger.info("Loading resident VAD model (reused across jobs)...")
            processor = load_vad(
                vad_method=cfg.vad_method,
                device=cfg.device,
                device_index=cfg.device_index,
                vad_onset=cfg.vad_onset,
                vad_offset=cfg.vad_offset,
                vad_pad_onset=cfg.vad_pad_onset,
                vad_pad_offset=cfg.vad_pad_offset,
                vad_min_duration_off=cfg.vad_min_duration_off,
                chunk_size=cfg.chunk_size,
                use_auth_token=cfg.hf_token,
            )
            self._vad_model = processor.vad_model
            self._vad_key = key
        return self._vad_model

    def run(
        self,
        audio_path: str,
        cfg: PipelineConfig,
        *,
        progress: Optional[ProgressSink] = None,
        **kwargs,
    ) -> PipelineResult:
        """Run one job. Serialized against other :meth:`run` calls in this process."""
        with self._lock:
            vad_model = self._get_vad_model(cfg) if self.resident else None
            return run_pipeline(audio_path, cfg, progress=progress, _vad_model=vad_model, **kwargs)


__all__ = [
    "PipelineResult",
    "PipelineService",
    "run_pipeline",
    "ConfigError",
    "InputError",
]

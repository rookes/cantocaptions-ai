"""Which settings produced a debug checkpoint, so a replay never reuses a stale one.

``--debug_dir`` writes each stage's output under ``<debug_dir>/<name>/<stage>/`` and
``--load_debug_dir`` reads it back. Those checkpoints used to be keyed by file name alone:
change the ASR model, the VAD thresholds or the vocal isolation mode and a replay would
quietly load the old stage output as if it were current.

Each checkpoint now carries a ``meta.json`` recording the settings that shaped it -- its own
stage's, plus every upstream stage's, since a transcript is only as current as the VAD cuts
it was decoded from. A checkpoint whose recorded settings differ from this run's is treated
as missing (and recomputed), with the changed settings named in the log. A checkpoint with
no ``meta.json`` predates this and is accepted, with one warning per stage.

Only settings that change a stage's *output* belong here. Batch sizes, devices and VRAM
knobs are deliberately absent: a different batch size must not throw away an hour of ASR.
"""
import hashlib
import json
import os
from typing import Any, Callable, Dict, Optional, Tuple

from cantocaptions_ai.utils.log_utils import get_logger

logger = get_logger(__name__)

META_FILENAME = "meta.json"

# PipelineConfig fields that shape each stage's own output.
_STAGE_FIELDS: Dict[str, Tuple[str, ...]] = {
    "vad": (
        "vad_method", "vad_onset", "vad_offset", "vad_pad_onset", "vad_pad_offset",
        "vad_min_duration_off", "chunk_size", "audio_start", "audio_end", "audio_downmix",
        "audio_normalize", "audio_track",
    ),
    "vocal_isolation": (
        "vocal_isolation_method", "vocal_isolation_segment_mode", "vocal_isolation_compute_type",
        "vocal_isolation_span_gap",
    ),
    "transcription": ("language", "asr_compute_type"),
    "ensemble": ("ensemble_model",),
    "llm_correction": ("llm_model", "reference_correction_semantic"),
    "diarization": ("diarize_model", "diarize_scope", "min_speakers", "max_speakers"),
    "realign": (
        "realign_anchor", "realign_window", "realign_commit_margin", "realign_normalize",
        "align_compute_type", "align_char_substitution", "align_substitutions",
    ),
}


def _vad_derived(cfg) -> Dict[str, Any]:
    expand = bool(cfg.asr_context and cfg.asr_context_vad_expand and cfg.reference_subtitle)
    return {
        # --realign asks VAD to tile the whole file rather than keep only speech.
        "cover_all": bool(cfg.realign),
        "reference_expansion": (
            [cfg.reference_subtitle, cfg.reference_offset, cfg.asr_context_padding]
            if expand else None
        ),
    }


def _transcription_derived(cfg) -> Dict[str, Any]:
    # The model that actually ran: an unset `model` is the language pack's default, and
    # recording the resolved name keeps a checkpoint valid whichever way it was spelled.
    from cantocaptions_ai.pipeline.model_profiles import resolve_model_name
    derived: Dict[str, Any] = {"model": resolve_model_name(cfg.model, cfg.language)}
    if not cfg.asr_context:
        derived["context"] = None
        return derived
    derived["context"] = [
        cfg.reference_subtitle, cfg.reference_offset, cfg.asr_context_template,
        cfg.asr_context_scope, cfg.asr_context_neighbours, cfg.asr_context_max_chars,
    ]
    return derived


def _realign_derived(cfg) -> Dict[str, Any]:
    from cantocaptions_ai.languages import get_language_pack
    return {"align_model": cfg.align_model or get_language_pack(cfg.language).default_align_model}


def _llm_derived(cfg) -> Dict[str, Any]:
    return {"reference": [cfg.reference_subtitle, cfg.reference_offset]
            if cfg.reference_subtitle else None}


# Settings that are not plain fields, or only matter when a feature is on (a context
# template must not invalidate a transcript that was decoded without any context).
_STAGE_DERIVED: Dict[str, Callable[[Any], Dict[str, Any]]] = {
    "vad": _vad_derived,
    "transcription": _transcription_derived,
    "llm_correction": _llm_derived,
    "realign": _realign_derived,
}

# The stages whose output each stage consumed, in pipeline order.
_UPSTREAM: Dict[str, Tuple[str, ...]] = {
    "vad": (),
    "vocal_isolation": ("vad",),
    "transcription": ("vad", "vocal_isolation"),
    "ensemble": ("vad", "vocal_isolation"),
    "llm_correction": ("vad", "vocal_isolation", "transcription", "ensemble"),
    "diarization": ("vad", "vocal_isolation"),
    "realign": ("vad", "vocal_isolation"),
}

STAGES = tuple(_STAGE_FIELDS)


def _own_settings(stage: str, cfg) -> Dict[str, Any]:
    settings = {field: getattr(cfg, field) for field in _STAGE_FIELDS[stage]}
    derive = _STAGE_DERIVED.get(stage)
    if derive is not None:
        settings.update(derive(cfg))
    return settings


def checkpoint_settings(cfg) -> Dict[str, Dict[str, Any]]:
    """Per stage, every setting its checkpoint depends on: its own and its upstreams'.

    Keys are ``"<stage>.<setting>"`` so a diff reads as ``vad.vad_onset: 0.15 -> 0.3``.
    """
    own = {stage: _own_settings(stage, cfg) for stage in STAGES}
    out: Dict[str, Dict[str, Any]] = {}
    for stage in STAGES:
        merged: Dict[str, Any] = {}
        for source in (*_UPSTREAM[stage], stage):
            merged.update({f"{source}.{key}": value for key, value in own[source].items()})
        out[stage] = merged
    return out


def fingerprint(settings: Dict[str, Any]) -> str:
    blob = json.dumps(settings, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _meta_path(debug_dir: str, name: str, stage: str) -> str:
    # Must match debug._stage_dir, which lays checkpoints out the same way.
    return os.path.join(debug_dir, name.strip(), stage, META_FILENAME)


def write_checkpoint_meta(debug_dir: str, name: str, stage: str, settings: Dict[str, Any]) -> None:
    """Record the settings behind a checkpoint that was just written."""
    path = _meta_path(debug_dir, name, stage)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": fingerprint(settings), "settings": settings}, f,
                  ensure_ascii=False, indent=2, sort_keys=True, default=str)


# (debug_dir, stage) pairs already warned about for lacking a meta.json, and stale
# checkpoints already reported -- the orchestrator and the stage both check each one.
_warned_unversioned: set = set()
_warned_stale: set = set()


def checkpoint_is_current(
    debug_dir: str, name: str, stage: str, settings: Optional[Dict[str, Any]],
) -> bool:
    """False if the checkpoint was produced under different settings than *settings*.

    ``settings=None`` means the caller is not versioning checkpoints (a direct stage call),
    so any checkpoint counts. A checkpoint without a ``meta.json`` is accepted once-warned.
    Says nothing about whether the checkpoint's data files exist; callers check that.
    """
    if settings is None:
        return True
    path = _meta_path(debug_dir, name, stage)
    try:
        with open(path, encoding="utf-8") as f:
            recorded = json.load(f)
    except FileNotFoundError:
        key = (os.path.abspath(debug_dir), stage)
        if key not in _warned_unversioned:
            _warned_unversioned.add(key)
            logger.warning(
                "Debug checkpoints for stage '%s' in %s record no settings (written by an "
                "older version); reusing them as-is. Delete them if the settings changed.",
                stage, debug_dir,
            )
        return True
    except (OSError, ValueError):
        logger.warning("Unreadable checkpoint metadata %s; recomputing %s", path, stage)
        return False

    current = fingerprint(settings)
    if recorded.get("fingerprint") == current:
        return True
    stale_key = (os.path.abspath(path), current)
    if stale_key in _warned_stale:
        return False
    _warned_stale.add(stale_key)
    before = recorded.get("settings") or {}
    missing = "<not recorded>"
    changed = sorted(k for k in set(before) | set(settings)
                     if before.get(k, missing) != settings.get(k, missing))
    detail = ", ".join(f"{k}: {before.get(k, missing)!r} -> {settings.get(k, missing)!r}"
                       for k in changed[:6])
    if len(changed) > 6:
        detail += f", and {len(changed) - 6} more"
    logger.warning(
        "Debug checkpoint %s/%s was made with different settings (%s); recomputing it.",
        name, stage, detail or "unknown difference",
    )
    return False

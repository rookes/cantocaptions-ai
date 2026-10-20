"""The embedded-reference step of a pipeline run: before VAD, and as all of --references_only.

Each input's subtitle tracks are written beside its outputs (extract.extract_references). With
``reference_subtitle = embedded`` the best Chinese track then becomes the run's reference
subtitle, exactly as if its path had been given. It runs before VAD because VAD expansion and
ASR context read the reference, and it runs to the end before any speech model loads because
the OCR worker holds the GPU while it works.

Under ``--debug_dir`` the written files are checkpointed in ``{debug_dir}/{name}/references/``;
``--load_debug_dir`` copies them back instead of extracting again while the settings that
shaped them are unchanged.
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Dict, List

from cantocaptions_ai.utils.log_utils import get_logger

logger = get_logger(__name__)

EMBEDDED = "embedded"
CHECKPOINT = "references"


def wanted(cfg) -> bool:
    return bool(cfg.extract_references or cfg.reference_subtitle == EMBEDDED)


def _settings(cfg, media: str) -> dict:
    try:
        st = os.stat(media)
        stamp = [st.st_size, int(st.st_mtime)]
    except OSError:
        stamp = None
    return {"references.media": [os.path.abspath(media), stamp],
            "references.reference_tracks": cfg.reference_tracks,
            "references.reference_ocr": cfg.reference_ocr,
            "references.reference_ocr_mode": cfg.reference_ocr_mode,
            "references.reference_ocr_lang": cfg.reference_ocr_lang}


def _checkpoint_dir(debug_dir: str, name: str) -> str:
    return os.path.join(debug_dir, name.strip(), CHECKPOINT)


def _save_checkpoint(cfg, name: str, media: str, result) -> None:
    from cantocaptions_ai.utils.checkpoints import write_checkpoint_meta
    d = _checkpoint_dir(cfg.debug_dir, name)
    os.makedirs(d, exist_ok=True)
    for t in result.written:
        shutil.copy2(t.path, os.path.join(d, os.path.basename(t.path)))
    shutil.copy2(result.manifest, os.path.join(d, os.path.basename(result.manifest)))
    write_checkpoint_meta(cfg.debug_dir, name, CHECKPOINT, _settings(cfg, media))


def _load_checkpoint(cfg, name: str, media: str):
    """The checkpointed ReferenceSet copied back into the output directory, or None."""
    from cantocaptions_ai.references.extract import MANIFEST_SUFFIX, load_manifest, manifest_path
    from cantocaptions_ai.utils.checkpoints import checkpoint_is_current
    d = _checkpoint_dir(cfg.load_debug_dir, name)
    saved = os.path.join(d, os.path.basename(name) + MANIFEST_SUFFIX)
    if not os.path.isfile(saved) or not checkpoint_is_current(
            cfg.load_debug_dir, name, CHECKPOINT, _settings(cfg, media)):
        return None
    result = load_manifest(saved)
    out_base = os.path.dirname(manifest_path(cfg.output_dir, name))
    os.makedirs(out_base or cfg.output_dir, exist_ok=True)
    for t in result.tracks:
        if t.path:
            src = os.path.join(d, os.path.basename(t.path))
            if not os.path.isfile(src):
                return None
            t.path = os.path.join(out_base, os.path.basename(t.path))
            shutil.copy2(src, t.path)
    from cantocaptions_ai.references.extract import best_chinese
    result.best_chinese = best_chinese(result.tracks)
    result.manifest = manifest_path(cfg.output_dir, name)
    with open(result.manifest, "w", encoding="utf-8") as f:
        json.dump(result.as_dict(), f, ensure_ascii=False, indent=2)
    return result


def extract_for_inputs(cfg, media_paths: List[str], names: Dict[str, str], summary,
                       progress=None) -> Dict[str, object]:
    """Extract every input's embedded subtitles under one console stage. Returns
    {media path: ReferenceSet}; an input whose extraction failed is logged and left out."""
    from cantocaptions_ai.references.extract import describe, extract_references
    from cantocaptions_ai.utils.log_utils import StageTimer
    results: Dict[str, object] = {}
    with StageTimer("References", summary, progress, track_vram=False,
                    title="Embedded reference subtitles") as timer:
        for media in media_paths:
            name = names[media]
            result = _load_checkpoint(cfg, name, media) if cfg.load_debug_dir else None
            if result is not None:
                logger.info("%s: reference subtitles reloaded from %s", name,
                            os.path.normpath(_checkpoint_dir(cfg.load_debug_dir, name)))
            else:
                timer.status(f"reading subtitle tracks of {os.path.basename(media)}")
                try:
                    result = extract_references(
                        media, cfg.output_dir, name, tracks=cfg.reference_tracks,
                        ocr=cfg.reference_ocr, device=cfg.reference_ocr_device,
                        mode=cfg.reference_ocr_mode, ocr_lang=cfg.reference_ocr_lang,
                        cache_only=cfg.model_cache_only, status=timer.status,
                        reporter=timer.reporter)
                except (RuntimeError, ValueError, OSError) as e:
                    logger.error("%s: could not extract its subtitle tracks: %s", name, e)
                    continue
                if cfg.debug_dir:
                    _save_checkpoint(cfg, name, media, result)
            lines = describe(result)
            if lines:
                logger.info("%s: %d reference subtitle(s) written\n  %s", name,
                            len(result.written), "\n  ".join(lines))
            else:
                logger.info("%s: no subtitle tracks to extract", name)
            _warn_skipped_ocr(result)
            results[media] = result
            timer.status("")
    return results


def _warn_skipped_ocr(result) -> None:
    from cantocaptions_ai.errors import CantocaptionsWarning
    import warnings
    missing = [s for s in result.skipped if "not installed" in s["reason"]]
    if missing:
        warnings.warn(
            f"{len(missing)} image subtitle track(s) skipped: OCR is not installed "
            "(uv sync --extra ocr, or pip install 'cantocaptions_ai[ocr]')", CantocaptionsWarning)

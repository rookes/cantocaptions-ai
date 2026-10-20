"""OCR engines for bitmap subtitles. Each reads EventImages (raster.py) and returns one text
per event, lines joined top to bottom with ``\\n``.

``OCR_BACKENDS`` maps a name to its class, as ``ASR_BACKENDS`` does for speech models. PaddleOCR
is the only one so far. Every backend is constructed and run inside the OCR worker process
(worker.py), never in the pipeline's own process.

PaddleOCR has two ways in:

- ``line`` (the default): the recogniser alone, on each text line raster.py cut out. A subtitle
  line is already one horizontal line of text on a clean background, which is the recogniser's
  input, so the detector adds nothing and its mistakes go away: it boxes a lone 、 or the dots of
  an ellipsis as text of their own (read as "0" or "000"), and splits one line in two at a wide
  gap.
- ``detect``: the full pipeline (text detection, then recognition per detected box) on the
  event's whole picture, boxes on one row joined left to right. Kept for pictures that are not
  plain lines of text (a sign drawn at an angle).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Type

from cantocaptions_ai.references.raster import EventImages

DEFAULT_BACKEND = "paddle"
OCR_MODES = ("line", "detect")
# One recogniser reads Simplified and Traditional Chinese, English and Japanese. PP-OCRv6's
# reads Chinese subtitles more accurately than PP-OCRv5's server model, on the words that change
# the meaning (幹 read as 乾 or 干, 奸 as 姦, 兇 as 兒), and faster; it drops an enumeration
# comma (、) more often and writes a few characters in their Simplified form (盜 as 盗).
# Measurements: docs/references.md.
PADDLE_REC_MODEL = "PP-OCRv6_medium_rec"
PADDLE_BATCH = 32
# A piece wider than this many times its height that reads as Latin text is read again in
# halves, cut at a word space near its middle. On a long, low line of English the recogniser
# drops and merges letters ("Surely come in the morngone day sy ello") and reads each half of
# the same picture exactly. Chinese is never cut: read in halves it loses the context that
# picks a character (管制 became 管製, 判斷 判断), and a long Chinese line reads fine whole.
LATIN_MAX_ASPECT = 12.0


class OcrBackend:
    name = ""

    def __init__(self, lang: str, device: str = "auto", mode: str = "line",
                 cache_only: bool = False) -> None:
        self.lang, self.device, self.mode, self.cache_only = lang, device, mode, cache_only

    def read(self, items: Sequence[EventImages]) -> List[str]:
        raise NotImplementedError


def _paddle_device(device: str) -> str:
    if device in ("cpu", "gpu"):
        return device
    if device == "cuda":
        return "gpu"
    import paddle
    return "gpu" if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() else "cpu"


class PaddleBackend(OcrBackend):
    name = "paddle"

    def __init__(self, lang: str, device: str = "auto", mode: str = "line",
                 cache_only: bool = False) -> None:
        super().__init__(lang, device, mode, cache_only)
        if mode not in OCR_MODES:
            raise ValueError(f"unknown OCR mode {mode!r}; one of {OCR_MODES}")
        # Without this PaddleX probes its model hosters on every start, which is a network
        # round trip per run and fails offline, though the models are already on disk.
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        if cache_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
        self._device = _paddle_device(device)
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        if self.mode == "line":
            from paddleocr import TextRecognition
            # The recogniser is language-specific only beyond the CJK/Latin set this model
            # covers; other languages get PaddleOCR's own choice for them.
            if self.lang in ("ch", "chinese_cht", "en", "japan"):
                self._model = TextRecognition(model_name=PADDLE_REC_MODEL, device=self._device)
            else:
                from paddleocr import PaddleOCR
                self._model = PaddleOCR(lang=self.lang, device=self._device, **_PIPELINE_OFF)
                self.mode = "detect"
        else:
            from paddleocr import PaddleOCR
            self._model = PaddleOCR(lang=self.lang, device=self._device, **_PIPELINE_OFF)
        return self._model

    def read(self, items: Sequence[EventImages]) -> List[str]:
        import numpy as np
        model = self._load()
        if self.mode == "detect":
            return [normalize_text(_join_boxes(model.predict(np.asarray(it.block.convert("RGB")))))
                    for it in items]
        pieces = [image for it in items for line in it.lines for _, image in line]
        texts: List[str] = []
        for i in range(0, len(pieces), PADDLE_BATCH):
            batch = [np.asarray(im.convert("RGB")) for im in pieces[i:i + PADDLE_BATCH]]
            texts += [str(r["rec_text"]).strip() for r in model.predict(batch, batch_size=len(batch))]
        texts = [self._reread_latin(model, im, t) for im, t in zip(pieces, texts)]
        read = iter(texts)
        out = []
        for it in items:
            lines = ["".join(joiner + next(read) for joiner, _ in line).strip() for line in it.lines]
            out.append(normalize_text("\n".join(t for t in lines if t)))
        return out


    def _reread_latin(self, model, image, text: str, depth: int = 0) -> str:
        import numpy as np
        if depth >= 3 or image.width <= LATIN_MAX_ASPECT * image.height or not _mostly_latin(text):
            return text
        cut = _word_gap(np.asarray(image) < 200)
        if cut is None:
            return text
        halves = [image.crop((0, 0, cut, image.height)), image.crop((cut, 0, image.width, image.height))]
        parts = []
        for half in halves:
            read = str(model.predict(np.asarray(half.convert("RGB")))[0]["rec_text"]).strip()
            parts.append(self._reread_latin(model, half, read, depth + 1))
        return " ".join(p for p in parts if p)


def _mostly_latin(text: str) -> bool:
    letters = [ch for ch in text if not ch.isspace()]
    return bool(letters) and sum(ch.isascii() and ch.isalpha() for ch in letters) >= 0.6 * len(letters)


def _word_gap(ink) -> "Optional[int]":
    """The column to cut a line of words at: the middle of the word space nearest its centre.
    Word spaces are the blank column runs at least half as wide as the widest one."""
    import numpy as np
    cols = ink.any(axis=0)
    filled = np.flatnonzero(cols)
    if filled.size == 0:
        return None
    gaps, x, end = [], int(filled[0]), int(filled[-1])
    while x < end:
        if not cols[x]:
            start = x
            while x < end and not cols[x]:
                x += 1
            gaps.append((start, x))
        else:
            x += 1
    if not gaps:
        return None
    widest = max(b - a for a, b in gaps)
    words = [(a, b) for a, b in gaps if b - a >= 0.5 * widest]
    mid = ink.shape[1] / 2
    a, b = min(words, key=lambda g: abs((g[0] + g[1]) / 2 - mid))
    return (a + b) // 2


# Subtitles are upright, flat, horizontal lines: page orientation, unwarping and per-line
# orientation only cost time. oneDNN inference on the CPU crashes on these graphs in paddle
# 3.3.1 ("ConvertPirAttribute2RuntimeAttribute not support"); it is unused on the GPU.
_PIPELINE_OFF = dict(enable_mkldnn=False, use_doc_orientation_classify=False,
                     use_doc_unwarping=False, use_textline_orientation=False)


_FULLWIDTH = {",": "，", "?": "？", "!": "！", ":": "：", ";": "；"}


def _is_cjk(ch: str) -> bool:
    return bool(ch) and ("\u3000" <= ch <= "\u9fff" or "\uf900" <= ch <= "\ufaff"
                         or "\U00020000" <= ch <= "\U0003134f")


def normalize_text(text: str) -> str:
    """Halfwidth , ? ! : ; written against Chinese text become their fullwidth forms, the way
    a Chinese subtitle writes them, and the space the recogniser may leave after one goes; the
    same marks inside English or a time (02:20) stay as they are."""
    out = list(text)
    for i, ch in enumerate(out):
        if ch in _FULLWIDTH:
            before = out[i - 1] if i else ""
            after = out[i + 1] if i + 1 < len(out) else ""
            if _is_cjk(before) or _is_cjk(after):
                out[i] = _FULLWIDTH[ch]
    text = "".join(out)
    for mark in _FULLWIDTH.values():
        text = text.replace(mark + " ", mark).replace(" " + mark, mark)
    return text


def _join_boxes(results) -> str:
    """One picture's detected boxes as lines: boxes whose vertical spans overlap by half the
    shorter one are one row, read left to right; rows top to bottom."""
    boxes = []
    for result in results:
        data = result.get("res", result) if hasattr(result, "get") else result
        polys = data.get("rec_polys")
        if polys is None:
            polys = data.get("dt_polys")
        for i, text in enumerate(data.get("rec_texts") or []):
            if polys is None or i >= len(polys):
                continue
            ys = [float(p[1]) for p in polys[i]]
            xs = [float(p[0]) for p in polys[i]]
            boxes.append((min(ys), max(ys), min(xs), str(text).strip()))
    rows: List[List[tuple]] = []
    for box in sorted(boxes):
        for row in rows:
            top, bottom = min(b[0] for b in row), max(b[1] for b in row)
            overlap = min(bottom, box[1]) - max(top, box[0])
            if overlap >= 0.5 * min(bottom - top, box[1] - box[0]):
                row.append(box)
                break
        else:
            rows.append([box])
    return "\n".join("".join(b[3] for b in sorted(row, key=lambda b: b[2])) for row in rows
                     if any(b[3] for b in row))


OCR_BACKENDS: Dict[str, Type[OcrBackend]] = {"paddle": PaddleBackend}


def ocr_available(backend: str = DEFAULT_BACKEND) -> Optional[str]:
    """None if *backend* can run here, else what is missing. Looks for the package without
    importing it, so the pipeline's process never loads an OCR library."""
    import importlib.util
    if backend not in OCR_BACKENDS:
        return f"unknown OCR backend {backend!r}"
    if backend == "paddle":
        missing = [m for m in ("paddleocr", "paddle") if importlib.util.find_spec(m) is None]
        if missing:
            return ("PaddleOCR is not installed (uv sync --extra ocr, or pip install "
                    "'cantocaptions_ai[ocr]')")
    return None

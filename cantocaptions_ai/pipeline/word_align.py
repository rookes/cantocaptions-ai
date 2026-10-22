"""Word-level forced alignment: models that take audio and a list of words and return each
word's start and end.

The align backends (``align_backends.py``) are CTC models: they give per-frame emissions over a
character vocabulary, and ``alignment.py`` finds the path through them itself. That makes
alignment only as multilingual as the vocabulary: the Cantonese model has no tokens for é or ò,
and no ear for Italian, so the times it gives an Italian word are guesses. A word aligner is
the other kind: it is told the words and answers with times, in any language it was trained
on.

Today the pipeline uses one where the run's own align model cannot follow the speech: the
stretches ASR repair kept in another language (``foreign_spans``; see the alignment stage).
The interface is not specific to that. A job is any audio with any words in a language the
aligner supports, so a language whose pack has no CTC model can align with one outright.

Families are looked up by the checkpoint's architecture (``word_aligner_family``), so another
model is one subclass and one ``WORD_ALIGNERS`` entry:

* ``qwen3-forced-aligner`` -- Qwen3-ForcedAligner (transformers' ``Qwen3ASRForTokenClassification``).
  Chinese, Cantonese, English, French, German, Italian, Japanese, Korean, Portuguese, Russian
  and Spanish; up to 5 minutes of audio per job; 80 ms resolution. About 2 GB of VRAM in
  float16, and about 0.15 s per 25 s chunk on an RTX 3080.
"""
import importlib
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
import torch

from cantocaptions_ai.text_profiles import LATIN_WORD
from cantocaptions_ai.utils.audio import SAMPLE_RATE
from cantocaptions_ai.utils.log_utils import get_logger
from cantocaptions_ai.utils.model_utils import (
    BatchExecutor,
    ensure_hf_model_downloaded,
    guard_model_load,
    resolve_torch_compute_dtype,
)

logger = get_logger(__name__)

DEFAULT_WORD_ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B-hf"

WordTimes = List[Optional[Tuple[float, float]]]


@dataclass(frozen=True)
class WordAlignJob:
    """Audio (16 kHz mono) and the words said in it, in order. Times come back relative to the
    start of ``audio``. ``language`` is a Qwen3-ASR language name ("Italian")."""

    audio: np.ndarray
    words: Sequence[str]
    language: Optional[str] = None


def spoken_tokens(text: str) -> List[Tuple[int, int]]:
    """The (start, end) offsets of the words in *text* a word aligner is given: each run of
    Latin letters and digits (``LATIN_WORD``) as one, and every other letter or digit on its
    own -- one Chinese character, one kana -- which is also how ``alignment.py`` splits text
    with no spaces into words. Spaces and punctuation are not words."""
    out: List[Tuple[int, int]] = []
    pos = 0
    for match in LATIN_WORD.finditer(text):
        out.extend(_single_letters(text, pos, match.start()))
        out.append((match.start(), match.end()))
        pos = match.end()
    out.extend(_single_letters(text, pos, len(text)))
    return out


def _single_letters(text: str, start: int, end: int) -> List[Tuple[int, int]]:
    return [(i, i + 1) for i in range(start, end)
            if unicodedata.category(text[i])[0] in "LN"]


class WordAligner(ABC):
    """Audio plus words in, word times out. See the module docstring."""

    #: Language names (as Qwen3-ASR writes them) the model was trained on.
    languages: FrozenSet[str] = frozenset()
    #: The longest audio one job may hold, in seconds.
    max_seconds: float = float("inf")

    def supports(self, language: Optional[str]) -> bool:
        return bool(language) and language.strip().lower() in {l.lower() for l in self.languages}

    @abstractmethod
    def align(self, jobs: Sequence[WordAlignJob], progress_callback=None) -> List[WordTimes]:
        """Per job, one ``(start, end)`` per word, or None for a word that could not be timed.
        A job the aligner cannot take (language, length) comes back all None."""


class Qwen3ForcedAligner(WordAligner):
    """Qwen3-ForcedAligner through transformers' native qwen3_asr support.

    The model reads the audio and the words, each followed by a timestamp slot, and classifies
    every slot into an 80 ms step. The processor builds the prompt from a transcript by its
    own word split, which keeps no punctuation and so fuses ``ragazza。Mi`` into one word; the
    words here are the caller's, so its times map back one to one.
    """

    max_seconds = 300.0

    def __init__(self, model, processor, batch_size: Optional[int] = 8):
        from transformers.models.qwen3_asr.processing_qwen3_asr import FORCED_ALIGNER_LANGUAGES

        self.model = model
        self.processor = processor
        self.batch_size = batch_size
        self.languages = frozenset(FORCED_ALIGNER_LANGUAGES)
        self.timestamp_token_id = model.config.timestamp_token_id

    def align(self, jobs: Sequence[WordAlignJob], progress_callback=None) -> List[WordTimes]:
        out: List[WordTimes] = [[None] * len(job.words) for job in jobs]
        todo = []
        for i, job in enumerate(jobs):
            words = [_kept(w) for w in job.words]
            if not any(words):
                continue
            if not self.supports(job.language) or len(job.audio) / SAMPLE_RATE > self.max_seconds:
                logger.debug("Word aligner: skipping a %.1f s %s job", len(job.audio) / SAMPLE_RATE,
                             job.language)
                continue
            todo.append((i, words))

        def infer_fn(batch):
            # Words with nothing the model can read (a lone mark) are left out of the prompt.
            lists = [[w for w in words if w] for _, words in batch]
            conversations = [
                [{"role": "user", "content": [{"type": "audio", "audio": jobs[i].audio},
                                              *({"type": "text", "text": w} for w in kept)]}]
                for (i, _), kept in zip(batch, lists)
            ]
            inputs = self.processor.apply_chat_template(
                conversations, tokenize=True, return_dict=True, padding=True)
            inputs = inputs.to(self.model.device, self.model.dtype)
            with torch.inference_mode():
                logits = self.model(**inputs).logits
            stamps = self.processor.decode_forced_alignment(
                logits, inputs["input_ids"], lists, self.timestamp_token_id)
            for (i, words), times in zip(batch, stamps):
                timed = iter(times)
                row: WordTimes = []
                for word in words:
                    if word:
                        stamp = next(timed)
                        row.append((stamp["start_time"], stamp["end_time"]))
                    else:
                        row.append(None)
                out[i] = unpile(row, len(jobs[i].audio) / SAMPLE_RATE)
            del inputs, logits

        BatchExecutor(self.batch_size, order_key=lambda job: len(jobs[job[0]].audio)).run(
            todo, infer_fn, reporter=progress_callback)
        return out


def unpile(times: WordTimes, duration: float, step: float = 0.08,
           per_word: float = 0.4) -> WordTimes:
    """Spread out words an aligner piled onto one instant.

    Qwen3-ForcedAligner answers in 80 ms steps and keeps times in order; a word it cannot
    find gets no time of its own but the same instant as its neighbours -- four words at
    once where the speech is sung, invented or too fast to follow. A cue of such words would
    be on screen for no time at all. Each run of two or more words sharing one start, with no
    length between them, is spread evenly from that instant towards the next timed word (or
    the end of the audio), but at no more than *per_word* seconds a word: a pile is mostly
    speech too fast for the aligner to resolve, and stretching it to the next word found has
    put "Chinese food? Sì, thank you" ten seconds late.
    """
    out = list(times)
    i = 0
    while i < len(out):
        if out[i] is None:
            i += 1
            continue
        j = i
        while (j + 1 < len(out) and out[j + 1] is not None
               and abs(out[j + 1][0] - out[i][0]) < step / 2 and out[j][1] - out[j][0] < step / 2):
            j += 1
        if j > i:
            lo = out[i][0]
            hi = next((t[0] for t in out[j + 1:] if t is not None), duration)
            width = min(max(hi - lo, 0.0) / (j - i + 1), per_word)
            for k in range(i, j + 1):
                out[k] = (round(lo + width * (k - i), 3), round(lo + width * (k - i + 1), 3))
        i = j + 1
    return out


def _kept(word: str) -> str:
    """The characters of *word* the aligner reads: letters, digits and apostrophes."""
    return "".join(c for c in word if c == "'" or unicodedata.category(c)[0] in "LN")


def load_qwen3_forced_aligner(model_id: str, device: str, device_index: int = 0,
                              compute_type: str = "float16", download_root: Optional[str] = None,
                              local_files_only: bool = False,
                              batch_size: Optional[int] = 8) -> Qwen3ForcedAligner:
    from transformers import AutoModelForTokenClassification, AutoProcessor

    try:
        ensure_hf_model_downloaded(model_id, cache_dir=download_root, local_files_only=local_files_only)
    except Exception as e:  # a cached copy may still load
        logger.warning("Could not download %r: %s -- using the cached copy if there is one.",
                       model_id, e)
    dtype = resolve_torch_compute_dtype(compute_type, device, "Word aligner")
    device_map = f"cuda:{device_index}" if device == "cuda" else device
    model = guard_model_load(
        "word aligner", "pass --align_foreign_model none to keep the align model's own timings",
        lambda: AutoModelForTokenClassification.from_pretrained(
            model_id, dtype=dtype, device_map=device_map, cache_dir=download_root,
            local_files_only=local_files_only).eval(),
    )
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=download_root,
                                              local_files_only=local_files_only)
    logger.info("Word aligner: %s", model_id)
    return Qwen3ForcedAligner(model, processor, batch_size=batch_size)


#: family -> "module:loader", imported only when that family is used.
WORD_ALIGNERS: Dict[str, str] = {
    "qwen3-forced-aligner": "cantocaptions_ai.pipeline.word_align:load_qwen3_forced_aligner",
}

_ARCHITECTURES = {"Qwen3ASRForTokenClassification": "qwen3-forced-aligner"}


def word_aligner_family(model_id: str, cache_dir: Optional[str] = None,
                        local_files_only: bool = False) -> str:
    """The family that runs *model_id*, from its checkpoint's architecture."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir,
                                        local_files_only=local_files_only)
    for arch in getattr(config, "architectures", None) or ():
        if arch in _ARCHITECTURES:
            return _ARCHITECTURES[arch]
    raise ValueError(f"{model_id!r} is not a word aligner this pipeline can run "
                     f"(known: {', '.join(sorted(_ARCHITECTURES))})")


def load_word_aligner(model_id: str = DEFAULT_WORD_ALIGNER, **kwargs) -> WordAligner:
    """Load the word aligner *model_id* with the loader its family names."""
    family = word_aligner_family(model_id, kwargs.get("download_root"),
                                 kwargs.get("local_files_only", False))
    module, _, func = WORD_ALIGNERS[family].partition(":")
    loader: Callable[..., WordAligner] = getattr(importlib.import_module(module), func)
    return loader(model_id, **kwargs)

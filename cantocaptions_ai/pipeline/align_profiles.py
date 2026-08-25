"""Per-alignment-model configuration.

The sibling of ``pipeline/model_profiles.py``, which does the same job for ASR models:
behaviour that depends on *how a given alignment model behaves* is pinned per model here
rather than hard-coded into the alignment stage. Every field defaults to a no-op, so an
align model with no entry runs exactly as it did before this module existed — and adding
one is a single ``ALIGN_PROFILES`` entry with no edits to ``alignment.py``.

Contrast ``pipeline/align_checks.py``, which is deliberately *not* per-model: it validates
alignment output for every model, including ones with no profile here.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

try:  # Protocol is stdlib on 3.8+; guarded only so type-checking imports stay optional.
    from typing import Protocol
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore[assignment,misc]


class AudioPrimer(Protocol):
    """Prepends left context to one VAD segment's audio before the encoder sees it.

    Implementations return ``prefix + audio`` and are **not** asked how much they
    prepended: ``alignment.py`` discards the prefix by measuring the encoder's own output
    length for the unprimed audio, so any prefix length works. That is what keeps this
    swappable — a future implementation is free to prepend a fixed shared buffer, room
    tone sampled once per file, or a canned clip, without the alignment stage changing.
    """

    def __call__(self, audio: np.ndarray, sample_rate: int) -> np.ndarray: ...


@dataclass(frozen=True)
class TailPrimer:
    """Prime with a copy of the segment's own tail, reversed by default.

    Why priming is needed at all: ``alvanlii/wav2vec2-BERT-cantonese`` was fine-tuned on
    clips that begin at speech onset and it reproduces that prior. Given audio whose first
    frames are *not* speech, it emits the utterance's first character at emission frame 0
    **and nowhere else** — on one measured segment the character scores ~0.999 at frame 0
    and ~1e-6 at its real onset 2.8 s later. Forced alignment then has no evidence to find,
    so the first character pins to the start of the VAD segment. This was invisible until
    ``vad_pad_onset`` began handing alignment segments that start before the speech does.

    Only real speech in front of the window satisfies the prior. Digital silence, low-level
    noise and masked left-padding were all measured and all leave the character at frame 0,
    so the primer must carry speech-like energy — it cannot be a silent buffer.

    The segment's own tail is used because it always exists, is the same voice and
    recording, and needs no state from outside the segment. It is reversed by default so
    the primer can never be mistaken for dialogue: ``alignment.py`` discards its frames
    before the trellis runs either way, and reversed and forward tails were measured to
    agree on 12 of 14 segments (the two exceptions differing by one frame and by 0.85 s on
    a 3.3 s segment).
    """

    seconds: float = 1.0
    reverse: bool = True

    def __call__(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        n = min(int(self.seconds * sample_rate), len(audio))
        if n <= 0:
            return audio
        prefix = audio[-n:][::-1] if self.reverse else audio[-n:]
        return np.concatenate([prefix, audio])


@dataclass(frozen=True)
class AlignProfile:
    """Per-model alignment behaviour. Every field defaults to a no-op."""

    primer: Optional[AudioPrimer] = None


DEFAULT_ALIGN_PROFILE = AlignProfile()

ALIGN_PROFILES: Dict[str, AlignProfile] = {
    "alvanlii/wav2vec2-BERT-cantonese": AlignProfile(primer=TailPrimer()),
}


def get_align_profile(model_name: Optional[str]) -> AlignProfile:
    """Resolve a model's profile, falling back to the all-no-op default."""
    return ALIGN_PROFILES.get(model_name or "", DEFAULT_ALIGN_PROFILE)

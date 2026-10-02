"""Per-alignment-model configuration.

The sibling of ``pipeline/model_profiles.py``, which does the same job for ASR models:
behaviour that depends on *how a given alignment model behaves* is pinned per model here
rather than hard-coded into the alignment stage. Every field defaults to a no-op, so an
align model with no entry runs exactly as it did before this module existed — and adding
one is a single ``ALIGN_PROFILES`` entry with no edits to ``alignment.py``.

Contrast ``pipeline/align_checks.py``, which is deliberately *not* per-model: it validates
alignment output for every model, including ones with no profile here. The division holds
even where the two meet — ``split_gap`` below says *whether and at what threshold* this
model's cues should be broken at an internal silence, while the code that finds the silence
and does the breaking stays model-agnostic over there, taking the number as an argument.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

# Bundled hand-picked substitution tables, one per align model; see AlignProfile.
SUBSTITUTIONS_DIR = Path(__file__).parent / "align_substitutions"

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

    Primed output is for forced alignment only; don't decode text from it. At the real onset
    the first character is often too weak to win the argmax (log-probability -1 to -3 where
    it wins at frame 0 unprimed), so a greedy decode drops it, and a one-word segment can
    decode to nothing. Sharing one primed pass between CTC transcription and alignment was
    tried and reverted for this reason: it lost 6 of 701 matched cues on the eval episodes.
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
    # Filename under SUBSTITUTIONS_DIR holding hand-picked character substitutions for this
    # model'"'"'s vocabulary. A substitution only means anything relative to one vocabulary --
    # 爹 -> 弟 is only useful because *this* model has 弟 and not 爹 -- so the table belongs
    # to the model, not to the pipeline. Loaded by align_vocab.bundled_substitutions and
    # merged *under* the caller'"'"'s --align_substitutions file; see pipeline/align_vocab.py.
    substitutions: Optional[str] = None
    # The least silence (seconds) between two adjacent characters of one cue at which this
    # model's alignment should be read as two separate utterances, so the cue is broken in
    # two. None -- the default -- never splits, which is the behaviour every caller had
    # before this field existed.
    #
    # It belongs to the align model because it is a statement about *that model's* emission:
    # how long a hole between two characters it will leave when it has nothing to place one
    # of them on. A model that dwells differently wants a different number, and one with no
    # measured number should not inherit another's. align_checks.SPLIT_INTERNAL_GAP is the
    # suggested starting point; --align_split_gap overrides whatever is set here.
    split_gap: Optional[float] = None
    # The shortest segment (in 16 kHz samples) the model's front end turns into a frame.
    # Shorter ones are skipped rather than crashing the extractor; see alignment.
    min_samples: int = 400
    # The --align_char_substitution level used when the config leaves it unset. Homophone
    # substitution reads Cantonese (Jyutping) pronunciations, so it is only right for a
    # Cantonese model's vocabulary; everything else defaults to no substitution.
    char_substitution: str = "off"
    # Whether the model's input is normalised to zero mean and unit variance before the
    # encoder sees it. None leaves the checkpoint's own processor to decide. Set False for
    # the Hugging Face copies of models torchaudio ships as pipeline bundles: torchaudio
    # feeds those the raw waveform, as they were trained, while the Hugging Face processor
    # normalises -- the one difference between the two (same weights, bit-identical
    # output given the same input).
    normalize_input: Optional[bool] = None


DEFAULT_ALIGN_PROFILE = AlignProfile()

# The Hugging Face releases of the torchaudio bundles that used to be these languages'
# defaults (languages/align_defaults.py), fed the same raw waveform the bundles were.
_RAW_INPUT = AlignProfile(normalize_input=False)

ALIGN_PROFILES: Dict[str, AlignProfile] = {
    "alvanlii/wav2vec2-BERT-cantonese": AlignProfile(
        primer=TailPrimer(),
        substitutions="wav2vec2-bert-cantonese.toml",
        char_substitution="homophone",
    ),
    "facebook/wav2vec2-base-960h": _RAW_INPUT,
    "facebook/wav2vec2-base-10k-voxpopuli-ft-fr": _RAW_INPUT,
    "facebook/wav2vec2-base-10k-voxpopuli-ft-de": _RAW_INPUT,
    "facebook/wav2vec2-base-10k-voxpopuli-ft-it": _RAW_INPUT,
}


def get_align_profile(model_name: Optional[str]) -> AlignProfile:
    """Resolve a model's profile, falling back to the all-no-op default."""
    return ALIGN_PROFILES.get(model_name or "", DEFAULT_ALIGN_PROFILE)

"""
Forced Alignment with Whisper
C. Max Bain
"""
import bisect
from dataclasses import dataclass
import math
from functools import lru_cache
import time
from typing import Callable, Dict, Iterable, Mapping, NamedTuple, Optional, Sequence, Union, List, Tuple

import numpy as np
import pandas as pd
import torch

from cantocaptions_ai.utils.audio import SAMPLE_RATE, load_audio, log_mel_spectrogram, resolve_device
from cantocaptions_ai.utils.output import PUNKT_LANGUAGES
from cantocaptions_ai.utils.schema import (
    AlignedTranscriptionResult,
    SingleSegment,
    SingleAlignedSegment,
    SingleWordSegment,
    SegmentData,
    ProgressCallback,
    VadAudioSegment,
    add_note,
    interpolate_nans,
)
from cantocaptions_ai.text_profiles import (
    DEFAULT_PUNCTUATION,
    PunctuationConfig,
    ScriptConfig,
    SpotCheck,
    script_for_language,
)
from cantocaptions_ai.pipeline.align_checks import (
    split_gapped_cues,
    warn_on_gapped_cues,
    warn_on_silent_starts,
    whole_file_region,
)
from cantocaptions_ai.pipeline.align_profiles import (
    DEFAULT_ALIGN_PROFILE,
    AudioPrimer,
    get_align_profile,
)
from cantocaptions_ai.pipeline.align_vocab import (
    VocabRepair,
    bundled_substitutions,
    filter_spotchecks,
    merge_substitutions,
    substitution_notes,
)

# How each model family is loaded and run lives in align_backends; the helpers are
# re-exported here, where callers (and the tests) have always found them.
from cantocaptions_ai.pipeline.align_backends import (  # noqa: F401
    ALIGN_BACKENDS,
    _compute_vad_emissions_batched,
    _input_key,
    _passes_attention_mask,
    _warn_alignment_vram,
    align_backend_for,
    get_align_backend,
)
from cantocaptions_ai.utils.log_utils import get_logger
from cantocaptions_ai.utils.model_utils import resolve_torch_compute_dtype

logger = get_logger(__name__)

# One 25 ms fbank frame at 16 kHz: the shortest audio the aligner's feature extractor turns
# into anything. Below 400 samples it yields no frames, and below 240 it raises. The default
# for AlignProfile.min_samples; wav2vec2's conv front end happens to need the same 400.
MIN_ALIGN_SAMPLES = 400

# The built-in align models by language live in languages/align_defaults.py (a torch-free
# module the language registry reads); re-exported here, where callers have always found them.
from cantocaptions_ai.languages.align_defaults import (  # noqa: E402
    DEFAULT_ALIGN_MODELS_HF,
    DEFAULT_ALIGN_MODELS_TORCH,
)

# https://huggingface.co/scottykwok/wav2vec2-large-xlsr-cantonese xlsr-53 + common voice
# scottykwok/wav2vec2-large-xlsr-cantonese xlsr-53 + common voice + more training
# wcfr/wav2vec2-conformer-rel-pos-base-cantonese
# alvanlii/wav2vec2-BERT-cantonese


# --- Dataclasses ---

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


# --- Low-level CTC alignment ---
# source: https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html

def get_trellis(emission, tokens, blank_id=0, free_end: bool = False):
    """Forced-alignment trellis over *emission* for *tokens*.

    ``free_end`` must match the value later passed to ``backtrack``. It drops the
    ``trellis[-num_tokens:, 0] = +inf`` sentinel, which exists purely to terminate the
    backward walk of a *forced* alignment. That +inf propagates diagonally through the
    recurrence, so by the final row it has flooded every token column but the last -- which
    would leave a free-end search with only one column to "choose", silently turning it back
    into a forced walk. A partial walk does not need the sentinel: it enters at a column the
    forward pass actually reached, so it has the frames to walk back to the start.
    """
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra dimensions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, blank_id], 0)
    trellis[0, -num_tokens:] = -float("inf")
    if not free_end:
        trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


def backtrack(trellis, emission, tokens, blank_id=0, free_end: bool = False):
    """Walk the trellis back to token 0, returning the best monotonic path.

    Without ``free_end`` (the default) the path must consume *every* token: it enters at the
    last token column and finds the frame that best completes it. That is right when the
    text is known to correspond to exactly this audio.

    With ``free_end=True`` the path instead enters at the *last frame* and takes whichever
    token column scores best there, so it consumes only as many tokens as the audio
    actually explains. --realign uses this to slide a window over a long recording and ask
    "how much of the remaining transcript fits in here?" without knowing the answer up
    front; see pipeline/realign.py. The trellis must have been built with the same
    ``free_end``, and is checked for it rather than being allowed to answer wrongly.
    """
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    if free_end:
        t_start = trellis.size(0) - 1
        row = trellis[t_start, 1:]
        if bool(torch.isposinf(row).any()):
            raise ValueError(
                "backtrack(free_end=True) needs a trellis built with get_trellis("
                "free_end=True); the forced-walk sentinel makes every column but the last "
                "infinite, which would silently reduce the search to a forced alignment."
            )
        # Column 0 is the start state -- entering there means nothing was consumed.
        j = 1 + torch.argmax(row).item()
    else:
        j = trellis.size(1) - 1
        t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[T-1]` is the emission at time frame `T` of trellis dimension.
        # Score for token staying the same from time frame T-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from J-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else blank_id].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        # failed
        return None

    return path[::-1]


def get_score(emission, tokens, blank_id=0):
    """Return average score for a token sequence against emissions."""
    trellis = get_trellis(emission, tokens, blank_id)
    path = backtrack(trellis, emission, tokens, blank_id)
    if path is None:
        return float("-inf")
    return sum(p.score for p in path) / len(path)


def merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


class AlignedCandidate(NamedTuple):
    """The winner of :func:`align_best_of`: which candidate text it was, its
    per-character segments (already run through `merge_repeats`), and its mean
    path score."""

    text: str
    segments: List["Segment"]
    mean_score: float


def align_best_of(emission, dictionary, blank_id, candidates: Sequence[str]) -> Optional[AlignedCandidate]:
    """Score each of `candidates` against the SAME `emission` and keep the best.

    Generic over any list of candidate strings -- not particle- or numeral-
    specific. Exists because the particle `SpotCheck` rescoring in
    `_align_segment` below cannot be reused for this: it swaps a single character
    for another single character at a fixed frame, which only works because every
    candidate has the same length and so leaves the rest of the path's frame
    assignments valid. A numeral reading changes the token count itself (`一千八
    百九十一` vs `一八九一`), so each candidate needs its own `get_trellis`/
    `backtrack` walk; only the (expensive) `emission` is shared across them.

    Candidates are tokenized against `dictionary`, dropping characters it has no
    token for (same policy as the rest of this module) -- a candidate that ends
    up with zero usable characters is skipped rather than raising. Returns `None`
    if every candidate fails to produce a usable token sequence or to align.
    """
    best: Optional[tuple[str, list, float]] = None
    for text in candidates:
        chars = [c for c in text if c in dictionary]
        if not chars:
            continue
        tokens = [dictionary[c] for c in chars]
        trellis = get_trellis(emission, tokens, blank_id)
        path = backtrack(trellis, emission, tokens, blank_id)
        if path is None:
            continue
        mean_score = sum(p.score for p in path) / len(path)
        if best is None or mean_score > best[2]:
            best = (text, chars, mean_score)
            best_path = path
    if best is None:
        return None
    text, chars, mean_score = best
    return AlignedCandidate(text, merge_repeats(best_path, chars), mean_score)


def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words


# --- Private helpers ---

def _run_model_inference(
    model: torch.nn.Module,
    model_type: str,
    audio: torch.Tensor,
    processor,
    device: str,
    lengths=None,
) -> torch.Tensor:
    """Single forward pass returning log-softmax emissions, by the model's backend."""
    return get_align_backend(model_type).forward(model, processor, audio, device, lengths=lengths)


def _get_blank_id(model_dictionary: dict) -> int:
    return next((code for char, code in model_dictionary.items() if char in ('[pad]', '<pad>')), 0)


def _spaced(model_lang: str, script: Optional[ScriptConfig]) -> bool:
    """Whether words are space-separated: the script's say, else the language's default."""
    return (script or script_for_language(model_lang)).spaced


def _get_sentence_spans(
    text: str, model_lang: str, punctuation: PunctuationConfig,
    script: Optional[ScriptConfig] = None,
) -> List[Tuple[int, int]]:
    """Split text into sentence span tuples using language-appropriate tokenization.

    A script without spaces splits at its own punctuation; a space-separated one uses the
    NLTK Punkt model for the language, which knows abbreviations are not sentence ends.
    """
    if not _spaced(model_lang, script):
        return punctuation.sentence_spans(text)
    splitter = _punkt_splitter(PUNKT_LANGUAGES.get(model_lang, 'english'))
    if splitter is None:
        return punctuation.sentence_spans(text)
    return list(splitter.span_tokenize(text))


@lru_cache(maxsize=None)
def _punkt_splitter(punkt_lang: str):
    """NLTK's Punkt sentence splitter for a language, downloading it on first use.

    None when it is neither installed nor downloadable (an offline machine): sentences are
    then split at the language's own punctuation instead, which only loses Punkt's
    knowledge that "Dr." does not end a sentence.
    """
    import nltk

    for attempt in range(2):
        try:
            return nltk.tokenize.PunktTokenizer(punkt_lang)
        except LookupError:
            if attempt == 0 and not nltk.download("punkt_tab", quiet=True):
                break
    logger.warning(
        "NLTK Punkt data for %r is unavailable (offline?); splitting sentences at punctuation",
        punkt_lang,
    )
    return None


def _preprocess_segment(
    text: str, model_lang: str, model_dictionary: dict,
    punctuation: PunctuationConfig = DEFAULT_PUNCTUATION,
    spans: Optional[List[Tuple[int, int]]] = None,
    script: Optional[ScriptConfig] = None,
) -> SegmentData:
    """Clean text and produce per-segment alignment metadata.

    ``spans`` lets a caller declare the cue boundaries as inclusive (start, end) index
    pairs into ``text`` instead of having them derived from punctuation. Alignment treats
    them as opaque, so the only requirement is that they index ``text``. Used by --realign,
    whose input arrives pre-split into cues; see utils/schema.py SingleSegment.cue_spans.
    """
    num_leading = len(text) - len(text.lstrip())
    num_trailing = len(text) - len(text.rstrip())

    spaced = _spaced(model_lang, script)
    per_word = text.split(" ") if spaced else text

    clean_char, clean_cdx = [], []
    for cdx, char in enumerate(text):
        char_ = char.lower()
        if spaced:
            char_ = char_.replace(" ", "|")
        if cdx < num_leading or cdx > len(text) - num_trailing - 1:
            continue
        if char_ in model_dictionary or char_ in punctuation.split_chars:
            clean_char.append(char_)
            clean_cdx.append(cdx)

    clean_wdx = [
        wdx for wdx, wrd in enumerate(per_word)
        if any(c in model_dictionary for c in wrd.lower())
    ]

    return {
        "clean_char": clean_char,
        "clean_cdx": clean_cdx,
        "clean_wdx": clean_wdx,
        "sentence_spans": (
            list(spans) if spans is not None
            else _get_sentence_spans(text, model_lang, punctuation, script)
        ),
    }

def _preprocess_transcript(
    transcript: List[SingleSegment],
    model_lang: str,
    model_dictionary: dict,
    punctuation: PunctuationConfig = DEFAULT_PUNCTUATION,
    print_progress: bool = False,
    script: Optional[ScriptConfig] = None,
) -> dict:
    """First pass: build SegmentData for every transcript segment."""
    total = len(transcript)
    segment_data = {}
    for sdx, segment in enumerate(transcript):
        segment_data[sdx] = _preprocess_segment(
            segment["text"], model_lang, model_dictionary, punctuation,
            spans=segment.get("cue_spans"), script=script,
        )
    return segment_data


_TIMESTAMP_TOLERANCE_S = 0.005


def _find_vad_segment_idx(vad_segments: List[VadAudioSegment], t: float) -> Optional[int]:
    # Half-open [start - tol, end) so that a timestamp exactly at a segment boundary
    # belongs to the next segment rather than the one that just ended.
    for i, seg in enumerate(vad_segments):
        if seg["start"] - _TIMESTAMP_TOLERANCE_S <= t < seg["end"]:
            return i
    # Fallback: t is at or just past the end of the last segment.
    if vad_segments and t <= vad_segments[-1]["end"] + _TIMESTAMP_TOLERANCE_S:
        return len(vad_segments) - 1
    return None


def _compute_vad_emissions(
    vad_segments: List[VadAudioSegment],
    model: torch.nn.Module,
    model_type: str,
    processor,
    device: str,
    batch_size: int = 4,
    vram_checks: bool = True,
    primer: Optional["AudioPrimer"] = None,
    min_samples: int = MIN_ALIGN_SAMPLES,
    dtype: Optional[torch.dtype] = None,
) -> List[Tuple[torch.Tensor, float]]:
    """Run inference on each full VAD segment. Returns (log_softmax_emission, frame_rate) per segment.

    Logs before/after regardless of which path runs below, so a slow pass (a file
    with many/long VAD segments) is visibly explained rather than looking like a
    hang — this ran with no progress feedback at all before batching was added.

    The work is the model's backend's (``align_backends``). ``primer`` reaches only a backend
    that ``supports_primer`` -- the batched Hugging Face one. torchaudio bundles, which run one
    segment at a time, carry no profile primer today; priming there would need the same exact
    frame-length bookkeeping for no current caller, so it warns instead of half-doing it.

    ``dtype`` casts each segment's emission as it leaves the device; ``None`` keeps the
    model's own. ``EmissionTimeline`` passes its storage dtype, because the conversion has to
    happen per batch: casting after this returns means the whole file sits here at float32
    first -- about 1 GB an hour of audio for the 2.7k-token Cantonese vocabulary, three times
    what the timeline itself keeps.
    """
    if not vad_segments:
        return []
    start = time.perf_counter()
    logger.info("Computing alignment emissions for %d VAD segments...", len(vad_segments))

    # A segment too short for one feature frame gets an empty emission rather than a trip
    # through the feature extractor, which fails outright below 240 samples. VAD should no
    # longer produce one (Binarize._split_long), but a crash here costs a whole batch of
    # files, and there is no speech in 25 ms to align anyway. Downstream already copes with
    # an empty emission: EmissionTimeline skips it, so a line there is treated as unmatched.
    usable = [i for i, seg in enumerate(vad_segments) if len(seg["audio"]) >= min_samples]
    if len(usable) < len(vad_segments):
        short = [seg for seg in vad_segments if len(seg["audio"]) < min_samples]
        logger.warning(
            "Skipping %d VAD segment(s) shorter than one alignment frame (%d samples): %s",
            len(short), min_samples,
            ", ".join(f"{s['start']:.3f}-{s['end']:.3f}s" for s in short[:5]),
        )
    segments = [vad_segments[i] for i in usable]

    backend = get_align_backend(model_type)
    if primer is not None and not backend.supports_primer:
        logger.warning(
            "Align model profile configures an audio primer, but the %s backend does not "
            "apply it. First-character timings may be pinned to each segment's start.",
            backend.name,
        )
        primer = None
    computed = backend.emissions(
        segments, model, processor, device, batch_size,
        vram_checks=vram_checks, primer=primer, dtype=dtype,
    ) if segments else []

    vocab = computed[0][0].shape[-1] if computed else 0
    empty = torch.zeros((0, vocab), dtype=dtype or torch.float32)
    results: List[Tuple[torch.Tensor, float]] = [(empty, 0.0)] * len(vad_segments)
    for i, result in zip(usable, computed):
        results[i] = result
    logger.info("Alignment emissions computed in %.1fs", time.perf_counter() - start)
    return results


def compute_vad_emissions(vad_segments, model, model_type, processor, device, batch_size: int = 4, vram_checks: bool = True, primer=None, min_samples: int = MIN_ALIGN_SAMPLES, dtype=None):
    """Public wrapper around _compute_vad_emissions, for callers outside this module."""
    return _compute_vad_emissions(vad_segments, model, model_type, processor, device, batch_size, vram_checks=vram_checks, primer=primer, min_samples=min_samples, dtype=dtype)


class EmissionTimeline:
    """The file's emissions as one timeline, sliceable by time across chunk boundaries.

    The chunks are contiguous and gap-free (``Vad.cover_chunks``), so a slice spanning a join
    is exactly as valid as one inside a single chunk: the tear exists because the encoder
    cannot take a whole film at once, not because anything changes in the audio there.

    That matters more than it sounds. Before this, a transcript segment could only be aligned
    against the *one* chunk holding its start (``alignment._get_emission_for_segment``), so
    ``build_align_input`` had to re-cut the file to keep every line inside a single chunk --
    and a coarse guess a second or two out could then put the boundary *in front of* the line
    it was meant to contain, handing forced alignment a chunk that does not hold the line at
    all. On the Police Story 2 head that produced a 0.2 s cue at score 0.000 for a line whose
    speech had ended 1.5 s before the chunk began. Measured across the fixtures, 22-44% of the
    spans this module wants to align in one piece cross a chunk join, so the old constraint
    was not a corner case.

    Frame times come from each chunk's *own* frame count rather than from one global rate:
    ``_compute_vad_emissions_batched`` reports up to 2% variation between chunks, and a single
    rate would smear a cross-join slice at every boundary.

    Emissions are held as float16 and upcast per slice. Two hours is then about 1 GB against
    the 2 GB float32 copy ``align()`` holds today, and the same copy serves both the placement
    search and the final alignment -- one encoder pass over the file instead of two. That
    figure holds only if nothing else keeps the float32 emissions alive: a compute_fn should
    return ``dtype`` already (``compute_vad_emissions(..., dtype=EmissionTimeline.dtype)``),
    and ``from_computed`` must not hold on to the list it was given.
    """

    dtype = torch.float16

    def __init__(
        self,
        vad_segments: Sequence[VadAudioSegment],
        compute_fn: Callable[[List[VadAudioSegment]], List[Tuple[torch.Tensor, float]]],
        frame_rate: Optional[float] = None,
    ):
        self._nominal_rate = frame_rate
        self._segments = list(vad_segments)
        self._compute = compute_fn
        self._emissions: List[Optional[np.ndarray]] = [None] * len(self._segments)
        self._times: List[Optional[np.ndarray]] = [None] * len(self._segments)
        self._starts = [float(s["start"]) for s in self._segments]
        self._ends = [float(s["end"]) for s in self._segments]
        self.computed = 0

    @property
    def frame_rate(self) -> float:
        """Emission frames per second: the align model's nominal rate (``align_metadata
        ["frame_rate"]``), or failing that the rate measured on the first computed chunk.

        Used to *budget* a search before its emissions exist -- frame times themselves always
        come from each chunk's own frame count (see the class docstring).
        """
        if self._nominal_rate:
            return self._nominal_rate
        if not any(e is not None for e in self._emissions):
            self.ensure(0, 1)
        for times in self._times:
            if times is not None and len(times) > 1:
                return 1.0 / float(times[1] - times[0])
        return 25.0

    @property
    def file_start(self) -> float:
        return self._starts[0] if self._starts else 0.0

    @property
    def file_end(self) -> float:
        return self._ends[-1] if self._ends else 0.0

    def chunk_range(self, t0: float, t1: float) -> Tuple[int, int]:
        """The half-open range of chunks covering [t0, t1]."""
        lo = max(0, bisect.bisect_right(self._starts, t0) - 1)
        hi = bisect.bisect_right(self._starts, t1)
        return lo, max(hi, lo + 1)

    def ensure(self, lo: int, hi: int) -> None:
        missing = [i for i in range(lo, min(hi, len(self._segments)))
                   if self._emissions[i] is None]
        if not missing:
            return
        self._store(missing, self._compute([self._segments[i] for i in missing]))

    def _store(self, indices: Sequence[int], results) -> None:
        for i, (emission, _rate) in zip(indices, results):
            # A no-copy view when the emission is already self.dtype on the CPU, so the
            # caller dropping its tensor leaves exactly one copy behind.
            arr = emission.detach().to("cpu", dtype=self.dtype).numpy()
            self._emissions[i] = arr
            n = arr.shape[0]
            step = (self._ends[i] - self._starts[i]) / n if n else 0.0
            self._times[i] = self._starts[i] + np.arange(n, dtype=np.float64) * step
            self.computed += 1

    def slice(self, t0: float, t1: float) -> Tuple[torch.Tensor, np.ndarray]:
        """(emission[frames, vocab] as float32, absolute start time of each frame)."""
        lo, hi = self.chunk_range(t0, t1)
        self.ensure(lo, hi)
        ems, ts = [], []
        for i in range(lo, min(hi, len(self._segments))):
            times = self._times[i]
            if times is None or not len(times):
                continue
            a = int(np.searchsorted(times, t0, side="left"))
            b = int(np.searchsorted(times, t1, side="right"))
            if b > a:
                ems.append(self._emissions[i][a:b])
                ts.append(times[a:b])
        if not ems:
            # A request narrower than one frame. Give back the frame covering t0 so no caller
            # has to special-case an empty emission.
            i = min(max(lo, 0), len(self._segments) - 1)
            self.ensure(i, i + 1)
            times = self._times[i]
            if times is None or not len(times):
                return torch.zeros((0, 0)), np.empty(0, dtype=np.float64)
            k = min(max(int(np.searchsorted(times, t0, side="right")) - 1, 0), len(times) - 1)
            ems, ts = [self._emissions[i][k:k + 1]], [times[k:k + 1]]
        emission = ems[0] if len(ems) == 1 else np.concatenate(ems)
        frame_times = ts[0] if len(ts) == 1 else np.concatenate(ts)
        return torch.from_numpy(np.asarray(emission, dtype=np.float32)), frame_times

    @classmethod
    def from_computed(cls, vad_segments, results, frame_rate: Optional[float] = None):
        """Wrap emissions that have already been computed, so nothing is encoded twice.

        Nothing here may keep ``results`` alive once it has been stored. This used to serve it
        through a compute_fn closure, which pinned every float32 emission for the life of the
        timeline beside the float16 copy and tripled alignment's memory: a user's 5.4k-line
        file ran Windows out of commit at the very end of the alignment pass.
        """
        results = list(results)
        if len(results) != len(vad_segments):
            raise ValueError(
                f"{len(results)} emissions for {len(vad_segments)} VAD segments")

        def _already_computed(segs):
            raise RuntimeError("EmissionTimeline.from_computed holds every chunk already")

        timeline = cls(vad_segments, _already_computed, frame_rate=frame_rate)
        timeline._store(range(len(results)), results)
        return timeline


def _get_emission_for_segment(
    t1: float,
    t2: float,
    audio,
    vad_segments: Optional[List[VadAudioSegment]],
    vad_seg_emissions: Optional[List[Tuple[torch.Tensor, float]]],
    model: torch.nn.Module,
    model_type: str,
    processor,
    device: str,
    timeline=None,
) -> Optional[Tuple[torch.Tensor, Optional[np.ndarray]]]:
    """Return (emission, frame times) for one segment, or None if no audio matches.

    With a ``timeline`` (see ``realign.EmissionTimeline``) the emission is cut straight out of
    the file's timeline and may span chunk joins, and the exact absolute time of every frame
    comes back with it. Without one the old behaviour stands: the segment is sliced out of the
    single VAD chunk containing ``t1``, and the caller derives times from the segment's own
    duration. Frame times are ``None`` in that case.
    """
    if timeline is not None:
        emission, frame_times = timeline.slice(t1, t2)
        return (emission, frame_times) if emission.size(0) else None
    if vad_seg_emissions is not None:
        vad_idx = _find_vad_segment_idx(vad_segments, t1)
        if vad_idx is None:
            return None
        vad_seg = vad_segments[vad_idx]
        full_emission, frame_rate = vad_seg_emissions[vad_idx]
        t1_local = t1 - vad_seg["start"]
        t2_local = t2 - vad_seg["start"]
        e1 = int(t1_local * frame_rate)
        e2 = max(int(t2_local * frame_rate), e1 + 1)
        return full_emission[e1:e2, :], None

    f1 = int(t1 * SAMPLE_RATE)
    f2 = int(t2 * SAMPLE_RATE)
    waveform_segment = audio[:, f1:f2]
    if waveform_segment.shape[-1] < 400:
        lengths = torch.as_tensor([waveform_segment.shape[-1]]).to(device)
        waveform_segment = torch.nn.functional.pad(
            waveform_segment, (0, 400 - waveform_segment.shape[-1])
        )
    else:
        lengths = None
    emissions = _run_model_inference(model, model_type, waveform_segment, processor, device, lengths=lengths)
    return emissions[0].cpu().detach(), None


# How long a character's aligned span may run before it is read as a dwell rather than a
# character.
#
# CTC is peaky: a character fires on a frame or two and the path then sits on it, emitting
# blank, until whatever comes next arrives. merge_repeats reports that whole wait as the
# character's span, and where the wait is long the span swallows it. Measured on Police
# Story 2:
#
#     我@330.26-344.19  覺@344.19-344.35  得@344.35-344.47  ...
#
# Thirteen seconds on the first character and a tenth of a second on each one after it, which
# put that subtitle on screen fourteen seconds before anyone spoke. The same shape at the far
# edge gave 「唔該警察叔叔」 an 11.8 s cue whose last nine seconds are a pause.
#
# No guard that compares *consecutive* characters can see this -- there is no gap between
# characters anywhere in those cues -- and neither can an energy envelope, because the pause
# is room tone a dozen dB under the dialogue rather than actual silence. The emission says it
# plainly: within the dwell, the character's own token peaks on one frame and is negligible on
# the rest.
#
# A real syllable, even drawn out, does not hold the CTC path for a second.
MAX_CHAR_DWELL_SECONDS = 1.0


def _reseat_dwelling_chars(char_segments, emission, tokens, blank_id, seconds_per_frame):
    """Move any character that merely waited onto the frame its own token peaks at.

    Its replacement span is the median length of the segment's other characters, so the
    character keeps a plausible duration instead of collapsing to a single frame, and it is
    clipped against the next character so the sequence stays ordered. See
    MAX_CHAR_DWELL_SECONDS.
    """
    if len(char_segments) != len(tokens) or seconds_per_frame <= 0:
        return 0
    limit = max(int(round(MAX_CHAR_DWELL_SECONDS / seconds_per_frame)), 2)
    lengths = [cs.end - cs.start for cs in char_segments]
    typical = max(int(np.median(lengths)), 1)
    moved = 0
    for idx, cs in enumerate(char_segments):
        if cs.end - cs.start <= limit or tokens[idx] == blank_id:
            continue
        window = emission[cs.start:cs.end, tokens[idx]]
        if window.numel() == 0:
            continue
        peak = cs.start + int(torch.argmax(window).item())
        ceiling = char_segments[idx + 1].start if idx + 1 < len(char_segments) else cs.end
        cs.start = peak
        cs.end = max(min(peak + typical, max(ceiling, peak + 1), cs.end), peak + 1)
        moved += 1
    if moved:
        logger.debug(
            "Re-seated %d character(s) that held the alignment path for more than %.1fs onto "
            "the frame their own token peaks at", moved, MAX_CHAR_DWELL_SECONDS,
        )
    return moved


def _align_segment(
    segment: SingleSegment,
    seg_data: SegmentData,
    emission: torch.Tensor,
    model_dictionary: dict,
    model_lang: str,
    blank_id: int,
    spacing_char_id: int,
    t1: float,
    t2: float,
    interpolate_method: str,
    return_char_alignments: bool,
    spotchecks: Mapping[str, SpotCheck],
    punctuation: PunctuationConfig,
    frame_times: Optional[np.ndarray] = None,
    script: Optional[ScriptConfig] = None,
) -> List[dict]:
    """Align one transcript segment against its emission, returning subsegment dicts.

    ``frame_times`` gives the absolute time of each emission frame. When present it is used
    verbatim, which is what makes an emission spanning several VAD chunks correct: those
    chunks can differ in frame rate by a couple of percent, and stretching one rate across the
    join would smear every timestamp after it. Without it the old assumption holds -- the
    emission covers exactly [t1, t2] at a constant rate.
    """
    text = segment["text"]
    avg_logprob = segment.get("avg_logprob")

    base_seg: SingleAlignedSegment = {"start": t1, "end": t2, "text": text, "words": [], "chars": None}
    if avg_logprob is not None:
        base_seg["avg_logprob"] = avg_logprob
    if return_char_alignments:
        base_seg["chars"] = []

    if len(seg_data["clean_char"]) == 0:
        logger.warning(f'Failed to align segment ("{text}"): no characters in this segment found in model dictionary, resorting to original')
        return [base_seg]

    text_clean = "".join(seg_data["clean_char"])

    # Replace punctuation with spacing token to better align breaks at sentence ends
    split_chars = punctuation.split_chars
    tokens = [model_dictionary[c] if c not in split_chars else spacing_char_id for c in text_clean]

    trellis = get_trellis(emission, tokens, blank_id)
    path = backtrack(trellis, emission, tokens, blank_id)

    # Spot checks: for each char with an interchangeable candidate set (per the model's
    # profile), pick the candidate whose acoustic log-prob at this char's aligned frame is
    # highest, plus any per-candidate bias weight. Empty `spotchecks` (the default for a
    # model whose output already uses the intended particles) makes this loop a no-op.
    if spotchecks and path is not None:
        logger.debug("Checking particle candidates for text: '%s'.", text_clean)

        lowercase_text = text.lower()
        t_i = 0
        for p_i, p in enumerate(text_clean):
            # Use t_i to mark the position in the base "text" var. Keep this updated to avoid conflicts.
            # TODO: roll text, text_clean, and seg_data["clean_char"] all up into a single dynamic type
            t_i = t_i + lowercase_text[t_i:].index(p) + 1

            sc = spotchecks.get(p)
            if sc is None or len(sc.candidates) <= 1:
                continue

            path_i = min(x.time_index for x in path if x.token_index == p_i)

            max_score = -math.inf
            best_candidate = None
            for c in sc.candidates:
                c_token = model_dictionary.get(c)
                if c_token is None:
                    logger.warning("Spot-check candidate %r absent from align model vocab; skipping.", c)
                    continue
                score = emission[path_i, c_token].item() + sc.weights.get(c, 0.0)
                if score > max_score:
                    best_candidate = c
                    max_score = score

            if best_candidate is None:
                continue

            if best_candidate != p:
                text_clean = text_clean[:p_i] + best_candidate + text_clean[p_i + 1:]
                text = text[:t_i - 1] + best_candidate + text[t_i:] # messy :(

            logger.debug("Best candidate for char '%d' ('%s'): '%s' (score %.3f).", p_i, p, best_candidate, max_score)

    seg_data["clean_char"] = [c for c in text_clean]

    if path is None:
        logger.warning(f'Failed to align segment ("{text}"): backtrack failed, resorting to original')
        return [base_seg]

    seconds_per_frame = (
        float(frame_times[1] - frame_times[0])
        if frame_times is not None and len(frame_times) > 1
        else (t2 - t1) / max(trellis.size(0) - 1, 1)
    )
    char_segments = merge_repeats(path, text_clean)
    _reseat_dwelling_chars(char_segments, emission, tokens, blank_id, seconds_per_frame)
    if frame_times is not None and len(frame_times):
        # merge_repeats reports an *exclusive* end, so the map needs one entry past the last
        # frame; extend by the final step rather than clamping, which would collapse the last
        # character to zero length.
        step = float(frame_times[-1] - frame_times[-2]) if len(frame_times) > 1 else 0.04
        edges = np.append(np.asarray(frame_times, dtype=np.float64), frame_times[-1] + step)
        last = len(edges) - 1

        def _at(frame: float) -> float:
            return float(edges[min(max(int(round(frame)), 0), last)])
    else:
        ratio = (t2 - t1) / max(trellis.size(0) - 1, 1)

        def _at(frame: float) -> float:
            return frame * ratio + t1

    char_segments_arr = []
    word_idx = 0
    for cdx, char in enumerate(text):
        start, end, score = None, None, None
        if cdx in seg_data["clean_cdx"]:
            char_seg = char_segments[seg_data["clean_cdx"].index(cdx)]
            start = round(_at(char_seg.start), 3)
            end = round(_at(char_seg.end), 3)
            score = round(char_seg.score, 3)
        char_segments_arr.append({"char": char, "start": start, "end": end, "score": score, "word-idx": word_idx})
        if not _spaced(model_lang, script):
            word_idx += 1
        elif cdx == len(text) - 1 or text[cdx + 1] == " ":
            word_idx += 1

    char_segments_arr = pd.DataFrame(char_segments_arr)
    char_segments_arr["sentence-idx"] = None
    aligned_subsegments = []

    for sdx2, (sstart, send) in enumerate(seg_data["sentence_spans"]):
        mask = (char_segments_arr.index >= sstart) & (char_segments_arr.index <= send)
        curr_chars = char_segments_arr.loc[mask]
        char_segments_arr.loc[mask, "sentence-idx"] = sdx2

        end_chars = curr_chars[curr_chars["char"] != ' ']
        if len(end_chars) == 0:
            continue

        sentence_text = text[sstart:send + 1]
        sentence_start = curr_chars["start"].min()
        last_char = end_chars.iloc[-1]
        sentence_end = end_chars["end"].max()
        # Sentences ending on punctuation get their end time released (extended) later,
        # in align(), once the position relative to *all* subsegments in the file
        # (not just this transcript segment) is known — see release_from below.
        release_from = last_char["start"] if last_char["char"] in split_chars else None

        sentence_words = []
        for word_idx in curr_chars["word-idx"].unique():
            word_chars = curr_chars.loc[curr_chars["word-idx"] == word_idx]
            word_text = "".join(word_chars["char"].tolist()).strip()
            if not word_text:
                continue
            word_chars = word_chars[word_chars["char"] != " "]
            word_start = word_chars["start"].min()
            word_end = word_chars["end"].max()
            word_score = round(word_chars["score"].mean(), 3)
            word_segment = {"word": word_text}
            if not np.isnan(word_start):
                word_segment["start"] = word_start
            if not np.isnan(word_end):
                word_segment["end"] = word_end
            if not np.isnan(word_score):
                word_segment["score"] = word_score
            sentence_words.append(word_segment)

        subsegment = {
            "text": sentence_text,
            "start": sentence_start,
            "end": sentence_end,
            "words": sentence_words,
            "release_from": release_from,
        }
        # A caller that declared its own cue spans may also say why a cue is doubtful
        # (--realign). Carry it onto the finished cue: the reason is known before alignment
        # runs and there is nothing downstream that could reconstruct it.
        cue_reasons = segment.get("cue_reasons")
        if cue_reasons and sdx2 < len(cue_reasons) and cue_reasons[sdx2]:
            subsegment["realign_reason"] = cue_reasons[sdx2]
        if avg_logprob is not None:
            subsegment["avg_logprob"] = avg_logprob
        aligned_subsegments.append(subsegment)

        if return_char_alignments:
            chars_out = curr_chars[["char", "start", "end", "score"]].copy()
            chars_out.fillna(-1, inplace=True)
            aligned_subsegments[-1]["chars"] = [
                {k: v for k, v in row.items() if v != -1}
                for row in chars_out.to_dict("records")
            ]

    aligned_subsegments = pd.DataFrame(aligned_subsegments)
    aligned_subsegments["start"] = interpolate_nans(aligned_subsegments["start"], method=interpolate_method)
    aligned_subsegments["end"] = interpolate_nans(aligned_subsegments["end"], method=interpolate_method)

    # Concatenate sentences with same timestamps
    if "realign_reason" not in aligned_subsegments.columns:
        aligned_subsegments["realign_reason"] = None
    agg_dict = {"text": " ".join, "words": "sum", "release_from": "first",
                "realign_reason": "first"}
    if not _spaced(model_lang, script):
        agg_dict["text"] = "".join
    if return_char_alignments:
        agg_dict["chars"] = "sum"
    if avg_logprob is not None:
        agg_dict["avg_logprob"] = "first"

    aligned_subsegments = aligned_subsegments.groupby(["start", "end"], as_index=False).agg(agg_dict)
    records = aligned_subsegments.to_dict("records")
    for row in records:
        # A column of Nones comes back from groupby as NaN, and NaN is *truthy* -- test the
        # type, not the value, or every cue in the file ends up carrying a float "reason".
        if not isinstance(row.get("realign_reason"), str) or not row["realign_reason"]:
            row.pop("realign_reason", None)
    return records


# --- Public functions ---

def _nominal_frame_rate(model, processor, model_type: str, device) -> Optional[float]:
    """Emission frames per second of audio, from the model's backend (see
    ``AlignBackend.frame_rate``), or None if it cannot say."""
    try:
        return get_align_backend(model_type).frame_rate(model, processor, device)
    except Exception as exc:  # the timeline falls back to measuring it
        logger.warning("Could not determine the align model's frame rate: %s", exc)
        return None


def load_align_model(
    language_code: str, device: str, device_index: int = 0, model_name: Optional[str] = None,
    model_dir=None, model_cache_only: bool = False, compute_type: str = "float32",
    vram_checks: bool = True,
    char_substitution: Optional[str] = None,
    substitution_overrides: Optional[Mapping[str, str]] = None,
):
    """Load the phoneme-alignment model.

    ``char_substitution`` None takes the model profile's level (homophone for the Cantonese
    model, off for anything else): the homophone tiers read Cantonese pronunciations.

    compute_type="float16" halves weight VRAM but the model is otherwise loaded and
    invoked exactly like float32 (no autocast) — inputs are cast to match in
    _run_model_inference/_compute_vad_emissions_batched, and this is deliberately
    opt-in with float32 as the default since it can measurably affect forced-alignment
    accuracy.
    """
    from cantocaptions_ai.languages import get_language_pack
    pack = get_language_pack(language_code)
    if model_name is None:
        # The language pack's choice; an unregistered language's generic pack takes the
        # built-in tables (languages/align_defaults.py).
        model_name = pack.default_align_model
        if model_name is None:
            logger.error(
                f"No default alignment model for language: {language_code}. "
                f"Please find a wav2vec2.0 model finetuned on this language at https://huggingface.co/models, "
                f"then pass the model name via --align_model [MODEL_NAME]"
            )
            raise ValueError(f"No default align-model for language: {language_code}")

    device = resolve_device(device, device_index)
    dtype = resolve_torch_compute_dtype(compute_type, device, "align")

    backend = align_backend_for(model_name, cache_dir=model_dir, local_files_only=model_cache_only)
    align_model, processor, align_dictionary = backend.load(
        model_name, device, dtype, model_dir=model_dir, cache_only=model_cache_only,
        vram_checks=vram_checks,
    )
    pipeline_type = backend.name

    profile = get_align_profile(model_name)
    if char_substitution is None:
        char_substitution = profile.char_substitution
    readings = pack.char_readings() if pack.char_readings is not None else None
    if readings is None and char_substitution != "off":
        logger.warning(
            "Align char substitution %r needs character readings, which language %r does not "
            "provide; only the align model's substitution table and --align_substitutions apply.",
            char_substitution, language_code,
        )
    align_metadata = {
        "language": language_code,
        "dictionary": align_dictionary,
        "type": pipeline_type,
        # The model's own feature extractor (None for a torchaudio bundle). Every stage that
        # runs the encoder takes it from here, so the model and its inputs cannot mismatch.
        "processor": processor,
        # Frames per second the model emits, read off its own length arithmetic; realign
        # budgets its searches with it before any emission exists.
        "frame_rate": _nominal_frame_rate(align_model, processor, pipeline_type, device),
        # Built here, next to the dictionary it edits, so every stage that tokenises text
        # against this model shares one vocabulary. Resolves nothing until a caller hands it
        # some text -- see align_vocab.VocabRepair.
        #
        # The model's own bundled table sits *under* whatever the caller passed, merged per
        # character, so --align_substitutions can correct one entry without restating the
        # rest. Both beat every automatic tier.
        "vocab_repair": VocabRepair(
            align_dictionary, char_substitution,
            merge_substitutions(
                bundled_substitutions(profile.substitutions) if profile.substitutions else None,
                substitution_overrides,
            ),
            readings=readings,
        ),
        # Resolved once here rather than in align(), which then has no idea which model it
        # is holding. Unknown models get the all-no-op default.
        "profile": profile,
    }
    return align_model, align_metadata


def align(
    transcript: Iterable[SingleSegment],
    model: torch.nn.Module,
    align_model_metadata: dict,
    audio: Union[str, np.ndarray, torch.Tensor, List[VadAudioSegment]],
    device: str,
    processor=None,
    align_padding: float = 0.04,
    align_release: float = 0.4,
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
    print_progress: bool = False,
    progress_callback: ProgressCallback = None,
    batch_size: int = 4,
    vram_checks: bool = True,
    spotchecks: Optional[Mapping[str, SpotCheck]] = None,
    punctuation: PunctuationConfig = DEFAULT_PUNCTUATION,
    timeline=None,
    split_gap: Optional[float] = None,
    script: Optional[ScriptConfig] = None,
) -> AlignedTranscriptionResult:
    """Align phoneme recognition predictions to known transcription.

    ``spotchecks`` and ``punctuation`` come from the ASR model's profile (see
    ``pipeline/model_profiles.py``); their defaults (no spot checks, standard
    punctuation) keep alignment independent of any specific model.

    ``split_gap`` breaks any cue holding a silence at least that long between two of its own
    adjacent characters into one cue per utterance (``align_checks.split_gapped_cues``).
    ``None`` defers to the *align* model's own profile (``AlignProfile.split_gap``), which
    is itself ``None`` -- never split -- unless a profile sets it; ``0`` turns it off whatever
    the profile says. A caller that declares its own cue structure should pass ``0``: under
    ``--realign`` the transcript's line breaks *are* the cue boundaries, and a cue is one
    whole transcript line by contract.

    ``timeline`` (``realign.EmissionTimeline``) replaces the per-chunk emission set: segments
    are then cut out of one continuous timeline, so a segment may span chunk joins and the
    encoder is not run a second time over audio the caller has already encoded.

    ``processor`` defaults to the align model's own, from ``align_model_metadata``.

    ``script`` (from the ASR model's profile) decides whether spaces separate words, which
    changes how words are counted, how sentences are split and how subsegments rejoin. None
    derives it from the align model's language.
    """
    spotchecks = spotchecks or {}
    if processor is None:
        processor = align_model_metadata.get("processor")

    # --- Audio setup ---
    vad_segments: Optional[List[VadAudioSegment]] = None
    if isinstance(audio, list):
        vad_segments = audio
        MAX_DURATION = max(seg["end"] for seg in vad_segments) if vad_segments else 0.0
    else:
        if not torch.is_tensor(audio):
            if isinstance(audio, str):
                audio = load_audio(audio)
            audio = torch.from_numpy(audio)
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        MAX_DURATION = audio.shape[1] / SAMPLE_RATE

    model_dictionary = align_model_metadata["dictionary"]
    model_lang = align_model_metadata["language"]
    model_type = align_model_metadata["type"]
    # .get so hand-built metadata dicts (tests, callers predating align_profiles) still work.
    profile = align_model_metadata.get("profile") or DEFAULT_ALIGN_PROFILE
    blank_id = _get_blank_id(model_dictionary)
    spacing_char_id = blank_id # model_dictionary['！']

    # One timeline for the file whichever way we got here. --realign's acoustic anchor hands
    # one in (already populated, so the encoder does not run twice over the same audio);
    # otherwise the emissions are computed here and wrapped. Either way a transcript segment
    # can be cut across chunk joins, which is what stops a boundary landing in front of the
    # line it was meant to contain.
    if timeline is None and vad_segments is not None:
        timeline = EmissionTimeline.from_computed(
            vad_segments,
            _compute_vad_emissions(
                vad_segments, model, model_type, processor, device, batch_size,
                vram_checks=vram_checks, primer=profile.primer, min_samples=profile.min_samples,
                dtype=EmissionTimeline.dtype,
            ),
            frame_rate=align_model_metadata.get("frame_rate"),
        )
    vad_seg_emissions = None

    # --- Preprocess transcript ---
    transcript = list(transcript)
    # Before anything reads the dictionary: give it a token for the characters it has none
    # for, so _preprocess_segment keeps them instead of dropping them. Idempotent, so under
    # --realign (where the coarse search already ran this over the same transcript against
    # the same dictionary) this is a no-op and the two passes cannot disagree.
    repair = align_model_metadata.get("vocab_repair")
    if repair is not None:
        repair.augment(segment.get("text", "") for segment in transcript)
        # A substituted character's token is some homophone's, so it cannot be asked which
        # of two particles the audio supports. Never fires for the shipped profiles.
        spotchecks = filter_spotchecks(spotchecks, repair.substitutions)
    segment_data = _preprocess_transcript(
        transcript, model_lang, model_dictionary, punctuation, print_progress, script=script,
    )

    # --- Align each segment ---
    aligned_segments: List[SingleAlignedSegment] = []
    untranscribed = 0

    for sdx, segment in enumerate(transcript):
        t1 = segment["start"]
        t2 = segment["end"]
        text = segment["text"]
        avg_logprob = segment.get("avg_logprob")

        # A segment ASR returned nothing for is not a cue and must not become one. Left in,
        # it is an empty cue covering that whole VAD segment, and cue assembly then treats it
        # as an ordinary neighbour: its blank text has no punctuation, so pass A reads the
        # join as clean and glues the cues on either side of it into one spanning the
        # silence. Note the test is the *text*, not whether anything aligned -- a segment
        # whose characters are all out of the align vocabulary keeps its cue and its text
        # (see align_vocab), it simply has no word timings.
        if not str(text).strip():
            untranscribed += 1
            if progress_callback is not None:
                progress_callback.advance(1)
            continue

        base_seg: SingleAlignedSegment = {"start": t1, "end": t2, "text": text, "words": [], "chars": None}
        if avg_logprob is not None:
            base_seg["avg_logprob"] = avg_logprob
        if return_char_alignments:
            base_seg["chars"] = []

        if t1 >= MAX_DURATION:
            logger.warning(f'Failed to align segment ("{text}"): original start time longer than audio duration, skipping')
            aligned_segments.append(base_seg)
            continue

        found = _get_emission_for_segment(
            t1, t2, audio, vad_segments, vad_seg_emissions,
            model, model_type, processor, device, timeline=timeline,
        )
        emission, frame_times = found if found is not None else (None, None)
        if emission is None:
            logger.warning(f'Failed to align segment ("{text}"): no VAD segment found for start time {t1}, skipping')
            aligned_segments.append(base_seg)
            continue

        subsegments = _align_segment(
            segment, segment_data[sdx], emission,
            model_dictionary, model_lang, blank_id, spacing_char_id,
            t1, t2, interpolate_method, return_char_alignments,
            spotchecks, punctuation, frame_times=frame_times, script=script,
        )
        aligned_segments += subsegments

        if progress_callback is not None:
            progress_callback.advance(1)

    # --- Release punctuation-terminated ends, then trim overlaps against the next
    # subsegment's start. Done once over the whole file (not per transcript segment)
    # so that a released end can't collide with the first subsegment of the next
    # VAD segment, which _align_segment has no visibility into.
    if aligned_segments:
        starts = pd.Series([seg["start"] for seg in aligned_segments], dtype="float64")
        ends = pd.Series([seg["end"] for seg in aligned_segments], dtype="float64")
        release_froms = pd.Series(
            [seg.pop("release_from", None) for seg in aligned_segments], dtype="float64"
        )

        release_mask = release_froms.notna()
        ends[release_mask] = (release_froms[release_mask] + align_release).round(3)

        next_starts = starts.shift(-1)
        overlap = ends > next_starts - align_padding
        ends[overlap] = (next_starts[overlap] - align_padding).round(3)

        for seg, new_end in zip(aligned_segments, ends):
            seg["end"] = float(new_end)

    # --- Break a cue that turned out to hold two utterances, if this align model's profile
    # (or the caller) asked for it. Opt-in, and off for every shipped profile: see
    # align_checks.split_gapped_cues for why reporting such a cue is free and breaking it is
    # a bet. Runs *after* the release/trim pass so the cut is made against final edges --
    # each piece keeps the outer edge it was given and takes its inner one from its own
    # characters, so nothing overlaps and no cue outside the split moves -- and *before* the
    # checks below, so they judge the cues the rest of the pipeline will see.
    split_gap = profile.split_gap if split_gap is None else split_gap
    if split_gap and aligned_segments:
        aligned_segments, breaks = split_gapped_cues(
            aligned_segments, split_gap, punctuation.split_chars,
        )
        if breaks:
            logger.info(
                "Broke %d cue boundary%s out of a silence of %.1fs or more inside a cue",
                breaks, "" if breaks == 1 else "s", split_gap,
            )

    # --- Validate. Model-agnostic and always on: a cue start sitting on silence is wrong
    # whichever align model produced it. Runs on final timings, after the release/trim
    # pass, so it judges what the rest of the pipeline will actually consume.
    silent = warn_on_silent_starts(
        aligned_segments,
        vad_segments if vad_segments is not None else whole_file_region(audio, MAX_DURATION),
    )
    # Land the finding on the cue rather than only printing it: a warning that names no cue
    # cannot be acted on across a 2000-line file. setdefault so a reason set at placement
    # time ("no audio for this line") outranks a symptom of it.
    for hit in silent:
        if 0 <= hit.index < len(aligned_segments):
            aligned_segments[hit.index].setdefault("realign_reason", "silent_start")

    # ...and the same question asked of the timings alone: a cue holding a silence between
    # two of its own adjacent characters is not one utterance, and its edges are the least
    # trustworthy in the file. A note rather than a reason or a repair -- see
    # align_checks.find_gapped_cues for why it is deliberately not fixed here.
    for gapped in warn_on_gapped_cues(aligned_segments, split_chars=punctuation.split_chars):
        add_note(aligned_segments[gapped.index],
                 f"internal_gap:{gapped.gap:.1f}s after {gapped.before}")

    # Record on each cue which of its characters the model could not read as written. Done
    # here rather than at substitution time because a substitution is per *character* over
    # the whole file, while what a reader wants to see is the handful of cues it touched.
    if repair is not None and (repair.substitutions or repair.unresolved):
        for seg in aligned_segments:
            for note in substitution_notes(
                seg.get("text", ""), repair.substitutions, repair.unresolved,
            ):
                add_note(seg, note)

    if untranscribed:
        logger.info(
            "%d segment(s) carried no transcribed text and produced no cue", untranscribed,
        )

    # --- Collect word segments ---
    word_segments: List[SingleWordSegment] = [w for seg in aligned_segments for w in seg["words"]]
    return {"segments": aligned_segments, "word_segments": word_segments}

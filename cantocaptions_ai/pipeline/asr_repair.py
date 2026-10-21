"""Catching ASR loops and foreign-language speech, and repairing the chunks they spoil.

A Qwen3-ASR chunk goes wrong in a few recognisable ways, each detected here from the first
pass's own output -- no audio, no model:

- **A repetition loop.** One phrase over and over (``我哋揀咗喇，`` ×16) until the token budget
  runs out, often ending on a half-written copy or a broken character (U+FFFD: the budget
  ran out part way through a multi-byte character). The processor's own repetition pass
  folds only runs of 20 or more, and silently, so a fold having fired (``raw_text``) is
  itself a loop. See :func:`find_loop` and :func:`loop_verdict`.
- **Speech in another language.** The fine-tuned Cantonese model writes Italian either as
  Italian, as invented Cantonese, or not at all, and it writes no language label to say so
  (the base model's ``language X<asr_text>`` prefix was trained out of it). What is left to
  go on: Latin-script text that is not English (:func:`latin_language`), a chunk the model
  wrote nothing for, and a loop -- the model lost on audio it cannot read.

Repair (``AsrRepairStage``, driven by :func:`plan_chunks`, :func:`run_ladder`,
:func:`run_language_check` and :func:`finish`) re-runs the flagged chunks. A loop, or a chunk
that changes script part way, is cut at its deepest pause and each piece re-transcribed by
the run's own model; a piece that still loops is cut again, up to ``split_depth``. Then a
multilingual model (base Qwen3-ASR) identifies each piece's language itself, and
:func:`combine` decides what the piece ends up saying: speech in a language the run does not
keep (``native``) is transcribed in that language, dropped, or only noted (``asr_foreign``).
Whatever still loops afterwards is trimmed to the house ``X…`` form by :func:`trim_loop`, in
the repair stage's tidy-up.

Everything here but the two model calls is pure (text and numpy arrays in, decisions out),
so the eval script (``scripts/eval_hallucination.py``) measures exactly what the pipeline
runs.
"""
import re
import zlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cantocaptions_ai.text_profiles import LATIN_WORD

# What may sit between two copies of a repeated phrase.
LOOP_SEPARATORS = "，,、…。.！!？? ～~\u3000 "
REPLACEMENT_CHAR = "\ufffd"
# The longest phrase a loop is looked for in (characters, separators included).
MAX_LOOP_UNIT = 40

_CJK = re.compile("[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002ebef\uf900-\ufaff]")


# --- Loops ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Loop:
    """A run of one phrase repeated back to back in a chunk's text.

    ``text[start:end]`` is the run; ``unit`` one copy of the phrase as it repeats (with the
    separator that follows it, if any); ``copies`` whole copies in the run, counting a last
    one that has no separator; ``truncated`` that the text ends in a part-copy -- a loop the
    token budget cut off.
    """

    unit: str
    copies: int
    start: int
    end: int
    truncated: bool = False

    @property
    def phrase(self) -> str:
        """The repeated phrase without its separators."""
        return self.unit.strip(LOOP_SEPARATORS)

    @property
    def covered(self) -> int:
        return self.end - self.start


def _strip_tail(text: str) -> str:
    return text.rstrip(REPLACEMENT_CHAR + LOOP_SEPARATORS)


def find_loop(text: str, max_unit: int = MAX_LOOP_UNIT) -> Optional[Loop]:
    """The repeated run in *text* that covers the most characters, or None.

    A run is one phrase (with its trailing separator, if it has one) repeated at least twice.
    A last copy without the separator still counts (``拜拜，拜拜，拜拜`` is three copies), and
    a part-copy at the very end of the text marks the run truncated. Of runs covering the
    same span, the one with the shortest phrase wins.
    """
    if not text:
        return None
    best: Optional[Loop] = None
    for size in range(1, min(max_unit, len(text) // 2) + 1):
        for match in re.finditer(r"(.{%d})\1+" % size, text, flags=re.DOTALL):
            candidate = _run_at(text, match.start(), size)
            if candidate is not None and (best is None or candidate.covered > best.covered):
                best = candidate
    return best


def _run_at(text: str, start: int, size: int) -> Optional[Loop]:
    """The run of ``text[start:start + size]`` from *start*, in the phase that ends each copy
    on its separator: a match found at ``晒，唔該晒，唔該…`` is read as ``唔該晒，`` copies, so
    the end of the text is not mistaken for a cut-off copy."""
    unit = text[start:start + size]
    if not unit.strip(LOOP_SEPARATORS):
        return None
    cut = max(unit.rfind(c) for c in LOOP_SEPARATORS)
    if 0 <= cut < size - 1 and unit[cut + 1] not in LOOP_SEPARATORS:
        start += cut + 1
        unit = text[start:start + size]
    copies, end = 0, start
    while text.startswith(unit, end):
        copies += 1
        end += size
    if copies < 2:
        return None
    rest = text[end:]
    phrase = unit.rstrip(LOOP_SEPARATORS)
    if phrase != unit and rest.startswith(phrase):
        copies += 1
        end += len(phrase)
        rest = text[end:]
    tail = _strip_tail(rest)
    if tail:
        truncated = len(tail) < len(phrase) and phrase.startswith(tail[:1])
    else:  # nothing after the run but separators, or the U+FFFD of a cut character
        truncated = REPLACEMENT_CHAR in rest
    return Loop(unit, copies, start, end, truncated)


@dataclass(frozen=True)
class LoopRule:
    """When a chunk's text counts as a loop (``asr_loop_min_copies`` and friends).

    Defaults chosen on the debug checkpoints by ``eval_hallucination.py scan``: runs of five
    or more copies were loops in every case looked at, while runs of three or four are
    mostly real speech (拜拜，拜拜，拜拜; 早晨 ×4).
    """

    min_copies: int = 5
    min_chars: int = 24
    # Copies that are enough when the text also ends in a cut-off part-copy.
    truncated_copies: int = 3
    # zlib compression ratio (bytes in / bytes out) above which a text counts as a loop;
    # 0 = not used. Only texts of at least ``compression_min_chars`` are judged.
    max_compression: float = 0.0
    compression_min_chars: int = 30

    def visible(self, loop: Optional[Loop]) -> bool:
        """Whether *loop* is long enough to count, from the text alone."""
        if loop is None:
            return False
        if loop.copies >= self.min_copies and loop.covered >= self.min_chars:
            return True
        return loop.truncated and loop.copies >= self.truncated_copies


def compression_ratio(text: str) -> float:
    """UTF-8 bytes over zlib-compressed bytes: high for a text that repeats itself."""
    data = text.encode("utf-8")
    if not data:
        return 0.0
    return len(data) / len(zlib.compress(data))


def loop_verdict(seg: dict, rule: LoopRule = LoopRule()) -> Tuple[Optional[str], Optional[Loop]]:
    """Why a segment counts as a repetition loop (None if it does not), and its longest run.

    Evidence, strongest first: a repaired piece still looped (``asr_looped``); the retry
    also spent the whole budget; the processor folded a loop away (``raw_text``); the text
    ends in a broken character; a long enough run of one phrase, or a run cut off at the
    end; a text that compresses too well.
    """
    text = seg.get("text") or ""
    loop = find_loop(text)
    if seg.get("asr_looped"):
        return "looped", loop
    if seg.get("retry_budget_hit"):
        return "retry_looped", loop
    if seg.get("raw_text"):
        return "folded", find_loop(seg["raw_text"]) or loop
    if text.rstrip().endswith(REPLACEMENT_CHAR):
        return "broken_char", loop
    if rule.visible(loop):
        return ("repeats" if loop.copies >= rule.min_copies else "truncated"), loop
    if (rule.max_compression and len(text) >= rule.compression_min_chars
            and compression_ratio(text) > rule.max_compression):
        return "compresses", loop
    return None, loop


def strip_broken_chars(text: str) -> str:
    """Drop U+FFFD, the half of a character a token budget cut through."""
    return text.replace(REPLACEMENT_CHAR, "")


def trim_loop(text: str, loop: Optional[Loop] = None, *, ellipsis: str = "…") -> str:
    """*text* with its repeated run written once, as ``phrase…`` (the house form for "said more
    than twice"), and anything after it that the loop left -- a cut-off copy, U+FFFD -- dropped.

    Text after the run that is not part of it (the loop ended and speech went on) is kept.
    Returns *text* with only U+FFFD removed when there is no run of at least two copies.
    """
    text = strip_broken_chars(text)
    loop = loop if loop is not None else find_loop(text)
    if loop is None or loop.copies < 2 or text[loop.start:loop.end].count(loop.phrase) < 2:
        return text
    head = text[:loop.start]
    rest = text[loop.end:]
    tail = _strip_tail(rest)
    if loop.truncated or (tail and len(tail) < len(loop.phrase) and loop.phrase.startswith(tail[:1])):
        rest = ""
    return f"{head}{loop.phrase}{ellipsis}{rest.lstrip(LOOP_SEPARATORS)}"


# --- Language -------------------------------------------------------------------------------

def cjk_share(text: str) -> float:
    """Chinese characters as a share of the text's letters (Han + Latin letters and digits)."""
    han = len(_CJK.findall(text))
    latin = sum(len(m.group(0)) for m in LATIN_WORD.finditer(text))
    total = han + latin
    return han / total if total else 0.0


# Function words common in each language and rare in English (and vice versa), lower case.
# Only "is this Latin-script text English?" matters to the pipeline; which other language it
# names is a best guess, used only when the multilingual model's own label is no help.
_STOPWORDS: Dict[str, frozenset] = {
    "English": frozenset(
        "the you and to is it that of what me my we are this for on with be have do don't not "
        "your he she they was can just but how where from about okay hey thank thanks please "
        "here there all get know like want i'm it's you're what's let's going would will yeah "
        "sorry sure right good why who when".split()),
    "Italian": frozenset(
        "che di il lo gli non è per una uno sono sei mi ti ci si ma con del della questo questa "
        "cosa perché grazie prego dai qua qui anche tutto molto tanto bene io tu lui lei noi voi "
        "loro ho hai ha siamo era al alla nel nella sulla sulle più dove quando ragazza "
        "insieme vieni andiamo sì".split()),
    "Spanish": frozenset(
        "que el los las y es por para muy pero está estoy gracias señor qué donde porque hola "
        "también ella nosotros usted bueno vamos aquí ahora".split()),
    "French": frozenset(
        "le les des et est je tu il elle nous vous ils pas ne qui une pour dans sur avec ce "
        "cette merci oui bonjour très mais où c'est suis".split()),
    "German": frozenset(
        "der die das und ist ich du er sie wir ihr nicht ein eine zu mit auf für von den dem ja "
        "nein danke bitte was wie wo aber auch sehr gut".split()),
    "Portuguese": frozenset(
        "os não um uma com eu você ele ela nós obrigado obrigada sim muito também".split()),
}
_DIACRITICS = re.compile("[àèéìòùáíóúâêîôûäöüßñçãõ]", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-zÀ-ÿ']+")


def latin_language(text: str, min_hits: int = 2) -> Optional[str]:
    """Which language *text*'s Latin-script words are in, from their function words: "English",
    another language's name, or None when there are too few to tell.

    Not a language identifier, only a check on one: the multilingual model labels whole
    chunks, and has called an Italian chunk "English" (and a Cantonese-and-Italian one
    "Cantonese"). Letters with diacritics count towards not-English.
    """
    words = [w.lower() for w in _WORD.findall(text)]
    if not words:
        return None
    hits = {lang: sum(w in vocab for w in words) for lang, vocab in _STOPWORDS.items()}
    english = hits.pop("English")
    accents = len(_DIACRITICS.findall(text))
    other, other_hits = max(hits.items(), key=lambda kv: kv[1])
    other_hits += accents
    if english + other_hits < min_hits:
        return None
    if english >= other_hits:
        return "English"
    return other


def latin_run(text: str) -> int:
    """The most Latin-script words in a row in *text* (punctuation and spaces between them
    allowed, Chinese characters not)."""
    best = run = 0
    pos = None
    for match in LATIN_WORD.finditer(text):
        run = run + 1 if pos is not None and not _CJK.search(text[pos:match.start()]) else 1
        best = max(best, run)
        pos = match.end()
    return best


def foreign_text(text: str, native: Sequence[str], min_words: int = 4) -> Optional[str]:
    """The non-native language a chunk's Latin-script text is in, or None: what flags a
    chunk the run's own model transcribed (or half-transcribed) in another language."""
    if latin_run(text) < min_words:
        return None
    latin = " ".join(m.group(0) for m in _WORD.finditer(text))
    lang = latin_language(latin)
    return lang if lang and not is_native(lang, native) else None


def is_native(label: Optional[str], native: Sequence[str]) -> bool:
    """Whether a language label is one this run keeps as it is.

    Matching ignores case, and any label naming Chinese ("Chinese", "Chinese (Sichuanese)")
    counts as Chinese. No label is native: there is nothing to act on.
    """
    if not label:
        return True
    low = label.strip().lower()
    names = {n.strip().lower() for n in native}
    if low in names:
        return True
    return "chinese" in names and "chinese" in low


def heard_language(label: Optional[str], text: str) -> Optional[str]:
    """The multilingual model's language label, corrected by its own text where that is
    mostly Latin script and plainly not in the labelled language (an Italian chunk labelled
    "English", say). Text that is largely Chinese keeps its label: a piece that is half
    Cantonese is not relabelled by its other half. ``"None"`` -- no speech heard -- stays."""
    if label and label.lower() == "none":
        return label
    if cjk_share(text) < 0.25:
        guess = latin_language(text)
        if guess and guess != "English" and (not label or label.lower() in ("english", "chinese", "cantonese")):
            return guess
    return label


# --- Splitting ------------------------------------------------------------------------------

def _smooth(db: np.ndarray, width: int) -> np.ndarray:
    """A moving average of frame levels, its edges padded with the edge level: zero padding
    would read as 0 dB, the loudest a frame can be."""
    if width <= 1:
        return db
    left = width // 2
    padded = np.pad(db, (left, width - 1 - left), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def split_point(
    seg_start: float,
    audio: np.ndarray,
    speech: Optional[Sequence[Sequence[float]]] = None,
    *,
    sample_rate: int = 16000,
    min_piece: float = 4.0,
    min_gap: float = 0.25,
) -> Optional[float]:
    """Where to cut a stretch of audio in two: a time on the file's timeline, or None if it is
    too short.

    Prefers the longest gap of at least *min_gap* between VAD's own speech regions
    (``speech``, on the file's timeline); failing that, the middle of the longest quiet
    stretch in the audio; and never within *min_piece* seconds of either end.
    """
    duration = len(audio) / sample_rate
    if duration < 2 * min_piece:
        return None
    lo, hi = seg_start + min_piece, seg_start + duration - min_piece

    if speech:
        gaps = []
        regions = sorted((float(a), float(b)) for a, b in speech)
        for (_, prev_end), (next_start, _) in zip(regions, regions[1:]):
            mid = (prev_end + next_start) / 2
            if next_start - prev_end >= min_gap and lo <= mid <= hi:
                gaps.append((next_start - prev_end, mid))
        if gaps:
            return max(gaps)[1]

    from cantocaptions_ai.pipeline.align_checks import frame_dbfs
    hop = 0.04
    db = frame_dbfs(np.asarray(audio, dtype=np.float32), sample_rate, frame_seconds=hop)
    if len(db) < 3:
        return None
    width = max(1, int(round(0.2 / hop)))
    smooth = _smooth(db, width)
    first = int(np.ceil(min_piece / hop))
    last = min(len(smooth), int(np.floor((duration - min_piece) / hop)) + 1)
    if last <= first:
        return None
    window = smooth[first:last]
    quiet = window <= np.percentile(smooth, 10) + 6.0
    # The longest run of quiet frames inside the window; its middle is the cut.
    best_len, best_mid, run_start = 0, None, None
    for i, q in enumerate(np.append(quiet, False)):
        if q and run_start is None:
            run_start = i
        elif not q and run_start is not None:
            if i - run_start > best_len:
                best_len, best_mid = i - run_start, (run_start + i - 1) / 2
            run_start = None
    if best_mid is None:
        best_mid = float(np.argmin(window))
    return seg_start + (first + best_mid + 0.5) * hop


def voiced_span(start: float, audio: np.ndarray, sample_rate: int = 16000,
                hop: float = 0.04) -> Tuple[float, float]:
    """The part of a stretch of audio that has sound in it, on the file's timeline: from the
    first to the last frame within 35 dB of the loudest (and 6 dB over the quietest tenth).
    The whole stretch when nothing stands out."""
    from cantocaptions_ai.pipeline.align_checks import frame_dbfs

    end = start + len(audio) / sample_rate
    db = frame_dbfs(np.asarray(audio, dtype=np.float32), sample_rate, frame_seconds=hop)
    if len(db) < 3:
        return start, end
    width = max(1, int(round(0.2 / hop)))
    smooth = _smooth(db, width)
    loud = np.nonzero(smooth >= max(np.percentile(smooth, 10) + 6.0, smooth.max() - 35.0))[0]
    if not len(loud):
        return start, end
    return start + loud[0] * hop, min(end, start + (loud[-1] + 1) * hop)


# --- Repair ---------------------------------------------------------------------------------

FOREIGN_MODES = ("transcribe", "drop", "note", "none")


@dataclass(frozen=True)
class RepairSettings:
    """What ASR repair does; see the ``asr_*`` fields of PipelineConfig."""

    rule: LoopRule = LoopRule()
    native: Tuple[str, ...] = ("Cantonese", "Chinese", "English")
    foreign: str = "transcribe"
    split_depth: int = 3
    split_min_piece: float = 4.0
    # A chunk of at least this many seconds the model wrote nothing for is asked about --
    # only in a file where some chunk already shows another language, when
    # empty_needs_foreign. Elsewhere an empty chunk is music or effects: across 8
    # Cantonese-only episodes all 21 checked came back unchanged, each costing a load of the
    # multilingual model for nothing.
    min_empty: float = 2.0
    empty_needs_foreign: bool = True

    @property
    def check_language(self) -> bool:
        return self.foreign != "none"


@dataclass
class Piece:
    """One stretch of a repaired chunk, on the file's timeline, and what it ends up saying.

    ``text``/``looped`` are the run's own model's; ``alt_text``/``alt_language`` the
    multilingual model's, when it was asked (``alt_language`` as corrected by
    :func:`heard_language`). ``source`` says which text is kept ("asr", "foreign",
    "dropped"), ``note`` what to tell the reader of notes.srt.
    """

    start: float
    end: float
    text: str = ""
    looped: bool = False
    alt_text: Optional[str] = None
    alt_language: Optional[str] = None
    source: str = "asr"
    note: Optional[str] = None
    # The run's own text, when another took its place.
    replaced: Optional[str] = None
    # The multilingual model's reading when told the run's own language, asked only where
    # its own call conflicts with clean text from the run's model (see combine).
    hinted_text: Optional[str] = None

    def to_dict(self) -> dict:
        out = {"start": round(self.start, 3), "end": round(self.end, 3), "text": self.text,
               "source": self.source}
        for key in ("alt_text", "alt_language", "hinted_text", "note", "replaced"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.looped:
            out["looped"] = True
        return out


@dataclass
class ChunkRepair:
    """A first-pass chunk being repaired: where it is, why, and its pieces so far."""

    file: int
    index: int
    start: float
    end: float
    audio: np.ndarray
    reasons: List[str]
    speech: Optional[List[Tuple[float, float]]] = None
    pieces: List[Piece] = field(default_factory=list)
    split: bool = False
    splits: int = 0

    def audio_for(self, piece: Piece, sample_rate: int = 16000) -> np.ndarray:
        a = int(round((piece.start - self.start) * sample_rate))
        b = int(round((piece.end - self.start) * sample_rate))
        return self.audio[max(a, 0):max(b, a + 1)]

    def speech_in(self, piece: Piece) -> Optional[List[Tuple[float, float]]]:
        if not self.speech:
            return None
        return [(max(a, piece.start), min(b, piece.end)) for a, b in self.speech
                if b > piece.start and a < piece.end]


def chunk_reasons(seg: dict, settings: RepairSettings) -> List[str]:
    """Why a first-pass chunk is worth repairing (empty: it is not).

    ``loop:<why>`` (see :func:`loop_verdict`); and, when the language check is on,
    ``foreign_label:<Lang>`` (the model said it heard a non-native language -- only a model
    that still writes its language label, such as stock Qwen3-ASR, ever does),
    ``foreign_text:<Lang>`` (it wrote Latin-script text that is not English, or not another
    native language) and ``empty`` (it wrote nothing for a stretch of sound).
    """
    reasons = []
    why, _ = loop_verdict(seg, settings.rule)
    if why:
        reasons.append(f"loop:{why}")
    if settings.check_language:
        label = seg.get("asr_language")
        if label and label != "none" and not is_native(label, settings.native):
            reasons.append(f"foreign_label:{label}")
        text = seg.get("text") or ""
        lang = foreign_text(text, settings.native)
        if lang:
            reasons.append(f"foreign_text:{lang}")
        if not text.strip() and seg["end"] - seg["start"] >= settings.min_empty:
            reasons.append("empty")
    return reasons


def plan_chunks(files: Sequence[Tuple[dict, Sequence[dict]]],
                settings: RepairSettings) -> List[ChunkRepair]:
    """The chunks to repair across *files* (each ``(transcription result, vad_segments)``,
    index-aligned), each holding one piece -- the whole chunk, with its first-pass text.

    A chunk is to be split when it loops, or when its foreign text sits beside Chinese (it
    changes language part way, and the language check needs the pieces apart).
    """
    chunks = []
    for f, (result, vad_segments) in enumerate(files):
        planned = [chunk_reasons(seg, settings) for seg in result["segments"]]
        if settings.empty_needs_foreign and not any(
                r.startswith("foreign_") for reasons in planned for r in reasons):
            planned = [[r for r in reasons if r != "empty"] for reasons in planned]
        for i, (seg, vad, reasons) in enumerate(zip(result["segments"], vad_segments, planned)):
            if not reasons:
                continue
            text = seg.get("text") or ""
            looped = any(r.startswith("loop") for r in reasons)
            mixed = any(r.startswith("foreign_text") for r in reasons) and cjk_share(text) >= 0.25
            speech = vad.get("speech")
            chunks.append(ChunkRepair(
                file=f, index=i, start=seg["start"], end=seg["end"], audio=vad["audio"],
                reasons=reasons, speech=[tuple(s) for s in speech] if speech else None,
                pieces=[Piece(seg["start"], seg["end"], text, looped=looped)],
                split=settings.split_depth > 0 and (looped or mixed),
            ))
    return chunks


Transcribe = Callable[[List[np.ndarray]], list]


def _looped(text, rule: LoopRule) -> bool:
    seg = {"text": str(text), "retry_budget_hit": getattr(text, "retry_budget_hit", False),
           "raw_text": getattr(text, "raw", None)}
    return loop_verdict(seg, rule)[0] is not None


def run_ladder(chunks: Sequence[ChunkRepair], transcribe: Transcribe, settings: RepairSettings,
               normalize: Callable[[str], str] = lambda t: t, sample_rate: int = 16000) -> int:
    """Cut each chunk marked ``split`` and re-transcribe the pieces with the run's own model
    (*transcribe*: audio arrays -> AsrTexts, with its own loop retry). A piece that still
    loops, or still has foreign text beside Chinese, is cut again, up to
    ``settings.split_depth`` levels. Returns the number of pieces transcribed.
    """
    def again(piece: Piece) -> bool:
        return piece.looped or (cjk_share(piece.text) >= 0.25
                                and foreign_text(piece.text, settings.native) is not None)

    done = 0
    for depth in range(settings.split_depth):
        new: List[Tuple[ChunkRepair, Piece]] = []
        for chunk in chunks:
            if not chunk.split:
                continue
            pieces = []
            for piece in chunk.pieces:
                if depth == 0 or again(piece):
                    cut = split_point(piece.start, chunk.audio_for(piece, sample_rate),
                                      chunk.speech_in(piece), sample_rate=sample_rate,
                                      min_piece=settings.split_min_piece)
                    if cut is not None:
                        halves = [Piece(piece.start, cut), Piece(cut, piece.end)]
                        pieces.extend(halves)
                        new.extend((chunk, h) for h in halves)
                        chunk.splits += 1
                        continue
                pieces.append(piece)
            chunk.pieces = pieces
        if not new:
            break
        texts = transcribe([chunk.audio_for(piece, sample_rate) for chunk, piece in new])
        for (chunk, piece), text in zip(new, texts):
            piece.text = normalize(str(text))
            piece.looped = _looped(text, settings.rule)
        done += len(new)
    return done


def run_language_check(chunks: Sequence[ChunkRepair], transcribe: Transcribe,
                       native: Sequence[str], transcribe_hinted: Optional[Transcribe] = None,
                       sample_rate: int = 16000) -> int:
    """Ask the multilingual model what every piece of every chunk says, and in what language
    (*transcribe*: audio arrays -> AsrTexts, the language left to the model).

    Where that reading would conflict with clean text from the run's own model -- another
    language, written in another script, over text that does not loop -- the piece is asked
    again with the run's language as a hint (*transcribe_hinted*). Told to expect
    Cantonese, the model still writes pure Italian as Italian (or nothing), but writes any
    Cantonese it hears as Cantonese; :func:`combine` reads which. Returns pieces asked.
    """
    units = [(chunk, piece) for chunk in chunks for piece in chunk.pieces]
    if not units:
        return 0
    texts = transcribe([chunk.audio_for(piece, sample_rate) for chunk, piece in units])
    for (chunk, piece), text in zip(units, texts):
        alt = strip_broken_chars(str(text)).strip()
        piece.alt_text = alt
        piece.alt_language = heard_language(getattr(text, "language", None), alt)
    asked = len(units)
    if transcribe_hinted is not None:
        doubtful = [(chunk, piece) for chunk, piece in units if _conflicts(piece, native)]
        if doubtful:
            hinted = transcribe_hinted([chunk.audio_for(p, sample_rate) for chunk, p in doubtful])
            for (_, piece), text in zip(doubtful, hinted):
                piece.hinted_text = strip_broken_chars(str(text)).strip()
            asked += len(doubtful)
    return asked


def _trivial(text: Optional[str]) -> bool:
    return not text or not text.strip(LOOP_SEPARATORS + " ")


def _conflicts(piece: Piece, native: Sequence[str]) -> bool:
    """The multilingual model heard another language, in another script, where the run's own
    model wrote clean text that is not itself in another language."""
    label, alt = piece.alt_language, piece.alt_text
    if not label or label.lower() == "none" or _trivial(alt) or is_native(label, native):
        return False
    if cjk_share(alt) >= 0.25:
        return False
    return not (_trivial(piece.text) or piece.looped or _own_foreign(piece.text) is not None)


def _own_foreign(text: str) -> Optional[str]:
    """The language of a piece's own text when it is Latin-script and not English, else None."""
    if _trivial(text) or cjk_share(text) >= 0.25:
        return None
    lang = latin_language(text)
    return lang if lang and lang != "English" else None


def combine(piece: Piece, *, mode: str, native: Sequence[str]) -> Piece:
    """Decide what a piece says, given the multilingual model's reading of it (in place).

    ``mode`` is ``asr_foreign``. A piece heard as a non-native language takes that model's
    text under ``transcribe`` (unless the run's own model already wrote it in that
    language), says nothing under ``drop``, and keeps the run's own text with a note under
    ``note``. Only when the evidence agrees, though:

    - the multilingual model's own text must be in another script (a reading that is
      largely Chinese is a piece that changes language part way: noted, never replaced);
    - and the run's own text for the piece must be suspect -- empty, still looping, or itself
      in that other language -- or the multilingual model, told to expect the run's
      language, must still have heard none of it (``hinted_text``). Otherwise the run's own
      text is kept, with a note of the disagreement: given audio that is part Cantonese, the
      multilingual model has been seen to write the Cantonese out as an Italian
      *translation*.

    A native reading never replaces or drops anything: a piece heard as English while the
    run's own model wrote Chinese is only noted, as is a looping piece the multilingual
    model heard no speech in.
    """
    label, alt = piece.alt_language, piece.alt_text
    if label is None or alt is None:
        return piece
    if label.lower() == "none" or _trivial(alt):
        if piece.looped and not _trivial(piece.text):
            piece.note = "asr_unheard:a second model heard no speech"
        return piece
    if is_native(label, native):
        if label.lower() == "english" and cjk_share(piece.text) >= 0.5:
            piece.note = "asr_lang_mismatch:English"
        return piece
    if cjk_share(alt) >= 0.25:
        piece.note = f"asr_lang_mixed:{label}"
        return piece
    own = _own_foreign(piece.text)
    # Text the run's own model already wrote in the other language names it better than a
    # whole-piece label does (Italian has been labelled Portuguese).
    label = own or label
    suspect = _trivial(piece.text) or piece.looped or own is not None
    if not suspect and (piece.hinted_text is None or cjk_share(piece.hinted_text) >= 0.25):
        piece.note = f"asr_lang_conflict:{label}"
        return piece
    if mode == "transcribe":
        if own is None:
            piece.replaced, piece.text, piece.source = piece.text, alt, "foreign"
        piece.note = f"asr_lang:{label}"
    elif mode == "drop":
        piece.replaced, piece.text, piece.source = piece.text, "", "dropped"
        piece.note = f"asr_lang_dropped:{label}"
    else:
        piece.note = f"asr_lang:{label}"
    return piece


_FIRST_PASS_FLAGS = ("budget_hit", "retried", "retry_budget_hit", "raw_text")


def finish(chunks: Sequence[ChunkRepair], results: Sequence[dict], settings: RepairSettings,
           join: Callable[[List[str]], str]) -> List[Tuple[int, float, str]]:
    """Write each repaired chunk back into its file's result (in place) and say what happened.

    A chunk's text becomes its pieces' texts joined (*join*: ``asr.join_clauses`` bound to
    the run's punctuation and script). The segment keeps its start and end -- one segment
    per VAD chunk, which later stages index by -- and gains ``repair`` (the reasons),
    ``pieces`` and ``notes_at``. Once a chunk has been re-transcribed, the first pass's loop
    flags describe text that is gone; they are dropped, and ``asr_looped`` set if a piece
    kept from the run's own model still loops. Returns ``(file, time, description)`` rows.
    """
    rows = []
    for chunk in chunks:
        for piece in chunk.pieces:
            combine(piece, mode=settings.foreign, native=settings.native)
        seg = dict(results[chunk.file]["segments"][chunk.index])
        retranscribed = chunk.splits > 0 or any(p.source != "asr" for p in chunk.pieces)
        if retranscribed:
            for key in _FIRST_PASS_FLAGS:
                seg.pop(key, None)
            seg["text"] = join([p.text for p in chunk.pieces if not _trivial(p.text)])
            if any(p.looped and p.source == "asr" for p in chunk.pieces):
                seg["asr_looped"] = True
        seg["repair"] = ";".join(chunk.reasons)
        seg["pieces"] = [p.to_dict() for p in chunk.pieces]
        notes = [(round(p.start, 3), round(p.end, 3), p.note) for p in chunk.pieces if p.note]
        if notes:
            seg["notes_at"] = list(seg.get("notes_at") or []) + notes
        spans = _foreign_spans(chunk, seg["text"], join)
        if spans:
            seg["foreign_spans"] = spans
        results[chunk.file]["segments"][chunk.index] = seg
        rows.append((chunk.file, chunk.start, describe(chunk)))
    return rows


def _foreign_spans(chunk: ChunkRepair, text: str, join, sample_rate: int = 16000) -> List[dict]:
    """The stretches of a chunk's final text that are in another language, each with the
    time its speech takes up (:func:`voiced_span`): what :func:`spread_foreign` times."""
    spans, cursor = [], 0
    for piece in chunk.pieces:
        if _trivial(piece.text) or not (piece.note or "").startswith("asr_lang:"):
            continue
        for written in (join([piece.text]), piece.text.strip()):
            at = text.find(written, cursor) if written else -1
            if at >= 0:
                t0, t1 = voiced_span(piece.start, chunk.audio_for(piece, sample_rate), sample_rate)
                spans.append({"text": written, "start": round(t0, 3), "end": round(t1, 3)})
                cursor = at + len(written)
                break
    return spans


def describe(chunk: ChunkRepair) -> str:
    """One line for the stage's INFO record: what was wrong and what was done."""
    what = []
    if any(r.startswith("loop") for r in chunk.reasons):
        what.append("repetition loop")
    for r in chunk.reasons:
        if r.startswith("foreign_text:"):
            what.append(f"{r.split(':', 1)[1]} text")
        elif r.startswith("foreign_label:"):
            what.append(f"heard as {r.split(':', 1)[1]}")
    if "empty" in chunk.reasons:
        what.append("no text")
    done = []
    if chunk.splits:
        done.append(f"split into {len(chunk.pieces)}")
    said = {"asr_lang": "{} speech kept as {}", "asr_lang_dropped": "{} speech dropped",
            "asr_lang_conflict": "a second model heard {}; the text was kept",
            "asr_lang_mixed": "part {}, kept as it was"}
    for kind, template in said.items():
        langs = sorted({p.note.split(":", 1)[1] for p in chunk.pieces
                        if p.note and p.note.split(":", 1)[0] == kind})
        done.extend(template.format(lang, lang) for lang in langs)
    if any(p.looped and p.source == "asr" for p in chunk.pieces):
        done.append("still loops")
    if not done:
        done.append("unchanged")
    return f"{', '.join(what)}: {'; '.join(done)}"


# --- Sanitizing -----------------------------------------------------------------------------

def sanitize_segment(seg: dict, rule: LoopRule = LoopRule(), *,
                     trim: bool = True) -> Tuple[dict, Optional[Loop]]:
    """A segment's text with broken characters removed and, if *trim*, a visible loop written
    once; plus notes for notes.srt. Returns (segment, the loop it trimmed or None).

    Only a loop the text itself shows (:meth:`LoopRule.visible`) is trimmed. A chunk flagged
    by how generation ended -- the budget ran out, the processor folded a loop away -- but
    whose text reads normally is left alone and only noted: those are where a recovered
    loop's text is most worth a look.
    """
    text = seg.get("text") or ""
    why, _ = loop_verdict(seg, rule)
    clean = strip_broken_chars(text)
    loop = find_loop(clean)
    out = dict(seg)
    notes = list(out.get("notes_at") or [])
    trimmed = None
    if trim and rule.visible(loop):
        clean = trim_loop(clean, loop)
        trimmed = loop
        end = out["end"]
        notes.append((max(out["start"], end - 1.0), end,
                      f"asr_loop:written once, {loop.copies}x {loop.phrase}"))
    elif why or seg.get("budget_hit"):
        start = out["start"]
        notes.append((start, min(out["end"], start + 0.5), f"asr_loop:{why or 'recovered by a retry'}"))
    out["text"] = clean
    if notes:
        out["notes_at"] = notes
    return out, trimmed


# --- Timing other-language text ---------------------------------------------------------------

# A clause of other-language text ends at one of these.
_SENTENCE_END = re.compile(r"[.!?。！？…]")
_CLAUSE_END = re.compile(r"[,;:，；：]")


def spread_foreign(segment: dict, pieces: List[dict], line_width: float = 17.5,
                   width: Optional[Callable[[str], float]] = None) -> List[dict]:
    """Retime the other-language words in one aligned chunk, and cut them into clauses.

    The align model for the run's language cannot follow speech in another language (the
    Cantonese one has no tokens for é, ò, ù, and no ear for Italian): the times it gives
    those words are guesses, and the gaps between them invented. Cue cutters that trust those
    gaps cut ``Si | può``. So each ``foreign_spans`` stretch's words are spread evenly by
    character over the time its speech takes up, and the cues they are in are cut at each
    sentence end (``.`` ``?`` ``!``), and at a comma only where a sentence is wider than one
    line (*line_width*, measured by *width*): even timing shows no pauses, so cue assembly's
    own cutters would otherwise cut such a stretch anywhere. Words are matched to the
    stretch by their place in the chunk's text; if they cannot be, *pieces* come back as
    they were.

    TODO: a forced aligner that speaks the language (Qwen3-ForcedAligner-0.6B covers Italian,
    French, German, Spanish, Portuguese, Russian, Japanese, Korean) would give real word
    times. See docs/asr-repair.md.
    """
    spans = segment.get("foreign_spans")
    text = str(segment.get("text") or "")
    if not spans or not pieces or not text:
        return pieces
    ranges, cursor = [], 0
    for span in spans:
        at = text.find(span["text"], cursor)
        if at < 0:
            continue
        ranges.append((at, at + len(span["text"]), float(span["start"]), float(span["end"])))
        cursor = at + len(span["text"])
    if not ranges:
        return pieces

    # Every word's place in the chunk's text, across the pieces alignment cut it into.
    places, cursor = [], 0
    for p, piece in enumerate(pieces):
        for w, word in enumerate(piece.get("words") or []):
            token = str(word.get("word", ""))
            if not token.strip():
                continue
            at = text.find(token, cursor)
            if at < 0:
                return pieces
            places.append((p, w, at, at + len(token)))
            cursor = at + len(token)

    spread = set()
    for a, b, t0, t1 in ranges:
        inside = [pl for pl in places if pl[2] >= a and pl[3] <= b]
        total = sum(pl[3] - pl[2] for pl in inside)
        if not total or t1 <= t0:
            continue
        done = 0
        for p, w, x, y in inside:
            word = pieces[p]["words"][w]
            word["start"] = round(t0 + (t1 - t0) * done / total, 3)
            done += y - x
            word["end"] = round(t0 + (t1 - t0) * done / total, 3)
            word.pop("peak", None)  # its peak was measured where the word is not
            spread.add((p, w))
    if not spread:
        return pieces

    out = []
    for p, piece in enumerate(pieces):
        if not any(q == p for q, _ in spread):
            out.append(piece)
            continue
        out.extend(_clauses(piece, {w for q, w in spread if q == p}, line_width,
                            width or (lambda t: len(t) / 2)))
    return out


def _clauses(piece: dict, spread: set, line_width: float,
             width: Callable[[str], float]) -> List[dict]:
    """*piece* cut before each spread word that a sentence end separates from the spread word
    before it -- or a comma, inside a sentence too wide for a line -- each part re-edged. In
    unspaced text a mark (``?``) is a word of its own, so the mark is looked for in
    everything between the two spoken words, words included."""
    from cantocaptions_ai.pipeline.align_checks import _word_offsets

    text = str(piece.get("text", ""))
    words = list(piece.get("words") or [])
    offsets = _word_offsets(text, words)
    if offsets is None:
        return [_edged(piece)]
    spoken = [w for w, word in enumerate(words) if LATIN_WORD.search(str(word.get("word", "")))
              or _CJK.search(str(word.get("word", "")))]
    pairs = [(a, b) for a, b in zip(spoken, spoken[1:]) if a in spread and b in spread]
    ends = [b for a, b in pairs if _SENTENCE_END.search(text[offsets[a][1]:offsets[b][0]])]
    commas = [b for a, b in pairs if _CLAUSE_END.search(text[offsets[a][1]:offsets[b][0]])]
    cuts = []
    for lo, hi in zip([0] + ends, ends + [len(words)]):
        cuts += _fill_lines(text, offsets, lo, hi, [c for c in commas if lo < c < hi],
                            line_width, width)
        if hi < len(words):
            cuts.append(hi)
    if not cuts:
        return [_edged(piece)]
    chars = piece.get("chars")
    chars = list(chars) if chars is not None and len(chars) == len(text) else None
    out, w_from, t_from = [], 0, 0
    for w_to in cuts + [len(words)]:
        t_to = offsets[w_to][0] if w_to < len(words) else len(text)
        part = dict(piece)
        part["text"] = text[t_from:t_to].rstrip()
        part["words"] = words[w_from:w_to]
        if chars is not None:
            part["chars"] = chars[t_from:t_to]
        part["notes"] = list(piece.get("notes") or [])
        out.append(_edged(part))
        w_from, t_from = w_to, t_to
    return [part for part in out if part["text"].strip()]


def _fill_lines(text: str, offsets, lo: int, hi: int, commas: List[int], line_width: float,
                width: Callable[[str], float]) -> List[int]:
    """Commas to cut a sentence (words lo..hi) at, filling each piece up to a line; none if
    the sentence fits on one."""
    def at(w):
        return offsets[w][0] if w < len(offsets) else len(text)

    if width(text[at(lo):at(hi)].strip()) <= line_width:
        return []
    cuts, start = [], lo
    for k, comma in enumerate(commas):
        following = commas[k + 1] if k + 1 < len(commas) else hi
        if width(text[at(start):at(following)].strip()) > line_width:
            cuts.append(comma)
            start = comma
    return cuts


def _edged(piece: dict) -> dict:
    """A cue's start and end from its timed words, released from its last word."""
    timed = [w for w in piece.get("words") or [] if w.get("start") is not None]
    if not timed:
        return piece
    piece["start"] = round(min(float(w["start"]) for w in timed), 3)
    piece["end"] = round(max(float(w["end"]) for w in timed), 3)
    piece["release_from"] = piece["end"]
    return piece

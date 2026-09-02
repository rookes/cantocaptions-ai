"""Forced alignment of an *untimed* transcript onto a recording (``--realign``).

The input is a complete transcript with no timings at all -- one cue per line, the shape a
human transcriptionist or a scraped film script produces. The job is to put every line on the
audio timeline so the ordinary alignment, cleaning and subtitle-writing stages can finish it.

This is not what ``--retime`` does. That feature searches for each cue *around its existing
timestamp* (see ``retime.py``), which is exactly the information a plain transcript lacks.

Three ideas carry the whole module:

1. **The transcript's line breaks are the cue boundaries.** ``split_chars`` exists because
   Qwen returns an undifferentiated block per segment that has to be cut into cues; a
   transcript arrives pre-cut. So the line structure is declared to alignment through
   ``SingleSegment['cue_spans']`` and punctuation is demoted to what it acoustically is --
   a pause. Deriving cues from punctuation here would be actively wrong: real transcripts
   are punctuated far too sparsely (10% of lines in the first one this was built for), so
   the cue count would come out both wrong and unstable.

2. **Segmentation must not discard audio.** Ordinary VAD keeps speech and drops the rest,
   but we hold a transcript line for every utterance in the file *including* the ones VAD
   scores below threshold -- sung, shouted, buried under music. Dropping that audio leaves
   the line nothing to align against. ``--realign`` therefore runs VAD in split-only mode
   (``Vad.cover_chunks``): VAD chooses where to cut, never what to keep.

3. **Where each line sits is found with a sliding free-end Viterbi.** A whole film cannot go
   in one trellis, and the line-to-chunk mapping is unknown up front -- the chicken-and-egg
   this module exists to break. A window of audio is aligned against a deliberate
   over-supply of the next unplaced lines, with the path allowed to *stop early*
   (``backtrack(free_end=True)``); however many lines it consumed is the answer for that
   window. Only the lines finishing comfortably inside the window are committed, and the
   next window restarts from there, so a window that goes wrong costs one window rather
   than the rest of the film.

The alternative anchor, ``assign_lines_via_asr``, runs the normal ASR stage and matches the
two character streams instead. It is slower but degrades better: a text match can leave a
line *unmatched*, whereas forced alignment must consume every token it is given and will
smear a transcript containing lines the recording does not actually have.
"""

from __future__ import annotations

import bisect
import difflib
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from cantocaptions_ai.cantonese.text import DEFAULT_PUNCTUATION, PunctuationConfig
from cantocaptions_ai.pipeline.alignment import (
    EmissionTimeline,
    _get_blank_id,
    _preprocess_segment,
    backtrack,
    get_trellis,
)
from cantocaptions_ai.utils.audio import SAMPLE_RATE
from cantocaptions_ai.utils.log_utils import get_logger
from cantocaptions_ai.utils.schema import SingleSegment, VadAudioSegment

logger = get_logger(__name__)


# A Unicode private-use character, so it cannot collide with anything a transcript may
# legitimately contain. One is appended to every line before the lines of a chunk are joined
# for alignment, which buys two things: the aligner gets an explicit pause token to park the
# inter-line silence on, and release_from (alignment.py) extends each cue's end into that
# pause. It is removed again by strip_sentinels() before the text reaches the cleaner.
REALIGN_SENTINEL = ""

# Punctuation as --realign wants it. Widening split_chars does *not* create cues here (the
# caller declares those via cue_spans); it only decides which characters become pause tokens
# rather than being dropped outright. The space earns its place because a transcript uses one
# where a comma would go, and the aligner should spend silence there rather than running the
# two clauses together.
REALIGN_PUNCTUATION = PunctuationConfig(
    split_chars=tuple(DEFAULT_PUNCTUATION.split_chars) + (" ", REALIGN_SENTINEL),
    mergeable_chars=DEFAULT_PUNCTUATION.mergeable_chars,
)

# Halfwidth marks are neither in the align vocabulary nor in split_chars, so left alone they
# are dropped outright and the pause they represent goes unmodelled. Only a mark that follows
# CJK is converted: that is a transcriber typing the wrong key after Chinese text, whereas the
# comma in "Thank you, sir." belongs to an English clause and must stay as written, since this
# text is the subtitle. It also keeps "1.5" and "3,000" intact for free.
_FULLWIDTH_PUNCT = {",": "，", ".": "。", "?": "？", "!": "！", ";": "；"}

# Transcripts in the wild use U+22EF (midline) or a doubled U+2026 for a trailing-off pause;
# only the single U+2026 is a split char.
_ELLIPSIS_FORMS = ("⋯⋯", "⋯", "……")

# Ceiling on how fast the transcript could possibly be spoken, used only to decide how many
# lines to *offer* a window. Over-supplying costs a few tokens; under-supplying would cap
# what the window can consume and stall progress, so this is deliberately generous.
MAX_CHAR_RATE = 12.0

# Fallback speaking rate, used to invent a span for a line the aligner could not place.
NOMINAL_CHAR_RATE = 5.0

# The slowest a line could conceivably be spoken, in characters per second. Anything slower
# is not slow speech, it is a failed alignment, and the span it claims must not be believed.
#
# This is the guard that keeps a bad window local. Without it the search has no incentive to
# consume tokens at all: the trellis maximises total score, and on audio the model cannot
# read (singing, music, effects) dwelling on blanks scores better than advancing through
# poorly-matching characters. A window then spends its whole 120 s on one line, the pointer
# advances by the whole window, and a few of those in a row walk off the end of the file --
# on the Doraemon fixture, five consecutive windows moved the pointer 527 s and stranded the
# remaining 295 lines at the end of the audio. Lines 0-185 before it were placed to within
# 0.2 s, so the failure is abrupt and local, and bounding it is enough to survive it.
MIN_CHAR_RATE = 1.0

# ...but a very short line can legitimately be drawn out, so the bound never goes below this.
MIN_LINE_SPAN = 3.0

# Ceiling on the characters an alignment chunk may carry, per second of its audio.
#
# CTC needs at least one emission frame per token, and the align model emits ~25 fps, so a 30 s
# chunk can hold a few hundred characters and no more. Grouping on time alone is therefore not
# enough: when the search strands a run of lines on one timestamp, they all fall inside a
# single chunk, the trellis has far more tokens than frames, backtrack fails, and
# _align_segment falls back to *one* cue carrying the lot. On the Police Story 2 run that was
# 1015 lines and 6492 characters in a single subtitle.
#
# Half the frame rate leaves room for the blanks a real alignment spends between characters.
MAX_CHARS_PER_SECOND = 12.0

# Distinct timestamps for lines the search could not separate, so a stranded run still comes
# out as an ordered sequence of cues rather than one.
MIN_LINE_STEP = 0.04

# The shortest cue that still exists as a subtitle. Two frames at the align model's ~25 fps.
#
# A zero-length cue does not merely flash, it never displays, and strict SRT parsers reject
# the whole file over it (suber trims an epsilon off the end and asserts the duration is
# non-negative). They arise because align()'s release/trim pass sets a cue's end from the
# *next* cue's start, so two cues beginning within align_padding of each other leave the first
# with nothing; segmentation's duration floor then finds no room to grow into and correctly
# declines to overlap the neighbour. That non-overlap guarantee is right for ASR output and is
# not weakened here -- realign is what produced coincident cue starts, so realign repairs it.
MIN_VISIBLE_DURATION = 0.08

# The shortest alignment chunk that can be fed to the encoder at all.
#
# The feature extractor needs one whole analysis window (400 samples at 16 kHz) and raises
# "negative dimensions are not allowed" from inside numpy rather than saying so. A chunk that
# short is reachable whenever the search ran out of audio and left the remaining lines on the
# final frame: the group's bounds then collapse onto file_end and _slice_audio returns nothing.
# On the Doraemon fixture, where the acoustic anchor genuinely does run out of film, that
# aborted the whole run in the alignment stage.
MIN_CHUNK_DURATION = 0.1


# A transcript line is shown as a single cue, so a character sitting seconds away from the
# rest of its line is not a long pause -- it is a misplacement, and the cue must not be
# stretched to cover it. CTC is peaky, and a leading interjection (噢, 嗚, 唉) is a long open
# vowel that scores well against almost any voiced frame, so the search will happily park it
# in earlier audio and drag the cue's start back with it.
#
# Measured on test/bluey: 7 interjection-initial lines drifted -2.62s on average (worst
# -8.81s) while the other 125 averaged +0.07s (worst -0.35s). The confidence score does not
# see this at all -- those misplacements scored 0.96-0.99, higher than the median -- because
# the model really is confident, just about the wrong frame. So the detached token has to be
# discarded on the geometry, which is what this threshold does.
MAX_INTERNAL_GAP = 1.0

# ...but only a *short* detached run is discarded, and only from the ends of a line.
#
# "Whichever run is longest wins" was the first version of this rule and it over-reaches. The
# failure it was built for is a single unmodellable token drifting -- a line-initial
# interjection is a long open vowel, so it scores well against almost any voiced frame -- but
# the same test also throws away a correctly-aligned clause whenever a line has a real pause
# in the middle of it. Police Story 2 cue 18, 「無錯喇，我哋係唔需要身手好嘅警察」: the emission
# puts 無 at 296.120 s (hand-measured: 296.114) and the comma after 喇 then dwells 1.400 s
# before 我 at 297.880. Longest-run kept the twelve-character clause and dropped the
# three-character opening, putting the subtitle on screen 1.77 s late.
#
# Trimming only runs of at most this many characters, and only at the edges, keeps that
# opening while still dropping the one-character interjection that motivated the guard.
MAX_DETACHED_RUN = 2

# Below this, a cue's characters did not align -- they were placed because CTC must place
# every token, not because the audio supports them. Police Story 2 cue 33, 「由今日開始」,
# whose speech ends 0.2 s *before* the chunk it was grouped into begins, scored exactly 0.000.
MIN_CUE_SCORE = 0.05

# --- Anchor search -------------------------------------------------------------------
#
# All measured on the three fixtures with real emissions (scripts/bench_realign_placement.py).

# A window that explains nothing at the current pointer is retried this far down the
# transcript before the search gives up on it and advances the audio instead. This is what
# lets a search that has lost its place re-acquire; the old sweep could only go forward one
# line at a time and so stayed lost for the rest of the file.
ANCHOR_OFFSETS = (0, 3, 8, 20, 50, 120, 300)

# Mean path score a window must reach before anything in it is believed at all.
ACQUIRE_MIN_SCORE = 0.45

# ...and what a single line must reach, and hold in characters, to become an anchor. Anchors
# are the load-bearing part: a wrong one misplaces everything filled around it, while a
# missing one costs nothing because the fill covers it. So this is deliberately strict.
ANCHOR_MIN_SCORE = 0.55
ANCHOR_MIN_CHARS = 4

# Consecutive windows overlap by this much, so a line near a window's far edge gets a second,
# better-centred look rather than being judged from the edge.
ANCHOR_WINDOW_OVERLAP = 20.0

# Anchor review: forced-align blocks of this many consecutive anchors, stepping by the stride,
# and drop an interior anchor that moves further than the tolerance. See verify_anchors.
VERIFY_BLOCK = 7
VERIFY_STRIDE = 3
VERIFY_TOLERANCE = 2.0

# Trellis cells (frames x tokens) a single alignment may allocate. At float32 this is 240 MB.
# On all three fixtures the largest fill needs under 1M, so the recursive split below it is a
# safety net for a long unanchored stretch rather than a hot path.
CELL_BUDGET = 60_000_000
MAX_FILL_DEPTH = 24

# How much more room than its text could possibly occupy a line's neighbours may leave before
# the line is called unconstrained.
#
# _plausible_span is already the *most* generous reading of a line (MIN_CHAR_RATE, one
# character per second), so a bracket holding more audio than that contains time the
# transcript does not account for, and forced alignment inside it is free to put the lines
# almost anywhere. That is exactly Police Story 2's ``May呀``: two characters bracketed by
# anchors 66 s apart, placed 48 s from the truth.
#
# This replaces the confidence score as the per-cue trust signal, which was measured and does
# not work: ``May呀`` scored 0.733, above the median, while 「唔該」 scored 0.001 and is right
# to within 0.2 s. --realign_min_score stays what its docstring says it is -- a
# transcript-versus-recording mismatch detector -- and is logged, not attached to cues.
UNCONSTRAINED_ROOM = 1.5


@dataclass
class TranscriptLine:
    """One cue of the source transcript, before it has a time."""
    index: int
    text: str


# Why a line's timing is not to be trusted.
#
# One bit ("placed") used to carry all of these, which is not usable on a feature-length
# transcript: "the recording ends 1015 lines early" and "this interjection has no character
# the align model can spell" were the same bit, and they call for completely different
# responses. Every cue carrying a reason is written to realign/suspect.srt so it can be
# watched against the film, and the run summary breaks the count down by reason.
REASON_NO_AUDIO = "no_audio"            # the recording ran out before this line
REASON_UNREADABLE = "unreadable"        # audio is there, the align model could not read it
REASON_NO_VOCABULARY = "no_vocabulary"  # not one character of this line is in the vocabulary
REASON_ISOLATED = "isolated"            # placed by its neighbours; no support of its own
REASON_LOW_SCORE = "low_score"          # aligned, but weakly
REASON_IMPLAUSIBLE = "implausible"      # final cue is longer than its text could be spoken in
REASON_SILENT_START = "silent_start"    # final cue starts on silence

REASON_HELP = {
    REASON_NO_AUDIO: "the recording ran out before this line",
    REASON_UNREADABLE: "the align model could not read the audio here",
    REASON_NO_VOCABULARY: "the align model has no character for this line",
    REASON_ISOLATED: "no other dialogue nearby to anchor against",
    REASON_LOW_SCORE: "weak acoustic support",
    REASON_IMPLAUSIBLE: "cue is longer than its text could be spoken in",
    REASON_SILENT_START: "cue starts on silence",
}

# Counted apart in the summary. A line the model has no characters for is a property of the
# transcript and the vocabulary, not a sign that anything went wrong, and on a Cantonese film
# it is by far the most common reason -- 3.76% of the Police Story 2 transcript is out of
# vocabulary. Mixed in with the rest it would drown the reasons that mean something.
BENIGN_REASONS = (REASON_NO_VOCABULARY,)


@dataclass
class LineTiming:
    """Where a transcript line was found in the audio."""
    index: int
    start: float
    end: float
    score: float = float("nan")
    reason: Optional[str] = None

    @property
    def placed(self) -> bool:
        """True when the span came from the audio rather than from the lines around it."""
        return self.reason is None


# --- Loading ------------------------------------------------------------------------

def _follows_cjk(text: str, i: int) -> bool:
    """True when text[i] is a CJK ideograph or a fullwidth mark."""
    if not 0 <= i < len(text):
        return False
    ch = text[i]
    return "㐀" <= ch <= "鿿" or "　" <= ch <= "〿" or "＀" <= ch <= "￯"


def normalize_transcript_text(text: str) -> str:
    """Fold a transcript line into the punctuation the aligner can actually use.

    Only punctuation and whitespace are touched -- never the words -- because this text *is*
    the subtitle. Anything a rule file should decide (particle conventions, character
    variants) is left to the cleaner, which runs after alignment.
    """
    for form in _ELLIPSIS_FORMS:
        text = text.replace(form, "…")
    text = text.replace(REALIGN_SENTINEL, "")
    out = []
    for i, ch in enumerate(text):
        repl = _FULLWIDTH_PUNCT.get(ch)
        if repl and _follows_cjk(text, i - 1):
            out.append(repl)
        else:
            out.append(ch)
    # A transcript uses runs of spaces as loose phrasing; one pause token is enough.
    return " ".join("".join(out).split())


def load_transcript_lines(path: str) -> List[TranscriptLine]:
    """Read a line-delimited transcript (or an SRT, discarding its timings).

    Handles a UTF-8 BOM and any mix of line endings. Blank lines are separators, not cues.
    """
    if path.lower().endswith((".srt", ".vtt")):
        from cantocaptions_ai.pipeline.retime import load_subtitle_file
        raw = [seg["text"] for seg in load_subtitle_file(path)]
    else:
        with open(path, "rb") as fh:
            raw = fh.read().decode("utf-8-sig").splitlines()

    lines: List[TranscriptLine] = []
    for text in raw:
        text = normalize_transcript_text(text.strip())
        if text:
            lines.append(TranscriptLine(index=len(lines), text=text))
    return lines


# --- Tokenization -------------------------------------------------------------------

def line_tokens(
    text: str, model_lang: str, model_dictionary: dict, blank_id: int,
    punctuation: PunctuationConfig = REALIGN_PUNCTUATION,
) -> List[int]:
    """Token ids for one line, matching exactly what _align_segment would build.

    Going through _preprocess_segment rather than reimplementing the filter keeps the coarse
    search and the final alignment agreeing on which characters are alignable. If they
    disagreed, the search would place a line using evidence the alignment then ignores.
    """
    seg_data = _preprocess_segment(text, model_lang, model_dictionary, punctuation)
    split_chars = punctuation.split_chars
    return [
        model_dictionary[c] if c not in split_chars else blank_id
        for c in seg_data["clean_char"]
    ]


# --- Emissions ----------------------------------------------------------------------

def _plausible_span(text: str) -> float:
    """The longest a line of *text* could credibly take to say. See MIN_CHAR_RATE."""
    return max(MIN_LINE_SPAN, len(text) / MIN_CHAR_RATE)


def _bounded_timing(
    line: TranscriptLine, start: float, end: float, floor: float,
    score: float = float("nan"),
) -> LineTiming:
    """Build a LineTiming whose span cannot exceed what its text could be spoken in.

    Every placement goes through here, including the fallback taken when a window commits
    nothing in its safe zone -- that path is exactly where an unreadable stretch of audio
    ends up, so leaving it unbounded (as the first version did) let a single line claim a
    whole 120 s window and defeated the guard entirely.

    A line over the limit falls back to the *nominal* speaking rate rather than to the limit
    itself, so the pointer advances at roughly the rate the text implies rather than five
    times it, and the search can resynchronise once readable speech resumes. See
    MIN_CHAR_RATE.

    **Hanging the invented span on the *end* instead was tried and measured, and is worse.**
    The theory was that the end is the edge the evidence supports -- a line whose audio sits
    later in the window than the search first put it dwells on blanks and then fires its
    characters when the speech arrives, so the last frame used is near the truth. That is
    exactly true of one case (Police Story 2's ``May呀``, alone in a 64 s stretch with no
    dialogue: the end lands within 0.3 s of the truth and the start is 58 s early). It is
    false of the general smear, where the line belongs at the *start* of the window and the
    dwell runs forward, and that case is far more common. Backing out to the end there
    reports every stranded line roughly a window late: on the Doraemon fixture p90 start
    error went 0.209 s -> 425.7 s, and bluey's worst cue 1.07 s -> 10.71 s. Both regressions
    reproduce with this change alone and with no other. Do not reintroduce it.
    """
    start = max(start, floor)
    smeared = end - start > _plausible_span(line.text)
    if smeared:
        end = start + max(0.4, len(line.text) / NOMINAL_CHAR_RATE)
    if end <= start:
        end = start + 0.04
    return LineTiming(line.index, round(start, 3), round(end, 3), score,
                      reason=REASON_UNREADABLE if smeared else None)

# A bound on the forward *jump* between consecutive lines was tried here too, on the theory
# that a complete transcript cannot leave two minutes unaccounted for. It made the Doraemon
# fixture far worse -- median error went from 0.05 s to 127 s -- because it cascades: a film
# does contain genuinely wordless stretches, and clamping the first legitimate one leaves the
# pointer lagging, which makes the next correctly-placed line look stranded too, until the
# whole transcript is running ahead of the audio. Do not reintroduce it. Over-advancing is
# recoverable only by letting the search look backwards, which it deliberately cannot do.


def _trim_detached_edges(runs: Sequence[Sequence]) -> List:
    """Flatten *runs*, dropping short ones from either end.

    A run in the middle is never dropped however short it is: whatever it is, the characters
    on both sides of it place it, so it is part of the line. Only an edge run is unanchored,
    and it goes only if it is both short (MAX_DETACHED_RUN) and outweighed by what is left --
    so a line the search split evenly in two keeps its opening rather than jumping forward,
    which is the tie-break the coarse pass has always had.
    """
    lo, hi = 0, len(runs)
    rest = sum(len(run) for run in runs)
    # A tie is resolved towards the *opening*: a line the search split evenly in two starts
    # where the line starts rather than jumping forward, which is the tie-break the coarse
    # pass has always had. Hence < at the head and <= at the tail.
    while hi - lo > 1 and len(runs[lo]) <= MAX_DETACHED_RUN and len(runs[lo]) < rest - len(runs[lo]):
        rest -= len(runs[lo])
        lo += 1
    while hi - lo > 1 and len(runs[hi - 1]) <= MAX_DETACHED_RUN and len(runs[hi - 1]) <= rest - len(runs[hi - 1]):
        rest -= len(runs[hi - 1])
        hi -= 1
    return [item for run in runs[lo:hi] for item in run]


def _core_token_run(
    body: Sequence[int],
    first_frame: Dict[int, int],
    last_frame: Dict[int, int],
    frame_times: np.ndarray,
    max_gap: float = MAX_INTERNAL_GAP,
) -> List[int]:
    """A line's tokens with any short, detached run trimmed off either end.

    See MAX_INTERNAL_GAP for why a detached token is dropped rather than spanned, and
    MAX_DETACHED_RUN for why only a short one is.
    """
    body = list(body)
    if len(body) < 2:
        return body
    last = len(frame_times) - 1

    def at(f: int) -> float:
        return float(frame_times[min(max(f, 0), last)])

    runs: List[List[int]] = [[body[0]]]
    for prev, cur in zip(body, body[1:]):
        if at(first_frame[cur]) - at(last_frame[prev]) > max_gap:
            runs.append([cur])
        else:
            runs[-1].append(cur)
    return _trim_detached_edges(runs)


# --- The acoustic anchor ------------------------------------------------------------
#
# Placement is bracketing, not a sweep. The old search walked the file with one pointer,
# asking each window "how much of the remaining transcript does this explain?" and advancing
# by whatever it consumed. On audio the model cannot read the honest answer is "none of it",
# but the window still had to move on, so *time* advanced by two minutes while the line
# pointer barely moved -- and because the pointer only ever went forward it could never get
# back. On the Doraemon fixture that put half the file more than two and a half minutes out;
# on Police Story 2 it ran off the end with 1015 lines unplaced.
#
# Instead: find the lines that can be placed with confidence (acquire), throw out the ones
# their neighbours contradict (sanitise, verify), and align everything else against exactly
# the audio between two confident neighbours (fill). A bad stretch is then trapped between
# two good anchors and cannot spread. Measured on Doraemon: median start error 153 s -> 0.04 s.


class _Aligner:
    """The one primitive everything here is built from: align lines [lo, hi) against [t0, t1].

    ``free`` picks the question. A *forced* alignment must consume every token, which is the
    right question once two anchors say these lines are what is spoken in this span. A
    *free-end* one may stop early, which is the right question while still searching.
    """

    def __init__(self, timeline: EmissionTimeline, tokens_per_line, blank_id: int):
        self.timeline = timeline
        self.tokens_per_line = tokens_per_line
        self.blank_id = blank_id
        self.trellises = 0
        self.cells = 0

    def spans(self, lo: int, hi: int):
        tokens: List[int] = []
        spans: List[Tuple[int, int]] = []
        for i in range(lo, hi):
            start = len(tokens)
            tokens.extend(self.tokens_per_line[i])
            tokens.append(self.blank_id)   # somewhere for the inter-line pause to live
            spans.append((start, len(tokens) - 1))
        return tokens, spans

    def cost(self, lo: int, hi: int, t0: float, t1: float) -> Tuple[int, int]:
        """(frames, tokens) for this request, without building anything."""
        a, b = self.timeline.chunk_range(t0, t1)
        frames = int(round((min(t1, self.timeline.file_end) - t0) * 25.0))
        tokens = sum(len(self.tokens_per_line[i]) + 1 for i in range(lo, hi))
        return max(frames, 0), tokens

    def run(self, lo: int, hi: int, t0: float, t1: float, free: bool):
        emission, frame_times = self.timeline.slice(t0, t1)
        tokens, spans = self.spans(lo, hi)
        if not tokens or emission.size(0) == 0:
            return []
        self.trellises += 1
        self.cells += (emission.size(0) + 1) * (len(tokens) + 1)
        trellis = get_trellis(emission, tokens, self.blank_id, free_end=free)
        path = backtrack(trellis, emission, tokens, self.blank_id, free_end=free)
        if not path:
            return []
        return _line_spans_impl(path, tokens, spans, frame_times, self.blank_id)


def _line_spans_impl(path, tokens, spans, frame_times, blank_id):
    first: Dict[int, int] = {}
    last: Dict[int, int] = {}
    scores: Dict[int, List[float]] = {}
    for point in path:
        first.setdefault(point.token_index, point.time_index)
        last[point.token_index] = point.time_index
        scores.setdefault(point.token_index, []).append(point.score)
    consumed = path[-1].token_index + 1
    n = len(frame_times)
    out: List[Tuple[int, float, float, float]] = []
    for k, (lo_tok, hi_tok) in enumerate(spans):
        if hi_tok > consumed:
            break
        body = [j for j in range(lo_tok, hi_tok) if j in first and tokens[j] != blank_id]
        core = _core_token_run(body, first, last, frame_times) if body else []
        if not core:
            # See the note in the module docstring: a line the align model has no characters
            # for still has a sentinel, and the path walked past it in order, so the sentinel
            # places the line between its neighbours. Abandoning the rest of the span here
            # failed 8 of 21 fills on the Police Story 2 head.
            if hi_tok not in first:
                continue
            out.append((k, float(frame_times[first[hi_tok]]),
                        float(frame_times[min(last[hi_tok] + 1, n - 1)]), float("nan")))
            continue
        start = float(frame_times[min(first[j] for j in core)])
        tail = max(last[j] for j in core)
        if core[-1] == body[-1]:
            tail = last.get(hi_tok, tail)
        end = float(frame_times[min(tail + 1, n - 1)])
        flat = [s for j in core for s in scores[j]]
        out.append((k, start, end, float(np.mean(flat)) if flat else float("nan")))
    return out


def _offer(tokens_per_line: Sequence[Sequence[int]], p: int, seconds: float, frames: int) -> int:
    """How many of the remaining lines to offer a free-end probe.

    Over-supply is the point -- the free end decides where to stop -- but CTC needs at least
    one frame per token, so the offer is also held well short of the frame count.
    """
    budget = max(1, int(seconds * MAX_CHAR_RATE))
    frame_cap = max(1, int(frames * 0.8))
    n_lines, n_tokens = 0, 0
    for i in range(p, len(tokens_per_line)):
        size = len(tokens_per_line[i]) + 1
        if n_lines and (n_tokens + size > budget or n_tokens + size > frame_cap):
            break
        n_tokens += size
        n_lines += 1
    return n_lines


def acquire(
    lines: Sequence[TranscriptLine], aligner: _Aligner, *, window_seconds: float,
    commit_margin: float, stats: dict,
) -> List[Tuple[int, float, float, float]]:
    """Find the lines that can be placed confidently. Anchors only; gaps are left for fill.

    Two things separate this from the old sweep. The line pointer may **jump**: when the probe
    at the current pointer explains nothing, the same window is retried further down the
    transcript, so a search that has lost its place can re-acquire instead of staying lost.
    And progress in *time* is guaranteed -- every iteration either anchors (the pointer moves)
    or advances the window -- so the loop cannot spin on unreadable audio.
    """
    timeline = aligner.timeline
    # The overlap has to leave the window somewhere to go: a caller may set a window smaller
    # than the default overlap, and t1 - overlap would then be *behind* t, so a blind window
    # would step backwards and the loop would never end.
    overlap = min(ANCHOR_WINDOW_OVERLAP, window_seconds / 2.0)
    anchors: List[Tuple[int, float, float, float]] = []
    p, t = 0, timeline.file_start
    while p < len(lines) and t < timeline.file_end - 0.5:
        t1 = min(t + window_seconds, timeline.file_end)
        frames = max(int(round((t1 - t) * 25.0)), 1)
        best = None
        for offset in ANCHOR_OFFSETS:
            q = p + offset
            if q >= len(lines):
                break
            hi = min(len(lines), q + _offer(aligner.tokens_per_line, q, t1 - t, frames))
            got = aligner.run(q, hi, t, t1, free=True)
            if got:
                mass = sum(len(lines[q + k].text) * s for k, _, _, s in got if s == s)
                quality = [s for _, _, _, s in got if s == s]
                quality = float(np.mean(quality)) if quality else 0.0
                if best is None or mass > best[0]:
                    best = (mass, quality, q, got)
            # The alternatives are a re-acquisition search; they are only worth their
            # trellises once the pointer itself has clearly failed.
            if offset == 0 and best is not None and best[1] >= ACQUIRE_MIN_SCORE:
                break

        found = []
        if best is not None and best[1] >= ACQUIRE_MIN_SCORE:
            _mass, _quality, q, got = best
            for k, start, end, score in got:
                i = q + k
                if end > t1 - commit_margin and t1 < timeline.file_end:
                    break     # only half seen; the next window gets a proper look at it
                if not (score == score) or score < ANCHOR_MIN_SCORE:
                    continue  # NaN fails every comparison silently, so reject it explicitly
                if len(lines[i].text) < ANCHOR_MIN_CHARS:
                    continue
                if not _bounded_timing(lines[i], start, end, floor=t).placed:
                    continue  # implausibly slow for its text: not something to anchor on
                found.append((i, start, end, score))
        if not found:
            stats["blind"] += 1
            t = max(t1 - overlap, t + MIN_LINE_STEP) if t1 < timeline.file_end                 else timeline.file_end
            continue
        anchors.extend(found)
        p = found[-1][0] + 1
        # An anchor whose end is at or behind the window start would leave the pointer where
        # it was; the line pointer has moved, but time must move too or the next window is
        # the same window.
        t = max(found[-1][2], t + MIN_LINE_STEP)
    return anchors


def sanitise_anchors(
    anchors: Sequence[Tuple[int, float, float, float]], lines: Sequence[TranscriptLine],
) -> List[Tuple[int, float, float, float]]:
    """The heaviest set of anchors that can all be true at once.

    Anchors must increase in time with the line index, and consecutive ones must leave at
    least ``chars_between / MAX_CHAR_RATE`` seconds for the text between them -- a physical
    bound, not a heuristic: that text has to be spoken somewhere. Weight is chars x score, so
    a long confident line outranks a short one. Anything off the chosen chain is discarded and
    filled from its neighbours instead.

    Only the *lower* bound on the gap is used. Capping the forward jump as well is the
    documented trap (see _bounded_timing): films contain genuinely wordless stretches, and
    clamping the first legitimate one cascades.
    """
    if not anchors:
        return []
    cumulative = np.cumsum([0] + [len(line.text) for line in lines])
    ordered = sorted(anchors, key=lambda a: a[0])
    n = len(ordered)
    weight = [len(lines[a[0]].text) * a[3] for a in ordered]
    best = list(weight)
    previous = [-1] * n
    for i in range(n):
        for j in range(i):
            if ordered[j][2] > ordered[i][1]:
                continue
            room = (cumulative[ordered[i][0]] - cumulative[ordered[j][0] + 1]) / MAX_CHAR_RATE
            if ordered[i][1] - ordered[j][2] < room - 1e-6:
                continue
            if best[j] + weight[i] > best[i]:
                best[i], previous[i] = best[j] + weight[i], j
    k = int(np.argmax(best))
    chain = []
    while k >= 0:
        chain.append(ordered[k])
        k = previous[k]
    return chain[::-1]


def verify_anchors(
    chain: Sequence[Tuple[int, float, float, float]], aligner: _Aligner, stats: dict,
) -> List[Tuple[int, float, float, float]]:
    """Drop anchors that a forced alignment of their neighbourhood contradicts.

    An anchor can be confidently wrong. On the Police Story 2 head two police-radio lines
    anchored at 214.8 s and 223.8 s with scores 0.92 and 0.89 while belonging at 195.8 and
    197.2. Nothing local separates them: the score is high and the implied speaking rate is
    fine. Two obvious global tests were built and measured, and **both fail** -- the emission's
    blank probability is not a voice-activity detector (CTC is peaky enough that a whole 360 s
    clip shows 24 s of non-blank frames, and the 15 s the search wrongly skipped measured
    0.00 s of "speech", the same as a genuinely wordless stretch), and the VAD speech spans do
    no better (1.41 s in the false jump against 1.52 s in a real 64 s silence). Do not
    reintroduce either.

    What does separate them is their neighbours. Forced-aligning a block of consecutive
    anchors against the audio those anchors bracket has to consume every token, so it puts
    each line where the audio supports it; an anchor that then moves by seconds was wrong.
    Blocks overlap so that every anchor is judged from inside one, not from its own edge where
    it would trivially agree with itself.
    """
    if len(chain) < 3:
        return list(chain)
    moved: Dict[int, List[float]] = {}
    for s in range(0, max(len(chain) - 1, 1), VERIFY_STRIDE):
        block = chain[s:s + VERIFY_BLOCK]
        if len(block) < 3:
            continue
        lo, hi = block[0][0], block[-1][0] + 1
        t0, t1 = block[0][1], block[-1][2]
        frames, tokens = aligner.cost(lo, hi, t0, t1)
        if frames < tokens or frames * tokens > CELL_BUDGET:
            continue
        got = aligner.run(lo, hi, t0, t1, free=False)
        stats["verified"] += 1
        if not got or got[-1][0] != hi - lo - 1:
            continue
        placed = {lo + k: start for k, start, _end, _score in got}
        for (i, start, _end, _score) in block[1:-1]:   # the ends define the span
            if i in placed:
                moved.setdefault(i, []).append(abs(placed[i] - start))
    keep = [a for a in chain
            if a[0] not in moved or float(np.median(moved[a[0]])) <= VERIFY_TOLERANCE]
    stats["dropped"] += len(chain) - len(keep)
    return keep


def _interpolate_span(
    lines: Sequence[TranscriptLine], lo: int, hi: int, t0: float, t1: float,
    reason: str, out: List[Optional[LineTiming]], stats: dict,
) -> None:
    """Lay lines [lo, hi) evenly across [t0, t1], each flagged with *reason*."""
    step = max(t1 - t0, 0.04) / max(hi - lo, 1)
    for k, i in enumerate(range(lo, hi)):
        start = t0 + k * step
        span = min(step, _plausible_span(lines[i].text))
        out[i] = LineTiming(lines[i].index, round(start, 3),
                            round(start + max(span, MIN_LINE_STEP), 3), reason=reason)
    stats["interpolated"] += hi - lo


def _patch_holes(
    lines: Sequence[TranscriptLine], lo: int, hi: int, t0: float, t1: float,
    out: List[Optional[LineTiming]], stats: dict,
) -> None:
    """Place any line the path could not time across the gap its neighbours leave."""
    i = lo
    while i < hi:
        if out[i] is not None:
            i += 1
            continue
        j = i
        while j < hi and out[j] is None:
            j += 1
        a = out[i - 1].end if i > lo and out[i - 1] is not None else t0
        b = out[j].start if j < hi and out[j] is not None else t1
        _interpolate_span(lines, i, j, a, max(b, a + 0.04), REASON_ISOLATED, out, stats)
        i = j


def fill(
    lines: Sequence[TranscriptLine], aligner: _Aligner, lo: int, hi: int,
    t0: float, t1: float, out: List[Optional[LineTiming]], stats: dict, depth: int = 0,
) -> None:
    """Place lines [lo, hi), known to lie in (t0, t1), by forced alignment against that audio.

    Forced alignment is the right tool here precisely because the brackets are trusted: these
    lines are what is said in this span, so every token must be consumed and each one lands
    where the audio supports it.
    """
    if lo >= hi:
        return
    frames, tokens = aligner.cost(lo, hi, t0, t1)
    if tokens == 0 or frames < tokens or depth > MAX_FILL_DEPTH:
        past_end = t0 >= aligner.timeline.file_end - 0.5
        _interpolate_span(lines, lo, hi, t0, t1,
                          REASON_NO_AUDIO if past_end else REASON_UNREADABLE, out, stats)
        return
    if frames * tokens <= CELL_BUDGET:
        got = aligner.run(lo, hi, t0, t1, free=False)
        stats["fills"] += 1
        # A forced alignment consumes every token, so reaching the last line means the whole
        # span was placed. Lines missing from the middle are holes to interpolate, not a
        # reason to re-split.
        if got and got[-1][0] == hi - lo - 1:
            for k, start, end, score in got:
                out[lo + k] = LineTiming(lines[lo + k].index, round(start, 3), round(end, 3),
                                         score, None if score == score else REASON_ISOLATED)
            _patch_holes(lines, lo, hi, t0, t1, out, stats)
            return
        stats["fill_failed"] += 1
    # Too big for one trellis (or it failed): split on the text and cut at the quietest frame
    # near where that lands, the same min-cut idea as Binarize._split_long.
    sizes = np.cumsum([len(lines[i].text) + 1 for i in range(lo, hi)])
    mid = min(max(lo + int(np.searchsorted(sizes, sizes[-1] / 2)) + 1, lo + 1), hi - 1)
    share = sizes[mid - lo - 1] / sizes[-1]
    guess = t0 + share * (t1 - t0)
    band = max(2.0, 0.15 * (t1 - t0))
    a, b = max(t0 + 0.5, guess - band), min(t1 - 0.5, guess + band)
    if b > a:
        emission, times = aligner.timeline.slice(a, b)
        cut = float(times[int(torch.argmax(emission[:, aligner.blank_id]).item())]) \
            if emission.numel() and len(times) else (t0 + t1) / 2
    else:
        cut = (t0 + t1) / 2
    stats["splits"] += 1
    fill(lines, aligner, lo, mid, t0, cut, out, stats, depth + 1)
    fill(lines, aligner, mid, hi, cut, t1, out, stats, depth + 1)


def assign_lines(
    lines: Sequence[TranscriptLine],
    vad_segments: Sequence[VadAudioSegment],
    timeline: EmissionTimeline,
    model_dictionary: dict,
    model_lang: str,
    *,
    blank_id: Optional[int] = None,
    window_seconds: float = 120.0,
    commit_margin: float = 10.0,
    punctuation: PunctuationConfig = REALIGN_PUNCTUATION,
    progress_callback=None,
) -> List[LineTiming]:
    """Place every transcript line on the audio timeline, in transcript order.

    Anchors first, then everything between anchors by forced alignment against exactly the
    audio those anchors bracket. Returns one LineTiming per input line; a ``reason`` marks a
    line whose span came from its neighbours rather than from the audio.
    """
    if not vad_segments:
        raise ValueError("realign needs at least one audio chunk")
    if blank_id is None:
        blank_id = _get_blank_id(model_dictionary)

    tokens_per_line = [
        line_tokens(line.text, model_lang, model_dictionary, blank_id, punctuation)
        for line in lines
    ]
    aligner = _Aligner(timeline, tokens_per_line, blank_id)
    stats = dict(blind=0, verified=0, dropped=0, fills=0, fill_failed=0, splits=0,
                 interpolated=0, unconstrained=0)

    anchors = acquire(lines, aligner, window_seconds=window_seconds,
                      commit_margin=commit_margin, stats=stats)
    chain = sanitise_anchors(anchors, lines)
    chain = sanitise_anchors(verify_anchors(chain, aligner, stats), lines)

    out: List[Optional[LineTiming]] = [None] * len(lines)
    for i, start, end, score in chain:
        out[i] = LineTiming(lines[i].index, round(start, 3), round(end, 3), score)
    if progress_callback is not None:
        progress_callback.advance(len(chain))

    bounds = [(-1, timeline.file_start)] + [(a[0], a[2]) for a in chain] + \
             [(len(lines), timeline.file_end)]
    for (i0, ta), (i1, tb) in zip(bounds, bounds[1:]):
        if i1 <= i0 + 1:
            continue
        fill(lines, aligner, i0 + 1, i1, ta, max(tb, ta + 0.1), out, stats)
        if progress_callback is not None:
            progress_callback.advance(i1 - i0 - 1)

    # A line with no alignable character was never going to be timed from the audio; say so
    # rather than letting it look like a placement failure. The test is for a *non-blank*
    # token: punctuation maps to the blank id, so a line of nothing but out-of-vocabulary
    # characters and a full stop still has a token list, just no evidence in it.
    for i, line in enumerate(lines):
        if out[i] is None:
            out[i] = LineTiming(line.index, round(timeline.file_end, 3),
                                round(timeline.file_end + 0.04, 3), reason=REASON_NO_AUDIO)
        elif (out[i].reason == REASON_ISOLATED
                and not any(t != blank_id for t in tokens_per_line[i])):
            out[i].reason = REASON_NO_VOCABULARY

    flag_unconstrained(out, lines, timeline.file_end, stats)
    logger.info(
        "realign: %d anchor(s) kept (%d dropped on review), %d line(s) filled between them; "
        "%d blind window(s), %d split(s)",
        len(chain), stats["dropped"], len(lines) - len(chain), stats["blind"], stats["splits"],
    )
    _report(out, lines)
    return out


def flag_unconstrained(
    timings: Sequence[LineTiming], lines: Sequence[TranscriptLine], file_end: float, stats: dict,
) -> None:
    """Flag lines whose neighbours leave far more room than the line could fill.

    This is the per-cue trust signal, and it is geometric rather than acoustic. A line placed
    between two neighbours that are seconds further apart than its text could ever span is not
    pinned down by anything: forced alignment had to put it somewhere in that gap and was free
    to choose. Police Story 2's ``May呀`` is the extreme -- two characters with 66 s of room --
    but the same shape accounts for every remaining error on Doraemon, all of them short lines
    sitting early in a gap their neighbours left.

    Measured against the confidence score, which does *not* work for this: ``May呀`` scored
    0.733, above the median, while a correctly-placed 「唔該」 scored 0.001. See
    UNCONSTRAINED_ROOM and warn_low_confidence.
    """
    by_index = {t.index: t for t in timings}
    order = [by_index[line.index] for line in lines if line.index in by_index]
    for k, timing in enumerate(order):
        previous = order[k - 1].end if k else 0.0
        following = order[k + 1].start if k + 1 < len(order) else file_end
        room = following - previous
        if room > UNCONSTRAINED_ROOM * max(_plausible_span(lines[k].text), MIN_LINE_SPAN):
            stats["unconstrained"] += 1
            if timing.reason is None:
                timing.reason = REASON_ISOLATED


def _report(timings: Sequence[LineTiming], lines: Sequence[TranscriptLine]) -> None:
    """Summarise by reason. One count would hide the only case worth stopping for."""
    scored = [t.score for t in timings if not math.isnan(t.score)]
    if scored:
        logger.info(
            "realign: placed %d line(s); mean path score %.3f (p10 %.3f)",
            len(timings), float(np.mean(scored)), float(np.percentile(scored, 10)),
        )
    counts: Dict[str, int] = {}
    for t in timings:
        if t.reason:
            counts[t.reason] = counts.get(t.reason, 0) + 1
    if not counts:
        return
    suspect = sum(n for r, n in counts.items() if r not in BENIGN_REASONS)
    logger.warning("realign: %d of %d line(s) need review", suspect or sum(counts.values()),
                   len(timings))
    for reason, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        logger.warning("    %4d  %-14s %s", n, reason, REASON_HELP.get(reason, ""))
    if counts.get(REASON_NO_AUDIO, 0) > max(10, 0.02 * len(timings)):
        logger.warning(
            "realign: %d line(s) have no audio at all -- the transcript and this recording "
            "are probably not the same content, or the search lost its place badly. Check "
            "before trusting any of the output.", counts[REASON_NO_AUDIO],
        )


def warn_low_confidence(
    timings: Sequence[LineTiming], lines: Sequence[TranscriptLine], min_score: float,
) -> List[LineTiming]:
    """Log the lines whose acoustic support is weak, and return them.

    A scattering is normal (interjections, digits, speech under music); a run of them in one
    stretch means the transcript and the audio have diverged there. That is all this is: a
    *document*-level mismatch signal.

    It deliberately does **not** mark the cues. Measured on Police Story 2, the score is a
    poor per-cue trust signal in both directions: it flagged 187 of 2172 lines, most of them
    correctly-placed short interjections (「唔該」 scores 0.001 and is right to within 0.2 s),
    and it missed the one line that is badly wrong (``May呀``, 48 s out at score 0.733). The
    cue-level signal is the bracket geometry instead -- see UNCONSTRAINED_ROOM.
    """
    by_index = {line.index: line for line in lines}
    weak = [t for t in timings if not math.isnan(t.score) and t.score < min_score]
    if not weak:
        return []
    logger.warning(
        "realign: %d line(s) below the confidence floor (%.2f); check these first:",
        len(weak), min_score,
    )
    for t in weak[:10]:
        logger.warning("  %8.2fs score=%.3f  %r", t.start, t.score, by_index[t.index].text[:48])
    if len(weak) > 10:
        logger.warning("  ... and %d more", len(weak) - 10)
    return weak


# --- The ASR anchor -----------------------------------------------------------------

# How much of each stream one matching block covers. The two streams are monotonic and
# roughly the same length, so a bounded window is enough and keeps the quadratic matcher off
# the whole film: 15k x 15k characters would be minutes of work, 600 x 900 is milliseconds.
_MATCH_BLOCK_CHARS = 600
_MATCH_LOOKAHEAD = 900


def _speech_by_segment(
    vad_segments: Optional[Sequence[VadAudioSegment]],
) -> Dict[int, List[List[float]]]:
    """Index the per-chunk ``speech`` spans (see vad.py) by rounded chunk start."""
    if not vad_segments:
        return {}
    return {
        int(round(float(seg["start"]) * 1000)): seg["speech"]
        for seg in vad_segments if seg.get("speech")
    }


def _char_times(t0: float, t1: float, n: int, speech: Optional[Sequence[Sequence[float]]]):
    """Lay *n* character slots across [t0, t1], over the speech in it where that is known.

    Spreading characters evenly over the whole span is only right when the span *is* speech.
    Under --realign it is not: split-only chunking hands ASR a contiguous 14-28 s block of
    film, so a chunk holding one line of dialogue and twenty seconds of car chase gets the
    line smeared across the lot. Measured on Police Story 2, the chunk at 382.5-409.8 s holds
    37 characters over 27.3 s -- 0.74 s each, uniformly -- which timed 「巡邏嘅伙記注意」 at
    382.5 s when it is spoken at 391.4, and 「請儘速趕去現場」 at 404.0 when it is at 397.2.
    Errors of that size move a line into the wrong alignment chunk, and once the chunk does
    not contain the audio, nothing downstream can recover it.

    Distributing the same characters over the *speech* inside the chunk instead costs one
    extra binarization of a score curve VAD already computed, and places both of those lines
    within a few tenths of a second. Where no speech was recorded (an ordinary speech-only
    VAD run, or a chunk VAD scored as silent yet ASR still read something) it falls back to
    spreading them evenly, which is the old behaviour and the only thing left to do.
    """
    spans = [(max(a, t0), min(b, t1)) for a, b in (speech or []) if min(b, t1) > max(a, t0)]
    total = sum(b - a for a, b in spans)
    if not spans or total <= 0:
        step = (t1 - t0) / n
        return [(t0 + k * step, t0 + (k + 1) * step) for k in range(n)]

    # Each speech region takes the share of the characters its duration is worth.
    out: List[Tuple[float, float]] = []
    assigned, consumed = 0, 0.0
    for i, (a, b) in enumerate(spans):
        consumed += b - a
        upto = n if i == len(spans) - 1 else int(round(n * consumed / total))
        count = upto - assigned
        if count <= 0:
            continue
        step = (b - a) / count
        out.extend((a + k * step, a + (k + 1) * step) for k in range(count))
        assigned = upto
    return out


def _hypothesis_stream(
    asr_segments: Sequence[dict],
    punctuation: PunctuationConfig = REALIGN_PUNCTUATION,
    vad_segments: Optional[Sequence[VadAudioSegment]] = None,
) -> Tuple[str, np.ndarray, np.ndarray]:
    """Flatten ASR output into (characters, start times, end times).

    Neither ASR backend emits per-character timings (``time_stamps`` is never set), so each
    segment's characters are laid across the audio it covers -- over the speech in it where
    ``vad_segments`` records that, evenly otherwise. See _char_times.
    """
    skip = set(punctuation.split_chars)
    speech_by_start = _speech_by_segment(vad_segments)
    chars, starts, ends = [], [], []
    for seg in asr_segments:
        text = [c for c in seg.get("text", "") if c not in skip and not c.isspace()]
        if not text:
            continue
        t0, t1 = float(seg["start"]), float(seg["end"])
        speech = speech_by_start.get(int(round(t0 * 1000)))
        for ch, (a, b) in zip(text, _char_times(t0, t1, len(text), speech)):
            chars.append(ch)
            starts.append(a)
            ends.append(b)
    return "".join(chars), np.asarray(starts), np.asarray(ends)


def _transcript_stream(
    lines: Sequence[TranscriptLine], punctuation: PunctuationConfig = REALIGN_PUNCTUATION,
) -> Tuple[str, List[int]]:
    """Flatten the transcript the same way, remembering which line each character came from."""
    skip = set(punctuation.split_chars)
    chars, owners = [], []
    for line in lines:
        for ch in line.text:
            if ch not in skip and not ch.isspace():
                chars.append(ch)
                owners.append(line.index)
    return "".join(chars), owners


def assign_lines_via_asr(
    lines: Sequence[TranscriptLine],
    asr_segments: Sequence[dict],
    punctuation: PunctuationConfig = REALIGN_PUNCTUATION,
    vad_segments: Optional[Sequence[VadAudioSegment]] = None,
) -> List[LineTiming]:
    """Time each transcript line by matching it against what ASR actually heard.

    The alternative to the acoustic search, and better in one specific way: forced alignment
    *must* consume every token it is given, so a transcript line the recording does not
    contain gets smeared over whatever audio is nearest. A text match can simply fail to match
    it, which is both the correct answer and a reportable one -- unmatched lines come back
    flagged as ``unreadable`` after being interpolated between their neighbours.

    Both streams are monotonic, so matching runs in bounded blocks rather than as one diff
    over the whole file; see _MATCH_BLOCK_CHARS.
    """
    hyp, hyp_start, hyp_end = _hypothesis_stream(asr_segments, punctuation, vad_segments)
    text, owners = _transcript_stream(lines, punctuation)
    if not hyp or not text:
        logger.warning("realign: ASR produced no usable text to match against")
        return _interpolate_unplaced(lines, {}, asr_segments)

    # Character index in the transcript -> character index in the hypothesis.
    matched: Dict[int, int] = {}
    t_pos = h_pos = 0
    while t_pos < len(text) and h_pos < len(hyp):
        t_end = min(len(text), t_pos + _MATCH_BLOCK_CHARS)
        h_end = min(len(hyp), h_pos + _MATCH_LOOKAHEAD)
        matcher = difflib.SequenceMatcher(
            None, text[t_pos:t_end], hyp[h_pos:h_end], autojunk=False
        )
        blocks = [b for b in matcher.get_matching_blocks() if b.size]
        if not blocks:
            t_pos, h_pos = t_end, h_end
            continue
        for block in blocks:
            for k in range(block.size):
                matched[t_pos + block.a + k] = h_pos + block.b + k
        last = blocks[-1]
        # Advance past everything consumed, leaving the tail of the block to be re-matched
        # with more context next time round.
        t_pos += last.a + last.size
        h_pos += last.b + last.size

    spans: Dict[int, Tuple[float, float]] = {}
    by_line: Dict[int, List[int]] = {}
    for t_idx, h_idx in matched.items():
        by_line.setdefault(owners[t_idx], []).append(h_idx)
    for index, hits in by_line.items():
        spans[index] = (float(hyp_start[min(hits)]), float(hyp_end[max(hits)]))

    logger.info(
        "realign: matched %d of %d line(s) against the ASR hypothesis (%d of %d characters)",
        len(spans), len(lines), len(matched), len(text),
    )
    return _interpolate_unplaced(lines, spans, asr_segments)


def _interpolate_unplaced(
    lines: Sequence[TranscriptLine],
    spans: Dict[int, Tuple[float, float]],
    asr_segments: Sequence[dict],
) -> List[LineTiming]:
    """Fill in the lines the text match could not place, between their nearest neighbours.

    An unmatched line is a claim the recording does not support, so it is given a share of
    the gap around it and flagged rather than being hidden. The alternative -- dropping it --
    would silently lose a line the user wrote down.

    A whole *run* of them is filled at once, and that matters. Placing each unmatched line
    immediately after the last matched one stacks every line of a run on a single instant,
    which is how four consecutive Police Story 2 cues came to share a timestamp -- and a text
    match fails in runs by its nature, because what defeats it is a stretch of repeated or
    near-identical short lines. Laying the run out across the gap instead, with the leftover
    silence shared equally before, between and after, keeps them ordered and separable, and
    for a single line in a long gap it centres the guess rather than jamming it against the
    near edge (which is the worst place for it: the error is then the whole gap).
    """
    file_end = max((float(s["end"]) for s in asr_segments), default=0.0)
    out: List[LineTiming] = []
    i = 0
    while i < len(lines):
        span = spans.get(lines[i].index)
        if span is not None:
            start, end = span
            out.append(LineTiming(lines[i].index, round(start, 3),
                                  round(max(end, start + 0.04), 3)))
            i += 1
            continue

        run_end = i
        while run_end < len(lines) and lines[run_end].index not in spans:
            run_end += 1
        prev = next((t.end for t in reversed(out) if t.placed), 0.0)
        following = spans[lines[run_end].index][0] if run_end < len(lines) else file_end

        run = lines[i:run_end]
        wanted = [max(0.4, len(ln.text) / NOMINAL_CHAR_RATE) for ln in run]
        room = max(following - prev, 0.0)
        if room < sum(wanted):  # not enough silence to be generous with
            scale = room / sum(wanted) if sum(wanted) else 0.0
            wanted = [max(0.04, w * scale) for w in wanted]
        pause = max(room - sum(wanted), 0.0) / (len(run) + 1)

        cursor = prev + pause
        for ln, want in zip(run, wanted):
            out.append(LineTiming(ln.index, round(cursor, 3),
                                  round(cursor + want, 3), reason=REASON_UNREADABLE))
            cursor += want + pause
        i = run_end
    unmatched = sum(1 for t in out if not t.placed)
    if unmatched:
        logger.warning(
            "realign: %d of %d line(s) had no match in the ASR hypothesis and were given "
            "interpolated timings. A run of these in one stretch means the transcript and "
            "the recording have genuinely diverged there.", unmatched, len(out),
        )
    return out


# --- Turning placements into alignment input ----------------------------------------

def _sanitize(
    timings: Sequence[LineTiming], file_start: float, file_end: float,
) -> List[LineTiming]:
    """Force placements into an in-bounds, strictly-forward sequence, in *transcript order*.

    Nothing downstream is allowed to depend on the search having behaved. A run of lines
    stranded on the last frame of the audio (which is what a transcript longer than its
    recording produces) otherwise yields identical spans, non-monotonic chunk boundaries and
    therefore a grouping pass that never splits -- one chunk covering the entire file, and an
    encoder forward pass over it that will exhaust memory rather than raise. Guaranteeing the
    invariants here is much cheaper than guaranteeing them at every call site.

    Ordering is by **line index**, never by time. Sorting by time looks harmless while the
    search is behaving and is wrong the moment it is not: when several lines are handed the
    same instant -- which is exactly what happens to a run the ASR text match could not place
    -- the tie is broken by their *ends*, silently permuting the transcript. On Police Story 2
    that reversed four consecutive 唔該 lines. The transcript's order is authoritative and is
    the one thing this function must not touch; a placement that goes backwards is clamped by
    the cursor below, which is the correct repair.
    """
    out: List[LineTiming] = []
    cursor = file_start
    for timing in sorted(timings, key=lambda t: t.index):
        start = min(max(timing.start, cursor), file_end)
        end = max(timing.end, start + MIN_LINE_STEP)
        out.append(LineTiming(timing.index, round(start, 3), round(end, 3),
                              timing.score, timing.reason))
        # Step the cursor even when two lines claim the same instant, so a stranded run keeps
        # a stable order. These are past the end of the audio and already flagged unplaced;
        # the point is only that they stay separable into cues.
        cursor = start + MIN_LINE_STEP if start >= file_end else start
    return out


def _slice_audio(
    vad_segments: Sequence[VadAudioSegment], start: float, end: float,
) -> np.ndarray:
    """Copy the audio covering [start, end) out of a contiguous chunk list."""
    pieces = []
    for seg in vad_segments:
        if seg["end"] <= start or seg["start"] >= end:
            continue
        a = int(round((max(start, seg["start"]) - seg["start"]) * SAMPLE_RATE))
        b = int(round((min(end, seg["end"]) - seg["start"]) * SAMPLE_RATE))
        if b > a:
            pieces.append(seg["audio"][a:b])
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(pieces)


def build_align_input(
    lines: Sequence[TranscriptLine],
    timings: Sequence[LineTiming],
    vad_segments: Sequence[VadAudioSegment],
    chunk_size: float,
    *,
    pad: float = 0.2,
) -> Tuple[List[VadAudioSegment], List[SingleSegment]]:
    """Group placed lines into alignment chunks, re-cut so no line straddles a boundary.

    _get_emission_for_segment slices a transcript segment out of the *one* chunk holding its
    start, so a line spanning a boundary would have its tail compressed into the first
    chunk. The coarse pass has already said where every line begins, so the boundaries are
    simply moved onto the gaps between lines: each chunk starts and ends midway through a
    silence, and every line lies wholly inside exactly one of them.

    Returns (chunks, transcript) ready for align(): one SingleSegment per chunk, carrying its
    lines joined by sentinels and one cue_spans entry per line.
    """
    if not timings:
        return [], []

    by_index = {line.index: line for line in lines}
    by_timing = {t.index: t for t in timings}
    file_start = vad_segments[0]["start"]
    file_end = vad_segments[-1]["end"]
    ordered = _sanitize(timings, file_start, file_end)

    # Boundary between consecutive lines: the middle of the gap they leave. Clamped to be
    # non-decreasing so the budget test below is always measuring a real duration.
    bounds = [max(file_start, ordered[0].start - pad)]
    for a, b in zip(ordered, ordered[1:]):
        bounds.append(max(bounds[-1], a.end, min(b.start, (a.end + b.start) / 2)))
    bounds.append(max(bounds[-1], min(file_end, ordered[-1].end + pad)))

    # Two budgets, both hard: the chunk's duration, and the characters its frames can carry.
    # See MAX_CHARS_PER_SECOND for why the second is not optional.
    #
    # The character budget is measured against the duration the group has *actually* reached,
    # not against chunk_size. Measuring against the maximum is what let a group that collapsed
    # onto a fraction of a second still be handed a full chunk's worth of text: on Police
    # Story 2 that produced a ~1 s chunk carrying 330 characters, far more than its frames can
    # hold, so backtrack failed and every line in it came out as one enormous subtitle. Two of
    # those survive a full run today.
    groups: List[List[int]] = []
    current: List[int] = []
    chars = 0
    for i in range(len(ordered)):
        size = len(by_index[ordered[i].index].text) + 1  # + the sentinel
        # Budget against the audio the group actually has. Past the end of the recording
        # there is none, and splitting there buys nothing -- the lines have no frames either
        # way -- so fall back to the flat budget rather than emitting a chunk per line.
        available = (min(bounds[i + 1], file_end) - min(bounds[current[0]], file_end)
                     if current else 0.0)
        budget = min(available, chunk_size) if available > MIN_CHUNK_DURATION else chunk_size
        max_chars = max(1, int(budget * MAX_CHARS_PER_SECOND))
        too_long = current and bounds[i + 1] - bounds[current[0]] > chunk_size
        too_much = current and chars + size > max_chars
        if too_long or too_much:
            groups.append(current)
            current, chars = [], 0
        current.append(i)
        chars += size
    if current:
        groups.append(current)

    chunks: List[VadAudioSegment] = []
    transcript: List[SingleSegment] = []
    previous_end = file_start
    for group in groups:
        start = bounds[group[0]]
        end = max(bounds[group[-1] + 1], start + 0.04)
        text_parts, spans = [], []
        cursor = 0
        for i in group:
            body = by_index[ordered[i].index].text + REALIGN_SENTINEL
            text_parts.append(body)
            spans.append((cursor, cursor + len(body) - 1))  # inclusive, sentinel included
            cursor += len(body)
        end = min(end, start + chunk_size)  # the budget is a hard guarantee, not a target
        audio = _slice_audio(vad_segments, start, end)
        if len(audio) < int(MIN_CHUNK_DURATION * SAMPLE_RATE):
            # Reach *backwards* for the missing samples: a chunk this short only happens at
            # the end of the audio, where there is nothing ahead to reach for. Never behind
            # the chunk in front of it, though -- alignment relies on these being sorted and
            # disjoint, and reaching past a neighbour would orphan its words. See
            # MIN_CHUNK_DURATION.
            start = max(file_start, previous_end, end - MIN_CHUNK_DURATION)
            end = max(end, start + MIN_LINE_STEP)
            audio = _slice_audio(vad_segments, start, end)
        previous_end = end
        chunks.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "audio": audio,
        })
        transcript.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "text": "".join(text_parts),
            "cue_spans": spans,
            "cue_reasons": [by_timing[ordered[i].index].reason for i in group],
        })

    over = [c for c in chunks if c["end"] - c["start"] > chunk_size + 1e-6]
    if over:
        # Only reachable when one line alone claims more than the budget, since the grouping
        # above splits on it otherwise. Worth saying, but it is a warning rather than a
        # failure: _slice_audio already capped the chunk at the budget.
        logger.warning(
            "realign: %d chunk(s) reached --chunk_size on a single line; the longest is "
            "%.1fs.", len(over), max(c["end"] - c["start"] for c in over),
        )
    logger.info(
        "realign: %d line(s) grouped into %d alignment chunk(s)", len(ordered), len(chunks),
    )
    return chunks, transcript


def tighten_cue_spans(
    segments: List[dict], max_gap: float = MAX_INTERNAL_GAP,
) -> int:
    """Anchor each cue's start on a real character rather than on a blank's dwell.

    _align_segment takes a subsegment's start as the minimum over *all* its characters, and
    under --realign a cue spans a whole transcript line, so two ordinary facts combine badly:

    * a character out of the align model's vocabulary (interjections such as 噢 and 嗚 are,
      and they are exactly what a line tends to open with) carries no timing at all; and
    * the punctuation after it was mapped to the blank token, and CTC will happily dwell on
      a blank for as long as the silence lasts.

    So the cue inherits its start from a comma that "began" ten seconds earlier in the
    preceding silence. On the bluey fixture that put one subtitle on screen 8.8s before
    anybody spoke, while its own sentence aligned correctly -- the cue was not misplaced, its
    left edge was. Anchoring on the first *timed, non-blank* character costs the unmodellable
    interjection (the cue starts a beat late instead of a beat early) and fixes the rest.

    The same run-splitting also drops a short run of characters stranded more than ``max_gap``
    from the body of its line, which is the coarse search's guard applied to the final
    timings; see MAX_INTERNAL_GAP and MAX_DETACHED_RUN.

    Only the cue's start and end move, and only inwards, so cue ordering cannot be disturbed
    and word/character timings stay as alignment produced them. A cue left too short is then
    handled by the duration floor in segmentation.assemble_cues like any other short cue.
    Returns the number of cues adjusted.
    """
    split_chars = set(REALIGN_PUNCTUATION.split_chars)
    adjusted = 0
    for seg in segments:
        words = [
            w for w in (seg.get("words") or [])
            if w.get("start") is not None and w.get("end") is not None
            and w.get("word", "").strip() not in split_chars
            and w.get("word", "").strip() != ""
        ]
        if not words:
            continue
        runs: List[List[dict]] = [[words[0]]]
        for prev, cur in zip(words, words[1:]):
            if cur["start"] - prev["end"] > max_gap:
                runs.append([cur])
            else:
                runs[-1].append(cur)
        core = _trim_detached_edges(runs)

        start = max(seg["start"], core[0]["start"])
        # The end is left alone when the line's last character is in the core: alignment
        # deliberately released it into the following pause (align_release), and that is a
        # readability decision, not a misplacement.
        end = seg["end"] if core[-1] is words[-1] else min(seg["end"], core[-1]["end"])
        if end <= start:
            continue
        if start > seg["start"] + 1e-6 or end < seg["end"] - 1e-6:
            seg["start"], seg["end"] = round(start, 3), round(end, 3)
            adjusted += 1
    if adjusted:
        logger.info(
            "realign: pulled %d cue edge(s) onto the first/last character actually aligned "
            "there", adjusted,
        )
    return adjusted


def enforce_cue_order(segments: List[dict]) -> int:
    """Clamp any cue that starts before its predecessor. Returns the count moved.

    An SRT whose cues are not in start order is rejected outright -- suber refuses to read the
    file at all, and players disagree about what to do with it -- so this is a validity
    guarantee, not a quality one.

    _sanitize already orders the *coarse* placements, but the final timings come from forced
    alignment inside each chunk and it has no such constraint: where the search ran out of
    audio and stranded the tail of the transcript, consecutive chunks collapse onto the last
    fraction of a second of the file and their cues can come back interleaved. On the Doraemon
    fixture, whose acoustic anchor genuinely does run out of film, exactly one pair did.

    Order is fixed by moving the later cue *forward*, never by sorting: the transcript's line
    order is authoritative, so a cue that appears to precede its predecessor is a misplacement
    to be clamped, not evidence that the two should swap.
    """
    moved = 0
    for prev, seg in zip(segments, segments[1:]):
        if seg["start"] < prev["start"]:
            seg["start"] = prev["start"]
            seg["end"] = max(seg["end"], seg["start"])
            moved += 1
    if moved:
        logger.warning(
            "realign: %d cue(s) came back before the cue in front of them and were clamped; "
            "an out-of-order SRT is rejected by strict readers. This only happens where the "
            "placements themselves collapsed, so check the timings around them.", moved,
        )
    return moved


def ensure_visible_cues(segments: List[dict], min_duration: float = MIN_VISIBLE_DURATION) -> int:
    """Give every cue a duration it can actually be displayed for. Returns the count fixed.

    Borrows from the silence *before* the cue where there is any, since moving a start earlier
    cannot disturb the cue after it. Only when the previous cue ends flush against this one
    does it extend forward instead, overlapping by at most ``min_duration`` -- two frames of
    overlap is a far smaller defect than a subtitle that never appears, and by then the two
    cues were already claiming the same instant.
    """
    fixed = 0
    for i, seg in enumerate(segments):
        if seg["end"] - seg["start"] >= min_duration:
            continue
        floor = segments[i - 1]["end"] if i else 0.0
        start = max(floor, seg["end"] - min_duration)
        if seg["end"] - start >= min_duration:
            seg["start"] = round(start, 3)
        else:
            seg["start"] = round(start, 3)
            seg["end"] = round(start + min_duration, 3)
        fixed += 1
    if fixed:
        logger.info("realign: gave %d cue(s) a displayable duration", fixed)
    return fixed


def find_implausible_cues(segments: Sequence[dict]) -> List[dict]:
    """Cues the audio cannot support: too long for their text, or aligned on nothing.

    This is ``_bounded_timing``'s test applied to the *final* timings, and it is the only
    signal that survives the failure it detects. When the coarse pass puts a line in an
    alignment chunk that does not contain its speech, forced alignment still has to place
    every token somewhere, so it spreads them over whatever the chunk does hold.

    **Confidence does not see this.** On Police Story 2 an eight-character line belonging at
    368.8 s was grouped into the chunk starting at 372.9 s and came out as an 18.5 s cue --
    scoring 0.983, above the median. The geometry is what gives it away: 2.3 s per character
    is not slow speech. The complementary case, where the line's audio ends before the chunk
    starts, leaves nothing to align against at all and shows up in the score instead.

    Reports rather than repairs. Neither edge of such a cue is trustworthy, so there is no
    honest correction to make here -- the repair belongs upstream, in whatever put the line in
    the wrong chunk. Returns the offending segments, annotated with ``realign_reason``.
    """
    out: List[dict] = []
    for seg in segments:
        text = seg.get("text", "")
        if not text:
            continue
        duration = seg["end"] - seg["start"]
        scores = [w["score"] for w in (seg.get("words") or []) if w.get("score") is not None]
        mean = float(np.mean(scores)) if scores else float("nan")
        if duration > _plausible_span(text):
            seg["realign_detail"] = f"{duration / max(len(text), 1):.1f}s per character"
        elif scores and mean < MIN_CUE_SCORE:
            seg["realign_detail"] = f"aligned at score {mean:.3f}"
        else:
            continue
        # Only ever *add* a reason: one set at placement time (no audio, no vocabulary) says
        # more about the cue than "the geometry is odd", which is a consequence of it.
        seg.setdefault("realign_reason", REASON_IMPLAUSIBLE)
        out.append(seg)
    return out


def warn_on_implausible_cues(segments: Sequence[dict]) -> List[dict]:
    """Log what find_implausible_cues found, and return it."""
    bad = find_implausible_cues(segments)
    if not bad:
        return []
    logger.warning(
        "realign: %d of %d cue(s) cannot be supported by the audio they were aligned "
        "against -- their line was almost certainly grouped into the wrong chunk:",
        len(bad), len(segments),
    )
    for seg in bad[:10]:
        logger.warning(
            "  %8.2f-%8.2fs  %-22s %r",
            seg["start"], seg["end"], seg.get("realign_detail", ""), seg.get("text", "")[:40],
        )
    if len(bad) > 10:
        logger.warning("  ... and %d more", len(bad) - 10)
    return bad


def strip_sentinels(segments: List[dict]) -> int:
    """Remove the injected sentinels from aligned cue text; return the fused-cue count.

    _align_segment collapses subsegments that came out with identical (start, end) into a
    single row, which for --realign means two transcript lines were handed the same span and
    have quietly become one cue. A cue holding more than one sentinel is exactly that case,
    so counting them here is a free integrity check on the 1:1 line-to-cue mapping.
    """
    fused = 0
    for seg in segments:
        text = seg.get("text", "")
        count = text.count(REALIGN_SENTINEL)
        if count > 1:
            fused += count - 1
        if count:
            seg["text"] = text.replace(REALIGN_SENTINEL, "")
    if fused:
        logger.warning(
            "realign: %d transcript line(s) shared a timing with their neighbour and were "
            "merged into one cue.", fused,
        )
    return fused

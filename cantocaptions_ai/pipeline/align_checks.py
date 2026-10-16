"""Model-agnostic sanity checks on alignment output.

Unlike ``pipeline/align_profiles.py``, nothing here is per-model and nothing here is
opt-in: these run for **every** alignment model, including one added later with no profile
entry. That is possible because the checks read only the aligned timings and the audio —
never the emission tensor, the blank id, or the model's vocabulary — so they stay correct
for any model that produces ``{start, end, text}``.

There are two of them. ``find_gapped_cues`` reads the timings alone and needs no audio;
``find_silent_starts`` needs the waveform. Neither repairs anything -- both say which cues
to distrust, which is the thing a 2000-cue file cannot be used without.

``split_gapped_cues`` is the one exception, and it is exception-shaped on purpose: it acts
on ``find_gapped_cues``' finding by breaking the cue in two, and it is the only thing here
that is per align model, enabled by ``AlignProfile.split_gap`` or per run by
``--align_split_gap``. Reporting a doubtful cue costs nothing; inventing a cue boundary
that was not there does, so it only believes a gap the evidence supports (``SPLIT_MIN_PEAK``,
``SPLIT_MIN_SILENCE``) and never cuts inside a word. The detection is shared -- one scan, two
verdicts -- so the two can never disagree about what a gap is.

``find_silent_starts`` catches a cue whose start time lands on audio where nothing is
being said. Alignment placing a character over silence is always wrong, whatever produced
it; the failure that motivated the check was ``alvanlii/wav2vec2-BERT-cantonese`` pinning
each VAD segment's first character to frame 0 (see ``align_profiles.TailPrimer``), but the
check is written against the symptom rather than that cause.

**It is precise, not exhaustive.** Measured over 96 segments in three files: when it fires
it is a real error ~94% of the time, but it catches only about half of them. It can only
see a lead-in that is genuinely near-silent, and the failure it was built for also happens
over music beds and room tone, which are speechless but not silent. Treat a clean run as
"nothing obvious", not as proof the alignment is sound.
"""

import unicodedata
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from cantocaptions_ai.utils.audio import SAMPLE_RATE
from cantocaptions_ai.utils.log_utils import get_logger
from cantocaptions_ai.utils.schema import add_note

logger = get_logger(__name__)

# One frame per 40 ms, matching the ~25 fps the CTC emission runs at for the current align
# model. Nothing depends on the two agreeing — this is only the resolution the audio's
# energy envelope is measured at — but a comparable scale keeps "gap" counts meaningful.
FRAME_SECONDS = 0.04

# A frame counts as silent within this much of the region's own noise floor, so the check
# adapts to whatever the floor happens to be rather than assuming digital zero.
FLOOR_MARGIN_DB = 6.0

# Below this much spread between floor and speech level there is no usable silence floor
# (a music bed sits a few dB under quiet speech). Skip the region rather than guess: both
# false positives seen during development were music-bed segments, where "floor + margin"
# swallowed real speech.
MIN_SEPARATION_DB = 30.0

# How far a start must sit from *any* sound before it is suspicious. Two effects stack to
# make small gaps normal: CTC legitimately leads the acoustic onset by a frame or two, and
# an unvoiced onset (就, 三, 開 — affricates and fricatives) carries little energy, so the
# envelope reads as silence where speech has in fact begun. Swept over three files with the
# fix in place: 4 frames is the smallest value that never fires on correct output, while
# still catching errors at the scale of one vad_pad_onset (0.2s = 5 frames). At 3 it fires
# on correct cues; at 6 it only catches gross failures.
MIN_GAP_FRAMES = 4

# The least silence between two adjacent characters of one cue that says the cue is not a
# single continuous utterance. Well clear of an ordinary breath or an unvoiced onset, which
# is where find_silent_starts' 4-frame threshold sits; a whole second between two characters
# that are written next to each other means one of them was placed somewhere it was not said.
MAX_INTERNAL_GAP = 1.0

# split_gapped_cues' default when a caller names no threshold. The pipeline's comes from the
# align model (``AlignProfile.split_gap``: 1.0 s for the Cantonese model) or
# ``--align_split_gap``. Unguarded, a break at 1.0 s would invent a cue boundary about as often
# as it found one (45% of the cues flagged there are within half a second of the reference);
# the guard below is what makes 1.0 s safe, and this stays the conservative choice for a
# caller with no evidence to give it.
SPLIT_INTERNAL_GAP = 1.5

# What split_gapped_cues needs before it believes a gap. The failure behind the 45% above is
# a character placed where it was not said, and the model says so: the misplaced side's
# characters were not heard there (``peak``). So a side whose Chinese characters have a median
# peak under SPLIT_MIN_PEAK (realign's LINE_SPLIT_MIN_PEAK, for the same question) blocks the
# split -- unless at least SPLIT_MIN_SILENCE of the gap is silence against its region's own
# noise floor, which says the characters cover all the speech there is and the gap is a real
# pause however unsure the model was of the words. That second route still needs one side
# not to be unheard: with both unheard, the "pause" is as likely to sit between two misplaced
# tokens. A side with no Chinese (code-switched English, which this model barely hears) is no
# evidence either way.
SPLIT_MIN_PEAK = 0.5
SPLIT_MIN_SILENCE = 0.8

# A frame is silent this far below its region's speech level (or within FLOOR_MARGIN_DB of
# the floor, whichever is higher). The floor alone does not work on isolated vocals: their
# floor is digital silence near -185 dBFS, which nothing in a real pause reaches. Measured on
# three files: real pauses sit 95-160 dB under speech; a held syllable or speech that alignment
# left uncovered sits 0-30 dB under it.
SILENCE_BELOW_SPEECH_DB = 40.0

# extend_to_sound: a cue whose last syllable is still sounding when it ends is extended to
# where that sound stops. CTC gives a character only the frames it fires on, so a syllable held
# by a slow speaker (a learner's 表現, 見工) outlasts its span by up to a second, and the cue
# left the screen before the word was finished. A frame is sound within SOUND_BELOW_SPEECH_DB
# of its region's speech level; a dip of SOUND_HANGOVER or less inside a syllable does not end
# it; and the sound must stop within SOUND_MAX_EXTEND of the word, because a held syllable ends
# and music, effects or the next speaker do not. Measured (2026-10) on a slow single-speaker
# vocabulary video: cues still on speech at their end, 44 of 210 -> 5. On bluey and Doraemon
# against their references: starts and cue counts unchanged, ends early by more than 0.3 s
# 9.1/9.3% -> 7.6/6.4%, late by more than 0.5 s 19.7/17.4% -> 19.7/18.2%. Without the "must
# stop" rule the same extension made 21-29% of their ends late, from 30 to 40 dB alike.
SOUND_MAX_EXTEND = 1.5
SOUND_HANGOVER = 0.12
SOUND_BELOW_SPEECH_DB = 40.0


@dataclass(frozen=True)
class GappedCue:
    """A cue holding a silence between two of its own adjacent characters."""

    index: int
    gap: float
    at: float
    before: str
    after: str


class InternalGap(NamedTuple):
    """One silence inside a cue, carrying indices into that cue's own ``words`` list.

    ``before_idx``/``after_idx`` are positions in the *full* words list, not in the timed
    subset the scan walks, so a caller can cut the cue at one without having to re-derive
    where the untimed characters in between belong.
    """

    before_idx: int
    after_idx: int
    gap: float
    at: float
    before: str
    after: str


def _internal_gaps(
    segment: Mapping, max_gap: float, split: set,
) -> Iterator[InternalGap]:
    """Every silence over ``max_gap`` between two adjacent *timed* characters of one cue.

    A character the align model has no token for carries no timing at all (see
    ``align_vocab``), so it is transparent here rather than gap-producing: the question is
    whether its two timed neighbours were said together, and an untimed character between
    them is not evidence either way.
    """
    timed = [(i, w) for i, w in enumerate(segment.get("words") or [])
             if w.get("start") is not None and w.get("end") is not None]
    for (i, before), (j, after) in zip(timed, timed[1:]):
        if before["word"] in split or after["word"] in split:
            continue
        gap = float(after["start"]) - float(before["end"])
        if gap > max_gap:
            yield InternalGap(i, j, round(gap, 3), round(float(before["end"]), 3),
                              before["word"], after["word"])


def find_gapped_cues(
    segments: Sequence[Mapping],
    max_gap: float = MAX_INTERNAL_GAP,
    split_chars: Iterable[str] = (),
) -> List[GappedCue]:
    """Cues whose characters are not one continuous utterance.

    CTC must place every token it is given, so when a cue's text contains something that was
    not said where the cue sits, the extra characters are put wherever scores least badly --
    typically seconds away, leaving a hole in the middle of the cue. On `experiments/fixtures/bluey`, ASR
    emitted 爸爸， once for what the reference has as two separate calls: the first 爸 landed
    at 56.44 s and the second at 59.80 s, so the subtitle went on screen **3.35 s** before
    anybody spoke (the reference cue starts at 59.790).

    Only *directly adjacent* characters are compared, which is what makes this safe to leave
    on. A punctuation mark is mapped to blank precisely so it can absorb a pause, and it does
    so by *spanning* it -- the mark is contiguous with both neighbours, so an ordinary clause
    break produces no gap at all. ``split_chars`` covers only the residual case where the mark
    itself sits inside the silence rather than filling it. Measured on the two ASR fixtures,
    every gap found is between two real characters with nothing between them.

    **This is a report, not a repair, and that is deliberate.** 45% of the cues it flags are
    still within half a second of the reference, so trimming or splitting on it would hurt
    about as often as it helped; and which side of the gap to keep has no general answer
    (realign's ``_trim_detached_edges`` resolves that tie towards the opening, which is right
    for a whole transcript line and wrong for a two-character cue). What it is good at is
    saying *which* cues to distrust, and it is much better at that than the CTC score:

    | cues | n | median start error | within 0.5 s |
    |---|---|---|---|
    | with a gap over 1 s | 12 | 1.25-3.35 s | 0-45% |
    | without | 545 | 0.05-0.09 s | 76-79% |
    """
    split = set(split_chars)
    hits: List[GappedCue] = []
    for index, segment in enumerate(segments):
        worst = max(_internal_gaps(segment, max_gap, split),
                    key=lambda g: g.gap, default=None)
        if worst is not None:
            hits.append(GappedCue(index, worst.gap, worst.at, worst.before, worst.after))
    return hits


def warn_on_gapped_cues(
    segments: Sequence[Mapping],
    max_gap: float = MAX_INTERNAL_GAP,
    split_chars: Iterable[str] = (),
) -> List[GappedCue]:
    """find_gapped_cues, with a line per hit at debug and a count at info."""
    hits = find_gapped_cues(segments, max_gap, split_chars)
    for hit in hits:
        logger.debug(
            "Cue %d holds %.2fs of silence between %r and %r at %.3fs: %r",
            hit.index, hit.gap, hit.before, hit.after, hit.at,
            str(segments[hit.index].get("text", ""))[:40],
        )
    if hits:
        logger.info(
            "%d cue(s) contain a silence over %.1fs between two of their own characters; "
            "their timings are the least trustworthy in the file (see notes.srt)",
            len(hits), max_gap,
        )
    return hits


def _word_offsets(
    text: str, words: Sequence[Mapping],
) -> Optional[List[Tuple[int, int]]]:
    """Where each word sits in the cue's own text, as (start, end) offsets.

    ``None`` if the two disagree — a word that cannot be found where it should be means the
    text was rewritten after alignment, and a caller that cannot locate a word cannot cut
    beside it either. Refusing is the only safe answer: a wrong offset would split the text
    at one place and the characters at another.
    """
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for word in words:
        token = str(word.get("word", ""))
        if not token:
            spans.append((cursor, cursor))
            continue
        at = text.find(token, cursor)
        if at < 0:
            return None
        spans.append((at, at + len(token)))
        cursor = at + len(token)
    return spans


def _split_one(segment: Mapping, gaps: Sequence[InternalGap],
               kind: str = "split_gap") -> Optional[List[dict]]:
    """Cut one cue at each of ``gaps``. ``None`` if it cannot be done safely.

    ``kind`` names the reason in each piece's note (``split_gap``: a silence;
    ``segmentation`` also cuts over-long cues here, as ``split_long``).
    """
    text = str(segment.get("text", ""))
    words = list(segment.get("words") or [])
    spans = _word_offsets(text, words)
    if spans is None:
        logger.debug("Not splitting cue %r: its words do not match its text", text[:40])
        return None
    # chars is 1:1 with text (every character of the cue, spaces included), so the same
    # offsets cut both. Anything else and the cue is left whole rather than desynchronised.
    chars = segment.get("chars")
    chars = list(chars) if chars is not None and len(chars) == len(text) else None

    pieces: List[dict] = []
    word_from, text_from = 0, 0
    start = float(segment["start"])

    def piece(word_to: Optional[int], text_to: Optional[int], end: float) -> dict:
        out = dict(segment)
        out["text"] = text[text_from:text_to]
        out["words"] = words[word_from:word_to]
        if chars is not None:
            out["chars"] = chars[text_from:text_to]
        out["start"] = round(start, 3)
        out["end"] = round(end, 3)
        out["notes"] = list(segment.get("notes") or [])
        return out

    for gap in gaps:
        # Untimed characters between the two neighbours go with the head: they were never
        # placed, so there is no evidence they belong to the clause after the silence.
        head = piece(gap.after_idx, spans[gap.after_idx - 1][1],
                     float(words[gap.before_idx]["end"]))
        if not head["text"].strip():
            return None
        add_note(head, f"{kind}:{gap.gap:.1f}s after {gap.before}")
        pieces.append(head)
        word_from, text_from = gap.after_idx, spans[gap.after_idx][0]
        start = float(words[gap.after_idx]["start"])

    tail = piece(None, None, float(segment["end"]))
    if not tail["text"].strip() or tail["end"] < tail["start"]:
        return None
    add_note(tail, f"{kind}:{gaps[-1].gap:.1f}s before {gaps[-1].after}")
    pieces.append(tail)
    return pieces


class _SilenceMap:
    """Which stretches of a file are silent, each region judged against its own levels.

    Like ``find_silent_starts``, but a frame also counts as silent SILENCE_BELOW_SPEECH_DB
    under the region's speech level. Time outside every region is silent (VAD found no speech
    there); a region with no usable floor (a music bed) is never silent, since nothing there
    can be told apart from speech.
    """

    def __init__(self, regions: Iterable[Mapping], sample_rate: int = SAMPLE_RATE,
                 frame_seconds: float = FRAME_SECONDS) -> None:
        self.frame_seconds = frame_seconds
        self.regions: List[Tuple[float, float, Optional[np.ndarray]]] = []
        self.levels: List[Tuple[np.ndarray, float, float]] = []
        for region in regions:
            audio = region.get("audio")
            if audio is None:
                continue
            db = frame_dbfs(audio, sample_rate, frame_seconds)
            mask = None
            floor = speech = float("nan")
            if db.size:
                floor, speech = np.percentile(db, [5, 95])
                if speech - floor >= MIN_SEPARATION_DB:
                    mask = db < max(floor + FLOOR_MARGIN_DB, speech - SILENCE_BELOW_SPEECH_DB)
            self.regions.append((float(region["start"]), float(region["end"]), mask))
            self.levels.append((db, float(floor), float(speech)))
        order = sorted(range(len(self.regions)), key=lambda i: self.regions[i][0])
        self.regions = [self.regions[i] for i in order]
        self.levels = [self.levels[i] for i in order]

    def describe(self, start: float, end: float) -> str:
        """The gap's median level and its region's floor and speech level, for the log."""
        mid = (start + end) / 2
        for (r0, r1, _), (db, floor, speech) in zip(self.regions, self.levels):
            if r0 <= mid < r1:
                frames = db[int((start - r0) / self.frame_seconds):int((end - r0) / self.frame_seconds)]
                level = float(np.median(frames)) if frames.size else float("nan")
                return f"{level:.0f} dBFS, floor {floor:.0f}, speech {speech:.0f}"
        return "outside every region"

    def _sound(self, t: float) -> Optional[bool]:
        """Whether the frame at *t* is within SOUND_BELOW_SPEECH_DB of its region's speech
        level; None outside every region or in one with no usable floor."""
        for (r0, r1, mask), (db, _floor, speech) in zip(self.regions, self.levels):
            if r0 <= t < r1:
                idx = int((t - r0) / self.frame_seconds)
                if mask is None or not 0 <= idx < db.size:
                    return None
                return bool(db[idx] > speech - SOUND_BELOW_SPEECH_DB)
        return None

    def sound_until(self, t: float, limit: float,
                    hangover: float = SOUND_HANGOVER) -> Optional[float]:
        """Where the sound running at *t* stops: the end of its last frame before a silence
        longer than ``hangover``. *t* itself if *t* is not on sound. None if it does not stop
        before *limit*, or runs where sound cannot be judged (no region, or a region with no
        usable floor): sound that will not stop is not one syllable being held."""
        last, quiet = t, 0.0
        step = self.frame_seconds
        while t < limit:
            found = self._sound(t)
            if found is None:
                return None
            if not found:
                quiet += step
                if quiet > hangover:
                    return last
            else:
                quiet = 0.0
                last = t + step
            t += step
        return None

    def share(self, start: float, end: float) -> Optional[float]:
        """The share of [start, end) that is silent; None with no audio to judge by."""
        if not self.regions or end <= start:
            return None
        times = np.arange(start, end, self.frame_seconds) + self.frame_seconds / 2
        silent = 0
        for t in times:
            region = next((r for r in self.regions if r[0] <= t < r[1]), None)
            if region is None:
                silent += 1
                continue
            mask = region[2]
            idx = int((t - region[0]) / self.frame_seconds)
            if mask is not None and 0 <= idx < mask.size and mask[idx]:
                silent += 1
        return silent / len(times)


def _heard(words: Sequence[Mapping]) -> Optional[bool]:
    """Whether the model heard these words where it put them: the median ``peak`` of their
    Chinese characters, or None where there are none (English, digits) or no peaks at all."""
    from cantocaptions_ai.text_profiles import display_columns

    def chinese(word: str) -> bool:
        return bool(word) and all(display_columns(c) == 2
                                  and not unicodedata.category(c).startswith(("P", "S"))
                                  for c in word)

    peaks = [float(w["peak"]) for w in words
             if w.get("peak") is not None and chinese(str(w.get("word", "")).strip())]
    if not peaks:
        return None
    return float(np.median(peaks)) >= SPLIT_MIN_PEAK


def split_gapped_cues(
    segments: Sequence[Mapping],
    min_gap: float = SPLIT_INTERNAL_GAP,
    split_chars: Iterable[str] = (),
    regions: Optional[Iterable[Mapping]] = None,
    script=None,
    release: float = 0.0,
    padding: float = 0.0,
) -> Tuple[List[dict], int]:
    """Break every cue holding a believable silence of ``min_gap`` or more into one cue per
    utterance.

    The one place in this module that *repairs* rather than reports — because breaking a cue
    is a bet, where saying "distrust this one" is not. See ``find_gapped_cues`` for the
    measurement: at the 1 s reporting threshold, 45% of the cues flagged are still within
    half a second of the reference. What those have in common is a character placed where it
    was not said, so a gap is only believed when the evidence says the placement is sound:
    neither side's Chinese characters went unheard (``SPLIT_MIN_PEAK``), or the gap is
    silence in the audio (``SPLIT_MIN_SILENCE`` of it, judged against ``regions`` — VAD
    segments or one whole-file region carrying ``audio``). Without ``regions`` only the
    first test can pass. Each half's own timings are then the ones alignment actually
    measured — a cue spanning the silence has a start or an end that was never spoken.
    A gap inside a word of ``script``'s (``ScriptConfig.inside_word``: 樂觀, or an English
    word) is never cut, however long; it stays a ``find_gapped_cues`` report. With
    ``release``, each head is then released into the pause the way alignment releases a cue
    ending on punctuation, stopping ``padding`` short of the piece after it.

    The cut is exactly the silence: the head keeps the cue's own start and ends on its last
    timed character before the gap, the tail begins on its first timed character after it
    and keeps the cue's own end. Nothing is retimed and nothing outside the cue moves, so a
    caller's release/trim pass and the non-overlap guarantee both survive untouched.

    Returns the new cue list (segments are copied, never mutated) and how many breaks were
    made. A cue whose words cannot be located in its own text is passed through whole.
    """
    split = set(split_chars)
    silence = _SilenceMap(regions) if regions is not None else None
    out: List[dict] = []
    breaks = 0
    for segment in segments:
        words = list(segment.get("words") or [])
        candidates = list(_internal_gaps(segment, min_gap, split))
        if script is not None and candidates:
            text = str(segment.get("text", ""))
            offsets = _word_offsets(text, words)
            if offsets is not None:
                candidates = [g for g in candidates
                              if not script.inside_word(text, offsets[g.after_idx][0])]
        gaps = []
        for k, gap in enumerate(candidates):
            # Each side is the run of words up to the next candidate gap: the piece this cut
            # would actually make.
            lo = candidates[k - 1].after_idx if k else 0
            hi = candidates[k + 1].after_idx if k + 1 < len(candidates) else len(words)
            head, tail = _heard(words[lo:gap.after_idx]), _heard(words[gap.after_idx:hi])
            quiet = silence.share(gap.at, gap.at + gap.gap) if silence is not None else None
            believed = (head is not False and tail is not False) or (
                quiet is not None and quiet >= SPLIT_MIN_SILENCE
                and (head is not False or tail is not False))
            logger.debug(
                "Gap of %.2fs at %.3fs between %r and %r: heard %s/%s, %s silent (%s) -> %s",
                gap.gap, gap.at, gap.before, gap.after, head, tail,
                "?" if quiet is None else f"{quiet:.0%}",
                silence.describe(gap.at, gap.at + gap.gap) if silence is not None else "no audio",
                "split" if believed else "kept",
            )
            if believed:
                gaps.append(gap)
        pieces = _split_one(segment, gaps) if gaps else None
        if pieces is None:
            out.append(dict(segment))
            continue
        if release:
            for head, tail in zip(pieces, pieces[1:]):
                head["end"] = round(max(head["end"], min(head["end"] + release,
                                                         tail["start"] - padding)), 3)
        breaks += len(pieces) - 1
        out.extend(pieces)
    return out, breaks


def extend_to_sound(
    segments: Sequence[dict],
    regions: Iterable[Mapping],
    padding: float = 0.0,
    max_extend: float = SOUND_MAX_EXTEND,
    file_end: Optional[float] = None,
    split_chars: Iterable[str] = (),
) -> int:
    """Extend each cue whose last syllable is still sounding when it ends (in place).

    The question is asked of the cue's last spoken word, not of its end: the sound must run
    unbroken from that word's aligned end past the cue's end. A cue whose end was already
    released past a silence -- the speaker stopped, and whatever sounds after that is music, an
    effect or someone else -- is left alone. The new end is where that sound stops, no more than
    ``max_extend`` after the word, and at least two ``padding`` short of the next cue, so an
    extension never brings two cues close enough for cue assembly to join them. Nothing moves
    where ``regions`` cannot tell silence from sound (see ``_SilenceMap``). Returns how many
    ends moved.
    """
    silence = regions if isinstance(regions, _SilenceMap) else _SilenceMap(regions)
    split = set(split_chars)
    moved = 0
    for k, seg in enumerate(segments):
        end = float(seg["end"])
        spoken = [w for w in seg.get("words") or []
                  if w.get("end") is not None and str(w.get("word", "")).strip()
                  and str(w.get("word", "")).strip() not in split]
        if not spoken:
            continue
        last = float(spoken[-1]["end"])
        nxt = segments[k + 1]["start"] if k + 1 < len(segments) else file_end
        limit = last + max_extend
        if nxt is not None:
            limit = min(limit, float(nxt) - 2 * padding)
        if limit <= end:
            continue
        sound_end = silence.sound_until(last, limit)
        if sound_end is not None and round(sound_end, 3) > end + 1e-3:
            sound_end = round(sound_end, 3)
            seg["end"] = sound_end
            moved += 1
    return moved


@dataclass(frozen=True)
class SilentStart:
    """One cue whose start time landed on silence."""

    time: float
    """Absolute start time of the cue, in seconds."""
    text: str
    gap: float
    """Seconds from that start to the nearest audio energy."""
    dbfs: float
    floor_dbfs: float
    index: int = -1
    """Position of the cue in the list that was checked, so a caller can annotate it."""


def frame_dbfs(
    audio: np.ndarray, sample_rate: int = SAMPLE_RATE, frame_seconds: float = FRAME_SECONDS
) -> np.ndarray:
    """RMS level per fixed-length frame, in dBFS.

    Frames are exactly ``frame_seconds`` long and start at the audio's first sample, so a
    time offset maps to a frame index by plain division — no windowing or hop bookkeeping.
    """
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    hop = max(int(round(frame_seconds * sample_rate)), 1)
    n = len(audio) // hop
    if n == 0:
        return np.empty(0)
    frames = audio[: n * hop].reshape(n, hop)
    return 20.0 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-12)


def _gap_to_energy(silent: np.ndarray) -> np.ndarray:
    """Frames from each position to the nearest non-silent frame, in either direction."""
    n = len(silent)
    far = n + 1
    gaps = np.full(n, far, dtype=np.int64)

    last = -far
    for t in range(n):
        if not silent[t]:
            last = t
        gaps[t] = t - last

    nxt = far
    for t in range(n - 1, -1, -1):
        if not silent[t]:
            nxt = t
        gaps[t] = min(gaps[t], nxt - t)
    return gaps


def find_silent_starts(
    aligned_segments: Sequence[Mapping],
    regions: Iterable[Mapping],
    sample_rate: int = SAMPLE_RATE,
    frame_seconds: float = FRAME_SECONDS,
    floor_margin_db: float = FLOOR_MARGIN_DB,
    min_separation_db: float = MIN_SEPARATION_DB,
    min_gap_frames: int = MIN_GAP_FRAMES,
) -> List[SilentStart]:
    """Cues whose start sits on near-silent audio, far from any sound.

    ``regions`` are dicts carrying ``start``/``end``/``audio`` — VAD segments as the
    pipeline builds them, or a single whole-file region. Each region establishes its own
    noise floor, since level varies widely between them.
    """
    hits: List[SilentStart] = []
    starts = [(i, seg.get("start")) for i, seg in enumerate(aligned_segments)]

    for region in regions:
        audio = region.get("audio")
        if audio is None:
            continue
        db = frame_dbfs(audio, sample_rate, frame_seconds)
        if db.size == 0:
            continue

        floor, speech = np.percentile(db, [5, 95])
        if speech - floor < min_separation_db:
            continue

        silent = db < floor + floor_margin_db
        gaps = _gap_to_energy(silent)
        r_start, r_end = region["start"], region["end"]

        for i, t in starts:
            if t is None or not (r_start <= t < r_end):
                continue
            idx = int((t - r_start) / frame_seconds)
            if not (0 <= idx < db.size) or not silent[idx] or gaps[idx] < min_gap_frames:
                continue
            hits.append(SilentStart(
                time=float(t),
                text=str(aligned_segments[i].get("text", "")),
                index=i,
                gap=float(gaps[idx] * frame_seconds),
                dbfs=float(db[idx]),
                floor_dbfs=float(floor),
            ))

    hits.sort(key=lambda h: h.time)
    return hits


def warn_on_silent_starts(
    aligned_segments: Sequence[Mapping],
    regions: Iterable[Mapping],
    max_examples: int = 3,
    **kwargs,
) -> List[SilentStart]:
    """Run ``find_silent_starts`` and log what it found. Returns the hits."""
    hits = find_silent_starts(aligned_segments, regions, **kwargs)
    if not hits:
        return hits

    from cantocaptions_ai.utils.output import format_timestamp

    examples = [
        f"  silent cue start at "
        f"{format_timestamp(hit.time, always_include_hours=True, decimal_marker=',')}: "
        f"{hit.gap:.2f}s of silence before {hit.text[:30]!r} "
        f"({hit.dbfs:.0f} dBFS, floor {hit.floor_dbfs:.0f})"
        for hit in hits[:max_examples]
    ]
    if len(hits) > max_examples:
        examples.append(f"  ...and {len(hits) - max_examples} more")
    logger.warning(
        "Alignment placed %d cue %s on silence — the audio there is at the segment's noise "
        "floor with no sound within %.2fs. Timings for %s are not trustworthy.\n%s",
        len(hits), "start" if len(hits) == 1 else "starts",
        max(h.gap for h in hits), "it" if len(hits) == 1 else "them", "\n".join(examples),
    )
    return hits


def region_levels(
    regions: Iterable[Mapping],
    sample_rate: int = SAMPLE_RATE,
    frame_seconds: float = FRAME_SECONDS,
) -> List[Tuple[float, np.ndarray]]:
    """Per-frame dBFS for each region, as (region start, levels), sorted by start.

    Levels only -- no thresholds and no verdict. Callers decide what quiet means, because the
    right comparison depends on the question.

    Be careful what you ask of it. Trimming a cue back off "silence" by level was tried and
    does not work on real material: inside the Police Story 2 cues that sit through a pause,
    the pause is continuous room tone about 12 dB under the dialogue and the first frame above
    any sane threshold is the cue's own first frame. What perceptually reads as silence is not
    silence to an envelope. The dwell those cues came from is visible in the *emission*
    instead -- see _align_segment's MAX_CHAR_DWELL_SECONDS.
    """
    out = []
    for region in regions:
        audio = region.get("audio")
        if audio is None:
            continue
        db = frame_dbfs(audio, sample_rate, frame_seconds)
        if db.size:
            out.append((float(region["start"]), db))
    out.sort(key=lambda item: item[0])
    return out


def whole_file_region(audio, duration: float) -> List[dict]:
    """Wrap a whole-file waveform as a single region for ``find_silent_starts``.

    Used when ``align()`` was handed a waveform instead of VAD segments; accepts the numpy
    array or torch tensor, mono or (1, N), that ``align()`` may be holding at that point.
    """
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = audio.reshape(audio.shape[0], -1)[0]
    return [{"start": 0.0, "end": duration, "audio": audio}]

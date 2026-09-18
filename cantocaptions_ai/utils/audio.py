import json
import os
import subprocess
from functools import lru_cache
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from cantocaptions_ai.utils.log_utils import get_logger
from cantocaptions_ai.utils.output import exact_div

logger = get_logger(__name__)

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token

def resolve_device(device: str, device_index: int = 0) -> str:
    """Return a torch-compatible device string (e.g. 'cuda:0', 'cpu')."""
    return f"cuda:{device_index}" if device == "cuda" else device


def probe_audio_tracks(file: str) -> List[dict]:
    """Return ffprobe metadata for all audio streams in *file*.

    Returns an empty list if the file has no audio streams, ffprobe cannot read
    the file, or ffprobe is unavailable. Raises RuntimeError if ffprobe is not
    found on PATH (i.e. ffmpeg is not installed).
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "a",
        file,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=False)
    except FileNotFoundError:
        raise RuntimeError("ffprobe not found; ensure ffmpeg is installed and on PATH")
    try:
        data = json.loads(result.stdout)
        return data.get("streams", [])
    except (json.JSONDecodeError, KeyError):
        return []


# Language codes and title keywords that identify a Chinese-language audio
# track (Cantonese, Mandarin, or an unspecified Chinese variant). Used as a
# fallback when no explicit Cantonese track is present, so a clearly-Chinese
# stream is preferred over ffmpeg's default (which is often English/Japanese).
_CHINESE_LANG_CODES = {
    "yue", "zh", "zho", "chi", "cmn", "nan", "hak", "wuu",
    "zh-hans", "zh-hant", "zh-hk", "zh-tw", "zh-cn", "zh-sg",
}
_CHINESE_TITLE_KEYWORDS = (
    "chinese", "cantonese", "mandarin", "putonghua", "guoyu", "huayu",
    "中文", "汉语", "漢語", "华语", "華語", "国语", "國語",
    "普通话", "普通話", "粤", "粵", "粵語", "粤语", "廣東話", "广东话",
)


def _is_chinese_track(stream: dict) -> bool:
    tags = stream.get("tags", {})
    lang = (tags.get("language") or "").lower()
    if lang in _CHINESE_LANG_CODES:
        return True
    title = (tags.get("title") or "").lower()
    return any(k in title for k in _CHINESE_TITLE_KEYWORDS)


def select_cantonese_track(streams: List[dict]) -> int:
    """Return the 0-based audio stream index for the best Chinese audio track.

    Preference order:
      1. An explicit Cantonese track (``language == "yue"`` or a title
         containing ``"cantonese"``).
      2. Otherwise the first Chinese track of any kind (Mandarin / generic
         ``zh`` / a Chinese-language title) so a clear Chinese option is used
         instead of falling through to whatever ffmpeg's default heuristic
         picks (frequently an English or Japanese dub).
      3. If no Chinese audio is present at all, return 0 and let ffmpeg choose.
    """
    for i, stream in enumerate(streams):
        tags = stream.get("tags", {})
        if tags.get("language") == "yue":
            return i
        if "cantonese" in tags.get("title", "").lower():
            return i
    for i, stream in enumerate(streams):
        if _is_chinese_track(stream):
            return i
    return 0


# Channel ORDER for every ffmpeg layout this module knows how to reduce to mono.
#
# Order, not just membership: the downmix below weights channels by role, and a
# `pan` expression has to address them somehow. It addresses them by INDEX,
# because the names are not portable between layouts that are otherwise
# interchangeable here -- 5.1 calls its rear pair BL/BR while 5.1(side) calls
# the same pair SL/SR, so an expression naming SL fails outright on a 5.1 file.
#
# Deriving _LAYOUTS_WITH_CENTER from this table also fixes a quiet bug: the old
# hand-written set listed 3.0(back) (FL FR BC), 6.0(front) and 6.1(front)
# (FLC/FRC, no centre) as having an FC, so `--audio_downmix center` built
# `pan=mono|c0=FC` for them and ffmpeg rejected it partway into the decode.
_LAYOUT_CHANNELS = {
    "mono": ("FC",),
    "3.0": ("FL", "FR", "FC"),
    "3.0(back)": ("FL", "FR", "BC"),
    "4.0": ("FL", "FR", "FC", "BC"),
    "5.0": ("FL", "FR", "FC", "BL", "BR"),
    "5.0(side)": ("FL", "FR", "FC", "SL", "SR"),
    "5.1": ("FL", "FR", "FC", "LFE", "BL", "BR"),
    "5.1(side)": ("FL", "FR", "FC", "LFE", "SL", "SR"),
    "6.0": ("FL", "FR", "FC", "BC", "SL", "SR"),
    "6.0(front)": ("FL", "FR", "FLC", "FRC", "SL", "SR"),
    "6.1": ("FL", "FR", "FC", "LFE", "BC", "SL", "SR"),
    "6.1(back)": ("FL", "FR", "FC", "LFE", "BL", "BR", "BC"),
    "6.1(front)": ("FL", "FR", "FLC", "FRC", "LFE", "SL", "SR"),
    "7.0": ("FL", "FR", "FC", "BL", "BR", "SL", "SR"),
    "7.0(front)": ("FL", "FR", "FC", "FLC", "FRC", "SL", "SR"),
    "7.1": ("FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"),
    "7.1(wide)": ("FL", "FR", "FC", "LFE", "BL", "BR", "FLC", "FRC"),
    "7.1(wide-side)": ("FL", "FR", "FC", "LFE", "FLC", "FRC", "SL", "SR"),
    "hexagonal": ("FL", "FR", "FC", "BL", "BR", "BC"),
    "octagonal": ("FL", "FR", "FC", "BL", "BR", "BC", "SL", "SR"),
}

# ffmpeg channel layouts that carry a discrete front-center channel. On a mixed
# soundtrack FC is the dialogue stem, so extracting it alone is close to free vocal
# isolation -- the reason --audio_downmix center exists. Layouts absent from this set
# (stereo, 2.1, quad) have no FC to take, and an unknown layout is not guessed at.
_LAYOUTS_WITH_CENTER = frozenset(
    layout for layout, channels in _LAYOUT_CHANNELS.items() if "FC" in channels)

_CHANNEL_ROLE = {"FC": "center", "FL": "front", "FR": "front",
                 "FLC": "front", "FRC": "front", "LFE": "lfe"}

# Weights for the multichannel -> mono downmix, by channel role.
#
# NOT ffmpeg's defaults, and deliberately so. Two things are wrong with letting
# swresample do this:
#
#   1. Gain. swr resolves `rematrix_maxval` to 1.0 for INTEGER output and to
#      unlimited for float. We decode to s16le, so it lands on the integer path
#      and scales the whole matrix down by its coefficient sum (~3.41 for 5.1),
#      costing every multichannel source ~9.4 dB for no reason but the output
#      sample format. Stereo never noticed: (L+R)/2 is already unity for centred
#      dialogue, which is why this hid for so long.
#
#   2. Balance. The conventional matrix puts surrounds 6 dB under the centre.
#      Background and off-screen speech is exactly the content panned off-centre,
#      so that spread is the wrong one for a transcription corpus -- it buries
#      the lines that are already hardest to hear. These weights narrow it to
#      3 dB.
#
# The centre sits BELOW the fronts (0.75 vs 0.53 each, so FL+FR together exceed
# it) which also buys headroom, because film peaks are centre-driven. Measured
# over all 55 multichannel sources in the corpus: 1 exceeded 0 dBFS, by 0.02 dB.
# With LFE included instead of dropped it was 2, by up to 1.03 dB -- the LFE term
# was the clipping, and it carries nothing above 120 Hz that speech needs.
_DOWNMIX_WEIGHTS = {"center": 0.75, "front": 0.53, "surround": 0.53, "lfe": 0.0}


def _mono_pan_filter(layout: str) -> Optional[str]:
    """A `pan` expression summing `layout` to mono, or None if it is not known.

    Channels weighted at zero are omitted rather than written as `0*cN`: the
    expression turns up in logs and in the dataset's downmix report, and "LFE is
    not in this mix" reads better as an absence than as arithmetic.
    """
    channels = _LAYOUT_CHANNELS.get(layout)
    if not channels:
        return None
    terms = []
    for index, name in enumerate(channels):
        weight = _DOWNMIX_WEIGHTS[_CHANNEL_ROLE.get(name, "surround")]
        if weight > 0:
            terms.append(f"{weight:g}*c{index}")
    return "pan=mono|c0=" + "+".join(terms) if terms else None


def _downmix_ffmpeg_args(file: str, audio_track: int, downmix: str) -> list:
    """Return the ffmpeg filter args implementing *downmix*, or [] for a plain downmix.

    ``center`` isolates the front-center channel. It is applied only when ffprobe reports
    a layout known to have one; anything else (stereo, an unnamed layout) falls back to
    the ordinary all-channel downmix with a warning rather than risking a filter error
    partway through decoding a feature-length file.

    ``mix`` states its matrix explicitly for any multichannel layout in
    ``_LAYOUT_CHANNELS`` (see ``_DOWNMIX_WEIGHTS`` for why it is not swresample's).
    Stereo and mono return no filter at all: ffmpeg's stereo downmix is already
    (L+R)/2, which is unity for centred dialogue and has nothing wrong with it.
    An unrecognised multichannel layout also returns no filter -- that leaves it
    on the old, quiet path, but a WRONG pan expression would be worse than a
    quiet one, and the warning says which layout to add to the table.
    """
    if downmix not in ("mix", "center"):
        raise ValueError(f"Unknown downmix mode: {downmix!r} (expected 'mix' or 'center')")

    streams = probe_audio_tracks(file)
    stream = streams[audio_track] if audio_track < len(streams) else None
    layout = (stream or {}).get("channel_layout", "")
    channels = int((stream or {}).get("channels", 0) or 0)

    if layout == "mono" or channels == 1:
        return []  # already one channel; nothing to combine or extract

    if downmix == "center":
        if layout in _LAYOUTS_WITH_CENTER:
            return ["-af", "pan=mono|c0=FC"]
        logger.warning(
            "--audio_downmix center: track %d has layout %r (%d channel(s)) with no front-center "
            "channel to extract; falling back to a full downmix.",
            audio_track, layout or "unknown", channels,
        )
        return []

    if channels <= 2:
        return []  # stereo: (L+R)/2 is already correct
    pan = _mono_pan_filter(layout)
    if pan:
        return ["-af", pan]
    logger.warning(
        "downmix: track %d has %d channels in unrecognised layout %r; falling back to "
        "ffmpeg's normalised matrix, which is roughly 9 dB quiet. Add the layout to "
        "_LAYOUT_CHANNELS to fix it.",
        audio_track, channels, layout or "unknown",
    )
    return []


# --- level normalisation -----------------------------------------------------
#
# Training and inference MUST measure level the same way or normalising is worse
# than not bothering: the model would learn one level distribution and meet
# another. That is the whole reason this lives here, in the decoder both sides
# call, rather than in the dataset repo where the corpus statistics were worked
# out. `cantocaptions_dataset.audio_cut` imports these constants so its staleness
# stamp invalidates the cut tree when they change.

#: Level a normalised file's speech is brought to, in dBFS.
NORMALIZE_TARGET_DBFS = -24.0

#: Never lift past this true peak, whatever the target asks for.
NORMALIZE_CEILING_DBFS = -1.0

#: Gating for the level estimate. A block below `_GATE_ABSOLUTE` is silence; one
#: more than `_GATE_RELATIVE` under the mean of the rest is background.
#:
#: The relative gate is deliberately far wider than EBU R128's 10 LU. R128 is
#: built to answer "how loud does this programme feel", so it gates hard toward
#: the loud content and reports that. The question here is the opposite one --
#: "how loud is the dialogue" -- and on a film the loud content is the score and
#: the action. Measured against per-segment ground truth over 26 episodes, a
#: 10 dB gate scores sd 1.94 dB and 30 dB scores 1.49; no relative gate at all
#: scores 1.44, so the gate costs almost nothing and buys protection against
#: material that is mostly ambience.
_BLOCK_S, _HOP_S = 0.400, 0.100
_GATE_ABSOLUTE, _GATE_RELATIVE = -60.0, 30.0


def gated_level(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Optional[float]:
    """Mean level in dBFS of the blocks that look like speech, or None.

    Gating is what makes this usable at inference, where there are no subtitle
    spans to measure over: silence and low-level ambience are excluded, so the
    result tracks the level of the loud, sustained content rather than the
    proportion of the file that happens to be quiet.

    The surviving blocks are reduced with a MEDIAN, not a mean, and that is the
    single choice that makes this work. Speech is the most COMMON audible thing
    in a film, but rarely the loudest; a mean is pulled upward by the score and
    the action, a median lands on the typical talking block. Measured against
    the dataset's per-segment ground truth over 26 episodes spanning its
    mastering styles:

        mean of gated blocks (EBU R128)   +2.49 dB offset, sd 2.74, worst 9.14
        median of gated blocks            -1.14 dB offset, sd 1.49, worst 3.93

    The mean's worst case is action cinema -- on Ip Man it read 11.6 dB HIGH,
    which would have left that film's dialogue 11.6 dB under target.

    The gate threshold is computed in the POWER domain even so. A threshold
    taken from the mean of the LEVELS sinks along with the background: on a file
    that is three quarters quiet it lands near the quiet part, nothing is gated
    out, and the estimate reports the silence.

    Known weakness: material that is mostly ambience within 30 dB of its speech
    will have that ambience as its median, reading LOW and so lifting the file
    too far. The peak ceiling in `normalize_gain` bounds the damage, and this
    corpus -- film and television, near-continuous soundtrack -- does not
    contain the case. The failure is bounded and rare where the mean's failure
    was unbounded and common, which is why this trade was taken.
    """
    block, hop = int(_BLOCK_S * sr), int(_HOP_S * sr)
    if audio.size < block:
        return None
    usable = audio.size - ((audio.size - block) % hop)
    blocks = np.lib.stride_tricks.sliding_window_view(
        audio[:usable].astype(np.float64), block)[::hop]
    power = (blocks ** 2).mean(axis=1)

    loud = power[10.0 * np.log10(power + 1e-20) > _GATE_ABSOLUTE]
    if loud.size == 0:
        return None                      # silence, or close enough to it
    threshold = 10.0 * np.log10(loud.mean() + 1e-20) - _GATE_RELATIVE
    speech = loud[10.0 * np.log10(loud + 1e-20) > threshold]
    kept = speech if speech.size else loud
    return float(10.0 * np.log10(np.median(kept) + 1e-20))


def normalize_gain(audio: np.ndarray, sr: int = SAMPLE_RATE,
                   target_db: Optional[float] = None,
                   ceiling_db: Optional[float] = None) -> float:
    """Gain in dB bringing `audio` to `target_db`, held under `ceiling_db` peak.

    A single linear gain, never compression: the recording's own dynamics are
    what make dialogue sound like dialogue, and the problem being solved is
    variation BETWEEN recordings, not within one.

    The targets default to the module constants, resolved HERE rather than in
    the signature. A default argument binds once at import, so spelling this
    `target_db: float = NORMALIZE_TARGET_DBFS` would freeze whatever was in
    scope when the module first loaded. Retuning the constant would then be
    picked up by the dataset's staleness stamp, mark the whole cut tree stale,
    re-cut all 401 episodes -- and write them at the OLD level regardless. The
    stamp would be describing audio that does not exist, which is worse than
    having no stamp at all.

    Returns 0.0 when there is nothing measurable, so a caller can apply it
    unconditionally.
    """
    target_db = NORMALIZE_TARGET_DBFS if target_db is None else target_db
    ceiling_db = NORMALIZE_CEILING_DBFS if ceiling_db is None else ceiling_db
    level = gated_level(audio, sr)
    if level is None:
        return 0.0
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak <= 0:
        return 0.0
    return min(target_db - level, ceiling_db - 20.0 * float(np.log10(peak)))


#: Announces that :func:`load_audio` puts sample 0 at the container's zero, so a
#: caller does not have to correct for a muxed audio delay itself -- and, more to
#: the point, so one that used to can stop and not double-correct. Read it with
#: ``getattr(audio, "HONOURS_CONTAINER_DELAY", False)``: a consumer pinned to an
#: older build of this package sees the absence and keeps its own correction.
HONOURS_CONTAINER_DELAY = True


def container_audio_delay(file: str, audio_track: int = 0) -> float:
    """Seconds of silence ffmpeg omits from the front of *file*'s audio track.

    A container may hold its audio later than its video -- Peppa Pig S1 puts the
    Cantonese dub at 1.000 s against video at 0.041 s. A player honours that and
    shows a subtitle at the time the subtitle says. ffmpeg decoding to a RAW
    format has no muxer to honour it and rebases the stream onto its own first
    packet, so the lead is dropped rather than padded and every later sample
    answers to a timestamp this much too small.

    The number is `audio_start_time - container_start_time`, and it is constant
    for the file -- this is a fixed mux offset, not drift, so nothing here has to
    track how far in we are. It is clamped at 0 because the container's start is
    the minimum over its streams, which leaves the audio unable to lead it.

    Returns 0.0 for anything unreadable rather than raising: a file ffprobe
    cannot parse has a much louder problem waiting one line later in the decode,
    and reporting it from here would blame the delay for it.
    """
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_streams", "-select_streams", "a", "-show_format", file]
    try:
        data = json.loads(subprocess.run(cmd, capture_output=True,
                                         check=False).stdout)
    except (OSError, json.JSONDecodeError):
        return 0.0

    streams = data.get("streams") or []
    if not streams:
        return 0.0
    stream = streams[audio_track] if audio_track < len(streams) else streams[0]

    def _start(obj) -> float:
        try:
            return float(obj.get("start_time"))
        except (TypeError, ValueError):
            return 0.0

    return max(_start(stream) - _start(data.get("format") or {}), 0.0)


def _clip_ffmpeg_args(audio_start: Optional[float], audio_end: Optional[float]) -> tuple:
    """Return (pre_input_args, post_input_args) implementing an [start, end) clip.

    ``-ss`` is placed before ``-i`` (fast, keyframe-accurate input seeking — plenty
    accurate for speech) and the clip length is bounded with ``-t``. Both bounds are
    optional; ``None`` means "from the very start" / "to the very end". Negative or
    inverted ranges raise ValueError so a bad request fails loudly rather than
    silently transcribing the whole file (the old no-op behavior).
    """
    if audio_start is not None and audio_start < 0:
        raise ValueError(f"audio_start must be >= 0, got {audio_start}")
    if audio_end is not None and audio_end < 0:
        raise ValueError(f"audio_end must be >= 0, got {audio_end}")
    if audio_start is not None and audio_end is not None and audio_end <= audio_start:
        raise ValueError(f"audio_end ({audio_end}) must be greater than audio_start ({audio_start})")

    pre: list = []
    post: list = []
    if audio_start is not None:
        pre += ["-ss", f"{audio_start:.3f}"]
    if audio_end is not None:
        duration = audio_end - (audio_start or 0.0)
        post += ["-t", f"{duration:.3f}"]
    return pre, post


def probe_duration_seconds(file: str) -> Optional[float]:
    """Return the media duration in seconds via ffprobe, or None if unavailable.

    Reads the container-level ``format.duration`` (present for essentially all real
    media). None means ffprobe could not determine a duration (e.g. a malformed or
    non-media file) — callers should treat that as "not a usable media file."
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        file,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=False)
    except FileNotFoundError:
        raise RuntimeError("ffprobe not found; ensure ffmpeg is installed and on PATH")
    try:
        data = json.loads(result.stdout)
        dur = data.get("format", {}).get("duration")
        return float(dur) if dur is not None else None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def validate_input_file(
    file: str,
    *,
    max_duration_s: Optional[float] = None,
    max_bytes: Optional[int] = None,
) -> float:
    """Fast pre-flight check that ``file`` is a usable media input; return its duration.

    Fails fast with :class:`InputError` — before any model loads — instead of the
    old behavior where the only signal was ffmpeg erroring partway into VAD. Checks:
    existence, non-empty, optional size cap, at least one readable audio stream, and
    a positive duration within an optional cap.
    """
    from cantocaptions_ai.errors import InputError

    if not os.path.isfile(file):
        raise InputError(f"File not found: {file}")
    size = os.path.getsize(file)
    if size == 0:
        raise InputError(f"File is empty: {file}")
    if max_bytes is not None and size > max_bytes:
        raise InputError(
            f"File is too large: {size} bytes exceeds the {max_bytes}-byte limit"
        )
    if not probe_audio_tracks(file):
        raise InputError(f"No readable audio stream found in {file}")
    duration = probe_duration_seconds(file)
    if duration is None or duration <= 0:
        raise InputError(f"Could not determine a positive audio duration for {file}")
    if max_duration_s is not None and duration > max_duration_s:
        raise InputError(
            f"Audio is too long: {duration:.1f}s exceeds the {max_duration_s:.0f}s limit"
        )
    return duration


def _restore_container_delay(audio: np.ndarray, file: str, audio_track: int,
                             audio_start: Optional[float],
                             audio_end: Optional[float], sr: int) -> np.ndarray:
    """Put `audio`'s sample 0 back at the container time the caller asked for.

    Both decode shapes need this and they need different amounts of it, which is
    why it is one function rather than a branch at each call site:

    * No ``-ss``. ffmpeg starts at the first audio packet, so the whole delay is
      missing and the whole delay is prepended.
    * ``-ss t`` with ``t`` at or past the delay. ffmpeg seeks on the container
      timeline and is already right; nothing is prepended. This is the ordinary
      case, and it is why the bug survived so long -- the clipped path, which is
      the one people reach for when a timing looks wrong, quietly disagreed with
      the unclipped one instead of being wrong alongside it.
    * ``-ss t`` with ``t`` inside the delay. There is no packet to seek to, so
      ffmpeg clamps to the first one and returns audio from `delay` while the
      caller believes it starts at `t`. The shortfall is `delay - t`.

    ``max`` covers all three: the first is `t = 0`, the second goes negative and
    clamps to nothing. The trim afterwards keeps the promise ``audio_end`` makes
    about LENGTH -- ffmpeg measured its ``-t`` from the clamp point, so padding
    the front without it would hand back a clip longer than the window asked
    for. It is skipped entirely when no padding happened, so a file with no
    delay decodes to the same samples, and the same count, it always did.
    """
    if not file or not os.path.exists(file):
        return audio                      # not a path we can probe; leave it be
    delay = container_audio_delay(file, audio_track)
    pad = int(round(max(delay - (audio_start or 0.0), 0.0) * sr))
    if pad <= 0:
        return audio
    logger.debug("%s: restoring %.3fs of container audio delay", file, pad / sr)
    audio = np.concatenate([np.zeros(pad, dtype=audio.dtype), audio])
    if audio_end is not None:
        audio = audio[:int(round((audio_end - (audio_start or 0.0)) * sr))]
    return audio


def load_audio(file: str,
               sr: int = SAMPLE_RATE,
               audio_track: int = 0,
               audio_start: Optional[float] = None,
               audio_end: Optional[float] = None,
               downmix: str = "mix",
               normalize: bool = False,
               ) -> np.ndarray:
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    audio_track: int
        The index of the audio track, if there are multiple

    audio_start: float
        Start of the clip to read, in seconds (None = from the beginning)

    audio_end: float
        End of the clip to read, in seconds (None = to the end)

    downmix: str
        How to reduce a multichannel track to mono. "mix" (default) lets ffmpeg
        downmix every channel; "center" takes the front-center channel alone, which
        on a film soundtrack is largely the dialogue stem.

    normalize: bool
        Apply one linear gain bringing the speech to ``NORMALIZE_TARGET_DBFS``.

        Defaults to FALSE, and that default is load-bearing. This function is
        called on two quite different things: source media, and clips that were
        already cut from source media and therefore already carry their
        episode's gain (``alignment`` reloads per-segment wavs that way). A
        default of True would normalise those a second time -- per clip, which
        flattens exactly the dynamics the per-file gain is designed to keep --
        and it would do it silently, in one stage only. So it is switched on
        explicitly where source media is read, and nowhere else.

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.

    Sample 0 is the container's zero -- the instant a player starts counting,
    and therefore the instant a subtitle file counts from -- or ``audio_start``
    where one is given. On a container that muxes its audio with a delay that is
    NOT what ffmpeg returns on its own; see ``_restore_container_delay``.
    """
    pre_input, post_input = _clip_ffmpeg_args(audio_start, audio_end)
    filter_args = _downmix_ffmpeg_args(file, audio_track, downmix)
    try:
        cmd = ["ffmpeg", "-nostdin", "-threads", "0", *pre_input, "-i", file]
        if audio_track != 0:
            cmd += ["-map", f"0:a:{audio_track}"]
        cmd += [*post_input, *filter_args,
                "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr), "-"]
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    audio = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    audio = _restore_container_delay(audio, file, audio_track, audio_start,
                                     audio_end, sr)
    if normalize:
        gain = normalize_gain(audio, sr)
        if gain:
            # Clipping is prevented by the ceiling in normalize_gain, so this
            # clamp only catches the float edge; it is not doing the work.
            audio = np.clip(audio * (10.0 ** (gain / 20.0)), -1.0, 1.0)
    return audio


def extract_clip_to_wav(
    src: str,
    dst: str,
    *,
    audio_start: Optional[float] = None,
    audio_end: Optional[float] = None,
    audio_track: int = 0,
    sr: int = SAMPLE_RATE,
    downmix: str = "mix",
) -> str:
    """Write a 16 kHz mono WAV of ``src``'s [audio_start, audio_end) clip to ``dst``.

    Used by the pipeline entry point to apply an audio clip *once*: every downstream
    stage then reads the already-clipped, single-track ``dst`` file (with
    ``audio_track=0``), so clipping works uniformly even for stages that reload the
    file by path (diarization, speaker verification). Output timestamps are relative
    to the clip start; the caller offsets them by ``audio_start`` to map back to the
    source timeline. Returns ``dst``.

    Decodes through ``load_audio`` and writes the samples here, rather than
    letting ffmpeg write the file, so that the clipped path and the whole-file
    path cannot drift apart. They already had: this function is what
    ``--audio_start`` goes through, and before the container-delay correction the
    two disagreed by the delay -- so the same media transcribed with the flag and
    without it produced subtitles a second apart. One decoder is the only way
    that stays fixed.

    Written with the stdlib ``wave`` module for the same reason the decode is a
    subprocess: this is a core path, and soundfile reaches this package only
    transitively through librosa.
    """
    import wave

    audio = load_audio(src, sr=sr, audio_track=audio_track,
                       audio_start=audio_start, audio_end=audio_end,
                       downmix=downmix)
    # Exact, not merely close: load_audio built these by dividing int16 by
    # 32768, a power of two, so every value is back to the integer it came from
    # with no rounding to argue about.
    samples = np.clip(audio * 32768.0, -32768.0, 32767.0).astype("<i2")
    try:
        with wave.open(dst, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(sr)
            out.writeframes(samples.tobytes())
    except OSError as e:
        raise RuntimeError(f"Failed to extract audio clip: {e}") from e
    return dst


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels in [80, 128], f"Unsupported n_mels: {n_mels}"
    # assets/ is one level up from utils/ (cantocaptions_ai/assets/)
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    with np.load(os.path.join(assets_dir, "mel_filters.npz")) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec

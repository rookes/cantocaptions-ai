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


# ffmpeg channel layouts that carry a discrete front-center channel. On a mixed
# soundtrack FC is the dialogue stem, so extracting it alone is close to free vocal
# isolation -- the reason --audio_downmix center exists. Layouts absent from this set
# (stereo, 2.1, quad) have no FC to take, and an unknown layout is not guessed at.
_LAYOUTS_WITH_CENTER = frozenset({
    "mono",
    "3.0", "3.0(back)", "4.0",
    "5.0", "5.0(side)", "5.1", "5.1(side)",
    "6.0", "6.0(front)", "6.1", "6.1(back)", "6.1(front)",
    "7.0", "7.0(front)", "7.1", "7.1(wide)", "7.1(wide-side)",
    "hexagonal", "octagonal",
})


def _downmix_ffmpeg_args(file: str, audio_track: int, downmix: str) -> list:
    """Return the ffmpeg filter args implementing *downmix*, or [] for a plain downmix.

    ``center`` isolates the front-center channel. It is applied only when ffprobe reports
    a layout known to have one; anything else (stereo, an unnamed layout) falls back to
    the ordinary all-channel downmix with a warning rather than risking a filter error
    partway through decoding a feature-length file.
    """
    if downmix == "mix":
        return []
    if downmix != "center":
        raise ValueError(f"Unknown downmix mode: {downmix!r} (expected 'mix' or 'center')")

    streams = probe_audio_tracks(file)
    stream = streams[audio_track] if audio_track < len(streams) else None
    layout = (stream or {}).get("channel_layout", "")
    channels = int((stream or {}).get("channels", 0) or 0)

    if layout == "mono" or channels == 1:
        return []  # already the center channel; nothing to extract
    if layout in _LAYOUTS_WITH_CENTER:
        return ["-af", "pan=mono|c0=FC"]

    logger.warning(
        "--audio_downmix center: track %d has layout %r (%d channel(s)) with no front-center "
        "channel to extract; falling back to a full downmix.",
        audio_track, layout or "unknown", channels,
    )
    return []


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


def load_audio(file: str,
               sr: int = SAMPLE_RATE,
               audio_track: int = 0,
               audio_start: Optional[float] = None,
               audio_end: Optional[float] = None,
               downmix: str = "mix",
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

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
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

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


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
    """
    pre_input, post_input = _clip_ffmpeg_args(audio_start, audio_end)
    filter_args = _downmix_ffmpeg_args(src, audio_track, downmix)
    cmd = ["ffmpeg", "-nostdin", "-y", "-threads", "0", *pre_input, "-i", src]
    if audio_track != 0:
        cmd += ["-map", f"0:a:{audio_track}"]
    cmd += [*post_input, *filter_args, "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr), dst]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to extract audio clip: {e.stderr.decode()}") from e
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

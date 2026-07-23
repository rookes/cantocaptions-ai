import json
import os
import subprocess
from functools import lru_cache
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from cantocaptions_ai.utils.output import exact_div

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


def load_audio(file: str, 
               sr: int = SAMPLE_RATE, 
               audio_track: int = 0, 
               audio_start: Optional[float] = None, 
               audio_end: Optional[float] = None
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
        The amount of audio to cut from the beginning

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        cmd = ["ffmpeg", "-nostdin", "-threads", "0", "-i", file]
        if audio_track != 0:
            cmd += ["-map", f"0:a:{audio_track}"]
        cmd += ["-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr), "-"]
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


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

"""Decoding puts sample 0 at the container's zero, not the audio stream's.

A container may mux its audio behind its video -- Peppa Pig S1 holds the
Cantonese dub at 1.000 s against video at 0.041 s. Players honour that; ffmpeg
decoding to a RAW format has no muxer to honour it and rebases the stream onto
its own first packet, dropping the lead instead of padding it. Every timestamp
the pipeline then emits is early by the delay, so the subtitles it writes are
early against the video by the same amount.

The nastier half was that the two decode shapes disagreed. ``-ss`` seeks on the
container timeline and was already correct, so ``--audio_start 5`` and no flag
at all produced subtitles a second apart for the same media -- the clipped path
being the one people reach for when a timing looks wrong. Both are pinned here.

Media is generated with ffmpeg rather than committed, and ``-itsoffset`` is what
gives it the same shape the real files have: container start 0, audio stream
start 1.0. A file that merely starts late reproduces nothing.
"""

import shutil
import subprocess
import wave

import numpy as np
import pytest

from cantocaptions_ai.utils.audio import (
    HONOURS_CONTAINER_DELAY,
    container_audio_delay,
    extract_clip_to_wav,
    load_audio,
)

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                reason="ffmpeg not on PATH")

SR = 16000
DELAY_S = 1.0
#: Where the burst sits inside the AUDIO stream; on the container timeline --
#: the one a subtitle counts in -- it is DELAY_S later.
TONE_IN_STREAM = (3.0, 4.0)
TONE_IN_CONTAINER = (TONE_IN_STREAM[0] + DELAY_S, TONE_IN_STREAM[1] + DELAY_S)


def _rms(audio, start_s, end_s) -> float:
    window = audio[int(start_s * SR):int(end_s * SR)]
    return float(np.sqrt((window ** 2).mean())) if len(window) else 0.0


def _holds_tone_at(audio, window) -> bool:
    """True if the burst sits at `window` rather than a delay away from it.

    A ratio, not an absolute floor: the gap is lossy-coded, so the quiet side
    carries the codec's ringing around the burst edge and never reaches digital
    silence. The order-of-magnitude separation is the claim; an absolute
    threshold would encode one aac build's noise floor into the suite.
    """
    other = (window[0] - DELAY_S, window[1] - DELAY_S)
    return _rms(audio, *window) > 5 * _rms(audio, *other)


@pytest.fixture(scope="module")
def delayed_media(tmp_path_factory):
    """(delayed, plain) -- the same 6 s tone, muxed with and without a delay."""
    tmp = tmp_path_factory.mktemp("delay")
    plain = tmp / "plain.mkv"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error",
         "-f", "lavfi", "-i", "color=c=black:s=64x64:r=25:d=6",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=6",
         "-af", f"volume=enable='between(t,{TONE_IN_STREAM[0]},"
                f"{TONE_IN_STREAM[1]})':volume=1,"
                f"volume=enable='not(between(t,{TONE_IN_STREAM[0]},"
                f"{TONE_IN_STREAM[1]}))':volume=0",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest",
         str(plain)], check=True)

    delayed = tmp / "delayed.mkv"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(plain),
         "-itsoffset", str(DELAY_S), "-i", str(plain),
         "-map", "0:v", "-map", "1:a", "-c", "copy", str(delayed)], check=True)
    return delayed, plain


def test_container_audio_delay_reads_the_mux_offset(delayed_media):
    delayed, plain = delayed_media
    assert container_audio_delay(str(delayed)) == pytest.approx(DELAY_S, abs=1e-3)
    assert container_audio_delay(str(plain)) == 0.0


def test_container_audio_delay_is_zero_for_unreadable_input(tmp_path):
    """A file ffprobe cannot parse has a louder problem one line later."""
    junk = tmp_path / "not-media.mkv"
    junk.write_bytes(b"absolutely not a container")
    assert container_audio_delay(str(junk)) == 0.0
    assert container_audio_delay(str(tmp_path / "missing.mkv")) == 0.0


def test_whole_file_decode_lands_on_the_container_timeline(delayed_media):
    delayed, _ = delayed_media
    assert _holds_tone_at(load_audio(str(delayed), sr=SR), TONE_IN_CONTAINER)


def test_clipped_and_unclipped_decodes_agree(delayed_media):
    """The regression that mattered: one file must not have two timelines.

    A clip starting at 2 s should contain the burst 3 s in, because on the
    container timeline that is where it is -- and that has to be the same answer
    the whole-file decode gives, or the flag changes the subtitles.
    """
    delayed, _ = delayed_media
    whole = load_audio(str(delayed), sr=SR)
    clip = load_audio(str(delayed), sr=SR, audio_start=2.0, audio_end=6.0)

    start = TONE_IN_CONTAINER[0] - 2.0
    assert _holds_tone_at(clip, (start, start + 1.0))
    assert _rms(clip, start, start + 1.0) == pytest.approx(
        _rms(whole, *TONE_IN_CONTAINER), rel=0.05)


def test_clip_starting_inside_the_delay_is_padded_not_clamped(delayed_media):
    """``-ss`` below the delay has no packet to seek to and clamps to the first.

    ffmpeg then returns audio from `delay` while the caller believes it has
    audio from `audio_start`. The shortfall is padded, so the burst still sits
    where the container puts it -- and the clip is still the length asked for,
    because ffmpeg measured its own ``-t`` from the clamp point.
    """
    delayed, _ = delayed_media
    clip = load_audio(str(delayed), sr=SR, audio_start=0.5, audio_end=6.0)
    assert len(clip) == pytest.approx(5.5 * SR, abs=2)
    start = TONE_IN_CONTAINER[0] - 0.5
    assert _holds_tone_at(clip, (start, start + 1.0))


def test_extract_clip_to_wav_matches_load_audio(delayed_media, tmp_path):
    """The temp WAV every stage reads must hold what the decoder would return."""
    delayed, _ = delayed_media
    dst = tmp_path / "clip.wav"
    extract_clip_to_wav(str(delayed), str(dst), audio_start=2.0, audio_end=6.0)

    with wave.open(str(dst), "rb") as f:
        assert (f.getnchannels(), f.getsampwidth(), f.getframerate()) == (1, 2, SR)
        written = np.frombuffer(f.readframes(f.getnframes()),
                                dtype="<i2").astype(np.float32) / 32768.0
    assert np.array_equal(
        written, load_audio(str(delayed), sr=SR, audio_start=2.0, audio_end=6.0))


def test_undelayed_media_decodes_exactly_as_before(delayed_media):
    """The correction must be a no-op, sample for sample, where none is needed.

    Most media has no delay, and the decoder's output feeds a trained model --
    "nearly the same" would be a silent change to every inference this package
    has ever run.
    """
    _, plain = delayed_media
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-threads", "0", "-i", str(plain),
         "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(SR), "-"],
        capture_output=True, check=True).stdout
    expected = np.frombuffer(raw, np.int16).flatten().astype(np.float32) / 32768.0
    assert np.array_equal(load_audio(str(plain), sr=SR), expected)


def test_capability_flag_is_advertised():
    """Consumers branch on this to avoid correcting a second time."""
    assert HONOURS_CONTAINER_DELAY is True

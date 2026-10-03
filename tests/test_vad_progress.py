"""VAD reports progress within a file, so a group of one file doesn't sit at 0% until done.

Decoding streams through ffmpeg to report how much is in, and the pyannote backend reports
its scoring through Inference's hook. Neither may change a single sample or segment.
"""
import shutil
import wave

import numpy as np
import pytest

from cantocaptions_ai.pipeline.vad import VadProcessor
from cantocaptions_ai.pipeline.vads import Silero
from cantocaptions_ai.utils.audio import load_audio

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")

SR = 16000


@pytest.fixture
def long_wav(tmp_path):
    """80 s: decodes to 2.5 MB, so the 1 MB reads report progress more than once."""
    rng = np.random.default_rng(0)
    seconds = 80
    t = np.arange(seconds * SR) / SR
    x = rng.normal(0, 0.001, len(t))
    voiced = (t % 10) < 4
    x[voiced] += 0.2 * sum(np.sin(2 * np.pi * 140 * k * t[voiced]) / k for k in range(1, 8))
    path = tmp_path / "long.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    return str(path)


def test_progress_does_not_change_the_samples(long_wav):
    seen = []
    with_progress = load_audio(long_wav, progress=seen.append)
    assert np.array_equal(with_progress, load_audio(long_wav))
    assert len(seen) >= 2
    assert seen == sorted(seen) and seen[-1] == pytest.approx(1.0, abs=0.01)


def test_progress_is_measured_against_the_clip(long_wav):
    seen = []
    clip = load_audio(long_wav, audio_start=10.0, audio_end=50.0, progress=seen.append)
    assert np.array_equal(clip, load_audio(long_wav, audio_start=10.0, audio_end=50.0))
    assert seen[-1] == pytest.approx(1.0, abs=0.01)


def test_a_failed_decode_still_raises(tmp_path):
    with pytest.raises(RuntimeError, match="Failed to load audio"):
        load_audio(str(tmp_path / "missing.wav"), progress=lambda f: None)


class _HookingSilero(Silero):
    """Silero scores in one call; this one reports through the hook the way pyannote does."""

    def __call__(self, audio, hook=None, **kwargs):
        if hook is not None:
            for done in (0, 50, 100):
                hook(completed=done, total=100)
        return super().__call__(audio)


class _Reporter:
    def __init__(self):
        self.partials, self.advanced = [], 0

    def set_total(self, total, unit="it"): pass
    def advance(self, n=1): self.advanced += n
    def partial(self, fraction): self.partials.append(fraction)


def _processor():
    return VadProcessor(_HookingSilero(vad_onset=0.5), vad_onset=0.5, vad_offset=0.363,
                        chunk_size=28)


def test_vad_moves_through_decoding_then_scoring(long_wav):
    reporter = _Reporter()
    item = {"audio_path": long_wav, "name": "long"}
    (out,) = _processor().run([item], progress_callback=reporter)
    decoding = [f for f in reporter.partials if f <= 0.5]
    scoring = [f for f in reporter.partials if f > 0.5]
    assert decoding and scoring
    assert reporter.partials == sorted(reporter.partials)
    assert reporter.partials[-1] == pytest.approx(1.0)
    assert reporter.advanced == 1

    # The same segments as a run with nothing to report to.
    (plain,) = _processor().run([item])
    assert [(s["start"], s["end"]) for s in out["vad_segments"]] == [
        (s["start"], s["end"]) for s in plain["vad_segments"]]


def test_a_reporter_without_partial_still_works(long_wav):
    class Plain:
        advanced = 0
        def set_total(self, total, unit="it"): pass
        def advance(self, n=1): Plain.advanced += n

    _processor().run([{"audio_path": long_wav, "name": "long"}], progress_callback=Plain())
    assert Plain.advanced == 1

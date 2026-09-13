"""Level normalisation, and the parity it exists to create.

The corpus is cut with one linear gain per file bringing its speech to
NORMALIZE_TARGET_DBFS. Inference must do the same thing by the same rule, or the
model learns one level distribution and meets another -- which is worse than not
normalising at all, since at least an unnormalised corpus matches unnormalised
input.
"""

import numpy as np
import pytest

from cantocaptions_ai.utils import audio as A

SR = A.SAMPLE_RATE


def _tone(seconds=4.0, dbfs=-20.0, freq=220.0, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return (10 ** (dbfs / 20.0) * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(seconds, sr=SR):
    return np.zeros(int(seconds * sr), dtype=np.float32)


# --- the estimator ------------------------------------------------------------

def test_gated_level_tracks_a_steady_tone():
    """A -20 dBFS sine has RMS 3 dB below its peak."""
    assert A.gated_level(_tone(dbfs=-20.0)) == pytest.approx(-23.0, abs=0.5)


def test_gated_level_ignores_silence():
    """The reason gating is used at all: a file that is mostly silence must
    report the level of its speech, not an average dragged toward the floor."""
    loud = _tone(seconds=2.0, dbfs=-20.0)
    padded = np.concatenate([_silence(6.0), loud, _silence(6.0)])
    assert A.gated_level(padded) == pytest.approx(A.gated_level(loud), abs=1.0)


def test_gated_level_ignores_background_under_the_absolute_gate():
    """Room tone below -60 dBFS is excluded outright."""
    speech = _tone(seconds=3.0, dbfs=-20.0)
    quiet = _tone(seconds=6.0, dbfs=-64.0, freq=90.0)
    mixed = np.concatenate([quiet, speech, quiet])
    assert A.gated_level(mixed) == pytest.approx(A.gated_level(speech), abs=1.5)


def test_gated_level_is_not_dragged_up_by_loud_music():
    """The failure this estimator exists to avoid.

    A mean over gated blocks reports the score rather than the dialogue -- on Ip
    Man it read 11.6 dB high, which would leave that film's speech 11.6 dB under
    target. A median lands on the typical talking block instead.
    """
    speech = _tone(seconds=6.0, dbfs=-26.0)
    music = _tone(seconds=3.0, dbfs=-8.0, freq=440.0)
    mixed = np.concatenate([speech, music, speech])
    assert A.gated_level(mixed) == pytest.approx(A.gated_level(speech), abs=1.5)


def test_gated_level_reports_ambience_when_it_dominates(caplog):
    """The known weakness, pinned so it is a decision rather than a surprise.

    Material that is mostly ambience WITHIN the relative gate has that ambience
    as its median. The peak ceiling in normalize_gain bounds the consequence,
    and this corpus does not contain the case; see gated_level's docstring.
    """
    speech = _tone(seconds=2.0, dbfs=-20.0)
    ambience = _tone(seconds=10.0, dbfs=-45.0, freq=90.0)
    mixed = np.concatenate([ambience, speech, ambience])
    assert A.gated_level(mixed) < A.gated_level(speech) - 10.0


def test_gated_level_returns_none_for_digital_silence():
    assert A.gated_level(_silence(3.0)) is None


def test_gated_level_returns_none_for_audio_shorter_than_a_block():
    assert A.gated_level(_tone(seconds=0.1)) is None


# --- the gain -----------------------------------------------------------------

def test_gain_brings_a_quiet_file_up_to_target():
    quiet = _tone(dbfs=-40.0)
    gain = A.normalize_gain(quiet)
    assert A.gated_level(quiet * 10 ** (gain / 20)) == pytest.approx(
        A.NORMALIZE_TARGET_DBFS, abs=0.5)


def test_gain_turns_a_loud_file_down():
    """A third of this corpus gets turned DOWN. Normalising is centring, not
    boosting, and a one-directional implementation would be wrong for them."""
    assert A.normalize_gain(_tone(dbfs=-6.0)) < 0


def test_gain_never_exceeds_the_peak_ceiling():
    """A quiet file with one loud transient must not be lifted into clipping."""
    signal = np.concatenate([_tone(seconds=4.0, dbfs=-45.0), _tone(seconds=0.05, dbfs=-2.0)])
    gain = A.normalize_gain(signal)
    peak_after = 20 * np.log10(np.abs(signal * 10 ** (gain / 20)).max())
    assert peak_after <= A.NORMALIZE_CEILING_DBFS + 1e-6


def test_gain_is_zero_for_unmeasurable_audio():
    """Callers apply the result unconditionally, so 'no answer' has to be a
    no-op rather than an exception or a huge lift of pure silence."""
    assert A.normalize_gain(_silence(3.0)) == 0.0
    assert A.normalize_gain(np.zeros(0, dtype=np.float32)) == 0.0


def test_target_is_read_at_call_time_not_bound_at_import(monkeypatch):
    """A default argument would freeze the constant, so retuning it would be
    recorded by the dataset's stamp, trigger a full re-cut, and then write the
    OLD level anyway -- a stamp describing audio that does not exist."""
    quiet = _tone(dbfs=-40.0)
    before = A.normalize_gain(quiet)
    monkeypatch.setattr(A, "NORMALIZE_TARGET_DBFS", A.NORMALIZE_TARGET_DBFS - 6.0)
    assert A.normalize_gain(quiet) == pytest.approx(before - 6.0, abs=0.01)


def test_gain_is_level_invariant():
    """Two copies of the same content at different levels must land together --
    this is the entire point."""
    quiet, loud = _tone(dbfs=-40.0), _tone(dbfs=-15.0)
    after_quiet = A.gated_level(quiet * 10 ** (A.normalize_gain(quiet) / 20))
    after_loud = A.gated_level(loud * 10 ** (A.normalize_gain(loud) / 20))
    assert after_quiet == pytest.approx(after_loud, abs=0.5)


def test_gain_does_not_compress():
    """One linear gain, so every level relationship inside the file survives."""
    signal = np.concatenate([_tone(seconds=2.0, dbfs=-35.0),
                             _tone(seconds=2.0, dbfs=-15.0)])
    scaled = signal * 10 ** (A.normalize_gain(signal) / 20)
    half = len(signal) // 2
    def rms(x):
        return 20 * np.log10(np.sqrt((x.astype(np.float64) ** 2).mean()))
    assert (rms(scaled[half:]) - rms(scaled[:half])) == pytest.approx(
        rms(signal[half:]) - rms(signal[:half]), abs=0.01)


# --- the default that prevents double application -----------------------------

def test_load_audio_does_not_normalize_by_default():
    """Load-bearing. `load_audio` is called both on source media AND on clips
    already cut from it, which already carry their episode's gain (alignment
    reloads per-segment wavs that way). Normalising by default would level those
    a second time, per clip, silently, in one stage only."""
    import inspect

    assert inspect.signature(A.load_audio).parameters["normalize"].default is False

"""The multichannel -> mono downmix matrix.

These lock down the fix for a bug that was invisible for a long time: decoding
to `s16le` puts libswresample on its integer path, where `rematrix_maxval`
resolves to 1.0 and the whole matrix is scaled down by its coefficient sum --
about 9.4 dB on any 5.1 source, and nothing at all on stereo, which is why it
hid. Stating the matrix explicitly is what makes the gain ours rather than a
side effect of the output sample format.
"""

import pytest

from cantocaptions_ai.utils import audio as A


def _probe(monkeypatch, layout, channels):
    monkeypatch.setattr(A, "probe_audio_tracks",
                        lambda _f: [{"channel_layout": layout,
                                     "channels": channels}])


def test_stereo_gets_no_filter(monkeypatch):
    """ffmpeg's stereo downmix is (L+R)/2, already unity for centred dialogue.

    Touching it would change 345 of this corpus's 401 episodes for no reason.
    """
    _probe(monkeypatch, "stereo", 2)
    assert A._downmix_ffmpeg_args("f.mkv", 0, "mix") == []


def test_mono_gets_no_filter(monkeypatch):
    _probe(monkeypatch, "mono", 1)
    assert A._downmix_ffmpeg_args("f.mkv", 0, "mix") == []


def test_five_one_gets_an_explicit_matrix(monkeypatch):
    _probe(monkeypatch, "5.1(side)", 6)
    args = A._downmix_ffmpeg_args("f.mkv", 0, "mix")
    assert args[0] == "-af"
    assert args[1] == "pan=mono|c0=0.53*c0+0.53*c1+0.75*c2+0.53*c4+0.53*c5"


def test_back_and_side_variants_produce_the_same_expression():
    """5.1 names its rear pair BL/BR and 5.1(side) names it SL/SR.

    They are the same six channels in the same order, so an INDEX-based
    expression is identical for both -- which is the reason indices are used
    instead of channel names, since `pan=...SL...` is rejected outright on a
    plain 5.1 file.
    """
    assert A._mono_pan_filter("5.1") == A._mono_pan_filter("5.1(side)")


def test_lfe_is_never_in_the_mix():
    """LFE carries nothing above 120 Hz that speech needs, and including it was
    measurably the cause of clipping: 2 of 55 sources over 0 dBFS with it,
    1 (by 0.02 dB) without."""
    for layout, channels in A._LAYOUT_CHANNELS.items():
        if "LFE" not in channels:
            continue
        expression = A._mono_pan_filter(layout)
        lfe_index = channels.index("LFE")
        assert f"*c{lfe_index}" not in expression, layout


def test_surrounds_sit_three_db_under_centre_not_six():
    """The whole point of the matrix: background and off-screen speech lives
    off-centre, and the conventional 6 dB spread buries it."""
    import math

    weights = A._DOWNMIX_WEIGHTS
    spread = 20 * math.log10(weights["center"] / weights["surround"])
    assert 2.5 < spread < 3.5


def test_centre_sits_below_the_front_pair_for_headroom():
    """Film peaks are centre-driven, so a centre weight under the summed fronts
    is what lets the surrounds run hot without clipping."""
    weights = A._DOWNMIX_WEIGHTS
    assert weights["center"] < 2 * weights["front"]


def test_unknown_multichannel_layout_falls_back_rather_than_guessing(monkeypatch, caplog):
    """A wrong pan expression is worse than a quiet one: ffmpeg rejects it
    partway through decoding a feature-length file."""
    _probe(monkeypatch, "22.2", 24)
    with caplog.at_level("WARNING"):
        assert A._downmix_ffmpeg_args("f.mkv", 0, "mix") == []
    assert "22.2" in caplog.text


def test_layouts_with_centre_excludes_layouts_that_have_none():
    """3.0(back) is FL FR BC, and the 'front' 6.x layouts use FLC/FRC.

    The old hand-written set listed all three as having an FC, so
    `--audio_downmix center` built `pan=mono|c0=FC` and ffmpeg refused it.
    """
    for layout in ("3.0(back)", "6.0(front)", "6.1(front)", "stereo", "quad"):
        assert layout not in A._LAYOUTS_WITH_CENTER, layout
    for layout in ("5.1", "5.1(side)", "7.1"):
        assert layout in A._LAYOUTS_WITH_CENTER, layout


def test_center_mode_still_extracts_fc(monkeypatch):
    _probe(monkeypatch, "5.1(side)", 6)
    assert A._downmix_ffmpeg_args("f.mkv", 0, "center") == ["-af", "pan=mono|c0=FC"]


def test_center_mode_falls_back_when_there_is_no_centre(monkeypatch, caplog):
    _probe(monkeypatch, "quad", 4)
    with caplog.at_level("WARNING"):
        assert A._downmix_ffmpeg_args("f.mkv", 0, "center") == []


def test_unknown_mode_is_rejected(monkeypatch):
    _probe(monkeypatch, "5.1", 6)
    with pytest.raises(ValueError, match="expected 'mix' or 'center'"):
        A._downmix_ffmpeg_args("f.mkv", 0, "loudest")


def test_every_known_layout_yields_a_usable_expression():
    """Guard against a typo in the channel table producing an empty or
    malformed filter for a layout nobody has media for yet."""
    for layout, channels in A._LAYOUT_CHANNELS.items():
        expression = A._mono_pan_filter(layout)
        assert expression.startswith("pan=mono|c0=")
        terms = expression.split("=", 2)[2].split("+")
        assert terms and all("*c" in t for t in terms), layout
        # Never addresses a channel the layout does not have.
        for term in terms:
            assert int(term.split("*c")[1]) < len(channels), layout

"""Audio-track selection: Cantonese first, then any Chinese track, then default."""

from cantocaptions_ai.utils.audio import select_cantonese_track


def _stream(language=None, title=None):
    tags = {}
    if language is not None:
        tags["language"] = language
    if title is not None:
        tags["title"] = title
    return {"tags": tags}


def test_prefers_explicit_cantonese_language():
    streams = [_stream("eng"), _stream("yue"), _stream("jpn")]
    assert select_cantonese_track(streams) == 1


def test_prefers_cantonese_title_even_without_language_tag():
    streams = [_stream("und", "Japanese"), _stream("und", "Cantonese Dub")]
    assert select_cantonese_track(streams) == 1


def test_falls_back_to_first_chinese_track_when_no_cantonese():
    # No yue/Cantonese; a Mandarin (zh) track should win over English/Japanese
    # instead of defaulting to track 0.
    streams = [_stream("eng"), _stream("jpn"), _stream("zh"), _stream("cmn")]
    assert select_cantonese_track(streams) == 2


def test_chinese_fallback_matches_language_variants_and_titles():
    assert select_cantonese_track([_stream("eng"), _stream("zh-HK")]) == 1
    assert select_cantonese_track([_stream("eng"), _stream("und", "國語")]) == 1
    assert select_cantonese_track([_stream("eng"), _stream("chi")]) == 1


def test_cantonese_beats_other_chinese_even_when_later():
    # A Mandarin track appears first, Cantonese later: Cantonese still wins.
    streams = [_stream("cmn"), _stream("eng"), _stream("yue")]
    assert select_cantonese_track(streams) == 2


def test_returns_zero_when_no_chinese_audio():
    streams = [_stream("eng"), _stream("jpn"), _stream("kor")]
    assert select_cantonese_track(streams) == 0

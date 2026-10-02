"""Language packs shipped as their own distributions, found through entry points.

Each test runs a fresh interpreter with a fake installed distribution on its path -- a
``*.dist-info`` with an ``entry_points.txt``, which is exactly what ``importlib.metadata``
reads for a real install -- so the registry's plugin loading is exercised for real and
never leaks into this process's registry.
"""
import json
import subprocess
import sys
import textwrap

PACK_MODULE = textwrap.dedent('''
    from cantocaptions_ai.languages import LanguagePack
    from cantocaptions_ai.text_profiles import LATIN_PUNCTUATION, SPACED_SCRIPT

    PACK = LanguagePack("tlh", SPACED_SCRIPT, LATIN_PUNCTUATION, default_model="some/tlh-asr")

    def make_yue():
        return LanguagePack("yue", SPACED_SCRIPT, LATIN_PUNCTUATION, default_model="other/yue")

    NOT_A_PACK = 42
''')


def _install(tmp_path, entries: str):
    (tmp_path / "toy_pack.py").write_text(PACK_MODULE)
    dist = tmp_path / "toy_pack-0.1.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: toy-pack\nVersion: 0.1\n")
    (dist / "entry_points.txt").write_text("[cantocaptions_ai.languages]\n" + entries)


def _run(tmp_path, code: str):
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
        env={"PYTHONPATH": str(tmp_path), "PATH": "/usr/bin:/bin"},
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1]), out.stderr


REPORT = (
    "import json, sys; from cantocaptions_ai.languages import get_language_pack, LANGUAGE_PACKS;"
    "p = get_language_pack('tlh');"
    "print(json.dumps({'tlh': p.default_model, 'codes': sorted(LANGUAGE_PACKS),"
    " 'yue': get_language_pack('yue').default_model, 'torch': 'torch' in sys.modules}))"
)


def test_an_installed_pack_is_registered(tmp_path):
    _install(tmp_path, "tlh = toy_pack:PACK\n")
    report, _ = _run(tmp_path, REPORT)
    assert report["tlh"] == "some/tlh-asr"
    assert report["codes"] == ["tlh", "yue"]
    assert report["yue"] == "cantocaptions-cantonese-ASR"
    assert report["torch"] is False


def test_a_broken_pack_is_skipped_with_a_warning(tmp_path):
    _install(tmp_path, "gone = toy_pack_missing:PACK\nodd = toy_pack:NOT_A_PACK\n"
                       "tlh = toy_pack:PACK\n")
    report, stderr = _run(tmp_path, REPORT)
    assert report["codes"] == ["tlh", "yue"]
    assert "'gone' from toy-pack is skipped" in stderr
    assert "'odd' from toy-pack is skipped" in stderr


def test_a_plugin_may_replace_a_built_in_pack(tmp_path):
    _install(tmp_path, "yue = toy_pack:make_yue\n")
    report, _ = _run(tmp_path, REPORT)
    assert report["yue"] == "other/yue"


def test_a_pack_registered_in_code_beats_a_plugin(tmp_path):
    _install(tmp_path, "tlh = toy_pack:PACK\n")
    report, _ = _run(tmp_path, (
        "from cantocaptions_ai.languages import LanguagePack, register_language_pack;"
        "from cantocaptions_ai.text_profiles import LATIN_PUNCTUATION, SPACED_SCRIPT;"
        "register_language_pack(LanguagePack('tlh', SPACED_SCRIPT, LATIN_PUNCTUATION,"
        " default_model='mine/tlh'));" + REPORT))
    assert report["tlh"] == "mine/tlh"

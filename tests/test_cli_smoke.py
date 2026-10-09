"""The installed entry point starts, parses and exits cleanly.

Run as a real subprocess so import-time failures (a missing optional dependency, a flag
defined twice, a bad default) surface here rather than on a user's first run.
"""
import subprocess
import sys


def _cli(*args):
    return subprocess.run([sys.executable, "-m", "cantocaptions_ai", *args],
                          capture_output=True, text=True, timeout=120)


def test_help_lists_the_core_flags():
    out = _cli("--help")
    assert out.returncode == 0, out.stderr
    for flag in ("--input_dir", "--language", "--realign", "--cfg", "--diarize"):
        assert flag in out.stdout


def test_version():
    out = _cli("--version")
    assert out.returncode == 0 and out.stdout.strip()


def test_no_input_is_a_usage_error():
    out = _cli()
    assert out.returncode == 2
    assert "provide at least one audio file, --input_dir, or --proofread_input" in out.stderr

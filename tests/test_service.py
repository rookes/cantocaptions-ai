"""Tests for the server-facing surface added for hosted use:

- validate_config raises ConfigError (never sys.exit) and normalizes language
- render_result renders in memory and rejects "all"
- _offset_result_times maps clip-relative times back to the source timeline
- _clip_ffmpeg_args builds correct ffmpeg seek/duration args and rejects bad ranges
- validate_input_file raises InputError on missing/empty inputs
"""
import os
import tempfile
import unittest

from cantocaptions_ai.errors import ConfigError, InputError
from cantocaptions_ai.pipeline.config import PipelineConfig
from cantocaptions_ai.pipeline.transcribe import validate_config, _offset_result_times
from cantocaptions_ai.utils.audio import _clip_ffmpeg_args, validate_input_file
from cantocaptions_ai.utils.output import render_result


class TestValidateConfig(unittest.TestCase):
    def test_reference_subtitle_requires_llm(self):
        cfg = PipelineConfig(reference_subtitle="ref.srt", llm_correction=False)
        with self.assertRaises(ConfigError):
            validate_config(cfg)

    def test_no_align_conflicts_with_word_options(self):
        cfg = PipelineConfig(no_align=True, max_line_width=18)
        with self.assertRaises(ConfigError):
            validate_config(cfg)

    def test_unsupported_language_raises(self):
        cfg = PipelineConfig(language="tlh")  # Klingon
        with self.assertRaises(ConfigError):
            validate_config(cfg)

    def test_language_is_normalized_in_place(self):
        cfg = PipelineConfig(language="YUE")
        validate_config(cfg)
        self.assertEqual(cfg.language, "yue")

    def test_valid_config_passes(self):
        validate_config(PipelineConfig())  # defaults must be valid

    def test_reference_subtitle_accepted_for_asr_context(self):
        # asr_context is the second consumer of the same file, so it satisfies the
        # requirement that used to be llm_correction-only.
        validate_config(PipelineConfig(reference_subtitle="ref.srt", asr_context=True))

    def test_asr_context_requires_reference_subtitle(self):
        with self.assertRaises(ConfigError):
            validate_config(PipelineConfig(asr_context=True))

    def test_asr_context_rejects_unknown_template(self):
        cfg = PipelineConfig(
            reference_subtitle="ref.srt", asr_context=True, asr_context_template="nope"
        )
        with self.assertRaises(ConfigError):
            validate_config(cfg)

    def test_asr_context_rejects_negative_neighbours(self):
        cfg = PipelineConfig(
            reference_subtitle="ref.srt", asr_context=True, asr_context_neighbours=-1
        )
        with self.assertRaises(ConfigError):
            validate_config(cfg)

    def test_asr_context_conflicts_with_retime(self):
        cfg = PipelineConfig(
            reference_subtitle="ref.srt", asr_context=True, retime="in.srt"
        )
        with self.assertRaises(ConfigError):
            validate_config(cfg)


class TestRenderResult(unittest.TestCase):
    RESULT = {
        "segments": [
            {"start": 0.0, "end": 1.5, "text": "你好"},
            {"start": 1.6, "end": 3.0, "text": "世界"},
        ],
        "language": "yue",
    }

    def test_srt_has_index_and_arrow(self):
        out = render_result(self.RESULT, "srt", {})
        self.assertIn("1\n00:00:00,000 --> 00:00:01,500", out)
        self.assertIn("你好", out)

    def test_vtt_header(self):
        self.assertTrue(render_result(self.RESULT, "vtt", {}).startswith("WEBVTT"))

    def test_txt_is_plain_lines(self):
        self.assertEqual(render_result(self.RESULT, "txt", {}).splitlines(), ["你好", "世界"])

    def test_all_is_rejected(self):
        with self.assertRaises(ValueError):
            render_result(self.RESULT, "all", {})


class TestOffsetTimes(unittest.TestCase):
    def test_offsets_segments_and_tokens(self):
        result = {
            "segments": [
                {"start": 1.0, "end": 2.0, "text": "a",
                 "words": [{"word": "a", "start": 1.0, "end": 2.0}]},
            ],
            "word_segments": [{"word": "a", "start": 1.0, "end": 2.0}],
        }
        _offset_result_times(result, 30.0)
        seg = result["segments"][0]
        self.assertEqual((seg["start"], seg["end"]), (31.0, 32.0))
        self.assertEqual(seg["words"][0]["start"], 31.0)
        self.assertEqual(result["word_segments"][0]["end"], 32.0)

    def test_zero_offset_is_noop(self):
        result = {"segments": [{"start": 5.0, "end": 6.0, "text": "a"}]}
        _offset_result_times(result, 0.0)
        self.assertEqual(result["segments"][0]["start"], 5.0)


class TestClipArgs(unittest.TestCase):
    def test_start_and_end(self):
        pre, post = _clip_ffmpeg_args(5.0, 12.0)
        self.assertEqual(pre, ["-ss", "5.000"])
        self.assertEqual(post, ["-t", "7.000"])

    def test_end_only(self):
        pre, post = _clip_ffmpeg_args(None, 8.0)
        self.assertEqual(pre, [])
        self.assertEqual(post, ["-t", "8.000"])

    def test_none_is_empty(self):
        self.assertEqual(_clip_ffmpeg_args(None, None), ([], []))

    def test_inverted_range_rejected(self):
        with self.assertRaises(ValueError):
            _clip_ffmpeg_args(10.0, 5.0)

    def test_negative_rejected(self):
        with self.assertRaises(ValueError):
            _clip_ffmpeg_args(-1.0, None)


class TestValidateInputFile(unittest.TestCase):
    def test_missing_file(self):
        with self.assertRaises(InputError):
            validate_input_file(os.path.join(tempfile.gettempdir(), "nope-does-not-exist.wav"))

    def test_empty_file(self):
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            with self.assertRaises(InputError):
                validate_input_file(path)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()

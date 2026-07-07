"""Tests for the layered CLI config system (cantocaptions_ai/pipeline/cli_config.py).

Uses the real parser built by cantocaptions_ai.__main__.build_parser() so
these tests can't drift out of sync with the actual flag set, but never
invokes sys.argv or the heavy pipeline itself.
"""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cantocaptions_ai.__main__ import build_parser
from cantocaptions_ai.pipeline.cli_config import (
    ensure_default_cfg_exists,
    load_cfg_file,
    resolve_cfg_path,
    resolve_pipeline_args,
)
from cantocaptions_ai.pipeline.config import PipelineConfig


def _silent_parser_error():
    """Redirect stderr so parser.error() (usage + message, then sys.exit(2))
    doesn't spam test output."""
    return contextlib.redirect_stderr(io.StringIO())


class TestPipelineConfigDefaults(unittest.TestCase):
    def test_covers_every_field(self):
        from dataclasses import fields
        defaults = PipelineConfig.defaults()
        self.assertEqual(set(defaults.keys()), {f.name for f in fields(PipelineConfig)})

    def test_device_prefers_cuda(self):
        with mock.patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(PipelineConfig.defaults()["device"], "cuda")

    def test_device_falls_back_to_mps(self):
        with mock.patch("torch.cuda.is_available", return_value=False), \
             mock.patch("torch.backends.mps.is_available", return_value=True):
            self.assertEqual(PipelineConfig.defaults()["device"], "mps")

    def test_device_falls_back_to_cpu(self):
        with mock.patch("torch.cuda.is_available", return_value=False), \
             mock.patch("torch.backends.mps.is_available", return_value=False):
            self.assertEqual(PipelineConfig.defaults()["device"], "cpu")


class TestLoadCfgFile(unittest.TestCase):
    def setUp(self):
        self.parser = build_parser()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _write(self, body: str) -> Path:
        path = Path(self.tmp.name) / "test.cfg"
        path.write_text(body, encoding="utf-8")
        return path

    def test_valid_partial_override(self):
        path = self._write("[pipeline]\ndevice = cpu\nbatch_size = 8\n")
        result = load_cfg_file(path, self.parser)
        self.assertEqual(result, {"device": "cpu", "batch_size": 8})

    def test_bool_flag_coerced(self):
        path = self._write("[pipeline]\nsuppress_numerals = True\n")
        result = load_cfg_file(path, self.parser)
        self.assertIs(result["suppress_numerals"], True)

    def test_unknown_key_fails_fast(self):
        path = self._write("[pipeline]\nnot_a_real_field = 1\n")
        with _silent_parser_error():
            with self.assertRaises(SystemExit):
                load_cfg_file(path, self.parser)

    def test_bad_choice_fails_fast(self):
        path = self._write("[pipeline]\nasr_compute_type = bogus\n")
        with _silent_parser_error():
            with self.assertRaises(SystemExit):
                load_cfg_file(path, self.parser)

    def test_bad_type_fails_fast(self):
        path = self._write("[pipeline]\ndevice_index = not_an_int\n")
        with _silent_parser_error():
            with self.assertRaises(SystemExit):
                load_cfg_file(path, self.parser)

    def test_missing_section_fails_fast(self):
        path = self._write("[wrong_section]\ndevice = cpu\n")
        with _silent_parser_error():
            with self.assertRaises(SystemExit):
                load_cfg_file(path, self.parser)


class TestEnsureDefaultCfgExists(unittest.TestCase):
    def test_creates_file_with_all_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "config"
            path = ensure_default_cfg_exists(config_dir)
            self.assertTrue(path.is_file())
            import configparser
            cp = configparser.ConfigParser()
            cp.read(path)
            from dataclasses import fields
            self.assertEqual(set(cp["pipeline"].keys()), {f.name for f in fields(PipelineConfig)})

    def test_does_not_clobber_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "config"
            path = ensure_default_cfg_exists(config_dir)
            path.write_text("[pipeline]\ndevice = marker_value\n", encoding="utf-8")

            ensure_default_cfg_exists(config_dir)

            self.assertIn("marker_value", path.read_text(encoding="utf-8"))


class TestResolveCfgPath(unittest.TestCase):
    def setUp(self):
        self.parser = build_parser()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config_dir = Path(self.tmp.name) / "config"

    def test_missing_named_cfg_fails_fast(self):
        with _silent_parser_error():
            with self.assertRaises(SystemExit):
                resolve_cfg_path("does_not_exist", self.parser, self.config_dir)

    def test_none_creates_default(self):
        path = resolve_cfg_path(None, self.parser, self.config_dir)
        self.assertEqual(path, self.config_dir / "default.cfg")
        self.assertTrue(path.is_file())

    def test_strips_redundant_extension(self):
        self.config_dir.mkdir(parents=True)
        (self.config_dir / "cpu.cfg").write_text("[pipeline]\ndevice = cpu\n", encoding="utf-8")
        path = resolve_cfg_path("cpu.cfg", self.parser, self.config_dir)
        self.assertEqual(path, self.config_dir / "cpu.cfg")

    def test_rejects_path_traversal(self):
        with _silent_parser_error():
            with self.assertRaises(SystemExit):
                resolve_cfg_path("../escape", self.parser, self.config_dir)


class TestResolvePipelineArgs(unittest.TestCase):
    def setUp(self):
        self.parser = build_parser()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config_dir = Path(self.tmp.name) / "config"

    def test_no_cfg_no_preset_no_explicit_equals_defaults(self):
        merged = resolve_pipeline_args(self.parser, {}, self.config_dir)
        self.assertEqual(merged, PipelineConfig.defaults())

    def test_cfg_file_overrides_default(self):
        self.config_dir.mkdir(parents=True)
        (self.config_dir / "cpu.cfg").write_text("[pipeline]\ndevice = cpu\n", encoding="utf-8")
        merged = resolve_pipeline_args(self.parser, {"cfg": "cpu"}, self.config_dir)
        self.assertEqual(merged["device"], "cpu")

    def test_preset_overrides_cfg_file(self):
        self.config_dir.mkdir(parents=True)
        (self.config_dir / "cpu.cfg").write_text(
            "[pipeline]\nalign_compute_type = float32\n", encoding="utf-8",
        )
        merged = resolve_pipeline_args(
            self.parser, {"cfg": "cpu", "align": "fast"}, self.config_dir,
        )
        self.assertEqual(merged["align_compute_type"], "float16")

    def test_explicit_granular_flag_overrides_preset(self):
        merged = resolve_pipeline_args(
            self.parser,
            {"align": "fast", "align_compute_type": "float32"},
            self.config_dir,
        )
        self.assertEqual(merged["align_compute_type"], "float32")

    def test_explicit_granular_flag_overrides_preset_regardless_of_order(self):
        # Same as above but with keys inserted in the opposite order --
        # merge correctness must not depend on dict insertion order.
        merged = resolve_pipeline_args(
            self.parser,
            {"align_compute_type": "float32", "align": "fast"},
            self.config_dir,
        )
        self.assertEqual(merged["align_compute_type"], "float32")

    def test_missing_cfg_name_fails_fast(self):
        with _silent_parser_error():
            with self.assertRaises(SystemExit):
                resolve_pipeline_args(self.parser, {"cfg": "does_not_exist"}, self.config_dir)


if __name__ == "__main__":
    unittest.main()

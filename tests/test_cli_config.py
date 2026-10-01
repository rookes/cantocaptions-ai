"""Tests for the layered CLI config system (cantocaptions_ai/pipeline/cli_config.py).

Uses the real parser built by cantocaptions_ai.__main__.build_parser() so
these tests can't drift out of sync with the actual flag set, but never
invokes sys.argv or the heavy pipeline itself.
"""

import contextlib
import io
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

from cantocaptions_ai.__main__ import build_parser
from cantocaptions_ai.pipeline import cli_config
from cantocaptions_ai.pipeline.cli_config import (
    PRESETS_DIR,
    default_config_dir,
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


class TestShippedDefaultCfg(unittest.TestCase):
    """presets/default.cfg ships with the package, so it must stay loadable and in step
    with PipelineConfig -- a divergence makes --help state a default no CLI run uses."""

    def _path(self):
        return PRESETS_DIR / "default.cfg"

    def test_loads_without_error(self):
        # Regression: configparser does NOT strip trailing '#' comments by default, so
        # every annotated value used to reach type=/choices= as a run-on string and the
        # shipped file aborted the run outright.
        parser = build_parser()
        with _silent_parser_error():
            loaded = load_cfg_file(self._path(), parser)
        self.assertEqual(loaded["attn_implementation"], "sdpa")
        self.assertEqual(loaded["batch_size"], 8)
        self.assertIsNone(loaded["model"])  # the language's own model
        self.assertIsNone(loaded["realign"])

    def test_every_value_matches_the_dataclass_default(self):
        parser = build_parser()
        with _silent_parser_error():
            loaded = load_cfg_file(self._path(), parser)
        defaults = PipelineConfig.defaults()
        self.assertEqual(
            {k: v for k, v in loaded.items() if defaults.get(k) != v}, {}
        )

    def test_omits_only_device_and_hf_token(self):
        # device must stay absent so the dataclass's cuda>mps>cpu detection still runs
        # on a machine without a GPU; hf_token so a secret never lands in a tracked file.
        parser = build_parser()
        with _silent_parser_error():
            loaded = load_cfg_file(self._path(), parser)
        missing = set(PipelineConfig.defaults()) - set(loaded)
        self.assertEqual(missing, {"device", "hf_token"})


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
        path = self._write("[pipeline]\nno_align = True\n")
        result = load_cfg_file(path, self.parser)
        self.assertIs(result["no_align"], True)

    def test_removed_key_is_skipped_with_a_warning(self):
        # A cfg written before a field was deleted must still load, not abort the run.
        # catch_warnings, not assertWarns: assertWarns walks sys.modules and trips over
        # transformers' lazy submodules once another test has imported transformers.
        path = self._write("[pipeline]\nfp16 = True\nbatch_size = 4\n")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = load_cfg_file(path, self.parser)
        self.assertEqual(result, {"batch_size": 4})
        self.assertTrue(any("fp16" in str(w.message) for w in caught))

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

    def test_none_is_the_shipped_default_and_creates_nothing(self):
        path = resolve_cfg_path(None, self.parser, self.config_dir)
        self.assertEqual(path, PRESETS_DIR / "default.cfg")
        self.assertFalse(self.config_dir.exists())

    def test_your_own_default_replaces_the_shipped_one(self):
        self.config_dir.mkdir(parents=True)
        (self.config_dir / "default.cfg").write_text("[pipeline]\ndevice = cpu\n", encoding="utf-8")
        self.assertEqual(resolve_cfg_path(None, self.parser, self.config_dir),
                         self.config_dir / "default.cfg")

    def test_a_named_preset_falls_back_to_the_shipped_one(self):
        self.assertEqual(resolve_cfg_path("cpu", self.parser, self.config_dir),
                         PRESETS_DIR / "cpu.cfg")
        self.assertEqual(cli_config.shipped_presets(), ["cpu", "fast_test"])

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

    def test_user_cfg_overrides_the_selected_cfg(self):
        self.config_dir.mkdir(parents=True)
        (self.config_dir / "cpu.cfg").write_text(
            "[pipeline]\ndevice = cpu\nbatch_size = 2\n", encoding="utf-8")
        (self.config_dir / "user.cfg").write_text(
            "[pipeline]\nbatch_size = 24\n", encoding="utf-8")
        merged = resolve_pipeline_args(self.parser, {"cfg": "cpu"}, self.config_dir)
        self.assertEqual((merged["device"], merged["batch_size"]), ("cpu", 24))

    def test_presets_and_flags_override_user_cfg(self):
        self.config_dir.mkdir(parents=True)
        (self.config_dir / "user.cfg").write_text(
            "[pipeline]\nalign_compute_type = float32\nbatch_size = 24\n", encoding="utf-8")
        merged = resolve_pipeline_args(
            self.parser, {"align": "fast", "batch_size": 4}, self.config_dir)
        self.assertEqual((merged["align_compute_type"], merged["batch_size"]), ("float16", 4))


class TestConfigDirDiscovery(unittest.TestCase):
    """Running outside the repo must read the checkout's config, never scatter new ones."""

    def setUp(self):
        self.parser = build_parser()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cwd = Path(self.tmp.name)

    def test_cwd_config_wins(self):
        (self.cwd / "config").mkdir()
        with mock.patch.object(Path, "cwd", return_value=self.cwd):
            self.assertEqual(default_config_dir(), self.cwd / "config")

    def test_falls_back_to_the_checkout_and_creates_nothing_here(self):
        with mock.patch.object(Path, "cwd", return_value=self.cwd):
            self.assertEqual(default_config_dir(), cli_config._REPO_CONFIG_DIR)
        self.assertFalse((self.cwd / "config").exists())

    def test_the_environment_variable_wins(self):
        (self.cwd / "config").mkdir()
        with mock.patch.object(Path, "cwd", return_value=self.cwd), \
                mock.patch.dict("os.environ", {"CANTOCAPTIONS_CONFIG_DIR": str(self.cwd / "mine")}):
            self.assertEqual(default_config_dir(), self.cwd / "mine")

    def test_an_installed_package_reads_the_user_config_dir(self):
        user_dir = self.cwd / "home-config"
        user_dir.mkdir()
        with mock.patch.object(Path, "cwd", return_value=self.cwd), \
                mock.patch.object(cli_config, "_REPO_CONFIG_DIR", self.cwd / "absent"), \
                mock.patch.object(cli_config, "_user_config_dir", return_value=user_dir):
            self.assertEqual(default_config_dir(), user_dir)

    def test_with_no_config_dir_the_shipped_presets_still_apply(self):
        with mock.patch.object(Path, "cwd", return_value=self.cwd), \
                mock.patch.object(cli_config, "_REPO_CONFIG_DIR", self.cwd / "absent"), \
                mock.patch.object(cli_config, "_user_config_dir", return_value=self.cwd / "none"):
            self.assertIsNone(default_config_dir())
            self.assertEqual(resolve_cfg_path(None, self.parser), PRESETS_DIR / "default.cfg")
            self.assertEqual(resolve_cfg_path("cpu", self.parser), PRESETS_DIR / "cpu.cfg")
            merged = resolve_pipeline_args(self.parser, {})
        self.assertEqual(merged, PipelineConfig.defaults())


if __name__ == "__main__":
    unittest.main()

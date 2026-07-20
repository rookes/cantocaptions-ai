"""Tests for the offline-fallback helpers (cantocaptions_ai/utils/model_utils.py).

When the machine is offline but all models are cached, HuggingFace loaders can raise
on their revision check instead of falling back to the cache. load_with_offline_fallback
retries such a failure against the local cache: it enables process-wide offline mode
and, if the loader exposes a local_files_only/model_cache_only flag, forces it True.

All CPU-only; no network or model files are touched (loaders are faked).
"""

import os
import unittest
from unittest import mock

import requests

import cantocaptions_ai.utils.model_utils as mu
from cantocaptions_ai.utils.model_utils import (
    load_with_offline_fallback,
    enable_hf_offline,
    _looks_offline,
)


class TestLooksOffline(unittest.TestCase):
    def test_typed_hub_and_requests_errors_are_offline(self):
        self.assertTrue(_looks_offline(requests.exceptions.ConnectionError("x")))
        self.assertTrue(_looks_offline(requests.exceptions.Timeout("x")))

    def test_transformers_oserror_message_sniff(self):
        # transformers wraps connectivity problems in a plain OSError.
        self.assertTrue(_looks_offline(OSError("We couldn't connect to 'https://huggingface.co'")))
        self.assertTrue(_looks_offline(OSError("Can't load processor for 'some/model'")))

    def test_unrelated_errors_are_not_offline(self):
        self.assertFalse(_looks_offline(OSError("permission denied")))
        self.assertFalse(_looks_offline(ValueError("bad arg")))


class TestEnableHfOffline(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        for k in self._saved:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_sets_env_and_constant(self):
        with mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", False):
            import huggingface_hub.constants as c
            enable_hf_offline()
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
            self.assertTrue(c.HF_HUB_OFFLINE)


class TestLoadWithOfflineFallback(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        for k in self._saved:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_no_error_passes_through_without_offline(self):
        result = load_with_offline_fallback(lambda local_files_only=False: ("ok", local_files_only))
        self.assertEqual(result, ("ok", False))
        self.assertNotIn("HF_HUB_OFFLINE", os.environ)

    def test_retries_and_flips_local_files_only(self):
        calls = []

        def loader(local_files_only=False):
            calls.append(local_files_only)
            if len(calls) == 1:
                raise requests.exceptions.ConnectionError("Max retries exceeded")
            return "MODEL"

        result = load_with_offline_fallback(loader, local_files_only=False)
        self.assertEqual(result, "MODEL")
        self.assertEqual(calls, [False, True])  # flipped True on retry
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")

    def test_retries_and_flips_model_cache_only(self):
        calls = []

        def loader(model_cache_only=False):
            calls.append(model_cache_only)
            if len(calls) == 1:
                raise OSError("Can't load processor for 'x'")
            return "MODEL"

        self.assertEqual(load_with_offline_fallback(loader, model_cache_only=False), "MODEL")
        self.assertEqual(calls, [False, True])

    def test_retry_without_flag_still_recovers(self):
        # A loader with no offline kwarg (e.g. NeMo) still gets a retry after
        # enable_hf_offline() has flipped the process into offline mode.
        calls = []

        def nemo_like():
            calls.append(1)
            if len(calls) == 1:
                raise requests.exceptions.ConnectionError("down")
            return "MODEL"

        self.assertEqual(load_with_offline_fallback(nemo_like), "MODEL")
        self.assertEqual(len(calls), 2)
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")

    def test_unrelated_error_propagates_without_retry(self):
        calls = []

        def loader():
            calls.append(1)
            raise ValueError("unrelated")

        with self.assertRaises(ValueError):
            load_with_offline_fallback(loader)
        self.assertEqual(len(calls), 1)  # not retried
        self.assertNotIn("HF_HUB_OFFLINE", os.environ)

    def test_genuinely_missing_reraises_clean_local_error(self):
        # Offline + not cached: the retry re-raises rather than looping.
        calls = []

        def loader(local_files_only=False):
            calls.append(local_files_only)
            raise requests.exceptions.ConnectionError("still down")

        with self.assertRaises(requests.exceptions.ConnectionError):
            load_with_offline_fallback(loader, local_files_only=False)
        self.assertEqual(calls, [False, True])  # exactly one retry, then give up


if __name__ == "__main__":
    unittest.main()

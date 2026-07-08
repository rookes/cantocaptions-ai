"""Tests for BatchExecutor and MemoryPolicy (cantocaptions_ai/utils/model_utils.py).

BatchExecutor owns the batched hot loop that used to be run_adaptive_batches plus
each stage's ad-hoc ordering: ordering policy, fixed-size batching, OOM-adaptive
halving, optional flush cadence, and progress reporting. MemoryPolicy bundles the
vram_checks/headroom knobs. All CPU-only; CUDA-dependent branches
(empty_cache/flush) are exercised by patching torch.cuda.
"""

import unittest
from unittest import mock

import cantocaptions_ai.utils.model_utils as mu
from cantocaptions_ai.utils.model_utils import BatchExecutor, MemoryPolicy


class _Reporter:
    def __init__(self):
        self.advanced = 0

    def advance(self, n):
        self.advanced += n


class TestBatchExecutorOrdering(unittest.TestCase):
    def _order_seen(self, jobs, **kw):
        seen = []
        BatchExecutor(2, **kw).run(list(jobs), lambda batch: seen.extend(batch))
        return seen

    def test_input_order_by_default(self):
        self.assertEqual(self._order_seen([3, 1, 2, 5, 4]), [3, 1, 2, 5, 4])

    def test_order_key_longest_first(self):
        self.assertEqual(self._order_seen([3, 1, 2, 5, 4], order_key=lambda x: x), [5, 4, 3, 2, 1])

    def test_order_key_ascending_when_desc_false(self):
        self.assertEqual(
            self._order_seen([3, 1, 2, 5, 4], order_key=lambda x: x, order_desc=False),
            [1, 2, 3, 4, 5],
        )

    def test_does_not_mutate_caller_list(self):
        jobs = [3, 1, 2]
        BatchExecutor(2, order_key=lambda x: x).run(jobs, lambda b: None)
        self.assertEqual(jobs, [3, 1, 2])

    def test_batches_are_fixed_size(self):
        batches = []
        BatchExecutor(2).run([1, 2, 3, 4, 5], lambda b: batches.append(list(b)))
        self.assertEqual(batches, [[1, 2], [3, 4], [5]])

    def test_none_or_zero_batch_size_is_single_batch(self):
        batches = []
        BatchExecutor(None).run([1, 2, 3], lambda b: batches.append(list(b)))
        self.assertEqual(batches, [[1, 2, 3]])

    def test_empty_jobs_is_noop(self):
        calls = []
        BatchExecutor(2).run([], lambda b: calls.append(b))
        self.assertEqual(calls, [])

    def test_reporter_advances_by_batch_len(self):
        r = _Reporter()
        BatchExecutor(2).run([1, 2, 3, 4, 5], lambda b: None, reporter=r)
        self.assertEqual(r.advanced, 5)


class TestBatchExecutorOomHalving(unittest.TestCase):
    def test_oom_halves_and_retries_same_work(self):
        state = {"failed": False}
        seen = []

        def infer_fn(batch):
            if not state["failed"]:
                state["failed"] = True
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            seen.extend(batch)

        BatchExecutor(4).run([1, 2, 3, 4], infer_fn)
        # halved from 4 -> 2; the failed batch is retried, so all four still processed.
        self.assertEqual(sorted(seen), [1, 2, 3, 4])

    def test_non_oom_runtimeerror_propagates(self):
        def infer_fn(batch):
            raise RuntimeError("some unrelated failure")

        with self.assertRaises(RuntimeError) as cm:
            BatchExecutor(2).run([1, 2], infer_fn)
        self.assertIn("unrelated", str(cm.exception))

    def test_oom_at_batch_size_1_raises_actionable(self):
        def infer_fn(batch):
            raise RuntimeError("CUDA out of memory")

        with self.assertRaises(RuntimeError) as cm:
            BatchExecutor(1).run([1, 2], infer_fn)
        self.assertIn("batch_size=1", str(cm.exception))


class TestBatchExecutorFlush(unittest.TestCase):
    def test_flush_every_calls_empty_cache_on_cadence(self):
        with mock.patch.object(mu.torch.cuda, "is_available", return_value=True), \
             mock.patch.object(mu.torch.cuda, "empty_cache") as ec:
            BatchExecutor(1, flush_every=2).run([1, 2, 3, 4, 5], lambda b: None)
        # 5 single-item batches; flush after the 2nd and 4th -> 2 flushes.
        self.assertEqual(ec.call_count, 2)

    def test_no_flush_by_default(self):
        with mock.patch.object(mu.torch.cuda, "is_available", return_value=True), \
             mock.patch.object(mu.torch.cuda, "empty_cache") as ec:
            BatchExecutor(1).run([1, 2, 3], lambda b: None)
        ec.assert_not_called()


class TestMemoryPolicy(unittest.TestCase):
    def test_enabled_reflects_flag(self):
        self.assertTrue(MemoryPolicy(vram_checks=True).enabled)
        self.assertFalse(MemoryPolicy(vram_checks=False).enabled)

    def test_disabled_warn_is_noop(self):
        # Disabled: warn short-circuits before ever calling check_vram_headroom.
        with mock.patch.object(mu, "check_vram_headroom") as chk:
            result = MemoryPolicy(vram_checks=False).warn("stage", "cpu", 100.0, "fix it")
        self.assertIsNone(result)
        chk.assert_not_called()

    def test_enabled_warn_off_cuda_returns_none(self):
        # Enabled but device is cpu -> vram_stats None -> check_vram_headroom returns None.
        self.assertIsNone(MemoryPolicy(vram_checks=True).warn("stage", "cpu", 100.0, "fix it"))

    def test_cap_after_load_noop_when_headroom_zero(self):
        with mock.patch.object(mu, "cap_cuda_memory") as cap:
            MemoryPolicy(vram_checks=True, headroom_mb=0).cap_after_load("cpu")
        # cap_after_load always delegates; cap_cuda_memory itself no-ops at <= 0.
        cap.assert_called_once_with("cpu", 0.0)

    def test_cap_after_load_off_cuda_does_not_raise(self):
        # Real cap_cuda_memory: cpu device -> vram_stats None -> silent no-op.
        MemoryPolicy(vram_checks=True, headroom_mb=512).cap_after_load("cpu")


if __name__ == "__main__":
    unittest.main()

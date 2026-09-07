import subprocess
import unittest
from unittest.mock import patch

from singularity_placeholder.monitor import GPU, MonitorError, NvidiaMonitor, Snapshot, select_gpus


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.inventory = tuple(GPU(index, f"GPU-test{index}", 0, 0) for index in range(8))

    def test_default_selects_exact_allocation(self):
        self.assertEqual(select_gpus(self.inventory, environ={}), tuple(g.uuid for g in self.inventory))

    def test_explicit_selection_intersects_both_masks(self):
        env = {"NVIDIA_VISIBLE_DEVICES": "GPU-test1,GPU-test3,GPU-test5", "CUDA_VISIBLE_DEVICES": "GPU-test3,GPU-test5"}
        self.assertEqual(select_gpus(self.inventory, count=1, selectors="GPU-test5", environ=env), ("GPU-test5",))
        with self.assertRaises(MonitorError):
            select_gpus(self.inventory, count=1, selectors="GPU-test1", environ=env)

    def test_explicit_indices_are_nvidia_smi_indices(self):
        self.assertEqual(select_gpus(self.inventory, count=2, selectors="2,6", environ={}), ("GPU-test2", "GPU-test6"))

    def test_complete_numeric_masks_are_safe(self):
        self.assertEqual(len(select_gpus(self.inventory, environ={"CUDA_VISIBLE_DEVICES": "7,6,5,4,3,2,1,0"})), 8)

    def test_partial_numeric_masks_fail_closed(self):
        for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
            with self.subTest(name=name), self.assertRaisesRegex(MonitorError, "ambiguous"):
                select_gpus(self.inventory, count=2, environ={name: "0,1"})

    def test_duplicate_and_mixed_masks_fail(self):
        for raw in ("0,0", "0,GPU-test1", "GPU-test1,", "GPU-test1,GPU-test1"):
            with self.subTest(raw=raw), self.assertRaises(MonitorError):
                select_gpus(self.inventory, count=2, environ={"CUDA_VISIBLE_DEVICES": raw})

    def test_disabled_visibility_does_not_fall_back_to_all(self):
        for raw in ("", "none", "void", "-1"):
            with self.subTest(raw=raw), self.assertRaises(MonitorError):
                select_gpus(self.inventory, environ={"NVIDIA_VISIBLE_DEVICES": raw})

    def test_extra_or_missing_gpus_are_not_silently_selected(self):
        with self.assertRaisesRegex(MonitorError, "exactly 4"):
            select_gpus(self.inventory, count=4, environ={})
        with self.assertRaises(MonitorError):
            select_gpus(self.inventory[:2], environ={})

    def test_mig_is_rejected(self):
        with self.assertRaisesRegex(MonitorError, "MIG"):
            select_gpus((GPU(0, "GPU-test", 0, 0, "Enabled"),), count=1, environ={})
        with self.assertRaisesRegex(MonitorError, "MIG"):
            select_gpus(self.inventory, count=1, selectors="MIG-123", environ={})

    def test_ambiguous_uuid_prefix_is_rejected(self):
        with self.assertRaises(MonitorError):
            select_gpus(self.inventory, count=1, selectors="GPU-test", environ={})


class MonitorTests(unittest.TestCase):
    def result(self, stdout, returncode=0, stderr=""):
        return subprocess.CompletedProcess([], returncode, stdout, stderr)

    @patch("singularity_placeholder.monitor.subprocess.run")
    def test_parses_uuid_activity_and_foreign_processes(self, run):
        run.side_effect = [
            self.result("0, GPU-A, 0, 100, Disabled\n1, GPU-B, 20, 1200, Disabled\n"),
            self.result("GPU-A, 111\nGPU-A, 222\nGPU-B, 333\n"),
        ]
        snapshot = NvidiaMonitor().sample(("GPU-A",))
        self.assertEqual(snapshot.foreign_pids({111}), {222})
        self.assertTrue(snapshot.idle(5, 256))
        self.assertIn("--query-compute-apps=gpu_uuid,pid", run.call_args.args[0])

    @patch("singularity_placeholder.monitor.subprocess.run")
    def test_empty_process_list_is_valid(self, run):
        run.side_effect = [self.result("0, GPU-A, 0, 0, Disabled\n"), self.result("")]
        self.assertEqual(NvidiaMonitor().sample(("GPU-A",)).processes, ())

    @patch("singularity_placeholder.monitor.subprocess.run")
    def test_unknown_utilization_fails_closed(self, run):
        run.return_value = self.result("0, GPU-A, N/A, 0, Disabled\n")
        with self.assertRaises(MonitorError):
            NvidiaMonitor().inventory()

    @patch("singularity_placeholder.monitor.subprocess.run")
    def test_permission_failure_is_not_idle(self, run):
        run.return_value = self.result("", 1, "Insufficient Permissions")
        with self.assertRaisesRegex(MonitorError, "Permissions"):
            NvidiaMonitor().inventory()

    @patch("singularity_placeholder.monitor.subprocess.run")
    def test_timeout_is_not_idle(self, run):
        run.side_effect = subprocess.TimeoutExpired("nvidia-smi", 10)
        with self.assertRaises(MonitorError):
            NvidiaMonitor().inventory()

    @patch("singularity_placeholder.monitor.subprocess.run")
    def test_gpu_disappearing_fails_closed(self, run):
        run.return_value = self.result("0, GPU-A, 0, 0, Disabled\n")
        with self.assertRaisesRegex(MonitorError, "allocation changed"):
            NvidiaMonitor().sample(("GPU-B",))

    @patch("singularity_placeholder.monitor.subprocess.run")
    def test_unreadable_pid_fails_closed(self, run):
        run.side_effect = [self.result("0, GPU-A, 0, 0, Disabled\n"), self.result("GPU-A, N/A\n")]
        with self.assertRaises(MonitorError):
            NvidiaMonitor().sample(("GPU-A",))

    def test_any_selected_gpu_busy_prevents_all_startup(self):
        self.assertFalse(Snapshot((GPU(0, "GPU-A", 0, 0), GPU(1, "GPU-B", 0, 300)), ()).idle(5, 256))


if __name__ == "__main__":
    unittest.main()

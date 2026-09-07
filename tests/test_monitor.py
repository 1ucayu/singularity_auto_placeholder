import subprocess
import unittest
from unittest.mock import patch

from singularity_placeholder.monitor import GPU, MonitorError, NvidiaMonitor, Snapshot, select_gpus, visibility_diagnostics


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.inventory = tuple(GPU(index, f"GPU-test{index}", 0, 0) for index in range(8))

    def test_default_selects_visible_allocation(self):
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

    def test_expected_count_does_not_truncate_or_expand_visible_allocation(self):
        self.assertEqual(
            select_gpus(self.inventory, count=4, environ={}),
            tuple(gpu.uuid for gpu in self.inventory),
        )
        self.assertEqual(
            select_gpus(self.inventory[:2], count=8, environ={}),
            ("GPU-test0", "GPU-test1"),
        )

    def test_explicit_subset_does_not_require_matching_expected_count(self):
        self.assertEqual(
            select_gpus(self.inventory, count=8, selectors="2,6", environ={}),
            ("GPU-test2", "GPU-test6"),
        )

    def test_mig_is_rejected(self):
        with self.assertRaisesRegex(MonitorError, "MIG"):
            select_gpus((GPU(0, "GPU-test", 0, 0, "Enabled"),), count=1, environ={})
        with self.assertRaisesRegex(MonitorError, "MIG"):
            select_gpus(self.inventory, count=1, selectors="MIG-123", environ={})

    def test_ambiguous_uuid_prefix_is_rejected(self):
        with self.assertRaises(MonitorError):
            select_gpus(self.inventory, count=1, selectors="GPU-test", environ={})


class CudaSelectionTests(unittest.TestCase):
    def setUp(self):
        self.inventory = tuple(GPU(index, f"GPU-test{index}", 0, 0) for index in range(8))
        self.visible = tuple(gpu.uuid for gpu in self.inventory)

    def test_actual_cuda_allocation_supersedes_stale_container_hint(self):
        for raw in ("", "none", "void", "-1", "all"):
            with self.subTest(raw=raw):
                self.assertEqual(
                    select_gpus(self.inventory, environ={"NVIDIA_VISIBLE_DEVICES": raw}, cuda_visible=self.visible),
                    self.visible,
                )

    def test_explicit_container_uuid_restriction_is_preserved(self):
        env = {"NVIDIA_VISIBLE_DEVICES": "GPU-test0"}
        actual = ("GPU-test0", "GPU-test1")
        self.assertEqual(
            select_gpus(self.inventory, count=2, environ=env, cuda_visible=actual),
            ("GPU-test0",),
        )
        with self.assertRaisesRegex(MonitorError, "outside"):
            select_gpus(self.inventory, count=1, selectors="1", environ=env, cuda_visible=actual)

    def test_container_and_cuda_uuid_restrictions_are_intersected(self):
        env = {"NVIDIA_VISIBLE_DEVICES": "GPU-test0,GPU-test1", "CUDA_VISIBLE_DEVICES": "GPU-test1,GPU-test2"}
        self.assertEqual(
            select_gpus(self.inventory, count=1, environ=env, cuda_visible=self.visible),
            ("GPU-test1",),
        )

    def test_partial_numeric_container_restriction_is_not_guessed(self):
        with self.assertRaisesRegex(MonitorError, "Partial numeric NVIDIA_VISIBLE_DEVICES=.*ambiguous"):
            select_gpus(self.inventory, count=2, environ={"NVIDIA_VISIBLE_DEVICES": "0,1"}, cuda_visible=self.visible[:2])

    def test_complete_numeric_container_hint_is_compatible_with_cuda_probe(self):
        self.assertEqual(
            select_gpus(self.inventory, environ={"NVIDIA_VISIBLE_DEVICES": "7,6,5,4,3,2,1,0"}, cuda_visible=self.visible),
            self.visible,
        )

    def test_unknown_container_uuid_restriction_is_rejected(self):
        with self.assertRaisesRegex(MonitorError, "does not identify"):
            select_gpus(self.inventory, environ={"NVIDIA_VISIBLE_DEVICES": "GPU-other-container"}, cuda_visible=self.visible)

    def test_container_filter_does_not_hide_inconsistent_numeric_cuda_probe(self):
        env = {"NVIDIA_VISIBLE_DEVICES": "GPU-test0,GPU-test1", "CUDA_VISIBLE_DEVICES": "0,1"}
        with self.assertRaisesRegex(MonitorError, "more devices than.*CUDA_VISIBLE_DEVICES"):
            select_gpus(self.inventory, count=2, environ=env, cuda_visible=self.visible)

    def test_cuda_zero_never_falls_back_to_nvidia_smi_or_container_hint(self):
        for env in ({"NVIDIA_VISIBLE_DEVICES": "all"}, {"CUDA_VISIBLE_DEVICES": ""}, {"CUDA_VISIBLE_DEVICES": "-1"}):
            with self.subTest(env=env), self.assertRaisesRegex(MonitorError, "safely resolved 0.*cuda_visible_count=0"):
                select_gpus(self.inventory, environ=env, cuda_visible=())

    def test_explicitly_disabled_cuda_mask_is_preserved(self):
        for raw in ("", "none", "void", "-1"):
            with self.subTest(raw=raw), self.assertRaisesRegex(MonitorError, "safely resolved 0"):
                select_gpus(self.inventory, environ={"CUDA_VISIBLE_DEVICES": raw}, cuda_visible=self.visible)

    def test_numeric_cuda_mask_uses_actual_uuid_mapping(self):
        # CUDA ordinal zero need not mean nvidia-smi index zero. The driver has
        # applied this process's CUDA_DEVICE_ORDER and remapped the ordinals.
        visible = ("GPU-test6", "GPU-test2")
        env = {"CUDA_VISIBLE_DEVICES": "0,1", "CUDA_DEVICE_ORDER": "FASTEST_FIRST", "NVIDIA_VISIBLE_DEVICES": ""}
        self.assertEqual(
            select_gpus(self.inventory, count=2, environ=env, cuda_visible=visible),
            ("GPU-test2", "GPU-test6"),
        )
        self.assertEqual(
            select_gpus(self.inventory, count=1, selectors="6", environ=env, cuda_visible=visible),
            ("GPU-test6",),
        )
        with self.assertRaisesRegex(MonitorError, "outside"):
            select_gpus(self.inventory, count=1, selectors="0", environ=env, cuda_visible=visible)

    def test_uuid_cuda_mask_still_constrains_actual_allocation(self):
        self.assertEqual(
            select_gpus(self.inventory, count=1, environ={"CUDA_VISIBLE_DEVICES": "GPU-test5"}, cuda_visible=self.visible),
            ("GPU-test5",),
        )

    def test_unknown_or_duplicate_cuda_uuid_is_rejected(self):
        for visible in (("GPU-not-in-inventory",), ("GPU-test0", "GPU-test0")):
            with self.subTest(visible=visible), self.assertRaises(MonitorError):
                select_gpus(self.inventory, count=len(visible), environ={}, cuda_visible=visible)

    def test_missing_uuid_in_cuda_mask_is_rejected(self):
        with self.assertRaisesRegex(MonitorError, "does not identify"):
            select_gpus(self.inventory, count=1, environ={"CUDA_VISIBLE_DEVICES": "GPU-not-present"}, cuda_visible=self.visible)

    def test_mig_cuda_allocation_is_rejected(self):
        with self.assertRaisesRegex(MonitorError, "MIG"):
            select_gpus(self.inventory, count=1, environ={}, cuda_visible=("MIG-slice0",))
        with self.assertRaisesRegex(MonitorError, "MIG"):
            select_gpus((GPU(0, "GPU-test0", 0, 0, "Enabled"),), count=1, environ={}, cuda_visible=("GPU-test0",))

    def test_fewer_cuda_visible_gpus_remain_usable(self):
        for actual_count in (1, 4, 7):
            with self.subTest(actual_count=actual_count):
                actual = self.visible[:actual_count]
                self.assertEqual(
                    select_gpus(self.inventory, count=8, environ={}, cuda_visible=actual),
                    actual,
                )

    def test_lower_expected_count_does_not_truncate_cuda_allocation(self):
        self.assertEqual(
            select_gpus(self.inventory, count=4, environ={}, cuda_visible=self.visible),
            self.visible,
        )

    def test_expected_count_never_adds_gpus_outside_cuda_allocation(self):
        actual = ("GPU-test2", "GPU-test6")
        self.assertEqual(
            select_gpus(self.inventory, count=8, environ={}, cuda_visible=actual),
            actual,
        )
        with self.assertRaisesRegex(MonitorError, "outside"):
            select_gpus(self.inventory, count=8, selectors="0,2,6", environ={}, cuda_visible=actual)

    def test_invalid_cuda_masks_still_fail(self):
        for raw in ("0,0", "0,GPU-test1", "GPU-test1,", "GPU-test1,GPU-test1", "0,1"):
            with self.subTest(raw=raw), self.assertRaises(MonitorError):
                select_gpus(self.inventory, environ={"CUDA_VISIBLE_DEVICES": raw}, cuda_visible=self.visible)

    def test_diagnostics_allowlist_and_escape_environment_values(self):
        detail = visibility_diagnostics(
            {"NVIDIA_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": "0\n1", "GITHUB_TOKEN": "must-stay-secret", "OTHER": "secret"},
            inventory_count=8, cuda_visible=(),
        )
        self.assertIn("NVIDIA_VISIBLE_DEVICES=''", detail)
        self.assertIn("CUDA_VISIBLE_DEVICES='0\\n1'", detail)
        self.assertIn("CUDA_DEVICE_ORDER=<unset>", detail)
        self.assertNotIn("\n", detail)
        self.assertNotIn("secret", detail)
        self.assertIn("nvidia-smi_count=8; cuda_visible_count=0", detail)


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

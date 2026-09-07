import ctypes
import json
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid

from singularity_placeholder.cuda_probe import CudaProbeError, _driver_uuids, discover_cuda_uuids


UUIDS = (
    "GPU-01234567-89ab-cdef-0123-456789abcdef",
    "GPU-fedcba98-7654-3210-fedc-ba9876543210",
)


class IsolatedProbeTests(unittest.TestCase):
    @patch("singularity_placeholder.cuda_probe.subprocess.run")
    def test_probe_preserves_exact_environment_and_uuid_order(self, run):
        env = {"CUDA_VISIBLE_DEVICES": "7,0", "NVIDIA_VISIBLE_DEVICES": "", "GITHUB_TOKEN": "keep-private"}
        run.return_value = subprocess.CompletedProcess([], 0, json.dumps({"uuids": UUIDS}), "")
        self.assertEqual(discover_cuda_uuids(environ=env), UUIDS)
        self.assertEqual(run.call_args.kwargs["env"], env)
        self.assertIsNot(run.call_args.kwargs["env"], env)
        self.assertTrue(run.call_args.args[0][1].endswith("cuda_probe.py"))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "7,0")

    @patch("singularity_placeholder.cuda_probe.subprocess.run")
    def test_default_probe_inherits_environment(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, '{"uuids": []}', "")
        self.assertEqual(discover_cuda_uuids(), ())
        self.assertIsNone(run.call_args.kwargs["env"])

    @patch("singularity_placeholder.cuda_probe.subprocess.run")
    def test_explicit_empty_cuda_mask_is_passed_unmodified(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, '{"uuids": []}', "")
        self.assertEqual(discover_cuda_uuids(environ={"CUDA_VISIBLE_DEVICES": ""}), ())
        self.assertEqual(run.call_args.kwargs["env"], {"CUDA_VISIBLE_DEVICES": ""})

    @patch("singularity_placeholder.cuda_probe.subprocess.run")
    def test_timeout_is_reported_without_captured_output(self, run):
        run.side_effect = subprocess.TimeoutExpired("probe", 3, output="secret", stderr="another secret")
        with self.assertRaisesRegex(CudaProbeError, "timed out after 3 seconds") as raised:
            discover_cuda_uuids(timeout=3)
        self.assertNotIn("secret", str(raised.exception))

    @patch("singularity_placeholder.cuda_probe.subprocess.run")
    def test_launch_failure_is_not_an_empty_allocation(self, run):
        run.side_effect = OSError("private path")
        with self.assertRaisesRegex(CudaProbeError, "Cannot launch") as raised:
            discover_cuda_uuids()
        self.assertNotIn("private path", str(raised.exception))

    @patch("singularity_placeholder.cuda_probe.subprocess.run")
    def test_child_driver_failure_is_distinct_from_no_devices(self, run):
        run.return_value = subprocess.CompletedProcess([], 1, '{"error": "cuInit failed with CUDA driver error 999."}', "secret")
        with self.assertRaisesRegex(CudaProbeError, "cuInit.*999") as raised:
            discover_cuda_uuids()
        self.assertNotIn("secret", str(raised.exception))

    @patch("singularity_placeholder.cuda_probe.subprocess.run")
    def test_malformed_response_never_becomes_a_valid_allocation(self, run):
        for payload in ("[]", "not json", '{"uuids": "GPU-test"}', '{"uuids": [null]}', '{"uuids": ["GPU-test"]}', json.dumps({"uuids": [UUIDS[0], UUIDS[0]]})):
            with self.subTest(payload=payload), self.assertRaisesRegex(CudaProbeError, "invalid output"):
                run.return_value = subprocess.CompletedProcess([], 0, payload, "secret")
                discover_cuda_uuids()

    @patch("singularity_placeholder.cuda_probe.subprocess.run")
    def test_crashed_child_without_structured_error_does_not_echo_stderr(self, run):
        run.return_value = subprocess.CompletedProcess([], -11, "", "private crash content")
        with self.assertRaisesRegex(CudaProbeError, "exit code -11") as raised:
            discover_cuda_uuids()
        self.assertNotIn("private", str(raised.exception))


class DriverApiTests(unittest.TestCase):
    def driver(self, uuids=UUIDS, *, v2=True):
        def count(ptr):
            ptr._obj.value = len(uuids)
            return 0

        def device(ptr, ordinal):
            ptr._obj.value = ordinal + 40
            return 0

        def get_uuid(ptr, selected):
            raw = uuid.UUID(uuids[selected.value - 40][4:]).bytes
            for index, value in enumerate(raw):
                ptr._obj.bytes[index] = value
            return 0

        result = SimpleNamespace(
            cuInit=Mock(return_value=0), cuDeviceGetCount=Mock(side_effect=count),
            cuDeviceGet=Mock(side_effect=device),
        )
        setattr(result, "cuDeviceGetUuid_v2" if v2 else "cuDeviceGetUuid", Mock(side_effect=get_uuid))
        return result

    @patch("singularity_placeholder.cuda_probe.ctypes.CDLL")
    def test_driver_probe_uses_cuda_ordinals_and_v2_full_uuids(self, cdll):
        driver = self.driver()
        cdll.return_value = driver
        self.assertEqual(_driver_uuids(), UUIDS)
        cdll.assert_called_once_with("libcuda.so.1")
        driver.cuInit.assert_called_once_with(0)
        self.assertEqual(driver.cuDeviceGetUuid_v2.call_count, 2)
        self.assertEqual(driver.cuDeviceGet.restype, ctypes.c_int)

    @patch("singularity_placeholder.cuda_probe.ctypes.CDLL")
    def test_older_uuid_symbol_is_supported(self, cdll):
        driver = self.driver(v2=False)
        cdll.return_value = driver
        self.assertEqual(_driver_uuids(), UUIDS)
        self.assertEqual(driver.cuDeviceGetUuid.call_count, 2)

    @patch("singularity_placeholder.cuda_probe.ctypes.CDLL")
    def test_no_device_at_init_is_empty_without_enumeration(self, cdll):
        driver = self.driver()
        driver.cuInit.return_value = 100
        cdll.return_value = driver
        self.assertEqual(_driver_uuids(), ())
        driver.cuDeviceGetCount.assert_not_called()

    @patch("singularity_placeholder.cuda_probe.ctypes.CDLL")
    def test_init_error_does_not_fall_back_to_nvml(self, cdll):
        driver = self.driver()
        driver.cuInit.return_value = 999
        cdll.return_value = driver
        with self.assertRaisesRegex(CudaProbeError, "cuInit.*999"):
            _driver_uuids()

    @patch("singularity_placeholder.cuda_probe.ctypes.CDLL")
    def test_missing_driver_is_actionable(self, cdll):
        cdll.side_effect = OSError("not found")
        with self.assertRaisesRegex(CudaProbeError, "libcuda.so.1.*exposed"):
            _driver_uuids()

    @patch("singularity_placeholder.cuda_probe.ctypes.CDLL")
    def test_uuid_query_error_fails_closed(self, cdll):
        driver = self.driver()
        driver.cuDeviceGetUuid_v2.side_effect = None
        driver.cuDeviceGetUuid_v2.return_value = 101
        cdll.return_value = driver
        with self.assertRaisesRegex(CudaProbeError, "cuDeviceGetUuid_v2.*101"):
            _driver_uuids()

    @patch("singularity_placeholder.cuda_probe.ctypes.CDLL")
    def test_duplicate_driver_uuids_fail_closed(self, cdll):
        cdll.return_value = self.driver((UUIDS[0], UUIDS[0]))
        with self.assertRaisesRegex(CudaProbeError, "duplicate"):
            _driver_uuids()


if __name__ == "__main__":
    unittest.main()

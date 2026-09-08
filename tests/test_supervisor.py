import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from singularity_placeholder.monitor import GPU, MonitorError, Snapshot
from singularity_placeholder.supervisor import (
    Config, DaemonLock, Reservations, Supervisor, Worker, WorkerPool, atomic_json,
    ensure_local_control_dir, identity_alive, identity_record, process_identity, resolve_run_gpus,
)


class FakeMonitor:
    def __init__(self):
        self.gpus = (GPU(0, "GPU-A", 0, 0), GPU(1, "GPU-B", 0, 0))
        self.processes = ()
        self.error = None

    def inventory(self):
        return self.gpus

    def sample(self, selected):
        if self.error:
            raise self.error
        return Snapshot(self.gpus, self.processes)


class FakePool:
    def __init__(self):
        self.workers = {}
        self.stops = 0
        self.starts = 0
        self.can_stop = True
        self.exit_codes = []

    def own_pids(self, gpu):
        return {self.workers[gpu]} if gpu in self.workers else set()

    def ready_gpus(self):
        return set(self.workers)

    def observe_processes(self, processes, ready_before_sample, *, now=None):
        foreign = {}
        for gpu, pid in processes:
            if pid not in self.own_pids(gpu):
                foreign.setdefault(gpu, set()).add(pid)
        return foreign, set(), {}

    def public(self):
        return self.workers.copy()

    def exited(self):
        return self.exit_codes

    def start(self, gpus):
        self.starts += 1
        self.workers.update({gpu: 1000 + (0 if gpu == "GPU-A" else 1) for gpu in gpus})

    def stop_all(self):
        return self.stop(set(self.workers))

    def stop(self, gpus):
        self.stops += 1
        if self.can_stop:
            for gpu in gpus:
                self.workers.pop(gpu, None)
            self.exit_codes = [(gpu, code) for gpu, code in self.exit_codes if gpu not in gpus]
        return not set(gpus).intersection(self.workers)


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.control = Path(self.temp.name) / "control"
        self.monitor, self.pool = FakeMonitor(), FakePool()
        self.supervisor = Supervisor(
            Config(self.control, Path(self.temp.name) / "logs", gpu_count=2, idle_seconds=10),
            monitor=self.monitor, pool=self.pool, environ={},
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_production_discovery_uses_cuda_devices_despite_empty_nvidia_hint(self):
        self.monitor.gpus = tuple(GPU(index, f"GPU-{index}", 0, 0) for index in range(8))
        cuda = tuple(gpu.uuid for gpu in self.monitor.gpus)
        with patch("singularity_placeholder.supervisor.NvidiaMonitor", return_value=self.monitor), \
                patch("singularity_placeholder.supervisor.discover_cuda_uuids", return_value=cuda) as probe:
            supervisor = Supervisor(
                Config(self.control, Path(self.temp.name) / "logs", gpu_count=8),
                pool=self.pool, environ={"NVIDIA_VISIBLE_DEVICES": ""},
            )
        self.assertEqual(supervisor.selected, cuda)
        probe.assert_called_once_with(environ={"NVIDIA_VISIBLE_DEVICES": ""})

    def test_gpu_count_mismatch_warns_and_uses_discovered_allocation(self):
        with self.assertLogs("singularity_placeholder", level="WARNING") as logs:
            supervisor = Supervisor(
                Config(self.control, Path(self.temp.name) / "logs", gpu_count=8),
                monitor=self.monitor, pool=self.pool, environ={},
            )
        self.assertEqual(supervisor.selected, ("GPU-A", "GPU-B"))
        self.assertIn("continuing with the visible allocation", "\n".join(logs.output))

    def test_auto_count_uses_visible_allocation_without_a_count_warning(self):
        self.monitor.gpus = tuple(GPU(index, f"GPU-{index}", 0, 0) for index in range(6))
        with patch("singularity_placeholder.supervisor.LOG.warning") as warning:
            supervisor = Supervisor(
                Config(self.control, Path(self.temp.name) / "logs"),
                monitor=self.monitor, pool=self.pool, environ={}, cuda_visible=("GPU-2", "GPU-5"),
            )
        self.assertEqual(supervisor.selected, ("GPU-2", "GPU-5"))
        self.assertFalse(any("Requested" in str(call) for call in warning.call_args_list))
        self.assertIs(supervisor.pool, self.pool)

    def test_small_expected_count_does_not_truncate_usable_allocation(self):
        with self.assertLogs("singularity_placeholder", level="WARNING"):
            supervisor = Supervisor(
                Config(self.control, Path(self.temp.name) / "logs", gpu_count=1),
                monitor=self.monitor, pool=self.pool, environ={},
            )
        self.assertEqual(supervisor.selected, ("GPU-A", "GPU-B"))

    def test_configured_matrix_size_reaches_default_worker_pool(self):
        supervisor = Supervisor(
            Config(self.control, Path(self.temp.name) / "logs", matrix_size=2048),
            monitor=self.monitor, environ={},
        )
        self.assertEqual(supervisor.pool.matrix_size, 2048)

    def test_invalid_runtime_configuration_is_rejected(self):
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                Config(self.control, Path(self.temp.name) / "logs", gpu_count=count)
        for size in (255, 8193, 1024.5, "1024"):
            with self.subTest(size=size), self.assertRaises(ValueError):
                Config(self.control, Path(self.temp.name) / "logs", matrix_size=size)

    def use_isolated_pool(self, *, ready=True):
        pool = WorkerPool()
        reader = threading.Thread(target=lambda: None)
        reader.start()
        reader.join()
        for gpu, pid in (("GPU-A", 42), ("GPU-B", 43)):
            process = Mock(pid=pid)
            process.poll.return_value = None
            worker = Worker(gpu, process, set(), reader, started_at=0)
            if ready:
                worker.ready.set()
            pool.workers[gpu] = worker

        def stop(gpus):
            for gpu in gpus:
                pool.workers.pop(gpu, None)
            return True

        pool.stop = Mock(side_effect=stop)
        pool.start = Mock()
        self.supervisor.pool = pool
        self.monitor.processes = (("GPU-A", 9000), ("GPU-B", 9001))
        return pool

    def test_isolated_container_workers_do_not_churn_on_host_pids(self):
        pool = self.use_isolated_pool()
        self.assertEqual(self.supervisor.step(1)["state"], "running")
        self.assertEqual(pool.own_pids("GPU-A"), {9000})
        self.assertEqual(pool.own_pids("GPU-B"), {9001})
        self.assertEqual(self.supervisor.step(2)["state"], "running")
        self.assertEqual(set(pool.workers), {"GPU-A", "GPU-B"})
        pool.start.assert_not_called()

    def test_isolated_worker_yields_only_gpu_with_extra_foreign_process(self):
        pool = self.use_isolated_pool()
        self.supervisor.step(1)
        self.monitor.processes += (("GPU-A", 9999),)
        result = self.supervisor.step(2)
        self.assertEqual(result["state"], "external_workload")
        self.assertEqual(result["external_pids"], [9999])
        self.assertEqual(set(pool.workers), {"GPU-B"})

    def test_ready_arriving_during_sample_waits_for_a_fresh_sample(self):
        pool = self.use_isolated_pool(ready=False)
        original_sample = self.monitor.sample

        def sample(selected):
            for worker in pool.workers.values():
                worker.ready.set()
            return original_sample(selected)

        self.monitor.sample = sample
        self.assertEqual(self.supervisor.step(1)["state"], "worker_starting")
        self.assertFalse(pool.own_pids("GPU-A"))
        self.assertEqual(self.supervisor.step(2)["state"], "running")
        self.assertEqual(pool.own_pids("GPU-A"), {9000})

    def test_reservation_still_releases_an_initializing_isolated_worker(self):
        pool = self.use_isolated_pool(ready=False)
        self.request(gpus=["GPU-A"])
        result = self.supervisor.step(1)
        self.assertEqual(result["state"], "reserved")
        self.assertEqual(set(pool.workers), {"GPU-B"})
        self.assertTrue((self.control / "acks" / "req.json").exists())

    def request(self, name="req", gpus=None):
        data = {"version": 1, "id": name, "owner": identity_record(os.getpid()), "gpus": gpus or ["GPU-A", "GPU-B"]}
        atomic_json(self.control / "requests" / f"{name}.json", data)
        return data

    def start(self):
        self.assertEqual(self.supervisor.step(0)["state"], "idle_grace")
        self.assertEqual(self.supervisor.step(10)["state"], "running")
        self.assertEqual(len(self.pool.workers), 2)

    def test_idle_grace_then_start_exactly_one_worker_per_gpu(self):
        self.start()
        self.monitor.processes = (("GPU-A", 1000), ("GPU-B", 1001))
        self.monitor.gpus = (GPU(0, "GPU-A", 100, 900), GPU(1, "GPU-B", 100, 900))
        self.assertEqual(self.supervisor.step(12)["state"], "running")
        self.assertEqual(self.pool.starts, 2)

    def test_foreign_cuda_pid_stops_only_its_gpu_even_at_zero_utilization(self):
        self.start()
        self.monitor.processes = (("GPU-A", 1000), ("GPU-B", 9999))
        result = self.supervisor.step(11)
        self.assertEqual(result["state"], "external_workload")
        self.assertEqual(result["external_pids"], [9999])
        self.assertEqual(self.pool.workers, {"GPU-A": 1000})
        self.assertEqual(self.supervisor.step(1000)["state"], "external_workload")
        self.assertEqual(self.pool.starts, 2)

    def test_monitor_failure_stops_workers_and_cannot_ack_running_workers(self):
        self.start()
        self.request()
        self.monitor.error = MonitorError("driver unavailable")
        self.pool.can_stop = False
        self.assertEqual(self.supervisor.step(11)["state"], "stopping")
        self.assertFalse((self.control / "acks" / "req.json").exists())
        self.pool.can_stop = True
        self.assertEqual(self.supervisor.step(12)["state"], "monitor_error")
        ack = json.loads((self.control / "acks" / "req.json").read_text())
        self.assertTrue(ack["workers_stopped"])
        self.assertEqual(self.pool.workers, {})

    def test_concurrent_reservations_hold_until_all_are_released(self):
        self.start()
        self.request("one")
        self.request("two")
        result = self.supervisor.step(11)
        self.assertEqual(result["state"], "reserved")
        self.assertEqual(result["reservations"], ["one", "two"])
        self.assertTrue((self.control / "acks" / "one.json").exists())
        Reservations(self.control).remove("one")
        self.assertEqual(self.supervisor.step(100)["state"], "reserved")
        Reservations(self.control).remove("two")
        self.assertEqual(self.supervisor.step(101)["state"], "idle_grace")
        self.assertEqual(self.supervisor.step(111)["state"], "running")

    def test_busy_memory_resets_only_its_gpu_idle_grace(self):
        self.supervisor.step(0)
        self.monitor.gpus = (GPU(0, "GPU-A", 0, 1000), GPU(1, "GPU-B", 0, 0))
        self.assertEqual(self.supervisor.step(9)["state"], "gpu_busy")
        self.monitor.gpus = (GPU(0, "GPU-A", 0, 0), GPU(1, "GPU-B", 0, 0))
        self.assertEqual(self.supervisor.step(10)["state"], "idle_grace")
        self.assertEqual(self.pool.workers, {"GPU-B": 1001})
        self.assertEqual(self.supervisor.step(19)["state"], "idle_grace")
        self.assertEqual(self.supervisor.step(20)["state"], "running")

    def test_pause_stops_and_resume_obeys_grace(self):
        self.start()
        atomic_json(self.control / "paused.json", {})
        self.assertEqual(self.supervisor.step(11)["state"], "paused")
        (self.control / "paused.json").unlink()
        self.assertEqual(self.supervisor.step(12)["state"], "idle_grace")

    def test_corrupt_reservation_fails_closed(self):
        self.start()
        (self.control / "requests" / "broken.json").write_text("{")
        self.assertEqual(self.supervisor.step(11)["state"], "reservation_error")
        self.assertFalse(self.pool.workers)

    def test_worker_failure_preserves_siblings_and_backs_off(self):
        self.start()
        self.pool.exit_codes = [("GPU-A", 1)]
        self.assertEqual(self.supervisor.step(11)["state"], "worker_error")
        self.assertEqual(self.pool.workers, {"GPU-B": 1001})
        self.assertEqual(self.supervisor.step(12)["state"], "worker_backoff")
        self.assertEqual(self.pool.starts, 2)

    def test_partial_request_keeps_other_gpu_placeholder_running(self):
        self.start()
        self.request(gpus=["GPU-A"])
        self.assertEqual(self.supervisor.step(11)["state"], "reserved")
        self.assertEqual(self.pool.workers, {"GPU-B": 1001})
        ack = json.loads((self.control / "acks" / "req.json").read_text())
        self.assertEqual(ack["gpus"], ["GPU-A"])
        self.assertTrue(ack["workers_stopped"])

    def test_overlapping_requests_release_only_unreserved_gpu(self):
        self.start()
        self.request("one", ["GPU-A"])
        self.request("both", ["GPU-A", "GPU-B"])
        self.supervisor.step(11)
        self.assertFalse(self.pool.workers)
        Reservations(self.control).remove("both")
        self.supervisor.step(12)
        self.supervisor.step(22)
        self.assertEqual(self.pool.workers, {"GPU-B": 1001})
        self.assertTrue((self.control / "acks" / "one.json").exists())

    def test_pid_ownership_is_per_gpu(self):
        self.start()
        self.monitor.processes = (("GPU-A", 1001),)
        self.supervisor.step(11)
        self.assertEqual(self.pool.workers, {"GPU-B": 1001})

    def test_unallocated_request_fails_closed(self):
        self.start()
        self.request(gpus=["GPU-OTHER"])
        self.assertEqual(self.supervisor.step(11)["state"], "reservation_error")
        self.assertFalse(self.pool.workers)
        self.assertFalse((self.control / "acks" / "req.json").exists())

    def test_ack_never_precedes_worker_exit(self):
        self.start()
        self.request()
        self.pool.can_stop = False
        self.supervisor.step(11)
        self.assertFalse((self.control / "acks" / "req.json").exists())
        self.pool.can_stop = True
        self.supervisor.step(12)
        self.assertTrue((self.control / "acks" / "req.json").exists())

    def test_lock_rejects_second_supervisor(self):
        with DaemonLock(self.control), self.assertRaisesRegex(RuntimeError, "Another supervisor"):
            with DaemonLock(self.control):
                pass


class IdentityAndReservationTests(unittest.TestCase):
    def test_wrapper_selectors_preserve_cuda_order(self):
        self.assertEqual(resolve_run_gpus("1,0", ["GPU-A", "GPU-B"]), ["GPU-B", "GPU-A"])
        self.assertEqual(resolve_run_gpus(None, ["GPU-A", "GPU-B"]), ["GPU-A", "GPU-B"])
        for invalid in ("2", "0,0", "GPU-OTHER", ""):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                resolve_run_gpus(invalid, ["GPU-A", "GPU-B"])
    def test_current_process_start_identity(self):
        record = identity_record(os.getpid())
        self.assertTrue(identity_alive(record))
        self.assertFalse(identity_alive({**record, "start": "a different start"}))

    @patch("singularity_placeholder.supervisor.identity_alive")
    @patch("singularity_placeholder.supervisor.group_alive")
    def test_dead_wrapper_live_child_keeps_reservation(self, group, alive):
        alive.side_effect = [False, True]
        self.assertFalse(Reservations._stale({"owner": {}, "child": {"pid": 12}}))
        group.assert_not_called()

    @patch("singularity_placeholder.supervisor.identity_alive")
    @patch("singularity_placeholder.supervisor.group_alive")
    def test_dead_owner_and_child_but_live_group_keeps_reservation(self, group, alive):
        alive.return_value = False
        group.return_value = True
        self.assertFalse(Reservations._stale({"owner": {}, "child": {"pid": 12}}))

    @patch("singularity_placeholder.supervisor.identity_alive")
    @patch("singularity_placeholder.supervisor.group_alive")
    def test_cleanup_requires_all_identities_and_group_dead(self, group, alive):
        alive.return_value = False
        group.return_value = False
        self.assertTrue(Reservations._stale({"owner": {}, "child": {"pid": 12}}))

    @patch("singularity_placeholder.supervisor.identity_alive", side_effect=RuntimeError("denied"))
    def test_unverifiable_owner_is_not_deleted(self, alive):
        with tempfile.TemporaryDirectory() as temp:
            control = ensure_local_control_dir(Path(temp) / "control")
            path = control / "requests" / "keep.json"
            atomic_json(path, {"version": 1, "id": "keep", "owner": {}})
            active, errors = Reservations(control).active()
            self.assertFalse(active)
            self.assertTrue(errors)
            self.assertTrue(path.exists())


DAEMON_SCRIPT = r'''
import os, signal, subprocess, sys, threading
from pathlib import Path
from singularity_placeholder.monitor import GPU, Snapshot
from singularity_placeholder.supervisor import Config, Supervisor, WorkerPool, Worker
class Monitor:
    def inventory(self): return (GPU(0, "GPU-test", 0, 0),)
    def sample(self, selected): return Snapshot(self.inventory(), ())
pool = WorkerPool(0.5)
worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"], start_new_session=True)
Path(sys.argv[2]).write_text(str(worker.pid))
reader = threading.Thread(target=lambda: None)
reader.start()
pool.workers["GPU-test"] = Worker("GPU-test", worker, {worker.pid}, reader)
Supervisor(Config(Path(sys.argv[1]), Path(sys.argv[1]) / "logs", gpu_count=1, idle_seconds=999, poll_seconds=0.05), monitor=Monitor(), pool=pool, environ={}).run()
'''


class ProtocolIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.control = Path(self.temp.name) / "control"
        self.pidfile = Path(self.temp.name) / "worker.pid"
        self.repo = Path(__file__).resolve().parents[1]
        self.env = os.environ.copy()
        self.env["PYTHONPATH"] = str(self.repo) + os.pathsep + self.env.get("PYTHONPATH", "")
        self.daemon = subprocess.Popen(
            [sys.executable, "-c", DAEMON_SCRIPT, str(self.control), str(self.pidfile)],
            cwd=self.repo, env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.addCleanup(self.cleanup_processes)
        self.wait_for(lambda: (self.control / "status.json").exists())

    def cleanup_processes(self):
        if self.daemon.poll() is None:
            self.daemon.terminate()
        try:
            self.daemon.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            self.daemon.kill()
            self.daemon.communicate()
        if self.pidfile.exists():
            try:
                os.kill(int(self.pidfile.read_text()), signal.SIGTERM)
            except ProcessLookupError:
                pass
        self.temp.cleanup()

    def wait_for(self, condition, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if condition():
                return
            if self.daemon.poll() is not None:
                self.fail(f"Fake supervisor failed with exit code {self.daemon.returncode}")
            time.sleep(0.02)
        self.fail("Timed out waiting for protocol condition")

    def command(self, code):
        return [sys.executable, "-m", "singularity_placeholder", "run", "--control-dir", str(self.control), "--ack-timeout", "3", "--stop-seconds", "0.3", "--", sys.executable, "-c", code]

    def test_command_only_runs_after_worker_exits_and_preserves_exit_code(self):
        marker = Path(self.temp.name) / "marker"
        worker_pid = int(self.pidfile.read_text())
        code = f"import os; from pathlib import Path; from singularity_placeholder.supervisor import process_identity; assert process_identity({worker_pid}) is None; assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-test'; Path({str(marker)!r}).write_text('started'); raise SystemExit(7)"
        result = subprocess.run(self.command(code), cwd=self.repo, env=self.env, capture_output=True, text=True, timeout=8)
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertTrue(marker.exists())
        self.assertFalse(list((self.control / "requests").glob("*.json")))
        self.assertIsNone(self.daemon.poll(), "Stopping a worker must not terminate the supervisor")

    def test_wrapper_sigterm_forwards_and_releases_after_command_stops(self):
        marker = Path(self.temp.name) / "child.pid"
        code = f"import os,time; from pathlib import Path; Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(300)"
        wrapper = subprocess.Popen(self.command(code), cwd=self.repo, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            self.wait_for(marker.exists)
            child_pid = int(marker.read_text())
            self.assertTrue(list((self.control / "requests").glob("*.json")))
            wrapper.terminate()
            output = wrapper.communicate(timeout=5)
            self.assertEqual(wrapper.returncode, 128 + signal.SIGTERM, output)
            self.assertIsNone(process_identity(child_pid))
            self.assertFalse(list((self.control / "requests").glob("*.json")))
        finally:
            if wrapper.poll() is None:
                wrapper.kill()
                wrapper.communicate()

    def test_dead_supervisor_never_authorizes_workload(self):
        self.daemon.terminate()
        self.daemon.communicate(timeout=5)
        marker = Path(self.temp.name) / "must-not-exist"
        cmd = self.command(f"from pathlib import Path; Path({str(marker)!r}).touch()")
        cmd[cmd.index("--ack-timeout") + 1] = "0.15"
        result = subprocess.run(cmd, cwd=self.repo, env=self.env, capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(marker.exists())
        self.assertIn("not launched", result.stderr)

    def test_sigkill_wrapper_retains_reservation_for_live_cpu_starting_child(self):
        marker = Path(self.temp.name) / "orphan.pid"
        code = f"import os,time; from pathlib import Path; Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(300)"
        wrapper = subprocess.Popen(self.command(code), cwd=self.repo, env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        child_pid = None
        try:
            self.wait_for(marker.exists)
            child_pid = int(marker.read_text())
            wrapper.kill()
            wrapper.wait(timeout=3)
            time.sleep(0.2)
            self.assertTrue(list((self.control / "requests").glob("*.json")))
            current = json.loads((self.control / "status.json").read_text())
            self.assertEqual(current["state"], "reserved")
            self.assertIsNotNone(process_identity(child_pid))
        finally:
            if wrapper.poll() is None:
                wrapper.kill()
                wrapper.wait(timeout=3)
            if child_pid is not None:
                try:
                    os.killpg(child_pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    unittest.main()

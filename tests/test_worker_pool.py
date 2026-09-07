import os
import io
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from singularity_placeholder.supervisor import Worker, WorkerPool, group_alive, process_identity
from singularity_placeholder.worker import _watch_parent


def finished_reader():
    reader = threading.Thread(target=lambda: None)
    reader.start()
    reader.join()
    return reader


class WorkerOwnershipTests(unittest.TestCase):
    def test_owned_pids_are_scoped_to_gpu(self):
        pool = WorkerPool()
        process_a = mock.Mock(pid=101)
        process_a.poll.return_value = None
        process_b = mock.Mock(pid=202)
        process_b.poll.return_value = None
        pool.workers = {
            "GPU-A": Worker("GPU-A", process_a, {101}, finished_reader()),
            "GPU-B": Worker("GPU-B", process_b, {202}, finished_reader()),
        }
        self.assertEqual(pool.own_pids("GPU-A"), {101})
        self.assertEqual(pool.own_pids("GPU-B"), {202})
        self.assertEqual(pool.own_pids("GPU-unknown"), set())
        process_a.poll.return_value = 0
        self.assertEqual(pool.own_pids("GPU-A"), set())

    def test_verified_host_namespace_uses_one_host_pid(self):
        with mock.patch("singularity_placeholder.supervisor.sys.platform", "linux"), \
                mock.patch("singularity_placeholder.supervisor.os.readlink", return_value="pid:[4026531836]"):
            self.assertEqual(WorkerPool._pid_aliases(12345), {12345})

    def test_isolated_namespace_launches_and_accepts_ready_marker(self):
        process = mock.Mock(pid=12345)
        process.poll.return_value = None
        process.stdout = io.StringIO("PLACEHOLDER_READY gpu=GPU-A pid=12345\n")
        with mock.patch("singularity_placeholder.supervisor.sys.platform", "linux"), \
                mock.patch("singularity_placeholder.supervisor.os.readlink", return_value="pid:[4026539999]"), \
                mock.patch("singularity_placeholder.supervisor.subprocess.Popen", return_value=process) as spawn:
            pool = WorkerPool()
            pool.start(("GPU-A",))
        self.assertTrue(pool.workers["GPU-A"].ready.wait(1))
        self.assertEqual(pool.own_pids("GPU-A"), set())
        self.assertIn("--parent-pid", spawn.call_args.args[0])

    def isolated_worker(self, *, ready=True):
        pool = WorkerPool()
        process = mock.Mock(pid=42)
        process.poll.return_value = None
        worker = Worker("GPU-A", process, set(), finished_reader(), started_at=0)
        if ready:
            worker.ready.set()
        pool.workers["GPU-A"] = worker
        return pool, worker

    def test_ready_singleton_maps_host_pid_without_container_alias(self):
        pool, worker = self.isolated_worker()
        observed = pool.observe_processes((("GPU-A", 9000),), pool.ready_gpus(), now=1)
        self.assertEqual(observed, ({"GPU-A": set()}, set(), {}))
        self.assertEqual(pool.own_pids("GPU-A"), {9000})
        self.assertNotIn(worker.process.pid, pool.own_pids("GPU-A"))
        self.assertEqual(pool.observe_processes((("GPU-A", 9000),), pool.ready_gpus(), now=2)[0], {"GPU-A": set()})

    def test_snapshot_started_before_ready_cannot_assign_ownership(self):
        pool, worker = self.isolated_worker(ready=False)
        ready_before_sample = pool.ready_gpus()
        worker.ready.set()  # The marker arrived while nvidia-smi was running.
        foreign, starting, errors = pool.observe_processes((("GPU-A", 9000),), ready_before_sample, now=1)
        self.assertEqual(pool.own_pids("GPU-A"), set())
        self.assertEqual((foreign, starting, errors), ({"GPU-A": set()}, {"GPU-A"}, {}))
        pool.observe_processes((("GPU-A", 9000),), pool.ready_gpus(), now=2)
        self.assertEqual(pool.own_pids("GPU-A"), {9000})

    def test_extra_process_is_foreign_before_and_after_association(self):
        pool, _ = self.isolated_worker()
        foreign, _, _ = pool.observe_processes((("GPU-A", 9000), ("GPU-A", 9001)), pool.ready_gpus(), now=1)
        self.assertEqual(foreign["GPU-A"], {9000, 9001})
        self.assertFalse(pool.own_pids("GPU-A"))
        pool.observe_processes((("GPU-A", 9000),), pool.ready_gpus(), now=2)
        foreign, _, _ = pool.observe_processes((("GPU-A", 9000), ("GPU-A", 9001)), pool.ready_gpus(), now=3)
        self.assertEqual(foreign["GPU-A"], {9001})

    def test_startup_without_ready_or_visible_pid_has_bounded_allowance(self):
        for ready in (False, True):
            with self.subTest(ready=ready):
                pool, _ = self.isolated_worker(ready=ready)
                self.assertEqual(pool.observe_processes((), pool.ready_gpus(), now=119)[1], {"GPU-A"})
                _, starting, errors = pool.observe_processes((), pool.ready_gpus(), now=120)
                self.assertFalse(starting)
                self.assertIn("GPU-A", errors)

    def test_exited_worker_never_claims_nvidia_pid(self):
        pool, worker = self.isolated_worker()
        ready = pool.ready_gpus()
        worker.process.poll.return_value = 1
        foreign, _, _ = pool.observe_processes((("GPU-A", 9000),), ready, now=1)
        self.assertEqual(foreign["GPU-A"], {9000})
        self.assertFalse(pool.own_pids("GPU-A"))

    def test_unreadable_namespace_uses_readiness_instead_of_rejecting(self):
        with mock.patch("singularity_placeholder.supervisor.sys.platform", "linux"), \
                mock.patch("singularity_placeholder.supervisor.os.readlink", side_effect=PermissionError):
            self.assertEqual(WorkerPool._pid_aliases(42), set())

    def test_watchdog_only_exits_its_own_process_after_parent_changes(self):
        with mock.patch("singularity_placeholder.worker.os.getppid", side_effect=[42, 42, 1]), \
                mock.patch("singularity_placeholder.worker.time.sleep") as sleep, \
                mock.patch("singularity_placeholder.worker.os._exit", side_effect=SystemExit) as leave:
            with self.assertRaises(SystemExit):
                _watch_parent(42)
        self.assertEqual(sleep.call_count, 2)
        leave.assert_called_once_with(0)

    def test_unstopped_group_remains_tracked_and_ack_is_withheld(self):
        pool = WorkerPool(0.1)
        process = mock.Mock(pid=101)
        process.poll.return_value = 0
        pool.workers["GPU-A"] = Worker("GPU-A", process, {101}, finished_reader())
        with mock.patch("singularity_placeholder.supervisor.group_alive", return_value=True), \
                mock.patch("singularity_placeholder.supervisor.signal_group") as send, \
                mock.patch("singularity_placeholder.supervisor.time.monotonic", side_effect=iter(range(100))), \
                mock.patch("singularity_placeholder.supervisor.time.sleep"):
            self.assertFalse(pool.stop(("GPU-A",)))
        self.assertIn("GPU-A", pool.workers)
        send.assert_any_call(101, signal.SIGTERM)
        send.assert_any_call(101, signal.SIGKILL)


class GroupLivenessTests(unittest.TestCase):
    def stat(self, state):
        # Only state, parent PID, and process group are used by group_alive.
        return f"12 (example) {state} 1 55 0 0 0"

    def test_linux_zombie_only_group_is_dead(self):
        with mock.patch("singularity_placeholder.supervisor.sys.platform", "linux"), \
                mock.patch("singularity_placeholder.supervisor.os.killpg"), \
                mock.patch.object(Path, "iterdir", return_value=iter([Path("/proc/12")])), \
                mock.patch.object(Path, "read_text", return_value=self.stat("Z")):
            self.assertFalse(group_alive(55))

    def test_linux_live_member_is_alive(self):
        with mock.patch("singularity_placeholder.supervisor.sys.platform", "linux"), \
                mock.patch("singularity_placeholder.supervisor.os.killpg"), \
                mock.patch.object(Path, "iterdir", return_value=iter([Path("/proc/12")])), \
                mock.patch.object(Path, "read_text", return_value=self.stat("S")):
            self.assertTrue(group_alive(55))

    def test_unreadable_procfs_does_not_release_group(self):
        with mock.patch("singularity_placeholder.supervisor.sys.platform", "linux"), \
                mock.patch("singularity_placeholder.supervisor.os.killpg"), \
                mock.patch.object(Path, "iterdir", return_value=iter([Path("/proc/12")])), \
                mock.patch.object(Path, "read_text", side_effect=PermissionError):
            self.assertTrue(group_alive(55))

    def test_invalid_group_rejected_instead_of_signalling_current_group(self):
        for pgid in (0, -1):
            with self.assertRaises(ValueError):
                group_alive(pgid)


class GroupCleanupIntegrationTests(unittest.TestCase):
    def test_worker_watchdog_exits_after_supervisor_sigkill(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            program = r'''
import os, subprocess, sys, time
from pathlib import Path
child_code = """
import sys, threading, time
from pathlib import Path
from singularity_placeholder.worker import _watch_parent
threading.Thread(target=_watch_parent, args=(int(sys.argv[1]),), daemon=True).start()
Path(sys.argv[2]).touch()
time.sleep(60)
"""
child = subprocess.Popen([sys.executable, "-c", child_code, str(os.getpid()), str(Path(sys.argv[1]) / "ready")], start_new_session=True)
Path(sys.argv[1], "child.pid").write_text(str(child.pid))
time.sleep(60)
'''
            parent = subprocess.Popen([sys.executable, "-c", program, temporary], start_new_session=True)
            child_pid = None
            try:
                deadline = time.monotonic() + 5
                while not (directory / "ready").exists() and time.monotonic() < deadline:
                    self.assertIsNone(parent.poll())
                    time.sleep(0.02)
                self.assertTrue((directory / "ready").exists())
                child_pid = int((directory / "child.pid").read_text())
                parent.kill()
                parent.wait(timeout=3)
                deadline = time.monotonic() + 5
                while process_identity(child_pid) is not None and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertIsNone(process_identity(child_pid), "Orphan worker must release itself after supervisor SIGKILL")
            finally:
                if parent.poll() is None:
                    parent.kill()
                parent.wait(timeout=3)
                if child_pid is None and (directory / "child.pid").exists():
                    child_pid = int((directory / "child.pid").read_text())
                if child_pid is not None and process_identity(child_pid) is not None:
                    os.killpg(child_pid, signal.SIGKILL)

    def test_stops_descendant_after_leader_exits_and_preserves_other_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "descendant.pid"
            program = ("import subprocess,sys; from pathlib import Path; "
                       "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                       "Path(sys.argv[1]).write_text(str(child.pid))")
            leader = subprocess.Popen([sys.executable, "-c", program, str(marker)], start_new_session=True)
            unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
            try:
                leader.wait(timeout=5)
                self.assertTrue(group_alive(leader.pid))
                pool = WorkerPool(0.2)
                pool.workers["GPU-A"] = Worker("GPU-A", leader, {leader.pid}, finished_reader())
                pool.workers["GPU-B"] = Worker("GPU-B", unrelated, {unrelated.pid}, finished_reader())
                self.assertTrue(pool.stop(("GPU-A",)))
                self.assertFalse(group_alive(leader.pid))
                self.assertNotIn("GPU-A", pool.workers)
                self.assertIn("GPU-B", pool.workers)
                self.assertIsNone(unrelated.poll())
                self.assertTrue(pool.stop_all())
                self.assertIsNotNone(unrelated.poll())
            finally:
                for process in (leader, unrelated):
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()

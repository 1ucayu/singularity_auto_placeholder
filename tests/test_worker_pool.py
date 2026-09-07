import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from singularity_placeholder.supervisor import Worker, WorkerPool, group_alive


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
                mock.patch("singularity_placeholder.supervisor.os.readlink", return_value="pid:[4026531836]"), \
                mock.patch.object(Path, "read_text", return_value="NSpid:\t12345\t8\n"):
            self.assertEqual(WorkerPool._pid_aliases(12345), {12345})

    def test_isolated_namespace_does_not_guess_or_launch_worker(self):
        with mock.patch("singularity_placeholder.supervisor.sys.platform", "linux"), \
                mock.patch("singularity_placeholder.supervisor.os.readlink", return_value="pid:[4026539999]"), \
                mock.patch("singularity_placeholder.supervisor.subprocess.Popen") as spawn:
            with self.assertRaisesRegex(RuntimeError, "isolated PID namespace"):
                WorkerPool().start(("GPU-A",))
            spawn.assert_not_called()

    def test_mismatched_host_pid_fails_closed(self):
        with mock.patch.object(WorkerPool, "_verify_pid_namespace"), \
                mock.patch("singularity_placeholder.supervisor.sys.platform", "linux"), \
                mock.patch.object(Path, "read_text", return_value="NSpid:\t9999\t8\n"):
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                WorkerPool._pid_aliases(12345)

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

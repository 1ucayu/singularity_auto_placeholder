from pathlib import Path
import selectors
import socket
import subprocess
import sys
import tempfile
import unittest

from singularity_placeholder import remote_socket


class SocketLeaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir="/tmp")
        self.root = Path(self.temp.name)
        self.path = self.root / "private/api.sock"
        self.path.parent.mkdir(mode=0o700)
        self.children = []
        self.listeners = []
        self.addCleanup(self.cleanup)

    def cleanup(self):
        for child in self.children:
            if child.poll() is None:
                child.stdin.close()
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            for stream in (child.stdin, child.stdout, child.stderr):
                if not stream.closed:
                    stream.close()
        for listener in self.listeners:
            listener.close()
        self.temp.cleanup()

    def listener(self, path=None):
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(path or self.path))
        listener.listen(4)
        self.listeners.append(listener)
        return listener

    def start(self):
        child = subprocess.Popen(
            [sys.executable, "-u", remote_socket.__file__, str(self.path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
        self.children.append(child)
        return child

    def marker(self, child, expected):
        with selectors.DefaultSelector() as events:
            events.register(child.stdout, selectors.EVENT_READ)
            self.assertTrue(events.select(timeout=3), "socket helper response timed out")
        value = child.stdout.readline().strip()
        if not value:
            self.fail(f"helper exited: {child.stderr.read()}")
        self.assertEqual(value, expected)

    def bind(self, child):
        listener = self.listener()
        child.stdin.write("BOUND\n")
        child.stdin.flush()
        self.marker(child, remote_socket.BOUND)
        return listener

    def test_stale_socket_recovered_and_own_socket_removed_on_clean_exit(self):
        self.listener().close()
        self.assertTrue(self.path.exists())
        child = self.start()
        self.marker(child, remote_socket.READY)
        self.assertFalse(self.path.exists())
        self.bind(child)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        child.stdin.close()
        self.assertEqual(child.wait(timeout=3), 0)
        self.assertFalse(self.path.exists())

    def test_competing_lease_cannot_replace_active_session(self):
        first = self.start()
        self.marker(first, remote_socket.READY)
        self.bind(first)
        identity = self.path.stat().st_ino
        second = self.start()
        self.assertEqual(second.wait(timeout=3), 1)
        self.assertIsNone(first.poll())
        self.assertEqual(self.path.stat().st_ino, identity)

    def test_live_socket_without_lease_is_never_removed(self):
        self.listener()
        identity = self.path.stat().st_ino
        child = self.start()
        self.assertEqual(child.wait(timeout=3), 1)
        self.assertIn("live or cannot be proven stale", child.stderr.read())
        self.assertEqual(self.path.stat().st_ino, identity)

    def test_socket_left_after_abrupt_exit_can_be_recovered(self):
        first = self.start()
        self.marker(first, remote_socket.READY)
        listener = self.bind(first)
        first.kill()
        first.wait(timeout=3)
        listener.close()
        self.assertTrue(self.path.exists())
        second = self.start()
        self.marker(second, remote_socket.READY)
        self.assertFalse(self.path.exists())

    def test_cleanup_preserves_replacement_inode(self):
        child = self.start()
        self.marker(child, remote_socket.READY)
        self.bind(child)
        self.path.unlink()
        self.listener()
        replacement = self.path.stat().st_ino
        child.stdin.close()
        self.assertEqual(child.wait(timeout=3), 0)
        self.assertEqual(self.path.stat().st_ino, replacement)

    def test_regular_file_and_unsafe_directory_are_preserved(self):
        self.path.write_text("do not remove")
        child = self.start()
        self.assertEqual(child.wait(timeout=3), 1)
        self.assertEqual(self.path.read_text(), "do not remove")
        self.path.unlink()
        self.path.parent.chmod(0o755)
        child = self.start()
        self.assertEqual(child.wait(timeout=3), 1)
        self.assertEqual(self.path.parent.stat().st_mode & 0o777, 0o755)


if __name__ == "__main__":
    unittest.main()

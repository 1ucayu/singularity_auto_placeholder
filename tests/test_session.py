import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from singularity_placeholder import session
from singularity_placeholder.session import prepare_paths, prepare_storage, write_environment


class SessionPathsTest(unittest.TestCase):
    def test_blob_and_local_are_separate_and_shell_paths_are_quoted(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            mount = base / "blob with ' quote"
            mount.mkdir()
            local = base / "local with ' quote"
            args = argparse.Namespace(blob_root=mount, blob_prefix="lucayu/sglang", local_root=local)
            paths = prepare_paths(args)
            prepare_storage(paths, 8)
            values = write_environment(paths, base / "repository")
            self.assertTrue((mount / "lucayu/sglang/models").is_dir())
            self.assertNotIn("TOKEN", (local / "env.sh").read_text())
            output = subprocess.check_output(["bash", "-c", 'source "$1"; printf "%s" "$BLOB_ROOT"', "_", str(local / "env.sh")], text=True)
            self.assertEqual(output, values["BLOB_ROOT"])
            subprocess.run(["bash", "-n", str(local / "bin/placeholder")], check=True)
            self.assertEqual(os.stat(local / "env.sh").st_mode & 0o777, 0o600)

    def test_missing_blob_still_produces_local_launcher(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            args = argparse.Namespace(blob_root=base / "missing", blob_prefix="lucayu/sglang", local_root=base / "local")
            paths = prepare_paths(args)
            write_environment(paths, base / "repo")
            self.assertTrue((base / "local/bin/placeholder").is_file())
            with self.assertRaisesRegex(OSError, "Blob mount is not available"):
                prepare_storage(paths, 8)
            self.assertFalse((base / "missing").exists())

    def test_rejects_local_inside_blob_and_prefix_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            mount = Path(directory) / "blob"
            mount.mkdir()
            for prefix, local in [("../escape", Path(directory) / "local"), ("lucayu/sglang", mount / "local")]:
                with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                    prepare_paths(argparse.Namespace(blob_root=mount, blob_prefix=prefix, local_root=local))


class ServiceLifecycleTest(unittest.TestCase):
    def wait_for(self, condition):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if condition():
                return
            time.sleep(0.02)
        self.fail("Service lifecycle timed out")

    def test_crashed_gpu_service_restarts_without_restarting_tunnel(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "started"
            code = (f"from pathlib import Path; import time; p=Path({str(marker)!r}); "
                    "first=not p.exists(); p.write_text('first' if first else 'restarted'); "
                    "time.sleep(0.1 if first else 60); raise SystemExit(7)")
            tunnel = session.Service("tunnel", [sys.executable, "-c", "import time; time.sleep(60)"], os.environ.copy())
            gpu = session.Service("GPU", [sys.executable, "-c", code], os.environ.copy())
            stopping = threading.Event()
            runner = threading.Thread(target=session.run_services, args=([tunnel, gpu], stopping, 0.1))
            with patch.object(session, "log"):
                runner.start()
                try:
                    self.wait_for(lambda: marker.exists() and tunnel.process is not None)
                    tunnel_child = tunnel.process
                    self.wait_for(lambda: marker.read_text() == "restarted")
                    self.assertIs(tunnel.process, tunnel_child)
                    self.assertIsNone(tunnel_child.poll())
                    self.assertTrue(runner.is_alive())
                finally:
                    stopping.set()
                    runner.join(timeout=5)
            self.assertFalse(runner.is_alive())
            self.assertIsNotNone(tunnel.process.poll())
            self.assertIsNotNone(gpu.process.poll())

    def test_spawn_error_does_not_block_sibling(self):
        broken = session.Service("missing GPU command", ["/nonexistent-placeholder-python"], os.environ.copy())
        live = session.Service("tunnel", [sys.executable, "-c", "import time; time.sleep(60)"], os.environ.copy())
        stopping = threading.Event()
        runner = threading.Thread(target=session.run_services, args=([broken, live], stopping, 0.1))
        with patch.object(session, "log"):
            runner.start()
            try:
                self.wait_for(lambda: live.process is not None)
                self.assertIsNone(live.process.poll())
                self.assertIsNone(broken.process)
            finally:
                stopping.set()
                runner.join(timeout=5)
        self.assertFalse(runner.is_alive())
        self.assertIsNotNone(live.process.poll())

    def test_tunnel_and_gpu_start_even_when_blob_setup_hangs(self):
        release_blob = threading.Event()
        blob_entered = threading.Event()
        children = []
        commands = []
        real_popen, real_run = subprocess.Popen, session.run_services

        def blocked_blob(*args):
            blob_entered.set()
            release_blob.wait(5)

        def spawn(command, **kwargs):
            commands.append(command)
            if "supervise" in command:
                self.assertNotIn("VSCODE_CLI_ACCESS_TOKEN", kwargs["env"])
            child = real_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
            children.append(child)
            return child

        def run(services, stopping, retry_seconds):
            timer = threading.Timer(0.6, stopping.set)
            timer.start()
            try:
                real_run(services, stopping, retry_seconds)
            finally:
                timer.cancel()

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            try:
                with patch.object(session, "prepare_storage", side_effect=blocked_blob), \
                        patch.object(session.subprocess, "Popen", side_effect=spawn), \
                        patch.object(session, "run_services", side_effect=run), \
                        patch.object(session, "log"), \
                        patch.dict(os.environ, {"VSCODE_CLI_ACCESS_TOKEN": "test-only"}):
                    result = session.main(["--blob-root", str(base / "missing"), "--local-root", str(base / "local"), "--tunnel"])
                self.assertEqual(result, 0)
                self.assertTrue(blob_entered.is_set())
                self.assertEqual(len(children), 2)
                self.assertIn("singularity_placeholder.tunnel", commands[0])
                self.assertIn("supervise", commands[1])
                self.assertTrue(all(child.poll() is not None for child in children))
                self.assertTrue((base / "local/env.sh").exists())
            finally:
                release_blob.set()
                for child in children:
                    if child.poll() is None:
                        os.killpg(child.pid, session.signal.SIGKILL)
                    child.wait(timeout=5)

    def test_blob_failure_retries_without_exiting_service(self):
        stopping = threading.Event()
        attempts = []

        def prepare(*args):
            attempts.append(1)
            if len(attempts) == 1:
                raise OSError("mount temporarily offline")

        with patch.object(session, "prepare_storage", side_effect=prepare), patch.object(session, "log"):
            session.storage_loop({"persistent": "/blob"}, 8, stopping, 0.01)
        self.assertEqual(len(attempts), 2)


if __name__ == "__main__":
    unittest.main()

import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from singularity_placeholder import session
from singularity_placeholder.session import prepare_paths, write_environment, supervisor_failure_action


class SessionPathsTest(unittest.TestCase):
    def test_blob_and_local_are_separate_and_shell_paths_are_quoted(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            mount = base / "blob with ' quote"
            mount.mkdir()
            local = base / "local with ' quote"
            args = argparse.Namespace(blob_root=mount, blob_prefix="lucayu/sglang", local_root=local)
            paths = prepare_paths(args)
            values = write_environment(paths, base / "repository")
            self.assertTrue((mount / "lucayu/sglang/models").is_dir())
            self.assertNotIn("TOKEN", (local / "env.sh").read_text())
            output = subprocess.check_output(["bash", "-c", 'source "$1"; printf "%s" "$BLOB_ROOT"', "_", str(local / "env.sh")], text=True)
            self.assertEqual(output, values["BLOB_ROOT"])
            subprocess.run(["bash", "-n", str(local / "bin/placeholder")], check=True)
            self.assertEqual(os.stat(local / "env.sh").st_mode & 0o777, 0o600)

    def test_rejects_local_inside_blob_and_prefix_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            mount = Path(directory) / "blob"
            mount.mkdir()
            for prefix, local in [("../escape", Path(directory) / "local"), ("lucayu/sglang", mount / "local")]:
                with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                    prepare_paths(argparse.Namespace(blob_root=mount, blob_prefix=prefix, local_root=local))


class SupervisorFailureTest(unittest.TestCase):
    def test_failed_supervisor_keeps_tunnel_for_bounded_grace(self):
        should_exit, deadline = supervisor_failure_action(1, tunnel_enabled=True, grace_seconds=600, now=10, deadline=None)
        self.assertFalse(should_exit)
        self.assertEqual(deadline, 610)
        self.assertEqual(supervisor_failure_action(1, tunnel_enabled=True, grace_seconds=600, now=20, deadline=deadline), (False, 610))
        self.assertEqual(supervisor_failure_action(1, tunnel_enabled=True, grace_seconds=600, now=610, deadline=deadline), (True, 610))

    def test_clean_exit_and_no_tunnel_do_not_keep_failed_session(self):
        for code, enabled, grace in [(0, True, 600), (1, False, 600), (1, True, 0)]:
            with self.subTest(code=code, enabled=enabled, grace=grace):
                self.assertTrue(supervisor_failure_action(code, tunnel_enabled=enabled, grace_seconds=grace, now=10, deadline=None)[0])

    def test_entrypoint_keeps_real_tunnel_child_then_cleans_up_and_returns_failure(self):
        real_popen = subprocess.Popen
        children = []

        def spawn(command, **kwargs):
            program = "raise SystemExit(7)" if "supervise" in command else "import time; time.sleep(60)"
            child = real_popen([session.sys.executable, "-c", program], **kwargs)
            children.append(child)
            return child

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "blob").mkdir()
            began = time.monotonic()
            try:
                with patch.object(session, "preflight"), patch.object(session.subprocess, "Popen", side_effect=spawn), patch.object(session, "log") as logger:
                    result = session.main([
                        "--blob-root", str(base / "blob"), "--local-root", str(base / "local"),
                        "--tunnel", "--debug-grace-seconds", "0.3",
                    ])
                self.assertEqual(result, 7)
                self.assertGreaterEqual(time.monotonic() - began, 0.3)
                self.assertEqual(len(children), 2)
                self.assertTrue(all(child.poll() is not None for child in children))
                self.assertTrue(any("Keeping the tunnel" in call.args[0] for call in logger.call_args_list))
            finally:
                for child in children:
                    if child.poll() is None:
                        os.killpg(child.pid, session.signal.SIGKILL)
                    child.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()

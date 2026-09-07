"""No network, GPU, GitHub login, or real tunnel required."""

import io
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from unittest import mock

from singularity_placeholder import parent_exec, tunnel


class TunnelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.logger = logging.getLogger("test-tunnel")
        self.logger.handlers = [logging.NullHandler()]
        self.logger.propagate = False

    def test_token_only_in_login_environment_never_arguments(self):
        environment = {"PATH": "/bin", "VSCODE_CLI_ACCESS_TOKEN": "sensitive-value",
                       "VSCODE_CLI_REFRESH_TOKEN": "refresh-value"}
        login = tunnel.child_environment(environment, self.root, login=True)
        self.assertEqual(login["VSCODE_CLI_ACCESS_TOKEN"], "sensitive-value")
        serving = tunnel.child_environment(environment, self.root)
        self.assertNotIn("VSCODE_CLI_ACCESS_TOKEN", serving)
        self.assertNotIn("VSCODE_CLI_REFRESH_TOKEN", serving)
        self.assertEqual(serving["VSCODE_CLI_DATA_DIR"], str(self.root / "cli-data"))
        self.assertEqual(serving["VSCODE_CLI_USE_FILE_KEYCHAIN"], "1")
        self.assertEqual(environment["VSCODE_CLI_ACCESS_TOKEN"], "sensitive-value")

    def test_redacts_tokens_in_console_and_file(self):
        output = io.StringIO()
        with mock.patch("sys.stdout", output):
            logger = tunnel.make_logger(self.root, ("opaque-secret",))
            logger.error("Failure: %s %s", "opaque-secret", "ghp_123456789012345678901234567890123456")
            for handler in logger.handlers:
                handler.flush()
            contents = (self.root / "tunnel.log").read_text()
        for text in (contents, output.getvalue()):
            self.assertNotIn("opaque-secret", text)
            self.assertNotIn("ghp_", text)
            self.assertIn("[REDACTED]", text)
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    def test_rejects_shared_runtime_but_allows_nested_local_mount(self):
        mounts = "1 0 0:1 / / rw - overlay overlay rw\n2 1 0:2 / /blob rw - fuse.blobfuse2 blob rw\n"
        with self.assertRaisesRegex(ValueError, "local"):
            tunnel.local_filesystem(Path("/blob/experiments/tunnel"), mounts)
        tunnel.local_filesystem(Path("/tmp/tunnel"), mounts)
        tunnel.local_filesystem(Path("/blob/local/tunnel"), mounts + "3 2 0:3 / /blob/local rw - tmpfs tmpfs rw\n")

    def test_rejects_downgrade_redirect(self):
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            tunnel.HTTPSRedirectHandler().redirect_request(None, None, 302, "", {}, "http://example.com/cli")

    def test_download_architectures(self):
        self.assertIn("cli-alpine-x64", tunnel.cli_download_url("x86_64"))
        self.assertIn("cli-alpine-arm64", tunnel.cli_download_url("aarch64"))
        with self.assertRaises(ValueError):
            tunnel.cli_download_url("riscv64")

    def test_archive_installs_only_code_without_extracting_paths(self):
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as bundle:
            for filename, content in (("code", b"#!/bin/sh\nexit 0\n"), ("../../outside", b"bad")):
                info = tarfile.TarInfo(filename)
                info.size = len(content)
                bundle.addfile(info, io.BytesIO(content))
        opener = mock.Mock()
        opener.open.return_value = io.BytesIO(archive.getvalue())
        with mock.patch.object(tunnel.platform, "system", return_value="Linux"), \
                mock.patch.object(tunnel.urllib.request, "build_opener", return_value=opener):
            binary = tunnel.install_cli(self.root)
        self.assertTrue(os.access(binary, os.X_OK))
        self.assertEqual(binary.read_bytes(), b"#!/bin/sh\nexit 0\n")
        self.assertFalse((self.root.parent / "outside").exists())

    def test_signal_stops_child_process_group(self):
        stop = threading.Event()
        timer = threading.Timer(0.3, stop.set)
        timer.start()
        self.addCleanup(timer.cancel)
        pidfile = self.root / "child.pid"
        program = ("import os,time; "
                   f"open({str(pidfile)!r}, 'w').write(str(os.getpid())); "
                   "time.sleep(60)")
        began = time.monotonic()
        result = tunnel.run_child([sys.executable, "-c", program], dict(os.environ), stop, self.logger)
        self.assertEqual(result, -signal.SIGTERM)
        self.assertLess(time.monotonic() - began, 5)
        self.assertTrue(pidfile.exists())
        with self.assertRaises(ProcessLookupError):
            os.kill(int(pidfile.read_text()), 0)

    def test_login_has_timeout(self):
        result = tunnel.run_child([sys.executable, "-c", "import time; time.sleep(60)"],
                                  dict(os.environ), threading.Event(), self.logger, timeout=0.2)
        self.assertEqual(result, 124)

    def test_child_exec_preserves_command_and_environment(self):
        output = self.root / "child-env"
        program = ("import os,sys; from pathlib import Path; "
                   "Path(sys.argv[1]).write_text(os.environ['TUNNEL_TEST_VALUE'] + ':' + sys.argv[2])")
        environment = dict(os.environ, TUNNEL_TEST_VALUE="unchanged value")
        result = tunnel.run_child([sys.executable, "-c", program, str(output), "one argument with spaces"],
                                  environment, threading.Event(), self.logger)
        self.assertEqual(result, 0)
        self.assertEqual(output.read_text(), "unchanged value:one argument with spaces")

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux parent-death signal integration")
    def test_crashed_tunnel_helper_terminates_its_detached_cli(self):
        pidfile = self.root / "orphan-cli.pid"
        child_code = ("import os,time; from pathlib import Path; "
                      f"Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(60)")
        helper_code = (
            "import logging,os,sys,threading; from singularity_placeholder import tunnel; "
            f"tunnel.run_child([sys.executable, '-c', {child_code!r}], dict(os.environ), "
            "threading.Event(), logging.getLogger('parent-death-integration'))"
        )
        helper = subprocess.Popen([sys.executable, "-c", helper_code], start_new_session=True,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cli_pid = None

        def running(pid):
            try:
                # Orphans may briefly remain zombies under a container PID 1.
                # A zombie has already exited and cannot hold a tunnel socket.
                status = Path(f"/proc/{pid}/stat").read_text()
                return status.rsplit(")", 1)[1].split()[0] != "Z"
            except FileNotFoundError:
                return False

        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    cli_pid = int(pidfile.read_text())
                    break
                except (FileNotFoundError, ValueError):
                    time.sleep(0.02)
            self.assertIsNotNone(cli_pid, "CLI was not launched")
            self.assertTrue(running(cli_pid))
            helper.kill()  # No helper finally block can run after SIGKILL.
            helper.wait(timeout=5)
            deadline = time.monotonic() + 5
            while running(cli_pid) and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertFalse(running(cli_pid), "CLI survived the tunnel helper's death")
        finally:
            if helper.poll() is None:
                helper.kill()
            helper.wait(timeout=5)
            if cli_pid is not None:
                try:
                    os.killpg(cli_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_failure_budget_and_backoff_are_bounded(self):
        stop = mock.Mock()
        stop.is_set.return_value = False
        with mock.patch.object(tunnel, "install_cli", side_effect=ValueError("offline")) as install:
            result = tunnel.supervise(self.root, "test", stop, self.logger, max_failures=3)
        self.assertEqual(result, 1)
        self.assertEqual(install.call_count, 3)
        self.assertEqual(stop.wait.call_args_list, [mock.call(2), mock.call(4)])

    def test_injected_secret_authenticates_then_runs_tunnel(self):
        stop = threading.Event()
        calls = []

        def run(command, environment, stop_event, logger, timeout=None):
            calls.append((command, environment))
            if "user" not in command:
                stop_event.set()
            return 0

        with mock.patch.dict(os.environ, {"VSCODE_CLI_ACCESS_TOKEN": "secret-for-test"}), \
                mock.patch.object(tunnel, "install_cli", return_value=Path("/local/code")), \
                mock.patch.object(tunnel, "run_child", side_effect=run):
            self.assertEqual(tunnel.supervise(self.root, "aml-lucayu", stop, self.logger), 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0][-5:], ["tunnel", "user", "login", "--provider", "github"])
        self.assertEqual(calls[0][1]["VSCODE_CLI_ACCESS_TOKEN"], "secret-for-test")
        self.assertNotIn("secret-for-test", " ".join(calls[0][0]))
        self.assertNotIn("VSCODE_CLI_ACCESS_TOKEN", calls[1][1])
        self.assertIn("--accept-server-license-terms", calls[1][0])

    def test_cached_credentials_skip_device_login(self):
        data = self.root / "cli-data"
        data.mkdir()
        (data / "token.json").write_text("{}")
        stop = threading.Event()

        def run(command, environment, stop_event, logger, timeout=None):
            self.assertNotIn("user", command)
            stop_event.set()
            return 0

        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(tunnel, "install_cli", return_value=Path("/local/code")), \
                mock.patch.object(tunnel, "run_child", side_effect=run):
            self.assertEqual(tunnel.supervise(self.root, "aml-lucayu", stop, self.logger), 0)


class ParentExecTests(unittest.TestCase):
    def test_parent_death_spawn_race_does_not_launch_cli(self):
        with mock.patch.object(parent_exec, "arm_parent_death"), \
                mock.patch.object(parent_exec.os, "getppid", return_value=1), \
                mock.patch.object(parent_exec.os, "execvp") as execute:
            self.assertEqual(parent_exec.main(["--parent-pid", "12345", "--", "/code", "tunnel"]), 0)
        execute.assert_not_called()

    def test_live_parent_executes_same_cli_arguments(self):
        with mock.patch.object(parent_exec, "arm_parent_death") as arm, \
                mock.patch.object(parent_exec.os, "getppid", return_value=12345), \
                mock.patch.object(parent_exec.os, "execvp") as execute:
            self.assertEqual(parent_exec.main(["--parent-pid", "12345", "--", "/code", "tunnel", "--name", "test"]), 0)
        arm.assert_called_once_with()
        execute.assert_called_once_with("/code", ["/code", "tunnel", "--name", "test"])

    def test_linux_arms_parent_death_signal(self):
        library = mock.Mock()
        library.prctl.return_value = 0
        with mock.patch.object(parent_exec.sys, "platform", "linux"), \
                mock.patch.object(parent_exec.ctypes, "CDLL", return_value=library):
            parent_exec.arm_parent_death()
        library.prctl.assert_called_once_with(1, signal.SIGTERM, 0, 0, 0)

    def test_unavailable_parent_death_setup_does_not_gate_exec(self):
        output = io.StringIO()
        with mock.patch.object(parent_exec.sys, "platform", "linux"), \
                mock.patch.object(parent_exec.ctypes, "CDLL", side_effect=OSError("unsupported")), \
                mock.patch.object(parent_exec.os, "getppid", return_value=12345), \
                mock.patch.object(parent_exec.os, "execvp") as execute, \
                mock.patch.object(parent_exec.sys, "stderr", output):
            self.assertEqual(parent_exec.main(["--parent-pid", "12345", "--", "/code"]), 0)
        execute.assert_called_once_with("/code", ["/code"])
        self.assertIn("continuing normally", output.getvalue())

    def test_non_linux_skips_optional_kernel_setup(self):
        with mock.patch.object(parent_exec.sys, "platform", "darwin"), \
                mock.patch.object(parent_exec.ctypes, "CDLL") as library:
            parent_exec.arm_parent_death()
        library.assert_not_called()


if __name__ == "__main__":
    unittest.main()

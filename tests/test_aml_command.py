"""Exercise generated AML commands with bounded local command shims only."""

import json
import os
from pathlib import Path
import shutil
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from singularity_placeholder import profile


REPO = Path(__file__).resolve().parents[1]


SHIM = r'''
import json, os, signal, sys, time
from pathlib import Path
kind = Path(sys.argv[0]).name
mode = os.environ["AML_COMMAND_TEST_MODE"]
events = Path(os.environ["AML_COMMAND_TEST_EVENTS"])
def record(event, **details):
    entry = {"event": event, "kind": kind, "pid": os.getpid(), "pgid": os.getpgrp(), **details}
    fd = os.open(events, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try: os.write(fd, (json.dumps(entry) + "\n").encode())
    finally: os.close(fd)
def stop(signum, frame):
    record("terminated", signal=signum)
    raise SystemExit(128 + signum)
for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(signum, stop)
def bounded_block():
    # No test shim can accidentally leave a real 99-day sleep behind.
    time.sleep(8)
    record("shim_deadline")
if kind == "bash":
    if "--noprofile" in sys.argv[1:]:
        record("clean_bash", args=sys.argv[1:3], bash_env=os.environ.get("BASH_ENV"))
        os.execv(os.environ["AML_COMMAND_REAL_BASH"], ["bash", *sys.argv[1:]])
    args = sys.argv[1:]
    config_path = Path(args[args.index("--config") + 1])
    config_text = config_path.read_text()
    record("session_started", args=args, config=json.loads(config_text), config_text=config_text)
    if mode == "session_failure":
        record("session_failed", code=42)
        raise SystemExit(42)
    bounded_block()
elif kind == "git":
    args = sys.argv[1:]
    if args[0] == "clone":
        record("clone_started", args=args)
        if mode == "clone_blocked":
            bounded_block()
            raise SystemExit(23)
        if mode == "clone_failure":
            record("clone_failed", code=23)
            raise SystemExit(23)
        target = Path(args[-1])
        (target / "scripts").mkdir(parents=True)
        (target / "scripts" / "aml_start.sh").write_text("# test fixture; the bash shim handles this\n")
        record("clone_succeeded")
    elif len(args) >= 4 and args[0] == "-C" and args[2] in ("fetch", "checkout"):
        record("revision_" + args[2], args=args, ref=args[-1])
    else:
        raise SystemExit("unexpected git command: " + repr(args))
elif kind == "sleep":
    duration = sys.argv[1]
    hold = duration == os.environ["AML_COMMAND_TEST_HOLD_SECONDS"]
    record("hold_started" if hold else "retry_sleep", duration=duration)
    if hold: bounded_block()
    else: time.sleep(0.04)
elif kind == "mktemp":
    import tempfile
    result = tempfile.mkdtemp(prefix="clone-", dir=os.environ["AML_COMMAND_TEST_TMP"])
    record("temp_created", path=result)
    print(result)
else:
    raise SystemExit("unexpected command shim")
'''


class AMLCommandLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.events = self.root / "events.jsonl"
        self.real_bash = shutil.which("bash")
        self.assertIsNotNone(self.real_bash)
        for name in ("bash", "git", "sleep", "mktemp"):
            script = self.bin / name
            script.write_text(f"#!{sys.executable}\n" + SHIM)
            script.chmod(0o755)
        self.process = None
        self.output = (self.root / "shell.log").open("w+")
        self.addCleanup(self.cleanup)

    def cleanup(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
                # Shims have an eight-second independent deadline, so even a
                # broken trap cannot leak an indefinitely running test process.
        self.output.close()
        self.temp.cleanup()

    def records(self):
        if not self.events.exists():
            return []
        result = []
        for line in self.events.read_text().splitlines():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                # Another shim can be in the middle of its single append.
                pass
        return result

    def wait_event(self, event, count=1, timeout=3):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            matching = [entry for entry in self.records() if entry["event"] == event]
            if len(matching) >= count:
                return matching
            if self.process.poll() is not None:
                self.output.flush()
                self.fail(f"AML shell exited {self.process.returncode}: {(self.root / 'shell.log').read_text()}")
            time.sleep(0.01)
        self.fail(f"Did not observe {event}: {self.records()}")

    def launch(self, mode, config=None, mount_path="/mock/blob mount"):
        if config is None:
            config = {
                "gpu_model": "H100",
                "gpu_count": 2,
                "gpus": "1,3",
                "tunnel": True,
                "tunnel_name": "test-h100",
                "aml": {
                    "input_name": "job_storage",
                    "datastore_uri": "azureml://datastores/test_store/paths/test_prefix/",
                    "ref": "release/test-config",
                },
            }
        self.resolved = profile.normalize(config)
        self.command, self.inputs = profile.render_aml(config)
        # AML substitutes its input expression before the shell runs. Do not
        # shell-escape the fixture here: the renderer owns safe transport.
        token = "${{inputs." + self.resolved["aml"]["input_name"] + "}}"
        command = self.command.replace(token, mount_path)
        env = os.environ.copy()
        env.pop("BASH_ENV", None)
        env.update({
            "PATH": str(self.bin) + os.pathsep + env["PATH"],
            "AML_COMMAND_TEST_MODE": mode,
            "AML_COMMAND_TEST_EVENTS": str(self.events),
            "AML_COMMAND_REAL_BASH": self.real_bash,
            "AML_COMMAND_TEST_TMP": str(self.root),
            "AML_COMMAND_TEST_HOLD_SECONDS": str(self.resolved["aml"]["keep_alive_seconds"]),
        })
        # The outer shell is only our AML launcher. Give the generated command
        # an inherited BASH_ENV and verify its clean inner Bash ignores it.
        bash_env = self.root / "inherited-bash-env.sh"
        bash_env.write_text("touch " + shlex.quote(str(self.root / "bash-env-ran")) + "\n")
        command = "BASH_ENV=" + shlex.quote(str(bash_env)) + " " + command
        self.process = subprocess.Popen(
            [self.real_bash, "--noprofile", "--norc", "-c", command],
            cwd=REPO, env=env, stdout=self.output, stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def test_sleep_reached_and_job_alive_when_clone_repeatedly_fails(self):
        self.launch("clone_failure")
        self.wait_event("hold_started")
        self.wait_event("clone_failed", count=2)
        self.assertIsNone(self.process.poll())
        self.assertFalse(any(row["event"] == "session_started" for row in self.records()))

    def test_sleep_starts_before_blocked_clone_returns(self):
        self.launch("clone_blocked")
        self.wait_event("clone_started")
        hold = self.wait_event("hold_started")[0]
        self.assertEqual(hold["duration"], "8553600")
        self.assertIsNone(self.process.poll())
        self.assertFalse(any(row["event"] == "clone_succeeded" for row in self.records()))

    def test_session_failure_does_not_end_sleep_or_job(self):
        self.launch("session_failure")
        hold = self.wait_event("hold_started")[0]
        self.wait_event("session_failed")
        time.sleep(0.1)
        self.assertIsNone(self.process.poll())
        self.assertFalse(any(row["event"] == "terminated" and row["pid"] == hold["pid"] for row in self.records()))

    def test_configured_lifetime_and_clone_retry_are_used(self):
        self.launch("clone_failure", {"aml": {"keep_alive_seconds": 91, "clone_retry_seconds": 7}})
        self.assertEqual(self.wait_event("hold_started")[0]["duration"], "91")
        self.assertEqual(self.wait_event("retry_sleep")[0]["duration"], "7")
        self.wait_event("clone_failed", count=2)
        self.assertIsNone(self.process.poll())

    def test_literal_mount_and_embedded_config_survive_shell_transport(self):
        substitutions = [self.root / "dollar-substitution-ran", self.root / "backtick-substitution-ran"]
        special = (
            "space 'single' \"double\" $(touch " + str(substitutions[0])
            + ") `touch " + str(substitutions[1]) + "`"
        )
        mount = str(self.root / ("blob " + special))
        config = {
            "gpu_model": "NVIDIA H100 80GB HBM3",
            "gpu_count": 2,
            "gpus": "1,3",
            "blob_prefix": "project " + special,
            "local_root": str(self.root / ("local " + special)),
            "poll_seconds": 0.5,
            "tunnel": True,
            "tunnel_name": "custom-h100",
            "aml": {
                "input_name": "custom_blob_input",
                "datastore_uri": "azureml://datastores/custom/paths/project/",
                "ref": "abcdef0123456789abcdef0123456789abcdef0123",
            },
        }
        self.launch("success", config, mount_path=mount)
        session = self.wait_event("session_started")[0]
        self.assertEqual(session["args"][session["args"].index("--blob-root") + 1], mount)
        self.assertEqual(session["config"], self.resolved)
        self.assertEqual(json.loads(session["config_text"]), self.resolved)
        self.assertIs(type(session["config"]["gpu_count"]), int)
        self.assertIs(type(session["config"]["poll_seconds"]), float)
        self.assertIs(type(session["config"]["tunnel"]), bool)
        self.assertIsNone(session["config"]["blob_root"])
        self.assertEqual(self.inputs, {
            "custom_blob_input": {
                "type": "uri_folder", "path": config["aml"]["datastore_uri"], "mode": "rw_mount",
            },
        })
        self.assertEqual(self.wait_event("revision_fetch")[0]["ref"], config["aml"]["ref"])
        checkout = self.wait_event("revision_checkout")[0]
        self.assertEqual(checkout["args"][-2:], ["--detach", "FETCH_HEAD"])
        for sentinel in substitutions:
            self.assertFalse(sentinel.exists(), f"Shell executed literal user path: {sentinel}")

    def test_no_blob_or_tunnel_plan_has_no_aml_input_reference(self):
        self.launch("success", {})
        self.assertEqual(self.inputs, {})
        self.assertNotIn("${{inputs.", self.command)
        session = self.wait_event("session_started")[0]
        self.assertNotIn("--blob-root", session["args"])
        self.assertNotIn("--tunnel", session["args"])
        self.assertIsNone(session["config"]["blob_root"])
        self.assertIs(session["config"]["tunnel"], False)

    def test_term_cleans_owned_groups_and_preserves_unrelated_process(self):
        unrelated = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(8)"], start_new_session=True,
        )
        try:
            self.launch("success")
            hold = self.wait_event("hold_started")[0]
            session = self.wait_event("session_started")[0]
            self.assertEqual(session["args"][session["args"].index("--blob-root") + 1], "/mock/blob mount")
            self.assertIs(session["config"]["tunnel"], True)
            clean_bash = self.wait_event("clean_bash")[0]
            self.assertEqual(clean_bash["args"], ["--noprofile", "--norc"])
            self.assertIsNone(clean_bash["bash_env"])
            self.assertFalse((self.root / "bash-env-ran").exists())
            self.assertNotEqual(hold["pgid"], session["pgid"])
            self.process.terminate()
            self.assertEqual(self.process.wait(timeout=3), 143)
            terminated = {row["pid"] for row in self.records() if row["event"] == "terminated"}
            self.assertIn(hold["pid"], terminated)
            self.assertIn(session["pid"], terminated)
            self.assertIsNone(unrelated.poll())
        finally:
            if unrelated.poll() is None:
                unrelated.terminate()
            unrelated.wait(timeout=3)


if __name__ == "__main__":
    unittest.main()

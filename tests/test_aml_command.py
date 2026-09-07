"""Run the documented AML shell lifecycle with local command shims only."""

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest


REPO = Path(__file__).resolve().parents[1]
COMMAND = REPO / "docs" / "aml-command.sh"


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
    record("session_started", args=sys.argv[1:])
    if mode == "session_failure":
        record("session_failed", code=42)
        raise SystemExit(42)
    bounded_block()
elif kind == "git":
    record("clone_started", args=sys.argv[1:])
    if mode == "clone_blocked":
        bounded_block()
        raise SystemExit(23)
    if mode == "clone_failure":
        record("clone_failed", code=23)
        raise SystemExit(23)
    target = Path(sys.argv[-1])
    (target / "scripts").mkdir(parents=True)
    (target / "scripts" / "aml_start.sh").write_text("# test fixture; the bash shim handles this\n")
    record("clone_succeeded")
elif kind == "sleep":
    duration = sys.argv[1]
    record("hold_started" if duration == "8553600" else "retry_sleep", duration=duration)
    if duration == "8553600": bounded_block()
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

    def launch(self, mode):
        command = COMMAND.read_text().replace("${{inputs.lucayu}}", "/mock/blob mount")
        env = os.environ.copy()
        env.pop("BASH_ENV", None)
        env.update({
            "PATH": str(self.bin) + os.pathsep + env["PATH"],
            "AML_COMMAND_TEST_MODE": mode,
            "AML_COMMAND_TEST_EVENTS": str(self.events),
            "AML_COMMAND_REAL_BASH": self.real_bash,
            "AML_COMMAND_TEST_TMP": str(self.root),
        })
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

    def test_term_cleans_owned_groups_and_preserves_unrelated_process(self):
        unrelated = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(8)"], start_new_session=True,
        )
        try:
            self.launch("success")
            hold = self.wait_event("hold_started")[0]
            session = self.wait_event("session_started")[0]
            self.assertEqual(session["args"][session["args"].index("--blob-root") + 1], "/mock/blob mount")
            self.assertIn("--tunnel", session["args"])
            self.assertEqual(self.wait_event("clean_bash")[0]["args"], ["--noprofile", "--norc"])
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

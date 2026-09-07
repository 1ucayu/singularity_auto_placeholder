"""Run the GPU supervisor and VS Code tunnel as independent, restartable services."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import threading
from dataclasses import dataclass
import uuid


def log(message: str) -> None:
    print(f"[session] {message}", flush=True)


def default_local_root() -> Path:
    return Path(f"/tmp/singularity-auto-{os.getuid()}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--blob-root", type=Path, required=True,
                        help="Actual mounted directory: pass Azure ML ${{inputs.lucayu}}")
    result.add_argument("--blob-prefix", default="lucayu/sglang")
    result.add_argument("--local-root", type=Path, default=default_local_root())
    result.add_argument("--gpu-count", type=int, default=8,
                        help="Expected GPU count; a mismatch does not block startup")
    result.add_argument("--gpus", help="Explicit allocated physical GPU UUIDs, comma separated")
    result.add_argument("--idle-seconds", type=float, default=30)
    result.add_argument("--poll-seconds", type=float, default=1)
    result.add_argument("--max-utilization", type=int, default=5)
    result.add_argument("--max-memory-mib", type=int, default=256)
    result.add_argument("--tunnel", action="store_true")
    result.add_argument("--tunnel-name", default="aml-lucayu")
    result.add_argument("--retry-seconds", type=float, default=10,
                        help="Delay before restarting a failed service or retrying Blob setup")
    return result


def prepare_paths(args: argparse.Namespace) -> dict[str, Path | str]:
    # Do not touch Blob here: a slow/unavailable mount must not delay the tunnel.
    mount = Path(os.path.abspath(args.blob_root.expanduser()))
    prefix = Path(args.blob_prefix)
    if prefix.is_absolute() or ".." in prefix.parts or not prefix.parts:
        raise ValueError("--blob-prefix must be a nonempty relative path without '..'")
    persistent = mount / prefix
    local = Path(os.path.abspath(args.local_root.expanduser()))
    if local.is_relative_to(mount) or mount.is_relative_to(local):
        raise ValueError("--local-root must be separate from Blob; use a node-local SSD directory")
    local = local.resolve()
    # Keep generated launchers and control files private. Blob ACLs are managed
    # by Azure, so chmod on the Blob mount is intentionally not used.
    local.mkdir(parents=True, exist_ok=True, mode=0o700)
    if local.stat().st_uid != os.getuid():
        raise ValueError("--local-root belongs to another user")
    local.chmod(0o700)
    for name in ("bin", "control", "logs", "workspace", "cache", "tmp", "outputs", "tunnel"):
        (local / name).mkdir(exist_ok=True, mode=0o700)
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    (local / "outputs" / run_id).mkdir()
    return {"mount": mount, "persistent": persistent, "local": local, "run_id": run_id}


def write_environment(paths: dict[str, Path | str], repo: Path) -> dict[str, str]:
    local, mount, persistent = (Path(paths[key]) for key in ("local", "mount", "persistent"))
    values = {
        "BLOB_MOUNT": str(mount),
        "BLOB_ROOT": str(persistent),
        "LOCAL_WORK_ROOT": str(local / "workspace"),
        "OUTPUT_ROOT": str(local / "outputs" / str(paths["run_id"])),
        "HF_HOME": str(local / "cache" / "huggingface"),
        "HF_HUB_CACHE": str(local / "cache" / "huggingface" / "hub"),
        "XDG_CACHE_HOME": str(local / "cache"),
        "PIP_CACHE_DIR": str(local / "cache" / "pip"),
        "UV_CACHE_DIR": str(local / "cache" / "uv"),
        "TMPDIR": str(local / "tmp"),
        "PLACEHOLDER_CONTROL_DIR": str(local / "control"),
        "PLACEHOLDER_REPO": str(repo),
        "SESSION_RUN_ID": str(paths["run_id"]),
    }
    env_text = "# Generated for this job. Source in every new VS Code terminal.\n"
    env_text += "".join(f"export {key}={shlex.quote(value)}\n" for key, value in values.items())
    env_text += f"export PATH={shlex.quote(str(local / 'bin'))}:\"$PATH\"\n"
    (local / "env.sh").write_text(env_text)
    (local / "env.sh").chmod(0o600)
    # A launcher bound to the image's CUDA-enabled Python; task commands still
    # use their own shell/PATH and may select a different virtual environment.
    launcher = "#!/usr/bin/env bash\nset -euo pipefail\n"
    launcher += f"export PYTHONPATH={shlex.quote(str(repo))}${{PYTHONPATH:+:$PYTHONPATH}}\n"
    launcher += 'if [[ "$#" -eq 0 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then\n'
    launcher += f"  exec {shlex.quote(sys.executable)} -m singularity_placeholder --help\nfi\n"
    launcher += 'action="$1"\nshift\n'
    launcher += (f"exec {shlex.quote(sys.executable)} -m singularity_placeholder \"$action\" "
                 f"--control-dir {shlex.quote(str(local / 'control'))} \"$@\"\n")
    (local / "bin" / "placeholder").write_text(launcher)
    (local / "bin" / "placeholder").chmod(0o700)
    return values



def prepare_storage(paths: dict[str, Path | str], gpu_count: int) -> None:
    """Best-effort durable layout; called in a thread, never on the tunnel path."""
    mount, persistent = Path(paths["mount"]), Path(paths["persistent"])
    if not mount.is_dir():
        raise OSError(f"Blob mount is not available: {mount}")
    for name in ("models", "datasets", "traces/input", "runs", "sessions", "code-snapshots"):
        (persistent / name).mkdir(parents=True, exist_ok=True)
    durable_session = persistent / "sessions" / str(paths["run_id"])
    durable_session.mkdir(exist_ok=True)
    manifest = {
        "run_id": paths["run_id"], "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "azureml_run_id": os.environ.get("AZUREML_RUN_ID"),
        "gpu_count_requested": gpu_count,
        "blob_root": str(persistent), "local_root": str(paths["local"]),
        "python": sys.version.split()[0],
    }
    (durable_session / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def storage_loop(paths: dict[str, Path | str], gpu_count: int,
                 stopping: threading.Event, retry_seconds: float) -> None:
    while not stopping.is_set():
        try:
            prepare_storage(paths, gpu_count)
            if not stopping.is_set():
                log(f"Blob personal directory ready: {paths['persistent']}")
            return
        except OSError as exc:
            log(f"Blob setup pending: {exc}; retrying in {retry_seconds:g}s. Local services continue.")
        stopping.wait(retry_seconds)


@dataclass
class Service:
    name: str
    command: list[str]
    env: dict[str, str]
    process: subprocess.Popen | None = None
    retry_at: float = 0

    def tick(self, now: float, retry_seconds: float) -> None:
        if self.process is not None:
            result = self.process.poll()
            if result is None:
                return
            log(f"{self.name} exited with code {result}; restarting in {retry_seconds:g}s. Other services continue.")
            self.process = None
            self.retry_at = now + retry_seconds
        if now < self.retry_at:
            return
        try:
            self.process = subprocess.Popen(self.command, env=self.env, start_new_session=True)
            log(f"Started {self.name} (pid {self.process.pid})")
        except OSError as exc:
            log(f"Cannot start {self.name}: {exc}; retrying in {retry_seconds:g}s")
            self.retry_at = now + retry_seconds


def run_services(services: list[Service], stopping: threading.Event,
                 retry_seconds: float) -> None:
    """A failed service never stops its siblings or the session."""
    try:
        while not stopping.is_set():
            for service in services:
                service.tick(time.monotonic(), retry_seconds)
            stopping.wait(0.25)
    finally:
        children = [service.process for service in services if service.process is not None]
        for child in reversed(children):
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 20
        for child in reversed(children):
            try:
                child.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,19}", args.tunnel_name):
        raise SystemExit("--tunnel-name must be 1-20 letters/digits/hyphens, starting with a letter or digit")
    if args.gpu_count < 1 or args.idle_seconds < 0 or args.poll_seconds <= 0 or args.retry_seconds <= 0:
        raise SystemExit("GPU count, poll interval and retry interval must be positive; idle interval must be nonnegative")
    paths = prepare_paths(args)
    local = Path(paths["local"])
    repo = Path(__file__).resolve().parents[1]
    import fcntl
    with (local / "session.lock").open("a") as session_lock:
        try:
            fcntl.flock(session_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("another AML session already owns this local root")
            return 2
        values = write_environment(paths, repo)
        child_env = os.environ.copy()
        child_env.update(values)
        child_env["PYTHONPATH"] = str(repo) + os.pathsep + child_env.get("PYTHONPATH", "")
        child_env["PATH"] = str(local / "bin") + os.pathsep + child_env.get("PATH", "")
        supervisor_env = child_env.copy()
        for name in ("VSCODE_CLI_ACCESS_TOKEN", "VSCODE_CLI_REFRESH_TOKEN"):
            supervisor_env.pop(name, None)
        log(f"In a new VS Code terminal: source {shlex.quote(str(local / 'env.sh'))}")
        log("Then: placeholder status | placeholder run -- python your_script.py")
        services = []
        # Start the tunnel first. No CUDA, PyTorch or Blob checks gate connectivity.
        if args.tunnel:
            services.append(Service("tunnel", [
                sys.executable, "-u", "-m", "singularity_placeholder.tunnel",
                "--runtime-dir", str(local / "tunnel"), "--name", args.tunnel_name,
                "--log-dir", str(local / "logs" / "tunnel"),
            ], child_env))
        command = [sys.executable, "-u", "-m", "singularity_placeholder", "supervise",
                   "--control-dir", str(local / "control"), "--state-dir", str(local / "logs"),
                   "--gpu-count", str(args.gpu_count), "--idle-seconds", str(args.idle_seconds),
                   "--poll-seconds", str(args.poll_seconds), "--max-utilization", str(args.max_utilization),
                   "--max-memory-mib", str(args.max_memory_mib)]
        if args.gpus:
            command += ["--gpus", args.gpus]
        services.append(Service("GPU supervisor", command, supervisor_env))
        stopping = threading.Event()
        old_handlers = {}
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            old_handlers[sig] = signal.signal(sig, lambda *_: stopping.set())
        storage = threading.Thread(target=storage_loop, args=(paths, args.gpu_count, stopping, args.retry_seconds),
                                   name="blob-setup", daemon=True)
        storage.start()
        try:
            run_services(services, stopping, args.retry_seconds)
        finally:
            stopping.set()
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        log(f"startup failed: {exc}")
        raise SystemExit(1)

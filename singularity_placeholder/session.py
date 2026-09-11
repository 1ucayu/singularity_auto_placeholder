"""Run GPU supervision, SSH relay and VS Code tunnel as independent services."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import threading
from dataclasses import dataclass
import uuid

from . import profile
from .tunnel import local_filesystem


def log(message: str) -> None:
    print(f"[session] {message}", flush=True)


def default_local_root() -> Path:
    return Path(f"/tmp/singularity-auto-{os.getuid()}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS)
    result.add_argument("--config", type=Path, help="JSON profile; explicit CLI flags override its values")
    result.add_argument("--dry-run", action="store_true", help="Print the resolved configuration without creating paths or starting services")
    blob = result.add_mutually_exclusive_group()
    blob.add_argument("--blob-root", help="Actual mounted directory on this node; optional")
    blob.add_argument("--no-blob", dest="blob_root", action="store_const", const=None,
                      help="Disable Blob setup, overriding the profile")
    result.add_argument("--blob-prefix", help="Relative suffix; empty means blob-root is already the final directory")
    result.add_argument("--local-root", help="Node-local directory; defaults to /tmp/singularity-auto-<UID>")
    result.add_argument("--gpu-count", type=lambda value: None if value.strip().lower() == "auto" else int(value),
                        help="Expected count or auto; never allocates, limits or expands GPUs")
    result.add_argument("--gpu-model", help="Planning metadata only, not a hardware filter")
    result.add_argument("--gpus", help="Explicit allocated physical GPU UUIDs, comma separated")
    result.add_argument("--idle-seconds", type=float)
    result.add_argument("--poll-seconds", type=float)
    result.add_argument("--max-utilization", type=int)
    result.add_argument("--max-memory-mib", type=int)
    result.add_argument("--matrix-size", type=int)
    result.add_argument("--worker-stop-seconds", type=float)
    result.add_argument("--worker-retry-seconds", type=float)
    result.add_argument("--tunnel", action=argparse.BooleanOptionalAction)
    result.add_argument("--tunnel-name", help="Unique 1-20 character name; generated when omitted")
    result.add_argument("--retry-seconds", type=float,
                        help="Delay before restarting a failed service or retrying Blob setup")
    return result


def parse_options(argv: list[str] | None = None) -> tuple[argparse.Namespace, bool]:
    overrides = vars(parser().parse_args(argv))
    config = profile.load(overrides.pop("config", None))
    dry_run = overrides.pop("dry_run", False)
    config.update(overrides)
    if "blob_root" in overrides:
        # The CLI has resolved or disabled the input. Persist an unambiguous
        # runtime profile that can itself be validated and reused.
        config["aml"]["datastore_uri"] = None
        if overrides["blob_root"] is None and "blob_prefix" not in overrides:
            config["blob_prefix"] = ""
    config["local_root"] = config["local_root"] or str(default_local_root())
    config = profile.normalize(config)
    if not dry_run and config["aml"]["datastore_uri"] and config["blob_root"] is None:
        raise ValueError("Pass --blob-root with the actual AML mount, or use profile render-aml for a new job")
    if config["blob_prefix"] and not (config["blob_root"] or config["aml"]["datastore_uri"]):
        raise ValueError("blob_prefix needs a Blob root; use --blob-root or --no-blob")
    return argparse.Namespace(**config), dry_run


def prepare_paths(args: argparse.Namespace) -> dict[str, Path | str | None]:
    # Do not touch Blob here: a slow/unavailable mount must not delay the tunnel.
    mount = Path(os.path.abspath(Path(args.blob_root).expanduser())) if args.blob_root is not None else None
    prefix = Path(args.blob_prefix)
    if prefix.is_absolute() or ".." in prefix.parts:
        raise ValueError("--blob-prefix must be relative without '..'")
    persistent = mount / prefix if mount is not None else None
    local = Path(os.path.abspath(Path(args.local_root).expanduser()))
    if mount is not None and (local.is_relative_to(mount) or mount.is_relative_to(local)):
        raise ValueError("--local-root must be separate from Blob; use a node-local SSD directory")
    local = local.resolve()
    if mount is not None and (local.is_relative_to(mount) or mount.is_relative_to(local)):
        raise ValueError("--local-root must be separate from Blob")
    local_filesystem(local)
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


def write_environment(paths: dict[str, Path | str | None], repo: Path) -> dict[str, str]:
    local = Path(paths["local"])
    values = {
        "BLOB_MOUNT": str(paths["mount"]) if paths["mount"] is not None else "",
        "BLOB_ROOT": str(paths["persistent"]) if paths["persistent"] is not None else "",
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



def prepare_storage(paths: dict[str, Path | str | None], gpu_count: int | None) -> None:
    """Best-effort durable layout; called in a thread, never on the tunnel path."""
    if paths["mount"] is None:
        return
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
        "profile": paths.get("profile"),
    }
    (durable_session / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def storage_loop(paths: dict[str, Path | str | None], gpu_count: int | None,
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
    args, dry_run = parse_options(argv)
    if dry_run:
        print(json.dumps(vars(args), indent=2, allow_nan=False))
        return 0
    if args.tunnel and args.tunnel_name is None:
        args.tunnel_name = "aml-" + uuid.uuid4().hex[:12]
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
        effective_profile = vars(args)
        (local / "profile.json").write_text(json.dumps(effective_profile, indent=2) + "\n")
        (local / "profile.json").chmod(0o600)
        paths["profile"] = effective_profile
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
            log(f"VS Code tunnel name: {args.tunnel_name}")
            services.append(Service("tunnel", [
                sys.executable, "-u", "-m", "singularity_placeholder.tunnel",
                "--runtime-dir", str(local / "tunnel"), "--name", args.tunnel_name,
                "--log-dir", str(local / "logs" / "tunnel"),
            ], child_env))
        if args.relay["enabled"]:
            # Credential/mount/SSH checks happen only inside this child. A new
            # job can occupy GPUs while node-local relay credentials are added.
            services.append(Service("SSH relay", [
                sys.executable, "-u", "-m", "singularity_placeholder.relay",
                "--config", str(local / "profile.json"),
                "--runtime-dir", str(local / "relay"),
            ], supervisor_env))
        command = [sys.executable, "-u", "-m", "singularity_placeholder", "supervise",
                   "--control-dir", str(local / "control"), "--state-dir", str(local / "logs"),
                   "--gpu-count", str(args.gpu_count) if args.gpu_count is not None else "auto", "--idle-seconds", str(args.idle_seconds),
                   "--poll-seconds", str(args.poll_seconds), "--max-utilization", str(args.max_utilization),
                   "--max-memory-mib", str(args.max_memory_mib), "--matrix-size", str(args.matrix_size),
                   "--worker-stop-seconds", str(args.worker_stop_seconds),
                   "--worker-retry-seconds", str(args.worker_retry_seconds)]
        if args.gpus:
            command += ["--gpus", args.gpus]
        services.append(Service("GPU supervisor", command, supervisor_env))
        stopping = threading.Event()
        old_handlers = {}
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            old_handlers[sig] = signal.signal(sig, lambda *_: stopping.set())
        if args.blob_root is not None:
            storage = threading.Thread(target=storage_loop, args=(paths, args.gpu_count, stopping, args.retry_seconds),
                                       name="blob-setup", daemon=True)
            storage.start()
        else:
            log("Blob storage disabled; outputs are node-local")
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

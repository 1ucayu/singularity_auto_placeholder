"""Azure ML entrypoint: local runtime, foreground supervisor, optional tunnel."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
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
    result.add_argument("--gpu-count", type=int, default=8)
    result.add_argument("--gpus", help="Explicit allocated physical GPU UUIDs, comma separated")
    result.add_argument("--idle-seconds", type=float, default=30)
    result.add_argument("--poll-seconds", type=float, default=1)
    result.add_argument("--max-utilization", type=int, default=5)
    result.add_argument("--max-memory-mib", type=int, default=256)
    result.add_argument("--tunnel", action="store_true")
    result.add_argument("--tunnel-name", default="aml-lucayu")
    result.add_argument("--debug-grace-seconds", type=float, default=600,
                        help="Keep an enabled tunnel alive this long after supervisor failure (0 disables)")
    return result


def supervisor_failure_action(returncode: int, *, tunnel_enabled: bool,
                              grace_seconds: float, now: float, deadline: float | None) -> tuple[bool, float | None]:
    """Return (should_exit, deadline), keeping a bounded tunnel debugging window."""
    if returncode == 0 or not tunnel_enabled or grace_seconds <= 0:
        return True, deadline
    if deadline is None:
        deadline = now + grace_seconds
    return now >= deadline, deadline


def prepare_paths(args: argparse.Namespace) -> dict[str, Path | str]:
    mount = args.blob_root.expanduser().resolve(strict=True)
    if not mount.is_dir():
        raise ValueError("--blob-root must be the mounted directory, not an azureml:// URI")
    prefix = Path(args.blob_prefix)
    if prefix.is_absolute() or ".." in prefix.parts or not prefix.parts:
        raise ValueError("--blob-prefix must be a nonempty relative path without '..'")
    persistent = (mount / prefix).resolve()
    if not persistent.is_relative_to(mount):
        raise ValueError("Blob prefix escapes the supplied mount")
    local = args.local_root.expanduser().resolve()
    if local.is_relative_to(mount) or mount.is_relative_to(local):
        raise ValueError("--local-root must be separate from Blob; use a node-local SSD directory")
    # Keep generated launchers and control files private. Blob ACLs are managed
    # by Azure, so chmod on the Blob mount is intentionally not used.
    local.mkdir(parents=True, exist_ok=True, mode=0o700)
    if local.stat().st_uid != os.getuid():
        raise ValueError("--local-root belongs to another user")
    local.chmod(0o700)
    for name in ("bin", "control", "logs", "workspace", "cache", "tmp", "outputs", "tunnel"):
        (local / name).mkdir(exist_ok=True, mode=0o700)
    for name in ("models", "datasets", "traces/input", "runs", "sessions", "code-snapshots"):
        (persistent / name).mkdir(parents=True, exist_ok=True)
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    (local / "outputs" / run_id).mkdir()
    (persistent / "sessions" / run_id).mkdir()
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


def preflight() -> None:
    if sys.platform != "linux":
        raise RuntimeError("The AML runtime must be Linux with NVIDIA CUDA GPUs")
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("nvidia-smi is missing; select a GPU-enabled Azure ML environment")
    subprocess.run(
        [sys.executable, "-c", "import torch; assert torch.cuda.is_available(), 'CUDA-enabled torch required'"],
        check=True, timeout=90,
    )


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,19}", args.tunnel_name):
        raise SystemExit("--tunnel-name must be 1-20 letters/digits/hyphens, starting with a letter or digit")
    if args.gpu_count < 1 or args.idle_seconds < 0 or args.poll_seconds <= 0 or args.debug_grace_seconds < 0:
        raise SystemExit("GPU count and poll interval must be positive; idle interval must be nonnegative")
    preflight()
    paths = prepare_paths(args)
    local = Path(paths["local"])
    repo = Path(__file__).resolve().parents[1]
    # A second entrypoint must not overwrite another session's launchers/env.
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
        manifest = {
            "run_id": paths["run_id"], "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "azureml_run_id": os.environ.get("AZUREML_RUN_ID"),
            "gpu_count_requested": args.gpu_count,
            "blob_root": str(paths["persistent"]), "local_root": str(local),
            "python": sys.version.split()[0],
        }
        durable_session = Path(paths["persistent"]) / "sessions" / str(paths["run_id"])
        (durable_session / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        log(f"Blob personal directory: {paths['persistent']}")
        log(f"In a new VS Code terminal: source {shlex.quote(str(local / 'env.sh'))}")
        log("Then: placeholder status | placeholder run -- python your_script.py")
        command = [sys.executable, "-u", "-m", "singularity_placeholder", "supervise",
                   "--control-dir", str(local / "control"), "--state-dir", str(local / "logs"),
                   "--gpu-count", str(args.gpu_count), "--idle-seconds", str(args.idle_seconds),
                   "--poll-seconds", str(args.poll_seconds), "--max-utilization", str(args.max_utilization),
                   "--max-memory-mib", str(args.max_memory_mib)]
        if args.gpus:
            command += ["--gpus", args.gpus]
        children: list[subprocess.Popen] = []
        stopping = False

        def stop(_signum: int, _frame: object) -> None:
            nonlocal stopping
            stopping = True

        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, stop)
        exit_code = 0
        try:
            supervisor = subprocess.Popen(command, env=supervisor_env, start_new_session=True)
            children.append(supervisor)
            if args.tunnel:
                tunnel = subprocess.Popen(
                    [sys.executable, "-u", "-m", "singularity_placeholder.tunnel",
                     "--runtime-dir", str(local / "tunnel"), "--name", args.tunnel_name,
                     "--log-dir", str(local / "logs" / "tunnel")],
                    env=child_env, start_new_session=True,
                )
                children.append(tunnel)
            tunnel_reported = False
            failure_deadline: float | None = None
            while not stopping:
                result = supervisor.poll()
                if result is not None:
                    exit_code = result if result >= 0 else 128 - result
                    first_failure = failure_deadline is None
                    should_exit, failure_deadline = supervisor_failure_action(
                        result, tunnel_enabled=args.tunnel and tunnel.poll() is None,
                        grace_seconds=args.debug_grace_seconds,
                        now=time.monotonic(), deadline=failure_deadline,
                    )
                    if should_exit:
                        log(f"supervisor exited with code {result}; ending job entrypoint")
                        break
                    if first_failure:
                        log(f"supervisor exited with code {result}; GPU supervision stopped; check worker state. "
                            f"Keeping the tunnel available for up to {args.debug_grace_seconds:g}s for diagnostics; "
                            "the platform may still reclaim the job.")
                if args.tunnel and tunnel.poll() is not None and not tunnel_reported:
                    supervisor_state = "GPU supervisor continues" if result is None else "GPU supervisor has also exited"
                    log(f"tunnel helper exited with code {tunnel.returncode}; see tunnel logs; {supervisor_state}")
                    tunnel_reported = True
                time.sleep(0.25)
        finally:
            for child in reversed(children):
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            deadline = time.monotonic() + 60
            for child in reversed(children):
                try:
                    child.wait(timeout=max(0.1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()
        return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        log(f"startup failed: {exc}")
        raise SystemExit(1)

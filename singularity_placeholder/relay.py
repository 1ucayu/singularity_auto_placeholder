"""Forward GPU loopback services to private Unix sockets on an SSH jump host.

No reverse TCP listener is used: sshd GatewayPorts may override a requested
loopback bind. A remote socket lease prevents reconnects replacing active
sessions and clears refused stale sockets left after a dropped SSH connection.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import selectors
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time

from . import profile, remote_socket
from .tunnel import local_filesystem, private_directory


def ssh_command(config: dict) -> list[str]:
    """Common SSH argv; no shell, password prompt, or agent forwarding."""
    relay = profile.normalize(config)["relay"]
    if not relay["enabled"]:
        raise ValueError("relay is disabled")
    return [
        "ssh", "-F", "/dev/null", "-T",
        "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
        "-o", "IdentityAgent=none", "-o", "ForwardAgent=no",
        "-o", "PasswordAuthentication=no", "-o", "KbdInteractiveAuthentication=no",
        "-o", "StrictHostKeyChecking=yes",
        "-o", f"UserKnownHostsFile={relay['known_hosts_file']}",
        "-o", "GlobalKnownHostsFile=/dev/null",
        "-o", "ExitOnForwardFailure=yes", "-o", "ControlPersist=no",
        "-o", f"ConnectTimeout={relay['connect_timeout']}",
        "-o", f"ServerAliveInterval={relay['server_alive_interval']}",
        "-o", f"ServerAliveCountMax={relay['server_alive_count_max']}",
        "-i", relay["identity_file"], "-p", str(relay["port"]),
    ]


def destination(config: dict) -> str:
    return f"{config['relay']['user']}@{config['relay']['host']}"


def forward_command(config: dict, control: Path) -> list[str]:
    command = ssh_command(config) + ["-S", str(control), "-O", "forward"]
    for item in config["relay"]["forwards"]:
        command += ["-R", f"{item['remote_socket']}:127.0.0.1:{item['local_port']}"]
    return command + ["--", destination(config)]


def check_credentials(config: dict) -> None:
    """Check metadata only; OpenSSH reads and validates the actual key/host key."""
    mount = Path(config["blob_root"]).resolve() if config["blob_root"] else None
    for name in ("identity_file", "known_hosts_file"):
        path = Path(config["relay"][name]).resolve(strict=True)
        local_filesystem(path)
        if mount is not None and path.is_relative_to(mount):
            raise ValueError(f"relay.{name} must not be stored on Blob")
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
            raise ValueError(f"relay.{name} must be a regular file owned by the session user")
        forbidden = 0o077 if name == "identity_file" else 0o022
        if metadata.st_mode & forbidden:
            raise ValueError(f"relay.{name} has unsafe permissions; use chmod 600")


def await_marker(process: subprocess.Popen, marker: str, timeout: float,
                 stopping: threading.Event) -> None:
    pending = b""
    deadline = time.monotonic() + timeout
    with selectors.DefaultSelector() as events:
        events.register(process.stdout, selectors.EVENT_READ)
        while not stopping.is_set() and time.monotonic() < deadline:
            if events.select(timeout=min(0.25, max(0, deadline - time.monotonic()))):
                data = os.read(process.stdout.fileno(), 4096)
                if not data:
                    raise RuntimeError(f"SSH ended before {marker}")
                pending += data
                if len(pending) > 65536:
                    raise RuntimeError("Unexpectedly large SSH socket-lease response")
                lines = pending.split(b"\n")
                pending = lines.pop()
                if marker.encode() in lines:
                    return
        raise RuntimeError(f"Stopped or timed out waiting for {marker}")


def run_master(config: dict, control: Path, stopping: threading.Event) -> int:
    source = Path(remote_socket.__file__).read_text()
    sockets = [item["remote_socket"] for item in config["relay"]["forwards"]]
    remote = shlex.join(["python3", "-u", "-c", source, *sockets])
    command = ssh_command(config) + ["-M", "-S", str(control), "--", destination(config), remote]
    # Inherit the relay process group. Session cleanup owns the master too.
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    timeout = config["relay"]["connect_timeout"] + 15
    try:
        await_marker(child, remote_socket.READY, timeout, stopping)
        result = subprocess.run(forward_command(config, control), capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f"SSH forwarding failed: {result.stderr.strip()}")
        child.stdin.write(b"BOUND\n")
        child.stdin.flush()
        await_marker(child, remote_socket.BOUND, timeout, stopping)
        print(f"[relay] Private socket forwards established: {', '.join(sockets)}", flush=True)
        while not stopping.wait(0.25):
            result_code = child.poll()
            if result_code is not None:
                return result_code if result_code >= 0 else 128 - result_code
        return 0
    finally:
        # EOF releases the remote lease and unlinks only the socket inodes
        # created by this connection. Broken connections are recovered later
        # by the stale-socket check under the same advisory lock.
        try:
            child.stdin.close()
        except OSError:
            pass
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        child.stdout.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = profile.load(args.config)
        if not config["relay"]["enabled"]:
            raise ValueError("relay is disabled")
        runtime = args.runtime_dir or Path(config["local_root"] or f"/tmp/singularity-auto-{os.getuid()}") / "relay"
        if args.dry_run:
            print(shlex.join(forward_command(config, runtime / "control.sock")))
            return 0
        check_credentials(config)
        if shutil.which("ssh") is None:
            raise ValueError("OpenSSH client is missing from this image; install openssh-client")
        runtime = private_directory(runtime)
        stopping = threading.Event()
        handlers = {sig: signal.signal(sig, lambda *_: stopping.set())
                    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
        try:
            with tempfile.TemporaryDirectory(prefix="connection-", dir=runtime) as directory:
                control = Path(directory) / "ssh.sock"
                if len(str(control).encode()) > 100:
                    raise ValueError("SSH control socket path is too long; use a shorter node-local runtime directory")
                return run_master(config, control, stopping)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"[relay] Unavailable: {exc}. GPU supervision continues; the session will retry.", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

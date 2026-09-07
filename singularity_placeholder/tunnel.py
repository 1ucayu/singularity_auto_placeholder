"""Optional VS Code tunnel supervisor; all credentials stay on local disk.

VS Code CLI sources for the supported options:
https://github.com/microsoft/vscode/blob/main/cli/src/commands/args.rs
https://github.com/microsoft/vscode/blob/main/cli/src/auth.rs

Inject VSCODE_CLI_ACCESS_TOKEN through an approved secret mechanism for an
unattended first login. Otherwise GitHub device authorization is required.
The helper never saves credentials to the persistent experiment directory.
"""

from __future__ import annotations

import argparse
import fcntl
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import ssl
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request


MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024
SECRET_NAMES = ("VSCODE_CLI_ACCESS_TOKEN", "VSCODE_CLI_REFRESH_TOKEN")


def local_filesystem(path: Path, mountinfo: str | None = None) -> None:
    """Reject known shared/FUSE mounts before placing OAuth state there."""
    if mountinfo is None:
        source = Path("/proc/self/mountinfo")
        if not source.exists():  # Unit tests also run on macOS.
            return
        mountinfo = source.read_text()
    resolved = path.resolve()
    selected: tuple[int, str] | None = None
    for line in mountinfo.splitlines():
        columns = line.split()
        try:
            separator = columns.index("-")
            mount = Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), columns[4]))
            resolved.relative_to(mount)
        except (ValueError, IndexError):
            continue
        candidate = (len(mount.parts), columns[separator + 1])
        if selected is None or candidate[0] > selected[0]:
            selected = candidate
    if selected and (selected[1].startswith(("fuse", "nfs")) or selected[1] in {
        "cifs", "smb3", "ceph", "lustre", "glusterfs", "9p", "afs",
    }):
        raise ValueError("Tunnel runtime and logs must use node-local disk, not a shared/Blob mount")


def private_directory(path: Path) -> Path:
    local_filesystem(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path.resolve()


class HTTPSRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.lower().startswith("https://"):
            raise ValueError("Refusing a non-HTTPS CLI download redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def cli_download_url(machine: str | None = None) -> str:
    machine = (machine or platform.machine()).lower()
    architecture = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(machine)
    if architecture is None:
        raise ValueError("Unsupported CPU architecture; set CODE_CLI_PATH to a compatible VS Code CLI")
    return f"https://update.code.visualstudio.com/latest/cli-alpine-{architecture}/stable"


def install_cli(runtime: Path, override: str | None = None) -> Path:
    if override:
        binary = Path(override).expanduser().resolve()
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError("CODE_CLI_PATH must point to an executable VS Code CLI")
        return binary
    binary = runtime / "bin" / "code"
    if binary.is_file() and os.access(binary, os.X_OK):
        return binary
    if platform.system() != "Linux":
        raise ValueError("Automatic CLI installation supports Linux only; set CODE_CLI_PATH")
    binary.parent.mkdir(mode=0o700, exist_ok=True)
    opener = urllib.request.build_opener(
        HTTPSRedirectHandler(), urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )
    with tempfile.TemporaryDirectory(prefix="download-", dir=runtime) as temporary:
        archive = Path(temporary) / "cli.tar.gz"
        request = urllib.request.Request(cli_download_url(), headers={"User-Agent": "singularity-placeholder/1"})
        with opener.open(request, timeout=45) as response, archive.open("wb") as destination:
            size = 0
            while block := response.read(1024 * 1024):
                size += len(block)
                if size > MAX_DOWNLOAD_BYTES:
                    raise ValueError("VS Code CLI download exceeded the size limit")
                destination.write(block)
        # Copy only the executable, never extract arbitrary archive paths/symlinks.
        with tarfile.open(archive, "r:gz") as bundle:
            members = [item for item in bundle if item.name in ("code", "./code") and item.isfile()]
            if len(members) != 1 or members[0].size > MAX_DOWNLOAD_BYTES:
                raise ValueError("Downloaded CLI archive does not contain one valid code executable")
            extracted = bundle.extractfile(members[0])
            if extracted is None:
                raise ValueError("Cannot read the downloaded code executable")
            staged = Path(temporary) / "code"
            with extracted, staged.open("wb") as destination:
                shutil.copyfileobj(extracted, destination)
            staged.chmod(0o700)
            os.replace(staged, binary)
    return binary


def child_environment(source: dict[str, str], runtime: Path, login: bool = False) -> dict[str, str]:
    environment = dict(source)
    environment["VSCODE_CLI_DATA_DIR"] = str(runtime / "cli-data")
    environment["VSCODE_CLI_USE_FILE_KEYCHAIN"] = "1"
    if not login:
        for name in SECRET_NAMES:
            environment.pop(name, None)
    return environment


def redact(message: str, secrets: tuple[str, ...]) -> str:
    for secret in sorted((value for value in secrets if value), key=len, reverse=True):
        message = message.replace(secret, "[REDACTED]")
    # Also cover conventional GitHub tokens should the CLI print an error with one.
    return re.sub(r"\b(?:gh[pousr]_[A-Za-z0-9_]{16,}|github_pat_[A-Za-z0-9_]{16,})\b", "[REDACTED]", message)


def make_logger(log_dir: Path, secrets: tuple[str, ...]) -> logging.Logger:
    logger = logging.getLogger("singularity_placeholder.tunnel")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for previous in logger.handlers[:]:
        previous.close()
        logger.removeHandler(previous)

    class Redactor(logging.Filter):
        def filter(self, record):
            record.msg = redact(record.getMessage(), secrets)
            record.args = ()
            return True

    handler = RotatingFileHandler(log_dir / "tunnel.log", maxBytes=5 * 1024 * 1024, backupCount=2)
    (log_dir / "tunnel.log").chmod(0o600)
    for target in (handler, logging.StreamHandler(sys.stdout)):
        target.setFormatter(logging.Formatter("%(asctime)s [tunnel] %(message)s"))
        target.addFilter(Redactor())
        logger.addHandler(target)
    return logger


def stop_process(process: subprocess.Popen, timeout: float = 10) -> None:
    # The helper created this process group; no other workload is signalled.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def run_child(command: list[str], environment: dict[str, str], stop: threading.Event,
              logger: logging.Logger, timeout: float | None = None) -> int:
    # Exec through a tiny child-side helper, avoiding preexec_fn in this
    # threaded process. On Linux the CLI receives SIGTERM if this helper is
    # killed, so the session's restart loop does not leave competing tunnels.
    guarded = [sys.executable, str(Path(__file__).with_name("parent_exec.py")),
               "--parent-pid", str(os.getpid()), "--", *command]
    process = subprocess.Popen(guarded, env=environment, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, errors="replace", bufsize=1, start_new_session=True)

    def relay():
        assert process.stdout is not None
        for line in process.stdout:
            logger.info("%s", line.rstrip())

    reader = threading.Thread(target=relay, daemon=True)
    reader.start()
    started = time.monotonic()
    timed_out = False
    try:
        while process.poll() is None:
            if stop.wait(0.2):
                break
            if timeout is not None and time.monotonic() - started >= timeout:
                timed_out = True
                logger.error("GitHub login timed out; authorization is still required")
                break
    finally:
        stop_process(process)
        reader.join(timeout=2)
        if process.stdout is not None and not reader.is_alive():
            process.stdout.close()
    return 124 if timed_out else int(process.returncode or 0)


def supervise(runtime: Path, name: str, stop: threading.Event, logger: logging.Logger,
              *, max_failures: int = 10, login_timeout: float = 1200) -> int:
    environment = dict(os.environ)
    failures = 0
    while not stop.is_set():
        tunnel_began: float | None = None
        try:
            binary = install_cli(runtime, environment.get("CODE_CLI_PATH"))
            base = [str(binary), "--cli-data-dir", str(runtime / "cli-data")]
            cached_credentials = runtime / "cli-data" / "token.json"
            if environment.get("VSCODE_CLI_ACCESS_TOKEN") or not cached_credentials.exists():
                logger.info("Preparing GitHub tunnel authentication; device login is needed if no valid secret was injected")
                result = run_child(base + ["tunnel", "user", "login", "--provider", "github"],
                                   child_environment(environment, runtime, login=True), stop, logger,
                                   timeout=login_timeout)
                if stop.is_set():
                    return 0
                if result:
                    raise RuntimeError(f"VS Code login exited with status {result}")
                # A stored credential is enough for subsequent restarts in this job.
                for secret_name in SECRET_NAMES:
                    environment.pop(secret_name, None)
            logger.info("Starting tunnel %s; connect with the same GitHub account", name)
            tunnel_began = time.monotonic()
            result = run_child(base + ["tunnel", "--accept-server-license-terms", "--name", name,
                                       "--server-data-dir", str(runtime / "server")],
                               child_environment(environment, runtime), stop, logger)
            if stop.is_set():
                return 0
            logger.warning("VS Code tunnel exited with status %s", result)
        except (OSError, ValueError, RuntimeError, tarfile.TarError, EOFError) as error:
            logger.error("%s", error)
        if tunnel_began is not None and time.monotonic() - tunnel_began >= 300:
            failures = 0
        failures += 1
        if failures >= max_failures:
            logger.error("Tunnel stopped after %s consecutive failures; GPU supervisor is independent", failures)
            return 1
        delay = min(60, 2 ** failures)
        logger.info("Retrying tunnel in %s seconds", delay)
        stop.wait(delay)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", required=True, type=Path, help="Node-local runtime root (never the Blob mount)")
    parser.add_argument("--log-dir", required=True, type=Path, help="Node-local log directory")
    parser.add_argument("--name", required=True, help="Tunnel name; use distinct names for simultaneous jobs")
    parser.add_argument("--max-failures", type=int, default=10)
    parser.add_argument("--login-timeout", type=float, default=1200)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,19}", args.name):
        parser.error("--name must contain 1–20 letters, digits or hyphens and start with a letter or digit")
    if args.max_failures < 1 or args.login_timeout <= 0:
        parser.error("--max-failures and --login-timeout must be positive")
    os.umask(0o077)
    try:
        runtime = private_directory(args.runtime_dir / "vscode-tunnel")
        log_dir = private_directory(args.log_dir)
        private_directory(runtime / "cli-data")
        private_directory(runtime / "server")
        lock = (runtime / "helper.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            parser.exit(1, "A tunnel helper is already using this runtime directory\n")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Tunnel setup failed: {error}\n")
    secrets = tuple(os.environ.get(name, "") for name in SECRET_NAMES)
    logger = make_logger(log_dir, secrets)
    stop = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        return supervise(runtime, args.name, stop, logger, max_failures=args.max_failures,
                         login_timeout=args.login_timeout)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())

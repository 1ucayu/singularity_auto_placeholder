"""Foreground supervisor and a local, acknowledged workload reservation protocol."""

from __future__ import annotations

import errno
import fcntl
import json
import logging
import logging.handlers
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .monitor import MonitorError, NvidiaMonitor, select_gpus


LOG = logging.getLogger("singularity_placeholder")


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    """Publish complete JSON without exposing partially written request/status files."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(data, output, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(name, path)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} does not contain a JSON object")
    return value


def process_identity(pid: int) -> str | None:
    """Return a process start identity, None only for a verified absent process.

    Linux boot ID + /proc start ticks survive exec and avoid PID-reuse errors.
    macOS support is for CPU tests and control tooling; GPU jobs run on Linux.
    Unreadable identity is an error, never evidence that a reservation is stale.
    """
    if pid <= 0:
        raise ValueError("PID must be positive")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError as exc:
        raise RuntimeError(f"Cannot verify ownership of PID {pid}") from exc
    if sys.platform.startswith("linux"):
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            fields = stat[stat.rfind(")") + 2:].split()
            if fields[0] in {"Z", "X"}:
                return None
            start_ticks = fields[19]
            boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            return f"linux:{boot}:{start_ticks}"
        except FileNotFoundError:
            # Confirm exit instead of treating an unusual /proc mount as death.
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return None
            raise RuntimeError(f"PID {pid} exists but its /proc identity is inaccessible")
        except (OSError, IndexError) as exc:
            raise RuntimeError(f"Cannot read start identity for PID {pid}: {exc}") from exc
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart="], capture_output=True, text=True,
        timeout=5, check=False,
    )
    if result.returncode or not result.stdout.strip():
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return None
        raise RuntimeError(f"Cannot read start identity for PID {pid}")
    return f"ps:{result.stdout.strip()}"


def identity_record(pid: int) -> dict[str, Any]:
    start = process_identity(pid)
    if start is None:
        raise RuntimeError(f"PID {pid} exited before its identity could be recorded")
    return {"pid": pid, "start": start}


def identity_alive(record: dict[str, Any]) -> bool:
    pid, start = record.get("pid"), record.get("start")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0 or not isinstance(start, str) or not start:
        raise ValueError("Invalid process identity record")
    return process_identity(pid) == start


def group_alive(pgid: int) -> bool:
    """Conservatively detect live group members, excluding verified Linux zombies."""
    if pgid <= 0:
        raise ValueError("Process group ID must be positive")
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    if not sys.platform.startswith("linux"):
        return True
    saw_group = False
    unreadable = False
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return True
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        try:
            stat = (entry / "stat").read_text()
            fields = stat[stat.rfind(")") + 2:].split()
            member_group = int(fields[2])
            if member_group == pgid:
                saw_group = True
                if fields[0] not in {"Z", "X"}:
                    return True
        except FileNotFoundError:
            continue  # Process disappeared during the snapshot.
        except (OSError, ValueError, IndexError):
            unreadable = True
    if saw_group and not unreadable:
        return False  # Every verified member is a zombie/dead; no CUDA context remains.
    try:
        os.killpg(pgid, 0)
        return True  # Missing or unreadable proc entries are not proof of death.
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def signal_group(pgid: int, signum: int) -> None:
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        pass


def ensure_local_control_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.stat().st_uid != os.getuid():
        raise RuntimeError("The control directory must be owned by the current user.")
    # Requests, flock, and atomic rename belong on local disk, never Blob/NFS.
    if sys.platform.startswith("linux"):
        try:
            candidates: list[tuple[int, str]] = []
            for line in Path("/proc/self/mountinfo").read_text().splitlines():
                left, right = line.split(" - ", 1)
                mountpoint = left.split()[4].replace("\\040", " ").replace("\\134", "\\")
                if path == Path(mountpoint) or Path(mountpoint) in path.parents:
                    candidates.append((len(mountpoint), right.split()[0]))
            fstype = max(candidates)[1] if candidates else "unknown"
        except (OSError, ValueError, IndexError) as exc:
            raise RuntimeError("Cannot verify that the control directory is on local disk.") from exc
        if fstype == "unknown" or fstype.startswith(("fuse", "nfs")) or fstype in {"cifs", "smbfs", "9p", "ceph", "lustre", "afs"}:
            raise RuntimeError(f"Control directory is on {fstype}; use node-local /tmp instead of Blob/shared storage.")
    os.chmod(path, 0o700)
    for name in ("requests", "acks"):
        directory = path / name
        directory.mkdir(exist_ok=True, mode=0o700)
        os.chmod(directory, 0o700)
    return path


class DaemonLock:
    def __init__(self, control: Path):
        self.path = control / "daemon.lock"
        self.fd: int | None = None

    def __enter__(self) -> "DaemonLock":
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(self.fd)
            self.fd = None
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise RuntimeError("Another supervisor already owns this control directory.") from exc
            raise
        return self

    def __exit__(self, *args: Any) -> None:
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None


class Reservations:
    def __init__(self, control: Path):
        self.control = control

    def _paths(self, request_id: str) -> tuple[Path, Path, Path]:
        return (
            self.control / "requests" / f"{request_id}.json",
            self.control / "acks" / f"{request_id}.json",
            self.control / "requests" / f"{request_id}.go",
        )

    def remove(self, request_id: str) -> None:
        for path in self._paths(request_id):
            path.unlink(missing_ok=True)

    @staticmethod
    def _stale(request: dict[str, Any]) -> bool:
        if identity_alive(request["owner"]):
            return False
        child = request.get("child")
        if child is not None:
            if identity_alive(child):
                return False
            # Retain the reservation while any descendants in its process group
            # survive, including the leader exiting before its GPU children.
            if group_alive(child["pid"]):
                return False
        return True

    def active(self) -> tuple[list[dict[str, Any]], list[str]]:
        active: list[dict[str, Any]] = []
        errors: list[str] = []
        for path in sorted((self.control / "requests").glob("*.json")):
            try:
                request = read_json(path)
                if request.get("version") != 1 or request.get("id") != path.stem:
                    raise ValueError("invalid reservation version or ID")
                if self._stale(request):
                    # Read again: wrapper may just have added its gated child.
                    latest = read_json(path)
                    if latest != request or not self._stale(latest):
                        active.append(latest)
                        continue
                    self.remove(path.stem)
                    LOG.info("Removed verified-dead reservation %s", path.stem)
                else:
                    active.append(request)
            except FileNotFoundError:
                continue
            except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
                # Corrupt or unverifiable reservations block placeholders.
                errors.append(f"{path.name}: {exc}")
        return active, errors

    def acknowledge(self, active: Sequence[dict[str, Any]], daemon: dict[str, Any]) -> None:
        for request in active:
            request_path, ack_path, _ = self._paths(request["id"])
            if request_path.exists():
                atomic_json(ack_path, {
                    "version": 1, "id": request["id"], "daemon": daemon,
                    "gpus": request.get("gpus"),
                    "workers_stopped": True, "acknowledged_at": time.time(),
                })


@dataclass
class Worker:
    gpu: str
    process: subprocess.Popen[str]
    aliases: set[int]
    reader: threading.Thread


class WorkerPool:
    def __init__(self, stop_seconds: float = 10.0):
        self.workers: dict[str, Worker] = {}
        self.stop_seconds = stop_seconds

    def own_pids(self, gpu: str) -> set[int]:
        worker = self.workers.get(gpu)
        if worker is None or worker.process.poll() is not None:
            return set()
        # Each set contains one verified NVML/host PID, never mixed namespace IDs.
        return set(worker.aliases)

    def exited(self) -> list[tuple[str, int]]:
        return [(gpu, code) for gpu, worker in self.workers.items() if (code := worker.process.poll()) is not None]

    @staticmethod
    def _verify_pid_namespace() -> None:
        if not sys.platform.startswith("linux"):
            return  # CPU-only control tests also run on macOS.
        try:
            current = os.readlink("/proc/self/ns/pid")
            proc_root = os.readlink("/proc/1/ns/pid")
        except OSError as exc:
            raise RuntimeError("Cannot verify the PID namespace used for NVIDIA process ownership") from exc
        # Linux reserves this nsfs inode for init_pid_ns (PROC_PID_INIT_INO).
        # NSpid is relative to the namespace mounting procfs, so its first value
        # alone is NOT evidence of a host PID when procfs is container-local.
        if current != "pid:[4026531836]" or proc_root != current:
            raise RuntimeError(
                "Cannot safely map nvidia-smi host PIDs to this isolated PID namespace. "
                "No placeholder was launched. Use a job with host PID visibility and "
                "an aligned host /proc mount; namespace-local NSpid aliases are insufficient."
            )

    @staticmethod
    def _pid_aliases(pid: int) -> set[int]:
        WorkerPool._verify_pid_namespace()
        if sys.platform.startswith("linux"):
            try:
                for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                    if line.startswith("NSpid:"):
                        pids = [int(token) for token in line.split()[1:]]
                        if not pids or pids[0] != pid:
                            raise RuntimeError("Worker PID does not match the verified host procfs namespace")
                        return {pids[0]}
            except (OSError, ValueError) as exc:
                raise RuntimeError("Cannot verify the worker's NVIDIA host PID") from exc
        return {pid}

    def start(self, gpus: Sequence[str]) -> None:
        requested = tuple(gpus)
        if len(set(requested)) != len(requested) or any(gpu in self.workers for gpu in requested):
            raise RuntimeError("Cannot start a duplicate worker on a GPU")
        if not requested:
            return
        self._verify_pid_namespace()
        added: list[str] = []
        try:
            for gpu in requested:
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu
                env["PYTHONUNBUFFERED"] = "1"
                for name in ("VSCODE_CLI_ACCESS_TOKEN", "VSCODE_CLI_REFRESH_TOKEN"):
                    env.pop(name, None)
                process = subprocess.Popen(
                    [sys.executable, "-m", "singularity_placeholder.worker", "--gpu", gpu],
                    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, start_new_session=True,
                )

                def consume(output: Any = process.stdout, label: str = gpu) -> None:
                    try:
                        for line in output:
                            LOG.info("worker %s: %s", label, line.rstrip()[:8192])
                    finally:
                        output.close()

                reader = threading.Thread(target=consume, name=f"log-{gpu}", daemon=True)
                worker = Worker(gpu, process, set(), reader)
                self.workers[gpu] = worker
                added.append(gpu)
                reader.start()
                worker.aliases = self._pid_aliases(process.pid)
                LOG.info("Started worker gpu=%s pid=%s", gpu, process.pid)
        except BaseException:
            self.stop(added)
            raise

    def start_all(self, gpus: Sequence[str]) -> None:
        """Backward-compatible spelling; start() also supports free subsets."""
        self.start(gpus)

    def stop(self, gpus: Sequence[str]) -> bool:
        workers = [self.workers[gpu] for gpu in dict.fromkeys(gpus) if gpu in self.workers]
        for worker in workers:
            # A departed leader can leave descendants (and GPU contexts) alive.
            worker.process.poll()
            if group_alive(worker.process.pid):
                signal_group(worker.process.pid, signal.SIGTERM)

        def pending() -> list[Worker]:
            result: list[Worker] = []
            for worker in workers:
                code = worker.process.poll()  # Reap owned leaders before zombie checks.
                if code is None or group_alive(worker.process.pid):
                    result.append(worker)
            return result

        deadline = time.monotonic() + self.stop_seconds
        while pending() and time.monotonic() < deadline:
            time.sleep(0.05)
        remaining = pending()
        for worker in remaining:
            signal_group(worker.process.pid, signal.SIGKILL)
        if remaining:
            deadline = time.monotonic() + 5.0
            while pending() and time.monotonic() < deadline:
                time.sleep(0.05)
        for worker in workers:
            if worker.process.poll() is None or group_alive(worker.process.pid):
                LOG.error("Worker group %s has not fully stopped; withholding ACK", worker.process.pid)
                continue
            self.workers.pop(worker.gpu, None)
            worker.reader.join(timeout=0.2)
            LOG.info("Stopped worker gpu=%s pid=%s", worker.gpu, worker.process.pid)
        return all(worker.gpu not in self.workers for worker in workers)

    def stop_all(self) -> bool:
        return self.stop(tuple(self.workers))

    def public(self) -> dict[str, int]:
        return {gpu: worker.process.pid for gpu, worker in self.workers.items()}


@dataclass
class Config:
    control_dir: Path
    state_dir: Path
    gpu_count: int = 8
    gpus: str | None = None
    idle_seconds: float = 30.0
    poll_seconds: float = 2.0
    max_utilization: int = 5
    max_memory_mib: int = 256
    worker_stop_seconds: float = 10.0
    worker_retry_seconds: float = 60.0


class Supervisor:
    def __init__(self, config: Config, *, monitor: Any = None, pool: Any = None, environ: Any = None):
        self.config = config
        self.control = ensure_local_control_dir(config.control_dir)
        self.monitor = monitor or NvidiaMonitor()
        self.pool = pool or WorkerPool(config.worker_stop_seconds)
        self.selected = select_gpus(
            self.monitor.inventory(), count=config.gpu_count, selectors=config.gpus, environ=environ,
        )
        self.reservations = Reservations(self.control)
        self.daemon = {**identity_record(os.getpid()), "session": uuid.uuid4().hex}
        self.idle_since: dict[str, float] = {}
        self.retry_after: dict[str, float] = {}
        self.stop_event = threading.Event()
        self.previous_state: str | None = None

    def _publish(self, state: str, reason: str, **details: Any) -> dict[str, Any]:
        status = {
            "version": 1, "updated_at": time.time(), "daemon": self.daemon,
            "state": state, "reason": reason, "gpus": list(self.selected),
            "workers": self.pool.public(), **details,
        }
        atomic_json(self.control / "status.json", status)
        summary = state + str(sorted(status["workers"]))
        if summary != self.previous_state:
            LOG.info("State %s (%s workers): %s", state, len(status["workers"]), reason)
            self.previous_state = summary
        return status

    def _request_gpus(self, request: dict[str, Any]) -> set[str]:
        values = request.get("gpus", list(self.selected))
        if not isinstance(values, list) or not values or any(not isinstance(gpu, str) for gpu in values):
            raise ValueError("Reservation must contain a non-empty GPU UUID list")
        if len(values) != len(set(values)) or not set(values) <= set(self.selected):
            raise ValueError("Reservation contains duplicate or unallocated GPU UUIDs")
        return set(values)

    def _acknowledge_stopped(self, active: Sequence[dict[str, Any]]) -> None:
        eligible = [r for r in active if not self._request_gpus(r).intersection(self.pool.workers)]
        self.reservations.acknowledge(eligible, self.daemon)

    def step(self, now: float | None = None) -> dict[str, Any]:
        now = time.monotonic() if now is None else now
        active, request_errors = self.reservations.active()
        reserved: set[str] = set()
        valid_active: list[dict[str, Any]] = []
        for request in active:
            try:
                reserved.update(self._request_gpus(request))
                valid_active.append(request)
            except ValueError as exc:
                request_errors.append(f"{request['id']}: {exc}")
        paused = (self.control / "paused.json").exists()
        snapshot, monitor_error = None, None
        foreign: dict[str, set[int]] = {gpu: set() for gpu in self.selected}
        try:
            snapshot = self.monitor.sample(self.selected)
            for gpu, pid in snapshot.processes:
                if pid not in self.pool.own_pids(gpu):
                    foreign[gpu].add(pid)
        except (MonitorError, OSError) as exc:
            monitor_error = str(exc)
        global_block: tuple[str, str] | None = None
        if self.stop_event.is_set():
            global_block = ("stopping", "Supervisor shutdown requested")
        elif monitor_error:
            global_block = ("monitor_error", monitor_error)
        elif request_errors:
            global_block = ("reservation_error", "; ".join(request_errors))
        elif paused:
            global_block = ("paused", "Manual global pause is active")
        if global_block:
            self.idle_since.clear()
            stopped = self.pool.stop_all()
            self._acknowledge_stopped(valid_active)
            if not stopped:
                global_block = ("stopping", "Owned workers have not fully exited; affected workload ACKs are withheld")
            return self._publish(
                *global_block, reservations=[r["id"] for r in active],
                external_pids=sorted({pid for pids in foreign.values() for pid in pids}),
                monitor_error=monitor_error, reservation_errors=request_errors, paused=paused,
            )
        assert snapshot is not None
        readings = {gpu.uuid: gpu for gpu in snapshot.gpus}
        exited = dict(self.pool.exited())
        gpu_states: dict[str, dict[str, Any]] = {}
        stop_set: set[str] = set()
        start_set: list[str] = []
        for gpu in self.selected:
            reading = readings[gpu]
            state, reason = "running", "Owned placeholder is running"
            if gpu in reserved:
                state, reason = "reserved", "Reserved by a cooperative workload"
            elif foreign[gpu]:
                state, reason = "external_workload", "External CUDA PID detected on this GPU"
                if gpu in self.pool.workers:
                    # An unknown host/container PID may be our own worker under
                    # an inaccessible PID namespace. Never classify it by guess.
                    self.retry_after[gpu] = now + self.config.worker_retry_seconds
                    reason += "; unknown PID identity is treated as foreign (including namespace mismatch)"
            elif gpu in exited:
                self.retry_after[gpu] = now + self.config.worker_retry_seconds
                state, reason = "worker_error", f"Owned worker exited {exited[gpu]}; retry after backoff"
            elif now < self.retry_after.get(gpu, 0):
                state, reason = "worker_backoff", "Waiting before retrying this GPU"
            elif gpu not in self.pool.workers:
                if reading.utilization > self.config.max_utilization or reading.memory_mib > self.config.max_memory_mib:
                    state, reason = "gpu_busy", "Utilization or memory exceeds the idle thresholds"
                else:
                    self.idle_since.setdefault(gpu, now)
                    idle_for = now - self.idle_since[gpu]
                    if idle_for < self.config.idle_seconds:
                        state, reason = "idle_grace", "Waiting for this GPU's idle grace period"
                    else:
                        state, reason = "start_pending", "This GPU passed the idle checks"
                        start_set.append(gpu)
            if state not in {"running", "idle_grace", "start_pending"}:
                self.idle_since.pop(gpu, None)
                stop_set.add(gpu)
            gpu_states[gpu] = {
                "state": state, "reason": reason, "external_pids": sorted(foreign[gpu]),
                "idle_for_seconds": max(0.0, now - self.idle_since.get(gpu, now)),
            }
        self.pool.stop(stop_set)
        for gpu in stop_set.intersection(self.pool.workers):
            gpu_states[gpu] = {"state": "stopping", "reason": "Owned worker has not fully exited"}
        self._acknowledge_stopped(valid_active)
        if start_set:
            # The wrapper cannot launch before its reservation is acknowledged;
            # a request racing this last check is handled on the next poll.
            new_active, new_errors = self.reservations.active()
            newly_reserved: set[str] = set()
            for request in new_active:
                try:
                    newly_reserved.update(self._request_gpus(request))
                except ValueError:
                    new_errors.append("Invalid GPU selection in a new reservation")
            if new_errors or (self.control / "paused.json").exists():
                start_set = []
            else:
                start_set = [gpu for gpu in start_set if gpu not in newly_reserved]
            for gpu in start_set:
                try:
                    self.pool.start([gpu])
                    gpu_states[gpu] = {"state": "running", "reason": "Started an owned worker on this GPU"}
                except (OSError, RuntimeError) as exc:
                    LOG.error("Worker startup failed gpu=%s: %s", gpu, exc)
                    self.retry_after[gpu] = now + self.config.worker_retry_seconds
                    self.idle_since.pop(gpu, None)
                    gpu_states[gpu] = {"state": "worker_error", "reason": f"Worker startup failed: {exc}"}
        states = {item["state"] for item in gpu_states.values()}
        priority = ("stopping", "reserved", "external_workload", "worker_error", "worker_backoff", "gpu_busy", "idle_grace", "start_pending", "running")
        aggregate = next(state for state in priority if state in states)
        return self._publish(
            aggregate, "Each GPU independently yields to real work and resumes after its idle grace period",
            gpu_states=gpu_states, reservations=[r["id"] for r in active],
            external_pids=sorted({pid for pids in foreign.values() for pid in pids}), paused=False,
        )

    def run(self) -> int:
        old_handlers: dict[int, Any] = {}
        with DaemonLock(self.control):
            for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                old_handlers[signum] = signal.signal(signum, lambda *_: self.stop_event.set())
            try:
                LOG.info("Supervising GPU UUIDs independently: %s", ", ".join(self.selected))
                while not self.stop_event.is_set():
                    self.step()
                    self.stop_event.wait(self.config.poll_seconds)
                return 0
            finally:
                stopped = self.pool.stop_all()
                self._publish("stopped" if stopped else "stop_failed", "Supervisor exited")
                for signum, handler in old_handlers.items():
                    signal.signal(signum, handler)


def status(control: Path) -> dict[str, Any]:
    try:
        value = read_json(control / "status.json")
    except FileNotFoundError:
        return {"state": "not_started", "daemon_alive": False, "workers": {}}
    try:
        value["daemon_alive"] = identity_alive(value["daemon"])
    except (KeyError, OSError, RuntimeError, ValueError, subprocess.SubprocessError):
        value["daemon_alive"] = False
        value["identity_verification_failed"] = True
    value["heartbeat_age_seconds"] = max(0.0, time.time() - value.get("updated_at", 0))
    if value["state"] in {"stopped", "stop_failed"}:
        value["daemon_alive"] = False
    return value


def pause(control: Path, timeout: float = 120.0) -> dict[str, Any]:
    control = ensure_local_control_dir(control)
    requested_at = time.time()
    atomic_json(control / "paused.json", {"requested_at": requested_at})
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = status(control)
        if current.get("daemon_alive") and current.get("paused") and not current.get("workers") and current.get("updated_at", 0) >= requested_at:
            return current
        time.sleep(0.1)
    raise RuntimeError("Pause remains requested, but no live supervisor confirmed that all workers stopped.")


def resume(control: Path) -> dict[str, Any]:
    control = ensure_local_control_dir(control)
    (control / "paused.json").unlink(missing_ok=True)
    return {"state": "resume_requested", "message": "Idle checks and grace period still apply."}


def reserved_child(control: Path, request_id: str, command: Sequence[str]) -> int:
    """Wait until our identity is durable before execing the foreground command."""
    request_path = control / "requests" / f"{request_id}.json"
    gate = control / "requests" / f"{request_id}.go"
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            request = read_json(request_path)
            if not identity_alive(request["owner"]):
                return 125
            if gate.exists() and request.get("child") == identity_record(os.getpid()):
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = ",".join(request["gpus"])
                os.execvpe(command[0], list(command), env)
        except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError):
            return 125
        time.sleep(0.05)
    return 125


def _cleanup_command_group(process: subprocess.Popen[Any], timeout: float) -> bool:
    pgid = process.pid
    if process.poll() is None or group_alive(pgid):
        signal_group(pgid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        process.poll()
        if not group_alive(pgid):
            process.wait()
            return True
        time.sleep(0.05)
    signal_group(pgid, signal.SIGKILL)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        return False
    # Zombie descendants may remain until the container init reaps them. Keep
    # the reservation in that case; the supervisor will GC only after death.
    return not group_alive(pgid)


def resolve_run_gpus(selectors: str | None, managed: Sequence[str]) -> list[str]:
    """CLI ordinals refer to the stable managed list shown by status."""
    if selectors is None:
        return list(managed)
    result: list[str] = []
    for token in (part.strip() for part in selectors.split(",")):
        if token.isdecimal():
            index = int(token)
            if index >= len(managed):
                raise ValueError(f"Managed GPU ordinal {index} is unavailable")
            result.append(managed[index])
        else:
            matches = [gpu for gpu in managed if token.startswith("GPU-") and gpu.startswith(token)]
            if len(matches) != 1:
                raise ValueError(f"GPU selector {token!r} is unavailable or ambiguous")
            result.append(matches[0])
    if not result or len(result) != len(set(result)):
        raise ValueError("--gpus must identify a non-empty set without duplicates")
    return result


def run_reserved(
    control: Path, command: Sequence[str], *, ack_timeout: float = 120.0,
    stop_seconds: float = 10.0, gpus: str | None = None,
) -> int:
    if not command:
        raise ValueError("run requires a command after --")
    control = ensure_local_control_dir(control)
    current = status(control)
    if not current.get("daemon_alive") or not current.get("gpus"):
        raise RuntimeError("No live supervisor exposes a managed allocation; workload was not launched.")
    selected = resolve_run_gpus(gpus, current["gpus"])
    request_id = uuid.uuid4().hex
    requests = Reservations(control)
    request_path, ack_path, gate = requests._paths(request_id)
    request = {"version": 1, "id": request_id, "owner": identity_record(os.getpid()), "created_at": time.time(), "gpus": selected}
    child: subprocess.Popen[Any] | None = None
    received_signal: int | None = None
    old_handlers: dict[int, Any] = {}

    def forward(signum: int, _frame: Any) -> None:
        nonlocal received_signal
        received_signal = signum
        if child is not None:
            signal_group(child.pid, signum)

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        old_handlers[signum] = signal.signal(signum, forward)
    atomic_json(request_path, request)
    clean = True
    try:
        deadline = time.monotonic() + ack_timeout
        while True:
            if received_signal is not None:
                return 128 + received_signal
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for a live supervisor ACK; workload was not launched.")
            try:
                ack = read_json(ack_path)
                current = status(control)
                if (
                    ack.get("id") == request_id and ack.get("workers_stopped") is True
                    and ack.get("gpus") == selected
                    and current.get("daemon_alive")
                    and current.get("daemon") == ack.get("daemon")
                    and identity_alive(ack["daemon"])
                ):
                    break
            except (FileNotFoundError, KeyError, ValueError, OSError, RuntimeError):
                pass
            time.sleep(0.05)
        if received_signal is not None:
            return 128 + received_signal
        child = subprocess.Popen(
            [sys.executable, "-m", "singularity_placeholder", "_reserved-child", "--control-dir", str(control), "--request-id", request_id, "--", *command],
            start_new_session=True,
        )
        request["child"] = identity_record(child.pid)
        atomic_json(request_path, request)
        # Publishing the gate is the only operation that allows the command to
        # execute. If the wrapper dies earlier, the helper exits without exec.
        atomic_json(gate, {"ready": True})
        while child.poll() is None:
            if received_signal is not None:
                break
            time.sleep(0.1)
        if received_signal is not None:
            return 128 + received_signal
        code = child.returncode
        return 128 - code if code is not None and code < 0 else int(code or 0)
    finally:
        if child is not None:
            clean = _cleanup_command_group(child, stop_seconds)
        if clean:
            requests.remove(request_id)
        else:
            LOG.warning("Retaining reservation %s: command process group has not fully disappeared", request_id)
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def configure_logging(state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    LOG.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stdout = logging.StreamHandler()
    stdout.setFormatter(formatter)
    rotating = logging.handlers.RotatingFileHandler(
        state_dir / "placeholder.log", maxBytes=5 * 1024 * 1024, backupCount=3,
        encoding="utf-8",
    )
    rotating.setFormatter(formatter)
    LOG.handlers[:] = [stdout, rotating]

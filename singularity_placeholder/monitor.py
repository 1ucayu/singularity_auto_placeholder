"""Conservative, dependency-free NVIDIA GPU monitoring.

Only GPU UUIDs cross the worker boundary. Partial numeric visibility masks are
ambiguous between CUDA, NVML, and container ordinals, so they are rejected.
"""

from __future__ import annotations

import csv
import os
import subprocess
from dataclasses import dataclass
from typing import Mapping, Sequence


class MonitorError(RuntimeError):
    """The GPU allocation or current activity cannot safely be established."""


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    utilization: int
    memory_mib: int
    mig: str = "Disabled"


@dataclass(frozen=True)
class Snapshot:
    gpus: tuple[GPU, ...]
    processes: tuple[tuple[str, int], ...]

    def foreign_pids(self, own_pids: set[int]) -> set[int]:
        return {pid for _, pid in self.processes if pid not in own_pids}

    def idle(self, max_utilization: int, max_memory_mib: int) -> bool:
        return all(
            gpu.utilization <= max_utilization and gpu.memory_mib <= max_memory_mib
            for gpu in self.gpus
        )


def _uuid_token(token: str, inventory: Sequence[GPU]) -> str:
    if token.startswith("MIG-"):
        raise MonitorError("MIG allocations are not supported; use a full-GPU job.")
    matches = [gpu.uuid for gpu in inventory if gpu.uuid == token or gpu.uuid.startswith(token)]
    if not token.startswith("GPU-") or len(matches) != 1:
        raise MonitorError(f"GPU selector {token!r} does not identify exactly one visible GPU UUID.")
    return matches[0]


def _environment_mask(name: str, raw: str | None, inventory: Sequence[GPU]) -> set[str]:
    all_uuids = {gpu.uuid for gpu in inventory}
    if raw is None or raw.strip().lower() == "all":
        return all_uuids
    if raw.strip().lower() in {"", "none", "void", "-1"}:
        return set()
    tokens = [token.strip() for token in raw.split(",")]
    if any(not token for token in tokens) or len(tokens) != len(set(tokens)):
        raise MonitorError(f"Invalid {name} mask: empty or duplicate selectors.")
    if all(token.isdecimal() for token in tokens):
        # A complete numeric mask selects the entire inventory regardless of
        # CUDA ordering. A subset cannot be inferred safely from NVML indices.
        numeric = {int(token) for token in tokens}
        if numeric == {gpu.index for gpu in inventory} and len(numeric) == len(inventory):
            return all_uuids
        raise MonitorError(
            f"Partial numeric {name}={raw!r} is ambiguous in a container. "
            "Set the mask to the assigned full GPU UUIDs, or expose the complete allocation."
        )
    if any(token.isdecimal() for token in tokens):
        raise MonitorError(f"Mixed numeric and UUID selectors in {name} are not supported.")
    return {_uuid_token(token, inventory) for token in tokens}


def select_gpus(
    inventory: Sequence[GPU],
    *,
    count: int = 8,
    selectors: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Intersect explicit selection with both runtime allocation masks."""
    if count < 1 or not inventory:
        raise MonitorError("A positive GPU count and a non-empty NVIDIA inventory are required.")
    if len({gpu.uuid for gpu in inventory}) != len(inventory):
        raise MonitorError("NVIDIA returned duplicate GPU UUIDs.")
    if len({gpu.index for gpu in inventory}) != len(inventory):
        raise MonitorError("NVIDIA returned duplicate GPU indices.")
    env = os.environ if environ is None else environ
    allowed = {gpu.uuid for gpu in inventory}
    for name in ("NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        allowed &= _environment_mask(name, env.get(name), inventory)
    if selectors is not None:
        chosen: list[str] = []
        for token in (part.strip() for part in selectors.split(",")):
            if token.isdecimal():
                matches = [gpu.uuid for gpu in inventory if gpu.index == int(token)]
                if len(matches) != 1:
                    raise MonitorError(f"Explicit nvidia-smi GPU index {token!r} is unavailable.")
                chosen.append(matches[0])
            else:
                chosen.append(_uuid_token(token, inventory))
        if len(chosen) != len(set(chosen)):
            raise MonitorError("--gpus contains duplicate GPUs.")
        if not set(chosen) <= allowed:
            raise MonitorError("--gpus requests a GPU outside the environment's visible allocation.")
        allowed = set(chosen)
    selected = tuple(gpu.uuid for gpu in inventory if gpu.uuid in allowed)
    if len(selected) != count:
        raise MonitorError(f"Expected exactly {count} allocated GPUs; safely resolved {len(selected)}.")
    for gpu in inventory:
        if gpu.uuid in allowed and gpu.mig.lower() not in {"disabled", "n/a", "[n/a]", "not supported"}:
            raise MonitorError(f"GPU {gpu.uuid} has MIG mode {gpu.mig!r}; MIG is unsupported.")
    return selected


class NvidiaMonitor:
    def __init__(self, executable: str = "nvidia-smi", timeout: float = 10.0):
        self.executable = executable
        self.timeout = timeout

    def _query(self, fields: str, kind: str) -> list[list[str]]:
        try:
            result = subprocess.run(
                [self.executable, f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=self.timeout, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise MonitorError(f"nvidia-smi {kind} query failed: {exc}") from exc
        if result.returncode:
            raise MonitorError(
                f"nvidia-smi {kind} exited {result.returncode}: {result.stderr.strip()[:500]}"
            )
        return [[value.strip() for value in row] for row in csv.reader(result.stdout.splitlines()) if row]

    def inventory(self) -> tuple[GPU, ...]:
        rows = self._query("index,uuid,utilization.gpu,memory.used,mig.mode.current", "gpu")
        inventory: list[GPU] = []
        try:
            for row in rows:
                if len(row) != 5:
                    raise ValueError("expected five GPU fields")
                index, uuid, utilization, memory, mig = row
                gpu = GPU(int(index), uuid, int(utilization), int(memory), mig)
                if not uuid.startswith("GPU-") or not 0 <= gpu.utilization <= 100 or gpu.memory_mib < 0:
                    raise ValueError("invalid GPU UUID, utilization, or memory reading")
                inventory.append(gpu)
        except ValueError as exc:
            raise MonitorError(f"Cannot parse NVIDIA GPU inventory: {exc}") from exc
        if not inventory:
            raise MonitorError("NVIDIA reported no GPUs.")
        return tuple(inventory)

    def sample(self, selected: Sequence[str]) -> Snapshot:
        inventory = self.inventory()
        selected_set = set(selected)
        gpus = tuple(gpu for gpu in inventory if gpu.uuid in selected_set)
        if len(gpus) != len(selected) or {gpu.uuid for gpu in gpus} != selected_set:
            raise MonitorError("The selected GPU allocation changed or became unavailable.")
        for gpu in gpus:
            if gpu.mig.lower() not in {"disabled", "n/a", "[n/a]", "not supported"}:
                raise MonitorError("MIG mode changed; stopping workers.")
        processes: list[tuple[str, int]] = []
        rows = self._query("gpu_uuid,pid", "compute-apps")
        try:
            for row in rows:
                if len(row) != 2:
                    raise ValueError("expected GPU UUID and PID")
                uuid, raw_pid = row
                pid = int(raw_pid)
                if not uuid.startswith("GPU-") or pid <= 0:
                    raise ValueError("invalid GPU UUID or PID")
                if uuid in selected_set:
                    processes.append((uuid, pid))
        except ValueError as exc:
            raise MonitorError(f"Cannot parse NVIDIA compute-process inventory: {exc}") from exc
        return Snapshot(gpus, tuple(processes))

"""Discover the actual CUDA allocation in a short-lived isolated process.

NVML and ``nvidia-smi`` can expose more GPUs than CUDA can use. Container startup
hints such as NVIDIA_VISIBLE_DEVICES may also be stale inside an existing
container. Ask the CUDA driver without changing CUDA_VISIBLE_DEVICES, and return
full UUIDs so CUDA ordinal reordering cannot select another physical GPU.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Mapping
import uuid


class CudaProbeError(RuntimeError):
    """The actual CUDA-visible allocation could not be established."""


_UUID_PATTERN = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")


class _CUuuid(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_ubyte * 16)]


def _driver_uuids() -> tuple[str, ...]:
    """Run only in the child process; CUDA initialization stays out of the parent."""
    try:
        driver = ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        raise CudaProbeError("Cannot load libcuda.so.1; the NVIDIA CUDA driver must be exposed to this container.") from exc

    def function(name, argtypes):
        try:
            result = getattr(driver, name)
        except AttributeError as exc:
            raise CudaProbeError(f"The CUDA driver does not expose {name}.") from exc
        result.argtypes = argtypes
        result.restype = ctypes.c_int
        return result

    def checked(name, result):
        if result != 0:
            raise CudaProbeError(f"{name} failed with CUDA driver error {result}.")

    init = function("cuInit", [ctypes.c_uint])
    status = init(0)
    if status == 100:  # CUDA_ERROR_NO_DEVICE, including an empty visibility mask.
        return ()
    checked("cuInit", status)
    count = ctypes.c_int()
    checked("cuDeviceGetCount", function("cuDeviceGetCount", [ctypes.POINTER(ctypes.c_int)])(ctypes.byref(count)))
    if not 0 <= count.value <= 1024:
        raise CudaProbeError("CUDA returned an invalid device count.")
    device_get = function("cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int])
    uuid_name = "cuDeviceGetUuid_v2" if hasattr(driver, "cuDeviceGetUuid_v2") else "cuDeviceGetUuid"
    uuid_get = function(uuid_name, [ctypes.POINTER(_CUuuid), ctypes.c_int])
    found: list[str] = []
    for ordinal in range(count.value):
        device = ctypes.c_int()
        checked("cuDeviceGet", device_get(ctypes.byref(device), ordinal))
        raw = _CUuuid()
        checked(uuid_name, uuid_get(ctypes.byref(raw), device))
        found.append("GPU-" + str(uuid.UUID(bytes=bytes(raw.bytes))))
    if len(found) != len(set(found)):
        raise CudaProbeError("CUDA returned duplicate GPU UUIDs; this allocation is unsupported.")
    return tuple(found)


def discover_cuda_uuids(
    *, timeout: float = 20.0, environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return UUIDs in CUDA ordinal order without changing visibility settings.

    An empty tuple is a successful probe that found no usable devices. Driver
    failures, a hung driver, and invalid child output raise CudaProbeError.
    Only this function's structured output is consumed; arbitrary stderr or
    process environment values are never included in errors.
    """
    if timeout <= 0:
        raise ValueError("CUDA probe timeout must be positive.")
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            env=None if environ is None else dict(environ),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise CudaProbeError(f"CUDA allocation probe timed out after {timeout:g} seconds.") from exc
    except OSError as exc:
        raise CudaProbeError("Cannot launch the isolated CUDA allocation probe.") from exc
    try:
        payload = json.loads(result.stdout)
        if not isinstance(payload, dict):
            raise ValueError("expected object")
        if result.returncode:
            # Errors are emitted only by the fixed child implementation below;
            # do not echo captured stderr or unexpected payloads into job logs.
            error = payload.get("error")
            if not isinstance(error, str) or not error or len(error) > 500:
                raise ValueError("invalid error")
            raise CudaProbeError(f"CUDA allocation probe failed: {error}")
        values = payload["uuids"]
        if not isinstance(values, list) or len(values) > 1024:
            raise ValueError("invalid UUID list")
        if any(not isinstance(value, str) or not _UUID_PATTERN.fullmatch(value) for value in values):
            raise ValueError("invalid UUID")
        if len(values) != len(set(values)):
            raise ValueError("duplicate UUIDs")
    except (KeyError, TypeError, ValueError) as exc:
        raise CudaProbeError(
            f"CUDA allocation probe returned invalid output (exit code {result.returncode})."
        ) from exc
    return tuple(values)


def _main() -> int:
    try:
        values = _driver_uuids()
    except CudaProbeError as exc:
        print(json.dumps({"error": str(exc)}), flush=True)
        return 1
    print(json.dumps({"uuids": values}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

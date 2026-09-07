"""Exec a tunnel CLI that receives SIGTERM if its Linux helper disappears."""

from __future__ import annotations

import argparse
import ctypes
import os
import signal
import sys


def arm_parent_death() -> None:
    """Use the existing Linux kernel API; other platforms need no extra setup."""
    if not sys.platform.startswith("linux"):
        return
    try:
        library = ctypes.CDLL(None, use_errno=True)
        prctl = library.prctl
        prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
        prctl.restype = ctypes.c_int
        if prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
    except (AttributeError, OSError) as exc:
        # Connectivity remains available if a container disallows this optional
        # cleanup mechanism. The ordinary helper shutdown path still applies.
        print(f"[tunnel] Parent-death cleanup unavailable: {exc}; continuing normally", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pid", required=True, type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.parent_pid < 1 or not command:
        parser.error("a positive parent PID and a command are required")
    arm_parent_death()
    # The helper can disappear between Popen and prctl. Do not launch another
    # CLI after reparenting has already occurred.
    if os.getppid() != args.parent_pid:
        return 0
    os.execvp(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

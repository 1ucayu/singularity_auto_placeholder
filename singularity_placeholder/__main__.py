"""python -m singularity_placeholder ..."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .supervisor import Config, Supervisor, configure_logging, pause, reserved_child, resume, run_reserved, status


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Cooperative, foreground GPU placeholder supervisor")
    commands = root.add_subparsers(dest="action", required=True)
    default_control = os.environ.get("PLACEHOLDER_CONTROL_DIR", "/tmp/singularity-placeholder")
    for name in ("supervise", "run", "pause", "resume", "status", "_reserved-child"):
        item = commands.add_parser(name)
        item.add_argument("--control-dir", type=Path, default=Path(default_control), help="Node-local control path; never a Blob/shared mount")
        if name == "supervise":
            item.add_argument("--state-dir", type=Path, default=None, help="Bounded log directory (defaults to control-dir/logs)")
            item.add_argument("--gpu-count", type=int, default=8, help="Expected GPU count; a mismatch is logged and usable GPUs still run")
            item.add_argument("--gpus", help="Comma-separated allocated GPU UUIDs or explicit nvidia-smi indices")
            item.add_argument("--idle-seconds", type=float, default=30.0)
            item.add_argument("--poll-seconds", type=float, default=2.0)
            item.add_argument("--max-utilization", type=int, default=5)
            item.add_argument("--max-memory-mib", type=int, default=256)
            item.add_argument("--worker-stop-seconds", type=float, default=10.0)
            item.add_argument("--worker-retry-seconds", type=float, default=60.0)
        elif name == "run":
            item.add_argument("--gpus", help="Comma-separated managed GPU ordinals or UUIDs; defaults to all. Sets the workload CUDA_VISIBLE_DEVICES.")
            item.add_argument("--ack-timeout", type=float, default=120.0)
            item.add_argument("--stop-seconds", type=float, default=10.0)
            item.add_argument("command", nargs=argparse.REMAINDER)
        elif name == "pause":
            item.add_argument("--timeout", type=float, default=120.0)
        elif name == "_reserved-child":
            item.add_argument("--request-id", required=True)
            item.add_argument("command", nargs=argparse.REMAINDER)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.action == "supervise":
            if args.gpu_count < 1 or args.idle_seconds < 0 or args.poll_seconds <= 0 or args.worker_stop_seconds <= 0 or args.worker_retry_seconds <= 0:
                raise ValueError("GPU count and polling/stop/retry durations must be positive; idle duration cannot be negative")
            if not 0 <= args.max_utilization <= 100 or args.max_memory_mib < 0:
                raise ValueError("Invalid idle utilization or memory thresholds")
            state_dir = args.state_dir or args.control_dir / "logs"
            configure_logging(state_dir)
            return Supervisor(Config(
                control_dir=args.control_dir, state_dir=state_dir, gpu_count=args.gpu_count,
                gpus=args.gpus, idle_seconds=args.idle_seconds, poll_seconds=args.poll_seconds,
                max_utilization=args.max_utilization, max_memory_mib=args.max_memory_mib,
                worker_stop_seconds=args.worker_stop_seconds, worker_retry_seconds=args.worker_retry_seconds,
            )).run()
        if args.action in {"run", "_reserved-child"}:
            command = args.command[1:] if args.command[:1] == ["--"] else args.command
            if not command:
                raise ValueError("Provide a foreground command after --")
            if args.action == "_reserved-child":
                return reserved_child(args.control_dir, args.request_id, command)
            if args.ack_timeout <= 0 or args.stop_seconds <= 0:
                raise ValueError("Timeout durations must be positive")
            return run_reserved(args.control_dir, command, ack_timeout=args.ack_timeout, stop_seconds=args.stop_seconds, gpus=args.gpus)
        if args.action == "pause":
            if args.timeout <= 0:
                raise ValueError("Timeout must be positive")
            result = pause(args.control_dir, args.timeout)
        elif args.action == "resume":
            result = resume(args.control_dir)
        else:
            result = status(args.control_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"singularity-placeholder: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

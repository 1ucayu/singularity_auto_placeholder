"""Small, interruptible CUDA GEMM workload. Owned and stopped by the supervisor."""

from __future__ import annotations

import argparse
import os
import signal
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, help="Physical GPU UUID, never a CUDA ordinal")
    parser.add_argument("--matrix-size", type=int, default=4096)
    args = parser.parse_args(argv)
    if not args.gpu.startswith("GPU-"):
        parser.error("--gpu must be a full NVIDIA GPU UUID")
    if not 256 <= args.matrix_size <= 8192:
        parser.error("--matrix-size must be between 256 and 8192")

    stopping = False

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    # Set before importing torch or initializing CUDA. Respect the container's
    # NVIDIA_VISIBLE_DEVICES boundary; only narrow CUDA visibility to this GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    try:
        import torch

        if stopping:
            return 0
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("worker requires exactly one visible CUDA GPU")
        torch.set_num_threads(1)
        with torch.inference_mode():
            size = args.matrix_size
            a = torch.randn((size, size), device="cuda:0", dtype=torch.float16)
            b = torch.randn_like(a)
            result = torch.empty_like(a)
            print(f"worker ready gpu={args.gpu} pid={os.getpid()} matrix={size}", flush=True)
            while not stopping:
                torch.mm(a, b, out=result)
                # Bound queued work and return to Python after each GEMM so a
                # signal can release the CUDA context promptly.
                torch.cuda.synchronize()
            del a, b, result
    except Exception as exc:
        print(f"worker failed gpu={args.gpu}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

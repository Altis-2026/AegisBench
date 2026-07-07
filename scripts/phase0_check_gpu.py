#!/usr/bin/env python3
"""Phase 0 checkpoint: prove the GPU actually works before anything else.

Prints torch/CUDA versions, the detected device, the compiled arch list,
whether the device's compute capability is covered, and times a real
matmul on the GPU. Exits nonzero if CUDA is unavailable or the smoke test
fails — do not proceed to later phases on a nonzero exit.
"""

import sys
import time


def main() -> int:
    try:
        import torch
    except ImportError:
        print("FAIL: torch is not installed. Run scripts/setup_env_sm120.sh")
        return 1

    print(f"torch version        : {torch.__version__}")
    print(f"built with CUDA      : {torch.version.cuda}")
    print(f"cuda available       : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available. On an RTX 5070 (Blackwell, "
              "sm_120) a stable torch wheel that predates sm_120 support "
              "will do exactly this — install the nightly cu128+ wheel "
              "(scripts/setup_env_sm120.sh).")
        return 1

    dev = torch.cuda.get_device_properties(0)
    cap = f"sm_{dev.major}{dev.minor}"
    arch_list = torch.cuda.get_arch_list()
    print(f"device               : {dev.name}")
    print(f"compute capability   : {cap}")
    print(f"total VRAM           : {dev.total_memory / 2**30:.1f} GiB")
    print(f"compiled arch list   : {arch_list}")
    if cap not in arch_list:
        print(f"WARNING: {cap} not in compiled arch list; kernels may run "
              "via PTX JIT or fail. Prefer a wheel listing "
              f"{cap} explicitly.")

    x = torch.randn(2048, 2048, device="cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        x = (x @ x).clamp(-1e3, 1e3)
    torch.cuda.synchronize()
    dt = time.time() - t0
    checksum = float(x.float().abs().mean())
    print(f"GPU matmul smoke test: 20x 2048^2 in {dt:.3f}s "
          f"(checksum {checksum:.4f})")
    if dt > 30:
        print("WARNING: suspiciously slow — check the GPU is not falling "
              "back to PTX JIT or the laptop is on battery power limits.")
    print("PHASE 0: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Sanity check: is the ROCm build of PyTorch installed and using the Radeon GPU?

Run with:  uv run python scripts/check_gpu.py
"""

import time

import torch


def main() -> None:
    print(f"torch          : {torch.__version__}")
    print(f"HIP (ROCm)     : {torch.version.hip}")
    # ROCm builds reuse the torch.cuda API names.
    available = torch.cuda.is_available()
    print(f"GPU available  : {available}")
    if not available:
        raise SystemExit("GPU not detected. Check the AMD driver and the ROCm wheel.")

    props = torch.cuda.get_device_properties(0)
    print(f"device         : {props.name} ({getattr(props, 'gcnArchName', '?')})")
    print(f"VRAM           : {props.total_memory / 1024**3:.1f} GB")

    # bf16 matmul: the dtype ColPali/ColQwen run in.
    n = 8192
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    torch.matmul(a, b)  # warm-up
    torch.cuda.synchronize()

    iters = 20
    start = time.perf_counter()
    for _ in range(iters):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tflops = 2 * n**3 * iters / elapsed / 1e12
    print(f"bf16 matmul    : {tflops:.1f} TFLOPS ({n}x{n}, {iters} iters)")

    # Cross-check a small result against the CPU.
    x = torch.randn(256, 256)
    gpu = (x.cuda() @ x.cuda()).cpu()
    max_err = (gpu - x @ x).abs().max().item()
    print(f"fp32 GPU vs CPU: max abs diff {max_err:.2e}")
    print("OK" if max_err < 1e-2 and torch.isfinite(c).all() else "MISMATCH")


if __name__ == "__main__":
    main()

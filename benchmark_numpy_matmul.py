#!/usr/bin/env python3
"""
Benchmark NumPy matmul (1024×1024 float32) for comparison with matmul-cpu 7.cpp.
Reports mean time (ms) and FLOP/s using the same FLOP count: 2*M*N*K.
"""

import time
import numpy as np

M, N, K = 1024, 1024, 1024
FLOPs_PER_MATMUL = 2.0 * M * N * K  # multiply-add per element
NUM_ITERATIONS = 20                  # run multiple times for stable mean
WARMUP = 3                            # warmup runs (not timed)

def main():
    rng = np.random.default_rng(42)
    A = rng.random((M, K), dtype=np.float32) * 2 - 1
    B = rng.random((K, N), dtype=np.float32) * 2 - 1
    C = np.zeros((M, N), dtype=np.float32)

    # Warmup
    for _ in range(WARMUP):
        C.fill(0)
        C += A @ B

    # Timed runs (C += A@B to match 7.cpp semantics)
    times_ms = []
    for _ in range(NUM_ITERATIONS):
        C.fill(0)
        t0 = time.perf_counter()
        C += A @ B
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)

    mean_ms = np.mean(times_ms)
    std_ms = np.std(times_ms)
    flops_per_sec = (FLOPs_PER_MATMUL / 1e9) / (mean_ms / 1000)

    print("--- NumPy matmul (1024×1024 float32) ---")
    print(f"Iterations: {NUM_ITERATIONS}  (warmup: {WARMUP})")
    print(f"Time:       {mean_ms:.2f} ms  (std: {std_ms:.2f} ms)")
    print(f"FLOP/s:     {flops_per_sec:.2f} G/s")
    print("(Compare with 7.cpp: BM_RowColParallelInnerTilingRegisterAcc_1024)")

if __name__ == "__main__":
    main()

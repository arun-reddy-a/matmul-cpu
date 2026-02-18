# Matrix Multiplication (CPU) — Optimization Suite

A progressive series of dense matrix-matrix multiply implementations (1024×1024) demonstrating the impact of compiler flags, loop structure, cache awareness, tiling, and multithreading. Each step adds one or more optimizations; all are benchmarked with [Google Benchmark](https://github.com/google/benchmark).

---

## Prerequisites & Installation

Install dependencies with Homebrew (macOS):

```bash
brew install google-benchmark
brew install pkg-config
brew install libomp
```

- **google-benchmark** — benchmarking framework  
- **pkg-config** — optional; for build systems that use it to find benchmark  
- **libomp** — OpenMP runtime and headers (required for `7.cpp`)

**Python (for NumPy benchmark):** use a virtual environment and install NumPy inside it:

```bash
cd matmul-cpu
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install numpy
```

To run the NumPy benchmark later, activate the same environment (`source .venv/bin/activate`) then run `python3 benchmark_numpy_matmul.py` or `./run_all_benchmarks.sh`.

---

## How to Run

From the `matmul-cpu` directory:

```bash
./run_all_benchmarks.sh
```

This script:

- Compiles `1.cpp` and `2.cpp` with **default** clang (no extra optimization flags).
- Compiles `3.cpp`–`7.cpp` with **`-O3 -march=native -ffast-math`**.
- Links `7.cpp` with OpenMP (`-lomp` and libomp include/lib paths).
- Runs each benchmark and appends results to **`benchmark_results.txt`**.
- Runs **NumPy** matmul (same 1024×1024 float32) for comparison with `7.cpp`; requires a Python environment with NumPy (see **Prerequisites**).

---

## Optimizations by File

| File   | Name / focus | Optimizations |
|--------|----------------|---------------|
| **1.cpp** | Vanilla matmul | Naive triple loop, runtime dimensions; no compiler optimizations in the script. |
| **2.cpp** | Vanilla++ matmul | Same algorithm as 1, but **templated** on `M, N, K` so the compiler can specialize and optimize the inner loops. |
| **3.cpp** | Vanilla++ with flags | Same templated naive kernel as 2, built with **`-O3 -march=native -ffast-math`** to enable vectorization and arch-specific tuning. |
| **4.cpp** | Register accumulation | Templated kernel plus an **explicit scalar accumulator** in the inner loop (reduces repeated memory writes and helps the compiler keep the value in a register). |
| **5.cpp** | Cache-aware loop order | Register accumulation with **loop order** chosen for cache locality (e.g. row-major access, reordered loops) to improve L1/L2 reuse. |
| **6.cpp** | Cache-aware + inner tiling | Same cache-friendly structure with **inner-loop tiling** (blocking) so that working set fits better in cache and reuse is maximized. |
| **7.cpp** | Tiling + multithreading | Same tiled, register-accumulator kernel with **OpenMP** parallelization over rows/columns (or tiles), scaling across cores. |

---

## Benchmark Results (Sample Run)

All benchmarks compute **C += A×B** for **1024×1024** float matrices (2·M·N·K ≈ 2.15G FLOPs per run). Times below are from a single run; re-run the script to regenerate `benchmark_results.txt`.

| # | Implementation | Time (ms) | CPU (ms) | FLOP/s |
|---|----------------|-----------|----------|--------|
| 1 | Vanilla matmul | 6489 | 6373 | 337 M/s |
| 2 | Vanilla++ (templated) | 6387 | 6312 | 340 M/s |
| 3 | Vanilla++ + flags | 1537 | 1530 | 1.40 G/s |
| 4 | Register accumulation | 1435 | 1427 | 1.50 G/s |
| 5 | Cache-aware loop order | 138 | 138 | 15.6 G/s |
| 6 | Cache-aware + inner tiling | 132 | 131 | 16.4 G/s |
| 7 | Tiling + OpenMP (multi-thread) | 49.8 | 39.4 | 54.5 G/s |
| — | **NumPy** (`A @ B`, float32) | 31.3 | — | 68.7 G/s |

*Time*: wall-clock; *CPU*: total CPU time (for C++ benchmarks; for 7, sum across threads). NumPy uses wall-clock only. FLOP/s is reported by each benchmark.

### Comparing with NumPy

To benchmark **NumPy** matmul (same problem size) after setting up the environment:

```bash
source .venv/bin/activate   # if not already active
python3 benchmark_numpy_matmul.py
```

The script reports mean time (ms) and FLOP/s. It is also run automatically at the end of **`./run_all_benchmarks.sh`** (with whatever `python3` is in your `PATH`); its output is appended to **`benchmark_results.txt`**.

---

## Summary

Going from the naive implementation (1) to the fully optimized, tiled, and parallel one (7) yields a large speedup (on the order of **~100×** in wall time and **~160×** in FLOP/s in the sample above), driven by:

1. **Compilation**: templating + `-O3 -march=native -ffast-math`  
2. **Register usage**: explicit accumulator in the inner loop  
3. **Memory hierarchy**: cache-friendly loop order and tiling  
4. **Parallelism**: OpenMP over rows/columns or tiles  

For full, machine-specific numbers and metadata, inspect **`benchmark_results.txt`** after running **`./run_all_benchmarks.sh`**.

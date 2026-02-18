// matmul_bench.cpp
#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

#include <omp.h>

// ------------------------------------------------------------
// Row+Col tiling (256x256), inner tiling, OpenMP parallel over (rowTile, colTile)
// Register-acc variant: hoist A[row,inner] load out of the col loop, and
// use scalar acc for the update.
//
// NOTE: With loop order rowTile -> colTile -> innerTile -> row -> inner -> col,
// you *cannot* keep C[row,col] in a register across all inners without changing
// loop nesting. So "register acc" here means per-update scalar acc + hoisted A.
// ------------------------------------------------------------
template <int rows, int columns, int inners,
          int tileSize = 32 /* ROW_COL_PARALLEL_INNER_TILING_TILE_SIZE */>
inline void matmulImplRowColParallelInnerTilingRegisterAcc(const float* left,
                                                           const float* right,
                                                           float* result) {
#pragma omp parallel for shared(result, left, right) default(none) collapse(2) num_threads(8)
  for (int rowTile = 0; rowTile < rows; rowTile += 256) {
    for (int columnTile = 0; columnTile < columns; columnTile += 256) {
      for (int innerTile = 0; innerTile < inners; innerTile += tileSize) {
        for (int row = rowTile; row < rowTile + 256; row++) {
          const int innerTileEnd = std::min(inners, innerTile + tileSize);
          for (int inner = innerTile; inner < innerTileEnd; inner++) {
            const float a = left[row * inners + inner];  // reuse across cols in this tile
            for (int col = columnTile; col < columnTile + 256; col++) {
              float acc = result[row * columns + col];
              acc += a * right[inner * columns + col];
              result[row * columns + col] = acc;
            }
          }
        }
      }
    }
  }
}

// ------------------------------------------------------------
// Helpers (fixed seeds for reproducibility)
// ------------------------------------------------------------
static void FillRandom(std::vector<float>& v, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : v) x = dist(rng);
}

// ------------------------------------------------------------
// Benchmark: fixed 1024 x 1024 x 1024
// ------------------------------------------------------------
static void BM_RowColParallelInnerTilingRegisterAcc_1024(benchmark::State& state) {
  constexpr int M = 1024;
  constexpr int N = 1024;
  constexpr int K = 1024;
  constexpr int T = 32; // match your default tile size if needed

  std::vector<float> A(static_cast<size_t>(M) * K);
  std::vector<float> B(static_cast<size_t>(K) * N);
  std::vector<float> C(static_cast<size_t>(M) * N);

  FillRandom(A, 123u);
  FillRandom(B, 456u);
  std::fill(C.begin(), C.end(), 0.0f);

  for (auto _ : state) {
    // Kernel is C += A*B
    std::fill(C.begin(), C.end(), 0.0f);

    benchmark::DoNotOptimize(A.data());
    benchmark::DoNotOptimize(B.data());
    benchmark::DoNotOptimize(C.data());

    matmulImplRowColParallelInnerTilingRegisterAcc<M, N, K, T>(
        A.data(), B.data(), C.data());

    benchmark::ClobberMemory();
    benchmark::DoNotOptimize(C.data());
  }

  const double flops = 2.0 * static_cast<double>(M) * N * K;
  state.counters["FLOP/s"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_RowColParallelInnerTilingRegisterAcc_1024)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();

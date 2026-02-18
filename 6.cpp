// matmul_bench.cpp
#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

// ------------------------------------------------------------
// Tiled over K (inners). Register-acc variant for the innermost update.
//
// Loop structure preserved:
// innerTile -> row -> inner -> column
//
// "Register acc" here means: load C[row,col] into a scalar, update, store back,
// and hoist A[row,inner] out of the column loop.
// (Because inner is outside column, we cannot keep a per-(row,col) accumulator
// across all inners without changing loop nesting.)
// ------------------------------------------------------------
template <int rows, int columns, int inners, int tileSize>
inline void matmulImplTilingRegisterAcc(const float* left,
                                        const float* right,
                                        float* result) {
  for (int innerTile = 0; innerTile < inners; innerTile += tileSize) {
    for (int row = 0; row < rows; row++) {
      const int innerTileEnd = std::min(inners, innerTile + tileSize);
      for (int inner = innerTile; inner < innerTileEnd; inner++) {
        const float a = left[row * inners + inner];  // reuse across all columns
        for (int column = 0; column < columns; column++) {
          float acc = result[row * columns + column];
          acc += a * right[inner * columns + column];
          result[row * columns + column] = acc;
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
// Benchmark: fixed 1024 x 1024 x 1024, fixed tileSize
// ------------------------------------------------------------
static void BM_MatmulTilingRegisterAcc_1024(benchmark::State& state) {
  constexpr int M = 1024;
  constexpr int N = 1024;
  constexpr int K = 1024;
  constexpr int T = 32;   // change tile size here

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

    matmulImplTilingRegisterAcc<M, N, K, T>(A.data(), B.data(), C.data());

    benchmark::ClobberMemory();
    benchmark::DoNotOptimize(C.data());
  }

  const double flops = 2.0 * static_cast<double>(M) * N * K;
  state.counters["FLOP/s"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_MatmulTilingRegisterAcc_1024)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();

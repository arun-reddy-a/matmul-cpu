// matmul_bench.cpp
#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

// ------------------------------------------------------------
// Loop order: row -> inner -> col
// Register-accumulating version
// C += A * B
// ------------------------------------------------------------
template <int rows, int columns, int inners>
inline void matmulImplLoopOrderRegisterAcc(const float* left,
                                           const float* right,
                                           float* result) {
  for (int row = 0; row < rows; row++) {
    for (int inner = 0; inner < inners; inner++) {
      const float a = left[row * inners + inner];  // reused across cols
      for (int col = 0; col < columns; col++) {
        float acc = result[row * columns + col];
        acc += a * right[inner * columns + col];
        result[row * columns + col] = acc;
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
static void BM_MatmulLoopOrderRegisterAcc_1024(benchmark::State& state) {
  constexpr int M = 1024;
  constexpr int N = 1024;
  constexpr int K = 1024;

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

    matmulImplLoopOrderRegisterAcc<M, N, K>(
        A.data(), B.data(), C.data());

    benchmark::ClobberMemory();
    benchmark::DoNotOptimize(C.data());
  }

  const double flops =
      2.0 * static_cast<double>(M) * N * K;

  state.counters["FLOP/s"] =
      benchmark::Counter(flops,
          benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_MatmulLoopOrderRegisterAcc_1024)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();

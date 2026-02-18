// matmul_bench.cpp
#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

// Naive matmul (same style: no local accumulator).
// C[rows x columns] += A[rows x inners] * B[inners x columns]
template <int rows, int columns, int inners>
inline void matmulImplNaive(const float* left, const float* right, float* result) {
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < columns; col++) {
      for (int inner = 0; inner < inners; inner++) {
        result[row * columns + col] +=
            left[row * inners + inner] * right[inner * columns + col];
      }
    }
  }
}

static void FillRandom(std::vector<float>& v, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : v) x = dist(rng);
}

static void BM_MatmulNaive_Template_1024(benchmark::State& state) {
  constexpr int M = 1024;
  constexpr int N = 1024;
  constexpr int K = 1024;

  // Fixed seeds => identical matrices across runs/implementations.
  std::vector<float> A(static_cast<size_t>(M) * K);
  std::vector<float> B(static_cast<size_t>(K) * N);
  std::vector<float> C(static_cast<size_t>(M) * N);

  FillRandom(A, 123u);
  FillRandom(B, 456u);
  std::fill(C.begin(), C.end(), 0.0f);

  for (auto _ : state) {
    // Kernel is C += A*B, so reset each iteration for fair timing.
    std::fill(C.begin(), C.end(), 0.0f);

    benchmark::DoNotOptimize(A.data());
    benchmark::DoNotOptimize(B.data());
    benchmark::DoNotOptimize(C.data());

    matmulImplNaive<M, N, K>(A.data(), B.data(), C.data());

    benchmark::ClobberMemory();
    benchmark::DoNotOptimize(C.data());
  }

  // 2*M*N*K FLOPs (mul+add)
  const double flops = 2.0 * static_cast<double>(M) * N * K;
  state.counters["FLOP/s"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_MatmulNaive_Template_1024)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();

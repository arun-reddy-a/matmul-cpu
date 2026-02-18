#!/usr/bin/env bash
# Run all seven matmul benchmarks: 1,2 with default clang; 3-7 with -O3 -march=native -ffast-math.
# Requires: clang++, Google Benchmark (libbenchmark). Output is written to benchmark_results.txt.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
cd "$SCRIPT_DIR"
RESULTS_FILE="${SCRIPT_DIR}/benchmark_results.txt"
BIN_DIR="${SCRIPT_DIR}/build"
mkdir -p "$BIN_DIR"

# Same as your working command: Homebrew include/lib + benchmark linkage
BASE_CXX="clang++ -std=c++17 -I/opt/homebrew/include"
BASE_LD="-L/opt/homebrew/lib -lbenchmark -lpthread"

echo "========================================" | tee "$RESULTS_FILE"
echo "Matmul CPU benchmark results" | tee -a "$RESULTS_FILE"
echo "Generated: $(date)" | tee -a "$RESULTS_FILE"
echo "========================================" | tee -a "$RESULTS_FILE"

run_one() {
  local src="$1"
  local name="$2"
  local extra_cxx="$3"
  local extra_ld="${4:-}"
  local src_path="${SRC_DIR}/$src"
  local base="${BIN_DIR}/bench_${name}"
  local ld_flags="$BASE_LD $extra_ld"
  echo "" | tee -a "$RESULTS_FILE"
  echo "--- $src ($name) ---" | tee -a "$RESULTS_FILE"
  echo "Compile: $BASE_CXX $extra_cxx $src_path -o ${base} $ld_flags" | tee -a "$RESULTS_FILE"
  if ! $BASE_CXX $extra_cxx "$src_path" -o "${base}" $ld_flags; then
    echo "FAILED to compile $src" | tee -a "$RESULTS_FILE"
    return 1
  fi
  echo "Running..." | tee -a "$RESULTS_FILE"
  if "${base}" 2>&1 | tee -a "$RESULTS_FILE"; then
    echo "Done: $src" | tee -a "$RESULTS_FILE"
  else
    echo "FAILED to run $src" | tee -a "$RESULTS_FILE"
    return 1
  fi
}

# 1.cpp and 2.cpp: regular clang (no -O3/-march/-ffast-math)
run_one "1.cpp" "1" ""
run_one "2.cpp" "2" ""

# 3–7: optimized flags
OPT_FLAGS="-O3 -march=native -ffast-math"
run_one "3.cpp" "3" "$OPT_FLAGS"
run_one "4.cpp" "4" "$OPT_FLAGS"
run_one "5.cpp" "5" "$OPT_FLAGS"
run_one "6.cpp" "6" "$OPT_FLAGS"
# 7.cpp uses OpenMP: need libomp include path, -fopenmp, and -lomp
OMP_FLAGS="-I/opt/homebrew/opt/libomp/include -Xpreprocessor -fopenmp"
run_one "7.cpp" "7" "$OPT_FLAGS $OMP_FLAGS" "-L/opt/homebrew/opt/libomp/lib -lomp"

# NumPy matmul (same 1024×1024 float32, for comparison with 7.cpp)
echo "" | tee -a "$RESULTS_FILE"
echo "--- NumPy matmul (1024×1024 float32) ---" | tee -a "$RESULTS_FILE"
if command -v python3 &>/dev/null; then
  python3 "${SRC_DIR}/benchmark_numpy_matmul.py" 2>&1 | tee -a "$RESULTS_FILE" || true
else
  echo "python3 not found; skip NumPy benchmark" | tee -a "$RESULTS_FILE"
fi

echo "" | tee -a "$RESULTS_FILE"
echo "========================================" | tee -a "$RESULTS_FILE"
echo "All benchmarks finished. Results in: $RESULTS_FILE" | tee -a "$RESULTS_FILE"

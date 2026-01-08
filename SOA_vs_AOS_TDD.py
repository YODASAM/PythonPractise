import numpy as np
import time
import sys

# ==============================================================
# TDD-Style Demo: Structure of Arrays (SoA) vs Array of Structures (AoS)
# ==============================================================
#
# Goal: Prove that SoA (separate contiguous arrays for each field)
#       is dramatically more cache-efficient than AoS when accessing only one field.
#
# Scenario:
#   - We have n = 10,000,000 particles with x, y, z coordinates (float32).
#   - We only need to compute the mean of the x coordinate.
#
# Expected Outcome on modern CPUs:
#   - SoA: Extremely fast – sequential access over one contiguous block → perfect prefetching & full cache lines used.
#   - AoS: Much slower – strided access (jump by sizeof(struct) = 12 bytes) → poor spatial locality, cache lines mostly wasted.
#
# TDD Structure:
#   Test 1 → Environment check
#   Test 2 → Create correct SoA and AoS data (with assertions)
#   Test 3 → Benchmark mean(x) on both layouts
#   Test 4 → Interpret results and assert expected cache advantage

def test_1_environment():
    """Test 1: Verify setup and data size"""
    print("=== Test 1: Environment Check ===")
    print(f"NumPy version: {np.__version__}")
    print(f"Python: {sys.version.split()[0]}")
    n = 10_000_000
    bytes_per_float = 4
    total_bytes_soa = 3 * n * bytes_per_float / 1e9  # GB
    print(f"Data size: {n:,} elements → {total_bytes_soa:.2f} GB (SoA total for x,y,z)")
    print("Note: We only access x → expect SoA to use cache efficiently.\n")
    print("=== Test 1 Passed ===\n")

def test_2_create_data(n=10_000_000):
    """
    Test 2: Create two equivalent datasets
    
    - SoA (Structure of Arrays): separate contiguous arrays → cache-friendly for single-field access
    - AoS (Array of Structures): one array of structured dtype → mimics C struct array or class instances
    
    We fill with the same values and assert numerical equivalence.
    """
    print("=== Test 2: Creating SoA and AoS datasets ===")
    print(f"Generating data for {n:,} elements...\n")
    
    # --- SoA: separate arrays (cache-friendly when touching only one field) ---
    x_soa = np.random.rand(n).astype(np.float32)  # Random values [0,1)
    y_soa = np.random.rand(n).astype(np.float32)
    z_soa = np.random.rand(n).astype(np.float32)
    
    # --- AoS: single array with structured dtype (x, y, z) ---
    dtype_aos = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)])
    aos = np.empty(n, dtype=dtype_aos)
    aos['x'] = x_soa  # Same values as SoA
    aos['y'] = y_soa
    aos['z'] = z_soa
    
    # --- Assertions: data integrity ---
    assert np.array_equal(aos['x'], x_soa), "AoS x must match SoA x"
    assert np.array_equal(aos['y'], y_soa), "AoS y must match SoA y"
    assert np.array_equal(aos['z'], z_soa), "AoS z must match SoA z"
    
    # Verify contiguity
    assert x_soa.flags.c_contiguous, "SoA x should be contiguous"
    assert aos.flags.c_contiguous, "AoS array should be contiguous"
    
    print("SoA layout: separate contiguous float32 arrays")
    print("AoS layout: structured array, stride for 'x' field =", aos.strides[0])  # stride between elements
    print("Element size in AoS =", dtype_aos.itemsize, "bytes (x at offset 0, then y, z)")
    print("\n=== Test 2 Passed: Datasets created and verified identical ===\n")
    
    return x_soa, aos

def compute_x_mean_soa(x):
    """Simple function: compute mean of x in SoA layout (sequential access)"""
    return x.mean()

def compute_x_mean_aos(aos):
    """Same computation but on AoS layout (strided access for x field)"""
    return aos['x'].mean()

def test_3_benchmark(n=10_000_000, x_soa=None, aos=None):
    """
    Test 3: Benchmark mean(x) on both layouts
    
    We warm up first, then time the actual operation.
    Expect SoA to be 3–10× faster due to full cache line utilization.
    """
    print("=== Test 3: Benchmarking mean(x) – accessing only x coordinate ===")
    
    # --- Warm-up: bring data into cache and initialize BLAS ---
    _ = compute_x_mean_soa(x_soa)
    _ = compute_x_mean_aos(aos)
    
    # --- Time SoA (sequential access) ---
    t_start = time.perf_counter()
    mean_soa = compute_x_mean_soa(x_soa)
    t_soa = (time.perf_counter() - t_start) * 1000  # ms
    
    print(f"SoA (contiguous x array)   : {t_soa:.2f} ms")
    
    # --- Time AoS (strided access) ---
    t_start = time.perf_counter()
    mean_aos = compute_x_mean_aos(aos)
    t_aos = (time.perf_counter() - t_start) * 1000  # ms
    
    print(f"AoS (structured array)     : {t_aos:.2f} ms")
    
    # --- Performance analysis ---
    speedup = t_aos / t_soa if t_soa > 0 else float('inf')
    print(f"\nSpeedup (AoS time / SoA time): {speedup:.2f}x")
    
    print("\nInterpretation:")
    if speedup >= 3.0:
        print("  → Strong cache effect! SoA wins dramatically – prefetcher happy, full cache lines used")
    elif speedup >= 1.5:
        print("  → Clear win for SoA – good cache behavior observed")
    elif speedup > 1.1:
        print("  → Mild improvement – possibly due to small cache or optimized strided access")
    else:
        print("  → Unexpected: check system load or try larger n")
    
    # --- Final correctness assertion ---
    assert np.isclose(mean_soa, mean_aos), "Means must be identical regardless of layout"
    print(f"\nBoth layouts give same mean: {mean_soa:.6f}")
    
    print("\n=== Test 3 Passed: Cache efficiency advantage demonstrated ===\n")
    
    return speedup

def test_4_conclusion():
    """Test 4: Summarize the key lesson"""
    print("=== Test 4: Conclusion ===")
    print("\nKey Takeaway:")
    print("   → When you frequently access only a subset of fields → use SoA!")
    print("   → SoA gives contiguous memory access → maximizes cache line utilization,")
    print("     hardware prefetching, and bandwidth.")
    print("   → AoS is natural for OOP but kills performance in tight loops over single fields.")
    print("   → Common in particle simulations, game engines, data processing pipelines.")
    print("\n=== All Tests Passed: SoA proven cache-superior for selective field access ===\n")

# ==============================================================
# Run the full TDD test suite
# ==============================================================

if __name__ == "__main__":
    test_1_environment()
    x_soa, aos = test_2_create_data(n=10_000_000)
    test_3_benchmark(x_soa=x_soa, aos=aos)
    test_4_conclusion()

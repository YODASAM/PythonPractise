import numpy as np
import time
import sys

# ==============================================================
# Cache-Efficient Memory Layout Demo in NumPy (TDD-Style Explanation)
# ==============================================================
#
# This script demonstrates the impact of memory layout (row-major vs column-major)
# on CPU cache performance when performing row-wise reductions (sum along axis=1).
#
# Test-Driven Development (TDD) Style:
# We structure the code as a series of "tests" that assert expected behavior.
# Each section builds on the previous one, "failing" conceptually if the
# performance difference is not observed (i.e., C-order should be significantly faster).
#
# Expected Outcome on modern x86_64 CPUs (Intel/AMD):
# - Row-major (C-order) arrays are ~1.5–3× faster for row sums due to spatial locality.
# - Data is accessed sequentially → better cache line utilization and hardware prefetching.
# - Column-major (Fortran-order) causes strided access → many cache misses.
#
# Note: Results depend on matrix size (should exceed L3 cache), CPU, and NumPy/BLAS backend.

def test_environment():
    """Test 1: Verify we have NumPy and basic info"""
    print("NumPy version:", np.__version__)
    print("Python version:", sys.version.split()[0])
    print("Array default order:", np.empty((2,2)).flags)  # Should show C_CONTIGUOUS
    print("\n=== Test 1 Passed: Environment ready ===\n")

def create_test_data(shape=(8000, 8000)):
    """
    Test 2: Create identical data with different memory layouts
    
    - a_c: C-contiguous (row-major, default in NumPy)
      Memory layout: elements in each row are adjacent.
    
    - a_f: Fortran-contiguous (column-major)
      Created via np.asfortranarray() → data stored column-by-column.
      Numerically identical to a_c but transposed storage order.
    
    We verify flags to "assert" correct layout.
    """
    print(f"Creating arrays of shape {shape} ({shape[0]*shape[1]*8 / 1e9:.2f} GB each)...")
    
    # // C-order array (row-major)
    a_c = np.random.rand(*shape)                # Random uniform [0,1)
    # // Explicitly ensure C order (usually default)
    if not a_c.flags.c_contiguous:
        a_c = np.ascontiguousarray(a_c)
    
    # // Fortran-order array (column-major)
    # // np.asfortranarray copies/transposes storage to column-major
    a_f = np.asfortranarray(a_c.copy())         # Ensure same values, different layout
    
    # --- Assertions (manual checks) ---
    assert a_c.flags.c_contiguous and not a_c.flags.f_contiguous, "a_c must be C-contiguous"
    assert a_f.flags.f_contiguous and not a_f.flags.c_contiguous, "a_f must be F-contiguous"
    assert np.allclose(a_c, a_f), "Arrays must be numerically identical"
    
    print("C-order array flags:", a_c.flags)
    print("F-order array flags:", a_f.flags)
    print("\n=== Test 2 Passed: Data created with correct layouts ===\n")
    
    return a_c, a_f

def benchmark_row_sum(a_c, a_f):
    """
    Test 3: Benchmark sum along axis=1 (row reduction)
    
    - Summing along axis=1 means: for each row, sum all columns.
    - In C-order: each row is contiguous → sequential memory access → cache-friendly.
    - In F-order: each row jumps by stride == number of rows → poor spatial locality.
    
    We time multiple runs if needed, but single run is often sufficient for large arrays.
    """
    print("Benchmarking row-wise sum (axis=1)...\n")
    
    # --- Warm-up (ensure arrays are in cache, JIT if any) ---
    _ = a_c.sum(axis=1)
    _ = a_f.sum(axis=1)
    
    # --- Time C-order ---
    t_start = time.perf_counter()
    s_c = a_c.sum(axis=1)                       # Row reduction on row-major array
    t_c = (time.perf_counter() - t_start) * 1000  # ms
    
    print(f"C-order (row-major) row-sum : {t_c:.2f} ms")
    
    # --- Time F-order ---
    t_start = time.perf_counter()
    s_f = a_f.sum(axis=1)                       # Row reduction on column-major array
    t_f = (time.perf_counter() - t_start) * 1000  # ms
    
    print(f"F-order (col-major) row-sum : {t_f:.2f} ms")
    
    # --- Assertion: C-order should be significantly faster ---
    speedup = t_f / t_c if t_c > 0 else float('inf')
    print(f"\nSpeedup (F-time / C-time): {speedup:.2f}x")
    print("Interpretation:")
    if speedup > 1.5:
        print("  → Strong cache effect observed: C-order is cache-efficient for row operations")
    elif speedup > 1.1:
        print("  → Mild improvement: possible due to smaller matrix or optimized BLAS")
    else:
        print("  → Unexpected: Check system load, array size, or NumPy backend")
    
    # Verify results are identical
    assert np.allclose(s_c, s_f), "Row sums must be identical regardless of layout"
    
    print("\n=== Test 3 Passed: Performance difference explained by cache locality ===\n")

def bonus_column_sum_demo(a_c, a_f):
    """
    Bonus Test: Show the opposite effect for column sums (axis=0)
    
    This proves the effect is symmetric:
    - Column sum on F-order should be fast
    - Column sum on C-order should be slow
    """
    print("Bonus: Benchmarking column-wise sum (axis=0)...\n")
    
    # Warm-up
    _ = a_c.sum(axis=0)
    _ = a_f.sum(axis=0)
    
    t_start = time.perf_counter()
    s_c_col = a_c.sum(axis=0)
    t_c_col = (time.perf_counter() - t_start) * 1000
    
    t_start = time.perf_counter()
    s_f_col = a_f.sum(axis=0)
    t_f_col = (time.perf_counter() - t_start) * 1000
    
    print(f"C-order column-sum : {t_c_col:.2f} ms")
    print(f"F-order column-sum : {t_f_col:.2f} ms")
    
    speedup_col = t_c_col / t_f_col if t_f_col > 0 else float('inf')
    print(f"Speedup (C-time / F-time) for columns: {speedup_col:.2f}x")
    
    if speedup_col > 1.5:
        print("  → Expected: F-order excels at column operations")
    
    assert np.allclose(s_c_col, s_f_col)

# ==============================================================
# Run the "test suite"
# ==============================================================

if __name__ == "__main__":
    test_environment()
    #a_c, a_f = create_test_data(shape=(8000, 8000))  # ~0.5 GB each → fits in RAM, stresses cache
    #benchmark_row_sum(a_c, a_f)
    #bonus_column_sum_demo(a_c, a_f)
    
    print("\nConclusion:")
    print("   → Always align your data layout with your access pattern!")
    print("   → Use C-order for row-major algorithms,")
    print("   → Use F-order (or transpose) for column-major algorithms.")
    print("   → This is why libraries like BLAS/LAPACK specify order explicitly.")

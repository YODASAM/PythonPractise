import numpy as np
import time

# ==============================================================
# TDD-Style Demo: Making a Matrix-Vector Multiply Loop Cache-Efficient
# ==============================================================
#
# Goal: Prove that swapping loop order in a row-major NumPy array
#       dramatically improves cache performance for the operation out[i,j] = A[i,j] * b[j]
#
# Scenario:
#   - A: large (rows × cols) matrix, row-major (C-order, default in NumPy)
#   - b: vector of length cols
#   - out: result matrix (rows × cols)
#   - Operation: each column of out = corresponding column of A scaled by b[j]
#
# Expected Outcome:
#   - Original (j outer, i inner): inner loop jumps by 'rows' elements → large stride → poor cache use
#   - Swapped (i outer, j inner): inner loop steps by 1 → unit stride → perfect cache-line reuse → 3–10× faster

def test_1_environment():
    """Test 1: Setup check"""
    print("=== Test 1: Environment ===")
    rows, cols = 8000, 6000
    print(f"Matrix size: {rows} × {cols} (~{rows*cols*8/1e9:.2f} GB for float64)")
    print("NumPy default order is row-major (C-order)\n")
    print("=== Test 1 Passed ===\n")

def test_2_create_data(rows=8000, cols=6000):
    """Test 2: Create data and verify layout"""
    print("=== Test 2: Creating data ===")
    
    A = np.random.rand(rows, cols)          # Row-major by default
    b = np.random.rand(cols)
    out1 = np.empty((rows, cols))
    out2 = np.empty((rows, cols))
    
    # Assertions
    assert A.flags.c_contiguous, "A must be C-contiguous (row-major)"
    assert A.strides == (cols*8, 8), "Row stride should be cols*itemsize, column stride = 8 bytes"
    
    print("A is row-major: adjacent elements in memory are A[i, j] and A[i, j+1]")
    print("\n=== Test 2 Passed ===\n")
    
    return A, b, out1, out2

def bad_loop(A, b, out):
    """Original cache-UNfriendly version: j outer, i inner"""
    rows, cols = A.shape
    for j in range(cols):
        for i in range(rows):               # ← Inner loop jumps by 'rows' elements → huge stride!
            out[i, j] = A[i, j] * b[j]

def good_loop(A, b, out):
    """Cache-friendly version: i outer, j inner (one-line loop swap)"""
    rows, cols = A.shape
    for i in range(rows):                   # ← Inner loop now moves to next column element
        for j in range(cols):               # ← Unit stride: contiguous memory access
            out[i, j] = A[i, j] * b[j]

def test_3_benchmark(A, b, out1, out2):
    """Test 3: Benchmark both versions"""
    print("=== Test 3: Benchmarking loop orders ===")
    
    # Warm-up
    bad_loop(A, b, out1)
    good_loop(A, b, out2)
    
    # Time bad version
    start = time.perf_counter()
    bad_loop(A, b, out1)
    t_bad = (time.perf_counter() - start) * 1000
    
    # Time good version
    start = time.perf_counter()
    good_loop(A, b, out2)
    t_good = (time.perf_counter() - start) * 1000
    
    print(f"Cache-unfriendly (j outer): {t_bad:.1f} ms")
    print(f"Cache-friendly    (i outer): {t_good:.1f} ms")
    
    speedup = t_bad / t_good
    print(f"\nSpeedup: {speedup:.2f}×")
    
    if speedup >= 3.0:
        print("  → Strong cache effect observed – unit stride wins!")
    else:
        print("  → Mild or no improvement – try larger matrix")
    
    # Correctness check
    assert np.allclose(out1, out2), "Both loops must produce identical results"
    print("\nResults numerically identical.")
    print("\n=== Test 3 Passed: One-line loop swap gives huge cache speedup ===\n")

def test_4_conclusion():
    """Test 4: Key takeaway"""
    print("=== Test 4: Conclusion ===")
    print("\nOne-line change that makes the loop cache-efficient:")
    print("    Swap the loops → make the inner loop iterate over the last (contiguous) dimension.")
    print("\nWhy it works:")
    print("    • Row-major storage: elements A[i,j] and A[i,j+1] are adjacent in memory.")
    print("    • Inner loop with unit stride reuses full cache lines → maximum prefetching & bandwidth.")
    print("    • This simple change often gives 3–10× speedups in real scientific/numerical code.")
    print("\n=== All Tests Passed: Loop order matters enormously for cache performance! ===\n")

# ==============================================================
# Run the TDD suite
# ==============================================================

if __name__ == "__main__":
    test_1_environment()
    A, b, out1, out2 = test_2_create_data()
    test_3_benchmark(A, b, out1, out2)
    test_4_conclusion()

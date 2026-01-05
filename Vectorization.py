import numpy as np
import time

n = 10_000_000

# Python loop + list comprehension
start = time.time()
python_result = [i**2 + 2*i + 1 for i in range(n)]
python_time = time.time() - start

# NumPy vectorized
arr = np.arange(n)
start = time.time()
np_result = arr**2 + 2*arr + 1
np_time = time.time() - start

print(f"Python loop:       {python_time:.3f} seconds")
print(f"NumPy vectorized:  {np_time:.3f} seconds")
print(f"Speedup:           {python_time / np_time:.1f}×")
assert np.allclose(python_result, np_result)

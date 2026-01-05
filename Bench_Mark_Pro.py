import numpy as np
import pandas as pd
import timeit
import matplotlib.pyplot as plt
from memory_profiler import memory_usage  # pip install memory-profiler for memory tracking

# 1. Vectorization vs Loops
def loop_square(n=1_000_000):
    lst = list(range(n))
    return [x**2 for x in lst]

def vectorized_square(n=1_000_000):
    arr = np.arange(n)
    return arr**2

print("Vectorization vs Loops:")
print("Loop time:", timeit.timeit(lambda: loop_square(), number=10))
print("Vectorized time:", timeit.timeit(lambda: vectorized_square(), number=10))

# 2. Broadcasting Example
A = np.random.rand(5, 1, 3)
B = np.random.rand(3, 7)
C = A + B  # Shape: (5, 7, 3)
print("\nBroadcasting result shape:", C.shape)

# 3. View vs Copy
arr = np.arange(10)
view = arr[::2]          # View
copy = arr[::2].copy()   # Explicit copy
view[0] = 99
print("\nView modifies original:", arr[0] == 99)
copy[0] = 88
print("Copy does not modify original:", arr[0] != 88)

# 4. Memory Layout & Cache Efficiency
def row_major_sum(arr):
    return np.sum(arr, axis=1)  # Fast for C-order

def col_major_sum(arr):
    return np.sum(arr, axis=0)  # Slower for C-order

img_c = np.random.rand(1000, 1000, 3)  # C-order (default)
img_f = np.asfortranarray(img_c)       # F-order

print("\nC-order row sum time:", timeit.timeit(lambda: row_major_sum(img_c), number=100))
print("C-order col sum time:", timeit.timeit(lambda: col_major_sum(img_c), number=100))
print("F-order col sum time:", timeit.timeit(lambda: col_major_sum(img_f), number=100))

# 5. Categorical Memory Savings
df_obj = pd.DataFrame({'col': np.random.choice(['A', 'B', 'C', 'D', 'E'], size=50_000_000)})
df_cat = df_obj.copy()
df_cat['col'] = df_cat['col'].astype('category')

print("\nObject dtype memory (MB):", df_obj.memory_usage(deep=True).sum() / 1e6)
print("Categorical dtype memory (MB):", df_cat.memory_usage(deep=True).sum() / 1e6)

# 6. GroupBy Reuse Efficiency
df = pd.DataFrame({
    'dept': np.random.choice(['HR', 'IT', 'Sales'], 1_000_000),
    'salary': np.random.randn(1_000_000)
})

gb = df.groupby('dept')['salary']
print("\nReused GroupBy mean:", gb.mean())
print("Reused GroupBy std:", gb.std())  # Reuses cached groups → faster than new groupby

# 7. Chunked Processing for Large CSV
def process_large_csv(path="large.csv", chunksize=1_000_000):
    total, count = 0.0, 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        total += chunk['value'].sum()
        count += chunk['value'].count()
    return total / count if count else np.nan

# Simulate large CSV (run once)
# pd.DataFrame({'value': np.random.randn(200_000_000)}).to_csv('large.csv', index=False)
# print("\nMean from large CSV:", process_large_csv())

print("\nProject complete! Run sections individually for benchmarks.")

# 03_vectorization_filtering_conditionals.py
import numpy as np
import time

# Simulate sensor data: temperature readings from 1 million devices
n = 1_000_000
temperatures = np.random.normal(25, 10, n)  # Mean 25°C, std 10

# --- Loop version: Find all devices with temp > 35°C ---
start = time.time()
hot_devices_loop = []
for temp in temperatures:
    if temp > 35:
        hot_devices_loop.append(temp)
loop_time = time.time() - start

# --- Vectorized version: Boolean mask + filtering ---
start = time.time()
hot_mask = temperatures > 35                # Boolean array: True/False
hot_devices_vec = temperatures[hot_mask]    # One line filtering!
vec_time = time.time() - start

print(f"Finding overheating devices ({len(hot_devices_vec)} found):")
print(f"Loop time:      {loop_time:.3f} seconds")
print(f"Vectorized time:{vec_time:.3f} seconds")
print(f"Speedup:        {loop_time / vec_time:.1f}×")

assert np.allclose(hot_devices_loop, hot_devices_vec)

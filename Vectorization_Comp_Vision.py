# 02_vectorization_image_processing.py
import numpy as np
import time

# Simulate a 1920x1080 grayscale image (2 million pixels)
height, width = 1080, 1920
image = np.random.randint(0, 256, size=(height, width), dtype=np.uint8)

# --- Loop version: Adjust brightness pixel by pixel ---
start = time.time()
brightened_loop = np.zeros_like(image)
for i in range(height):
    for j in range(width):
        pixel = image[i, j]
        new_pixel = min(255, pixel + 50)  # Increase brightness by 50
        brightened_loop[i, j] = new_pixel
loop_time = time.time() - start

# --- Vectorized version: Adjust ALL pixels at once ---
start = time.time()
brightened_vec = np.clip(image + 50, 0, 255).astype(np.uint8)
vec_time = time.time() - start

print(f"Image brightening (2M pixels):")
print(f"Loop time:      {loop_time:.3f} seconds")
print(f"Vectorized time:{vec_time:.3f} seconds")
print(f"Speedup:        {loop_time / vec_time:.1f}×")

assert np.array_equal(brightened_loop, brightened_vec)

# 02_vectorization_image_processing_with_visuals.py

import numpy as np
import time
import matplotlib.pyplot as plt

# Simulate a 1920x1080 grayscale image (~2 million pixels)
height, width = 1080, 1920
print("Generating simulated 1920x1080 grayscale image...")
image = np.random.randint(0, 256, size=(height, width), dtype=np.uint8)

# --- Python nested loop version (SLOW) ---
print("\nRunning slow pixel-by-pixel loop...")
start = time.time()
brightened_loop = np.zeros_like(image)
for i in range(height):
    for j in range(width):
        pixel = image[i, j]
        new_pixel = min(255, pixel + 50)  # Increase brightness by 50
        brightened_loop[i, j] = new_pixel
loop_time = time.time() - start

# --- NumPy vectorized version (FAST) ---
print("Running fast vectorized operation...")
start = time.time()
brightened_vec = np.clip(image + 50, 0, 255).astype(np.uint8)
vec_time = time.time() - start

# --- Performance results ---
print("\n" + "="*50)
print(f"Image brightening (2,073,600 pixels):")
print(f"Python nested loop:  {loop_time:.3f} seconds")
print(f"NumPy vectorized:    {vec_time:.3f} seconds")
print(f"Speedup:             {loop_time / vec_time:.1f}×")
print("="*50)

# --- Verify both methods give same result ---
assert np.array_equal(brightened_loop, brightened_vec)
print("✓ Results are identical!")

# --- VISUALIZATION: See the difference! ---
print("\nDisplaying images...")

plt.figure(figsize=(16, 9))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.title("Original Simulated Grayscale Image", fontsize=16)
plt.axis('off')

# Brightened image
plt.subplot(1, 2, 2)
plt.imshow(brightened_vec, cmap='gray', vmin=0, vmax=255)
plt.title(f"Brightened (+50) — Vectorized\nSpeedup: {loop_time / vec_time:.1f}×", fontsize=16)
plt.axis('off')

plt.suptitle("NumPy Vectorization Demo: Image Brightness Adjustment", fontsize=20, y=0.95)
plt.tight_layout()
plt.show()

# Optional: Save images for later
# plt.imsave('original.png', image, cmap='gray')
# plt.imsave('brightened.png', brightened_vec, cmap='gray')

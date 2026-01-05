# broadcasting_all_cases_demo.py

import numpy as np
import matplotlib.pyplot as plt

print("NumPy Broadcasting: All 4 Cases in One Program\n" + "="*60)

# Case 1: Scalar + 1D Array (Simplest stretching)
print("CASE 1: Scalar + 1D Array")
a = np.array([1, 2, 3, 4])  # shape (4,)
scalar = 10                # shape () → treated as scalar
result1 = a + scalar       # scalar stretches to [10,10,10,10]
print(f"a:     {a}")
print(f"+ {scalar} → stretches to [10,10,10,10]")
print(f"Result: {result1}")
print(f"Shapes: a={a.shape}, scalar=(), result={result1.shape}\n")

# Case 2: 2D Matrix + 1D Vector (Row-wise addition)
print("CASE 2: 2D Matrix + 1D Vector (Row-wise)")
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])  # shape (3,3)
vector = np.array([10, 20, 30])  # shape (3,)
result2 = matrix + vector        # vector → (1,3) → stretched to (3,3)
print(f"Matrix:\n{matrix}")
print(f"+ vector {vector} → stretches vertically")
print(f"Result:\n{result2}")
print(f"Shapes: matrix={matrix.shape}, vector={vector.shape}, result={result2.shape}\n")

# Case 3: Column Vector (n,1) + Row Vector (1,m) → Full (n,m) Matrix
print("CASE 3: Column + Row Vector → Outer Addition (Full Matrix)")
col_vec = np.array([[100],
                    [200],
                    [300]])      # shape (3,1)
row_vec = np.array([1, 2, 3])     # shape (3,) → treated as (1,3)
result3 = col_vec + row_vec       # Both stretch → (3,3)
print(f"Column vector:\n{col_vec}")
print(f"+ Row vector: {row_vec} → stretches both ways")
print(f"Result (outer addition):\n{result3}")
print(f"Shapes: col={col_vec.shape}, row={row_vec.shape}, result={result3.shape}\n")

# Case 4: Incompatible Shapes → Error
print("CASE 4: Incompatible Shapes → Broadcasting Fails")
try:
    arr1 = np.array([1, 2])           # shape (2,)
    arr2 = np.array([[10, 20, 30]])   # shape (1,3)
    result4 = arr1 + arr2             # Cannot align: 2 ≠ 3
except ValueError as e:
    print(f"Error: {e}")
    print(f"Shapes: arr1={arr1.shape}, arr2={arr2.shape} → incompatible!\n")

# === VISUALIZATION: See Case 3 (Outer Addition) as a Heatmap ===
print("Visualizing Case 3: Outer Addition Result as Heatmap")
plt.figure(figsize=(8, 6))
plt.imshow(result3, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Value')
plt.title("Broadcasting Case 3: (3,1) + (1,3) → (3,3) Outer Addition", fontsize=14)
plt.xticks([0,1,2], row_vec)
plt.yticks([0,1,2], col_vec.flatten())
plt.xlabel("Added from Row Vector →")
plt.ylabel("← Added from Column Vector")
for i in range(3):
    for j in range(3):
        plt.text(j, i, result3[i,j], ha='center', va='center', color='white', fontweight='bold')
plt.tight_layout()
plt.show()

print("="*60)
print("All 4 broadcasting cases demonstrated!")

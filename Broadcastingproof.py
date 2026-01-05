import numpy as np

A = np.random.rand(5, 1, 3)   # Shape (5, 1, 3)
B = np.random.rand(7, 3)      # Shape (7, 3)

# Broadcasting: A stretched to (5,7,3), B stretched to (1,7,3)
C = A + B
print("Shape of A:", A.shape)
print("Shape of B:", B.shape)
print("Result shape after broadcasting:", C.shape)  # (5, 7, 3)

# Visual proof
print("\nFirst slice (A[0] + B):")
print((A[0] + B).shape)  # (7, 3) — same row of A added to all of B

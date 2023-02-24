### Import the libraries first

import numpy as np


A = np.array([[4, 9, 9], [9, 1, 6], [9, 2, 3]])
print("Matrix A (3 by 3):\n", A)

B = np.array([[2, 2], [5, 7], [4, 4]])
print("Matrix B (3 by 2):\n", B)

# You can multiply matrices A and B using np.matmul

np.matmul(A,B)

# Python @ operator also works for matrix multiplication

A@B

#### MATRIX CONVENTION and BROADCASTING
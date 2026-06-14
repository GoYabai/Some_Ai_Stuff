import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A = np.array(A)
    row, col = A.shape
    re = np.zeros((col, row))
    for j in range(col):
        for i in range(row):
            re[j][i] = A[i][j]
    return re

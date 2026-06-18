import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # Write code here
    a = np.array(a)
    b = np.array(b)
    re1 = np.dot(a, b)
    re2 = np.linalg.norm(a) * np.linalg.norm(b)
    if re2 == 0:
        return 0
    else:
        return re1 / re2
# --- Hopfield Network Functions ---
import numpy as np
from data import x1, x2, x3, x4, x5

def flatten_pattern(matrix):
    """Flatten a 2D pattern matrix to a 1D vector using typewriter scheme."""
    return np.array(matrix).flatten()

def hebb_weight_matrix(patterns):
    """
    Compute weight matrix using Hebb's rule for a list of patterns.
    Diagonal weights are set to zero.
    patterns: list of numpy arrays (each pattern is a 1D array of -1/1)
    Returns: weight matrix (numpy array)
    """
    patterns = np.array(patterns)
    n = patterns.shape[1]
    W = np.zeros((n, n))
    for p in patterns:
        W += np.outer(p, p)
    W /= len(patterns)
    np.fill_diagonal(W, 0)
    return W

def signum(x):
    return 1 if x >= 0 else -1

def hopfield_async_update(W, s, max_iter=100):
    """
    Asynchronous deterministic update in typewriter order.
    W: weight matrix
    s: initial state vector
    Returns: final state vector
    """
    n = len(s)
    s = s.copy()
    for _ in range(max_iter):
        prev = s.copy()
        for i in range(n):
            h = np.dot(W[i], s)
            s[i] = signum(h)
        if np.array_equal(s, prev):
            break
    return s

def pattern_index(final, patterns):
    """
    Classify the final pattern according to the scheme.
    Returns: index (mu), -mu (inverted), or 6 (other)
    """
    for idx, p in enumerate(patterns, 1):
        if np.array_equal(final, p):
            return idx
        if np.array_equal(final, -p):
            return -idx
    return 6

# --- Prepare patterns ---
patterns = [flatten_pattern(x1), flatten_pattern(x2), flatten_pattern(x3), flatten_pattern(x4), flatten_pattern(x5)]
W = hebb_weight_matrix(patterns)

# --- Example: Feed a test pattern ---
if __name__ == "__main__":
    # Feed the provided pattern
    input_pattern = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, -1, -1, -1, -1, 1, 1, 1],
        [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1],
        [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
        [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
        [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
        [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
        [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
        [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
        [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
        [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
        [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
        [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
        [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1],
        [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    ]
    test_pattern = flatten_pattern(input_pattern)
    result = hopfield_async_update(W, test_pattern)
    idx = pattern_index(result, patterns)
    print("Converged to pattern index:", idx)
    print("Steady state pattern in input format:")
    output_matrix = result.reshape(16, 10).tolist()
    print("[")
    for i, row in enumerate(output_matrix):
        print("  [" + ", ".join(str(v) for v in row) + "]" + ("," if i < 15 else ""))
    print("]")

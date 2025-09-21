# --- Hopfield Network Functions ---
from typing import List, Any
import numpy as np
from data import x1, x2, x3, x4, x5, x_input1, x_input2, x_input3


def flatten_pattern(matrix: List[List[int]]) -> np.ndarray:
    return np.array(matrix).flatten()


def hebb_weight_matrix(patterns: List[np.ndarray]) -> np.ndarray:
    patterns = np.array(patterns)  # type:ignore
    n: int = patterns.shape[1]  # type:ignore
    W: np.ndarray = np.zeros((n, n))  # type:ignore
    for p in patterns:
        W += np.outer(p, p)  # type:ignore
    W /= len(patterns)  # type:ignore
    np.fill_diagonal(W, 0)
    return W


def signum(x: float) -> int:
    return 1 if x >= 0 else -1


def hopfield_async_update(
    W: np.ndarray, s: np.ndarray, max_iter: int = 100
) -> np.ndarray:
    n: int = len(s)
    s = s.copy()
    for _ in range(max_iter):
        prev: np.ndarray = s.copy()
        for i in range(n):
            h: float = np.dot(W[i], s)
            s[i] = signum(h)
        if np.array_equal(s, prev):
            break
    return s


def pattern_index(final: np.ndarray, patterns: List[np.ndarray]) -> int:
    for idx, p in enumerate(patterns, 1):
        if np.array_equal(final, p):
            return idx
        if np.array_equal(final, -p):
            return -idx
    return 6


patterns: List[np.ndarray] = [
    flatten_pattern(x1),
    flatten_pattern(x2),
    flatten_pattern(x3),
    flatten_pattern(x4),
    flatten_pattern(x5),
]
W: np.ndarray = hebb_weight_matrix(patterns)

if __name__ == "__main__":
    input_pattern: Any = x_input3
    test_pattern: np.ndarray = flatten_pattern(input_pattern)
    result: np.ndarray = hopfield_async_update(W, test_pattern)
    idx: int = pattern_index(result, patterns)
    print("Converged to pattern index:", idx)
    print("Steady state pattern in input format:")
    output_matrix: List[List[int]] = result.reshape(16, 10).tolist()
    print("[")
    for i, row in enumerate(output_matrix):
        print("  [" + ", ".join(str(v) for v in row) + "]" + ("," if i < 15 else ""))
    print("]")

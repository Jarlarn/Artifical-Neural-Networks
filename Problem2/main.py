# --- Hopfield Network Functions ---
from typing import List, Any
import numpy as np
from data import x1, x2, x3, x4, x5, x_input1, x_input2, x_input3


def flatten_pattern(matrix: List[List[int]]) -> np.ndarray:
    return np.array(matrix).flatten()


def hebb_weight_matrix(patterns: List[np.ndarray]) -> np.ndarray:
    patterns = np.array(patterns)  # type:ignore
    n: int = patterns.shape[1]  # type:ignore
    weight_matrix: np.ndarray = np.zeros((n, n))  # type:ignore
    for p in patterns:
        weight_matrix += np.outer(p, p)
    weight_matrix /= len(patterns)
    np.fill_diagonal(weight_matrix, 0)
    return weight_matrix


def signum(x: float) -> int:
    return 1 if x >= 0 else -1


def hopfield_async_update(
    weight_matrix: np.ndarray, s: np.ndarray, max_iter: int = 100
) -> np.ndarray:
    n: int = len(s)
    s = s.copy()
    for _ in range(max_iter):
        prev: np.ndarray = s.copy()
        for i in range(n):
            h: float = np.dot(weight_matrix[i], s)
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


data_patterns: List[np.ndarray] = [
    flatten_pattern(x1),
    flatten_pattern(x2),
    flatten_pattern(x3),
    flatten_pattern(x4),
    flatten_pattern(x5),
]
weight_matrix: np.ndarray = hebb_weight_matrix(data_patterns)

if __name__ == "__main__":
    input_pattern: Any = x_input3
    test_pattern: np.ndarray = flatten_pattern(input_pattern)
    result: np.ndarray = hopfield_async_update(weight_matrix, test_pattern)
    idx: int = pattern_index(result, data_patterns)
    print("Converged to pattern index:", idx)
    print("Steady state pattern in input format:")
    output_matrix: List[List[int]] = result.reshape(16, 10).tolist()
    print("[")
    for i, row in enumerate(output_matrix):
        print("  [" + ", ".join(str(v) for v in row) + "]" + ("," if i < 15 else ""))
    print("]")

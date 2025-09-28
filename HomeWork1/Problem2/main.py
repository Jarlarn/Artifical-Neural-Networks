# This code has been written thanks to documentation on numpy, stackoverflow, course book and AI.
from typing import List, Optional
import numpy as np
from data import x1, x2, x3, x4, x5, x_input1, x_input2, x_input3


class HopfieldNetwork:
    def __init__(self, patterns: List[np.ndarray]) -> None:
        self.patterns: np.ndarray = np.array(patterns)
        self.n: int = self.patterns.shape[1]
        self.weight_matrix: np.ndarray = self._compute_weight_matrix()

    @staticmethod
    def flatten_pattern(matrix: List[List[int]]) -> np.ndarray:
        return np.array(matrix).flatten()

    def _compute_weight_matrix(self) -> np.ndarray:
        weight_matrix: np.ndarray = np.zeros((self.n, self.n))
        for p in self.patterns:
            weight_matrix += np.outer(p, p)
        weight_matrix /= len(self.patterns)
        np.fill_diagonal(weight_matrix, 0)
        return weight_matrix

    @staticmethod
    def signum(x: float) -> int:
        return 1 if x >= 0 else -1

    def async_update(self, s: np.ndarray, max_iter: int = 100) -> np.ndarray:
        s = s.copy()
        for _ in range(max_iter):
            prev = s.copy()
            for i in range(self.n):
                h = np.dot(self.weight_matrix[i], s)
                s[i] = self.signum(h)
            if np.array_equal(s, prev):
                break
        return s

    def pattern_index(self, final: np.ndarray) -> Optional[int]:
        for idx, p in enumerate(self.patterns, 1):
            if np.array_equal(final, p):
                return idx
            if np.array_equal(final, -p):
                return -idx
        return None

    def run(self, input_pattern: List[List[int]]) -> None:
        test_pattern = self.flatten_pattern(input_pattern)
        result = self.async_update(test_pattern)
        idx = self.pattern_index(result)
        print("Converged to pattern index:", idx if idx is not None else "No match")
        print("Steady state pattern in input format:")
        output_matrix = result.reshape(16, 10).tolist()
        print("[")
        for i, row in enumerate(output_matrix):
            print(
                "  [" + ", ".join(str(v) for v in row) + "]" + ("," if i < 15 else "")
            )
        print("]")


if __name__ == "__main__":
    data_patterns = [
        HopfieldNetwork.flatten_pattern(x1),
        HopfieldNetwork.flatten_pattern(x2),
        HopfieldNetwork.flatten_pattern(x3),
        HopfieldNetwork.flatten_pattern(x4),
        HopfieldNetwork.flatten_pattern(x5),
    ]
    network = HopfieldNetwork(data_patterns)
    network.run(x_input1)
    network.run(x_input2)
    network.run(x_input3)

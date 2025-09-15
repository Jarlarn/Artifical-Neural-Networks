# src/boolean_function.py
import itertools
from typing import Iterator
import numpy as np


class BooleanFunction:
    """
    Represents an n-dimensional Boolean function f: {-1,+1}^n -> {-1,+1}.
    """

    def __init__(self, n: int, outputs: np.ndarray = None):
        """
        Initialize a Boolean function.
        - n: number of inputs
        - outputs: array of length 2^n, with values in {-1, +1}
          If outputs is None, generate random outputs.
        """
        if outputs is None:
            outputs = np.random.choice([-1, 1], size=2**n)
        assert len(outputs) == 2**n, "Outputs must have length 2^n."
        self.n = n
        self.X = self._generate_all_inputs(n)
        self.Y = np.array(outputs)

    @staticmethod
    def _generate_all_inputs(n: int) -> np.ndarray:
        """Generate all input vectors of length n in {-1,+1}^n."""
        return np.array(list(itertools.product([-1, 1], repeat=n)))

    @classmethod
    def random(cls, n: int) -> "BooleanFunction":
        """Generate a random Boolean function."""
        outputs = np.random.choice([-1, 1], size=2**n)
        return cls(n, outputs)

    @classmethod
    def enumerate(cls, n: int) -> Iterator["BooleanFunction"]:
        """Yield all Boolean functions for n inputs"""
        for outputs in itertools.product([-1, 1], repeat=2**n):
            yield cls(n, np.array(outputs))

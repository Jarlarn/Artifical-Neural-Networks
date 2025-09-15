from typing import Set, Tuple
from src.boolean_function import BooleanFunction
from src.perceptron import Perceptron


class BooleanFunctionSampler:
    """
    Samples Boolean functions and estimates fraction linearly separable.
    """

    def __init__(self, n: int):
        self.n = n
        self.seen: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]] = set()

    def sample_unique(self, num_samples: int) -> float:
        """Sample unique Boolean functions and return fraction linearly separable."""
        seperable_functions = 0
        unique_funcs = 0

        for _ in range(num_samples):
            bf = BooleanFunction.random(self.n)
            key = (tuple(map(tuple, bf.X)), tuple(bf.Y))  # type: ignore
            if key in self.seen:
                continue
            self.seen.add(key)  # type: ignore
            unique_funcs += 1

            if Perceptron.is_linearly_separable(bf.X, bf.Y):
                seperable_functions += 1

        return seperable_functions / unique_funcs if unique_funcs > 0 else 0.0

import itertools
import numpy as np


class BooleanFunction:
    """
    Represents an n-dimensional Boolean function f: {-1,+1}^n -> {-1,+1}.
    """

    def __init__(self, n: int, outputs: np.ndarray = None):  # type: ignore
        """
        Initialize a Boolean function.
        - n: number of inputs
        - outputs: array of length 2^n, with values in {-1, +1}
          If outputs is None, generate random outputs.
        """
        self.n = n
        if outputs is None:  # type: ignore
            outputs = np.random.choice([-1, 1], size=2**n)
        assert len(outputs) == 2**n, "Outputs must have length 2^n."
        self.X = self._generate_all_inputs()
        self.Y = np.array(outputs)

    def _generate_all_inputs(self) -> np.ndarray:
        """Generate all input vectors of length n in {-1,+1}^n."""
        grid = np.array(np.meshgrid(*[[-1, 1]] * self.n)).T.reshape(-1, self.n)
        return grid

    @classmethod
    def random(cls, n: int) -> "BooleanFunction":
        """Generate a random Boolean function."""
        rng = np.random.default_rng()
        outputs = rng.choice([-1, 1], size=2**n)
        return cls(n, outputs)

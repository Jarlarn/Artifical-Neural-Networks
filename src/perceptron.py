import numpy as np
from boolean_function import BooleanFunction


class Perceptron:
    def __init__(self, n_inputs: int, threshold: float = 0.0) -> None:
        self.n_inputs: int = n_inputs
        self.threshold: float = threshold
        self.weight_matrix: np.ndarray = np.random.normal(
            0.0, 1.0 / np.sqrt(n_inputs), size=n_inputs
        )

    def activation_function(self, b: float) -> int:
        """Signum function."""
        if b > 0:
            return 1
        elif b < 0:
            return -1
        else:
            return 0

    def compute_b(self, x: np.ndarray) -> float:
        """Compute activation b = sum_j w_j * x_j - theta."""
        return float(np.sum(self.weight_matrix * x) - self.threshold)

    def train(
        self,
        x: np.ndarray,
        t: np.ndarray,
        eta: float = 0.05,
        max_epochs: int = 20,
    ) -> None:
        for _ in range(max_epochs):
            for mu in range(x.shape[0]):
                x_mu: np.ndarray = x[mu]
                t_mu: int = t[mu]
                o_mu: int = self.activation_function(
                    np.dot(self.weight_matrix, x_mu) - self.threshold
                )
                error: int = t_mu - o_mu
                if error != 0:
                    self.weight_matrix += eta * error * x_mu
                    self.threshold -= eta * error

    def predict(self, x: np.ndarray) -> int:
        """Predict using the learned perceptron (sign activation)."""
        b: float = self.compute_b(x)
        return self.activation_function(b)


def is_linearly_separable(
    X: np.ndarray,
    T: np.ndarray,
    n_inputs: int,
    eta: float = 0.05,
    max_epochs: int = 20,
) -> bool:
    p: Perceptron = Perceptron(n_inputs=n_inputs)
    p.train(X, T, eta=eta, max_epochs=max_epochs)
    predictions: np.ndarray = np.array([p.predict(xi) for xi in X])
    return np.array_equal(predictions, T)


def estimate_fraction(n: int, num_samples: int) -> float:
    count_separable: int = 0
    for _ in range(num_samples):
        bf: BooleanFunction = BooleanFunction.random(n)
        X: np.ndarray = bf.X
        T: np.ndarray = bf.Y
        if is_linearly_separable(X, T, n):
            count_separable += 1
    return count_separable / num_samples


def sample_unique_linearly_separable(n: int, num_samples: int) -> int:
    seen = set()  # type: ignore
    count_separable = 0
    unique_boolean_funcs = 0
    for _ in range(num_samples):
        bf = BooleanFunction.random(n)
        # Use the outputs as a tuple for uniqueness
        key = (tuple(map(tuple, bf.X)), tuple(bf.Y))
        if key in seen:
            continue  # Skip duplicates
        unique_boolean_funcs += 1

        seen.add(key)  # type: ignore
        X = bf.X
        T = bf.Y
        if is_linearly_separable(X, T, n):
            count_separable += 1
    return count_separable


# for n in range(2, 6):
#     frac: float = estimate_fraction(n, num_samples=1000)
#     print(f"n={n}: fraction linearly separable ≈ {frac:.4f}")

for n in range(2, 6):
    num_samples = 10000
    num_separable = sample_unique_linearly_separable(n, num_samples)
    print(
        f"n={n}: unique linearly separable Boolean functions found: {num_separable} out of {num_samples}"
    )

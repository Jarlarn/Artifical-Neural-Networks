import numpy as np


class Perceptron:
    """
    Simple perceptron with sign activation for Boolean function separability.
    """

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
        return float(np.dot(self.weight_matrix, x) - self.threshold)

    def train(
        self,
        X: np.ndarray,
        T: np.ndarray,
        eta: float = 0.05,
        max_epochs: int = 20,
    ) -> None:
        """Train perceptron on dataset (delta rule)."""
        for _ in range(max_epochs):
            for x_mu, t_mu in zip(X, T):
                o_mu = self.activation_function(self.compute_b(x_mu))
                error: int = t_mu - o_mu
                if error != 0:
                    self.weight_matrix += eta * error * x_mu
                    self.threshold -= eta * error

    def predict(self, x: np.ndarray) -> int:
        """Predict using the learned perceptron (sign activation)."""
        b: float = self.compute_b(x)
        return self.activation_function(b)

    def predict_all(self, X: np.ndarray) -> np.ndarray:
        """Predict for all input patterns."""
        return np.array([self.predict(x) for x in X])

    @staticmethod
    def is_linearly_separable(
        X: np.ndarray,
        T: np.ndarray,
        eta: float = 0.05,
        max_epochs: int = 20,
    ) -> bool:
        """Check if dataset (X, T) is linearly separable with a perceptron."""
        p = Perceptron(n_inputs=X.shape[1])
        p.train(X, T, eta=eta, max_epochs=max_epochs)
        return np.array_equal(p.predict_all(X), T)

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# pylint: disable=invalid-name

FloatArray = NDArray[np.float64]


class Data:
    """Load CSV with 2 inputs and target ±1 in third column."""

    def __init__(self, path: str) -> None:
        self.training_set: FloatArray = np.loadtxt(
            path, delimiter=",", dtype=np.float64
        )
        self.x: FloatArray = self.training_set[:, 0:2]
        self.t: FloatArray = self.training_set[:, 2]


class NeuralNetwork:

    def __init__(
        self,
        n_inputs: int,
        M1: int,
        M2: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()
        self.M1 = M1
        self.M2 = M2
        self.w1: FloatArray = self.rng.normal(
            0, 1 / np.sqrt(n_inputs), size=(M1, n_inputs)
        )
        self.theta1: FloatArray = np.zeros(M1, dtype=np.float64)
        self.w2: FloatArray = self.rng.normal(0, 1 / np.sqrt(M1), size=(M2, M1))
        self.theta2: FloatArray = np.zeros(M2, dtype=np.float64)
        self.w3: FloatArray = self.rng.normal(0, 1 / np.sqrt(M2), size=M2)
        self.theta3: float = 0.0

    def forward(self, x: FloatArray) -> tuple[FloatArray, FloatArray, float]:
        """Forward pass for a single sample x (shape (2,))."""
        net1: FloatArray = np.dot(self.w1, x) - self.theta1
        V1: FloatArray = np.tanh(net1)
        net2: FloatArray = np.dot(self.w2, V1) - self.theta2
        V2: FloatArray = np.tanh(net2)
        net3: float = float(np.dot(self.w3, V2) - self.theta3)
        O: float = float(np.tanh(net3))
        return V1, V2, O

    def train_epoch(self, X: np.ndarray, T: np.ndarray, eta: float) -> None:
        idx = np.arange(X.shape[0])
        self.rng.shuffle(idx)
        for i in idx:
            x = X[i]
            t = T[i]
            V1, V2, O = self.forward(x)
            delta3 = (O - t) * (1 - O**2)
            delta2 = (1 - V2**2) * (self.w3 * delta3)
            delta1 = (1 - V1**2) * (self.w2.T @ delta2)
            self.w3 -= eta * delta3 * V2
            self.theta3 += eta * delta3
            self.w2 -= eta * np.outer(delta2, V1)
            self.theta2 += eta * delta2
            self.w1 -= eta * np.outer(delta1, x)
            self.theta1 += eta * delta1

    # def train_epoch(self, X: np.ndarray, T: np.ndarray, eta: float):
    #     return


def main():
    training_data = Data("training_set.csv")
    validation_data = Data("validation_set.csv")  # unused yet

    M1, M2 = 6, 4
    eta = 0.01
    epochs = 200  # unused yet

    rng = np.random.default_rng(42)
    net = NeuralNetwork(n_inputs=2, M1=M1, M2=M2, rng=rng)
    net.train_epoch(training_data.x, training_data.t, eta)


if __name__ == "__main__":
    main()

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pathlib import Path

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
    """Two-hidden-layer tanh MLP with scalar tanh output."""

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
        # Fan-in style init
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
        net1: FloatArray = self.w1 @ x - self.theta1
        V1: FloatArray = np.tanh(net1)
        net2: FloatArray = self.w2 @ V1 - self.theta2
        V2: FloatArray = np.tanh(net2)
        net3: float = float(self.w3 @ V2 - self.theta3)
        O: float = float(np.tanh(net3))
        return V1, V2, O

    def train_epoch(self, X: FloatArray, T: FloatArray, eta: float) -> None:
        """One stochastic epoch (pattern shuffling)."""
        idx = np.arange(X.shape[0])  # type: ignore
        self.rng.shuffle(idx)  # type: ignore
        for i in idx:  # type: ignore
            x: FloatArray = X[i]
            t: float = float(T[i])
            V1, V2, O = self.forward(x)
            # Backprop deltas
            delta3: float = (O - t) * (1 - O**2)
            delta2: FloatArray = (1 - V2**2) * (self.w3 * delta3)
            delta1: FloatArray = (1 - V1**2) * (self.w2.T @ delta2)
            # Gradient descent updates
            self.w3 -= eta * delta3 * V2
            self.theta3 += eta * delta3
            self.w2 -= eta * np.outer(delta2, V1)
            self.theta2 += eta * delta2
            self.w1 -= eta * np.outer(delta1, x)
            self.theta1 += eta * delta1

    def predict_raw(self, X: FloatArray) -> FloatArray:
        """Return raw tanh outputs."""
        return np.array([self.forward(x)[2] for x in X], dtype=np.float64)

    def predict_sign(self, X: FloatArray) -> FloatArray:
        """Return ±1 predictions."""
        return np.where(self.predict_raw(X) >= 0.0, 1.0, -1.0)

    def classification_error(self, X: FloatArray, T: FloatArray) -> float:
        """C = (1/2) mean |sign(O) - t|."""
        preds = self.predict_sign(X)
        return 0.5 * float(np.mean(np.abs(preds - T)))

    def save_parameters(self, out_dir: str | Path = ".") -> None:
        """Save weights and thresholds as CSV files (comma separated)."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        np.savetxt(out_path / "w1.csv", self.w1, delimiter=",")
        np.savetxt(out_path / "w2.csv", self.w2, delimiter=",")
        np.savetxt(out_path / "w3.csv", self.w3.reshape(-1, 1), delimiter=",")
        np.savetxt(out_path / "t1.csv", self.theta1.reshape(-1, 1), delimiter=",")
        np.savetxt(out_path / "t2.csv", self.theta2.reshape(-1, 1), delimiter=",")
        np.savetxt(out_path / "t3.csv", np.array([[self.theta3]]), delimiter=",")


def main() -> None:
    train = Data("training_set.csv")
    val = Data("validation_set.csv")

    net = NeuralNetwork(n_inputs=2, M1=6, M2=4, rng=np.random.default_rng())

    eta: float = 0.01
    epochs: int = 300
    target_C: float = 0.12

    for ep in range(1, epochs + 1):
        net.train_epoch(train.x, train.t, eta)
        if ep % 10 == 0 or ep == 1:
            C_val = net.classification_error(val.x, val.t)
            print(f"Epoch {ep:4d}  Validation C = {C_val:.4f}")
            if C_val < target_C:
                print("Target reached.")
                break

    final_C = net.classification_error(val.x, val.t)
    print(f"Final validation C = {final_C:.4f}")
    net.save_parameters(".")


if __name__ == "__main__":
    main()

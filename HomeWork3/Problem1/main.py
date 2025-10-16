import numpy as np
import matplotlib.pyplot as plt


# -------------------------------
# Data loader
# -------------------------------
class Data:
    def __init__(self, path):
        training_data = np.loadtxt(path, delimiter=",")
        self.x = training_data[:, 0]
        self.y = training_data[:, 1]
        self.z = training_data[:, 2]


# -------------------------------
# Discrete-time Reservoir Network
# -------------------------------
class ReservoirNetwork:
    def __init__(self, Ni, Nr, spectral_radius=0.9, seed=None) -> None:
        """
        Ni : number of input dimensions
        Nr : number of reservoir neurons
        spectral_radius : scaling factor for reservoir weights
        """
        self.Ni = Ni
        self.Nr = Nr
        self.r = np.zeros(self.Nr)
        self.input_weights = None
        self.reservoir_weights = None
        self.output_weights = None
        self._init_weights(spectral_radius, seed)

    def _init_weights(self, spectral_radius, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Input weight variance
        variance_in = 0.002
        self.input_weights = np.random.normal(
            0.0, np.sqrt(variance_in), size=(self.Nr, self.Ni)
        )

        # Random reservoir weights
        variance_reservoir = 1.0 / self.Nr
        W = np.random.normal(0.0, np.sqrt(variance_reservoir), size=(self.Nr, self.Nr))

        # Scale to achieve desired spectral radius
        eigvals = np.linalg.eigvals(W)
        rho = np.max(np.abs(eigvals))
        self.reservoir_weights = W * (spectral_radius / rho)

    def update_reservoir(self, xk):
        """
        xk : input vector at current time step, shape (Ni,)
        Updates internal reservoir state.
        """
        self.r = np.tanh(self.reservoir_weights @ self.r + self.input_weights @ xk)
        return self.r

    def train_output_weights(self, reservoir_states, target_outputs, ridge_param=1e-2):
        """
        Train output weights using ridge regression.
        reservoir_states: shape (T, Nr)
        target_outputs: shape (T, n_outputs)
        """
        T, Nr = reservoir_states.shape
        I = np.eye(Nr)
        RtR = reservoir_states.T @ reservoir_states
        RtY = reservoir_states.T @ target_outputs
        # Ridge regression: W_out = (RᵀR + λI)⁻¹ RᵀY
        self.output_weights = np.linalg.solve(
            RtR + ridge_param * I, RtY
        ).T  # shape (n_outputs, Nr)

    def calculate_reservoir_output(self, reservoir_input):
        """
        reservoir_input: current reservoir state (Nr,)
        Returns network output (n_outputs,)
        """
        return self.output_weights @ reservoir_input

    def predict_future(self, initial_input, n_steps=500):
        """
        Predict n_steps into the future, starting from initial_input.
        Returns array of shape (n_steps, n_outputs).
        """
        predictions = []
        O_k = initial_input.copy()

        # Reset reservoir to avoid influence from previous data
        self.r[:] = 0.0

        for _ in range(n_steps):
            # Update reservoir with previous output as input (closed loop)
            self.r = np.tanh(self.reservoir_weights @ self.r + self.input_weights @ O_k)
            # Compute new output
            O_k = self.output_weights @ self.r
            predictions.append(O_k.copy())

        return np.array(predictions)


# -------------------------------
# Main program
# -------------------------------
if __name__ == "__main__":
    # 1. Load training data
    data = Data("training-set.csv")

    # 2. Initialize reservoir
    network = ReservoirNetwork(Ni=3, Nr=500, spectral_radius=0.9, seed=3211)

    # 3. Run training data through the reservoir
    reservoir_states = []
    for i in range(len(data.x) - 1):
        xk = np.array([data.x[i], data.y[i], data.z[i]])
        r = network.update_reservoir(xk)
        reservoir_states.append(r.copy())

    reservoir_states = np.array(reservoir_states)
    target_outputs = np.column_stack([data.x[1:], data.y[1:], data.z[1:]])

    # 4. Train readout weights
    network.train_output_weights(reservoir_states, target_outputs, ridge_param=1e-2)

    # 5. Load test data and update reservoir (open-loop)
    test_data = Data("test-set.csv")
    for i in range(len(test_data.x)):
        xk = np.array([test_data.x[i], test_data.y[i], test_data.z[i]])
        network.update_reservoir(xk)

    # 6. Predict future (closed-loop, generative mode)
    last_output = np.array([test_data.x[-1], test_data.y[-1], test_data.z[-1]])
    predictions_xyz = network.predict_future(last_output, n_steps=500)

    # 7. Save y-component of predictions
    np.savetxt("prediction.csv", predictions_xyz[:, 1], delimiter=",")

    # 8. Optional: visualize predicted trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(predictions_xyz[:, 0], predictions_xyz[:, 1], predictions_xyz[:, 2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.title("Predicted Lorenz Attractor (Reservoir Output)")
    plt.show()

import numpy as np
import matplotlib.pyplot as plt


# -------------------------------
# Data loader & normalization
# -------------------------------
class Data:
    def __init__(self, path):
        arr = np.loadtxt(path, delimiter=",")
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError("CSV must have at least 3 columns (x,y,z).")
        self.x = arr[:, 0]
        self.y = arr[:, 1]
        self.z = arr[:, 2]

    def to_array(self):
        return np.column_stack([self.x, self.y, self.z])


# -------------------------------
# Discrete-time Reservoir Network (with fixes)
# -------------------------------
class ReservoirNetwork:
    def __init__(
        self,
        Ni,
        Nr,
        spectral_radius=0.9,
        input_scaling=1.0,
        bias_scaling=0.2,
        seed=None,
    ):
        """
        Ni : number of input dims (3)
        Nr : number of reservoir neurons
        spectral_radius : desired spectral radius for W_res
        input_scaling : scale applied to W_in (important!)
        bias_scaling : scale of constant bias input
        """
        self.Ni = Ni
        self.Nr = Nr
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.r = np.zeros(self.Nr)
        self.W_in = None  # shape (Nr, Ni)
        self.W_res = None  # shape (Nr, Nr)
        self.W_out = None  # shape (n_outputs, Nr + 1) includes bias
        self._init_weights(seed)

    def _init_weights(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Input weights; scaled by input_scaling
        W_in = np.random.normal(0.0, 1.0, size=(self.Nr, self.Ni))
        # normalize columns to unit std then scale
        W_in = W_in / np.std(W_in)
        self.W_in = W_in * self.input_scaling

        # Reservoir weights: initialize and scale to desired spectral radius
        W = np.random.normal(0.0, 1.0 / np.sqrt(self.Nr), size=(self.Nr, self.Nr))
        eigvals = np.linalg.eigvals(W)
        rho = np.max(np.abs(eigvals))
        if rho == 0:
            raise RuntimeError("Initialized reservoir has zero spectral radius.")
        self.W_res = W * (self.spectral_radius / rho)

    def reset_state(self):
        self.r[:] = 0.0

    def step(self, u):
        """
        Discrete update (no leak). u shape (Ni,)
        """
        pre = self.W_res @ self.r + self.W_in @ u
        self.r = np.tanh(pre)
        return self.r

    def collect_states(self, inputs, washout=100):
        """
        Run reservoir on inputs, discard first `washout` states.
        inputs: array shape (T, Ni)
        returns: states array shape (T - washout, Nr)
        """
        T = inputs.shape[0]
        states = []
        self.reset_state()
        for t in range(T):
            u = inputs[t]
            self.step(u)
            if t >= washout:
                states.append(self.r.copy())
        return np.array(states)

    def train_output_weights(self, reservoir_states, target_outputs, ridge_param=1e-6):
        """
        reservoir_states: (T_eff, Nr)
        target_outputs: (T_eff, n_outputs)
        We add a bias column of ones.
        """
        T_eff = reservoir_states.shape[0]
        if T_eff != target_outputs.shape[0]:
            raise ValueError("reservoir_states and target_outputs length mismatch.")

        # Add bias column
        R = np.hstack([reservoir_states, np.ones((T_eff, 1))])  # shape (T_eff, Nr+1)
        RtR = R.T @ R
        RtY = R.T @ target_outputs
        # Solve (RtR + ridge I) W^T = RtY  => W = ( ... )^T
        I = np.eye(RtR.shape[0])
        W = np.linalg.solve(RtR + ridge_param * I, RtY)
        self.W_out = W.T  # shape (n_outputs, Nr+1)

    def output_from_state(self):
        """
        Compute current output from self.r using bias.
        returns vector shape (n_outputs,)
        """
        if self.W_out is None:
            raise RuntimeError("Output weights not trained yet.")
        vec = np.concatenate([self.r, [1.0]])  # (Nr+1,)
        return self.W_out @ vec

    def predict_closed_loop(self, initial_output, n_steps=500):
        """
        Closed-loop prediction:
        - initial_output: shape (Ni,) a last observed x,y,z (normalized if trained that way)
        - uses previous prediction as next input
        returns array (n_steps, n_outputs)
        """
        if self.W_out is None:
            raise RuntimeError("Output weights not trained yet.")

        preds = []
        self.reset_state()  # important to start fresh (or warm up if desired)
        O_k = initial_output.copy()

        for _ in range(n_steps):
            # feed previous output as next input
            self.step(O_k)
            # compute new output
            vec = np.concatenate([self.r, [1.0]])
            O_k = self.W_out @ vec
            preds.append(O_k.copy())

        return np.array(preds)


# -------------------------------
# Utility: normalize and denormalize (fit on training)
# -------------------------------
class Normalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, arr):
        self.mean = np.mean(arr, axis=0)
        self.std = np.std(arr, axis=0)
        # avoid zero std
        self.std[self.std == 0] = 1.0

    def transform(self, arr):
        return (arr - self.mean) / self.std

    def inverse(self, arr):
        return arr * self.std + self.mean


# -------------------------------
# Main (script)
# -------------------------------
if __name__ == "__main__":
    # PARAMETERS
    NR = 500
    INPUT_SCALING = 0.5  # increase from near-zero; tune if needed
    SPECTRAL_RADIUS = 0.9
    WASHOUT = 100
    RIDGE = 1e-6
    SEED = 3211

    # 1) Load training data
    train = Data("training-set.csv")
    X_train = train.to_array()  # shape (T, 3)
    T = X_train.shape[0]

    # 2) Normalize (fit on training set)
    norm = Normalizer()
    norm.fit(X_train)
    X_train_n = norm.transform(X_train)

    # 3) Initialize reservoir
    net = ReservoirNetwork(
        Ni=3,
        Nr=NR,
        spectral_radius=SPECTRAL_RADIUS,
        input_scaling=INPUT_SCALING,
        seed=SEED,
    )

    # 4) Run reservoir on training inputs and collect states (with washout)
    states = net.collect_states(X_train_n, washout=WASHOUT)  # shape (T - washout, Nr)
    # Targets are one-step-ahead predictions (also normalized)
    targets = X_train_n[
        WASHOUT + 1 : T
    ]  # because we collected from t>=washout, state at t corresponds to input at t
    # Note: states length = T - washout; but since we want to predict next step, align accordingly:
    # We'll use state at time t to predict input at time t+1. So drop final state to match targets length.
    states = states[:-1, :]
    assert states.shape[0] == targets.shape[0], "Alignment mismatch."

    # 5) Train readout
    net.train_output_weights(states, targets, ridge_param=RIDGE)

    # 6) Load test data, normalize, and warm up reservoir on test set (open-loop)
    test = Data("test-set.csv")
    X_test = test.to_array()
    X_test_n = norm.transform(X_test)

    # warm up with test inputs (do not collect)
    net.reset_state()
    for t in range(X_test_n.shape[0]):
        net.step(X_test_n[t])

    # 7) Predict closed-loop for n steps using last test point as seed (normalized)
    last_obs = X_test_n[-1]
    preds_n = net.predict_closed_loop(
        last_obs, n_steps=500
    )  # shape (500, 3) in normalized space

    # 8) Denormalize predictions
    preds = norm.inverse(preds_n)

    # 9) Save y-component
    np.savetxt("prediction.csv", preds[:, 1], delimiter=",")

    # 10) Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(preds[:, 0], preds[:, 1], preds[:, 2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.title("Predicted Lorenz Attractor (Reservoir Output)")
    plt.show()

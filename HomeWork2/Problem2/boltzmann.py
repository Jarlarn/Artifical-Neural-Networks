import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def xor_patterns():
    pats = np.array([[-1, -1, -1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=int)
    probs = np.full(4, 0.25)
    return pats, probs


def enumerate_visible_states():
    return np.array(
        [[(1 if (i >> b) & 1 else -1) for b in range(3)] for i in range(8)], dtype=int
    )


class RBM:
    def __init__(self, n_visible, n_hidden, seed=0, scale=0.1):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = rng.normal(0, scale, (n_visible, n_hidden))
        self.a = np.zeros(n_visible)
        self.b = np.zeros(n_hidden)

    # P(h_j=+1|v) = σ(2(b_j + Σ_i v_i W_ij))
    def sample_h(self, v):
        act = 2 * (self.b + v @ self.W)
        p = sigmoid(act)
        h = np.where(self.rng.random(p.shape) < p, 1, -1)
        return h, p

    # P(v_i=+1|h) = σ(2(a_i + Σ_j W_ij h_j))
    def sample_v(self, h):
        act = 2 * (self.a + h @ self.W.T)
        p = sigmoid(act)
        v = np.where(self.rng.random(p.shape) < p, 1, -1)
        return v, p

    # Exact unnormalized P(v)
    def unnormalized_pv(self, V):
        linear = V @ self.a
        hidden_terms = np.prod(2 * np.cosh(self.b + V @ self.W), axis=1)
        return np.exp(linear) * hidden_terms

    def exact_visible_distribution(self):
        V = enumerate_visible_states()
        u = self.unnormalized_pv(V)
        P = u / u.sum()
        return V, P


def kl_divergence(p_target_dict, p_model_dict, eps=1e-12):
    kl = 0.0
    for k, pt in p_target_dict.items():
        pm = max(p_model_dict.get(k, 0.0), eps)
        kl += pt * (np.log(pt + eps) - np.log(pm))
    return kl


def dict_from(V, P):
    return {tuple(v.tolist()): float(p) for v, p in zip(V, P)}


def step_by_step_demo(epochs=5, k=1, lr=0.05):
    print("=== RBM XOR DEBUG DEMO ===")
    data, data_probs = xor_patterns()
    rbm = RBM(n_visible=3, n_hidden=2, seed=123)

    print("Initial W:\n", rbm.W)
    print("Initial a:", rbm.a)
    print("Initial b:", rbm.b)
    print()

    for epoch in range(1, epochs + 1):
        # Single full-batch CD-k (batch = all four XOR patterns)
        batch = data
        m = batch.shape[0]

        # Positive phase
        h0, ph0 = rbm.sample_h(batch)  # sampled h0 and probs
        Eh0 = 2 * ph0 - 1  # expectations for ±1 units
        pos_W_sample = batch.T @ h0 / m
        pos_W_expect = batch.T @ Eh0 / m  # lower variance version
        pos_a = batch.mean(axis=0)
        pos_b_sample = h0.mean(axis=0)
        pos_b_expect = Eh0.mean(axis=0)

        # Gibbs chain k steps (starting from data)
        v_neg = batch.copy()
        h_neg = h0.copy()
        for _ in range(k):
            v_neg, _ = rbm.sample_v(h_neg)
            h_neg, ph_neg = rbm.sample_h(v_neg)
        Eh_neg = 2 * ph_neg - 1

        neg_W_sample = v_neg.T @ h_neg / m
        neg_W_expect = v_neg.T @ Eh_neg / m
        neg_a = v_neg.mean(axis=0)
        neg_b_sample = h_neg.mean(axis=0)
        neg_b_expect = Eh_neg.mean(axis=0)

        # Choose expectation-based update (comment to use sampled)
        dW = pos_W_expect - neg_W_expect
        da = pos_a - neg_a
        db = pos_b_expect - neg_b_expect

        rbm.W += lr * dW
        rbm.a += lr * da
        rbm.b += lr * db

        print(f"--- Epoch {epoch} ---")
        print("Batch (data) v:\n", batch)
        print("Positive phase hidden probs ph0:\n", ph0)
        print("Positive phase Eh0:\n", Eh0)
        print("Negative sample v_neg:\n", v_neg)
        print("Negative phase hidden probs ph_neg:\n", ph_neg)
        print("Eh_neg:\n", Eh_neg)
        print("pos_W (expect):\n", pos_W_expect)
        print("neg_W (expect):\n", neg_W_expect)
        print("dW:\n", dW)
        print("da:", da, " db:", db)
        print("Updated W:\n", rbm.W)
        print("Updated a:", rbm.a)
        print("Updated b:", rbm.b)

        # Exact model distribution after update
        V_all, P_all = rbm.exact_visible_distribution()
        model_dict = dict_from(V_all, P_all)
        target_dict = dict_from(*xor_patterns())
        kl = kl_divergence(target_dict, model_dict)
        print("Exact P(v):")
        for v_row, p in zip(V_all, P_all):
            mark = "*" if tuple(v_row.tolist()) in target_dict else " "
            print(f"  {tuple(v_row)} : {p:.5f}{mark}")
        print(f"KL(target || model) = {kl:.6f}")
        print()

    print("Demo complete.")


if __name__ == "__main__":
    step_by_step_demo(epochs=5, k=1, lr=0.05)

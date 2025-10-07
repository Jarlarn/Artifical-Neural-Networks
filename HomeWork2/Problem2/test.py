import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


# Target XOR distribution over 3 visible ±1 bits:
# Patterns (v1,v2,v3) with v3 = XOR(v1,v2) under mapping 0->-1, 1->+1:
# (-1,-1,-1), (-1,+1,+1), (+1,-1,+1), (+1,+1,-1)
def xor_patterns():
    pats = np.array([[-1, -1, -1], [-1, +1, +1], [+1, -1, +1], [+1, +1, -1]], dtype=int)
    probs = np.full(len(pats), 1 / 4.0)
    return pats, probs


def enumerate_visible_states():
    # All 8 patterns of 3 ±1 bits
    vs = np.array(
        [[(1 if (i >> b) & 1 else -1) for b in range(3)] for i in range(8)], dtype=int
    )
    return vs


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class RBM:
    n_visible: int
    n_hidden: int
    rng: np.random.Generator

    def __post_init__(self):
        scale = 0.1
        self.W = self.rng.normal(0, scale, (self.n_visible, self.n_hidden))
        self.a = np.zeros(self.n_visible)  # visible biases
        self.b = np.zeros(self.n_hidden)  # hidden biases

    # For ±1 units: P(h_j=+1|v) = sigmoid(2(b_j + W_j^T v))
    def sample_h(self, v):
        act = 2 * (self.b + v @ self.W)
        p = sigmoid(act)
        h = np.where(self.rng.random(p.shape) < p, 1, -1)
        return h, p

    # P(v_i=+1|h) = sigmoid(2(a_i + W_i·h))
    def sample_v(self, h):
        # Supports h shape (n_hidden,) or (batch, n_hidden)
        act = 2 * (
            self.a + h @ self.W.T
        )  # FIX: was self.W @ h (dimension error for batches)
        p = sigmoid(act)
        v = np.where(self.rng.random(p.shape) < p, 1, -1)
        return v, p

    def cd_k(self, data, k=1, lr=0.05, batch_size=16):
        n = data.shape[0]
        idx = self.rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = data[idx[start : start + batch_size]]
            # Positive phase
            h0, ph0 = self.sample_h(batch)
            pos_W = batch.T @ h0
            pos_a = batch.sum(axis=0)
            pos_b = h0.sum(axis=0)
            # Gibbs chain
            v = batch
            h = h0
            for _ in range(k):
                v, _ = self.sample_v(h)
                h, _ = self.sample_h(v)
            neg_W = v.T @ h
            neg_a = v.sum(axis=0)
            neg_b = h.sum(axis=0)
            m = batch.shape[0]
            self.W += lr * (pos_W - neg_W) / m
            self.a += lr * (pos_a - neg_a) / m
            self.b += lr * (pos_b - neg_b) / m

    # Exact unnormalized probability: exp(a^T v) * Π_j 2 cosh(b_j + W_j^T v)
    def unnormalized_pv(self, V):
        # V: (N, n_visible)
        linear = V @ self.a
        hidden_terms = np.prod(2 * np.cosh(self.b + V @ self.W), axis=1)
        return np.exp(linear) * hidden_terms

    def model_distribution_exact(self):
        V = enumerate_visible_states()
        unnorm = self.unnormalized_pv(V)
        Z = unnorm.sum()
        return V, unnorm / Z

    def gibbs_chain_visible(self, steps=50000, burn_in=5000, thin=10):
        # Start random
        v = self.rng.choice([-1, 1], size=self.n_visible)
        counts = {}
        total = 0
        for t in range(steps):
            h, _ = self.sample_h(v)
            v, _ = self.sample_v(h)
            if t >= burn_in and (t - burn_in) % thin == 0:
                key = tuple(v.tolist())
                counts[key] = counts.get(key, 0) + 1
                total += 1
        V = enumerate_visible_states()
        p = np.zeros(len(V))
        mapping = {tuple(v.tolist()): i for i, v in enumerate(V)}
        for k, c in counts.items():
            p[mapping[k]] = c / total
        return V, p


def kl_divergence(p_data_dict, p_model_dict, eps=1e-12):
    kl = 0.0
    for v, pd in p_data_dict.items():
        pm = max(p_model_dict.get(v, 0.0), eps)
        kl += pd * (np.log(pd + eps) - np.log(pm))
    return kl


def dict_from(V, P):
    return {tuple(v.tolist()): float(p) for v, p in zip(V, P)}


def train_and_evaluate(
    hidden_list=(1, 2, 4, 8), epochs=2000, lr=0.05, k=1, seed=0, eval_chain_steps=200000
):
    rng = np.random.default_rng(seed)
    data_pats, _ = xor_patterns()
    dataset = np.repeat(data_pats, 25, axis=0)  # 100 samples (25 of each)
    target_V, target_P = (
        xor_patterns()
    )  # FIX: simpler & clearer than lambda indirection
    target_dict = dict_from(target_V, target_P)
    results = []
    for M in hidden_list:
        rbm = RBM(3, M, rng)
        for e in range(epochs):
            rbm.cd_k(dataset, k=k, lr=lr, batch_size=16)
        # Empirical (chain)
        V_chain, P_chain = rbm.gibbs_chain_visible(steps=eval_chain_steps)
        chain_dict = dict_from(V_chain, P_chain)
        # Exact
        V_exact, P_exact = rbm.model_distribution_exact()
        exact_dict = dict_from(V_exact, P_exact)
        kl_chain = kl_divergence(target_dict, chain_dict)
        kl_exact = kl_divergence(target_dict, exact_dict)
        results.append(
            {"M": M, "KL_chain": kl_chain, "KL_exact": kl_exact, "P_exact": exact_dict}
        )
        print(f"M={M:2d}  KL(chain)={kl_chain:.5f}  KL(exact)={kl_exact:.5f}")
    return results


def plot_results(results, outfile="results.png"):
    Ms = [r["M"] for r in results]
    kl_chain = [r["KL_chain"] for r in results]
    kl_exact = [r["KL_exact"] for r in results]
    plt.figure(figsize=(5, 3))
    plt.plot(Ms, kl_chain, "o-", label="KL (Gibbs estimate)")
    plt.plot(Ms, kl_exact, "s--", label="KL (exact)")
    plt.xlabel("Number of hidden units M")
    plt.ylabel("KL divergence")
    plt.title("RBM on XOR distribution")
    plt.grid(True, ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    print(f"Saved plot to {outfile}")


def print_best_table(results):
    xor_pats, xor_probs = xor_patterns()
    target = {tuple(p): pr for p, pr in zip(xor_pats, xor_probs)}
    best = min(results, key=lambda r: r["KL_exact"])
    print(f"\nBest model: M={best['M']}  KL_exact={best['KL_exact']:.6f}")
    print("Pattern        Target   Model    AbsDiff")
    print("-----------------------------------------")
    for pat in enumerate_visible_states():
        key = tuple(pat.tolist())
        tprob = target.get(key, 0.0)
        mprob = best["P_exact"].get(key, 0.0)
        print(f"{key}  {tprob:7.4f}  {mprob:7.4f}  {abs(tprob-mprob):7.4f}")


if __name__ == "__main__":
    # Hyperparameters (tune as needed)
    results = train_and_evaluate(
        hidden_list=(1, 2, 4, 8),
        epochs=10000,  # increase for better fit
        lr=0.02,
        k=5,  # try k=1,2,5
        seed=42,
        eval_chain_steps=100000,
    )
    plot_results(results)
    print_best_table(results)

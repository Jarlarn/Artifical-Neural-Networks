import numpy as np
import itertools
import math
import time
from dataclasses import dataclass
import matplotlib.pyplot as plt
from collections import Counter

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Four XOR-like patterns (probability 1/4 each)
# Given: [[-1,-1,-1],[1,-1,1],[-1,1,1],[1,1,-1]]
data_patterns = np.array([
    [-1, -1, -1],
    [ 1, -1,  1],
    [-1,  1,  1],
    [ 1,  1, -1],
], dtype=int)

def get_data_distribution():
    probs = {tuple(p): 0.25 for p in data_patterns}
    return probs

ALL_VISIBLE = np.array(list(itertools.product([-1, 1], repeat=3)), dtype=int)

@dataclass
class RBMConfig:
    n_visible: int
    n_hidden: int
    lr: float = 0.05
    weight_scale: float = 0.1
    k: int = 1
    l2: float = 0.0
    seed: int = 0

class RBMPlusMinus:
    def __init__(self, cfg: RBMConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.W = rng.normal(0, cfg.weight_scale, size=(cfg.n_visible, cfg.n_hidden))
        self.a = np.zeros(cfg.n_visible)
        self.b = np.zeros(cfg.n_hidden)
        self.rng = rng

    def p_hidden_plus(self, v):
        field = self.b + v @ self.W
        return sigmoid(2 * field)

    def sample_hidden(self, v):
        p_plus = self.p_hidden_plus(v)
        r = self.rng.uniform(size=p_plus.shape)
        return np.where(r < p_plus, 1, -1)

    def p_visible_plus(self, h):
        field = self.a + h @ self.W.T
        return sigmoid(2 * field)

    def sample_visible(self, h):
        p_plus = self.p_visible_plus(h)
        r = self.rng.uniform(size=p_plus.shape)
        return np.where(r < p_plus, 1, -1)

    def gibbs_step(self, v0):
        h = self.sample_hidden(v0)
        v = self.sample_visible(h)
        return v, h

    def cd_step(self, batch):
        h_pos = self.sample_hidden(batch)
        pos_w = batch.T @ h_pos / batch.shape[0]
        pos_a = batch.mean(axis=0)
        pos_b = h_pos.mean(axis=0)

        v_neg = batch.copy()
        h_neg = None
        for _ in range(self.cfg.k):
            h_neg = self.sample_hidden(v_neg)
            v_neg = self.sample_visible(h_neg)

        neg_w = v_neg.T @ h_neg / batch.shape[0]
        neg_a = v_neg.mean(axis=0)
        neg_b = h_neg.mean(axis=0)

        dW = pos_w - neg_w - self.cfg.l2 * self.W
        da = pos_a - neg_a
        db = pos_b - neg_b

        self.W += self.cfg.lr * dW
        self.a += self.cfg.lr * da
        self.b += self.cfg.lr * db

    def train(self, data, epochs=4000, batch_size=None, verbose_every=1000):
        if batch_size is None or batch_size > len(data):
            batch_size = len(data)
        n = len(data)
        for ep in range(1, epochs + 1):
            idx = self.rng.integers(0, n, size=batch_size)
            batch = data[idx]
            self.cd_step(batch)
            if verbose_every and ep % verbose_every == 0:
                h = self.sample_hidden(data)
                v_rec = self.sample_visible(h)
                recon_err = np.mean(np.sum((data - v_rec) != 0, axis=1))
                print(f"[M={self.cfg.n_hidden}] Epoch {ep}: recon diff bits={recon_err:.3f}")

    def unnormalized_p_visible(self, v):
        if v.ndim == 1:
            v = v[None, :]
        res = []
        for vv in v:
            fields = self.b + vv @ self.W
            # exp(a·v) * Π_j 2 cosh(fields_j)
            val = np.exp(self.a @ vv) * np.prod(2 * np.cosh(fields))
            res.append(val)
        return np.array(res)

    def exact_visible_distribution(self):
        unnorm = np.array([self.unnormalized_p_visible(v)[0] for v in ALL_VISIBLE])
        Z = unnorm.sum()
        return {tuple(ALL_VISIBLE[i]): unnorm[i] / Z for i in range(len(ALL_VISIBLE))}

    def long_gibbs_chain(self, steps=50000, burn_in=5000, thinning=10, init=None):
        if init is None:
            init = self.rng.choice([-1, 1], size=self.cfg.n_visible)
        v = init.copy()
        samples = []
        for t in range(steps):
            h = self.sample_hidden(v[None, :])[0]
            v = self.sample_visible(h[None, :])[0]
            if t >= burn_in and (t - burn_in) % thinning == 0:
                samples.append(tuple(v.tolist()))
        return samples

def kl_divergence(p_dist, q_dist, eps=1e-12):
    kl = 0.0
    for v, p in p_dist.items():
        q = q_dist.get(v, 0.0)
        if p > 0:
            kl += p * (math.log(p + eps) - math.log(q + eps))
    return kl

def theoretical_kl(M_values, N=3):
    """
    Implements Eq. (4.40) bound (interpreted):
    D_KL <= log(2) * ( N - floor(log2(M+1)) - (M+1)/2^{floor(log2(M+1))} ) for M < 2^{N-1} - 1
    D_KL = 0 for M >= 2^{N-1} - 1
    Natural logarithm assumed (log = ln).
    """
    bound = []
    cutoff = 2**(N-1) - 1
    for M in M_values:
        if M < cutoff:
            k = int(np.floor(np.log2(M + 1)))
            term = N - k - (M + 1) / (2**k)
            bound.append(np.log(2) * term)
        else:
            bound.append(0.0)
    return bound

def run_experiment(hidden_list=(1,2,4,8),
                   ks=(1,2,5),
                   lrs=(0.05, 0.02),
                   epochs=6000,
                   sampling_steps=80000,
                   burn_in=8000):
    data_dist = get_data_distribution()
    data_array = np.array(list(data_dist.keys()), dtype=int)

    results = []
    for M in hidden_list:
        best_kl = float('inf')
        best_cfg = None
        best_exact_dist = None
        for k in ks:
            for lr in lrs:
                cfg = RBMConfig(n_visible=3, n_hidden=M, lr=lr, k=k, seed=123 + 17*M + k)
                rbm = RBMPlusMinus(cfg)
                rbm.train(data_array, epochs=epochs, verbose_every=epochs//3)

                exact_dist = rbm.exact_visible_distribution()
                kl_exact = kl_divergence(data_dist, exact_dist)

                samples = rbm.long_gibbs_chain(steps=sampling_steps,
                                               burn_in=burn_in,
                                               thinning=10)
                counts = Counter(samples)
                total = sum(counts.values())
                sampled_dist = {v: counts.get(v, 0) / total for v in data_dist.keys()}
                kl_sample = kl_divergence(data_dist, sampled_dist)

                print(f"M={M} k={k} lr={lr}: KL_exact={kl_exact:.6f} KL_sample={kl_sample:.6f}")

                if kl_exact < best_kl:
                    best_kl = kl_exact
                    best_cfg = (k, lr)
                    best_exact_dist = exact_dist

        results.append((M, best_kl, best_cfg, best_exact_dist))

    Ms = [r[0] for r in results]
    KLs = [r[1] for r in results]
    theory = theoretical_kl(Ms, N=3)

    plt.figure(figsize=5,4)
    plt.plot(Ms, KLs, 'o-', label='Empirical KL (best per M)')
    plt.plot(Ms, theory, 's--', label='Theory bound (Eq 4.40)')
    plt.xlabel('Number of hidden units M')
    plt.ylabel('KL(P_data || P_model)')
    plt.title('RBM vs XOR distribution')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Summary:")
    for (M, klv, (k, lr), _) in results:
        print(f"M={M}: KL={klv:.6f} (k={k}, lr={lr}), theory_bound={theory[Ms.index(M)]:.6f}")

if __name__ == "__test__":
    run_experiment()
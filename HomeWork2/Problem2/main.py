import numpy as np
import itertools
import math
from dataclasses import dataclass
import matplotlib.pyplot as plt
from collections import Counter


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# Data patterns (prob = 1/4 each)
data_patterns = np.array(
    [
        [-1, -1, -1],
        [1, -1, 1],
        [-1, 1, 1],
        [1, 1, -1],
    ],
    dtype=int,
)


def get_data_distribution():
    return {tuple(p): 0.25 for p in data_patterns}


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
        return np.where(self.rng.uniform(size=p_plus.shape) < p_plus, 1, -1)

    def p_visible_plus(self, h):
        field = self.a + h @ self.W.T
        return sigmoid(2 * field)

    def sample_visible(self, h):
        p_plus = self.p_visible_plus(h)
        return np.where(self.rng.uniform(size=p_plus.shape) < p_plus, 1, -1)

    def cd_step(self, batch):
        h_pos = self.sample_hidden(batch)
        pos_w = batch.T @ h_pos / batch.shape[0]
        pos_a = batch.mean(axis=0)
        pos_b = h_pos.mean(axis=0)

        v_neg = batch.copy()
        for _ in range(self.cfg.k):
            h_neg = self.sample_hidden(v_neg)
            v_neg = self.sample_visible(h_neg)

        neg_w = v_neg.T @ h_neg / batch.shape[0]
        neg_a = v_neg.mean(axis=0)
        neg_b = h_neg.mean(axis=0)

        self.W += self.cfg.lr * (pos_w - neg_w - self.cfg.l2 * self.W)
        self.a += self.cfg.lr * (pos_a - neg_a)
        self.b += self.cfg.lr * (pos_b - neg_b)

    def train(self, data, epochs=6000, batch_size=None, verbose_every=2000):
        if batch_size is None or batch_size > len(data):
            batch_size = len(data)
        n = len(data)
        for ep in range(1, epochs + 1):
            idx = self.rng.integers(0, n, size=batch_size)
            self.cd_step(data[idx])
            if verbose_every and ep % verbose_every == 0:
                h = self.sample_hidden(data)
                v_rec = self.sample_visible(h)
                recon_err = np.mean(np.sum(data != v_rec, axis=1))
                print(
                    f"[M={self.cfg.n_hidden}] Epoch {ep}: recon diff bits={recon_err:.2f}"
                )

    def unnormalized_p_visible(self, v):
        if v.ndim == 1:
            v = v[None, :]
        vals = []
        for vv in v:
            fields = self.b + vv @ self.W
            vals.append(np.exp(self.a @ vv) * np.prod(2 * np.cosh(fields)))
        return np.array(vals)

    def exact_visible_distribution(self):
        un = np.array([self.unnormalized_p_visible(v)[0] for v in ALL_VISIBLE])
        Z = un.sum()
        return {tuple(ALL_VISIBLE[i]): un[i] / Z for i in range(len(ALL_VISIBLE))}

    def long_gibbs_chain(self, steps=80000, burn_in=8000, thinning=10):
        v = self.rng.choice([-1, 1], size=self.cfg.n_visible)
        samples = []
        for t in range(steps):
            h = self.sample_hidden(v[None, :])[0]
            v = self.sample_visible(h[None, :])[0]
            if t >= burn_in and (t - burn_in) % thinning == 0:
                samples.append(tuple(v.tolist()))
        return samples


def kl_divergence(p, q, eps=1e-12):
    s = 0.0
    for v, pv in p.items():
        qv = q.get(v, 0.0)
        if pv > 0:
            s += pv * (math.log(pv + eps) - math.log(qv + eps))
    return s


# Equation (4.40) bound with cutoff 2^(N-1)-1
def theoretical_kl(M_values, N=3):
    cutoff = 2 ** (N - 1) - 1
    out = []
    for M in M_values:
        if M < cutoff:
            k = int(np.floor(np.log2(M + 1)))
            term = N - k - (M + 1) / (2**k)
            out.append(np.log(2) * term)
        else:
            out.append(0.0)
    return out


def estimate_sampling_length(rbm, target_std=0.005, max_steps=120000, check_every=5000):
    data_dist = get_data_distribution()
    counts = Counter()
    collected = 0
    burn_in = 5000
    thinning = 10
    v = rbm.rng.choice([-1, 1], size=rbm.cfg.n_visible)
    for t in range(max_steps):
        h = rbm.sample_hidden(v[None, :])[0]
        v = rbm.sample_visible(h[None, :])[0]
        if t >= burn_in and (t - burn_in) % thinning == 0:
            counts[tuple(v.tolist())] += 1
            collected += 1
            if collected % (check_every // thinning) == 0:
                freqs = np.array(
                    [counts.get(p, 0) / collected for p in data_dist.keys()]
                )
                if len(freqs) > 0:
                    std = freqs.std()
                    if std < target_std:
                        return collected, {
                            k: counts.get(k, 0) / collected for k in data_dist.keys()
                        }
    return collected, {k: counts.get(k, 0) / collected for k in data_dist.keys()}


def run_experiment(
    hidden_list=(1, 2, 4, 8), ks=(1, 2, 5), lrs=(0.05, 0.02), epochs=6000
):
    data_dist = get_data_distribution()

    data_array = np.array(list(data_dist.keys()), dtype=int)
    results = []
    for M in hidden_list:
        best = (float("inf"), None, None)
        for k in ks:
            for lr in lrs:
                cfg = RBMConfig(3, M, lr=lr, k=k, seed=100 + 13 * M + k)
                rbm = RBMPlusMinus(cfg)
                rbm.train(data_array, epochs=epochs, verbose_every=epochs // 3)
                dist = rbm.exact_visible_distribution()
                klv = kl_divergence(data_dist, dist)
                print(f"M={M} k={k} lr={lr}: KL_exact={klv:.6f}")
                if klv < best[0]:
                    best = (klv, (k, lr), dist)
        results.append((M, *best))
    Ms = [r[0] for r in results]
    KLs = [r[1] for r in results]
    theory = theoretical_kl(Ms, N=3)
    plt.figure(figsize=(5, 4))
    plt.plot(Ms, KLs, "o-", label="Empirical KL (best)")
    plt.plot(Ms, theory, "s--", label="Theory bound")
    plt.xlabel("Hidden units M")
    plt.ylabel("KL(P_data || P_model)")
    plt.title("KL vs M")
    plt.legend()
    plt.tight_layout()
    plt.show()
    for M, klv, cfg, _ in results:
        print(
            f"M={M}: KL={klv:.6f} best_cfg={cfg} theory_bound={theory[Ms.index(M)]:.6f}"
        )


if __name__ == "__main__":

    run_experiment()

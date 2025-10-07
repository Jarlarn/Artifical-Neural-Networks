# Written with the help of AI, blogs, videos and documentation
import numpy as np
import itertools
import math
from dataclasses import dataclass
from typing import Optional
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
    seed: Optional[int] = None


class RBMPlusMinus:
    def __init__(self, cfg: RBMConfig):
        self.cfg = cfg
        if cfg.seed is None:
            rng = np.random.default_rng()
            self.seed_used = None
        else:
            rng = np.random.default_rng(cfg.seed)
            self.seed_used = cfg.seed
        self.W = rng.normal(0, cfg.weight_scale, size=(cfg.n_visible, cfg.n_hidden))
        self.visible_bias = np.zeros(cfg.n_visible)
        self.hidden_bias = np.zeros(cfg.n_hidden)
        self.rng = rng

    def p_hidden_plus(self, visible_states):
        hidden_field = self.hidden_bias + visible_states @ self.W
        return sigmoid(2 * hidden_field)

    def sample_hidden(self, visible_states):
        p_hidden_is_plus = self.p_hidden_plus(visible_states)
        return np.where(
            self.rng.uniform(size=p_hidden_is_plus.shape) < p_hidden_is_plus, 1, -1
        )

    def p_visible_plus(self, hidden_states):
        visible_field = self.visible_bias + hidden_states @ self.W.T
        return sigmoid(2 * visible_field)

    def sample_visible(self, hidden_states):
        p_visible_is_plus = self.p_visible_plus(hidden_states)
        return np.where(
            self.rng.uniform(size=p_visible_is_plus.shape) < p_visible_is_plus, 1, -1
        )

    def cd_step(self, visible_batch):
        batch_size = visible_batch.shape[0]

        # Positive phase
        positive_hidden_sample = self.sample_hidden(visible_batch)
        positive_weight_expectation = (
            visible_batch.T @ positive_hidden_sample / batch_size
        )
        positive_visible_mean = visible_batch.mean(axis=0)
        positive_hidden_mean = positive_hidden_sample.mean(axis=0)

        # Negative phase via Gibbs chain (CD-k)
        model_visible = visible_batch.copy()
        for _ in range(self.cfg.k):
            model_hidden_sample = self.sample_hidden(model_visible)
            model_visible = self.sample_visible(model_hidden_sample)

        final_hidden_sample = self.sample_hidden(model_visible)
        negative_weight_expectation = model_visible.T @ final_hidden_sample / batch_size
        negative_visible_mean = model_visible.mean(axis=0)
        negative_hidden_mean = final_hidden_sample.mean(axis=0)

        # Parameter updates
        self.W += self.cfg.lr * (
            positive_weight_expectation
            - negative_weight_expectation
            - self.cfg.l2 * self.W
        )
        self.visible_bias += self.cfg.lr * (
            positive_visible_mean - negative_visible_mean
        )
        self.hidden_bias += self.cfg.lr * (positive_hidden_mean - negative_hidden_mean)

    def train(
        self, training_visible_data, epochs=6000, batch_size=None, verbose_every=2000
    ):
        if batch_size is None or batch_size > len(training_visible_data):
            batch_size = len(training_visible_data)
        num_cases = len(training_visible_data)
        for epoch in range(1, epochs + 1):
            batch_indices = self.rng.integers(0, num_cases, size=batch_size)
            self.cd_step(training_visible_data[batch_indices])
            if verbose_every and epoch % verbose_every == 0:
                sampled_hidden = self.sample_hidden(training_visible_data)
                reconstructed_visible = self.sample_visible(sampled_hidden)
                reconstruction_bit_diff = np.mean(
                    np.sum(training_visible_data != reconstructed_visible, axis=1)
                )
                print(
                    f"[M={self.cfg.n_hidden}] Epoch {epoch}: recon diff bits={reconstruction_bit_diff:.2f}"
                )

    def unnormalized_p_visible(self, visible_vector):
        if visible_vector.ndim == 1:
            visible_vector = visible_vector[None, :]
        vals = []
        for single_visible in visible_vector:
            hidden_fields = self.hidden_bias + single_visible @ self.W
            vals.append(
                np.exp(self.visible_bias @ single_visible)
                * np.prod(2 * np.cosh(hidden_fields))
            )
        return np.array(vals)

    def exact_visible_distribution(self):
        unnormalized = np.array(
            [self.unnormalized_p_visible(v)[0] for v in ALL_VISIBLE]
        )
        Z = unnormalized.sum()
        return {
            tuple(ALL_VISIBLE[i]): unnormalized[i] / Z for i in range(len(ALL_VISIBLE))
        }

    def long_gibbs_chain(self, steps=80000, burn_in=8000, thinning=10):
        current_visible = self.rng.choice([-1, 1], size=self.cfg.n_visible)
        collected_samples = []
        for t in range(steps):
            current_hidden = self.sample_hidden(current_visible[None, :])[0]
            current_visible = self.sample_visible(current_hidden[None, :])[0]
            if t >= burn_in and (t - burn_in) % thinning == 0:
                collected_samples.append(tuple(current_visible.tolist()))
        return collected_samples


def kl_divergence(p_dist, q_dist, eps=1e-12):
    total = 0.0
    for state, p_val in p_dist.items():
        q_val = q_dist.get(state, 0.0)
        if p_val > 0:
            total += p_val * (math.log(p_val + eps) - math.log(q_val + eps))
    return total


def theoretical_kl(num_hidden_list, N=3):
    cutoff = 2 ** (N - 1) - 1
    bounds = []
    for M in num_hidden_list:
        if M < cutoff:
            k_val = int(np.floor(np.log2(M + 1)))
            term = N - k_val - (M + 1) / (2**k_val)
            bounds.append(np.log(2) * term)
        else:
            bounds.append(0.0)
    return bounds


def estimate_sampling_length(rbm, target_std=0.005, max_steps=120000, check_every=5000):
    data_dist = get_data_distribution()
    pattern_counts = Counter()
    collected = 0
    burn_in = 5000
    thinning = 10
    current_visible = rbm.rng.choice([-1, 1], size=rbm.cfg.n_visible)
    for t in range(max_steps):
        current_hidden = rbm.sample_hidden(current_visible[None, :])[0]
        current_visible = rbm.sample_visible(current_hidden[None, :])[0]
        if t >= burn_in and (t - burn_in) % thinning == 0:
            pattern_counts[tuple(current_visible.tolist())] += 1
            collected += 1
            if collected % (check_every // thinning) == 0:
                freqs = np.array(
                    [pattern_counts.get(p, 0) / collected for p in data_dist.keys()]
                )
                if len(freqs) > 0:
                    std = freqs.std()
                    if std < target_std:
                        return collected, {
                            k: pattern_counts.get(k, 0) / collected
                            for k in data_dist.keys()
                        }
    return collected, {
        k: pattern_counts.get(k, 0) / collected for k in data_dist.keys()
    }


def run_experiment(
    hidden_list=(1, 2, 4, 8),
    ks=(1, 2, 5),
    lrs=(0.05, 0.02),
    epochs=6000,
    sample_steps=40000,
    burn_in=8000,
    thinning=10,
):
    data_dist = get_data_distribution()
    training_visible_data = np.array(list(data_dist.keys()), dtype=int)
    results = []
    for num_hidden in hidden_list:
        best = (float("inf"), None, None)
        for k_steps in ks:
            for learning_rate in lrs:
                # Use a random seed each time (cfg.seed=None) unless caller specifies otherwise
                cfg = RBMConfig(
                    3,
                    num_hidden,
                    lr=learning_rate,
                    k=k_steps,
                    seed=None,
                )
                rbm = RBMPlusMinus(cfg)
                rbm.train(
                    training_visible_data, epochs=epochs, verbose_every=epochs // 3
                )

                exact_dist = rbm.exact_visible_distribution()
                kl_exact_val = kl_divergence(data_dist, exact_dist)

                samples = rbm.long_gibbs_chain(
                    steps=sample_steps, burn_in=burn_in, thinning=thinning
                )
                sample_counts = Counter(samples)
                total_samples = len(samples)
                sampled_dist = {
                    pat: sample_counts.get(pat, 0) / total_samples
                    for pat in data_dist.keys()
                }
                kl_sample_val = kl_divergence(data_dist, sampled_dist)

                print(
                    f"M={num_hidden} k={k_steps} lr={learning_rate}: "
                    f"KL_exact={kl_exact_val:.6f} KL_sample={kl_sample_val:.6f} (samples={total_samples})"
                )

                if kl_exact_val < best[0]:
                    best = (
                        kl_exact_val,
                        (k_steps, learning_rate),
                        exact_dist,
                        sampled_dist,
                        kl_sample_val,
                    )

        results.append((num_hidden, *best))
    Ms = [r[0] for r in results]
    KLs = [r[1] for r in results]
    theory = theoretical_kl(Ms, N=3)
    plt.figure(figsize=(5, 4))
    plt.plot(Ms, KLs, "o-", label="Empirical best (exact KL)")
    plt.plot(Ms, theory, "s--", label="Theory bound")
    plt.xlabel("Hidden units M")
    plt.ylabel("KL(P_data || P_model)")
    plt.title("KL vs M")
    plt.legend()
    plt.tight_layout()
    plt.show()
    for (
        num_hidden,
        kl_exact_val,
        best_cfg,
        exact_dist,
        sampled_dist,
        kl_sample_val,
    ) in results:
        print(
            f"M={num_hidden}: KL_exact={kl_exact_val:.6f} KL_sample={kl_sample_val:.6f} "
            f"best_cfg={best_cfg} theory_bound={theory[Ms.index(num_hidden)]:.6f}"
        )
        print("  Sampled frequencies:", sampled_dist)


if __name__ == "__main__":
    run_experiment()

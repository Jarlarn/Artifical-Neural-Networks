import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


# Target XOR distribution over 3 visible ±1 bits:
# Patterns (v1,v2,v3) with v3 = XOR(v1,v2) under mapping 0->-1, 1->+1:
# (-1,-1,-1), (-1,+1,+1), (+1,-1,+1), (+1,+1,-1)
def xor_patterns():
    xor_visible_patterns = np.array(
        [[-1, -1, -1], [-1, +1, +1], [+1, -1, +1], [+1, +1, -1]], dtype=int
    )
    xor_target_probabilities = np.full(len(xor_visible_patterns), 1 / 4.0)
    return xor_visible_patterns, xor_target_probabilities


def enumerate_visible_states():
    # All 8 patterns of 3 ±1 bits
    all_visible_states = np.array(
        [[(1 if (i >> bit) & 1 else -1) for bit in range(3)] for i in range(8)],
        dtype=int,
    )
    return all_visible_states


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class RBM:
    n_visible: int
    n_hidden: int
    rng: np.random.Generator

    # yes
    def __post_init__(self):
        weight_init_scale = 0.1
        self.weights = self.rng.normal(
            0, weight_init_scale, (self.n_visible, self.n_hidden)
        )
        self.visible_threshold = np.zeros(self.n_visible)
        self.hidden_threshold = np.zeros(self.n_hidden)

    # yes
    def sample_hidden(self, visible_batch):
        field_hidden = visible_batch @ self.weights - self.hidden_threshold
        prob_hidden = sigmoid(2 * field_hidden)
        hidden_sample = np.where(
            self.rng.random(prob_hidden.shape) < prob_hidden, 1, -1
        )
        return hidden_sample, prob_hidden

    # yes
    def sample_visible(self, hidden_batch):
        field_visible = hidden_batch @ self.weights.T - self.visible_threshold
        prob_visible = sigmoid(2 * field_visible)
        visible_sample = np.where(
            self.rng.random(prob_visible.shape) < prob_visible, 1, -1
        )
        return visible_sample, prob_visible

    def cd_k(self, training_data, k=1, learning_rate=0.05, batch_size=16):
        n_samples = training_data.shape[0]
        shuffled = self.rng.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            v0 = training_data[shuffled[start : start + batch_size]]

            # Positive phase (data)
            h0_sample, h0_prob = self.sample_hidden(v0)
            pos_W = v0.T @ h0_sample
            pos_v_theta = v0.sum(axis=0)
            pos_h_theta = h0_sample.sum(axis=0)

            # Gibbs chain k steps
            v_chain = v0
            h_chain = h0_sample
            for _ in range(k):
                v_chain, _ = self.sample_visible(h_chain)
                h_chain, _ = self.sample_hidden(v_chain)

            # Negative phase (model)
            neg_W = v_chain.T @ h_chain
            neg_v_theta = v_chain.sum(axis=0)
            neg_h_theta = h_chain.sum(axis=0)

            m = v0.shape[0]
            self.weights += learning_rate * (pos_W - neg_W) / m
            # For thresholds θ we subtract gradient that previously added to bias (bias = -θ)
            self.visible_threshold -= learning_rate * (pos_v_theta - neg_v_theta) / m
            self.hidden_threshold -= learning_rate * (pos_h_theta - neg_h_theta) / m

    # Unnormalized P(v) = exp(-θ^(v)·v) * Π_j 2 cosh( (v W)_j - θ_j^(h) )
    def unnormalized_visible_prob(self, visible_states):
        field_hidden = visible_states @ self.weights - self.hidden_threshold
        hidden_terms = np.prod(2 * np.cosh(field_hidden), axis=1)
        linear_term = -(visible_states @ self.visible_threshold)
        return np.exp(linear_term) * hidden_terms

    def model_distribution_exact(self):
        all_visible_states = enumerate_visible_states()
        unnormalized = self.unnormalized_visible_prob(all_visible_states)
        partition_function = unnormalized.sum()
        return all_visible_states, unnormalized / partition_function

    def gibbs_chain_visible(self, steps=50000, burn_in=5000, thin=10):
        current_visible = self.rng.choice([-1, 1], size=self.n_visible)
        occurrence_counts = {}
        total_kept = 0
        for t in range(steps):
            current_hidden, _ = self.sample_hidden(current_visible)
            current_visible, _ = self.sample_visible(current_hidden)
            if t >= burn_in and (t - burn_in) % thin == 0:
                key = tuple(current_visible.tolist())
                occurrence_counts[key] = occurrence_counts.get(key, 0) + 1
                total_kept += 1
        all_visible_states = enumerate_visible_states()
        empirical_probabilities = np.zeros(len(all_visible_states))
        index_lookup = {
            tuple(state.tolist()): i for i, state in enumerate(all_visible_states)
        }
        for pattern_key, count in occurrence_counts.items():
            empirical_probabilities[index_lookup[pattern_key]] = count / total_kept
        return all_visible_states, empirical_probabilities


def kl_divergence(target_prob_dict, model_prob_dict, eps=1e-12):
    divergence = 0.0
    for state_key, target_prob in target_prob_dict.items():
        model_prob = max(model_prob_dict.get(state_key, 0.0), eps)
        divergence += target_prob * (np.log(target_prob + eps) - np.log(model_prob))
    return divergence


def dict_from(visible_states, probabilities):
    return {
        tuple(state.tolist()): float(p)
        for state, p in zip(visible_states, probabilities)
    }


def train_and_evaluate(
    hidden_unit_list=(1, 2, 4, 8),
    epochs=2000,
    learning_rate=0.05,
    k=1,
    seed=0,
    eval_chain_steps=200000,
):
    rng = np.random.default_rng(seed)
    xor_data_patterns, _ = xor_patterns()
    training_dataset = np.repeat(xor_data_patterns, 25, axis=0)
    target_states, target_probs = xor_patterns()
    target_prob_dict = dict_from(target_states, target_probs)
    results = []
    for n_hidden_units in hidden_unit_list:
        rbm = RBM(3, n_hidden_units, rng)
        for _ in range(epochs):
            rbm.cd_k(training_dataset, k=k, learning_rate=learning_rate, batch_size=16)
        # Empirical (sampling)
        sampled_states, sampled_probs = rbm.gibbs_chain_visible(steps=eval_chain_steps)
        sampled_prob_dict = dict_from(sampled_states, sampled_probs)
        # Exact
        exact_states, exact_probs = rbm.model_distribution_exact()
        exact_prob_dict = dict_from(exact_states, exact_probs)
        kl_sampling = kl_divergence(target_prob_dict, sampled_prob_dict)
        kl_exact = kl_divergence(target_prob_dict, exact_prob_dict)
        results.append(
            {
                "M": n_hidden_units,
                "KL_chain": kl_sampling,
                "KL_exact": kl_exact,
                "P_exact": exact_prob_dict,
            }
        )
        print(
            f"M={n_hidden_units:2d}  KL(chain)={kl_sampling:.5f}  KL(exact)={kl_exact:.5f}"
        )
    return results


def plot_results(results, outfile="results.png"):
    hidden_sizes = [r["M"] for r in results]
    kl_chain_values = [r["KL_chain"] for r in results]
    kl_exact_values = [r["KL_exact"] for r in results]
    plt.figure(figsize=(5, 3))
    plt.plot(hidden_sizes, kl_chain_values, "o-", label="KL (Gibbs estimate)")
    plt.plot(hidden_sizes, kl_exact_values, "s--", label="KL (exact)")
    plt.xlabel("Number of hidden units M")
    plt.ylabel("KL divergence")
    plt.title("RBM on XOR distribution")
    plt.grid(True, ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    print(f"Saved plot to {outfile}")


def print_best_table(results):
    xor_states, xor_target_probs = xor_patterns()
    target_prob_map = {tuple(s): p for s, p in zip(xor_states, xor_target_probs)}
    best_result = min(results, key=lambda r: r["KL_exact"])
    print(f"\nBest model: M={best_result['M']}  KL_exact={best_result['KL_exact']:.6f}")
    print("Pattern        Target   Model    AbsDiff")
    print("-----------------------------------------")
    for state in enumerate_visible_states():
        key = tuple(state.tolist())
        target_p = target_prob_map.get(key, 0.0)
        model_p = best_result["P_exact"].get(key, 0.0)
        print(f"{key}  {target_p:7.4f}  {model_p:7.4f}  {abs(target_p-model_p):7.4f}")


if __name__ == "__main__":
    results = train_and_evaluate(
        hidden_unit_list=(1, 2, 4, 8),
        epochs=20000,
        learning_rate=0.01,
        k=5,
        seed=42,
        eval_chain_steps=100000,
    )
    plot_results(results)
    print_best_table(results)

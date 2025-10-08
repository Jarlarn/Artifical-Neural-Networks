import numpy as np
import matplotlib.pyplot as plt
from scipy.special import xlogy
from itertools import product
from collections import Counter

class RBM:
    def __init__(self, n_visible, n_hidden, seed=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        
        self.rng = np.random.default_rng(seed)

        self.W = self.rng.normal(scale=0.1, size=(n_hidden, n_visible))
        self.b_h = np.zeros(n_hidden)
        self.b_v = np.zeros(n_visible)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def train(self, data, epochs, learning_rate, k):
        shuffled_data = data.copy()
        
        for epoch in range(epochs):
            self.rng.shuffle(shuffled_data)
            
            for v0 in shuffled_data:
                hidden_field_0 = self.b_h + np.dot(self.W, v0)
                
                prob_h_0 = self._sigmoid(2 * hidden_field_0)
                h_k = np.where(self.rng.random(size=self.n_hidden) < prob_h_0, 1, -1)

                for step in range(k):
                    visible_field_k = self.b_v + np.dot(self.W.T, h_k)
                    prob_v_k = self._sigmoid(2 * visible_field_k)
                    v_k = np.where(self.rng.random(size=self.n_visible) < prob_v_k, 1, -1)

                    hidden_field_k = self.b_h + np.dot(self.W, v_k)
                    prob_h_k = self._sigmoid(2 * hidden_field_k)
                    h_k = np.where(self.rng.random(size=self.n_hidden) < prob_h_k, 1, -1)

                # to change
                prob_h_0 = self._sigmoid(2 * hidden_field_0)
                h_0 = np.where(self.rng.random(size=self.n_hidden) < prob_h_0, 1, -1)

                self.W += learning_rate * (np.outer(h_0, v0) - np.outer(h_k, v_k))
                self.b_v += learning_rate * (v0 - v_k)
                self.b_h += learning_rate * (h_0 - h_k)


    def energy(self, v, h):
        return -np.dot(h, np.dot(self.W, v)) - np.dot(self.b_v, v) - np.dot(self.b_h, h)

    def get_model_distribution(self):
        all_v = np.array(list(product([-1, 1], repeat=self.n_visible)))
        all_h = np.array(list(product([-1, 1], repeat=self.n_hidden)))
        
        probs = np.zeros(len(all_v))
        for i, v in enumerate(all_v):
            log_sum_exp = np.logaddexp.reduce([-self.energy(v, h) for h in all_h])
            probs[i] = log_sum_exp
        
        log_Z = np.logaddexp.reduce(probs)
        final_probs = np.exp(probs - log_Z)
        
        return dict(zip([tuple(v) for v in all_v], final_probs))
        
    def sample_dynamics(self, n_steps, initial_v=None):
        if initial_v is None:
            v = self.rng.choice([-1, 1], size=self.n_visible)
        else:
            v = initial_v
            
        samples = []
        for _ in range(n_steps):
            hidden_field = self.b_h + np.dot(self.W, v)
            prob_h = self._sigmoid(2 * hidden_field)
            h = np.where(self.rng.random(self.n_hidden) < prob_h, 1, -1)
            
            visible_field = self.b_v + np.dot(self.W.T, h)
            prob_v = self._sigmoid(2 * visible_field)
            v = np.where(self.rng.random(self.n_visible) < prob_v, 1, -1)
            
            samples.append(tuple(v))
            
        return samples

def compute_kl_divergence(p_data, p_model_dict):
    kl_div = 0.0
    for v_tuple, p_v_data in p_data.items():
        if p_v_data > 0:
            p_v_model = p_model_dict.get(v_tuple, 1e-12) # Use a small epsilon for stability
            kl_div += xlogy(p_v_data, p_v_data / p_v_model) if p_v_model > 0 else float('inf')
    return kl_div

def get_theoretical_bound(N, M):
    if M < 2**(N - 1) - 1:
        log2_M1 = np.log2(M + 1)
        k_val = int(np.floor(log2_M1))
        term1 = N - k_val
        term2 = (M + 1) / (2**k_val)
        return np.log(2) * (term1 - term2)
    else:
        return 0.0

def plot_distribution(ax, p_data, p_model_dict, title):
    sorted_keys = sorted(p_data.keys())
    labels = [''.join(map(str, k)).replace('-1','0') for k in sorted_keys]
    x = np.arange(len(labels))
    width = 0.35
    
    data_probs = [p_data[k] for k in sorted_keys]
    model_probs = [p_model_dict.get(k, 0) for k in sorted_keys]

    ax.bar(x - width/2, data_probs, width, label='P_data (Target)')
    ax.bar(x + width/2, model_probs, width, label='P_model (Learned)')
    ax.set_ylabel('Probability')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)


xor_data_patterns = [(-1, -1, -1), (1, -1, 1), (-1, 1, 1), (1, 1, -1)]
xor_data_np = np.array(xor_data_patterns)
N_VISIBLE = 3
all_possible_v_tuples = list(product([-1, 1], repeat=N_VISIBLE))
P_data = {v: 1/4 if v in xor_data_patterns else 0 for v in all_possible_v_tuples}



M_values = [1, 2, 4, 8]
K_values = [1, 5, 10] 
LR_values = [0.05, 0.01, 0.005] 
EPOCHS = 10000

best_kl_results = []
best_trained_rbms = {}


for M in M_values:
    print(f"\n Searching for best hyperparameters for M = {M} ")
    
    best_kl_for_this_M = float('inf')
    best_rbm_for_this_M = None
    best_hyperparams_for_this_M = {}

    for K_CD in K_values:
        for LEARNING_RATE in LR_values:
            print(f"  Training with k={K_CD}, lr={LEARNING_RATE}...")
            
            seed = 13*M + 7*K_CD + int(LEARNING_RATE*1000)
            rbm = RBM(n_visible=N_VISIBLE, n_hidden=M, seed=seed)
            rbm.train(xor_data_np, epochs=EPOCHS, learning_rate=LEARNING_RATE, k=K_CD)
            
            p_model = rbm.get_model_distribution()
            kl = compute_kl_divergence(P_data, p_model)
            print(f"  --> Final KL Divergence: {kl:.4f}")

            if kl < best_kl_for_this_M:
                best_kl_for_this_M = kl
                best_rbm_for_this_M = rbm
                best_hyperparams_for_this_M = {'k': K_CD, 'lr': LEARNING_RATE}

    print(f"\n--- Best result for M={M}: KL={best_kl_for_this_M:.4f} with params {best_hyperparams_for_this_M} ---")
    best_kl_results.append(best_kl_for_this_M)
    best_trained_rbms[M] = best_rbm_for_this_M




# Plot KL Divergence vs. Number of Hidden Neurons
plt.figure(figsize=(14, 6))
ax1 = plt.subplot(1, 2, 1)
ax1.plot(M_values, best_kl_results, 'o-', label='Empirical KL Divergence (Best Run)')
bounds = [get_theoretical_bound(N_VISIBLE, M) for M in M_values]
ax1.plot(M_values, bounds, 's--', label='Theoretical Upper Bound (Eq. 4.40)')
ax1.set_xlabel("Number of Hidden Neurons (M)")
ax1.set_ylabel("Kullback-Leibler Divergence")
ax1.set_title("RBM Performance on XOR")
ax1.set_xticks(M_values)
ax1.set_ylim(bottom=-0.1)
ax1.grid(True)
ax1.legend()

best_M_overall = M_values[np.argmin(best_kl_results)]
best_rbm_overall = best_trained_rbms[best_M_overall]
print(f"\nAnalyzing dynamics for the overall best model (M={best_M_overall})")
n_dynamic_steps = 50000
burn_in = 5000
samples = best_rbm_overall.sample_dynamics(n_dynamic_steps)
samples_after_burn_in = samples[burn_in:]

sample_counts = Counter(samples_after_burn_in)
total_samples = len(samples_after_burn_in)

p_dynamics = {v_tuple: sample_counts.get(v_tuple, 0) / total_samples for v_tuple in all_possible_v_tuples}

ax2 = plt.subplot(1, 2, 2)
plot_distribution(ax2, P_data, p_dynamics, f"Generated Distribution (Best Model, M={best_M_overall})")

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
axes = axes.flatten()
for i, M in enumerate(M_values):
    p_model_exact = best_trained_rbms[M].get_model_distribution()
    plot_distribution(axes[i], P_data, p_model_exact, f'Best Exact Distribution (M={M})')
fig.suptitle("Learned Distributions vs. Target (Best Runs)")
plt.show()
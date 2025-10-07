import numpy as np


# Tested!
def intitialize_boltzmann(M: int):
    # Intialize weights: first index hidden, second index visible
    w = np.random.normal(0.0, 0.5, (M, 3))

    # Intitialize thresholds
    theta_v = np.zeros(3)
    theta_h = np.zeros(M)

    return w, theta_v, theta_h


# Tested!
def sample_data_distribution(data, p_0: int):
    indices = np.random.randint(4, size=p_0)
    return data[indices, :]


# Tested!
def update_hidden(V, M, w, theta_h, beta):
    B_h = np.matmul(w, V) - theta_h
    p_B_h = (1 + np.exp(-2 * beta * B_h)) ** (-1)
    random_values = np.random.rand(M)

    return np.where(random_values < p_B_h, 1, -1)


# Tested!
def update_visible(H, w, theta_v, beta):
    B_v = np.matmul(H, w) - theta_v
    p_B_v = (1 + np.exp(-2 * beta * B_v)) ** (-1)
    random_values = np.random.rand(3)

    return np.where(random_values < p_B_v, 1, -1)


# Tested!
def compute_deltas(eta, V_0, V_k, w, theta_h, beta):
    B_h_0 = np.matmul(w, V_0) - theta_h
    E_h_0 = np.tanh(beta * B_h_0)

    B_h_k = np.matmul(w, V_k) - theta_h
    E_h_k = np.tanh(beta * B_h_k)

    delta_w = eta * (np.outer(E_h_0, V_0) - np.outer(E_h_k, V_k))

    delta_theta_v = -eta * (V_0 - V_k)
    delta_theta_h = -eta * (E_h_0 - E_h_k)

    return delta_w, delta_theta_v, delta_theta_h


def kl_divergence(w, theta_v, theta_h, beta):
    all_patterns = np.array(
        [
            [-1, -1, -1],
            [1, -1, 1],
            [-1, 1, 1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, -1],
            [1, 1, 1],
        ]
    )

    unnormalized_log_probs = np.zeros(all_patterns.shape[0])

    for i, V in enumerate(all_patterns):
        V_term = np.dot(theta_v, V)

        B_h = np.matmul(w, V) - theta_h
        H_term = np.sum(np.log(2 * np.cosh(beta * B_h)))

        unnormalized_log_probs[i] = beta * V_term + H_term

    max_log_prob = np.max(unnormalized_log_probs)
    log_Z = max_log_prob + np.log(np.sum(np.exp(unnormalized_log_probs - max_log_prob)))

    log_model_probs = unnormalized_log_probs - log_Z
    model_probs = np.exp(log_model_probs)

    data_probs = np.array([0.25, 0.25, 0.25, 0.25])
    log_data_probs = np.log(data_probs)

    kl_div = np.sum(data_probs * (log_data_probs - log_model_probs[:4]))

    return kl_div, model_probs


def kl_divergence_simulation(w, theta_v, theta_h, beta, n_samples=10000, k_gibbs=100):
    all_patterns = np.array(
        [
            [-1, -1, -1],
            [1, -1, 1],
            [-1, 1, 1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, -1],
            [1, 1, 1],
        ]
    )
    V = np.random.choice([-1, 1], 3)

    M = w.shape[0]
    for _ in range(k_gibbs):
        H = update_hidden(V, M, w, theta_h, beta)
        V = update_visible(H, w, theta_v, beta)

    samples = []
    for _ in range(n_samples):
        H = update_hidden(V, M, w, theta_h, beta)
        V = update_visible(H, w, theta_v, beta)
        samples.append(V.copy())

    counts = np.zeros(4)
    for sample in samples:
        for i, pattern in enumerate(all_patterns[:4]):
            if np.array_equal(sample, pattern):
                counts[i] += 1
                break

    model_probs = counts / n_samples
    eps = 1e-10
    model_probs = np.maximum(model_probs, eps)

    data_probs = np.array([0.25, 0.25, 0.25, 0.25])
    kl_div = np.sum(data_probs * (np.log(data_probs) - np.log(model_probs[:4])))

    return kl_div, model_probs


# Tested!
def main(M, k, beta, eta, nu_max, p_0):
    data = np.array([[-1, -1, -1], [1, -1, 1], [-1, 1, 1], [1, 1, -1]])
    print(data.shape)

    w, theta_v, theta_h = intitialize_boltzmann(M)

    for nu in range(nu_max):
        print(f"nu: {nu}")
        sample = sample_data_distribution(data, p_0)

        delta_w = np.zeros((M, 3))
        delta_theta_v = np.zeros(3)
        delta_theta_h = np.zeros(M)

        for mu in range(sample.shape[0]):
            V = sample[mu, :]

            H = update_hidden(V, M, w, theta_h, beta)

            for t in range(k):
                V = update_visible(H, w, theta_v, beta)
                H = update_hidden(V, M, w, theta_h, beta)

            cur_delta_w, cur_delta_theta_v, cur_delta_theta_h = compute_deltas(
                eta, sample[mu, :], V, w, theta_h, beta
            )
            delta_w += cur_delta_w
            delta_theta_v += cur_delta_theta_v
            delta_theta_h += cur_delta_theta_h

        w += delta_w
        theta_v += delta_theta_v
        theta_h += delta_theta_h

    kl_div, _ = kl_divergence_simulation(w, theta_v, theta_h, beta)
    print(f"sim kl div: {kl_div}")


# main(M = 1, k = 3, beta = 0.4, eta = 0.01, nu_max = 50000, p_0 = 4)
main(M=2, k=3, beta=0.4, eta=0.01, nu_max=50000, p_0=4)
# main(M = 4, k = 5, beta = 0.4, eta = 0.005, nu_max = 50000, p_0 = 4)
# main(M = 8, k = 5, beta = 0.4, eta = 0.005, nu_max = 20000, p_0 = 4)

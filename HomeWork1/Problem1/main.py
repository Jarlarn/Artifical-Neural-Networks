from sampler import BooleanFunctionSampler

if __name__ == "__main__":
    NUM_SAMPLES = 10000
    for n in range(2, 6):
        sampler = BooleanFunctionSampler(n)
        frac = sampler.sample_unique(NUM_SAMPLES)
        print(f"n={n}: fraction linearly separable ≈ {frac:.6f}")

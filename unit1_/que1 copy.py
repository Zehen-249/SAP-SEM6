import numpy as np
import matplotlib.pyplot as plt

# Parameters
M = 10000  # Number of samples
N_values = [5, 10, 30, 50]  # Different sample sizes
population_size = 100000  # Population size

# Generate data from a non-normal population (e.g., uniform distribution)
population = np.random.uniform(low=0, high=10, size=population_size)

plt.figure(figsize=(10, 8))
plt.suptitle("Demonstrating the Central Limit Theorem", fontsize=16)

for i, N in enumerate(N_values):
    sample_means = []  # Store sample means

    # Draw M samples of size N
    for _ in range(M):
        sample = np.random.choice(population, size=N, replace=False)
        sample_means.append(np.mean(sample))

    # Plot histogram of sample means
    plt.subplot(2, 2, i + 1)
    plt.hist(sample_means, bins=50, density=True, alpha=0.7, color="blue", label="Sample Means")

    # Plot Gaussian fit
    mean = np.mean(sample_means)
    std = np.std(sample_means)
    x = np.linspace(mean - 4 * std, mean + 4 * std, 1000)
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    plt.plot(x, y, color="red", label="Gaussian Fit")

    plt.title(f"Sample Size (N) = {N}")
    plt.xlabel("Sample Mean")
    plt.ylabel("Density")
    plt.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

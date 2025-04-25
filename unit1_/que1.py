import matplotlib.pyplot as plt
import numpy as np

N = [10, 20, 30, 40]
M = 1000
n= 10

# Random Distribution data generation
data_binom = [np.random.binomial(n=n, p=0.2, size=(i, M)) for i in N]
# data_poiss = [np.random.poisson(5, size=(n, M)) for n in N]
# data_norm = [np.random.normal(loc=0, scale=1, size=(n, M)) for n in N]
# data_cauchy = [np.random.standard_cauchy(size=(n, M)) for n in N]

# Sample Space
data_binom = np.array(data_binom, dtype=object)
# data_poiss = np.array(data_poiss, dtype=object)
# data_norm = np.array(data_norm, dtype=object)
# data_cauchy = np.array(data_cauchy, dtype=object)
# S = [data_binom, data_poiss, data_norm, data_cauchy]
S = [data_binom]
s_mean = [[] for i in range(len(S))]
s_std = [[] for i in range(len(S))]
for i in range(len(S)):
    for j in range(len(S[i])):
        # print(np.mean(S[j]))
        s_mean[i].append(np.mean(S[i][j]))
        s_std[i].append(np.std(S[i][j]))

# Plotting
for j in range(len(S)):
    plt.figure(figsize=(8, 8))
    if j == 0:
        plt.suptitle("Binomial")
        plt.xlim(0,10)
    elif j == 1:
        plt.suptitle("Poisson")
    elif j == 2:
        plt.suptitle("Normal")
    elif j == 3:
        plt.suptitle("Cauchy (Lorentzian)")
    
    for i in range(len(S[j])):
        mean = np.mean(S[j][i], axis=0)
        plt.subplot(2, 2, i + 1)
        # plt.xlim(np.min(mean), np.max(mean))
        # plt.xticks(np.linspace(np.min(mean), np.max(mean), num=5))

        counts, bins, _ = plt.hist(mean, bins=100, density=True, alpha=0.5, color="blue", label="Histogram")
        
        # y = 10*(1 / (s_std[j][i] * np.sqrt(2 * np.pi))) * np.exp(-(((mean - s_mean[j][i]) ** 2) / (2 * (s_std[j][i] ** 2))))
        # x= x = np.linspace(np.min(mean), np.max(mean), 100)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])  # Use bin centers for Gaussian x-values
        y = (1 / (s_std[j][i] * np.sqrt(2 * np.pi))) * np.exp(-((bin_centers - s_mean[j][i]) ** 2) / (2 * (s_std[j][i] ** 2)))
        y *= max(counts) / max(y)

        plt.plot(bin_centers, y, color="red", label="Gaussian Fit")
        plt.title(f"N = {N[i]}")
        plt.legend()

plt.show()

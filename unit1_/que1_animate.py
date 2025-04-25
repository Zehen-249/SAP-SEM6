import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Parameters
N = np.arange(1, 52, 5)
M = 10000

data_binom = [np.random.binomial(n=n, p=0.5, size=(n, M)) for n in N]
data_poiss = [np.random.poisson(5, size=(n, M)) for n in N]
data_norm = [np.random.normal(loc=0, scale=1, size=(n, M)) for n in N]
data_cauchy = [np.random.standard_cauchy(size=(n, M)) for n in N]

# Sample Space
data_binom = np.array(data_binom, dtype=object)
data_poiss = np.array(data_poiss, dtype=object)
data_norm = np.array(data_norm, dtype=object)
data_cauchy = np.array(data_cauchy,dtype=object)
S = [data_binom, data_poiss, data_norm, data_cauchy]

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlabel('X-axis')
ax.set_ylabel('Density')

hist_bars = None
gaussian_curve = None

def update(frame):
    global hist_bars, gaussian_curve
    
    # Determine which distribution to use
    j = frame // len(N)  
    i = frame % len(N)   

    mean = np.mean(S[j][i], axis=0)
    std_S = np.std(mean)
    
    # Clear the previous plot
    ax.clear()
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Density')

    counts, bins, _ = ax.hist(mean, bins=100, density=True, alpha=0.5, color="blue", label="Histogram")
    

    y = (1 / (std_S * np.sqrt(2 * np.pi))) * np.exp(-(bins[:-1] - np.mean(mean)) ** 2 / (2 * std_S ** 2))
    gaussian_curve, = ax.plot(bins[:-1], y, color="red", label="Gaussian Fit")

    max_hist = np.max(counts)
    ax.set_xlim(np.min(mean) - 1, np.max(mean) + 1)
    ax.set_ylim(0, max_hist + 0.05)  

    dist_titles = ["Binomial", "Poisson", "Normal", "Cauchy"]
    ax.set_title(f"{dist_titles[j]} Distribution - N = {N[i]}")

    return hist_bars, gaussian_curve

ani = FuncAnimation(fig, update, frames=len(N) * len(S), repeat=False, blit=False, interval=300)

# Save the animation
ani.save("Central Limit Theorem.mp4", fps=0.5)

# plt.show()

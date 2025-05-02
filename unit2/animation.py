import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm, binom

def likelihood(theta, M, N):
    return binom.pmf(M, N, theta)

def compute_posterior(prior, likelihood_vals, theta):
    unnormalized_posterior = prior * likelihood_vals
    return unnormalized_posterior / np.trapezoid(unnormalized_posterior, theta)

theta = np.linspace(0, 1, 1000)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].set_xlabel('θ')
axs[0].legend()
axs[1].set_xlabel('θ')
axs[1].legend()

line_prior_beta, = axs[0].plot([], [], 'b--', label='Beta Prior')
line_likelihood_beta, = axs[0].plot([], [], 'k:', label='Likelihood')
line_posterior_beta, = axs[0].plot([], [], 'r-', label='Posterior')

line_prior_gauss, = axs[1].plot([], [], 'b--', label='Gaussian Prior')
line_likelihood_gauss, = axs[1].plot([], [], 'k:', label='Likelihood')
line_posterior_gauss, = axs[1].plot([], [], 'r-', label='Posterior')

params = [
    [0,0],[1,1],[2,2],[2,3],[2,4],[3,8],[5,16],[10,32],
    [20,64],[40,128],[80,256],[160,512],[320,1024],[640,2048],[1280,4096]
]

for M, N in params:
    likelihood_vals = likelihood(theta, M, N)

    alpha_val = 4
    beta_val = 2
    prior_beta = beta.pdf(theta, alpha_val, beta_val)
    prior_gauss = norm.pdf(theta, 0.5, 0.1)

    # Posteriors
    posterior_beta = compute_posterior(prior_beta, likelihood_vals, theta)
    posterior_gauss = compute_posterior(prior_gauss, likelihood_vals, theta)
    axs[0].set_xlim(np.min(theta), np.max(theta))
    axs[0].set_ylim(0, np.max(posterior_beta) +1)
    axs[1].set_xlim(np.min(theta), np.max(theta))
    axs[1].set_ylim(0, np.max(posterior_gauss) +1) 

    line_prior_beta.set_data(theta, prior_beta)
    line_likelihood_beta.set_data(theta, likelihood_vals / np.max(likelihood_vals))
    line_posterior_beta.set_data(theta, posterior_beta)

    line_prior_gauss.set_data(theta, prior_gauss)
    line_likelihood_gauss.set_data(theta, likelihood_vals / np.max(likelihood_vals))
    line_posterior_gauss.set_data(theta, posterior_gauss)

    axs[0].set_title(f'Beta Prior (M={M}, N={N})')
    axs[1].set_title(f'Gaussian Prior (M={M}, N={N})')

    fig.canvas.draw()
    plt.pause(1) 
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import (
    beta,
    norm,
    binom
) 

def likelihood(theta, M, N):
    return binom.pmf(M, N, theta)

def compute_posterior(prior, likelihood_vals):
    unnormalized_posterior = prior * likelihood_vals
    return unnormalized_posterior / np.trapezoid(unnormalized_posterior, theta)

def plot_distribution(theta, prior_beta, prior_gauss, likelihood_vals, posterior_beta, posterior_gauss, title):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(theta, prior_beta, 'b--', label='Beta Prior')
    plt.plot(theta, likelihood_vals / np.max(likelihood_vals), 'k:', label='Likelihood')
    plt.plot(theta, posterior_beta, 'r-', label='Posterior')
    plt.title(title)
    plt.xlabel('θ')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(theta, prior_gauss, 'b--', label='Gaussian Prior')
    plt.plot(theta, likelihood_vals / np.max(likelihood_vals), 'k:', label='Likelihood')
    plt.plot(theta, posterior_gauss, 'r-', label='Posterior')
    plt.title(title)
    plt.xlabel('θ')
    plt.legend()

def solve(M,N, theta, myu,sigma):
    likelihood_vals = likelihood(theta, M, N)
    alpha_val = M + 1
    beta_val = N - M + 1
    prior_beta = beta.pdf(theta, alpha_val, beta_val)
    prior_gauss = norm.pdf(theta, myu, sigma)
    posterior_beta = compute_posterior(prior_beta, likelihood_vals)
    posterior_gauss = compute_posterior(prior_gauss, likelihood_vals)
    plot_distribution(theta, prior_beta,prior_gauss, likelihood_vals, posterior_beta,posterior_gauss, f'Posterior Distribution (M={M}, N={N})')
    plt.show()
theta = np.linspace(0.0001, 0.9999, 998)
params = [
    [0,0],[1,1],[2,2],[2,3],[2,4],[3,8],[5,16],[10,32],
    [20,64],[40,128],[80,256],[160,512],[320,1024],[640,2048],[1280,4096]
]
for M, N in params:
    solve(M, N, theta, 0.5, 0.1)
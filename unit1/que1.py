
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson, binom

def plot_distribution_with_clt(N: int, M: int, plot_num: list, dist: str = 'normal', param: tuple = None):
    if param :
        if dist == 'normal':
            data = np.random.normal(loc=param[0], scale=param[1], size=(N, M))
        elif dist == "poisson":
            data = np.random.poisson(lam=param[0], size=(N, M))
        elif dist == "binomial":
            data = np.random.binomial(n=param[0], p=param[1], size=(N, M))
        elif dist =="cauchy":
            data = np.random.standard_cauchy(size=(N, M)) 
        
        mean_vect = data.mean(axis=0)
        mean_data = mean_vect.mean()
        std_data = mean_vect.std()

        
        axs[plot_num[0],plot_num[1]].hist(mean_vect, bins=30, density=True, color='blue', edgecolor='k', label=dist)
        x = np.linspace(min(mean_vect), max(mean_vect), 100)
        axs[plot_num[0],plot_num[1]].plot(x, norm.pdf(x, mean_data, std_data), 'r--', label='Normal Fit')
        axs[plot_num[0],plot_num[1]].set_title(f"{dist} Distribution")
        axs[plot_num[0],plot_num[1]].legend()

fig, axs = plt.subplots(2, 2, figsize=(8, 8))
fig.suptitle(f'N = {100} and N = {1000}', fontsize=16)
plot_distribution_with_clt(N=100, M=1000,plot_num=[0,0], dist="normal", param=(50,0.5)) 
plot_distribution_with_clt(N=100, M=1000,plot_num=[0,1], dist="poisson", param=(50,)) 
plot_distribution_with_clt(N=100, M=1000,plot_num=[1,0], dist="binomial", param=(10,0.2)) 
plot_distribution_with_clt(N=100, M=1000,plot_num=[1,1], dist="cauchy", param=(10,)) 
plt.tight_layout()
plt.show()

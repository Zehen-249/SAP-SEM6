import numpy as np
from scipy.stats import binomtest

def simulate_coin_tosses(n, q):
    tosses = np.random.binomial(n=1, p=q, size=n)
    num_heads = np.sum(tosses)
    return num_heads


n = 100  
q_actual = 0.5  
num_heads = simulate_coin_tosses(n, q_actual)

# Binomial Test
# Null hypothesis: q = 0.5
result = binomtest(num_heads, n, p=0.5, alternative='two-sided')

print(f"Number of heads: {num_heads} out of {n}")
print(f"P-value: {result.pvalue:.4f}")
if result.pvalue < 0.05:
    print("Reject the null hypothesis (evidence that coin is not fair).")
else:
    print("Fail to reject the null hypothesis (no strong evidence against fairness).")

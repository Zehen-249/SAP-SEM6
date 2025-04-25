import numpy as np
from scipy.stats import binomtest

def simulate_coin_tosses(n, q):
    tosses = np.random.binomial(n=1, p=q, size=n)
    num_heads = np.sum(tosses)
    return num_heads


n = 1000  
myu = 0.4
num_heads = simulate_coin_tosses(n, myu)
x_bar = num_heads / n
print(f"Number of heads: {num_heads} out of {n}")

# Binomial Test
# Null hypothesis(H0): q = 0.5

print("TWO-SIDED TEST")
# Alternate Hypothesis Ha: q != 0.5
print("Z-test")
z_two = (x_bar - 0.5) / np.sqrt((myu * (1 - myu)) / n)
print(f"Z-value: {z_two}")
if abs(z_two) > 1.96:
    print("Reject the null hypothesis (evidence that coin is not fair).")
else:
    print("Fail to reject the null hypothesis (no strong evidence against fairness).")
print("p-value test")
result_two = binomtest(num_heads, n, p=0.5, alternative='two-sided')
print(f"p-value: {result_two.pvalue}")
if result_two.pvalue < 0.05:
    print("Reject the null hypothesis (evidence that coin is not fair).")
else:
    print("Fail to reject the null hypothesis (no strong evidence against fairness).")

print("LEFT-SIDED TEST")
# Alternate Hypothesis Ha: q < 0.5
print("Z-test")
z_left = (x_bar - 0.5) / np.sqrt((myu * (1 - myu)) / n)
print(f"Z-value: {z_two}")
if z_left < -1.645:
    print("Reject the null hypothesis (evidence that coin is biased for tails).")
else:
    print("Fail to reject the null hypothesis (no strong evidence that coin is biased for tails).")
print("p-value test")
result_left = binomtest(num_heads, n, p=0.5, alternative='less')
if result_left.pvalue < 0.05:
    print("Reject the null hypothesis (evidence that coin is biased for tails).")
else:
    print("Fail to reject the null hypothesis (no strong evidence that coin is biased for tails).")


print("RIGHT-SIDED TEST")
# Alternate Hypothesis Ha: q > 0.5
print("Z-test")
z_left = (x_bar - 0.5) / np.sqrt((myu * (1 - myu)) / n)
print(f"Z-value: {z_two}")
if z_left > 1.645:
    print("Reject the null hypothesis (evidence that coin is biased for heads).")
else:
    print("Fail to reject the null hypothesis (no strong evidence that coin is biased for heads).")
print("p-value test")
result_right = binomtest(num_heads, n, p=0.5, alternative='greater')
if result_right.pvalue < 0.05:
    print("Reject the null hypothesis (evidence that coin is biased for heads).")
else:
    print("Fail to reject the null hypothesis (no strong evidence that coin is biased for heads).")


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

np.random.seed(42)

n = 100
Xi = np.random.uniform(0, 10, size=n)
Yi = 3 * Xi + 5 + np.random.normal(0, 2, size=n)  
sigma_X = 0.5  
sigma_Y = 2   

A = np.vstack([Xi, np.ones_like(Xi)]).T
beta1, beta0 = np.linalg.lstsq(A, Yi, rcond=None)[0]

Yi_pred = beta0 + beta1 * Xi


residuals = Yi - Yi_pred


correlation_matrix = np.corrcoef(Xi, Yi)
correlation_coefficient = correlation_matrix[0, 1]

n = len(Xi)
mean_X = np.mean(Xi)
mean_Y = np.mean(Yi)


residual_variance = np.sum(residuals**2) / (n - 2)


se_beta1 = np.sqrt(residual_variance / np.sum((Xi - mean_X)**2))
se_beta0 = np.sqrt(residual_variance * (1/n + mean_X**2 / np.sum((Xi - mean_X)**2)))


t_90 = stats.t.ppf(0.95, n - 2) 


CI_beta1 = (beta1 - t_90 * se_beta1, beta1 + t_90 * se_beta1)
CI_beta0 = (beta0 - t_90 * se_beta0, beta0 + t_90 * se_beta0)

plt.figure(figsize=(8, 6))
plt.errorbar(Xi, Yi, xerr=sigma_X, yerr=sigma_Y, fmt='o', label="Data points", color='blue', alpha=0.5)
plt.plot(Xi, Yi_pred, label=f'Regression line: Y = {beta0:.2f} + {beta1:.2f} * X', color='red', linewidth=2)

X_range = np.linspace(min(Xi), max(Xi), 100)
Y_range = beta0 + beta1 * X_range
Y_range_upper = Y_range + t_90 * se_beta1
Y_range_lower = Y_range - t_90 * se_beta1
plt.fill_between(X_range, Y_range_upper, Y_range_lower, color='orange', alpha=0.3, label='90% Confidence Interval')


plt.xlabel("X (Independent variable)")
plt.ylabel("Y (Dependent variable)")
plt.title("Scatter plot with Linear Regression, Error Bars, and Confidence Interval")

plt.legend()

plt.show()

print(f"Regression Line: Y = {beta0:.2f} + {beta1:.2f} * X")
print(f"Correlation Coefficient: {correlation_coefficient:.3f}")
print(f"90% Confidence Interval for Slope (β1): {CI_beta1}")
print(f"90% Confidence Interval for Intercept (β0): {CI_beta0}")

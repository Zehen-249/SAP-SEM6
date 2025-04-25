import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Step 1: Generate continuous random data (Example: Normal distribution)
np.random.seed(42)
X_cont = np.random.normal(loc=10, scale=2, size=1000)  # Mean=10, StdDev=2
Y_cont = np.random.normal(loc=1, scale=1, size=1000)   # Mean=5, StdDev=1

# Step 2: Create a 2D histogram (binned joint probability)
bin_size = 30  # Number of bins
hist, x_edges, y_edges = np.histogram2d(X_cont, Y_cont, bins=bin_size, density=True)

# Convert bin edges to centers
x_centers = (x_edges[:-1] + x_edges[1:]) / 2
y_centers = (y_edges[:-1] + y_edges[1:]) / 2

# Step 3: Convert to DataFrame
df_cont = pd.DataFrame(hist, index=x_centers, columns=y_centers)

# Step 4: Plot Heatmap
plt.figure(figsize=(7, 6))
sns.heatmap(df_cont, cmap="viridis", annot=False, linewidths=0)

plt.xlabel("Y Values")
plt.ylabel("X Values")
plt.title("Joint Probability Density (Continuous Variables)")

plt.show()

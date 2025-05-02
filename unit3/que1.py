import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# LSF
def LST(X,Y,n):
    sum_xiyi = np.sum(X*Y)
    sum_xi = np.sum(X)
    sum_xi_squared = np.sum(X**2)
    sum_yi = np.sum(Y)

    m = (n * sum_xiyi - sum_xi * sum_yi) / (n * sum_xi_squared - sum_xi ** 2)
    b = (sum_yi - m * sum_xi) / n

    return m ,b

X = np.arange(0,30,1)
Y = 0.95 * X + 0.8 + np.random.normal(0, 5, size=len(X)) 

df = pd.DataFrame({'X': X, 'Y': Y})

m,b = LST(X,Y,len(X))
plt.figure(figsize=(10, 6))
plt.scatter(X,Y,label='Data Points', color='blue')

x = np.arange(0,50,0.1)
plt.plot(x,m*x + b, label='Fitted Line', color='red')
plt.xlabel('X Points')
plt.ylabel('Y Points')
plt.title('Least Squares Fit manually')
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 6))
reg_plot= sns.regplot(x=X, y=Y, ci=90, line_kws={'color': 'red'}, scatter_kws={'color': 'blue'})
reg_plot.set_title('Least Squares Fit using regplot')
reg_plot.set_xlabel('X Points')
reg_plot.set_ylabel('Y Points')

g = sns.jointplot(x='X', y='Y', data=df, kind='reg', ci=90, color='red',height=4, ratio=5)
g.ax_joint.plot(x, m*x + b, label='LSF Line (Manual)', color='blue') #for verification
g.ax_joint.legend()
g.figure.suptitle('Jointplot with Regression Line')
plt.tight_layout()

cov_XY = np.sum((X - np.mean(X)) * (Y - np.mean(Y))) / (len(X) )
cor_XY = cov_XY / (np.std(X) * np.std(Y))
print(f"Correlation_Coefficient, Manually: {cor_XY}")
print(f"Correlation_Coefficient, Using Panda: {df.corr()}")

error = Y - (m*X + b)
std_error = np.std(error)
print(f"Standard Error: {std_error}")
plt.figure(figsize=(10, 6))
plt.errorbar(X, Y, yerr=std_error, fmt='o', label='Data with error bars', capsize=5, color='blue')
plt.plot(X, (m*X + b), 'r-', label='Fitted Line',color='red')
plt.xlabel('X Points')
plt.ylabel('Y Points')
plt.title('Data with Error Bars')
plt.legend()
plt.tight_layout()
plt.show()  
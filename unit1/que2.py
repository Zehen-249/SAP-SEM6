import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def coutFreq(pair,X_,Y_):
    x=pair[0]
    y = pair[1]
    count = 0

    for i in range(len(X_)):
        if X_[i]==x and Y_[i]==y:
            count+=1
    return count

def createDF(data):
    rows = sorted(set(k[0] for k in data.keys()))
    cols = sorted(set(k[1] for k in data.keys()))

    df = pd.DataFrame(index=rows, columns=cols)

    for (x, y), freq in data.items():
        df.loc[x, y] = freq

    return df

N=1000 
# X_real = np.random.randint(0,50,N)
# Y_real = np.random.randint(50,100,N)

X_real = np.random.poisson(lam = 5,size = N)
Y_real = np.random.binomial(p=0.5,n=10,size =N)
data = {}
for i in X_real:

    for j in Y_real:
        data[(i,j)] = 0

# for i,j in data.items():
#     print(i,j)

for pair,freq in data.items():
    data[pair] = coutFreq(pair,X_real,Y_real)/N
    # print(f"{pair} | {data[pair]}")

    
# print_table(data)
df_real = createDF(data)
# df_real["marginal Prob of X"] = df_real.sum(axis=1)
# df_real.loc["marginal Prob of Y"] = df_real.sum(axis=0)
print(df_real)

df_real = df_real.apply(pd.to_numeric)


plt.figure(figsize=(6, 5))  
sns.heatmap(df_real, annot=True, cmap="coolwarm", fmt=".3f", linewidths=0)


plt.xlabel("Y-Axis Values")
plt.ylabel("X-Axis Values")
plt.title("Heatmap of Data")


plt.show()
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


def createDF(X,Y):
    data = {}
    for i in X:
        for j in Y:
            data[(i,j)] = 0

    for pair,freq in data.items():
        data[pair] = coutFreq(pair,X,Y)/N

    rows = sorted(set(k[0] for k in data.keys()))
    cols = sorted(set(k[1] for k in data.keys()))

    df = pd.DataFrame(index=rows, columns=cols)

    for (x, y), freq in data.items():
        df.loc[x, y] = freq

    return df

N=1000  #No. of pairs
X_dis = np.random.poisson(lam = 5,size = N)
Y_dis = np.random.binomial(p=0.5,n=10,size =N)
    
df_dis = createDF(X_dis,Y_dis)

# df_dis["marginal Prob of X"] = df_dis.sum(axis=1)
# df_dis.loc["marginal Prob of Y"] = df_dis.sum(axis=0)
print(df_dis)

df_dis = df_dis.apply(pd.to_numeric)

plt.figure(figsize=(6, 5)) 
sns.heatmap(df_dis, annot=True, cmap="coolwarm", fmt=".3f", linewidths=0)

plt.xlabel("Y-Axis Values")
plt.ylabel("X-Axis Values")
plt.title("Heatmap of Data")

plt.tight_layout()

plt.show()
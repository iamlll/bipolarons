import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fig = plt.figure(figsize=(5.5,4.5))
ax = fig.add_subplot(111)
#read in CSV as Pandas dataframe
df = pd.read_csv("gauss_fixedU.csv")
pd.set_option("display.max.columns", None)
#print(df.head())
#df.plot(x="eta", y=["E_inf"],ax=ax)
x = df["eta"].values
f = df["E_opt"].values
g = df["E_inf"].values
stop = np.where(x<0.095)[0]
print((x[stop],f[stop]))
ax.plot(x,g,label='$E_\infty$')
ax.plot(x[stop], f[stop], label='$E_{opt}$')
idx = np.argwhere(np.diff(np.sign(f - g))).flatten()
print(x[idx])
ax.plot(x[idx], f[idx], 'ro')
ax.set_xlim(0,0.2)
ax.set_ylim(bottom=-1,top=-0.6)
ax.set_xlabel("$\eta$")
ax.set_ylabel("E/K")
ax.legend(loc=2)

plt.tight_layout()
plt.savefig("E_vs_eta_fixedU.eps")
plt.show()

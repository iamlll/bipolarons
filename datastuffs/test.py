import multiprocessing 
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.optimize import minimize
from scipy.special import erfcx
import time
import warnings
from itertools import product
import sys
from scipy.optimize import curve_fit
import nagano_cy as nag
import matplotlib.pyplot as plt
import pandas as pd

def Plot_E_vs_a(csvname):
    '''ONLY use with files generated from GenE_vs_a() !!!'''

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    #read in CSV as Pandas dataframe
    df = pd.read_csv(csvname)
    alphas = np.array([(1-n)*U/2. for n,U in zip(df["eta"].values, df['U'].values)])
    df['alpha'] = alphas
    alphas = np.unique(alphas)

    for alpha in alphas:
        #divide into energy arrays by fixed alpha value
        Es = df[df['alpha'] == alpha]['E_inf'].values
        sigs = df[df['alpha'] == alpha]['y'].values
        ayes = df[df['alpha'] == alpha]['a'].values
        ax.plot(ayes,Es,label='$\\alpha=$' + str(alpha))
        ax2.plot(ayes,sigs,label='$\\alpha=$' + str(alpha))

    ax.legend(loc=2)
    ax.set_xlabel("$a$")
    ax.set_ylabel("E(a)/K")
    ax2.legend(loc=2)
    ax2.set_xlabel("$a$")
    ax2.set_ylabel("$y$")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    import pandas as pd
    Plot_E_vs_a("./data/nak_E(a).csv")



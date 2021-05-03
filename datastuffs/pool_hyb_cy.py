import multiprocessing 
import numpy as np
from matplotlib.ticker import MaxNLocator
#from scipy import optimize, integrate
#import matplotlib.pyplot as plt
#from scipy.special import erf, erfc
from scipy.optimize import minimize
import time
import warnings
from itertools import product
import sys
import baby_integral as bee

#define various constants
elec = 1.602E-19*2997924580 #convert C to statC
hbar = 1.054E-34 #J*s
m = 9.11E-31 #kg
w = 0.1*1.602E-19/hbar
epssr = 23000
epsinf = 2.394**2
conv = 1E-9/1.602E-19 #convert statC (expressions with elec^2) to eV
convJ = 1/1.602E-19 #convert J to eV
eta_STO = epsinf/epssr
alpha = (elec**2*1E-9)/hbar*np.sqrt(m/(2*hbar*w))*1/epsinf*(1 - epsinf/epssr) #convert statC to J*m
l = np.sqrt(hbar/(2*m*w))   
U_STO = elec**2/(epsinf*hbar)*np.sqrt(2*m/(hbar*w))*1E-9 #convert statC in e^2 term into J to make U dimensionless

def format_filename(filename, param_val):
    addon_str = str(param_val).split(".")
    if len(addon_str)>1:
        if addon_str[1] == '0': addon_str.remove(addon_str[1])
    for i,string in enumerate(addon_str):
        filename += addon_str[i]
        if i < len(addon_str)-1: filename += "-"
    return filename

if __name__ == '__main__': 
    import pandas as pd
    a = float(sys.argv[1])
    print(a)
    csvname = "./data/cy_" + format_filename("hyb_a_",a) + ".csv"
    
    #print(csvname)
    ayes = np.linspace(1E-15,1,2)
    ns = np.linspace(0,.999,40) #60, eta values
    #ns = [eta_STO]
    Us = np.linspace(0.001,5,15) #30
    #Us = [U_STO]

    df={}
    quantities=['eta','U','s_opt','y_opt','E_opt','s_inf','y_inf','E_inf','E_binding']
    for i in quantities:
        df[i]=[]

    tic = time.perf_counter()
    with multiprocessing.Pool(processes=4) as pool:
        job_args = [(a, n,u) for a,n,u in product(ayes,ns,Us)]
        results = pool.map(bee.minimization, job_args)
        print(results[0])
        for res in results:
            for name, val in zip(quantities, res): 
                df[name].append(val)
    toc = time.perf_counter()
    print(f"time taken: {toc-tic:0.4f} s, {(toc-tic)/60:0.3f} min")
    data = pd.DataFrame(df)
    print(data)
    data.to_csv(csvname,sep=',',index=False)

import multiprocessing 
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.optimize import minimize
import time
import warnings
from itertools import product
import sys
import var_a_cy as avar
import asym_U1 as asym
import nagano_cy as nag
import indep_params as ip

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
    #ns = [eta_STO]
    #Us = [U_STO]
    
    #STO, KTO, PbS, PbSe, PbTe, SnTe, GeTe
    #ns = [eta_STO, 0.0012, 9.05E-2, 8.18E-2, 8.26E-2, 3.75E-2, 8E-2]
    #Us = [U_STO, 12.6, 2.6, 2.58, 1.87, 3.12, 0.47]

    #strong coupling?
    #ns=[0.]
    #Us=[10.,100.,500.,1000.,5000.]

    #zoomed in
    #ns = np.linspace(0,.2,100) #60, eta values
    
    #"regular" settings
    #ns = np.linspace(0,.999,60) #60, eta values
    #Us = np.linspace(0.001,10,80)
    
    #zoom in on large Us to study power law behavior
    #ns = np.linspace(0,0.5,100)
    #Us = np.geomspace(.1,16,num=200)

    #log spacing to study scaling behavior over decades
    #Us = np.geomspace(1E-5,5000,num=100)
    
    #range of Us for eta=0 to compare E_inf behavior with 2*single polaron behavior (as fn of alpha)
    ns=[0.]
    Us = np.linspace(1E-3,20,100)

    #csvname = "./data/asym_U1_v2.csv"
    #csvname = "./data/nagano.csv"
    csvname = "./data/hyb_alpha.csv"
    #csvname = "./data/indep_params.csv"
    #csvname = "./data/PDMatParams.csv" #material parameters for the hybrid calc

    df={}
    quantities=['eta','U','a_opt','s_opt','y_opt','E_opt','a_inf','s_inf','y_inf','E_inf','E_binding']
    #quantities=['eta','U','a_r','s_r','a_R','s_R','E_bi']
    #quantities = ['eta','U','s','y','E_bi']

    for i in quantities:
        df[i]=[]

    tic = time.perf_counter()
    with multiprocessing.Pool(processes=4) as pool:
        job_args = [(n,u) for n,u in product(ns,Us)]
        #results = pool.map(nag.min_E_a_1, job_args)
        results = pool.map(avar.minimization, job_args)
        print(results[0])
        for res in results:
            for name, val in zip(quantities, res): 
                df[name].append(val)
    toc = time.perf_counter()
    print(f"time taken: {toc-tic:0.4f} s, {(toc-tic)/60:0.3f} min")
    data = pd.DataFrame(df)
    pd.set_option("display.max.columns",None)
    print(data)
    data.to_csv(csvname,sep=',',index=False)
    

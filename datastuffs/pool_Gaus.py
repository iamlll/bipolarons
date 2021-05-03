import multiprocessing 
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy import optimize, integrate
import matplotlib.pyplot as plt
from sympy import solve, symbols
from scipy.special import erf
from scipy.optimize import minimize
from math import isnan, isinf
import time
import warnings
from itertools import product

#various constants
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

def opt_Gauss(x, n, U):
    '''
    Inputs:
        x[0] = y = d/sigma
        x[1] = s = sigma/l
        U = ratio of Coulomb energy unit to KE energy unit (effective Rydberg/hw): U = e^2/(epsinf*l) / hbar^2/(2ml^2)
        n = eta = epsinf/epssr
    Output:
        E/K, ratio of total energy expectation value to kinetic energy coeff = K = hbar^2/(2ml^2)
    '''
    KE = 1/x[1]**2 *(3-1/2* x[0]**2/(np.exp(x[0]**2/2)+1))
    coul = U/x[1] * (1/x[0]*erf(x[0]/np.sqrt(2)) + np.sqrt(2/np.pi)* np.exp(-x[0]**2/2))/(1+np.exp(-x[0]**2/2))
    e_ph = -(1-n)*U/(x[1]*(1 + np.exp(-x[0]**2/2))**2)* np.sqrt(2/np.pi) * (1+ 2*np.exp(-x[0]**2) + \
            1/x[0]*np.sqrt(np.pi/2)*(erf(x[0]/np.sqrt(2)) + 8*np.exp(-x[0]**2/2)*erf(x[0]/(2*np.sqrt(2)))))
    E = KE + e_ph + coul
    return E

def E_ana(n,U):
    '''Optimized symmetric Gaussian energy evaluated at y->infty, normalized by KE'''
    return -(1-n)**2*U**2/(6*np.pi)

def minimization(n,u):
    bnds = ((1E-3,10), (1E-3, 10)) #y,s
    guess = np.array([1.,1.,])
    result = minimize(opt_Gauss,guess,args=(n,u),bounds=bnds)
    minvals = result.x
    E_opt = opt_Gauss(minvals,n,u)
    E_inf = E_ana(n,u)
    E_binding = (E_opt - E_inf)/np.abs(E_inf)
    return n,u,minvals[1],minvals[0],E_opt,0.,0.,E_inf,E_binding

if __name__ == '__main__': 
    import pandas as pd
    ns = np.linspace(0,.999,200) #60; eta values
    #ns = [eta_STO]
    #Us = np.linspace(0.001,5,30)
    U_STO = elec**2/(epsinf*hbar)*np.sqrt(2*m/(hbar*w))*1E-9 #convert statC in e^2 term into J to make U dimensionless
    Us = [U_STO]

    tic = time.perf_counter()
    df={}
    quantities=['eta','U','s_opt','y_opt','E_opt','s_inf','y_inf','E_inf','E_binding']
    for i in quantities:
        df[i]=[]

    with multiprocessing.Pool(processes=4) as pool:
        results = pool.starmap(minimization, product(ns,Us))
        print(results[0])
        for res in results:
            for name, val in zip(quantities, res): 
                df[name].append(val)
    toc = time.perf_counter()
    print(f"time taken: {toc-tic:0.4f} s, {(toc-tic)/60:0.3f} min")
    data = pd.DataFrame(df)
    print(data)
    data.to_csv("./data/gauss_fixedU.csv",sep=',',index=False)
    

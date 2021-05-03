import multiprocessing 
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy import optimize, integrate
import matplotlib.pyplot as plt
from sympy import solve, symbols
from scipy.special import erf, erfc
from scipy.optimize import minimize
from math import isnan, isinf
import time
import warnings
from itertools import product
import sys

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
U_STO = elec**2/(epsinf*hbar)*np.sqrt(2*m/(hbar*w))*1E-9 #convert statC in e^2 term into J to make U dimensionless

#a=1E-15 #magic param

def format_filename(filename, param_val):
    addon_str = str(param_val).split(".")
    if len(addon_str)>1:
        if addon_str[1] == '0': addon_str.remove(addon_str[1])
    for i,string in enumerate(addon_str):
        filename += addon_str[i]
        if i < len(addon_str)-1: filename += "-"
    return filename

def i1(b,c,x,option=0):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        if option == 0:
            try:
                integral = integrate.quad(lambda u: np.exp(-u**2)/(u**2+b) * (c*(1+2*np.exp(-x[0]**2)) + \
                    np.sin(c*u)/u + 8*np.exp(-x[0]**2/2)*np.sin(c*u/2)/u),0,10)
                return integral[0]
            except integrate.IntegrationWarning:
                #print ('Uh oh: option ' + str(option) + '\tb: ' + str(b) + '\tc: ' + str(c))
                integral = integrate.quad(lambda u: np.exp(-u**2)/(u**2+b) * ((np.sin(c*u)/u-c) + \
                    4*np.exp(-x[0]**2/2)*(2*np.sin(c*u/2)/u - c)),0,10)
                val = erfc(np.sqrt(b))
                if val == 0: val2 = 0
                else: val2 = np.exp(b)
                ana = (1+2*np.exp(-x[0]**2/2) + np.exp(-x[0]**2))*np.pi*c/np.sqrt(b) * val2* val
                return integral[0] + ana

            except RuntimeWarning:
                print ('Raised! option ' + str(option) + '\tb: ' + str(b) + '\tc: ' + str(c))
        else:
            try:
                integral = integrate.quad(lambda u: np.exp(-u**2)/(u**2+b) * (c+ np.sin(c*u)/u ),0,10)
                return integral[0]

            except integrate.IntegrationWarning:
                #print ('Uh oh: option ' + str(option) + '\tx: ' + str(x) + '\tb: ' + str(b) + '\tc: ' + str(c))
                integral = integrate.fixed_quad(lambda u: np.exp(-u**2)/(u**2+b) * (np.sin(c*u)/u-c),0,10,n=50)
                val = erfc(np.sqrt(b))
                if val == 0: val2 = 0
                else: 
                    val2 = np.exp(b)
                print("val ",val,"val2 ",val2)
                ana = np.pi*c/np.sqrt(b) * val* val2
                return integral[0] + ana
            except RuntimeWarning:
                print ('Raised! option ' + str(option) + '\tx: ' + str(x) + '\tb: ' + str(b) + '\tc: ' + str(c))

def opt_hybE(x, n, U, a):
    '''
    Inputs:
        x[0] = y = d/sigma
        x[1] = s = sigma/l
        U = ratio of Coulomb energy unit to KE energy unit (effective Rydberg/hw): U = e^2/(epsinf*l) / hbar^2/(2ml^2)
        n = eta = epsinf/epssr
        a = magic transformation parameter from Huybrechts
    Output:
        E/K, ratio of total energy expectation value to kinetic energy coeff = K = hbar^2/(2ml^2)
    '''
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            A = (1-a/2)**2 + (a/2)**2
            b = A*x[1]**2/a**2
            c = x[0]*np.sqrt(2/A)
            KE = 1/x[1]**2 *(3-1/2* x[0]**2/(np.exp(x[0]**2/2)+1))
            coul = U/x[1] * (1/x[0]*erf(x[0]/np.sqrt(2)) + np.sqrt(2/np.pi)* np.exp(-x[0]**2/2))/(1+np.exp(-x[0]**2/2))
            const = -(1-n)*U* 2/(np.pi*(1 + np.exp(-x[0]**2/2))**2)* A*x[1]/(a**2*x[0])
            e_ph = const* i1(b,c,x,0)
            E = KE + e_ph + coul
            return E
        except RuntimeWarning:
            print ('Raised! x: ' + str(x) + "\tn: " + str(n) + "\tU: " + str(U))

def E_infty(x,n,U,a):
    '''Hybrid calc energy evaluated at y->infty, normalized by KE'''
    A = (1-a/2)**2 + (a/2)**2
    b = A*x[1]**2/a**2
    c = x[0]*np.sqrt(2/A)
    KE = 3/x[1]**2
    coul = U/x[1] * 1/x[0]*erf(x[0]/np.sqrt(2))
    const = -(1-n)*U* 2/np.pi* A*x[1]/(a**2*x[0])
    e_ph = const* i1(b,c,x,1)
    E = KE + e_ph + coul
    return E


def minimization(args):
    a,n,u = args
    bnds = ((1E-3,10), (1E-3, 10)) #y,s
    guess = np.array([1.,1.,])
    guess_inf = np.array([5000.,1.,])
    bnds_inf = ((5000,5010), (1E-3, 1E10)) #y,s

    result = minimize(opt_hybE,guess,args=(n,u,a),bounds=bnds)
    minvals = result.x
    E_opt = opt_hybE(minvals,n,u,a)

    #Find E_infty semi-analytically
    res_inf = minimize(E_infty,guess_inf,args=(n,u,a),bounds=bnds_inf)
    m_inf = res_inf.x
    E_inf = E_infty(m_inf,n,u,a)

    E_binding = (E_opt - E_inf)/np.abs(E_inf)
    return n,u,minvals[1],minvals[0],E_opt,m_inf[1],m_inf[0],E_inf,E_binding

if __name__ == '__main__': 
    import pandas as pd
    a = float(sys.argv[1])
    print(a)
    csvname = "./data/" + format_filename("hyb_a_",a) + ".csv"
    print(csvname)

    ns = np.linspace(0,.999,60) #eta values
    #ns = [eta_STO]
    Us = np.linspace(0.001,5,30)
    #Us = [U_STO]

    df={}
    quantities=['eta','U','s_opt','y_opt','E_opt','s_inf','y_inf','E_inf','E_binding']
    for i in quantities:
        df[i]=[]

    tic = time.perf_counter()
    with multiprocessing.Pool(processes=4) as pool:
        job_args = [(a, n,u) for n,u in product(ns,Us)]
        results = pool.map(minimization, job_args)
        #results = pool.starmap(minimization, product(ns,Us))
        print(results[0])
        for res in results:
            for name, val in zip(quantities, res): 
                df[name].append(val)
    toc = time.perf_counter()
    print(f"time taken: {toc-tic:0.4f} s, {(toc-tic)/60:0.3f} min")
    data = pd.DataFrame(df)
    print(data)
    #data.to_csv(csvname,sep=',',index=False)

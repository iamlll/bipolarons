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
#print(l)
U_STO = elec**2/(epsinf*hbar)*np.sqrt(2*m/(hbar*w))*1E-9 #convert statC in e^2 term into J to make U dimensionless
#print(alpha*hbar*w*convJ)

def V_eff(x,n,U):
    coul = U/x[1] * (1/x[0]*erf(x[0]/np.sqrt(2)) + np.sqrt(2/np.pi)* np.exp(-x[0]**2/2))/(1+np.exp(-x[0]**2/2))
    e_ph = -(1-n)*U/(x[1]*(1 + np.exp(-x[0]**2/2))**2)* np.sqrt(2/np.pi) * (1+ 2*np.exp(-x[0]**2) + \
            1/x[0]*np.sqrt(np.pi/2)*(erf(x[0]/np.sqrt(2)) + 8*np.exp(-x[0]**2/2)*erf(x[0]/(2*np.sqrt(2)))))
    V = e_ph + coul
    return V

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
    effpot = V_eff(x,n,U)
    E = KE + effpot
    return E

def E_ana(n,U):
    '''Optimized symmetric Gaussian energy evaluated at y->infty, normalized by KE'''
    return -(1-n)**2*U**2/(6*np.pi)

def EvaluateE(xlist, ylist,n,U):
    '''use if want to plot contours for a single value of eta and U'''
    arr_E = [[opt_Gauss((y,x),n,U) for x in xlist] for y in ylist]
    return np.array(arr_E)

def minimization(args):
    n,u = args
    bnds = ((1E-3,10), (1E-3, 10)) #y,s
    guess = np.array([1.,1.,])
    result = minimize(opt_Gauss,guess,args=(n,u),bounds=bnds)
    minvals = result.x
    E_opt = opt_Gauss(minvals,n,u)
    E_inf = E_ana(n,u)
    E_binding = (E_opt - E_inf)/np.abs(E_inf)
    V_opt = V_eff(minvals,n,u)
    return n,u,minvals[1],minvals[0], E_opt,0.,0.,E_inf,E_binding

def plotContour(X,Y,Z,axlabels,xmax,ymax,saveas="",colors="viridis"):
    xtitle, ytitle, ztitle = axlabels
    fig = plt.figure(figsize=(6,4.5))
    ax = fig.add_subplot(111)
    cp = ax.contourf(X, Y, Z, levels = MaxNLocator(nbins=20).tick_values(max(-1,Z.min()), min(1,Z.max())), cmap=colors)
    cbar=fig.colorbar(cp) # Add a colorbar to a plot
    cbar.ax.set_ylabel(ztitle)
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    if xmax != X.max(): ax.set_xlim(right=xmax)
    if ymax != Y.max():ax.set_ylim(top=ymax)
    plt.tight_layout()
    if len(saveas) > 0: plt.savefig(saveas)
    plt.show()

def plotGraph(X,Y,axlabels,xmax,styles=[],leglabels=[],saveas=""):
    xtitle, ytitle = axlabels
    fig = plt.figure(figsize=(6,4.5))
    ax = fig.add_subplot(111)
    if len(leglabels) == 0: 
        [ax.plot(X,Y[i]) for i in range(len(Y))]
    else: 
        [ax.plot(X,Y[i],styles[i],label=leglabels[i]) for i in range(len(Y))]
        ax.legend()
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    if xmax != X.max(): ax.set_xlim(right=xmax)
    #if ymax != Y.max():ax.set_ylim(top=ymax)
    plt.tight_layout()
    if len(saveas) > 0: plt.savefig(saveas)
    plt.show()

if __name__ == '__main__': 
    import pandas as pd
    ns = np.linspace(0,.999,1000) #eta values
    #ns = [eta_STO]
    
    tic = time.perf_counter()
    '''
    csvname = 'gaus_fixedU.csv'
    df={}
    quantities=['eta','U','s_opt','y_opt','E_opt','s_inf','y_inf','E_inf','E_binding']
    for i in quantities:
        df[i]=[]

    tic = time.perf_counter()
    with multiprocessing.Pool(processes=4) as pool:
        job_args = [(n,U_STO) for n in ns]
        results = pool.map(minimization, job_args)
        print(results[0])
        for res in results:
            for name, val in zip(quantities, res): 
                df[name].append(val)
    toc = time.perf_counter()
    print(f"time taken: {toc-tic:0.4f} s, {(toc-tic)/60:0.3f} min")
    data = pd.DataFrame(df)
    #pd.set_option("display.max.columns",None)
    print(data)
    data.to_csv(csvname,sep=',',index=False)
    '''
    n1 = eta_STO #eta < 0.04
    n2 = 0.05
    n3 = 0.2
    
    #print energy contour + physical param values 
    sls = np.linspace(1E-10,10,300)
    ys = np.linspace(1E-10,10,500)
    EZ1 = EvaluateE(sls,ys,n1, U_STO)
    print(EZ1.shape)
    E_opt = EZ1.min()
    result = np.where(EZ1 == E_opt)
    print(result)
    sl_opt = sls[result[1]]
    y_opt = ys[result[0]]
    print((sl_opt,y_opt,E_opt))
    K = hbar**2/(2*m*l**2)
    print((sl_opt*l,y_opt*sl_opt*l,E_opt*K*convJ))
    #plotContour(sls,ys,EZ1,["$\sigma/l$","$y$","$E/K$"],sls.max(),ys.max(),colors="twilight_shifted",saveas='E_gaus_sym.png') 
     
    #Plot potential at different etas (stable, metastable, unstable)
    #plot energy vs y at fixed sigma/l = sl_opt
    E_y = EZ1[:,result[1]].transpose()[0]
    #print(E_y)
    #print(E_y.shape)
    #Metastable soln: 0<eta<0.1
    EZ2 = EvaluateE(sls,ys,n2, U_STO)
    result2 = np.where(EZ2 == EZ2.min())
    E_y2 = EZ2[:,result2[1]].transpose()[0]
    #horizontal line at y = infty value
    E_const = [E_y2[E_y2.shape[0]-1] for i in range(len(ys))]

    #Now obtain result for unstable soln: eta > 0.1
    EZ3 = EvaluateE(sls,ys,n3, U_STO)
    result3 = np.where(EZ3 == EZ3.min())
    E_y3 = EZ3[:,result3[1]].transpose()[0]
    plotGraph(ys,[E_y,E_y2,E_const,E_y3], ["$y$","$E/K$"], ys.max(),styles=['-','-','--','-'],
            saveas='E_vs_y_eta_comp.eps',
            leglabels=["$\eta=\eta_{STO}$","$\eta = 0.05$","$\eta=0.05(y \\rightarrow \infty)$","$\eta = 0.2$"])
    
    toc = time.perf_counter()
    print(f"time taken: {toc-tic:0.4f} s, {(toc-tic)/60:0.3f} min")

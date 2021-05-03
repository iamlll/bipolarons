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
import var_a_cy as avar

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

def EvaluateGaussE(xlist, ylist,n,U):
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

def EvalHybE(args):
    '''
    Find y vs sigmas array of hybrid energies for fixed a, eta, U values
    '''
    xlist,ylist,a,n,U = args
    arr_E = np.array([[avar.opt_hybE(np.array([y,x,a]),n,U) for x in xlist] for y in ylist])
    return n,arr_E
    
if __name__ == '__main__': 
    tic = time.perf_counter()
    ns_gaus = np.array([eta_STO, 0.05, 0.2]) #this works for the gaussian case
    ns_hyb = np.array([eta_STO,0.25,0.45]) #sto, 0.25, 0.45 this should work for the hybrid case

    a = 0.67 #This is around the optimal a value for STO

    #print energy contour + physical param values 
    sls = np.linspace(1E-10,10,300)
    ys = np.geomspace(1E-10,10,500)
    E_ys = []
    act = avar.minimization((eta_STO,U_STO)) #the actual STO answer
    print("a: ",act[2],"sl: ",act[3],"y: ",act[4],"E_opt: ",act[5])
    a = act[2]
    job_args = [(sls,ys,a,n,U_STO) for n in ns_hyb]
    with multiprocessing.Pool(processes=4) as pool:
        res = pool.map(EvalHybE, job_args)
        #print(res)
        for vals in res:
            eta, EZ = vals
        
            yidx,sidx = np.where(EZ == EZ.min())
            sidx = sidx[0]
            yidx = yidx[0]
            print("a: ",a,"eta: ",eta,"sl: ",sls[sidx],"y: ",ys[yidx], "E_opt: ",EZ[yidx,sidx])
            idx = np.where(ns_hyb == eta)[0][0]
            #print(idx)
            #E_ys[idx] = EZ[:,sidx].transpose()
            #print(E_ys[idx])
            E_ys.append(EZ[:,sidx].transpose())
    E_const = np.array([E_ys[1][E_ys[1].shape[0]-1] for i in range(len(ys))]) #constant line for metastable solution
    E_ys = np.insert(np.array(E_ys),2,E_const,axis=0)
    toc = time.perf_counter()
    print(f"time taken: {toc-tic:0.4f} s, {(toc-tic)/60:0.3f} min")
    plotGraph(ys,E_ys, ["$y$","$E/K$"], ys.max(),styles=['-','-','--','-'],
            saveas="hyb_E_y_eta_comp.eps",
            leglabels=["$\eta=\eta_{STO}$","$\eta = 0.25$","$\eta=0.25(y \\rightarrow \infty)$","$\eta = 0.45$"])
    
    #fig = plt.figure(figsize=(6,4.5))
    #ax = fig.add_subplot(111)
    #ax.plot(ys,E_ys[0])
    #plt.show()
    '''
    EZ1 = EvaluateGaussE(sls,ys,n1, U_STO)
    print(EZ1.shape)
    E_opt = EZ1.min()
    result = np.where(EZ1 == E_opt)
    print(result)
    sl_opt = sls[result[1]]
    y_opt = ys[result[0]]
    print((sl_opt,y_opt,E_opt))
     
    #Plot potential at different etas (stable, metastable, unstable)
    #plot energy vs y at fixed sigma/l = sl_opt
    E_y = EZ1[:,result[1]].transpose()[0]
    #print(E_y)
    #print(E_y.shape)
    #Metastable soln: 0<eta<0.1
    EZ2 = EvaluateGaussE(sls,ys,n2, U_STO)
    result2 = np.where(EZ2 == EZ2.min())
    E_y2 = EZ2[:,result2[1]].transpose()[0]
    #horizontal line at y = infty value
    E_const = [E_y2[E_y2.shape[0]-1] for i in range(len(ys))]

    #Now obtain result for unstable soln: eta > 0.1
    EZ3 = EvaluateGaussE(sls,ys,n3, U_STO)
    result3 = np.where(EZ3 == EZ3.min())
    E_y3 = EZ3[:,result3[1]].transpose()[0]
    plotGraph(ys,[E_y,E_y2,E_const,E_y3], ["$y$","$E/K$"], ys.max(),styles=['-','-','--','-'],
            saveas='E_vs_y_eta_comp.eps',
            leglabels=["$\eta=\eta_{STO}$","$\eta = 0.05$","$\eta=0.05(y \\rightarrow \infty)$","$\eta = 0.2$"])
    '''

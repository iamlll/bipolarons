import multiprocessing 
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.optimize import minimize
import time
import warnings
from itertools import product
import sys
import var_a_cy as avar

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

from scipy.optimize import curve_fit
def fitdata(filename):
    def fitweak(x,a,b):
        f = a*x + b*x**2 
        return f
    def fitstrong(x,a,b):
        f = a*x**2 +b
        return f

    fig = plt.figure(figsize=(10,4.5))
    ax = fig.add_subplot(121)
    #read in CSV as Pandas dataframe
    df = pd.read_csv(filename)
    alphas = df["alpha"].values
    Elist = df["E_pol"].values
    idx = np.where(alphas <= 0.5)[0]
    
    #separate lists into "weak coupling" (alpha < 5) and "strong coupling" (alpha > 5) - this is suggested by Feynman
    E_wk = Elist[idx]
    al_wk = alphas[idx]
    idx2 = np.where((alphas > 5) & (alphas<10))
    E_str = Elist[idx2]
    al_str = alphas[idx2]
    #print(al_str)

    bnds_w = ([-10,-10],[5,5]) #bounds for weak coupling fit
    guess_w =[-1,-3]
    param_w, p_cov_w = curve_fit(fitweak,al_wk, E_wk, p0=guess_w,bounds=bnds_w)
    print(param_w)
    print(np.sqrt(np.diag(p_cov_w))) #standard deviation of the parameters in the fit
    a,b = param_w
    
    ax.plot(al_wk, E_wk,label='data')
    ax.set_xlabel('$\\alpha$')
    ax.set_ylabel('$E_{opt}$')
    ans_w = np.array([fitweak(al,a,b) for al in al_wk])
    ax.plot(al_wk,ans_w,color='red',label='fit')
    
    textstr = '\n'.join((
        r'$y(\alpha) = a\alpha + b\alpha^2$',
        r'$a=%.2f$' % (a, ),
        r'$b=%.2f$' % (b, )
        ))

    ax.text(0.05, 0.45, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top')

    ax.legend(loc=1)
        
    bnds_s = ([-10,-10],[5,5]) #bounds for weak coupling fit
    guess_s =[-1,-3]
    param_s, p_cov_s = curve_fit(fitstrong,al_str, E_str, p0=guess_s,bounds=bnds_s)
    print(param_s)
    print(np.sqrt(np.diag(p_cov_s))) #standard deviation of the parameters in the fit
    c,d = param_s
  
    ax2 = fig.add_subplot(122)
    ax2.plot(al_str, E_str,label='data')
    ax2.set_xlabel('$\\alpha$')
    ax2.set_ylabel('$E_{opt}$')
    ans_s = np.array([fitstrong(al,c,d) for al in al_str])
    ax2.plot(al_str,ans_s,color='red',label='fit')
    
    textstr = '\n'.join((
        r'$y(\alpha) = c\alpha^2 + d$',
        r'$c=%.2f$' % (c, ),
        r'$d=%.2f$' % (d, )
        ))

    ax2.text(0.05, 0.45, textstr, transform=ax2.transAxes, fontsize=14,
        verticalalignment='top')

    ax2.legend(loc=1)
    plt.tight_layout()
    #if save==True: plt.savefig("sing_hyb_a_min.png")
    plt.show()

import matplotlib.pyplot as plt
import pandas as pd
def plotstuff(filename,save=False,scale='linear'):
    fig = plt.figure(figsize=(10,4.5))
    ax = fig.add_subplot(121)
    #read in CSV as Pandas dataframe
    df = pd.read_csv(filename)
    pd.set_option("display.max.columns", None)
    etas = df['eta'].values
    Us = df['U'].values
    alphas = np.array([(1-n)*u/2 for n,u in zip(etas,Us)])
    f = df["E_inf"].values
    h = df["s_inf"].values

    df2 = pd.read_csv("./data/cy_var_Einf_zoom.csv")
    g = df2['E_inf'].values
    k = df2["s_inf"].values
    if scale == 'loglog':
        f = np.array([-en if en<0 else 0. for en in f])
        g = np.array([-en if en<0 else 0. for en in g])
        ax.loglog(alphas,f,label='$-E_\infty\, (\sigma\\to\infty)$')
        ax.loglog(alphas,g,label='$-E_\infty$')
        ax.legend(loc=2)
    else:
        ax.plot(alphas,f,label='$E_\infty\, (\sigma\\to\infty)$')
        ax.plot(alphas,g,label='$E_\infty$')
        ax.legend(loc=1)
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("$E_\infty$")
    #ax.set_ylim(-20,10)
    #ax.set_xlim(0,5)

    ax2 = fig.add_subplot(122)
    if scale == 'loglog':
        ax2.loglog(alphas,h,label='$\sigma_\infty\, (\sigma\\to\infty)$')
        ax2.loglog(alphas,k,label='$\sigma_\infty$')
        
        inds, alnew, signew = extrapolate_sig(ax2,alphas,k)
        extrapolate_Einf(ax, inds, alnew, signew, etas, Us, df2['y_inf'].values,df2['a'].values)    
        ax2.legend(loc=3)
    else:
        ax2.plot(alphas,h,label='$\sigma_\infty\, (\sigma\\to\infty)$')
        ax2.plot(alphas,k,label='$\sigma_\infty$')
        ax2.legend(loc=1)
    ax2.set_xlabel("$\\alpha$")
    ax2.set_ylabel("$\sigma_\infty$")
    #ax2.set_ylim(0,15)
    #ax2.set_xlim(0,5)

    plt.tight_layout()
    if save==True: plt.savefig("lg_sig_comp_loglogzoom_extrap.png")
    plt.show()
    '''
    fig = plt.figure(figsize=(5,4.5))
    ax3 = fig.add_subplot(111)
    ax3.loglog(alphas,df2['y_inf'].values)
    ax3.set_xlabel('$\\alpha$')
    ax3.set_ylabel('$y$')
    plt.show()
    '''

def extrapolate_sig(ax, alphas, sigs):
    def fitpwrlaw(x,a,b):
        f = a*x**b
        return f
    idx = np.where((alphas <= 0.67) & (alphas > 0.08))
    alpha_interp = alphas[idx]
    sig_interp = sigs[idx]
    idx2 = np.where(alphas<=0.08)
    alnew = alphas[idx2]
    bnds = ((0,-5),(10,0))
    guess =[1,-1]
    param, p_cov = curve_fit(fitpwrlaw,alpha_interp, sig_interp, p0=guess,bounds=bnds)
    print(param)
    print(np.sqrt(np.diag(p_cov))) #standard deviation of the parameters in the fit
    c,d = param
    ans = np.array([fitpwrlaw(al,c,d) for al in alnew])
    ax.loglog(alnew,ans,color='green',label='extrapolation')
    
    textstr = '\n'.join((
        r'$\sigma(\alpha) = c\alpha^d$',
        r'$c=%.2f$' % (c, ),
        r'$d=%.2f$' % (d, )
        ))

    ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top')
    return idx2, alnew, ans

def extrapolate_Einf(ax, inds, alphas, sigs, etas, Us, ys, ayes):
    #find energies corresponding to interpolated sigmas 
    etas = etas[inds]
    Us = Us[inds]
    ys = ys[inds]
    ayes = ayes[inds]
    #ayes = np.ones(len(inds[0]))
    Einf_extrap = np.array([avar.E_infty(np.array([y,sl,a]),n,u) for y,sl,a,n,u in zip(ys,sigs,ayes,etas,Us)])
    ax.loglog(alphas,-Einf_extrap,color='green',label='extrapolation')
    ax.legend(loc=2)

if __name__ == '__main__': 
    import pandas as pd
    #range of Us for eta=0 to compare E_inf behavior with 2*single polaron behavior (as fn of alpha)
    ns=[0.]
    Us = np.linspace(1E-3,40,400)
    #Us = np.geomspace(1E-5,10,400)

    #csvname = "./data/cy_var_lg_sig_zoom.csv"
    #csvname = "./data/cy_var_Einf_zoom.csv"
    csvname = "./data/cy_var_Einf2.csv"
    
    
    df={}
    quantities=['eta','U','a','s_inf','y_inf','E_inf']
    for i in quantities:
        df[i]=[]
    
    tic = time.perf_counter()
    with multiprocessing.Pool(processes=4) as pool:
        job_args = [(n,u) for n,u in product(ns,Us)]
        #results = pool.map(avar.min_lg_sig, job_args)
        results = pool.map(avar.min_Einf, job_args)
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
     
    #plotstuff(csvname, save=False,scale='loglog')

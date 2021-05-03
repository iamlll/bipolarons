'''Compare hybrid polaron, Huybrechts polaron, and bipolaron y->inf energies to see if they match up.'''

import multiprocessing 
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.optimize import minimize
from scipy.special import erfcx
import time
import warnings
from itertools import product
import sys

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

from scipy.optimize import fsolve
def bipolSolveFora(a,alfa,sl):
    '''Solve for a, given sigma/l'''
    A = (1-a/2)**2 + (a/2)**2
    b = A*sl**2/a**2
    
    val = erfcx(np.sqrt(b))
    return 3/sl**3 + alfa*np.sqrt(2*A)/a**2 *(np.sqrt(b)*val*val2 - 1/np.sqrt(np.pi))

def bipolEinf2(x,alfa,a):
    '''
    Single polaron energy in hybrid scheme, in units of hw
    x = [sigma/l]
    df = Pandas dataframe, to store a value
    '''
    A = (1-a/2)**2 + (a/2)**2
    b = A*x**2/a**2
    val = erfcx(np.sqrt(b))
    return 3/(x**2) - alfa*np.sqrt(2)/a* val

def bipolEinf(x,alfa,a_arr):
    '''
    Single polaron energy in hybrid scheme, in units of hw
    x = [sigma/l]
    df = Pandas dataframe, to store a value
    '''
    a =fsolve(bipolSolveFora,[0.75],args=(alfa,x))[0]
    a_arr.append(a)
    A = (1-a/2)**2 + (a/2)**2
    b = A*x**2/a**2
    val = erfcx(np.sqrt(b))
    return 3/(x**2) - alfa*np.sqrt(2)/a* val

def min_bipol(args):
    alfa,a_arr, a_arr2 = args
    bnds = ((1E-3,1E10),) #s
    guess = np.array([1.])
    result = minimize(bipolEinf,guess,args=(alfa,a_arr),bounds=bnds)
    minvals = result.x
    E_opt = bipolEinf2(minvals[0],alfa,a_arr[-1])

    return alfa,a_arr[-1],minvals[0],E_opt

from scipy.optimize import curve_fit
def fitdata(filename,save=''):
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
    
    ax.plot(al_wk, E_wk,label='pol data')
    ax.set_xlabel('$\\alpha$')
    ax.set_ylabel('$E_{Huy}$')
    ans_w = np.array([fitweak(al,a,b) for al in al_wk])
    ax.plot(al_wk,ans_w,color='red',label='fit')
    
    textstr = '\n'.join((
        r'$E(\alpha) = a\alpha + b\alpha^2$',
        r'$a=%.2f$' % (a, ),
        r'$b=%.2f$' % (b, )
        ))

    ax.text(0.05, 0.45, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top')

    ax.legend(loc=1)
        
    bnds_s = ([-10,-10],[5,5]) #bounds for strong coupling fit
    guess_s =[-1,-3]
    param_s, p_cov_s = curve_fit(fitstrong,al_str, E_str, p0=guess_s,bounds=bnds_s)
    print(param_s)
    print(np.sqrt(np.diag(p_cov_s))) #standard deviation of the parameters in the fit
    c,d = param_s
  
    ax2 = fig.add_subplot(122)
    ax2.plot(al_str, E_str,label='pol data')
    ax2.set_xlabel('$\\alpha$')
    ax2.set_ylabel('$E_{Huy}$')
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
    if len(save)>0: plt.savefig(save)
    plt.show()

import matplotlib.pyplot as plt
import pandas as pd
def plotstuff(csvname,opt='linear',save=False):
    fig = plt.figure(figsize=(12,4))
    ax = fig.add_subplot(131)
    #read in CSV as Pandas dataframe
    if opt == 'linear':
        df = pd.read_csv("./data/hyb_pol.csv")
        df3 = pd.read_csv("./data/Huy_pol.csv")
        df2 = pd.read_csv("./data/hyb_Einf.csv")
        df4 = pd.read_csv(csvname) #different U1, y->infty result
        df5 = pd.read_csv("./data/asym_U1.csv")
    elif opt == 'log':
        df = pd.read_csv("./data/hyb_pol_log.csv")
        df3 = pd.read_csv("./data/Huy_pol_log.csv")
        df2 = pd.read_csv("./data/hyb_Einf_log.csv")

    x = df["alpha"].values
    f = df["E_pol"].values
    g = df["a_pol"].values
    c = df['s_pol'].values
    m = df2['s_bi'].values
    h = df2['E_bi'].values
    k = df2['a_bi'].values
    huy_E = df3["E_pol"].values
    huy_a = df3["a_pol"].values
    huy_s = df3['s_pol'].values 
    alt_alpha = [(1-n)*u/2 for n,u in zip(df4['eta'].values, df4['U'].values)]
    alt_sinf = df5['s_bi'].values
    alt_Einf = df5['E_bi'].values
    alt_a = df5['a_bi'].values
    alt2_sinf_R = df4['s_bi'].values
    alt2_sinf_r = df4['y_bi'].values
    alt2_Einf = df4['E_bi'].values
    alt2_a = df4['a_bi'].values

    ax.set_xlabel("$\\alpha$")
    ax2 = fig.add_subplot(132)
    ax2.set_xlabel("$\\alpha$")
    ax2.set_ylabel("a")
    ax3 = fig.add_subplot(133)
    ax3.set_xlabel("$\\alpha$")
    ax3.set_ylabel("$\\tilde{\sigma}$")

    if opt == 'linear':
        #ax.plot(x,2*f,label='$2E_{pol}$')
        #ax.plot(x,2*huy_E,label='$2E_{Huy}$')
        #ax.plot(x,h,label='$E_{bi}$')
        #ax.plot(alt_alpha,alt_Einf,label='$E_{alt}$')
        ax.plot(alt_alpha,alt2_Einf,label='$E_{rR}$')
        ax.set_ylabel("E/K")
        ax.set_xlim(0,10)
        #ax2.plot(x,g,label='$a_{pol}$')
        #ax2.plot(x,huy_a,label='$a_{Huy}$')
        #ax2.plot(x,k,label='$a_{bi}$')
        #ax2.plot(alt_alpha,alt_a,label='$a_{alt}$')
        ax2.plot(alt_alpha,alt2_a,label='$a_{rR}$')
        #ax3.plot(x,c,label='$\\tilde{\sigma}_{pol}$')
        #ax3.plot(x,huy_s,label='$\\tilde{\sigma}_{Huy}$')
        #ax3.plot(x,m,label='$\\tilde{\sigma}_{bi}$')
        #ax3.plot(alt_alpha,alt_sinf,label='$\\tilde{\sigma}_{alt}$')
        ax3.plot(alt_alpha,alt2_sinf_r,label='$\\tilde{\sigma}_{r}$')
        ax3.plot(alt_alpha,alt2_sinf_R,label='$\\tilde{\sigma}_{R}$')
        ax3.set_ylim(0,10)
    elif opt == 'log':
        ax.loglog(x,-2*f,label='$2E_{pol}$')
        ax.set_ylabel("-E/K")
        ax2.loglog(x,g,label='$a_{pol}$')
        ax2.plot(x,k,label='$a_{bi}$')
        ax3.loglog(x,c,label='$\\tilde{\sigma}_{pol}$')
        ax3.loglog(x,m,label='$\\tilde{\sigma}_{bi}$')
    ax.legend(loc=3)
    ax2.legend(loc=1)
    ax3.legend(loc=1)
    #ax3.set_ylim(0,10)

    plt.tight_layout()
    if save==True: plt.savefig("comp_Einfs.png")
    plt.show()

def plot_indep_params():
    fig = plt.figure(figsize=(12,4))
    ax = fig.add_subplot(131)
    df = pd.read_csv("./data/indep_params.csv")
    alphas = [(1-n)*u/2 for n,u in zip(df['eta'].values, df['U'].values)]
    s_r = df['s_r'].values
    Einf = df['E_bi'].values
    a_r = df['a_r'].values
    a_R = df['a_R'].values
    s_R = df['s_R'].values
    ax.plot(alphas,Einf,label='$E_\infty$')

    ax2 = fig.add_subplot(132)
    ax2.plot(alphas,a_r,label='$a_{r}$')
    ax2.plot(alphas,a_R,label='$a_{R}$')
    ax3 = fig.add_subplot(133)
    ax3.plot(alphas,s_r,label='$\\tilde{\sigma}_{r}$')
    ax3.plot(alphas,s_R,label='$\\tilde{\sigma}_{R}$')

    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("$E$")
    ax2.set_xlabel("$\\alpha$")
    ax2.set_ylabel("a")
    ax3.set_xlabel("$\\alpha$")
    ax3.set_ylabel("$\\tilde{\sigma}$")
    ax.legend(loc=1)
    ax2.legend(loc=1)
    ax3.legend(loc=1)
    plt.show()

def hybE_3(x,alfa):
    '''
    Single polaron energy in hybrid scheme, in units of hw
    x = [lambda,t]
    '''
    if x[1] == 0: return 3/4*x[0] - alfa
    else:
        val = erfcx(x[1])
        return 3/4*x[0] - alfa*(1+x[1]*np.sqrt(x[0])) * val

def min_hyb_3(alfa):
    bnds = ((1E-100,1E3),(-1,1E5)) #lambda, t
    guess = np.array([10,0.5]) #lambda, t
    result = minimize(hybE_3,guess,args=(alfa,),bounds=bnds)
    minvals = result.x
    E_pol = hybE_3(minvals, alfa)
    #return sigma/l = sqrt(2/lam), a = 1/(1+t*sqrt(lambda))
    return alfa,1/(1+minvals[1]*np.sqrt(minvals[0])),np.sqrt(2/minvals[0]),E_pol,0,0,0

def HuyE_2(x,alfa):
    '''
    Single polaron energy in Huybrechts scheme, in units of hw
    x = [lambda,t]
    '''
    val = erfcx(x[1])
    return 3/4*x[0]*1/(1+1/(x[1]*np.sqrt(x[0])))**2 - alfa* (1+x[1]*np.sqrt(x[0])) * val

def min_Huy_2(alfa):
    bnds = ((1E-6,1E3),(1E-3,1E5)) #lambda, t
    guess = np.array([10.,0.5])
    result = minimize(HuyE_2,guess,args=(alfa,),bounds=bnds)
    minvals = result.x
    E_pol = HuyE_2(minvals, alfa)
    #return sigma/l = sqrt(2/lam)
    return alfa,1/(1+minvals[1]*np.sqrt(minvals[0])),np.sqrt(2/minvals[0]),E_pol,0,0,0

def Einf2(x,alfa):
    '''
    y->infty bipolaron energy in hybrid scheme, in units of hw
    x = [lambda, b]
    '''
    val = erfcx(np.sqrt(x[1]))
    if x[1]*x[0] == 1: 
        return 3./2*x[0] - alfa*2*np.sqrt(2)*val
    else: 
        return 3./2*x[0] - alfa*2*np.sqrt(2)*(x[1]*x[0]-1) / (-1+np.sqrt( 1 + 2*(x[1]*x[0]-1) ) ) * val

def min_Einf2(alfa):
    bnds = ((1E-5,1E3),(1E-3,1E5)) #lambda,b
    guess = np.array([5.,1.])
    cons = ({'type': 'ineq', 'fun': lambda x:  x[0]*x[1]-0.5},)
    result = minimize(Einf2,guess,args=(alfa,),bounds=bnds, constraints=cons)
    minvals = result.x
    E_opt = Einf2(minvals,alfa)
    if minvals[0]*minvals[1] == 1: a=1
    else: a= (-1+np.sqrt( 1 + 2*(minvals[0]*minvals[1]-1))) /(minvals[1]*minvals[0]-1)
    return alfa,0,0,0,a,np.sqrt(2/minvals[0]),E_opt

def Einf3(x,alfa):
    '''
    y->infty bipolaron energy in hybrid scheme, in units of hw
    x = [lambda, a]
    '''
    A = (1-x[1]/2)**2+(x[1]/2)**2
    b = A*2/(x[0]*x[1]**2)
    val = erfcx(np.sqrt(b))
    return 3./2*x[0] - alfa/x[1]*2*np.sqrt(2)*val

def min_Einf3(alfa):
    bnds = ((1E-5,1E3),(1E-5,2)) #lambda,a
    guess = np.array([5.,1.])
    result = minimize(Einf3,guess,args=(alfa,),bounds=bnds)
    minvals = result.x
    E_opt = Einf3(minvals,alfa)
    return alfa,0,0,0,minvals[1],np.sqrt(2/minvals[0]),E_opt

def Einf_ALT(x,alfa):
    '''
    Corresponds to Nagano calc Einf
    y->infty bipolaron energy in hybrid scheme, in units of hw
    x = [lambda, a]
    '''
    b1 = (1- x[1])**2/(2*x[0]*x[1]**2)
    b2 = (1+ x[1]**2)/(2*x[0]*x[1]**2)
    val1 = erfcx(np.sqrt(b1))
    val2 = erfcx(np.sqrt(b2))
    return 3./2*x[0] - alfa*np.sqrt(2)/(x[1])*(val1 + val2)

def min_Einf_ALT(alfa):
    bnds = ((1E-5,1E3),(1E-5,2)) #lambda,a
    guess = np.array([5.,1.])
    result = minimize(Einf_ALT,guess,args=(alfa,),bounds=bnds)
    minvals = result.x
    E_opt = Einf_ALT(minvals,alfa)
    return alfa,0,0,0,minvals[1],np.sqrt(2/minvals[0]),E_opt

def Einf_ALT2(x,alfa):
    '''
    Corresponds to asymmetric U1 w/ indep a1, a2 params
    y->infty bipolaron energy in hybrid scheme, in units of hw
    x = [sigma/l, a1,a2]
    '''
    if float(x[1])==float(x[2]):
        b = ((1-x[1])**2+x[1]**2)/(2*x[0]*x[1]**2)
        val = erfcx(np.sqrt(b))
        return 3/2*x[0] - alfa*np.sqrt(2)/x[1]*val
    else:
        b1 = ((1- x[1])**2+x[2]**2)/(x[0]*(x[1]**2+x[2]**2))
        b2 = ((1- x[2])**2+x[1]**2)/(x[0]*(x[1]**2+x[2]**2))
        val1 = erfcx(np.sqrt(b1))
        val2 = erfcx(np.sqrt(b2))
        return 3/2*x[0] - alfa/(2*np.sqrt(x[1]**2+x[2]**2))*(val1 + val2)

def min_Einf_ALT2(alfa):
    bnds = ((1E-5,1E3),(0,1),(1E-10,1)) #lambda,a1,a2
    guess = np.array([5.,0.6,0.5])
    result = minimize(Einf_ALT2,guess,args=(alfa,),bounds=bnds)
    minvals = result.x
    E_opt = Einf_ALT2(minvals,alfa)
    return alfa,0,0,0,minvals[1],minvals[2],np.sqrt(2/minvals[0]),E_opt

def Comp_params():
    df = pd.read_csv("./data/Huy_pol.csv")
    df2 = pd.read_csv("./data/hyb_pol.csv")
    df3 = pd.read_csv("./data/hyb_Einf.csv") #taking y->infty

    fig = plt.figure(figsize=(10,9))
    ax = fig.add_subplot(221)
    #read in CSV as Pandas dataframe
    alphas = df["alpha"].values
    huy = df["E_pol"].values
    huy_sig = df["s_pol"].values
    huy_a = df["a_pol"].values
    hyb = df2["E_pol"].values
    hyb_sig = df2["s_pol"].values
    hyb_a = df2["a_pol"].values
    Einf = df3["E_bi"].values
    sig_inf = df3["s_bi"].values
    a_inf = df3["a_bi"].values

    ax.plot(alphas,huy,label='Huybrechts')
    ax.plot(alphas,hyb,label='hybrid ($\lambda,b$)')
    ax.plot(alphas,Einf,label='hyb ($\lambda,a,y\gg 1$)')
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("E/K")
    ax.legend(loc=3)
    ax2 = fig.add_subplot(222)
    ax2.plot(alphas,huy_a,label='Huybrechts')
    ax2.plot(alphas,hyb_a,label='hybrid ($\lambda,b$)')
    ax2.plot(alphas,a_inf,label='bipol ($y\gg 1$)')
    ax2.set_xlabel("$\\alpha$")
    ax2.set_ylabel("a")
    ax2.legend(loc=1)
    ax3 = fig.add_subplot(223)
    ax3.plot(alphas,huy_sig,label='Huybrechts')
    ax3.plot(alphas,hyb_sig,label='hybrid ($\lambda$)')
    ax3.plot(alphas,sig_inf,label='bipol ($y\gg 1$)')
    ax3.set_xlabel("$\\alpha$")
    ax3.set_ylabel("$\sigma/l$")
    ax3.legend(loc=1)

    plt.tight_layout()
    plt.savefig('huy_hyb_comp.png')
    plt.show()

def Comp_params_Einf():
    df = pd.read_csv("./data/hyb_Einf.csv") #taking y-> infty
    #df2 = pd.read_csv("./data/cy_var_Einf.csv") #taking y>>1, minimizing wrt sigma/l,y, and a
    df3 = pd.read_csv("./data/hyb_Einf_ALT.csv") #Nagano
    
    fig = plt.figure(figsize=(10,9))
    ax = fig.add_subplot(221)
    #read in CSV as Pandas dataframe
    alphas = df["alpha"].values
    hyb = df["E_bi"].values
    hyb_sig = df["s_bi"].values
    hyb_a = df["a_bi"].values
    hyb2 = df2["E_inf"].values
    hyb_sig2 = df2["s_inf"].values
    hyb_a2 = df2["a"].values
    hyb3 = df3["E_inf"].values
    hyb_sig3 = df3["s_inf"].values
    hyb_a3 = df3["a"].values

    ax.plot(alphas,hyb,label='hybrid ($\lambda,b, y\\to \infty$)')
    ax.plot(alphas,hyb2,label='hybrid ($\sigma,a,y \gg 1$)')
    ax.plot(alphas,hyb3,label='hybrid ($\lambda,a, y\gg 1$)')
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("E/K")
    ax.legend(loc=3)
    ax2 = fig.add_subplot(222)
    ax2.plot(alphas,hyb_a,label='hybrid ($\lambda,b, y\\to \infty$)')
    ax2.plot(alphas,hyb_a2,label='hybrid ($\sigma,a,y \gg 1$)')
    ax2.plot(alphas,hyb_a3,label='hybrid ($\lambda,a, y\gg 1$)')
    ax2.set_xlabel("$\\alpha$")
    ax2.set_ylabel("a")
    ax2.legend(loc=1)
    ax3 = fig.add_subplot(223)
    ax3.plot(alphas,hyb_sig,label='hybrid ($\lambda,b, y\\to \infty$)')
    ax3.plot(alphas,hyb_sig2,label='hybrid ($\sigma,a,y \gg 1$)')
    ax3.plot(alphas,hyb_sig3,label='hybrid ($\lambda,a, y\gg 1$)')
    ax3.set_xlabel("$\\alpha$")
    ax3.set_ylabel("$\sigma/l$")
    ax3.legend(loc=1)

    plt.tight_layout()
    plt.show()

def PlotE(csvname,fit=True):
    '''Plot energy, sep dist (y), elec size (sigma) of a single data file'''

    df = pd.read_csv(csvname) 
    
    fig = plt.figure(figsize=(10,9))
    ax = fig.add_subplot(221)
    #read in CSV as Pandas dataframe
    alphas = np.array([(1-n)*U/2. for n,U in zip(df["eta"].values, df['U'].values)])
    E = df["E_bi"].values
    sigs = df["s"].values
    ys = df["y"].values

    ax.plot(alphas,E,label='energy (a=1)')
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("E/K")
    ax2 = fig.add_subplot(222)
    ax2.plot(alphas,sigs)
    ax2.set_xlabel("$\\alpha$")
    ax2.set_ylabel("$\sigma/l$")
    ax3 = fig.add_subplot(223)
    ax3.plot(alphas,ys)
    ax3.set_xlabel("$\\alpha$")
    ax3.set_ylabel("$y$")

    if fit == True:
        def fitweak(x,a,b):
            f = a*x + b*x**2 
            return f

        bnds_w = ([-10,-10],[5,5]) #bounds for weak coupling fit
        guess_w =[-1,-3]
        param_w, p_cov_w = curve_fit(fitweak,alphas, E, p0=guess_w,bounds=bnds_w)
        print(param_w)
        print(np.sqrt(np.diag(p_cov_w))) #standard deviation of the parameters in the fit
        a,b = param_w
    
        ans_w = np.array([fitweak(al,a,b) for al in alphas])
        ax.plot(alphas,ans_w,color='red',label='fit')
    
        textstr = '\n'.join((
            r'$E(\alpha) = a\alpha + b\alpha^2$',
            r'$a=%.2f$' % (a, ),
            r'$b=%.2f$' % (b, )
            ))

        ax.text(0.05, 0.45, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top')

        ax.legend(loc=1)

    plt.tight_layout()
    plt.show()

def CompareEs():
    '''
    Compare my best hybrid E_inf result with Devreese (2x), LLP (2x), Feynman (2x)
    '''

    #df2 = pd.read_csv("./data/cy_var_Einf.csv") #y>>1
    df4 = pd.read_csv("./data/hyb_Einf.csv") #y->infty
    df = pd.read_csv("./data/Huy_pol.csv") #huybrechts
    df3 = pd.read_csv("./data/hyb_pol.csv") #hybrid single polaron
    df5 = pd.read_csv("./data/asym_U1_weak.csv")
    df6 = pd.read_csv("./data/asym_U1_strong.csv")
    
    etas = df5['eta'].values
    Us = df5['U'].values
    alphas = np.array([(1-n)*u/2 for n,u in zip(etas,Us)])
    #alphas = df['alpha'].values

    #devreese = np.array([alfa**2/(3*np.pi) for alfa in alphas])
    #LLP = np.array([alfa+0.007*alfa**2 for alfa in alphas])
    #feynman = np.array([alfa+0.98*alfa**2/100 for alfa in alphas])

    #Einfs = df2['E_inf'].values #original optimization w/ integrals
    huybrechts = df['E_pol'].values
    hybpol = df3['E_pol'].values
    Einf2 = df4['E_bi'].values #y->infty limit
    alt_Einf_wk = df5['E_bi'].values #y->infty limit
    alt_Einf_str = df6['E_bi'].values #y->infty limit

    fig = plt.figure(figsize=(5,4.5))
    ax = fig.add_subplot(111)
    #ax.plot(alphas, 2*LLP,label='LLP (\'52)')
    #ax.plot(alphas,2*feynman,label='Feynman (\'55)')
    #ax.plot(alphas,2*devreese,label='Devreese (\'91)')
    #ax.plot(alphas,-2*huybrechts,label='Huybrechts (\'77)')
    #ax.plot(alphas,-2*hybpol,label='hybrid polaron')
    #ax.plot(alphas, -Einf2,label='$E_\infty$')
    ax.plot(alphas, -alt_Einf_wk,label='$E_\infty$ ($a\\to 1$)')
    ax.plot(alphas, -alt_Einf_str,label='$E_\infty$ ($a\\to 0$)')

    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("$-E/(\hbar \omega)$")
    ax.legend(loc=2)
    plt.tight_layout()
    #plt.savefig("Einf_litcomp.eps")
    plt.show()

if __name__ == '__main__': 
    import pandas as pd
    alphas = np.linspace(0.001,20,400)
    #alphas = np.geomspace(1E-5,10,400)
    #ns = np.linspace(0,0.5,100)
    #Us = np.geomspace(.1,16,num=200)

    csvname = "./data/hyb_pol.csv"
    csvname2 = "./data/Huy_pol.csv"
    csvname3 = "./data/hyb_Einf.csv"
    csvname4 = "./data/hyb_Einf_ALT.csv"
    csvname5 = "./data/hyb_Einf_ALT2.csv"
    csvname6 = "./data/asym_U1.csv"
    csvname7 = "./data/asym_U1_v2.csv"
    csvname8 = "./data/nagano.csv"
    csvname9 = "./data/hyb_rR.csv"
    '''
    df={}
    #quantities=['alpha','a_pol','s_pol','E_pol','a_1','a_2','s_bi','E_bi']
    quantities=['alpha','a_pol','s_pol','E_pol','a_bi','s_bi','E_bi']
    for i in quantities:
        df[i]=[]

    tic = time.perf_counter()
     
    with multiprocessing.Pool(processes=4) as pool:
        #results = pool.map(min_hyb_3, alphas)
        #results = pool.map(min_Huy_2, alphas)
        results = pool.map(min_Einf2, alphas)
        #results = pool.map(min_Einf_ALT, alphas)
        print(results[0])
        for res in results:
            for name, val in zip(quantities, res): 
                df[name].append(val)
            
    data = pd.DataFrame(df)
    pd.set_option("display.max.columns",None)
    print(data)
    
    toc = time.perf_counter()
    print(f"time taken: {toc-tic:0.4f} s, {(toc-tic)/60:0.3f} min")
    data.to_csv(csvname3,sep=',',index=False)
    ''' 
    PlotE(csvname8)
    #plot_indep_params()
    #GlobMinEinf()
    #Comp_params()
    #fitdata(csvname8)
    #fitdata(csvname,save='hyb_pol_fit.png')
    #plotstuff(csvname9,save=False)
    #CompareEs()
    #Comp_params_Einf()

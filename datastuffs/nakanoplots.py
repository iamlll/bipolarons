import multiprocessing 
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.optimize import minimize
from scipy.special import erfcx, dawsn
import time
import warnings
from itertools import product
import sys
from scipy.optimize import curve_fit
import nagano_cy as nag
import matplotlib.pyplot as plt
import pandas as pd

#get preexisting plotting functions for phase diagrams
from sys import path
from os.path import abspath
path.append(abspath(".."))

from data import plotstuff as ps

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

#############################################################################################################################
'''Plotting functions'''

def PlotE(csvname,fit=False):
    '''Plot energy, sep dist (y), elec size (sigma) of a single data file'''

    df = pd.read_csv(csvname) 
    
    fig = plt.figure(figsize=(10,9))
    ax = fig.add_subplot(221)
    #read in CSV as Pandas dataframe
    alphas = np.array([(1-n)*U/2. for n,U in zip(df["eta"].values, df['U'].values)])
    E = df["E"].values
    sigs = df["s"].values
    ys = df["y"].values
    ayes = df["a"].values

    ax.plot(alphas,E,label='bipol')
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("E/K")
    ax2 = fig.add_subplot(222)
    ax2.plot(alphas,sigs,label='bipol')
    ax2.set_xlabel("$\\alpha$")
    ax2.set_ylabel("$\sigma/l$")
    #ax2.set_ylim(0,10)

    ax3 = fig.add_subplot(224)
    ax3.plot(alphas,ys,label='bipol')
    ax3.set_xlabel("$\\alpha$")
    ax3.set_ylabel("$y$")
    ax4 = fig.add_subplot(223)
    ax4.plot(alphas,ayes,label='bipol')
    ax4.set_xlabel("$\\alpha$")
    ax4.set_ylabel("$a$")
    '''
    df2 = pd.read_csv("./data/nagano_a_1_inf.csv") #a=1, y->inf limit
    alphas2 = np.array([(1-n)*U/2. for n,U in zip(df2["eta"].values, df2['U'].values)])
    Ea1 = df2["E_inf"].values
    sa1 = df2["s"].values
    ax.plot(alphas2,Ea1,label='$a=1$')
    #percentdiff = np.abs(Ea1-E)/Ea1 
    #ax.plot(alphas,percentdiff,label='% difference')
    ax2.plot(alphas2,sa1,label='$a=1$')
    '''
    '''
    df2b = pd.read_csv("./data/nak_alpha_yfin2.csv") #new optimization
    E2b = df2b["E"].values
    s2b = df2b["s"].values
    y2b = df2b["y"].values
    a2b = df2b["a"].values
    ax.plot(alphas,E2b,label='v2 opt')
    ax2.plot(alphas,s2b,label='v2 opt')
    ax3.plot(alphas,y2b,label='v2 opt')
    ax4.plot(alphas,a2b,label='v2 opt')
    ''' 
    
    df3 = pd.read_csv("./data/nak_alpha_yinf.csv") #y->inf limit
    alphas3 = np.array([(1-n)*U/2. for n,U in zip(df3["eta"].values, df3['U'].values)])
    E3 = df3["E"].values
    s3 = df3["s"].values
    a3 = df3["a"].values
    ax.plot(alphas3,E3,label='$y\\to \infty$')
    ax2.plot(alphas3,s3,label='$y\\to \infty$')
    ax4.plot(alphas3,a3,label='$y\\to \infty$')
    
    #compare with hybrid calc results
    df4 = pd.read_csv("./data/hyb_alpha.csv")
    alphas4 = np.array([(1-n)*U/2. for n,U in zip(df4["eta"].values, df4['U'].values)])
    Ehyb = df4["E_opt"].values
    shyb = df4["s_opt"].values
    yhyb = df4["y_opt"].values
    ahyb = df4["a_opt"].values
    ax.plot(alphas4,Ehyb,label='hybrid')
    ax2.plot(alphas4,shyb,label='hybrid')
    ax3.plot(alphas4,yhyb,label='hybrid')
    ax4.plot(alphas4,ahyb,label='hybrid')
    
    '''
    df5 = pd.read_csv("./data/nak_alpha_yfin.csv")
    alphas5 = np.array([(1-n)*U/2. for n,U in zip(df5["eta"].values, df5['U'].values)])
    E5 = df5["E"].values
    s5 = df5["s"].values
    y5 = df5["y"].values
    a5 = df5["a"].values
    ax.plot(alphas5,E5,label='nak_yfin')
    ax2.plot(alphas5,s5,label='nak_yfin')
    ax3.plot(alphas5,y5,label='nak_yfin')
    ax4.plot(alphas5,a5,label='nak_yfin')
    '''
    ax2.legend(loc=1)
    ax3.legend(loc=1)
    ax4.legend(loc=1)
    
    #expectedE_str = np.array([-2*al**2/(3*np.pi) for al in alphas])
    #ax.plot(alphas,expectedE_str,label='$-2\\alpha^2/(3\pi)$')
    #feyn = np.array([-al**2/(3*np.pi) -3*np.log(2)-0.75 for al in alphas])
    #ax.plot(alphas,2*feyn,label='Feyn55')


    #bigsig=100.;bigy=5000.;
    #eph = nag.Integral(bigy,bigsig)
    #E_lgsig = np.array([3./bigsig**2 + U/(bigsig*bigy)-(1-n)*U/2.* 2/(np.pi* bigsig)*eph for n,U in zip(df["eta"].values, df['U'].values)])
    #ax.plot(alphas,E_lgsig,label='energy ($\sigma \gg 1$)')

    ax.legend(loc=3)

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
        #ax.plot(alphas,ans_w,color='red',label='fit')
    
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

def PlotBindingE(csvnames):
    '''Plot binding energy (E_bi - E_inf)/|E_inf| as fxn of alpha'''
    csv, csv_inf = csvnames
    df = pd.read_csv(csv) 
    df2 = pd.read_csv(csv_inf)
    
    fig = plt.figure(figsize=(5,4.5))
    ax = fig.add_subplot(111)
    #read in CSV as Pandas dataframe
    alphas = np.array([(1-n)*U/2. for n,U in zip(df["eta"].values, df['U'].values)])
    E_bi = df["E"].values

    E_inf = df2["E"].values
    dE = (E_bi-E_inf)/np.abs(E_inf)

    ax.plot(alphas,dE,label='$\Delta E$')
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("$\Delta E/|E_\infty|$")
    ax.set_ylim(-0.2,0.2)
    ax.plot(alphas,[0.]*len(alphas),label='$\Delta E$')

    #ax.legend(loc=1)

    plt.tight_layout()
    plt.show()

###########################################################################################################################
'''Single polaron energy calcs for variational and fixed a'''

def hybpol(x,eta, U):
    '''
    Single polaron energy in hybrid scheme, in units of hw
    x = [lambda,t]
    '''
    if x[1] == 0: return 3/4*x[0] - (1-eta)*U/2
    else:
        val = erfcx(x[1])
        return 3/4*x[0] - (1-eta)*U/2 *(1+x[1]*np.sqrt(x[0])) * val

def min_hybpol(args):
    n,U = args
    bnds = ((1E-4,1E2),(1E-5,1E3)) #lambda, t
    guess = np.array([10,0.5]) #lambda, t
    result = minimize(hybpol,guess,args=(n,U,),bounds=bnds)
    minvals = result.x
    E_pol = hybpol(minvals, n,U)
    #return sigma/l = sqrt(2/lam), a = 1/(1+t*sqrt(lambda))
    return n,U,1/(1+minvals[1]*np.sqrt(minvals[0])),np.sqrt(2/minvals[0]),0,E_pol

def hybpol_afix(x,eta, U, a):
    '''
    Single polaron energy in hybrid scheme, in units of hw. Hold a = 1/(1+t*sqrt(lambda)) fixed
    x = [lambda]
    '''
    return 3/4*x[0] - (1-eta)*U/2 *1/a * erfcx((1-a)/(a*np.sqrt(x[0])))

def min_hybpol_afix(args):
    n,U,a = args
    bnds = ((1E-4,1E2),) #lambda
    guess = np.array([10.]) #lambda
    result = minimize(hybpol_afix,guess,args=(n,U,a),bounds=bnds)
    minvals = result.x
    E_pol = hybpol_afix(minvals, n,U,a)
    #return sigma/l = sqrt(2/lam), a = 1/(1+t*sqrt(lambda))
    return n,U,a,np.sqrt(2/minvals[0]),0,E_pol

###################################################################################################################
'''Find and plot E(a) at fixed alpha'''

def Plot_E_vs_a(csvname, xvar = 'a',plotcoulomb=False):
    '''ONLY use with files generated from GenE_vs_a() !!!'''

    fig = plt.figure(figsize=(10,4.5))
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    #read in CSV as Pandas dataframe
    df = pd.read_csv(csvname)
    alphas = np.array([(1-n)*U/2. for n,U in zip(df["eta"].values, df['U'].values)])
    df['alpha'] = alphas
    alphas = np.unique(alphas)

    for alpha in alphas:
        #divide into energy arrays by fixed alpha value
        Es = df[df['alpha'] == alpha]['E'].values
        ys = df[df['alpha'] == alpha]['y'].values
        ayes = df[df['alpha'] == alpha]['a'].values
        sigs = df[df['alpha'] == alpha]['s'].values
        ns = df[df['alpha'] == alpha]['eta'].values
        Us = df[df['alpha'] == alpha]['U'].values
        if xvar == 'y':
            ax.plot(ys,Es,label='$\\alpha=$' + str(alpha))
            ax2.plot(ys,ayes, label='$\\alpha=$' + str(alpha))
            ax.set_xlabel("$y$")
            ax2.set_xlabel("$y$")
            ax2.set_ylabel("$a$")
        elif xvar == 'a':
            ax.plot(ayes,Es,label='$\\alpha=$' + str(alpha))
            ax2.plot(ayes,ys, label='$\\alpha=$' + str(alpha))
            ax.set_xlabel("$a$")
            ax2.set_xlabel("$a$")
            ax2.set_ylabel("$y$")
        elif xvar == 's':
            ax.plot(sigs,Es,label='$\\alpha=$' + str(alpha))
            ax2.plot(sigs,ayes, label='$\\alpha=$' + str(alpha))
            ax.set_xlabel("$\sigma/l$")
            ax.set_xscale('log')
            ax2.set_xlabel("$\sigma/l$")
            ax2.set_ylabel("$a$")
        if plotcoulomb==True: 
            coul = np.array([n*U/(s*y) for n,U,s,y in zip(ns, Us, sigs, ys)]) #semiclassical Coulomb repulsion (screened by phonons) - to get order of magnitude variation
            ax.plot(ys,coul,label='Coulomb, $\\alpha=$' + str(alpha))

    ax.set_ylabel("E/K")
    #ax.set_ylim(-20,10)
    ax.legend(loc=4)
    ax2.legend(loc=4)

    plt.tight_layout()
    plt.show()

def GenE_vs_a():
    '''run multiprocessing to generate energy as a function of a: E(a) at a couple different values of alpha'''
    ns=[0.]
    Us = [2.,10,16,20] #corresponds to alpha = 1,5,8,10
    #Us = [20.]
    ayes = np.linspace(1E-5,1,100)
    #ayes = np.linspace(0.2,0.4,20)
    ys = np.linspace(1E-3,5,80)
    ss = np.linspace(-5,5,30)
    #ys = [1E-3]
    y = 10. #500 for y->inf limit, 5-10 for finite/bipolaron/wigner crystal limit (check for numerical integration trouble)
    z_c = 10.
    a_c = 0.6

    #csvname = "./data/testnak.csv"
    csvname = "./data/nak_E(y)_v2.csv" #nakano bipolaron for finite y
    #csvname = "./data/nak_E(a)_inf_eta0-2.csv" #nakano bipolaron for y->inf
    #csvname = "./data/pol_E(a).csv" #nakano/hybrid single polaron

    df={}
    quantities = ['eta','U','a','s','y','E']

    for i in quantities:
        df[i]=[]

    tic = time.perf_counter()
    with multiprocessing.Pool(processes=4) as pool:
        for n,U in product(ns,Us):
            #polaron run
            #job_args = [(n,U,a) for a in ayes]
            #results = pool.map(min_hybpol_afix, job_args)
            
            #bipolaron run, E(a)
            #job_args = [(n,U,z_c,a_c,a) for a in ayes]
            #results = pool.map(nag.min_E_afix_inf, job_args)

            #do same for E(y) now
            job_args = [(n,U,z_c,a_c, y) for y in ys]
            results = pool.map(nag.min_E_bip_yfix_ln, job_args)

            #and E(sig)
            #job_args = [(s, 5., 1., n, U, z_c, a_c) for s in ss]
            #results = pool.map(nag.E_bip_aysfix, job_args)
            for res in results:
                for name, val in zip(quantities, res):
                    df[name].append(val)

    toc = time.perf_counter()
    print(f"time taken: {toc-tic:0.4f} s, {(toc-tic)/60:0.3f} min")
    data = pd.DataFrame(df)
    pd.set_option("display.max.columns",None)
    print(data)
    data.to_csv(csvname,sep=',',index=False)

    Plot_E_vs_a(csvname,'y')

##########################################################################################################################
'''Generate maps of E(a,sigma) at a couple fixed values of y=0, 10, "infty" to help better understand contradictions in E(a) and E(y) results'''
def E_asig(y, n, U):
    csvname = "./data/nak_E(a,s)_n_" + str(n) + "_U_" + str(U) + "_y_" + str(y) + '.csv'
    z_c = 10.
    a_c = 0.6
    ayes = np.linspace(0,1,100)
    sigs = np.geomspace(1E-2,1E2,100)

    df={}
    quantities = ['eta','U','a','s','y','E']

    for i in quantities:
        df[i]=[]

    tic = time.perf_counter()
    with multiprocessing.Pool(processes=4) as pool:
        job_args = [(s,y,a,n,U,z_c,a_c) for a,s in product(ayes, sigs)]
        results = pool.map(nag.E_bip_aysfix, job_args)

        for res in results:
            for name, val in zip(quantities, res):
                df[name].append(val)

    toc = time.perf_counter()
    print(f"time taken: {toc-tic:0.4f} s, {(toc-tic)/60:0.3f} min")
    data = pd.DataFrame(df)
    pd.set_option("display.max.columns",None)
    print(data)
    data.to_csv(csvname,sep=',',index=False)
    return csvname

def plotContour(filename, colnames,xlims=(), ylims=(), zlims=(),save=False, zero=False,suffix='_phasedia.png', logplot=0,minmax=0):
    '''
    inputs:
        save: whether to save plot as a file
        zero: plot Z==0 contour
        suffix: save file name suffix
        point: plot the STO value for eta and U
    '''
    #phase diagram!
    fig = plt.figure(figsize=(6,4.5))
    ax = fig.add_subplot(111)
    #df = add_d(filename)
    df = pd.read_csv(filename)
    ns, Us, Z = ps.parse_CSV(df,colnames)
    a,b,c = colnames
    print(Z.min())

    #set color bar min/max
    if len(zlims)>0: cpmin, cpmax = zlims
    elif minmax == 0:
        cpmin = max(-1,Z.min())
        cpmax = min(1,Z.max())
    elif minmax == 1: #for binding energy
        cpmin = max(-0.65,Z.min())
        cpmax=0.
    elif minmax == 2:
        cpmin = Z.min()
        cpmax=10.
    else:
        cpmin = Z.min()
        cpmax = Z.max()
    cp = ax.contourf(ns, Us, Z, levels = MaxNLocator(nbins=20).tick_values(cpmin, cpmax))

    #set limits on x and y axes if argument given
    if len(xlims) >0: ax.set_xlim(xlims[0],xlims[1])
    if len(ylims) >0: ax.set_ylim(ylims[0],ylims[1])

    if logplot == 1:
        ax.semilogy()
    if logplot == 2:
        ax.loglog()

    cbar=fig.colorbar(cp) # Add a colorbar to a plot
    cbar.ax.set_ylabel(c)
    ax.set_xlabel(a)
    ax.set_ylabel(b)
    plt.tight_layout()
    print(ps.format_filename(filename))
    if save == True:
        plt.savefig(ps.SavePath(filename,suffix))
    plt.show()

def E_binding(filename1, filename2, colnames,save=False):
    '''
    Plot binding energy for phase diagram
    inputs:
        filename1: E_bip file
        filename2: E_inf file
        save: whether to save plot as a file
        colnames: array of [xname, yname, zname]
    '''
    #phase diagram!
    fig = plt.figure(figsize=(6,4.5))
    ax = fig.add_subplot(111)
    df = pd.read_csv(filename1)
    ns, Us, Z = ps.parse_CSV(df,colnames)
    a,b,c = colnames
    print(Z.min())
    df2 = pd.read_csv(filename2)
    _,_,Z2 = ps.parse_CSV(df2,colnames)
    Zbind = (Z-Z2)/np.abs(Z2)

    cp = ax.contourf(ns, Us, Zbind, levels = MaxNLocator(nbins=20).tick_values(max(-0.65,Zbind.min()),0))
    cbar=fig.colorbar(cp) # Add a colorbar to a plot
    cbar.ax.set_ylabel(c)
    ax.set_xlabel(a)
    ax.set_ylabel(b)
    plt.tight_layout()
    print(ps.format_filename(filename1))
    if save == True:
        plt.savefig(ps.SavePath(filename1,".png"))
    plt.show()

##########################################################################################################################

def PoolParty(csvname):
    '''run multiprocessing to generate energy CSV for range of alpha vals'''
    ns=np.linspace(0,0.035,20)
    Us = np.linspace(16,30,50)
    #ns = [0.]
    #Us = np.linspace(1E-3,20,50)
    a=1.
    z_c = 10.
    y = 500.
    a_c = 0.6 #dividing a value

    df={}
    quantities = ['eta','U','a','s','y','E']

    for i in quantities:
        df[i]=[]

    tic = time.perf_counter()
    with multiprocessing.Pool(processes=4) as pool:
        #bipolaron runs for y->inf
        job_args = [(n,u,z_c,a_c) for n,u in product(ns,Us)]
        #results = pool.map(nag.min_E_afix_inf, job_args)
        #results = pool.map(nag.min_E_avar_inf, job_args)
        #results = pool.map(nag.min_E_a01_inf, job_args)

        #bipolaron run for finite y
        results = pool.map(nag.min_E_bip_ln2, job_args)
        #results = pool.map(nag.min_E_bip_sfix, job_args)

        #polaron run
        #job_args = [(n,u) for n,u in product(ns,Us)]
        #results = pool.map(min_hybpol, job_args)

        for res in results:
            for name, val in zip(quantities, res):
                df[name].append(val)

    toc = time.perf_counter()
    print(f"time taken: {toc-tic:0.4f} s, {(toc-tic)/60:0.3f} min")
    data = pd.DataFrame(df)
    pd.set_option("display.max.columns",None)
    print(data)
    data.to_csv(csvname,sep=',',index=False)

################################################################################################################################
'''Find the fourier transform of the phonon displacement function f(k) to visualize the induced potential
   See https://stackoverflow.com/questions/39269804/fft-normalization-with-numpy for reference
'''

def FFT(a=1,y=1,s=1,opt='gaus'):
    '''Plot xy and yz contour plots of f(r) for the Nakano and Gaussian calculations. I found this site very helpful for understanding how to use NumPy's FFT algorithm: https://scipython.com/book/chapter-6-numpy/examples/blurring-an-image-with-a-two-dimensional-fft/'''

    def gaus_f_k(kx,ky,kz, y, s, alf):
        '''Phonon displacement function f(kx,ky,kz) - ignore -1j prefactor and set V/l^3 = 1'''
        return 2.*s/(1+np.exp(-y**2/2)) * np.sqrt(4*np.pi*alf) * np.exp(-0.25* (kx**2+ky**2+kz**2)) * (np.cos(0.5*y*kz) + np.exp(-0.5*y**2) )* 1/ np.sqrt(kx**2+ky**2+kz**2)
    
    def gaus_f_r(rx,ry,rz, y, s, alf):
        return 2./(s**2* (1+np.exp(-y**2/2)) ) * np.sqrt(alf/np.pi**3) * ( 2*np.exp(-y**2) * dawsn(np.sqrt( rx**2+ry**2+rz**2) )/np.sqrt( rx**2+ry**2+rz**2) + dawsn(np.sqrt( rx**2+ry**2+(rz+y/2)**2) )/np.sqrt( rx**2+ry**2+(rz+y/2)**2) + dawsn(np.sqrt( rx**2+ry**2+(rz-y/2)**2) )/np.sqrt( rx**2+ry**2+(rz-y/2)**2) )

    def nak_f_k(kx,ky,kz,a,y,s,alf):
        '''Phonon displacement function f(kx,ky,kz) for the Nakano calc'''
        numer = np.exp(-0.25* (1-a)**2 * (kx**2+ky**2+kz**2)) * (np.cos(0.5*(1-a)* y*kz) + np.exp(-y**2/2) ) + np.exp(-0.25*(1+a**2)*(kx**2+ky**2+kz**2)) * (np.cos(0.5* (1+a)* y*kz)+ np.exp(-y**2/2) )
        denom = 1 + (a/s)**2* (kx**2+ky**2+kz**2) + np.exp(-a**2 * (kx**2+ky**2+kz**2)/2) / (1+ np.exp(-y**2/2) ) * (np.cos(a*y*kz)+ np.exp(-y**2/2) )
        return s* np.sqrt(4*np.pi*alf)/(1+np.exp(-y**2/2)) * numer/denom * 1/ np.sqrt(kx**2+ky**2+kz**2) #setting V/l^3 = 1
    
    rmax = 5
    dr = 0.1
    rs = np.arange(-rmax,rmax, dr) #shift away from 0
    ks = np.fft.fftfreq(len(rs),dr)+0.001 #ordered as 0, 1,...,kmax,-kmax,...,-1 for inverse FFT input 
    kord = np.fft.fftshift(ks) #ordered ks from -kmax, ...,0,..., kmax for FFT input
    ky,kz = np.meshgrid(ks,ks, sparse=True) #returns (knum,1) and (1,knum) matrices in weird order
    kx=np.zeros(len(ks))
    fig = plt.figure(figsize=(10,9))
    ax = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    if opt == 'gaus':
        f_ks_yz = gaus_f_k(kx,ky,kz,y,s,alpha) #calc w/ disordered k's
        f_ks_yz_ord = np.fft.fftshift(f_ks_yz) #then shift into ordered arrangement
    else:
        f_ks_yz = nak_f_k(kx,ky,kz,a,y,s,alpha)
        f_ks_yz_ord = np.fft.fftshift(f_ks_yz)
        f_ks_xy = nak_f_k(ky,kz,kx,a,y,s,alpha)
        f_ks_xy_ord = np.fft.fftshift(f_ks_xy)

        cp3 = ax3.contourf(kord,kord,f_ks_xy_ord.real,levels = MaxNLocator(nbins=20).tick_values(0, 10))
        cbar3 = fig.colorbar(cp3, ax=ax3)
        cbar3.set_label("$f_k$")
        ax3.set_xlabel("$k_x$")
        ax3.set_ylabel("$k_y$")
    
    #cp = ax.contourf(kord,kord,f_ks_yz_ord.real)
    cp = ax.contourf(kord,kord,f_ks_yz_ord.real,levels = MaxNLocator(nbins=20).tick_values(0, 10))
    cbar = fig.colorbar(cp, ax=ax)
    cbar.set_label("$f_k$")
    ax.set_xlabel("$k_y$")
    ax.set_ylabel("$k_z$")

    ifft_yz = np.fft.ifft2(f_ks_yz) #needs weird input order, prod "normal" output order?
    #print(ifft_yz)
    cent_ifft_yz = np.fft.ifftshift(ifft_yz) #center the inv FFT frequencies to match up with normal ordered positions
    cp2 = ax2.contourf(rs, rs, cent_ifft_yz.real, levels = MaxNLocator(nbins=20).tick_values(ifft_yz.min(), ifft_yz.max()))
    cbar2 = fig.colorbar(cp2, ax=ax2) 
    cbar2.set_label("$f_r$ (iFFT)")
    ax2.set_xlabel("$r_y$")
    ax2.set_ylabel("$r_z$")

    if opt=='gaus':
        #find FFT to check the Fourier Transform
        fft_yz = np.fft.fft2(ifft_yz) #output is wonky order? Or "regular order"
        fft_yz = np.fft.fftshift(fft_yz) #center the frequencies
        cp4 = ax4.contourf(kord,kord,fft_yz.real)
        cbar4 = fig.colorbar(cp4, ax=ax4) 
        cbar4.set_label("$f_k$ (reco)")
        ax4.set_xlabel("$k_y$")
        ax4.set_ylabel("$k_z$")

        ry,rz = np.meshgrid(rs,rs, sparse=True) #returns (knum,1) and (1,knum) matrices
        rx=np.zeros(len(rs))
        f_rs_yz = gaus_f_r(rx,ry,rz,y,s,alpha)
        #print(f_rs)
        cp3 = ax3.contourf(rs,rs,f_rs_yz)
        cbar3 = fig.colorbar(cp3, ax=ax3) 
        cbar3.set_label("$f_r$ (ana)")
        ax3.set_xlabel("$r_y$")
        ax3.set_ylabel("$r_z$")
    else:
        #i.e. opt == 'nak'
        ifft_xy = np.fft.ifft2(f_ks_xy) #needs weird input order, prod "normal" output order?
        cent_ifft_xy = np.fft.ifftshift(ifft_xy) #center the inv FFT frequencies to match up with normal ordered positions
        cp4 = ax4.contourf(rs, rs, cent_ifft_xy.real, levels = MaxNLocator(nbins=20).tick_values(ifft_xy.min(), ifft_xy.max()))
        cbar4 = fig.colorbar(cp4, ax=ax4) 
        cbar4.set_label("$f_r$ (iFFT)")
        ax4.set_xlabel("$r_x$")
        ax4.set_ylabel("$r_y$")

    plt.tight_layout()
    plt.show()
    
def f_r_special(gen=False):
    '''Plot f(r) for r || k and r perp k, r in units of sigma'''
    csvname = "./data/nak_fr_para.csv"
    rs = np.linspace(0,10,100)
    a = 1.
    y = 1.
    s = 1.
    if gen == True:
        df={}
        quantities = ['r','fr_para','fr_perp']

        for i in quantities:
            df[i]=[]

        tic = time.perf_counter()
        with multiprocessing.Pool(processes=4) as pool:
            job_args = [(r,a,y,s,alpha) for r in rs]
            results = pool.map(nag.fr_special, job_args)

            for res in results:
                for name, val in zip(quantities, res):
                    df[name].append(val)

        toc = time.perf_counter()
        print(f"time taken: {toc-tic:0.4f} s, {(toc-tic)/60:0.3f} min")
        data = pd.DataFrame(df)
        pd.set_option("display.max.columns",None)
        print(data)
        data.to_csv(csvname,sep=',',index=False)
    
    fig = plt.figure(figsize=(6,4.5))
    ax = fig.add_subplot(111)
    df = pd.read_csv(csvname)
    fr_para = [complex(str(v).replace(" ", "")) for v in df['fr_para'].to_numpy()]
    fr_perp = [complex(str(v).replace(" ", "")) for v in df['fr_perp'].to_numpy()]
    ax.plot(df['r'].values, np.real(fr_para),label='para (real)')
    ax.plot(df['r'].values, np.real(fr_perp),label='perp (real)')
    ax.plot(df['r'].values, np.imag(fr_para),label='para (imag)')
    ax.plot(df['r'].values, np.imag(fr_perp),label='perp (imag)')
    ax.set_xlabel('r')
    ax.set_ylabel('f(r)')
    ax.legend(loc=1)
    plt.tight_layout()
    plt.show()

def DensityPlot3D(s=1,y=1):
    '''Plot electron density rho(r) for nakano'''
    def rho(rx,ry,rz,s,y):
        return 1./(s**3 * (1 + np.exp(-y**2/2))) * np.exp(-(rx**2 + ry**2))* (np.exp(-(rz - y/2)**2) + np.exp(-(rz + y/2)**2) + 2*np.exp(-(rz**2 + y**2/2)))

    rs = np.linspace(-3, 3, 30)
    ry = np.linspace(-3, 3, 30)

    rx,ry = np.meshgrid(rs,rs)
    dens = rho(rx,ry,rs,s,y)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(rx,ry,dens, 50, cmap='binary')
    ax.set_xlabel('rx')
    ax.set_ylabel('ry')
    ax.set_zlabel('$\\rho$')
    plt.show()

###############################################################################################################################

if __name__ == '__main__':
    import pandas as pd
    csvname = "./data/nakano_yfin_zoom.csv"
    csvname3 = "./data/nakano_yinf_zoom.csv"
    csvname2 = "./data/nak_alpha_s0-5.csv" #orig optimization formulation
    csvname2b = "./data/nak_alpha_yfin.csv"
    #csvname2 = "./data/nagano_a01_inf.csv"
    #csvname = "./data/nagano_pol.csv"
    #FFT(a=1,y=10,s=1,opt='nak')
    #f_r_special()
    DensityPlot3D()
    #PoolParty(csvname)
    #PlotE(csvname, fit=False)
    #PlotBindingE([csvname,csvname3])
    #GenE_vs_a()
    #Plot_E_vs_a("./data/nak_E(y)_eta0-1.csv")
    #name = E_asig(10,0,20)
    #name = "./data/nak_E(a,s)_n_0_U_20_y_1e-3.csv"
    #plotContour(csvname3, colnames=['eta','U','E'],xlims=(), ylims=(), zlims=(),save=False, minmax=5)
    #E_binding(csvname, csvname3, colnames=['eta','U','E'])



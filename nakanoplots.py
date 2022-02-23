import multiprocessing 
import numpy as np
from matplotlib.ticker import MaxNLocator, LogLocator
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
from skimage import measure
from matplotlib.colors import LogNorm
#from mpl_toolkits.mplot3d import Axes3D

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
epssr = 23000 #of the material, exptly measured static dielectric const at 0 freq
epsinf = 2.394**2 #optical dielectric const = (index of refraction)**2
conv = 1E-9/1.602E-19 #convert statC (expressions with elec^2) to eV
convJ = 1/1.602E-19 #convert J to eV
eta_STO = epsinf/epssr
alpha = (elec**2*1E-9)/hbar*np.sqrt(m/(2*hbar*w))*1/epsinf*(1 - epsinf/epssr) #convert statC to J*m
l = np.sqrt(hbar/(2*m*w)) #in units of m 
U_STO = elec**2/(epsinf*hbar)*np.sqrt(2*m/(hbar*w))*1E-9 #convert statC in e^2 term into J to make U dimensionless
epsinfKTO = 4.6; eps0KTO = 3800; wKTO = 299792458*826*100; UKTO = elec**2/(epsinfKTO*hbar)*np.sqrt(2*m/(hbar*wKTO))*1E-9; etaKTO = epsinfKTO/eps0KTO

#############################################################################################################################
'''Plotting functions'''

def PlotE(csvname,fit=False,opt='fin', multiplot=False):
    '''Plot energy, sep dist (y), elec size (sigma) of a single data file
    Input:
        opt: 'fin' or 'inf' determines whether to plot for finite y or y->inf data

    '''
    def multicolor_ylabel(ax,list_of_strings,list_of_colors,axis='x',anchorpad=0,**kw):
        """this function creates axes labels with multiple colors
        ax specifies the axes object where the labels should be drawn
        list_of_strings is a list of all of the text items
        list_if_colors is a corresponding list of colors for the strings
        axis='x', 'y', or 'both' and specifies which label(s) should be drawn"""

        from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

        # x-axis label
        if axis=='x' or axis=='both':
            boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',**kw)) 
                    for text,color in zip(list_of_strings,list_of_colors) ]
            xbox = HPacker(children=boxes,align="center",pad=0, sep=5)
            anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=anchorpad,frameon=False,bbox_to_anchor=(0.2, -0.09),
                                          bbox_transform=ax.transAxes, borderpad=0.)
            ax.add_artist(anchored_xbox)

        # y-axis label
        if axis=='y' or axis=='both':
            boxes = [TextArea(text, textprops=dict(color=color, ha='right',va='bottom',rotation=90,**kw)) 
                    for text,color in zip(list_of_strings[::-1],list_of_colors[::-1]) ]
            ybox = VPacker(children=boxes,align="center", pad=0, sep=5)
            anchored_ybox = AnchoredOffsetbox(loc=1, child=ybox, pad=anchorpad, frameon=False, bbox_to_anchor=(1.185, 0.57), 
                                          bbox_transform=ax.transAxes, borderpad=0.)
            ax.add_artist(anchored_ybox)

    df = pd.read_csv(csvname) 
    
    fig = plt.figure(figsize=(10,9))
    ax = fig.add_subplot(221)
    #read in CSV as Pandas dataframe
    alphas = np.array([(1-n)*U/2. for n,U in zip(df["eta"].values, df['U'].values)])
    if opt == 'fin':
        E = df["E"].values
        sigs = df["s"].values
        ys = df["y"].values
        ayes = df["a"].values
    else:
        E = df["Einf"].values
        sigs = df["sinf"].values
        ys = df["yinf"].values
        ayes = df["ainf"].values

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
    #compare with hybrid calc results
    df4 = pd.read_csv("./data/hyb_alpha.csv")
    alphas4 = np.array([(1-n)*U/2. for n,U in zip(df4["eta"].values, df4['U'].values)])
    '''
    if opt == 'fin':
        #just plot the energy comparison plot with Devreese
        fig1 = plt.figure(figsize=(6,4.5))
        ax1 = fig1.add_subplot(111)
        df3 = pd.read_csv("./data/nakano_yinf_U40.csv") #y->inf limit
        alphas3 = np.array([(1-n)*U/2. for n,U in zip(df3["eta"].values, df3['U'].values)])
        E3 = df3["E"].values
        df5 = pd.read_csv("./data/devreese0.csv")
        alphas5 = np.array([(1-n)*U/2. for n,U in zip(df5["eta"].values, df5['U'].values)])
        df7 = pd.read_csv("./data/gauss_U40.csv")
        alphas7 = np.array([(1-n)*U/2. for n,U in zip(df7["eta"].values, df7['U'].values)])
        ax1.plot(alphas,E + 2*alphas,'r',label='$E_{opt}$')
        ax1.plot(alphas3, E3 + 2*alphas3,'k:', label='$E_\infty$')
        ax1.plot(alphas7, df7['E'].values +2*alphas7,'g-.', label='gaus')
        ax1.plot(alphas5, df5['E'].values +2*alphas5, 'c--',label='DEV92')
        ax1.set_xlabel("$\\alpha$")
        ax1.set_ylabel("$E/K + 2\\alpha$")
        ax1.legend(loc=3)
        ax1.set_xlim(0,15)
        ax1.set_ylim(bottom=-25)
    ax2.legend(loc=1)
    ax3.legend(loc=1)
    ax4.legend(loc=1)
    
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
   
    if multiplot == True: 
        for val in np.unique(df['eta'].values):
            val, Us, Ys = ps.FindArrs(df, ['U','E','a','y','eta'], 'eta', val)
            Es,ayes,ys,ns = Ys
            alphas = np.array([(1-n)*U/2. for n,U in zip(ns, Us)])

            #plot metastable solutions
            dfstr = pd.read_csv('./data/nakano_U20_str.csv') #strong coupling solution - plot as dashed line
            _, Ustr, Ystr = ps.FindArrs(dfstr, ['U','E','a','y','eta'], 'eta', val)
            Estr, astr, ystr, nstr = Ystr
            idx = np.where(astr < 0.5)[0][1:]
            astr= astr[idx]
            Estr = Estr[idx]
            ystr = ystr[idx]
            alphastr = np.array([(1-n)*U/2. for n,U in zip(nstr[idx], Ustr[idx])])

            dfwk = pd.read_csv('./data/nakano_U40_wk.csv') #strong coupling solution - plot as dashed line
            _, Uwk, Ywk = ps.FindArrs(dfwk, ['U','E','a','y','eta'], 'eta', val)
            Ewk, awk, ywk, nwk = Ywk
            alphawk = np.array([(1-n)*U/2. for n,U in zip(nwk, Uwk)])

            #Plot E vs y on two y axes on the same plot
            fig2 = plt.figure(figsize=(6,4.5))
            axe = fig2.add_subplot(111)
            axe.plot(alphas,Es,color='red', label='$\eta = %.2f$' %val)
            axe.plot(alphastr, Estr, 'r--')
            axe.plot(alphawk, Ewk, 'r:')
            axe.set_xlabel("$\\alpha$")
            axe.set_ylabel("$E/K$",color='red')
            #axe.legend(loc=3)
            tax = axe.twinx()
            yidx = np.where(ys>ys.min())[0]
            yphys = ys[yidx[0]:] #ignore part where y diverges
            al_ys = alphas[yidx[0]:]
            tax.plot(al_ys,yphys,color='blue',label='$y + 0.1$')
       
            tax.plot(alphastr,ystr,'b--')
            tax.plot(alphas,ayes,color='green',label='$a$')
            tax.plot(alphastr,astr,'g--')
            tax.plot(alphawk,awk,'g:')
            #tax.semilogy()
            multicolor_ylabel(tax,('$y$',',','$a$'),('b','k','g'),axis='y')
            plt.tight_layout()
            plt.show()

def PlotBindingE(csvnames, realval=False):
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
    if realval==False:
        dE = (E_bi-E_inf)/np.abs(E_inf)
        ax.set_ylabel("$\Delta E/|E_\infty|$")
    else: 
        dE = E_bi-E_inf
        ax.set_ylabel("$\Delta E$")

    ax.plot(alphas,dE,label='$\Delta E$')
    ax.set_xlabel("$\\alpha$")
    #ax.set_ylim(-0.2,0.2)
    ax.plot(alphas,[0.]*len(alphas),label='$\Delta E=0$')

    #ax.legend(loc=1)

    plt.tight_layout()
    plt.show()

###########################################################################################################################
'''Create inset in phase diagram paper plot (Fig 1b)'''
#Fig 1 - binding energy
def FormatPhaseDia(csvname2, csvname2b):
    import os
    import string
    import mpl_toolkits.axes_grid1.inset_locator as mpl_il
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    path = os.path.dirname(os.path.abspath(__file__))
    path += "/data/"

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)

    #first, plot the data
    E_binding(csvname2, csvname2b, colnames=['eta','U','E'], point = True, logplot='y', realval = False, fig = fig, ax = ax, show=False)

    # Create a set of inset Axes: these should fill the bounding box allocated to
    # them.
    ax2 = plt.axes([0,0,1,1])
    # Manually set the position and relative size of the inset axes within ax1
    ip = mpl_il.InsetPosition(ax, [0.3,0.12,0.45,0.45]) #left, bottom, width, height
    ax2.set_axes_locator(ip)
    # Mark the region corresponding to the inset axes on ax1 and draw lines
    # in grey linking the two axes.
    mpl_il.mark_inset(ax, ax2, loc1=1, loc2=3, fc="none", ec='0.5')
    #E_binding(csvname2, csvname2b, colnames=['eta','U','E'], point = True, logplot='y', xlims=(0,0.003),ylims=(1,), fig=fig, ax = ax2, realval = True, scinot=True, cbscale=True)

    df = pd.read_csv(csvname2)
    df2 = pd.read_csv(csvname2b)
    dEs = np.array([(E-Einf) for E,Einf in zip(df['E'].values, df2['E'].values)])
    df['dE'] = dEs
    colnames=['eta','U','dE']
    ns, Us, Zbind = ps.parse_CSV(df,colnames)
    a,b,c = colnames
    cp = ax2.contourf(ns, Us, Zbind, levels = MaxNLocator(nbins=20).tick_values(Zbind.min(), 0))
    ax2.set_xlim(0,0.003)
    ax2.set_ylim(1, Us.max())
    ax2.semilogy()

    #Plot various material param values
    ax2.plot(eta_STO,U_STO,color='black',marker='.') #STO, strontium titanate
    ax2.annotate('STO', (eta_STO, U_STO), xytext=(8, -8), textcoords='offset pixels')
    ax2.plot(etaKTO,UKTO,color='black',marker='.') #KTO, potassium tantalate
    ax2.annotate('KTO', (etaKTO, UKTO), xytext=(8, -8), textcoords='offset pixels')

    cbar=fig.colorbar(cp, ax=ax2, format='%.2e', shrink=0.348, location='right', anchor=(-1.8,0.308), pad=0.01) # Add a colorbar to a plot; scientific notation
    cbar.ax.set_ylabel('$\Delta E$')

    ax2.set_xlabel(a)
    ax2.set_ylabel(b)
    plt.show()

def FormatWeakPD(generate=True, point=True, loglog=False, findconst=False, alt=False):
    '''
        alt: alternative method for generating phase diagrams, just by calculating the integral (depends on y, sigma) once for the el-ph contribution and then multiplying by different coefficients
        point: whether or not to label material points on weak coupling PD
        generate: generate data files for different values of sigma
        loglog: log-log plotting option (with fit)
        findconst: find individual contributions from KE, coul, eph terms in energy, compare to 2 infinitely separated electrons' energy (2E_pol = -2alpha)
    '''

    def fitweak(x,A,B):
        f = A + B*x
        return f
    #ss = [1,5,10,20]
    #ss = [2,5,8,10,12] #for _sub_ext
    ss = [.1,.5,1,2,3,5,8]
    #ss = [3,5,6,8,10]#5, 8, 12 for y=1000 
    csvnames = [""]*len(ss)
    csvnames_alt = [""]*len(ss)
    colors = ['red','green','blue','purple', 'orange','black','magenta']
    styles = ['solid','dotted','dashed','dashdot','solid','dotted','dashed']

    if loglog == True:
        #ext = '_loglog.csv'
        ext = '_loglog_y10.csv'
        #ext = '_loglog_sub_ext.csv' #subtraction of screened from bare Coulomb
        #ext = '_loglog_infty.csv' 
        ns=np.geomspace(2E-4,1,50)
    else:
        ext = '.csv'
        ns=np.linspace(0,0.095,50)
    #zoom in on weak-coupling regime; doesn't make sense for strong coupling which optimizes to its own sigma ~ 1
    Us = np.geomspace(1E-5,15,70)
    
    for i,sval in enumerate(ss):
        csvnames[i] = "./data/nakano_wkPD_s" + str(sval) + ext
        csvnames_alt[i] = "./data/nakano_wkPD_energy_consts_sval_" + str(sval) + ext

    print(csvnames)
       
    if alt == True & generate == True: 
        '''alternative calculation method, finding 1 integral value per combo of y and sigma (fixing y = 0, 500) and then varying coefficients/prefactors to generate rest of PD'''

        z_c = 10.
        yopt = 1.
        yinf = 10.
        dfs = [{}]*len(ss)
        quantities = ['eta','U','a','s','y','E', 'yinf', 'Einf', 'dE']
        for n,sval in enumerate(ss): 
            print(sval)
            for i in quantities: 
                dfs[n][i]=[]

            integral = nag.Weak_Eph(sval, yopt, z_c) 
            integral_inf = nag.Weak_Eph(sval, yinf, z_c) 

            tic = time.perf_counter()
            with multiprocessing.Pool(processes=4) as pool:
                job_args = [(nval,u,sval, yopt, integral, yinf, integral_inf) for nval,u in product(ns,Us)]
                results = pool.map(nag.Weak_E, job_args)

                for res in results:
                    for name, val in zip(quantities, res):
                        dfs[n][name].append(val)
            toc = time.perf_counter()
            print(f"time taken: {toc-tic:0.4f} s, {(toc-tic)/60:0.3f} min")
            data = pd.DataFrame(dfs[n])
            data.to_csv(csvnames[n],sep=',',index=False)

    elif generate == True:
        '''run multiprocessing to generate energy CSV for range of alpha vals'''
    
        a=1.
        z_c = 10.
        y = 500. #default: 500
        a_c = 0.6 #dividing a value
        
        dfs = [{}]*len(ss)
        quantities = ['eta','U','a','s','y','E', 'ainf', 'sinf', 'yinf', 'Einf', 'dE']

        for n, sval in enumerate(ss):
            print(sval)
            for i in quantities: 
                dfs[n][i]=[]
            tic = time.perf_counter()
            with multiprocessing.Pool(processes=4) as pool:
                job_args = [(n,u,z_c,a_c,y,sval) for n,u in product(ns,Us)]
                results = pool.map(nag.min_E_nak, job_args)

                for res in results:
                    for name, val in zip(quantities, res):
                        dfs[n][name].append(val)

            toc = time.perf_counter()
            print(f"time taken: {toc-tic:0.4f} s, {(toc-tic)/60:0.3f} min")
            data = pd.DataFrame(dfs[n])
            pd.set_option("display.max.columns",None)
            print(data)
            
            data.to_csv(csvnames[n],sep=',',index=False)
        
    #Now plot dE = 0 curves on phase diagram
    fig = plt.figure(figsize=(6,4.5))
    ax = fig.add_subplot(111)
    ax.set_xlabel('$\eta$')
    ax.set_ylabel('$U$')
    fig2 = plt.figure(figsize=(4.5,7.5))
    ax2 = fig2.add_subplot(211)
    ax2.set_xlabel('$\ln \eta$')
    ax2.set_ylabel('$\ln U$')
    ax3 = fig2.add_subplot(212)
    ax3.set_ylabel('Fit const $A$')
    ax3.set_xlabel('$s$')

    #labelpos = [[(0.00158,0.0002)],[(0.005,0.0002)],[(0.00158,0.0002)],[(0.00158,0.0002)],[(0.00158,0.0002)],[(0.00158,0.0002)]]
    labelpos = [[(0.00158,0.0002)],[(0.00158,0.0002)],[(0.00158,0.0002)],[(0.00158,0.0002)],[(0.00158,0.0002)],[(0.00158,0.0002)],[(0.00158,0.0002)]]
    As = np.zeros(len(ss))
    etaUs = [[]]*len(ss)
    
    for n,sval in enumerate(ss):
        print(sval)
        df = pd.read_csv(csvnames[n])

        ns, Us, Zbind = ps.parse_CSV(df,['eta','U','dE'])
        ax.semilogy()

        #plot a colored contour plot of one of the fixed sigma vals to check that binding is permitted below the dE=0 curve (not above)
        '''
        if n == len(ss)-1:
            fig2,axx = plt.subplots()
            s8cp = axx.contourf(ns, Us, Zbind) #plot s = 8 contour plot
            cbar2=fig2.colorbar(s8cp) # Add a colorbar to a plot
            axx.set_xscale('log')
            axx.set_yscale('log')
            plt.show()
        '''

        #plot binding energy = 0 contour
        zerocont = ax.contour(ns, Us, Zbind, [0.], colors=(colors[n],), linewidths=(1,), linestyles=(styles[n],), origin='lower') #plot dE = 0 curve
        e0,u0 = ps.FindCoords(zerocont)
        if len(e0) > 1:
            e0 = np.concatenate(e0,axis=None)
            u0 = np.concatenate(u0,axis=None)
        else:    
            e0 = e0[0]
            u0 = u0[0]
        print(u0[-1])
        #if sval == 4:
        #    print(list(zip(e0,u0)))
        etaUs[n] = np.array([n*u if n<1 else 1E-10*u for n,u in zip(e0,u0)])
         
        label = '$s = $' + str(ss[n])
        zerocont.collections[0].set_label(label)
        #ax.clabel(zerocont, zerocont.levels, inline=True, manual=labelpos[n], fmt=label, fontsize=10)

        if loglog == True:
            ax.semilogx()
            #ax.set_xlim(left=1E-4)
            logx = np.log(e0)
            logy = np.log(u0)
            
            ax2.plot(logx, logy, label= '$s = $' + str(ss[n]))

            bnds_w = ([-50,-5],[10,5]) # bounds for weak coupling fit
            guess_w =[0,0]
            if ss[n] < 15:
                if ss[n] < 2:
                    idx = np.where(logx >= -1)[0]
                else:
                    idx = np.where(logx >= -1.7)[0]
            else: idx = np.where(logx >= -2)[0]
            #print(idx)
            if len(idx) == 0: continue

            coeffs, p_cov_w = curve_fit(fitweak,logx[idx], logy[idx], p0=guess_w,bounds=bnds_w)
            As[n] = coeffs[0] #collect constant exponent of loglog fit
            print(coeffs)
            print(np.sqrt(np.diag(p_cov_w))) #standard deviation of the parameters in the fit
            ans_w = fitweak(logx[idx],coeffs[0],coeffs[1])
            ax2.plot(logx[idx],ans_w,color='red',label='fit')

            textstr = r'$\ln (U_{%d}) = %.2f + %.2f \ln (\eta)$' %(sval,coeffs[0], coeffs[1],)
            ax2.text(0.02, 0.01+(len(ss)-n)*0.05, textstr, transform=ax2.transAxes, fontsize=10, verticalalignment='top')

    ax.legend(loc=3)

    #scaling for sigma plots
    if loglog == True:
        textstr = r'$\ln (U_s) = A + B \ln (\eta)$'
        ax2.text(0.02, 0.01+(len(ss)+1)*0.05, textstr, transform=ax2.transAxes, fontsize=10, verticalalignment='top')
        names = ['s','A'] 
        ans, textstr = FitData(ss, As, names)
        ax3.plot(ss,ans,color='red',label='fit')
        ax3.plot(ss,As,'ko',label='data')

        ax3.text(0.02, 0.25, textstr, transform=ax3.transAxes, fontsize=12, verticalalignment='top')

    #Plot various material param values
    if point == True: 
        elems = ['STO','KTO','PbS','PbSe','PbTe','SnTe','GeTe']
        elemloc = [(eta_STO,U_STO), (etaKTO,UKTO), (9.05E-2,2.6), (8.18E-2,2.58), (8.26E-2,1.87), (3.75E-2,3.12), (8E-2,0.47)]
        textloc = [(8,-17),(8,-17),(-25,15),(-40,10),(6,-25),(-18,-18),(-70,-20)]
        textloc_log = [(8,-17),(-60, -17),(10,-5),(-60,10),(10,-20),(-70,0),(-70,-20)]
        if loglog == False:
            for elem, elemxy, textxy in zip(elems, elemloc, textloc):
                ax.plot(elemxy[0],elemxy[1],color='black',marker='.') 
                ax.annotate(elem, elemxy, xytext=textxy, textcoords='offset pixels')
        else:
            for elem, elemxy, textxy in zip(elems, elemloc, textloc_log):
                ax.plot(elemxy[0],elemxy[1],color='black',marker='.') 
                ax.annotate(elem, elemxy, xytext=textxy, textcoords='offset pixels')
            ax.set_xlim(left=2E-4)
    plt.tight_layout()
    plt.show()

def FindMatParams(constrained=True):
    elems = ['STO','KTO','PbS','PbSe','PbTe','SnTe','GeTe']
    ns = [eta_STO, etaKTO, 9.05E-2, 8.18E-2, 8.26E-2, 3.75E-2, 8E-2]
    Us = [U_STO, UKTO, 2.6, 2.58, 1.87, 3.12, 0.47] 
    z_c = 10.
    yopt = 1.
    yinf = 500.
    k_B = 1.380649E-23 #J/K, Boltzmann constant
    csvname = './data/nak_mats_constrained.csv' 
 
    if constrained:
        # Cheng et al. 2015 used 5 nm nanowires - convert to sigma/l = \tilde sigma:
        sig = [5E-9,60E-9]/l
        ss = np.log(sig)
        #specified max sigma value, do extrinsically constrained bipolaron calc
        df = {}
        df['matname'] = []
        quantities = ['eta','U','a','sig','y','E', 'yinf', 'Einf', 'dE']
        for i in quantities: 
            df[i]=[]
        for n,sval in enumerate(ss): 
            print(sval)
            integral = nag.Weak_Eph(sval, yopt, z_c) 
            integral_inf = nag.Weak_Eph(sval, yinf, z_c) 

            tic = time.perf_counter()
            with multiprocessing.Pool(processes=4) as pool:
                job_args = [(nval,u,sval, yopt, integral, yinf, integral_inf) for nval,u in zip(ns,Us)]
                results = pool.map(nag.Weak_E, job_args)
                
                for i,res in enumerate(results):
                    df['matname'].append(elems[i])
                    for name, val in zip(quantities, res):
                        df[name].append(val)
                    
        toc = time.perf_counter()
        print(f"time taken: {toc-tic:0.4f} s, {(toc-tic)/60:0.3f} min")
        data = pd.DataFrame(df)
        data['dE_real'] = df['dE']*np.abs(df['Einf'])* 0.1  #binding energy in real units, found from dE*abs(Einf)* (hbar w), hbar w = 100 meV for the dominant LO mode in STO. Units of eV
        data['Tp'] = data['dE_real']/(k_B*convJ) #preformed pairing temp estimate, found from dE = k_B T_p. Need to convert J to eV
        data.to_csv(csvname,sep=',',index=False)
        print(data)

def FitData(xvals, yvals, varnames, guess=[-1,-3],yerr=[], fit='lin', extrap=[]):
    def fitlinear(x,a,b):
        f = a*x + b 
        return f

    bnds = ([-10,-10],[5,5]) #bounds for weak coupling fit
    if len(yerr) > 0:
        param, p_cov = curve_fit(fitlinear,xvals, yvals, sigma=yerr, p0=guess,bounds=bnds)
    else:
        param, p_cov = curve_fit(fitlinear,xvals, yvals, p0=guess,bounds=bnds)
    #print(param)
    a,b = param
    X,Y = varnames
    aerr, berr = np.sqrt(np.diag(p_cov)) #standard deviation of the parameters in the fit
    
    if len(extrap) > 0:
        ans = np.array([fitlinear(x,a,b) for x in extrap])
    else:    
        ans = np.array([fitlinear(x,a,b) for x in xvals])
    
    textstr = '\n'.join((
        r'$%s(%s) = a%s + b$' % (Y, X, X),
        r'$a=%.7f \pm %.7f$' % (a, aerr),
        r'$b=%.6f \pm %.6f$' % (b, berr)
        ))

    print(r'$b=%.4f \pm %.4f$' % (b, berr))
    return ans, textstr

#generate table of El-ph integral values for y = y_opt, y_inf over a large range of sigma to hopefully find out something about its functional form, contributions to dE, etc.
def FindIntegral():
    ss = np.linspace(1,20,50)
    z_c = 10.
    yopt = 1.
    yinf = 1000.
    csvname = "./data/nak_eph_integral_wkcoupling_y" + str(yinf) +".csv"
    df2 = {}
    '''
    qtys = ['s','y','I_fin','yinf','I_inf'] #I denotes el-ph integral
    for i in qtys: 
        df2[i]=[]
    for n,sval in enumerate(ss): 
        integral = nag.Weak_Eph(sval, yopt, z_c) 
        integral_inf = nag.Weak_Eph(sval, yinf, z_c) 
        for name, val in zip(qtys, [sval,yopt,integral, yinf, integral_inf]):
            df2[name].append(val)
    data = pd.DataFrame(df2)
    pd.set_option("display.max.columns",None)
    print(data)
    data.to_csv(csvname,sep=',',index=False)
    '''
    fig = plt.figure(figsize=(6,4.5))
    ax = fig.add_subplot(111)
    df = pd.read_csv(csvname) 
    I_fin = df['I_fin'].values 
    I_inf = df['I_inf'].values 
    s = df['s'].values
    ax.plot(s,np.log10(I_fin-I_inf),label='I_fin')
    ax.legend()
    plt.tight_layout()
    plt.show()

def LgSigma(generate):
    ns=[0,0.01]
    Us = [0.1] 
    ss = np.linspace(0,15,200)
    csvname = "./data/nakano_E(s)_y10.csv"
    if generate == True:
        z_c = 10.
        yopt = 1.
        yinf = 10.
        df = {}
        quantities = ['eta','U','a','s','y','E', 'yinf', 'Einf', 'dE']
        for i in quantities: 
            df[i]=[]

        integral = [nag.Weak_Eph(sval, yopt, z_c) for sval in ss]
        integral_inf = [nag.Weak_Eph(sval, yinf, z_c) for sval in ss]
        print('done')
        tic = time.perf_counter()
    
        with multiprocessing.Pool(processes=4) as pool:
            for n,U in product(ns,Us):
                job_args = [(n,U,sval, yopt, integral[i], yinf, integral_inf[i]) for i,sval in enumerate(ss)]
                results = pool.map(nag.Weak_E, job_args)

                for res in results:
                    for name, val in zip(quantities, res):
                        df[name].append(val)
        toc = time.perf_counter()
        print(f"time taken: {toc-tic:0.4f} s, {(toc-tic)/60:0.3f} min")
        data = pd.DataFrame(df)
        data.to_csv(csvname,sep=',',index=False)
    
    fig, ax = plt.subplots(2,1,sharex='col')
    df = pd.read_csv(csvname) 
    etas = df['eta'].values
    dEs = df['dE'].values
    idx = np.where(etas == 0)[0]
    idx2 = np.where(etas == 0.01)[0]
    svals = np.log(df['s'].values)
    s1 = svals[idx]
    #dE1 = np.log(np.abs(dEs[idx]))
    dE1 = np.log(-(dEs[idx]))
    s2 = svals[idx2]
    dE2 = np.log(np.abs(dEs[idx2]))
    varnames = ['s','\ln(-\Delta E)'] 
    #nat log fit params
    idx = np.where((s1 >= 7) & (s1 <= 14))[0]
    idx2 = np.where((s2 >= 8) & (s2 <= 14))[0]
    #log base 10 fit params
    #idx = np.where((s1 >= 3) & (s1 <= 7))[0]
    #idx2 = np.where((s2 >= 3) & (s2 <= 14))[0]
    fit1, txt1 = FitData(s1[idx],dE1[idx],varnames)
    ax[0].plot(s1,dE1,label='$\eta = 0$')
    ax[0].plot(s1[idx],fit1,label='fit')
    ax[0].text(0.02, 0.27, txt1, transform=ax[0].transAxes, fontsize=10, verticalalignment='top')
    ax[0].set_ylabel('$' + varnames[1] + '$')
    varnames = ['s','\ln (|\Delta E|)'] 
    fit2, txt2 = FitData(s2[idx2],dE2[idx2],varnames)
    ax[1].text(0.02, 0.27, txt2, transform=ax[1].transAxes, fontsize=10, verticalalignment='top')
    ax[1].plot(s2,dE2,label='$\eta = 0.01$')
    ax[1].plot(s2[idx2],fit2,label='fit')
    ax[1].set_xlabel('$' + varnames[0] + '$')
    ax[1].set_ylabel('$' + varnames[1] + '$')
    ax[0].legend(loc=1)
    ax[1].legend(loc=1)
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

def Plot_E_vs_a(csvname, xvar = 'a',plotcoulomb=False,logplot=0, plotdE = False):
    '''ONLY use with files generated from GenE_vs_a() !!!'''

    fig = plt.figure(figsize=(5.5,4.5))
    ax = fig.add_subplot(111)
    fig2 = plt.figure(figsize=(5.5,4.5))
    ax2 = fig2.add_subplot(111)

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
            ax2.plot(ys,sigs, label='$\\alpha=$' + str(alpha))
            ax.set_xlabel("$y$")
            ax2.set_xlabel("$y$")
            ax2.set_ylabel("$a$")
            #ax2.set_ylabel("$\sigma/l$")
            
            if logplot == 1:
                ax.semilogx()
                ax2.semilogx()
                
            if plotcoulomb==True: 
                coul = np.array([n*U/(s*y) for n,U,s,y in zip(ns, Us, sigs, ys)]) #semiclassical Coulomb repulsion (screened by phonons) - to get order of magnitude variation
                ax.plot(ys,coul,label='Coulomb, $\\alpha=$' + str(alpha))
        elif xvar == 'a':
            ax.plot(ayes,Es,label='$\\alpha=$' + str(alpha))
            ax2.plot(ayes,sigs, label='$\\alpha=$' + str(alpha))
            if logplot == 1:
                ax.semilogx()
                ax2.semilogx()
            ax.set_xlabel("$a$")
            ax2.set_xlabel("$a$")
            ax2.set_ylabel("$\sigma/l$")
        elif xvar == 's':
            if plotdE == True:
                #dfinf = pd.read_csv("./data/testnak_inf.csv")
                #Einf_sfix = dfinf[df['alpha'] == alpha]['E'].values
                #dE_sfix = (Es-Einf_sfix)/np.abs(Einf_sfix)
                dE_sfix = df[df['alpha'] == alpha]['dE'].values
                ax.plot(sigs,dE_sfix,label='$\\alpha=$' + str(alpha))
            else:
                ax.plot(sigs,Es,label='$\\alpha=$' + str(alpha))
            ax2.plot(sigs,ayes, label='$\\alpha=$' + str(alpha))
            ax.set_xlabel("$\sigma/l$")
            ax.semilogx()
            if logplot == 1:
                ax.semilogy()
            #ax.set_ylim(bottom=0)
            ax2.set_xlabel("$\sigma/l$")
            ax2.set_ylabel("$a$")

    ax.set_ylabel("E/K")
    #ax.set_ylim(-20,10)
    ax.legend(loc=4)
    ax2.legend(loc=1)
    plt.tight_layout()
    plt.show()

def GenE_vs_a():
    '''run multiprocessing to generate energy as a function of a or y: E(a) or E(y) at a couple different values of alpha'''
    ns=[0,0.01]
    Us = [0.1] 
    #ayes = np.geomspace(1E-5,1,100)
    #ss = [np.log(60E-9/l), np.log(5E-9/l)]
    ss = np.linspace(0,15,150)
    #ys = np.geomspace(500,10000,300)
    ys = [1.,1000.]
    y = 1000. #500 for y->inf limit, 5-10 for finite/bipolaron/wigner crystal limit (check for numerical integration trouble)
    z_c = 10.
    a_c = 0.6

    #csvname = "./data/confined_STO.csv"
    #csvname = "./data/testnak.csv"
    csvname = "./data/nak_E(s).csv"
    #csvname = "./data/pol_E(a).csv" #nakano/hybrid single polaron

    df={}
    #quantities = ['eta','U','a','s','y','E']
    quantities = ['eta','U','a','s','y','E', 'ainf', 'sinf', 'yinf', 'Einf', 'dE']
    #quantities = ['eta','U','a','s','y','ke','eph','coul','E','Einf', 'dE'] #constants in format E= C1 -alpha*C2 + U*C3


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
            #job_args = [(n, U,z_c,a_c, y) for y in ys]
            #results = pool.map(nag.min_E_bip_yfix_ln, job_args)
            #job_args = [(n, U,z_c,a_c, 0.,y) for y in ys]
            #results = pool.map(nag.min_E_bip_ayfix, job_args)

            #and E(sig)
            job_args = [(n, U,z_c,a_c,s,y) for s in ss]
            #results = pool.map(nag.min_E_bip_sfix, job_args)
            #results = pool.map(nag.min_E_inf_sfix, job_args)
            results = pool.map(nag.min_E_sfix, job_args)

            #job_args = [(7., yval, 1., n, U, z_c, a_c) for yval in ys]
            #results = pool.map(nag.E_bip_aysfix, job_args)
            #display individual contributions from KE, eph, coul
            #results = pool.map(nag.FindNakConsts, job_args)
  
            for res in results:
                for name, val in zip(quantities, res):
                    df[name].append(val)

    toc = time.perf_counter()
    print(f"time taken: {toc-tic:0.4f} s, {(toc-tic)/60:0.3f} min")
    data = pd.DataFrame(df)
    pd.set_option("display.max.columns",None)
    print(data)
    data.to_csv(csvname,sep=',',index=False)

    Plot_E_vs_a(csvname,'y',logplot=1, plotdE=False)

def GenE_vs_eta_y_fixedU(generate = False):
    '''Generate a) an E vs eta plot at fixed U, plotting the bipolaron and 2*polaron energies; b) a plot of E(y) at different values of eta corresponding to regions before, slightly past, and far past the crossover value of binding eta_c to illustrate stable, metastable, and no minima at strong coupling'''
    ns = np.linspace(0,0.15,45) #eta vals
    Us = [30.]
    ys = np.linspace(0.05,10,85)
    z_c = 10.
    a_c = 0.6

    csvname = "./data/nak_E(eta).csv"
    csvname2 = "./data/nak_E(eta)_inf.csv"
    csvname3 = "./data/nak_E(y)_comp.csv" #nakano bipolaron for finite y
    qtys = ['eta','U','a','s','y','E']
    qtyinf = ['eta','U','a_inf','s_inf','y_inf','E_inf']

    df={}
    df_inf={}
    df3={}

    for i in qtys:
        df[i]=[]
        df3[i]=[]
    for i in qtyinf:
        df_inf[i] = []
    
    if generate == True:
        ''' 
        tic = time.perf_counter()
        with multiprocessing.Pool(processes=4) as pool:
        
            #produce E(eta)
            jargs = [(n, Us[0],z_c,a_c,500.) for n in ns]
            res = pool.map(nag.min_E_bip_ln2, jargs)
            for r in res:
                for name, val in zip(qtys, r):
                    df[name].append(val)
            resinf = pool.map(nag.min_E_inf, jargs)
            for r in resinf:
                for name, val in zip(qtyinf, r):
                    df_inf[name].append(val)
        
        toc = time.perf_counter()
        print(f"time taken: {toc-tic:0.4f} s, {(toc-tic)/60:0.3f} min")
    
        data = pd.DataFrame(df)
        pd.set_option("display.max.columns",None)
        print(data)
        data.to_csv(csvname,sep=',',index=False)
        pd.DataFrame(df_inf).to_csv(csvname2,sep=',',index=False)
        ''' 
        nvars = [0.,0.06, 0.15]
        tic = time.perf_counter()
        with multiprocessing.Pool(processes=4) as pool:
            #now generate E(y) at 3 vals of eta
            for n,U in product(nvars,Us):
                print(n,U)
                job_args = [(n,U,z_c,a_c, y) for y in ys]
                results = pool.map(nag.min_E_bip_yfix_ln, job_args)

                for res in results:
                    for name, val in zip(qtys, res):
                        df3[name].append(val)
        
        toc = time.perf_counter()
        print(f"time taken: {toc-tic:0.4f} s, {(toc-tic)/60:0.3f} min")
        pd.DataFrame(df3).to_csv(csvname3,sep=',',index=False)
        
    fig = plt.figure(figsize=(5,4.5))
    ax = fig.add_subplot(111)
    fig2 = plt.figure(figsize=(5,4.5))
    ax2 = fig2.add_subplot(111)
    #if want subplots
    #fig = plt.figure(figsize=(4.5,9))
    #ax = fig.add_subplot(211)
    #ax2 = fig.add_subplot(212)

    #read in CSV as Pandas dataframe
    df = pd.read_csv(csvname)
    df2 = pd.read_csv(csvname2)
    df3 = pd.read_csv(csvname3)
    idx = np.where(df['eta'].values <= 0.1)
    ax.plot(df['eta'].values[idx], df['E'].values[idx], label='$E_{opt}$')
    ax.plot(df2['eta'].values, df2['E_inf'].values, label='$E_\infty$')
    ax.set_xlabel('$\eta$')
    ax.set_ylabel('$E$')
    ax.legend(loc=4)
    #find crossover point of E & E_inf
    intersect = np.argwhere(np.diff(np.sign(df['E'].values - df2['E_inf'].values))).flatten()
    print(intersect)
    print(df['eta'].values[intersect])
    fig.set_tight_layout(True) #use if making 2 diff plots to prevent y-axis label from getting cut off

    etas = np.unique(df3['eta'].values)
    for eta in etas:
        #divide into energy arrays by fixed alpha value
        Es = df3[df3['eta'] == eta]['E'].values
        ys = df3[df3['eta'] == eta]['y'].values
        sigs = df3[df3['eta'] == eta]['s'].values
        ax2.plot(ys,Es, label='$\eta=$' + str(eta))
        if eta == etas[1]:
            ax2.plot(ys, np.full(len(ys),Es[-1]),'--')
    ax2.set_xlabel('$y$')
    ax2.set_ylabel('$E$')
    ax2.legend(loc=1)
    plt.tight_layout()
    plt.show()

from scipy.interpolate import interp1d
def eta_c_vs_U(generate = False, fit = 'lin', xfit=(), yfit=()):
    '''Generate a plot of eta_c vs U in the strong-coupling regime. eta_c = value of eta at which E_infty = E_opt (i.e. energy cost of bipolaron/single polaron formation is equal). Basically have to generate E(eta) datasets for each value of (eta, U), then find the crossover point and collect those into a separate data set and plot that against U.'''
    ns = np.linspace(0,0.08,30) #eta vals
    Us = np.geomspace(20,100,100)
    z_c = 10.
    a_c = 0.6
    y = 500.

    csvname = "./data/nak_E(eta_U).csv"
    quantities = ['eta','U','a','s','y','E', 'ainf', 'sinf', 'yinf', 'Einf', 'dE']
    df={}

    for i in quantities:
        df[i]=[]
    
    if generate == True:
        tic = time.perf_counter()
        with multiprocessing.Pool(processes=4) as pool:
        
            job_args = [(n,u,z_c,a_c,y,20.) for n,u in product(ns,Us)]
            res = pool.map(nag.min_E_nak, job_args)
            for r in res:
                    for name, val in zip(quantities, r):
                        df[name].append(val)
        
        toc = time.perf_counter()
        print(f"time taken: {toc-tic:0.4f} s, {(toc-tic)/60:0.3f} min")
    
        data = pd.DataFrame(df)
        pd.set_option("display.max.columns",None)
        print(data)
        data.to_csv(csvname,sep=',',index=False)
    
    #read in CSV as Pandas dataframe
    df = pd.read_csv(csvname)
    #for each unique value of U, find the eta_c crossover point of binding
    eta_cs = np.zeros(len(Us))
    for i,u in enumerate(Us):
        #divide into energy arrays by fixed U value
        Es = df[df['U'] == u]['E'].values
        Einfs = df[df['U'] == u]['Einf'].values

        #find crossover point of E & E_inf
        intersect = np.argwhere(np.diff(np.sign(Es - Einfs))).flatten()
        if intersect.size > 0:
            print(intersect, df[df['U'] == u]['eta'].values[intersect], ns[intersect])
            eta_cs[i] = ns[intersect]
        else: continue

    fig = plt.figure(figsize=(5,4.5))
    ax = fig.add_subplot(111)
    
    if fit == 'lin' or fit == 'exp':    
        logx = np.log10(Us)
        logy = np.log10(eta_cs)
        #ax.plot(logx, logy)
        
        def fitfxn(x,A,B):
            f = A + B*x
            return f
        def fitexp(x,A,B):
            return A* np.exp(B*x)       

        bnds_w = ([-50,-5],[10,5]) # #bounds for weak coupling fit
        guess_w =[0,0]
        
        if len(yfit) == 2:
            idx = np.where((logy <= yfit[1]) & (logy >= yfit[0]))[0] #for eta = 0
        elif len(xfit) == 2:
            idx = np.where((logx <= xfit[1]) & (logx >= xfit[0]))[0] #for eta = 0
        else:
            idx = np.where((logy <= -2) & (logy >= -10))[0] #for eta = 0
        
        #interpolate b/w points cuz they're all jaggedy
        interpfn = interp1d(logx,logy)
        new_ys = interpfn(logx[idx])
        ax.plot(np.log10(logx[idx]),np.log10(new_ys),'bo',label='interp')
        if fit == 'lin':
            coeffs, p_cov_w = curve_fit(fitfxn,logx[idx], new_ys, p0=guess_w,bounds=bnds_w)
            ans_w = fitfxn(logx[idx],coeffs[0],coeffs[1])
            textstr = '\n'.join((
                r'$\log (dE) = A + B \log U$',
                r'$A=%.4f$' % (coeffs[0], ),
                r'$B=%.4f$' % (coeffs[1], )
                ))
        elif fit == 'exp':
            coeffs, p_cov_w = curve_fit(fitexp,logx[idx], new_ys, p0=guess_w,bounds=bnds_w)
            ans_w = fitexp(logx[idx],coeffs[0],coeffs[1])
            textstr = '\n'.join((
                r'$\log (dE) = A*10^{B \log U}$',
                r'$A=%.4f$' % (coeffs[0], ),
                r'$B=%.4f$' % (coeffs[1], )
                ))
        print(coeffs)
        print(np.sqrt(np.diag(p_cov_w))) #standard deviation of the parameters in the fit
        #ax.plot(logx[idx],ans_w,color='red',label='fit')


        ax.text(0.05, 0.45, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top')
        ax.set_ylabel('$\log \eta_c$')
        ax.set_xlabel('$\log U$')
    else:
        ax.plot(Us, eta_cs,'.')
        ax.set_ylabel('$\eta_c$')
        ax.set_xlabel('$U$')

    ax.legend(loc=4)
    plt.tight_layout()
    plt.show()

##########################################################################################################################
'''Generate maps of E(a,sigma) at a couple fixed values of y=0, 10, "infty" to help better understand contradictions in E(a) and E(y) results'''
def E_asig(var, n, U,fixed='y'):
    if fixed == 'y':
        csvname = "./data/nak_E(a,s)_n_" + str(n) + "_U_" + str(U) + "_y_" + str(var) + '.csv'
    else:
        csvname = "./data/nak_E(y,s)_n_" + str(n) + "_U_" + str(U) + "_a_" + str(var) + '.csv'
    z_c = 10.
    a_c = 0.6
    ayes = np.linspace(0,1,100)
    sigs = np.linspace(-4,-1,100)
    ys = np.linspace(0.1,1,100)

    df={}
    quantities = ['eta','U','a','s','y','E']

    for i in quantities:
        df[i]=[]

    tic = time.perf_counter()
    with multiprocessing.Pool(processes=4) as pool:
        if fixed == 'y':
            job_args = [(s,var,a,n,U,z_c,a_c) for a,s in product(ayes, sigs)]
        else:
            job_args = [(s,y,var,n,U,z_c,a_c) for y,s in product(ys, sigs)]
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
    if fixed == 'y':
        plotContour(csvname, ['a','s','E'], minmax=5)
    else:
        plotContour(csvname, ['y','s','E'], minmax=5)
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

def PlotAtFixedVal(filenames, colnames, fixedqty, fixedvals, realval = False,logplot=0, fit=False, yfit=(),xfit=(),xlim=(),labels=[]):
    '''
    input:
        filenames: length 1 or 2 array
        colnames: length 2 array of [x,y]
        fixedqty: name of quantity held fixed while plotting
        fixedval: value of fixed quantity
    '''
    df = pd.read_csv(filenames[0])
    a = colnames[0] # this is the x axis column
    b = colnames[1] #these are the quantities to plot on the y-axis
    print(b)

    fig = plt.figure(figsize=(6,4.5))
    ax = fig.add_subplot(111)
    for val in fixedvals:
        if b == 'dE' and 'dE' not in df.columns: #and dE not in df col headers - need to check
            val, fixed_x, Efins = ps.FindArrs(df, [a,'E'], fixedqty, val)
            df2 = pd.read_csv(filenames[1])
            _, _, Einfs = ps.FindArrs(df2, [a,'E'], fixedqty, val)
            if realval == False: 
                ylist = [(Efin-Einf)/np.abs(Einf) for Efin, Einf in zip(Efins, Einfs)][0] #plot binding energy in relative units (of 2*polaron energy)
            else:
                ylist = [(Efin-Einf) for Efin, Einf in zip(Efins, Einfs)][0] #plot binding energy in abs units (hw)
            if logplot == 3:
                b = '-dE'
                ylist = -ylist
        else:
            #Einf = np.array([-(1-n)*U for n,U in zip(df['eta'].values,df['U'].values)])
            #df['dE'] = (df['E'].values-Einf)/np.abs(Einf)
            val, fixed_x, fixed_ys = ps.FindArrs(df, colnames, fixedqty, val)
            ylist = fixed_ys[0]
            
            if len(filenames) > 1:
                for i,name in enumerate(filenames[1:]):
                    df2 = pd.read_csv(name)
                    _,fixed_x2, fixed_y2 = ps.FindArrs(df2, colnames, fixedqty, val)
                    ylist2 = fixed_y2[0]
                    if len(xlim) > 0:
                        idx = np.where((fixed_x2 >= xlim[0]-1) & (fixed_x2 <= xlim[1]))[0]
                        if len(idx) > 0:
                            fixed_x2 = fixed_x2[idx]
                            ylist2 = ylist2[idx]
                    #ax.plot(fixed_x2, ylist2,label= '$' + b + '$, $' + fixedqty + ' = $%.3f' %val)
        if a == 'eta': ax.axvline(0.00158, c='red')
        if fixedqty == 'eta': fixedqty = '\\' + fixedqty

    if logplot == 3:
        #print(list(zip(np.log10(fixed_x),ylist)))
        if b == 'dE': 
            if val > 0:
                b = '|\Delta E|'
                ylist = np.abs(ylist)
            else: #eta = 0
                b = '|\Delta E|'
                ylist = np.abs(ylist)
                #b = '-\Delta E'
                #ylist = -ylist
        elif b == 'E':
            ylist = -ylist
            if len(filenames) > 1: 
                ylist2 = -ylist2
            b = '-E'
        if a == 's':
            a = '\\tilde \sigma'
        logx = np.log(fixed_x)
        logy = np.log(ylist)
        if len(filenames) > 1:
            logx2 = np.log(fixed_x2)
            logy2 = np.log(ylist2)
        if len(labels) > 0:
            ax.plot(logx, logy, label=labels[0])
            if len(filenames) > 1: ax.plot(logx2, logy2, label=labels[1])
        else:
            ax.plot(logx, logy, label= '$' + b + '$, $' + fixedqty + ' = $%.2f' %val)
        if fit == True:
            def fitweak(x,A,B):
                f = A + B*x
                return f

            bnds_w = ([-50,-5],[10,5]) # #bounds for weak coupling fit
            guess_w =[0,0]
            if len(yfit) == 2:
                idx = np.where((logy <= yfit[1]) & (logy >= yfit[0]))[0] #for eta = 0
            elif len(xfit) == 2:
                idx = np.where((logx <= xfit[1]) & (logx >= xfit[0]))[0] #for eta = 0
            else:
                idx = np.where((logy <= -2) & (logy >= -10))[0] #for eta = 0

            coeffs, p_cov_w = curve_fit(fitweak,logx[idx], logy[idx], p0=guess_w,bounds=bnds_w)
            print(coeffs)
            print(np.sqrt(np.diag(p_cov_w))) #standard deviation of the parameters in the fit
            ans_w = fitweak(logx[idx],coeffs[0],coeffs[1])
            ax.plot(logx[idx],ans_w,color='red',label='fit')
            if len(xlim) == 1:
                ax.set_xlim(left=xlim[0])

            textstr = '\n'.join((
                r'$\log (%s) = A + B \log (%s)$' % (b,a,),
                r'$A=%.2f$' % (coeffs[0], ),
                r'$B=%.2f$' % (coeffs[1], )
                ))

            ax.text(0.05, 0.25, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top')

        ax.set_ylabel('$\ln (' + b + ')$')
        if a == 'eta': a = "\\" + a
        elif a == 's': a = "\sigma"
        ax.set_xlabel('$\ln(' + a + ')$')
        ax.legend()
    else:
        ax.plot(fixed_x, ylist,label= '$' + b + '$, $' + fixedqty + ' = $%.3f' %val)
        if len(filenames) > 1:
           ax.plot(fixed_x2, ylist2,label= '$' + b + '$, $' + fixedqty + ' = $%.3f' %val)
        if logplot == 1:
            ax.semilogx()
        elif logplot == 2:
            ax.semilogy()
        ax.set_ylabel('$' + b + '$')
        if a == 'eta': a = "\\" + a
        elif a == 's': a = "\sigma"
        ax.set_xlabel('$' + a + '$')
        ax.legend()
    if len(xlim) > 0:
        if logplot == 3:
            ax.set_xlim(np.log(xlim[0]),np.log(xlim[1]))
        else: ax.set_xlim(xlim[0],xlim[1])
        

    plt.tight_layout()
    plt.show()

def E_binding(filename1, filename2, colnames,xlims=(),ylims=(),zlims=(),save=False, point=False, logplot='', realval = False, fig=None, ax=None, scinot = False, show = True):
    '''
    Plot binding energy for phase diagram
    inputs:
        filename1: E_bip file
        filename2: E_inf file
        save: whether to save plot as a file
        colnames: array of [xname, yname, zname]
    '''
    #phase diagram!
    #if have a preexisting figure/axis I want to plot on, use it
    if fig == None or ax == None:
        fig = plt.figure(figsize=(6,4.5))
        ax = fig.add_subplot(111)
    df = pd.read_csv(filename1)
    if 'Ebind' in df.columns: 
        print('harrumph')
        df['dE'] = df['Ebind']
        colnames[-1] = 'dE'
    elif (filename1 != filename2) or ('dE' not in df.columns): #and dE not in df col headers - need to check
        df2 = pd.read_csv(filename2)
        if realval == False: #binding energies relative to the single polaron energy
            dEs = np.array([(E-Einf)/np.abs(Einf) for E,Einf in zip(df['E'].values, df2['E'].values)])
        else:  #absolute energy differences (in units of hw)
            dEs = np.array([(E-Einf) for E,Einf in zip(df['E'].values, df2['E'].values)])
        df['dE'] = dEs
        colnames[-1] = 'dE'

    #else dE is already a column name

    ns, Us, Zbind = ps.parse_CSV(df,colnames)
    a,b,c = colnames
    print(Zbind.min())

    if len(zlims)>0: 
        zmin, zmax = zlims
    else: 
        zmin = Zbind.min()
        zmax = 0.
    
    #set limits on x and y axes if argument given
    if len(xlims) == 1: ax.set_xlim(xlims[0], ns.max())
    elif len(xlims) == 2: ax.set_xlim(xlims[0],xlims[1])

    if len(ylims) == 1: ax.set_ylim(ylims[0], Us.max())
    elif len(ylims) == 2: ax.set_ylim(ylims[0],ylims[1])
     
    idx = np.where(Zbind < -1E-10)[0]
    Zbind = Zbind[idx]
    ns = ns[idx]
    Us = Us[idx]
    
    if logplot == 'y':
        ax.semilogy()
    if logplot == 'x':
        ax.semilogx()
    if logplot == 'z':
        Zbind = -Zbind
        cp = ax.contourf(ns, Us, Zbind, levels = MaxNLocator(nbins=20).tick_values(-zmin,1.), norm=LogNorm()) #put color bar on log scale
    else:
        cp = ax.contourf(ns, Us, Zbind, levels = MaxNLocator(nbins=20).tick_values(zmin, zmax))

    #Plot various material param values
    if point == True: 
        #formatted to show up correctly in .eps file format; .png will look a bit off
        ax.plot(eta_STO,U_STO,color='black',marker='.') #STO, strontium titanate
        ax.annotate('STO', (eta_STO, U_STO), xytext=(6, -6), textcoords='offset pixels')
        ax.plot(etaKTO,UKTO,color='black',marker='.') #KTO, potassium tantalate
        ax.annotate('KTO', (etaKTO, UKTO), xytext=(6, -6), textcoords='offset pixels')
        ax.plot(9.05E-2, 2.6 ,color='black',marker='.') #PbS Lead sulfide, alloy
        ax.annotate('PbS', (9.05E-2, 2.6), xytext=(-18, 6), textcoords='offset pixels')
        ax.plot(8.18E-2, 2.58,color='black',marker='.') #PbSe Lead selenide
        ax.annotate('PbSe', (8.18E-2, 2.58), xytext=(-28, 5), textcoords='offset pixels')
        ax.plot(8.26E-2, 1.87,color='black',marker='.') #PbTe Lead telluride
        ax.annotate('PbTe', (8.26E-2, 1.87), xytext=(4, -9), textcoords='offset pixels')
        ax.plot(3.75E-2, 3.12,color='black',marker='.') #SnTe Tin telluride
        ax.annotate('SnTe', (3.75E-2, 3.12), xytext=(6, 0), textcoords='offset pixels')
        ax.plot(8E-2, 0.47,color='black',marker='.') #GeTe Germanium telluride
        ax.annotate('GeTe', (8E-2, 0.47), xytext=(-27,-10), textcoords='offset pixels')

    #plot binding energy = 0 contour
    #zerocont = ax.contour(ns, Us, Zbind, [0.], colors=('r',), linewidths=(1,), origin='lower') #plot dE = 0 curve
    #e0,u0 = ps.FindCoords(zerocont)
    #for elist, ulist in zip(e0,u0):
    #    print([(1-et)*yu/2 if yu < 20. else 0. for et,yu in zip(elist,ulist)]) #print alpha values for the bipolaron binding boundary region
    '''
    #plot weak-to-strong transition curve (i.e. where a = 1 --> a < 1)
    yoos = np.unique(df[b].values)
    etas = np.zeros(len(yoos))
    for i,u in enumerate(yoos):
        #for each unique value of U (or eta), find the maximum a != 1 and corresponding index
        aas = df[df[b] == u]['a'].values
        aidx = np.where(aas < 1)[0]
        ayes = aas[aidx]
        if (len(ayes) > 0) & (len(ayes) < len(aas)):
            dE = df[df[b] == u]['dE'].values[aidx]
            print(u)
            idx = np.where(dE < 0)[0][-1] #pick last value where dE < 0 (this is right before it switches to a=1, dE > 0
            print(idx)
            #idx = np.where(ayes == ayes.max())[0]
            etas[i] = df[df[b] == u][a].values[idx]
            print(etas[i])
        else: continue
    #now get rid of all the redundant eta = 0 values 
    idx = np.where((etas > 0) & (etas < 0.08))[0]
    aplt = ax.plot(etas[idx], yoos[idx], color='black') 
    '''
    if scinot == True:
        cbar=fig.colorbar(cp, ax=ax, format='%.2e') # Add a colorbar to a plot; scientific notation
    else:
        cbar=fig.colorbar(cp, ax=ax) # Add a colorbar to a plot

    if realval == False:
        cbar.ax.set_ylabel('$\Delta E/|E_\infty|$')
    else: 
        cbar.ax.set_ylabel('$\Delta E$')

    ax.set_xlabel('$\\' + a + '$')
    ax.set_ylabel('$' + b + '$')

    if show == True:
        plt.tight_layout()
        print(ps.format_filename(filename1))
        if save == True:
            plt.savefig(ps.SavePath(filename1,".png"))
        plt.show()

##########################################################################################################################
'''Check E_inf integrand plot (w/ approximations)'''
def Check_Einf_Integrand(y,s,a):
    fig = plt.figure(figsize=(6,4.5))
    ax = fig.add_subplot(111)
    zs = np.linspace(1E-3,10,300)
    vals = [nag.Integrand_ln(z, y, s,1) for z in zs]
    vals2 = [nag.zIntegrand_a1_yfin(z,y,s,2) for z in zs]

    ax.plot(zs,vals,'.', label='double')
    ax.plot(zs, vals2, '.', label='single')
    ax.legend(loc=1)
    plt.show()

###########################################################################################################################

def PoolParty(csvname):
    '''run multiprocessing to generate energy CSV for range of alpha vals'''
    #zoom in on weak-coupling regime
    ns=np.linspace(0,0.095,50)
    Us = np.geomspace(1E-3,50,200)
    #Us = np.geomspace(1E-5,15,70)

    #For E vs alpha plots
    #ns = [0.,0.05]
    #Us = np.linspace(1E-3,40,80)
    #Us = np.linspace(1E-3,20,80) #strong coupling U
    #Us = np.linspace(15,40,80) #weak coupling U
    #Us = np.geomspace(1E-20,1,200) #study small U behavior

    #Phase diagram capturing edge of horn
    #ns=np.linspace(0,0.08,60)
    #Us = np.linspace(10,50,200) #originally lower bound was 0.001

    #spherically symmetric wfn
    #ns=np.linspace(0,0.08,30)
    #Us = np.linspace(10,50,50) #originally lower bound was 0.001
    
    #STO, KTO, PbS, PbSe, PbTe, SnTe, GeTe
    #ns = [eta_STO, etaKTO, 9.05E-2, 8.18E-2, 8.26E-2, 3.75E-2, 8E-2]
    #Us = [U_STO, UKTO, 2.6, 2.58, 1.87, 3.12, 0.47] 
    
    a=1.
    z_c = 10.
    y = 10.
    a_c = 0.6 #dividing a value

    df={}
    #quantities = ['eta','U','a','s','y','E']
    quantities = ['eta','U','a','s','y','E', 'ainf', 'sinf', 'yinf', 'Einf', 'dE']
    #quantities = ['eta','U','Omega','Omega1','E','Einf','Ebinding'] #Devreese

    for i in quantities:
        df[i]=[]

    tic = time.perf_counter()
    with multiprocessing.Pool(processes=4) as pool:
        #bipolaron runs for y->inf
        job_args = [(n,u,z_c,a_c,y,20.) for n,u in product(ns,Us)] #for phase diagram
        #job_args = [(n,u,z_c,a_c,y) for n,u in zip(ns,Us)] #for materials ONLY

        #results = pool.map(nag.min_sym_E, job_args) #radially symmetric wave function
        #results = pool.map(nag.min_sym_E_inf, job_args) #radially symmetric wave function
        #results = pool.map(nag.min_E_avar_inf, job_args) #gives same answer as just fixing a at a=0 or a=1
        results = pool.map(nag.min_E_nak, job_args)
        #results = pool.map(nag.min_E_nak_strong, job_args)
        #results = pool.map(nag.min_E_nak_weak, job_args)

        #bipolaron run for finite y
        #results = pool.map(nag.min_E_bip_ln2, job_args) #bipolaron min energy
        #results = pool.map(nag.min_E_bip_strong, job_args) #strong coupling result
        #results = pool.map(nag.min_E_bip_weak, job_args) #strong coupling result
        #results = pool.map(nag.min_E_bip_asfix, job_args) #weak coupling result

        #polaron run
        #job_args = [(n,u, 1) for n,u in product(ns,Us)]
        #results = pool.map(min_hybpol, job_args)

        #Devreese
        #results = pool.map(nag.min_E_dev, job_args)

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

def FFT(a=1,y=1,s=1,opt='gaus',ext='.png'):
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
    
    rmax = 1.
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
    
    if opt !='gaus':
        #fig2 = plt.figure(figsize=(5,5))
        #axe = fig.add_subplot(111)
        fig, ax = plt.subplots()
        C = ax.contourf(rs, rs, cent_ifft_yz.real, levels = MaxNLocator(nbins=20).tick_values(8.94, ifft_yz.max())) #set min to 2.76 for rmax = 1.8
        #remove axis frame and render background transparent
        fig.patch.set_visible(False)
        ax.axis('off')
        plt.savefig("nak_contyz_y" + str(y) + ext)
        plt.show()
        #do same for xy - first clear the contour from the plot
        plt.contourf(rs, rs, cent_ifft_xy.real, levels = MaxNLocator(nbins=20).tick_values(8.94, ifft_xy.max()))
        plt.gcf().patch.set_visible(False)
        plt.gca().axis('off')
        plt.savefig("nak_contxy_y" + str(y) + ext)
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

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cbook import get_sample_data
import matplotlib.image as mpimg
def DensityPlot3D(a=1,y=1,s=1):
    '''Plot electron density rho(r) for nakano - plots the density rho just fine but can't seem to figure out how to visualize 2D and 3D plots together (i.e. force the contour plot to stay 2D and/or project images of them onto the boundary box xy and yz-planes. 
        Workaround: produce .eps contour plots in Python, then read them into Mathematica to tidy up the visualization/graphing since Mathematica has better support for 3D graphics rendering'''
    def rho(rx,ry,rz,s,y):
        return 1./(s**3 * (1 + np.exp(-y**2/2))) * np.exp(-(rx**2 + ry**2))* (np.exp(-(rz - y/2)**2) + np.exp(-(rz + y/2)**2) + 2*np.exp(-(rz**2 + y**2/2)))
    
    def nak_f_k(kx,ky,kz,a,y,s,alf):
        '''Phonon displacement function f(kx,ky,kz) for the Nakano calc'''
        numer = np.exp(-0.25* (1-a)**2 * (kx**2+ky**2+kz**2)) * (np.cos(0.5*(1-a)* y*kz) + np.exp(-y**2/2) ) + np.exp(-0.25*(1+a**2)*(kx**2+ky**2+kz**2)) * (np.cos(0.5* (1+a)* y*kz)+ np.exp(-y**2/2) )
        denom = 1 + (a/s)**2* (kx**2+ky**2+kz**2) + np.exp(-a**2 * (kx**2+ky**2+kz**2)/2) / (1+ np.exp(-y**2/2) ) * (np.cos(a*y*kz)+ np.exp(-y**2/2) )
        return s* np.sqrt(4*np.pi*alf)/(1+np.exp(-y**2/2)) * numer/denom * 1/ np.sqrt(kx**2+ky**2+kz**2) #setting V/l^3 = 1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    rmax = 3
    dr = 0.1
    rs = np.arange(-rmax,rmax, dr) #shift away from 0
    ndivs = (2*rmax/dr)*1j
    X, Y, Z = np.mgrid[-rmax:rmax:ndivs, -rmax:rmax:ndivs, -rmax:rmax:ndivs]
    
    dens = rho(X, Y, Z,s,y)
    iso_val=0.5
    verts, faces, _, _ = measure.marching_cubes_lewiner(dens, iso_val, spacing=(0.1, 0.1, 0.1))

    ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], cmap='Spectral', lw=1)
    
    '''
    arr = mpimg.imread("./nak_contyz_y10.png")
    # 10 is equal length of x and y axises of your surface
    stepX, stepY = 10. / arr.shape[0], 10. / arr.shape[1]

    X1 = np.arange(-rmax, rmax, stepX)
    Y1 = np.arange(-rmax, rmax, stepY)
    X1, Y1 = np.meshgrid(X1, Y1)
    # stride args allows to determine image quality
    # stride = 1 work slow
    ax.plot_surface(X1, Y1, np.full(X1.shape,-2.01), rstride=1, cstride=1, facecolors=arr)
    '''
    plt.show()

###############################################################################################################################

if __name__ == '__main__':
    import pandas as pd
    csvname = "./data/nakano_yinf_U60_eta0-1.csv" #minimizing between a=0 and a=1
    csvname1b = "./data/nakano_yfin_U60_eta0-1.csv" #eta extends up to 0.1
    csvname1c = "./data/nakano_yfin_U50.csv" #eta goes to 0.08
    csvname1d = "./data/nakano_yinf_U50.csv"
    csvname1e = "./data/nakano_yinf_U15_eta0-1.csv" #eta goes to 0.1 ish, zoom in on weak-binding region
    csvname1f = "./data/nakano_yfin_U15_eta0-1.csv"
    csvname1g = "./data/nakano_yfin_wkzoom_s20.csv" #zoom in on weak coupling phase space, with sigma set to sig = exp(20) - FOR PHASE DIAGRAM GENERATION
    csvname1h = "./data/nakano_yinf_wkzoom_s20.csv"
    csvname1j = "./data/nakano_yfin_wkzoom_s40.csv" #zoom in on weak coupling phase space, with sigma set to sig = exp(20)
    csvname1k = "./data/nakano_yinf_wkzoom_s40.csv"
    csvname1l = "./data/nakano_fullPD.csv"
    csvname1n = "./data/nakano_fullPD_y10.csv"
    csvname1m = "./data/nakano_zoomPD.csv"
    
    csvname2 = "./data/nakano_yinf_logU_v2_avar.csv"
    csvname3 = "./data/nakano_U40.csv"
    csvname3c = './data/nakano_U20_str.csv' #strong coupling soln only
    csvname3d = './data/nakano_U40_wk.csv' #strong coupling soln only
    csvname4 = './data/devreese1.csv' #strong coupling soln only
    csvname_mat = './data/nak_mats.csv' #optimized params for various materials where bipolarons might be found -- this particular run (using PoolParty) lets sigma optimize freely (so corresponds to s=148). Need to generate data for s = 8?
    csvname_mat_inf = './data/nak_mats_inf.csv' #optimized params (y->inf) for various materials where bipolarons might be found
    csvname_su_fin2 = './data/nak_smallU_yfin.csv'
    csvname_su_inf2 = './data/nak_smallU_yinf.csv'
    csvname_su_fin3 = './data/nak_smallU_yfin_s10.csv'
    csvname_su_inf3 = './data/nak_smallU_yinf_s10.csv'
    csvname_su2 = "./data/nak_smallU_s25_U-20.csv" #zoom in on weak coupling phase space, with sigma set to sig = exp(20) for E vs U loglog plot at fixed eta = 0, 0.05
    csvname_su = "./data/nak_smallU_s30_U-20.csv"
    csvname_sym = "./data/nak_symwfn.csv"
    csvname_sym_inf = "./data/nak_symwfn_inf.csv"

    #FFT(a=0.1,y=0,s=1,opt='nak',ext='.eps')
    #f_r_special()
    #DensityPlot3D()

    #PoolParty(csvname1n)
    #E_binding('./data/nakano_wkPD_s12_loglog.csv', './data/nakano_wkPD_s12_loglog.csv', colnames=['eta','U','dE'], realval = False, point=True)

    #GenE_vs_a()
    #Plot_E_vs_a("./data/testnak.csv",xvar='s',logplot=0, plotdE=True)
    #plotContour(csvname1b, colnames=['eta','U','E'],zlims=(), save=False, minmax=5,logplot=0)
    #FormatPhaseDia(csvname1g, csvname1h)
    
    #plot loglog plots to look at large-sigma behavior
    #PlotAtFixedVal(["./data/testnak.csv"], colnames=['s','dE'], fixedqty='eta', fixedvals=[0.01], logplot=3)
    #PlotAtFixedVal(["./data/testnak.csv"], colnames=['s','dE'], fixedqty='eta', fixedvals=[0.], logplot=3, fit=True, yfit=(-14,-4))
    #PlotAtFixedVal(["./data/nak_E(s)_sub.csv"], colnames=['s','E'], fixedqty='eta', fixedvals=[0.], logplot=3, fit=True, xfit=(3,7))
    #PlotAtFixedVal(["./data/nak_E(s)_sub.csv"], colnames=['s','dE'], fixedqty='eta', fixedvals=[0.], logplot=3, fit=True, xfit=(3,7))

    #PlotAtFixedVal(["./data/nak_E(s)_sub.csv"], colnames=['s','E'], fixedqty='eta', fixedvals=[0.01], logplot=1, fit=True, xfit=(4,8))
    #PlotAtFixedVal(["./data/nak_E(s)_sub.csv"], colnames=['s','Einf'], fixedqty='eta', fixedvals=[0.01], logplot=1, fit=True, xfit=(4,8))
    #PlotAtFixedVal(["./data/nak_E(s)_sub.csv"], colnames=['s','dE'], fixedqty='eta', fixedvals=[0.01], logplot=3, fit=True, xfit=(4,8))

    #PlotAtFixedVal(["./data/nak_E(y).csv"], colnames=['y','E'], fixedqty='eta', fixedvals=[0.], logplot=1, fit=False, xfit=(3,7))
    #PlotAtFixedVal(["./data/nak_E(y)_sub.csv"], colnames=['y','E'], fixedqty='eta', fixedvals=[0.], logplot=1, fit=False, xfit=(3,7))
    #eta_c_vs_U(generate=False, fit='ex', yfit=(-1.4,-1.1)) 

    #FindIntegral()

    '''PAPER PLOTS'''
    #E_binding(csvname1l, csvname1l, colnames=['eta','U','dE'], realval = False,point=True) #Fig 1, full phase diagram 

    #PlotE(csvname3, fit=False, opt='', multiplot=True) #use to create multiplot phase diagram, Fig 3
    #PlotE('./data/nakano_yfin_U40.csv', fit=False, opt='fin', multiplot=False) #plot energy comparison between E_opt, E_inf, Devreese, Fig 5
    #PlotAtFixedVal([csvname_sym, "./data/gauss_U40.csv"], colnames=['U','E'], fixedqty='eta', fixedvals=[0.],xlim=(10,40),labels=['sym','gaus'],logplot=3) #plot symmetric wfn results against Gaussian (a=0) result to show that symmetry breaking wfn super beats out a symmetric solution, Fig. 6
    #FormatWeakPD(generate=True,loglog=True, findconst=False,alt=True) #weak coupling phase diagram with lines of constant sigma; Figs. 1b, 9
    #LgSigma(False) #plot behavior of binding energy as a function of sigma - show attraction for eta = 0 (complete cancellation of Coulomb), incomplete cancellation for eta > 0 (dies as 1/sigma), Fig. 7
    #GenE_vs_eta_y_fixedU(False) #Fig 4
    #PoolParty(csvname1n)
    #E_binding(csvname1n, csvname1n, colnames=['eta','U','dE'], realval = False, point=True) 

    #GenE_vs_a()
    #PlotAtFixedVal(["./data/nak_E(s).csv"], colnames=['s','dE'], fixedqty='eta', fixedvals=[0.01], logplot=3,fit=True, xfit=(8,14))
    #PlotAtFixedVal(["./data/nak_E(s).csv"], colnames=['s','dE'], fixedqty='eta', fixedvals=[0.], logplot=3, fit=True, xfit=(10,15))

    #Find material parameters
    FindMatParams()

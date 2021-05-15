import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator
import sys
import os
from scipy.interpolate import interp1d
import scipy.ndimage
from scipy.special import erfc

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
U_STO = elec**2/(epsinf*hbar)*np.sqrt(2*m/(hbar*w))*1E-9 #convert statC in e^2 term into J to make U dimensionless
alpha = (elec**2*1E-9)/hbar*np.sqrt(m/(2*hbar*w))*1/epsinf*(1 - epsinf/epssr) #convert statC to J*m
l = np.sqrt(hbar/(2*m*w))

#KTO vals
epsinfKTO = 4.6; eps0KTO = 3800; wKTO = 299792458*826*100; UKTO = elec**2/(epsinfKTO*hbar)*np.sqrt(2*m/(hbar*wKTO))*1E-9; etaKTO = epsinfKTO/eps0KTO

def add_d(filename):
    df = pd.read_csv(filename)
    whys = df['y_opt'].values #array of ys
    sls = df['s_opt'].values #array of sigma/ls
    ds = np.array([y*s*l if y*s*l >= 1E-11 else 0 for s,y in zip(sls,whys)]) #elec separation distances - dimensionalized; setting cutoff boundary between 1E-11 and 1E-15 doesn't qualitatively change the shape of the boundary
    #ds = np.array([y*s*l for s,y in zip(sls,whys)]) #elec separation distances - dimensionalized; setting cutoff boundary between 1E-11 and 1E-15 doesn't qualitatively change the shape of the boundary
    df['d'] = ds

    ayes = df['a_inf'].values
    sinf = df['s_inf'].values
    alphas = [(1-n)*u/2 for n,u in zip(df['eta'].values,df['U'].values)]
    df['alpha'] = alphas
    Ayes = [(1-a/2)**2 + (a/2)**2 for a in ayes]
    E_ana = np.zeros(len(ayes))
    for i,s in enumerate(sinf):
        A = (1-ayes[i]/2)**2 + (ayes[i]/2)**2
        b = A*s**2/ayes[i]**2
        val2 = np.exp(b)
        if np.isinf(val2): E_ana[i] = 3/s**2
        else:
            #if val >1E-300: val2 = np.exp(b)
            E_ana[i] = 3/(s**2) - alphas[i]*np.sqrt(2)/ayes[i]* erfc(np.sqrt(b))*val2
    df['Einf_ana'] = E_ana
    return df

def parse_CSV(df_orig,colnames):
    '''
    inputs:
        filename: file from which to read into Pandas dataframe
        colnames: array of names of 3 columns (x,y,z) to plot
    outputs:
        arrays of x,y,z formatted for contour plotting
    '''
    #read in CSV as Pandas dataframe
    #df_orig = pd.read_csv(filename)
    df = df_orig[colnames]
    a,b,c = colnames
    #print(df)
    orgdf = df.groupby([a,b]).mean() #group by eta and U values; take the mean since there's only one of each combo anyways so it doesn't matter
    odf_reset = orgdf.reset_index()
    odf_reset.columns = colnames
    odf_pivot = odf_reset.pivot(a,b)
    pd.set_option("display.max.columns", None)
    #print(odf_pivot.columns.levels)
    #print(odf_pivot.index)
    #print(odf_pivot)
    Y = odf_pivot.columns.levels[1].values
    X = odf_pivot.index.values
    Z = odf_pivot.values.transpose()
    Xi,Yi = np.meshgrid(X, Y)
    return Xi, Yi, Z

def format_filename(path, param_val=''):
    filename = os.path.split(path)[1] #get tail of path
    head = filename.split(".")[0]
    addon_str = str(param_val).split(".")
    for i,string in enumerate(addon_str):
        head += addon_str[i]
        if i < len(addon_str)-1: filename += "-"
    return head

def SavePath(filename,suffix):
    my_path = os.path.abspath(__file__)
    #print(my_path)
    #get rid of tail
    my_path = os.path.split(my_path)[0]
    #print(my_path)
    saveas = os.path.join(my_path, 'plots' + os.sep, format_filename(filename) + suffix)
    print(saveas)
    return saveas

def FindArrs(df, colnames, fixedqty, fixedval):
    '''Find arrays of parameters at some fixed value of some other quantity (here shown for eta)'''
    #df = pd.read_csv(filename)
    values = [df[name].values for name in colnames]
    etas = df[fixedqty].values 

    #find the nearest index/value of eta in the etas array
    idx = (np.abs(etas-fixedval)).argmin()
    print("Requested: " + fixedqty + " = " + str(fixedval) + "\tfound: " + str(etas[idx]))
    idxs = np.where(etas==etas[idx])
    fixed_vals = [arr[idxs] for arr in values]
    return etas[idx],fixed_vals[0], fixed_vals[1:]

from scipy.optimize import curve_fit
def fitweak(x,a,b):
    f = a*x + b*x**2 
    return f
def fitstrong(x,a,b):
    f = a*x**2 +b
    return f
    
def FitPolaronData(ax,ax2):
    #read in CSV as Pandas dataframe
    df = pd.read_csv("hyb_polaron_2.csv")
    alphas = df["alpha"].values
    Elist = df["E_opt"].values
    idx = np.where(alphas <= 0.5)[0]
    
    #separate lists into "weak coupling" (alpha < 5) and "strong coupling" (alpha > 5) - this is suggested by Feynman
    E_wk = Elist[idx]*2 #multiply by 2 to get 2 polaron energy
    al_wk = alphas[idx]
    idx2 = np.where((alphas > 5) & (alphas <=10))
    #print(idx2)
    E_str = Elist[idx2]*2
    al_str = alphas[idx2]
    #print(al_str)

    bnds_w = ([-10,-10],[5,5]) #bounds for weak coupling fit
    guess_w =[-1,-3]
    param_w, p_cov_w = curve_fit(fitweak,al_wk, E_wk, p0=guess_w,bounds=bnds_w)
    print(param_w)
    print(np.sqrt(np.diag(p_cov_w))) #standard deviation of the parameters in the fit
    a,b = param_w
    
    ax.plot(al_wk, E_wk,label='2pol data')
    #ax.set_xlabel('$\\alpha$')
    #ax.set_ylabel('$E_{opt}$')
    ans_w = np.array([fitweak(al,a,b) for al in al_wk])
    ax.plot(al_wk,ans_w,color='green',label='2pol fit')
    
    textstr = '\n'.join((
        r'$E_{pol}(\alpha) = a\alpha + b\alpha^2$',
        r'$a=%.2f$' % (a, ),
        r'$b=%.2f$' % (b, )
        ))

    ax.text(0.05, 0.65, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top')

    ax.legend(loc=1)
        
    bnds_s = ([-10,-10],[5,5]) #bounds for weak coupling fit
    guess_s =[-1,-3]
    param_s, p_cov_s = curve_fit(fitstrong,al_str, E_str, p0=guess_s,bounds=bnds_s)
    print(param_s)
    print(np.sqrt(np.diag(p_cov_s))) #standard deviation of the parameters in the fit
    c,d = param_s
  
    #ax2 = fig.add_subplot(122)
    ax2.plot(al_str, E_str,label='2pol data')
    #ax2.set_xlabel('$\\alpha$')
    #ax2.set_ylabel('$E_{opt}$')
    ans_s = np.array([fitstrong(al,-.2,-2.5) for al in al_str]) #c = -0.1, d=-1.75 for single polaron, c=-0.2,d=2.5 for 2*polaron
    ax2.plot(al_str,ans_s,color='green',label='2pol fit')
    
    textstr = '\n'.join((
        r'$E_{pol}(\alpha) = c\alpha^2 + d$',
        r'$c=%.2f$' % (-0.2, ),
        r'$d=%.2f$' % (-2.5, )
        ))

    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=12,
        verticalalignment='top')

    ax2.legend(loc=1)

def Plot_vs_Alpha(filename, opt=1,save=False):
    '''
    Checking energy relationship to alpha to possible clarify some things about weak/strong coupling. 
    '''
    #first, extract data of U at fixed eta ~ 0
    tryeta = 0.05
    df = add_d(filename)
    Enames = ['U','a_opt','E_opt','E_inf']
    eta, Us, Es = FindArrs(df, Enames,'eta',tryeta)
    avals = Es[0]
    Es = Es[1:]

    alphas = (1-eta)*Us/2
    idx = np.where(alphas <= 0.5)[0]
    idx2 = np.where((alphas > 5) & (alphas <=10))
        
    #separate lists into "weak coupling" (alpha < 5) and "strong coupling" (alpha > 5) - this is suggested by Feynman
    a_wk = avals[idx]
    al_wk = alphas[idx]
    al_str = alphas[idx2]
    #print(al_str)

    i = 0 #plot counter
    fig = plt.figure(figsize=(10,9))
    if opt == 1:
        Elist = np.minimum(Es[0],Es[1]) #element-wise minimum between E_inf and E_opt; my algorithm is such that for small alpha ("weak coupling), E_opt hits a box instead of converging to E_inf. Hence the "true data" I should be plotting should be the minima between these two
        i += 1

        #print(a_wk)
        E_wk = Elist[idx]
        E_str = Elist[idx2]

        bnds_w = ([-10,-10],[5,5]) #bounds for weak coupling fit
        guess_w =[-1,-3]
        param_w, p_cov_w = curve_fit(fitweak,al_wk, E_wk, p0=guess_w,bounds=bnds_w)
        print(param_w)
        print(np.sqrt(np.diag(p_cov_w))) #standard deviation of the parameters in the fit
        a,b = param_w
        ax = fig.add_subplot(220 + i)
        ax.plot(al_wk, E_wk,label='data')
        ax.set_xlabel('$\\alpha$')
        ax.set_ylabel('$\min(E_{opt}, E_\infty)$')
        ans_w = np.array([fitweak(al,a,b) for al in al_wk])
        ax.plot(al_wk,ans_w,color='red',label='fit')
    
        textstr = '\n'.join((
            r'$\eta=%.2f$' % (eta),
            r'$E_{bi}(\alpha) = a\alpha + b\alpha^2$',
            r'$a=%.2f$' % (a, ),
            r'$b=%.2f$' % (b, )
            ))

        ax.text(0.05, 0.3, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top')
        ax.legend(loc=1)
        
        if len(E_str) > 0:
            bnds_s = ([-10,-10],[5,5]) #bounds for weak coupling fit
            guess_s =[-1,-3]
            param_s, p_cov_s = curve_fit(fitstrong,al_str, E_str, p0=guess_s,bounds=bnds_s)
            print(param_s)
            print(np.sqrt(np.diag(p_cov_s))) #standard deviation of the parameters in the fit
            c,d = param_s
            
            i+= 1
            ax2 = fig.add_subplot(220 + i)
            ax2.plot(al_str, E_str,label='data')
            ax2.set_xlabel('$\\alpha$')
            ax2.set_ylabel('$\min(E_{opt}, E_\infty)$')
            ans_s = np.array([fitstrong(al,c,d) for al in al_str])
            ax2.plot(al_str,ans_s,color='red',label='fit')
            textstr = '\n'.join((
                r'$\eta=%.2f$' % (eta),
                r'$E_{bi}(\alpha) = c\alpha^2 + d$',
                r'$c=%.2f$' % (c, ),
                r'$d=%.2f$' % (d, )
                ))

            ax2.text(0.05, 0.7, textstr, transform=ax2.transAxes, fontsize=12,
                verticalalignment='top')
            
            FitPolaronData(ax,ax2)
            
            ax2.legend(loc=1)
    
    if opt == 2: 
        for name, Elist in zip(Enames[2:],Es):
            i += 1

            E_wk = Elist[idx]
            E_str = Elist[idx2]
            bnds_w = ([-10,-10],[5,5]) #bounds for weak coupling fit
            guess_w =[-5,-3]
            param_w, p_cov_w = curve_fit(fitweak,al_wk, E_wk, p0=guess_w,bounds=bnds_w)
            print(param_w)
            print(np.sqrt(np.diag(p_cov_w))) #standard deviation of the parameters in the fit
            a,b = param_w
            pre,post = name.split('_')
            if post == 'inf': post = '\infty'
            ax = fig.add_subplot(220 + i)
            ax.plot(al_wk, E_wk,label='data')
            ax.set_xlabel('$\\alpha$')
            ax.set_ylabel('$E_{' + post + '}$')
            ans_w = np.array([fitweak(al,a,b) for al in al_wk])
            ax.plot(al_wk,ans_w,color='red',label='fit')
    
            textstr = '\n'.join((
                r'$\eta=%.2f$' % (eta),
                r'$y(\alpha) = a\alpha + b\alpha^2$',
                r'$a=%.2f$' % (a, ),
                r'$b=%.2f$' % (b, )
                ))

            ax.text(0.05, 0.45, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top')

            ax.legend(loc=1)

            if len(E_str) == 0: continue
            bnds_s = ([-10,-10],[5,5]) #bounds for weak coupling fit
            guess_s =[-1,-3]
            param_s, p_cov_s = curve_fit(fitstrong,al_str, E_str, p0=guess_s,bounds=bnds_s)
            print(param_s)
            print(np.sqrt(np.diag(p_cov_s))) #standard deviation of the parameters in the fit
            c,d = param_s
            
            i+= 1
            ax2 = fig.add_subplot(220 + i)
            ax2.plot(al_str, E_str,label='data')
            ax2.set_xlabel('$\\alpha$')
            ax2.set_ylabel('$E_{' + post + '}$')
            ans_s = np.array([fitstrong(al,c,d) for al in al_str])
            ax2.plot(al_str,ans_s,color='red',label='fit')
    
            textstr = '\n'.join((
                r'$\eta=%.2f$' % (eta),
                r'$y(\alpha) = c\alpha^2 + d$',
                r'$c=%.2f$' % (c, ),
                r'$d=%.2f$' % (d, )
                ))

            ax2.text(0.05, 0.45, textstr, transform=ax2.transAxes, fontsize=14,
                verticalalignment='top')

            ax2.legend(loc=1)
            if i==4:
                FitPolaronData(ax,ax2)

    plt.tight_layout()
    if save == True: plt.savefig(SavePath(filename,"_E_vs_alpha.png"))
    plt.show()
    
def PlotAtFixedVal(filename, colnames, fixedqty, fixedvals, logplot=0,save = False,savetype='png'):
    '''
    input:
        colnames: length 2 array of [x,y]
        fixedqty: name of quantity held fixed while plotting
        fixedval: value of fixed quantity
    '''
    df = add_d(filename)
    a = colnames[0] # this is the x axis column
    b = colnames[1:] #these are the quantities to plot on the y-axis
    if len(b) == 1:
        if len(fixedvals) == 1:
            suffix = '_' + b[0] + '_vs_' + a + '_fixed_' + fixedqty + '_' + ("%.2f" %fixedvals[0]).replace(".","-")
        else: 
            suffix = '_' + b[0] + '_vs_' + a + '_fixed_' + fixedqty
    else:
        if len(fixedvals) == 1:
            suffix = '_' + b[0].split('_')[0] + '_vs_' + a + '_fixed_' + fixedqty + '_' + ("%.2f" %fixedvals[0]).replace(".","-")
        else: 
            suffix = '_' + b[0].split('_')[0] + '_vs_' + a + '_fixed_' + fixedqty
    suffix += "." + savetype
    print(suffix)

    fig = plt.figure(figsize=(6,4.5))
    ax = fig.add_subplot(111)
    for val in fixedvals:
        val, fixed_x, fixed_ys = FindArrs(df, colnames, fixedqty, val)
        if fixedqty == 'eta': fixedqty = '\\' + fixedqty
        for i,ylist in enumerate(fixed_ys):
            pre,post = b[i].split('_')
            if b[i] =='E_opt':
                #cut off values where bipolaron formation not possible (E_opt)
                stop = np.where(fixed_x<0.44)[0]
                ax.plot(fixed_x[stop], ylist[stop],label= '$' + pre + '_{' + post + '}$, $' + fixedqty + ' = $%.2f' %val)
            else:    
                ax.plot(fixed_x, ylist,label= '$' + pre + '_{' + post + '}$, $' + fixedqty + ' = $%.2f' %val)
        #find intersection point
    if (len(fixed_ys)==2) & ('E_opt' in b) & ('E_inf' in b):
        idx = np.argwhere(np.diff(np.sign(fixed_ys[0] - fixed_ys[1]))).flatten()
        ax.plot(fixed_x[idx], fixed_ys[0][idx], 'ro')

    if logplot == 1:
        ax.semilogx()
    if logplot == 2:
        ax.loglog()

    if len(b) > 1: ax.set_ylabel("$" + b[0].split('_')[0] + "/K$")
    else: ax.set_ylabel('$' + b[0] + '$')
    if a == 'eta': a = "\\" + a
    ax.set_xlabel('$' + a + '$')
    ax.legend()

    plt.tight_layout()
    #print(format_filename(filename))

    if save == True: 
        plt.savefig(SavePath(filename,suffix))
    plt.show()

def fitfn(x,a,x0,n):
    f = a*(np.abs(x-x0))**n 
    #print((f,x,a,x0,n))
    return f

def yPowerLaw(filename, etaval, bnds, guess=[0.5,7.5,0.5], zoom=False, plot=True,save=False): 
    '''
    inputs:
        bnds: tuple of 2 arrays for lower + upper bounds for the parameters a, x0, n
        guess: initial guess
    '''

    suffix = "_yPwrLaw_eta_" + str(etaval).replace(".","-")
    print(suffix)
    df = pd.read_csv(filename)
    whys = df['y_opt'].values #array of ys
    Us = df['U'].values #array of U values
    etas = df['eta'].values 

    #find the nearest index/value of eta in the etas array
    idx = (np.abs(etas-etaval)).argmin()
    print("Requested: eta = " + str(etaval) + "\tfound: " + str(etas[idx]))
    idxs = np.where(etas==etas[idx])
    fixedUs = Us[idxs]
    ys = whys[idxs]

    #fit array of nonzero y values to power law curve
    #print(ys) 
    newys = np.array([y if (y >= 1E-3) else 0 for y in ys])
    ys = np.unique(newys)
    #print(ys)
    yoos = np.zeros(len(ys))

    for i,y in enumerate(ys):
        findUs = np.where(newys==y)
        maxU = Us[findUs].max()
        yoos[i] = maxU

    if zoom == True:
        Uidx = np.where(yoos < 8)[0]
        yoos = yoos[Uidx]
        ys = ys[Uidx]

    param, param_cov = curve_fit(fitfn,yoos,ys, p0=guess,bounds=bnds)
    #print(param)
    #print(np.sqrt(np.diag(param_cov))) #standard deviation of the parameters in the fit
    a,x0,n = param
    
    if plot==True:
        fig = plt.figure(figsize=(10,4.5))
        ax = fig.add_subplot(121)
        ax.plot(yoos, ys,label='data')
        ax.set_xlabel('U')
        ax.set_ylabel('y')
        ans = np.array([fitfn(u,a,x0,n) for u in yoos])
        ax.plot(yoos,ans,color='red',label='fit')
        ax.plot(x0,0,color='green',marker='o')
        manfit = np.array([fitfn(u,a,x0,0.5) for u in yoos])
        #ax.plot(yoos,manfit,color='green',label='manfit')
    
        textstr = '\n'.join((
        r'$\eta=%.2f$' % (etas[idx]),
        r'$y(U) = C|U-U_0|^n$',
        r'$C=%.2f$' % (a, ),
        r'$U_0=%.2f$' % (x0, ),
        r'$n=%.2f$' % (n, )))

        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top')

        ax.legend(loc=4)

        ax2 = fig.add_subplot(122)
        minU = 0.5*(np.floor(yoos[0]) + yoos[0])
        maxU = 0.5*(np.ceil(yoos[0]) + yoos[0])

        #print(yoos[0])
        sweepU0 = [ (U_0,[np.abs(u-U_0) for u in yoos]) for U_0 in np.linspace(minU,maxU,5)]
        #deltaUs = [np.abs(u-x0) for u in yoos]
        for U0, Ulist in sweepU0: #plot family of curves sweeping through possible U_0 values
            lab = '$U_0 = %.3f$' %U0 
            ax2.plot(Ulist, ys,label=lab)
        ax2.set_xlabel('$|U-U_0|$')
        ax2.set_ylabel('y')
        ax2.set_xlim(left=5E-3)
        ax2.loglog()
        ax2.legend(loc=4)

        plt.tight_layout()
    
        if save == True: 
            plt.savefig(SavePath(filename,suffix))

        plt.show()
    return a,x0,n

def y_Across_etas(filename):
    '''
    Examining how the power low fit exponent changes as a function of eta
    inputs:
        bnds: tuple of 2 arrays for lower + upper bounds for the parameters a, x0, n
        guess: initial guess
    '''

    suffix = "_PwrLawExp(eta)_"
    print(suffix)

    etavals = np.linspace(0,.18,20)
    #etavals=np.array([0.1])
    fitexpnts = np.zeros(len(etavals))

    for i, val in enumerate(etavals):
        if val==0: 
            guess = [0.5,7.5,0.5]
            bnds = ([1E-10,7,1E-10],[1,8,5]) 
        elif val > 0.16 and val < 0.18:
            guess = [0.01,9,0.1]
            bnds = ([1E-10,7,1E-10],[1,10,1]) 
        elif val >= 0.18:
            guess = [0.01,9,1]
            bnds = ([1E-10,7,1E-10],[1,10,2]) 
        else:
            guess = [0.5,8,0.5]
            bnds = ([1E-10,7,1E-10],[1,10,5]) 

        _,_,fitexpnts[i] = yPowerLaw(filename, val, bnds, guess,plot=True)

    print(fitexpnts)
    fig = plt.figure(figsize=(6,4.5))
    ax = fig.add_subplot(111)
    ax.plot(etavals, fitexpnts,marker='.')
    ax.set_xlabel('$\eta$')
    ax.set_ylabel('n')
    plt.show()

def Find_d(filename,xcolname,ycolname):
    '''Find points where d=0, and plot them on contour plot. Should return array of x's and y's to feed into plot.'''
    df = pd.read_csv(filename)
    whys = df['y_opt'].values #array of ys
    sls = df['s_opt'].values #array of sigma/ls
    
    ds = np.array([y*s*l if y*s*l >= 1E-11 else 0 for s,y in zip(sls,whys)]) #elec separation distances - dimensionalized; setting cutoff boundary between 1E-11 and 1E-15 doesn't qualitatively change the shape of the boundary
    a = xcolname; b=ycolname;
    ayes = df[a].values
    bees = df[b].values
    idxs = np.where(ds==0)
    print(len(idxs[0]))
    apts = ayes[idxs]
    bpts = bees[idxs]
    xpts = np.unique(apts)
    ymaxpts = [0.]*len(xpts)
    yminpts = [0.]*len(xpts)
    #for each x value, find the MAXIMUM and MINIMUM y-values associated with it. This will let me plot a phase boundary for d=0.
    for i,x in enumerate(xpts):
        findys = np.where(apts==x)
        maxy = bpts[findys].max()
        miny = bpts[findys].min()
        ymaxpts[i] = maxy
        yminpts[i] = miny
    xpts = np.append(xpts,xpts[::-1])
    ypts = np.append(ymaxpts,yminpts[::-1])
    #print([(x,y) for x,y in zip(xpts,ypts)])
    return xpts,ypts
    
def FindCoords(contour):
    #pull coordinates of the delta E = 0 contour
    etas_0 = [[]]*len(contour.collections[0].get_paths())
    Us_0 = [[]]*len(contour.collections[0].get_paths())
    for i,p in enumerate(contour.collections[0].get_paths()):
        v = p.vertices
        etas_0[i] = v[:,0]
        Us_0[i] = v[:,1]
    #print(etas_0)
    #for n,U in zip(etas_0[0],Us_0[0]):
    #    print((n,U))
    return etas_0, Us_0

def plotContour(filename, colnames,xlims=(), ylims=(), zlims=(),save=False, zero=False,suffix='_phasedia.png', logplot=0,point=True,minmax=0,dcont = False):
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
    ns, Us, Z = parse_CSV(df,colnames)
    a,b,c = colnames
    print(Z.min())
    whar = np.where(Z==Z.min())
    #print(whar)
    #print(ns[0][whar[0][0]],Us[whar[1][0]][0])
    #ax.plot(ns[0][whar[0][0]],Us[whar[1][0]][0],'ko')

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
        cpmax=0.
    else:
        cpmin = Z.min()
        cpmax = Z.max()
    cp = ax.contourf(ns, Us, Z, levels = MaxNLocator(nbins=20).tick_values(cpmin, cpmax))

    #set limits on x and y axes if argument given
    if len(xlims) >0: ax.set_xlim(xlims[0],xlims[1])
    if len(ylims) >0: ax.set_ylim(ylims[0],ylims[1])
    
    if zero== True: 
        #draw contour where binding energy = 0
        if 'E_binding' not in colnames:
            _,_, Eb = parse_CSV(df,[a,b,'E_binding'])
            zerocont = ax.contour(ns, Us, Eb, [0.], colors=('r',), linewidths=(1,), origin='lower')
        else: zerocont = ax.contour(ns, Us, Z, [0.], colors=('r',), linewidths=(1,), origin='lower')
        e0,u0 = FindCoords(zerocont)
        #for elist, ulist in zip(e0,u0):
        #    print([(1-et)*yu/2 for et,yu in zip(elist,ulist)]) #print alpha values for the bipolaron binding boundary region
    
    #Plot STO values for eta and U
    if point == True: 
        ax.plot(eta_STO,U_STO,color='magenta',marker='o')
        ax.plot(etaKTO,UKTO,color='red',marker='o') #KTO, potassium tantalate
        ax.plot(9.05E-2, 2.6 ,color='red',marker='o') #PbS
        ax.plot(8.18E-2, 2.58,color='red',marker='o') #PbSe
        ax.plot(8.26E-2, 1.87,color='red',marker='o') #PbTe
        ax.plot(3.75E-2, 3.12,color='red',marker='o') #SnTe
        ax.plot(8E-2, 0.47,color='red',marker='o') #GeTe
    
    #plot d=0 contour
    if dcont == True:
        xd0s, yd0s = Find_d(filename,a,b)
        ax.plot(xd0s,yd0s,color='orange')
        colnames[2] = 'd'
        #_,_,ds = parse_CSV(df,colnames)
        #ax.contour(ns, Us, ds, [1E-11], colors=('g',), linewidths=(1,), origin='lower')

    if logplot == 1:
        ax.semilogy()
    if logplot == 2:
        ax.loglog()

    cbar=fig.colorbar(cp) # Add a colorbar to a plot
    cbar.ax.set_ylabel(c)
    ax.set_xlabel(a)
    ax.set_ylabel(b)
    plt.tight_layout()
    #print(format_filename(filename))
    if save == True: 
        plt.savefig(SavePath(filename,suffix))
    plt.show()

if __name__ == "__main__":
    filename = sys.argv[1]

    #FitPolaronData
    #Plot_vs_Alpha(filename,opt=2,save=False)

    #PlotAtFixedVal(filename, ['y_opt','E_opt'], 'eta', [eta_STO], logplot=0,save = False)
    #PlotAtFixedVal(filename, ['eta','E_opt','E_inf'], 'U', [U_STO], savetype='eps',logplot=0,save = False)
    #PlotAtFixedVal(filename, ['U','a_opt'], 'eta', [0,0.1,0.2,0.5], logplot=2,save = True)
    #PlotAtFixedVal(filename, ['alpha','E_inf','Einf_ana'], 'eta', [0], logplot=0,save = False)

    #y_Across_etas(filename)
    #yPowerLaw(filename, 0,bnds = ([1E-10,7,1E-10],[1,8,5]), guess=[0.5,7.5,0.5],zoom=False, save=True) #for cy_var_a_zoom_lgU.csv file
    #yPowerLaw(filename, 0.1,bnds = ([1E-10,7,1E-10],[1,10,5]),guess=[0.5,8,0.5],zoom=False, save=False) #for cy_var_a_zoom_lgU.csv file
    #yPowerLaw(filename, 0.05,bnds = ([1E-10,7,1E-10],[1,10,5]),zoom=False, save=True) #for cy_var_a_zoom_lgU.csv file #doesn't work for eta > 1.9, no nonzero y vals

    #plotContour(filename,['eta','U','E_opt'],point=True,minmax=3,suffix='_phasedia.eps',logplot=0,save=False,zero=False,dcont=False)
    #plotContour(filename,['eta','U','E_inf'],point=True,minmax=5,suffix='_phasedia.eps',logplot=0,save=False,zero=True,dcont=True)
    #plotContour(filename,['eta','U','E_binding'],point=True,xlims=(0,0.2),ylims=(1E-5,13),minmax=1,suffix='_phasedia.png',logplot=0,save=True,zero=False,dcont=True)
    plotContour(filename,['eta','U','E_binding'],logplot=1,point=False,minmax=1, save=False,suffix='_zoom.eps',zero=False,dcont=False)
    #plotContour(filename,['eta','U','E_opt'],xlims=(0,0.2),point=False,minmax=2, save=True,suffix='_EoptPD',zero=False, dcont=True)
    #plotContour(filename,['eta','U','y_opt'],suffix='_yoptPD.png',save=True,point=False,dcont=True,minmax=2)
    #plotContour(filename,['eta','U','s_opt'],suffix='_soptPD.png',save=True,point=False,dcont=True,minmax=2)
    #plotContour(filename,['eta','U','a_opt'],suffix='_aoptPD.png',save=True,point=False,dcont=True,minmax=2)


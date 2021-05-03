import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import warnings

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

def integrand(u, b, c, rnorm,y):
    '''
    Inputs:
        c: contains c, c1, c2 doubles
        rnorm: contains |r/sig|, |\vec{r}/sig + \vec{y}|, |\vec{r}/sig - \vec{y}| 
        u: integration variable
        y: dist b/w elecs
    '''
    fn = np.exp(-u**2)/(u**2 +b) * (2*np.exp(-y**2/2)/rnorm[0]* np.sin(c[0]*u) + np.sin(c[1]*u)/rnorm[1] + np.sin(c[2]*u)/rnorm[2])
    return fn

def integral(x, a,r):
    '''
    input:
    x: numpy array of [y,sigma]
    r: vector of x,y,z
    d: vector of x,y,z (choose to point along z)
    '''
    A = (1-a/2)**2 + (a/2)**2
    b = A*x[1]**2/(2*a**2*l**2)
    y_arr = np.array([0,0,x[0]])
    d = y_arr*x[1] #d vector

    rnorm = np.linalg.norm(r)
    rnorm2 = np.linalg.norm(r+d)
    rnorm3 = np.linalg.norm(r-d)
    norms = np.array([rnorm,rnorm2,rnorm3])
    cs = 2*norms/(x[1]*np.sqrt(A))
    #print(cs)
    const = -x[1]/(a**2*l) *np.sqrt(A*alpha/(l*np.pi**3))* 1/ (1+np.exp(-x[0]**2/2))
    integral = integrate.quad(integrand,0,10,args=(b,cs,norms,x[0]))
    return const*integral[0]

if __name__ == '__main__': 
    #STO values, ish
    sig = 1.11E-9
    y = 1.3
    sl = sig/l
    a=0.67
    #y = 1000
    delta_r = np.linspace(1E-10,10,100)
    rs = [np.array([0,0,dr]) for dr in delta_r]
    points = [integral([y,sig],a,r) for r in rs] #to hold integral vals
    #print(points)
    fig = plt.figure(figsize=(10,4.5))
    ax = fig.add_subplot(111)
    ax.plot(delta_r, points,label='data')
    ax.set_xlabel('r')
    ax.set_ylabel('$f(\\vec{r})$')
    plt.tight_layout()
    plt.show()


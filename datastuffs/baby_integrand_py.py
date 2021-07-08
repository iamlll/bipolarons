import numpy as np
from scipy import integrate
from scipy.special import erf, erfc
import warnings

def py_integrand(u,b,c,x):
    '''
    Inputs:
        u: integration variable
        x: numpy array 
    '''
    number = np.exp(-u**2)/(u**2+b) * (c*(1+2*np.exp(-x[0]**2)) + np.sin(c*u)/u + 8*np.exp(-x[0]**2/2)*np.sin(c*u/2)/u)
    return number

def py_integrand_sing(u,b,c,x):
    number = np.exp(-u**2)/(u**2+b) * ((np.sin(c*u)/u-c) + 4*np.exp(-x[0]**2/2)*(2*np.sin(c*u/2)/u - c))
    return number

def py_integral_sing(b,c,x):
    integral = integrate.quad(py_integrand_sing,0,10, args=(b,c,x))
    val = erfc(np.sqrt(b))
    val2 = 0
    if val != 0: val2 = np.exp(b)
    ana = (1+2*np.exp(-x[0]**2/2) + np.exp(-x[0]**2))*np.pi*c/np.sqrt(b) * val2* val
    return integral[0] + ana

def py_integral(b,c,x):
    integral = integrate.quad(py_integrand,0,10,args=(b,c,x))
    return integral[0]

def py_integral_inf(b,c):
    integral = integrate.quad(lambda u: np.exp(-u**2)/(u**2+b) * (c+ np.sin(c*u)/u ),0,10)
    return integral[0]

def py_integral_sing_inf(b,c):
    integral = integrate.fixed_quad(lambda u: np.exp(-u**2)/(u**2+b) * (np.sin(c*u)/u-c),0,10,n=50)
    val = erfc(np.sqrt(b))
    val2 = 0
    if val != 0: val2 = np.exp(b)
    ana = np.pi*c/np.sqrt(b) * val* val2
    return integral[0] + ana

def I_py(b, c, x):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            result = py_integral(b,c,x)
            return result
        except integrate.IntegrationWarning:
            result = py_integral_sing(b,c,x)
            return result

def I_py_inf(b, c):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            result = py_integral_inf(b,c)
            return result
        except integrate.IntegrationWarning:
            result = py_integral_sing_inf(b,c)
            return result

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
    A = (1-a/2)**2 + (a/2)**2
    b = A*x[1]**2/a**2
    c = x[0]*np.sqrt(2/A)
    KE = 1/x[1]**2 *(3-1/2* x[0]**2/(np.exp(x[0]**2/2)+1))
    coul = U/x[1] * (1/x[0]*erf(x[0]/np.sqrt(2)) + np.sqrt(2/np.pi)* np.exp(-x[0]**2/2))/(1+np.exp(-x[0]**2/2))
    const = -(1-n)*U* 2/(np.pi*(1 + np.exp(-x[0]**2/2))**2)* A*x[1]/(a**2*x[0])
    e_ph = const* I_py(b,c,x)
    E = KE + e_ph + coul
    return E

def E_infty(x,n,U,a):
    '''Hybrid calc energy evaluated at y->infty, normalized by KE'''
    A = (1-a/2)**2 + (a/2)**2
    b = A*x[1]**2/a**2
    c = x[0]*np.sqrt(2/A)
    KE = 3/x[1]**2
    coul = U/x[1] * 1/x[0]*erf(x[0]/np.sqrt(2))
    const = -(1-n)*U* 2/np.pi* A*x[1]/(a**2*x[0])
    e_ph = const* I_py_inf(b,c)
    E = KE + e_ph + coul
    return E

import numpy as np
cimport numpy as np
from scipy import integrate
from scipy.optimize import minimize
import baby_integrand as bby
from libc.math cimport isinf, fabs, sqrt, pi, exp, sin, pow, erf, erfc
import warnings

def integral(double b,double c,tuple x):
    cdef tuple integral = integrate.quad(bby.integrand,0,10,args=(b,c,x))
    return integral[0]

def integral_sing(double b,double c, tuple x):
    cdef tuple integral = integrate.quad(bby.integrand_sing,0,10,args=(b,c,x))
    cdef double val = erfc(sqrt(b))
    cdef double val2 = 0.
    if val > 1E-300: 
        val2 = exp(b)
    #cdef double val, val2
    #if sqrt(b) < 26.7: #idk why but this seems to be the rounding limit for scipy erfc
    #    val = erfc(sqrt(b))
    #    val2 = exp(b)
    cdef double ana = (1+2*exp(-pow(x[0],2)/2) + exp(-pow(x[0],2)))*pi*c/sqrt(b) * val * val2
    return integral[0] + ana

def integral_inf(double b,double c):
    cdef tuple integral = integrate.quad(bby.integrand_inf,0,10,args=(b,c))
    return integral[0]

def integral_sing_inf(double b, double c):
    cdef tuple integral = integrate.fixed_quad(bby.integrand_sing_inf,0,10,n=50, args=(b,c))
    cdef double val = erfc(sqrt(b))
    cdef double val2 = 0.
    if val > 1E-300: 
        val2 = exp(b)
    
    cdef double ana = pi*c/sqrt(b) * val*val2
    if isinf(ana) == 1: 
        print("b ",b,"c ",c)
        print(val)
    return integral[0] + ana

def I_cy(double b, double c, np.ndarray[double, ndim=1] x):
    cdef double result = integral(b,c,tuple(x))
    return result

def I_cy_warn(double b, double c, np.ndarray[double, ndim=1] x):
    cdef double result
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            result = integral(b,c,tuple(x))
            return result
        except integrate.IntegrationWarning:
            result = integral_sing(b,c,tuple(x))
            return result

def I_cy_inf(double b, double c):
    cdef double result
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            result = integral_inf(b,c)
            return result
        except integrate.IntegrationWarning:
            result = integral_sing_inf(b,c)
            return result

def opt_hybE(np.ndarray[double, ndim=1] x, double n, double U, double a):
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
    cdef double A = pow(1-a/2,2) + pow(a/2,2)
    cdef double b = A*pow(x[1],2)/pow(a,2)
    cdef double c = x[0]*sqrt(2/A)
    cdef double KE = 1./pow(x[1],2) *(3.- 0.5* pow(x[0],2)/(exp(pow(x[0],2)/2)+1))
    cdef double coul = U/x[1] * (1/x[0]*erf(x[0]/sqrt(2)) + sqrt(2/pi)* exp(-pow(x[0],2)/2))/(1+exp(- pow(x[0],2)/2))
    cdef double const = -(1-n)*U* 2/(pi*pow(1 + exp(-pow(x[0],2)/2),2))* A*x[1]/(pow(a,2)*x[0])
    cdef double e_ph = const* I_cy(b,c,x)
    cdef double E = KE + e_ph + coul
    return E

def E_infty(np.ndarray[double, ndim=1] x, double n, double U, double a):
    '''Hybrid calc energy evaluated at y->infty, normalized by KE'''
    cdef double A = pow(1-a/2,2) + pow(a/2,2)
    cdef double b = A*pow(x[1],2)/pow(a,2)
    cdef double c = x[0]*sqrt(2/A)
    cdef double KE = 3./pow(x[1],2)
    cdef double coul = U/x[1] * 1./x[0]*erf(x[0]/sqrt(2))
    cdef double const = -(1-n)*U* 2./pi* A*x[1]/(pow(a,2)*x[0])
    cdef double e_ph = const* I_cy_inf(b,c)
    cdef double E = KE + e_ph + coul
    return E

def minimization(tuple args):
    cdef double a = args[0]
    cdef double n = args[1]
    cdef double u = args[2]
    cdef tuple bnds = ((1E-3,10), (1E-3, 10)) #y,s
    cdef tuple bnds_inf = ((5000,5010), (1E-3, 1E10)) #y,s
    cdef np.ndarray[double, ndim=1] guess = np.array([1.,1.])
    cdef np.ndarray[double, ndim=1] guess_inf = np.array([5000.,1.])
    result = minimize(opt_hybE,guess,args=(n,u,a),bounds=bnds)
    cdef np.ndarray[double, ndim=1] minvals = result.x
    cdef double E_opt = opt_hybE(minvals,n,u,a)

    #Find E_infty semi-analytically
    res_inf = minimize(E_infty,guess_inf,args=(n,u,a),bounds=bnds_inf)
    cdef np.ndarray[double, ndim=1] m_inf = res_inf.x
    cdef double E_inf = E_infty(m_inf,n,u,a)
    cdef double E_binding = (E_opt - E_inf)/fabs(E_inf)
    return n,u,minvals[1],minvals[0],E_opt,m_inf[1],m_inf[0],E_inf,E_binding

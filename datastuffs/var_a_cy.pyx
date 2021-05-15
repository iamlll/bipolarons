import numpy as np
cimport numpy as np
from scipy import integrate
from scipy.special.cython_special import erfcx
from scipy.optimize import minimize, basinhopping,shgo
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
    cdef double val = erfcx(sqrt(b))
    if isinf(val) == 1:
        return integral[0]
    else:
        return integral[0]+ pi*c/sqrt(b) * val

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

def opt_hybE(np.ndarray[double, ndim=1] x, double n, double U):
    '''
    Inputs:
        x[0] = y = d/sigma
        x[1] = s = sigma/l
        x[2] = a = magic param
        U = ratio of Coulomb energy unit to KE energy unit (effective Rydberg/hw): U = e^2/(epsinf*l) / hbar^2/(2ml^2)
        n = eta = epsinf/epssr
    Output:
        E/K, ratio of total energy expectation value to kinetic energy coeff = K = hbar^2/(2ml^2)
    '''
    cdef double A = pow(1-x[2]/2,2) + pow(x[2]/2,2)
    cdef double b = A*pow(x[1],2)/pow(x[2],2)
    cdef double c = x[0]*sqrt(2/A)
    cdef double KE = 1./pow(x[1],2) *(3.- 0.5* pow(x[0],2)/(exp(pow(x[0],2)/2)+1))
    cdef double coul = U/x[1] * (1/x[0]*erf(x[0]/sqrt(2)) + sqrt(2/pi)* exp(-pow(x[0],2)/2))/(1+exp(- pow(x[0],2)/2))
    cdef double const = -(1-n)*U* 2/(pi*pow(1 + exp(-pow(x[0],2)/2),2))* A*x[1]/(pow(x[2],2)*x[0])
    cdef double e_ph = const* I_cy(b,c,x)
    cdef double E = KE + e_ph + coul
    return E

def E_infty(np.ndarray[double, ndim=1] x, double n, double U):
    '''Hybrid calc energy evaluated at y>>1, normalized by KE'''
    cdef double A = pow(1-x[2]/2,2) + pow(x[2]/2,2)
    cdef double b = A*pow(x[1],2)/pow(x[2],2)
    cdef double c = x[0]*sqrt(2/A)
    cdef double KE = 3./pow(x[1],2)
    cdef double coul = U/x[1] * 1./x[0]*erf(x[0]/sqrt(2))
    cdef double const = -(1-n)*U* 2./pi* A*x[1]/(pow(x[2],2)*x[0])
    cdef double e_ph = const* I_cy_inf(b,c)
    cdef double E = KE + e_ph + coul
    return E

def minimization(tuple args):
    cdef double n = args[0]
    cdef double u = args[1]
    cdef tuple bnds = ((1E-15,10), (1E-3, 10),(1E-15,1.)) #y,s,a
    cdef tuple bnds_inf = ((5000,5010), (1E-3, 1E10),(1E-15,1.)) #y,s,a
    cdef np.ndarray[double, ndim=1] guess = np.array([1.,1.,0.5])
    cdef np.ndarray[double, ndim=1] guess_inf = np.array([5000.,1.,0.5])
    result = minimize(opt_hybE,guess,args=(n,u),bounds=bnds)
    cdef np.ndarray[double, ndim=1] minvals = result.x
    cdef double E_opt = opt_hybE(minvals,n,u)

    #Find E_infty semi-analytically
    res_inf = minimize(E_infty,guess_inf,args=(n,u),bounds=bnds_inf)
    cdef np.ndarray[double, ndim=1] m_inf = res_inf.x
    cdef double E_inf = E_infty(m_inf,n,u)
    cdef double E_binding = (E_opt - E_inf)/fabs(E_inf)
    return n,u,minvals[2],minvals[1],minvals[0],E_opt,m_inf[2],m_inf[1],m_inf[0],E_inf,E_binding

def min_Einf(tuple args):
    '''Just minimize E_inf to compare with lg sigma optimization'''
    cdef double n = args[0]
    cdef double u = args[1]
    cdef tuple bnds_inf = ((1E2,1E4), (1E-3, 1E3),(1E-5,2.)) #y,s,a
    cdef np.ndarray[double, ndim=1] guess_inf = np.array([1E4,10.,1])
    res_inf = minimize(E_infty,guess_inf,args=(n,u),bounds=bnds_inf)
    #res_inf = basinhopping(E_infty,guess_inf,niter=100,minimizer_kwargs={"method": "L-BFGS-B","args":(n,u)})
    cdef np.ndarray[double, ndim=1] m_inf = res_inf.x
    cdef double E_inf = E_infty(m_inf,n,u)
    return n,u,m_inf[2],m_inf[1],m_inf[0],E_inf

def Einf_lg_sig(np.ndarray[double, ndim=1] x, double n, double U):
    '''
    Inputs:
        x[0] = y = d/sigma
        x[1] = s = sigma/l
        U = ratio of Coulomb energy unit to KE energy unit (effective Rydberg/hw): U = e^2/(epsinf*l) / hbar^2/(2ml^2)
        n = eta = epsinf/epssr
    Output:
        E/K, ratio of total energy expectation value to kinetic energy coeff = K = hbar^2/(2ml^2)
    '''
    cdef double KE = 1./pow(x[1],2) *(3.- 0.5* pow(x[0],2)/(exp(pow(x[0],2)/2)+1))
    cdef double coul = U/x[1] * (1/x[0]*erf(x[0]/sqrt(2)) + sqrt(2./pi)* exp(-pow(x[0],2)/2))/(1+exp(- pow(x[0],2)/2))
    cdef double e_ph = -(1-n)*U/(pi*pow(1 + exp(-pow(x[0],2)/2),2))* (2*sqrt(pi)/x[1]* (1 + 2*exp(-pow(x[0],2)) -1/pow(x[1],2)*(1+ 6*exp(-pow(x[0],2)) + 8*exp(-3.*pow(x[0],2)/4)) ) + pi/(x[1]*x[0])*(erf(x[0]) + 8*exp(-pow(x[0],2)/2)*erf(x[0]/2) ) )
    cdef double E = KE + e_ph + coul
    return E

def min_lg_sig(tuple args):
    '''
    scipy.optimize.minimize gets stuck in local minima; 
    this method (optimize.basinhopping) gets stuck around the initial guess for y, but varies for sigma. Works better when no bounds are specified
    Another global scipy solver, shgo, has no guess specified but gets stuck at the initial param values attempted (i.e. halfway between upper/lower bounds)
    '''
    cdef double n = args[0]
    cdef double u = args[1]
    #we're only testing this limit for y, sigma both large and finite (but y>>sigma)
    cdef np.ndarray[double, ndim=1] guess_inf = np.array([8000.,200.])
    res_inf = basinhopping(Einf_lg_sig,guess_inf,niter=100,minimizer_kwargs={"method": "L-BFGS-B","args":(n,u)})
    cdef np.ndarray[double, ndim=1] m_inf = res_inf.x
    cdef double E_inf = res_inf.fun
    return n,u,1,m_inf[1],m_inf[0],E_inf


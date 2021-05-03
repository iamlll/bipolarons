import numpy as np
cimport numpy as np
from scipy import integrate
from scipy.optimize import minimize
import baby_integrand as bby
from libc.math cimport isinf, isnan, fabs, sqrt, pi, exp, sin, pow, erf, erfc
cimport scipy.special.cython_special as cy_scipy
import warnings

'''Direct electron-phonon potential terms'''
def integrand_direct_inf(double u, double b, double c):
    ''' integrand evaluated at y-> infty'''
    cdef double fn = exp(-pow(u,2))/(pow(u,2)+b) * (c+ sin(c*u)/u )
    return fn

from libc.stdio cimport printf
from cython.view cimport array as cvarray
def integrand_sing_direct_inf(np.ndarray[double, ndim=1] u, double b, double c):
    '''handling IntegrationWarning exceptions at y->infty'''
    cdef Py_ssize_t xmax = u.shape[0]
    cdef Py_ssize_t x
    #make a memoryview for the numpy array - see https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html#efficient-indexing-with-memoryviews
    result = np.zeros(xmax, dtype=np.float)
    cdef double[:] result_view = result
    for x in range(xmax):
        result_view[x] = exp(-pow(u[x],2))/(pow(u[x],2)+b) * (sin(c*u[x])/u[x] - c)
    return result

def integral_direct_inf(double b,double c):
    cdef tuple integral = integrate.quad(integrand_direct_inf,0,10,args=(b,c))
    return integral[0]

def integral_sing_direct_inf(double b, double c):
    if b==0: b=1E-10
    cdef tuple integral = integrate.fixed_quad(integrand_sing_direct_inf,0,10,n=50, args=(b,c))
    cdef double val = cy_scipy.erfcx(sqrt(b))
    return integral[0] + pi*c/sqrt(b)*val

def I_cy_direct_inf(double b, double c):
    cdef double result
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            result = integral_direct_inf(b,c)
            return result
        except integrate.IntegrationWarning:
            result = integral_sing_direct_inf(b,c)
            return result


def erfcxInt(double b):
    '''literally the erfcx() integral.'''
    def integrand(double u, double b):
        return exp(-pow(u,2))/(pow(u,2)+b)
    cdef tuple integral = integrate.quad(integrand, 0, 10, args=(b,))
    return integral[0]

from libc.stdio cimport printf
from cython.view cimport array as cvarray

def integral_direct_inf3(double b,double c):
    def integrand_direct_inf3(double u, double b, double c):
        ''' integrand evaluated at y-> infty'''
        cdef double fn = exp(-pow(u,2))/(pow(u,2)+b) * (sin(c*u)/(c*u) -1)
        return fn
    cdef tuple integral = integrate.quad(integrand_direct_inf3,0,10,args=(b,c))
    return integral[0]

def integral_sing_direct_inf3(double b, double c):
    def integrand_sing_direct_inf3(np.ndarray[double, ndim=1] u, double b, double c):
        '''handling IntegrationWarning exceptions at y->infty'''
        cdef Py_ssize_t xmax = u.shape[0]
        cdef Py_ssize_t x
        #make a memoryview for the numpy array - see https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html#efficient-indexing-with-memoryviews
        result = np.zeros(xmax, dtype=np.float)
        cdef double[:] result_view = result
        for x in range(xmax):
            result_view[x] = exp(-pow(u[x],2))/(pow(u[x],2)+b) * (sin(c*u[x])/(c*u[x]) - 1)
        return result
    cdef tuple integral = integrate.fixed_quad(integrand_sing_direct_inf3,0,10,n=50, args=(b,c))
    return integral[0]

def I_cy_direct_inf3(double b, double c):
    cdef double result
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            result = integral_direct_inf3(b,c)
            return result
        except integrate.IntegrationWarning:
            result = integral_sing_direct_inf3(b,c)
            return result

'''indirect (cross/exchange) electron-phonon potential term'''
def integrand_indirect_inf(double u, double b, double c, double a):
    ''' integrand evaluated at y-> infty'''
    cdef double fn = exp(-pow(u,2))/(pow(u,2)+b) * (sin(a*c*u)/(a*u) + sin(c*u)/u )
    return fn

from libc.stdio cimport printf
from cython.view cimport array as cvarray
def integrand_sing_indirect_inf(np.ndarray[double, ndim=1] u, double b, double c, double a):
    '''handling integrationwarning exceptions at y->infty'''
    cdef Py_ssize_t xmax = u.shape[0]
    cdef Py_ssize_t x
    #make a memoryview for the numpy array - see https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html#efficient-indexing-with-memoryviews
    result = np.zeros(xmax, dtype=np.float)
    cdef double[:] result_view = result
    for x in range(xmax):
        result_view[x] = exp(-pow(u[x],2))/(pow(u[x],2)+b) * (sin(a*c*u[x])/(a*u[x]) + sin(c*u[x])/u[x] - 2*c)
    return result

def integral_indirect_inf(double b,double c, double a):
    cdef tuple integral = integrate.quad(integrand_indirect_inf,0,10,args=(b,c, a))
    return integral[0]

def integral_sing_indirect_inf(double b, double c, double a):
    cdef tuple integral = integrate.fixed_quad(integrand_sing_indirect_inf,0,10,n=50, args=(b,c, a))
    cdef double val = cy_scipy.erfcx(sqrt(b))
    return integral[0]+ pi*c/sqrt(b) * val

def I_cy_indirect_inf(double b, double c, double a):
    cdef double result
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            result = integral_indirect_inf(b,c, a)
            return result
        except integrate.IntegrationWarning:
            result = integral_sing_indirect_inf(b,c, a)
            return result

def integral_indirect_inf2(double b,double c, double a):
    def integrand2(double u, double b, double c, double a):
        ''' integrand evaluated at y-> infty'''
        cdef double fn = exp(-pow(u,2))/(pow(u,2)+b) * (sin(a*c*u)/(a*u) - c)
        return fn
    cdef tuple integral = integrate.quad(integrand2,0,10,args=(b,c, a))
    cdef double val = pi*c/(2*sqrt(b)) * cy_scipy.erfcx(sqrt(b))
    return integral[0]+ val

def integral_sing_indirect_inf2(double b, double c, double a):
    def integrand_sing_indirect_inf2(np.ndarray[double, ndim=1] u, double b, double c, double a):
        '''handling integrationwarning exceptions at y->infty'''
        cdef Py_ssize_t xmax = u.shape[0]
        cdef Py_ssize_t x
        #make a memoryview for the numpy array - see https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html#efficient-indexing-with-memoryviews
        result = np.zeros(xmax, dtype=np.float)
        cdef double[:] result_view = result
        for x in range(xmax):
            result_view[x] = exp(-pow(u[x],2))/(pow(u[x],2)+b) * (sin(a*c*u[x])/(a*u[x]) - c)
        return result
    cdef tuple integral = integrate.fixed_quad(integrand_sing_indirect_inf2,0,10,n=50, args=(b,c, a))
    cdef double val = cy_scipy.erfcx(sqrt(b))
    return integral[0]+ pi*c/(2*sqrt(b)) * val

def I_cy_indirect_inf2(double b, double c, double a):
    cdef double result
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            result = integral_indirect_inf2(b,c, a)
            return result
        except integrate.IntegrationWarning:
            result = integral_sing_indirect_inf2(b,c, a)
            return result

'''Calculate and minimize energy for y->inf'''
def E_infty(np.ndarray[double, ndim=1] x, double n, double U):
    '''Hybrid calc energy evaluated at y>>1, normalized by KE'''
    cdef double b1 = pow(1-x[2],2) *pow(x[1],2)/(2*pow(x[2],2))
    cdef double b2 = (1+pow(x[2],2)) *pow(x[1],2)/(2*pow(x[2],2))
    cdef double b3 = (pow(1-x[2],2) + 1 + pow(x[2],2)) *pow(x[1],2)/(4*pow(x[2],2))
    cdef double c1 = x[0]*sqrt(2)
    cdef double c2 = x[0]* (1-x[2])*sqrt(2)/sqrt(1 + pow(x[2],2))
    cdef double c3 = 2*x[0]/sqrt( pow(1-x[2],2) + 1 + pow(x[2],2))
    cdef double KE = 3./pow(x[1],2)
    cdef double coul = U/x[1] * 1./x[0]*erf(x[0]/sqrt(2))
    cdef double directterms = (1-x[2])/pow(x[2],2) * I_cy_direct_inf(b1,c1) + sqrt(2)*sqrt(1+ pow(x[2],2))* x[0]/pow(x[2],2) * I_cy_direct_inf(b2,c2)
    cdef double crossterm = (pow(1-x[2],2) + 1 + pow(x[2],2))/pow(x[2],2) * I_cy_indirect_inf(b3,c3,x[2])
    cdef double e_ph = -(1-n)*U/2.* x[1]/(2*pi* x[0])*(directterms + crossterm)
    cdef double E = KE + e_ph + coul
    return E

def min_Einf(tuple args):
    '''Just minimize E_inf to compare with lg sigma optimization'''
    cdef double n = args[0]
    cdef double u = args[1]
    cdef tuple bnds_inf = ((9E3,1E4), (1E-2, 1E3),(1E-5,0.999999999)) #y,s,a
    cdef np.ndarray[double, ndim=1] guess_inf = np.array([1E4,1.,0.5])
    res_inf = minimize(E_infty,guess_inf,args=(n,u),bounds=bnds_inf)
    cdef np.ndarray[double, ndim=1] m_inf = res_inf.x
    cdef double E_inf = E_infty(m_inf,n,u)
    return n,u,m_inf[2],m_inf[1],m_inf[0],E_inf

'''Calculate and minimize energy for y->inf'''
def E_infty2(np.ndarray[double, ndim=1] x, double n, double U):
    '''Hybrid calc energy evaluated at y>>1, normalized by KE'''
    cdef double b1 = pow(1-x[2],2) *pow(x[1],2)/(2*pow(x[2],2))
    cdef double b2 = (1+pow(x[2],2)) *pow(x[1],2)/(2*pow(x[2],2))
    cdef double b3 = (pow(1-x[2],2) + 1 + pow(x[2],2)) *pow(x[1],2)/(4*pow(x[2],2))
    cdef double c1 = x[0]*sqrt(2)
    cdef double c2 = x[0]* (1-x[2])*sqrt(2)/sqrt(1 + pow(x[2],2))
    cdef double c3 = 2*x[0]/sqrt( pow(1-x[2],2) + 1 + pow(x[2],2))
    cdef double KE = 3./pow(x[1],2)

    #cdef double elph1 = pi/sqrt(b1)* cy_scipy.erfcx(sqrt(b1)) + I_cy_direct_inf3(b1,c1)
    #cdef double elph2 = pi/sqrt(b2)* cy_scipy.erfcx(sqrt(b2)) + I_cy_direct_inf3(b2,c2)
    cdef double elph1 = pi/(2*sqrt(b1))* cy_scipy.erfcx(sqrt(b1))
    cdef double elph2 = pi/(2*sqrt(b2))* cy_scipy.erfcx(sqrt(b2))
    cdef double directterms = (1-x[2])/pow(x[2],2) *sqrt(2)*elph1 + sqrt(2*(1+pow(x[2],2)))/pow(x[2],2)* elph2
    cdef double crossterm = (pow(1-x[2],2) + 1 + pow(x[2],2))/pow(x[2],2)* 1/x[0] * I_cy_indirect_inf(b3,c3,x[2])
    cdef double e_ph = -(1-n)*U/2.* x[1]/(2*pi)*(directterms + crossterm)

    cdef double E = KE + e_ph
    return E

def min_Einf2(tuple args):
    cdef double n = args[0]
    cdef double u = args[1]
    cdef tuple bnds_inf = ((9E3,1E4), (1E-3, 1E3),(1E-5,0.999999999)) #y,s,a
    cdef np.ndarray[double, ndim=1] guess_inf = np.array([1E4,1.,0.5])
    res_inf = minimize(E_infty2,guess_inf,args=(n,u),bounds=bnds_inf)
    cdef np.ndarray[double, ndim=1] m_inf = res_inf.x
    cdef double E_inf = E_infty2(m_inf,n,u)
    return n,u,m_inf[2],m_inf[1],m_inf[0],E_inf

'''Calculate and minimize energy for y->inf; minimize wrt indep params b1, b2'''
def E_infty3(np.ndarray[double, ndim=1] x, double n, double U):
    '''
    Hybrid calc energy evaluated at y>>1, normalized by KE
    x = [b1, b2,y]
    '''
    cdef double a = x[1]/(x[1]-x[0]) + sqrt(x[0]*(2*x[1]-x[0]))/abs(x[1]-x[0])
    cdef double sig = 1/(2*a) * sqrt(pow(1-a,2)/x[0] + (1+pow(a,2))/x[1])
    cdef double c3 = 2*x[2]/sqrt( pow(1-a,2) + 1 + pow(a,2))
    cdef double KE = 3./pow(sig,2)

    cdef double elph1 = cy_scipy.erfcx(sqrt(x[0]))
    cdef double elph2 = cy_scipy.erfcx(sqrt(x[1]))
    cdef double crossterm = (pow(1-a,2) + 1 + pow(a,2))/a * sig/(pi*x[2])* I_cy_indirect_inf2((x[0]+x[1])/2,c3,a)
    cdef double e_ph = -(1-n)*U/(4.*a) *( elph1 + elph2 + crossterm)
    cdef double E = KE + e_ph
    return E

def min_Einf3(tuple args):
    cdef double n = args[0]
    cdef double u = args[1]
    cdef tuple bnds_inf = ((1E-7,1E3),(1E-7,1E3),(9E3,1E4)) #b1,b2,y
    cdef tuple cons = ({'type': 'ineq', 'fun': lambda x:  x[1]-x[0]-1E-8},)
    cdef np.ndarray[double, ndim=1] guess_inf = np.array([10.,10.,1E4])
    res_inf = minimize(E_infty3,guess_inf,args=(n,u),bounds=bnds_inf, constraints=cons)
    cdef np.ndarray[double, ndim=1] m_inf = res_inf.x
    cdef double E_inf = E_infty3(m_inf,n,u)
    cdef double a = m_inf[1]/(m_inf[1]-m_inf[0]) + sqrt(m_inf[0]*(2*m_inf[1]-m_inf[0]))/abs(m_inf[1]-m_inf[0])
    cdef double sig = 1/(2*a) * sqrt(pow(1-a,2)/m_inf[0] + (1+pow(a,2))/m_inf[1])
    return n,u,a,sig,m_inf[2],E_inf

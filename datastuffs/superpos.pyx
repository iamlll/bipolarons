import numpy as np
cimport numpy as np
from scipy import integrate
from scipy.optimize import minimize
import baby_integrand as bby
from libc.math cimport isinf, isnan, fabs, sqrt, pi, exp, sin, pow, erf, erfc, erfi
cimport scipy.special.cython_special as cy_scipy
import warnings

'''Direct electron-phonon potential terms'''
def integrand_direct_inf(double u, double expo, double b, double c):
    ''' integrand for direct terms evaluated at y-> infty, u is the integration variable'''
    cdef double fn = exp(-expo* pow(u,2))/(b*pow(u,2)+1) * (sin(c*u)/(c*u) + 1)
    return fn

from libc.stdio cimport printf
from cython.view cimport array as cvarray
def integrand_sing_direct_inf(np.ndarray[double, ndim=1] u, double expo, double b, double c):
    '''handling IntegrationWarning exceptions at y->infty'''
    cdef Py_ssize_t xmax = u.shape[0]
    cdef Py_ssize_t x
    #make a memoryview for the numpy array - see https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html#efficient-indexing-with-memoryviews
    result = np.zeros(xmax, dtype=np.float)
    cdef double[:] result_view = result
    for x in range(xmax):
        result_view[x] = exp(-expo*pow(u[x],2))/(b* pow(u[x],2)+1) * (sin(c*u[x])/(c*u[x]) + 1)
    return result

def I_cy_direct_inf(double expo, double b, double c):
    #print("exponent: %.3f\t b: %.3f\t c: %.3f" %(expo,b,c))
    if c == 0: c = 1E-20
    cdef tuple integral
    cdef double val = 0
    #cdef double val = pi/sqrt(b) * cy_scipy.erfcx(sqrt(expo/b))
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            integral = integrate.quad(integrand_direct_inf,0,10,args=(expo, b,c))
            return integral[0] + val
        except integrate.IntegrationWarning:
            integral = integrate.fixed_quad(integrand_sing_direct_inf,0,10,n=50, args=(expo, b,c))
            return integral[0] + val

'''indirect (cross/exchange) electron-phonon potential term'''
def integrand_indirect_inf(double u, double expo, double b, double c, double a):
    ''' integrand evaluated at y-> infty'''
    cdef double fn = exp(-expo*pow(u,2))/(b*pow(u,2)+1 ) * (sin(2*a*c*u)/(2*a*c*u) + sin(c*u)/(c*u))
    return fn

def integrand_sing_indirect_inf(np.ndarray[double, ndim=1] u, double expo, double b, double c, double a):
    '''handling integrationwarning exceptions at y->infty'''
    cdef Py_ssize_t xmax = u.shape[0]
    cdef Py_ssize_t x
    #make a memoryview for the numpy array - see https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html#efficient-indexing-with-memoryviews
    result = np.zeros(xmax, dtype=np.float)
    cdef double[:] result_view = result
    for x in range(xmax):
        result_view[x] = exp(-expo*pow(u[x],2))/(b*pow(u[x],2)+ 1) * (sin(2*a*c*u[x])/(2*a*c*u[x]) + sin(c*u[x])/(c*u[x]))
    return result

def I_cy_indirect_inf(double expo, double b, double c, double a):
    cdef tuple integral
    cdef double val = 0
    #cdef double val = pi/sqrt(b) * cy_scipy.erfcx(sqrt(expo/b))
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            integral = integrate.quad(integrand_indirect_inf,0,10,args=(expo, b,c, a))
            return integral[0] + val
        except integrate.IntegrationWarning:
            integral = integrate.fixed_quad(integrand_sing_indirect_inf,0,10,n=50, args=(expo, b,c, a))
            return integral[0] + val

'''semianalytical integration with erfcx'''
def integrand_inf2(double u, double expo, double b, double c):
    ''' integrand evaluated at y-> infty'''
    cdef double fn = exp(-expo*pow(u,2))/(b*pow(u,2)+1 ) * (sin(c*u)/(c*u))
    return fn

def integrand_sing_inf2(np.ndarray[double, ndim=1] u, double expo, double b, double c):
    '''handling integrationwarning exceptions at y->infty'''
    cdef Py_ssize_t xmax = u.shape[0]
    cdef Py_ssize_t x
    #make a memoryview for the numpy array - see https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html#efficient-indexing-with-memoryviews
    result = np.zeros(xmax, dtype=np.float)
    cdef double[:] result_view = result
    for x in range(xmax):
        result_view[x] = exp(-expo*pow(u[x],2))/(b*pow(u[x],2)+ 1) * (sin(c*u[x])/(c*u[x]))
    return result

def I_cy_inf2(double expo, double b, double c):
    if c == 0: c = 1E-20
    cdef tuple integral
    cdef double val = 0
    #cdef double val = pi/(2*sqrt(b)) * cy_scipy.erfcx(sqrt(expo/b))
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            integral = integrate.quad(integrand_inf2,0,10,args=(expo, b,c))
            return integral[0] + val
        except integrate.IntegrationWarning:
            integral = integrate.fixed_quad(integrand_sing_inf2,0,10,n=50, args=(expo, b,c))
            return integral[0] + val

#actual integrands
def integrand3a(double u, double expo, double b, double c):
    ''' integrand evaluated at y-> infty'''
    cdef double fn = exp(-expo*pow(u,2))/(b*pow(u,2)+1 ) * (sin(c*u)/(c*u) + 1 )
    #cdef double fn = exp(-expo*pow(u,2))/(b*pow(u,2)+1 ) * (np.sinc(c*u/pi) + 1 )
    return fn

def integrand3a_sing(np.ndarray[double, ndim=1] u, double expo, double b, double c):
    '''handling integrationwarning exceptions at y->infty'''
    cdef Py_ssize_t xmax = u.shape[0]
    cdef Py_ssize_t x
    #make a memoryview for the numpy array - see https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html#efficient-indexing-with-memoryviews
    result = np.zeros(xmax, dtype=np.float)
    cdef double[:] result_view = result
    for x in range(xmax):
        result_view[x] = exp(-expo*pow(u[x],2))/(b*pow(u[x],2)+ 1) * (sin(c*u[x])/(c*u[x]) - 1)
    return result

def I3a(double expo, double b, double c):
    if c == 0: c = 1E-20
    cdef tuple integral
    cdef double val
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            integral = integrate.quad(integrand3a,0,10,args=(expo, b,c))
            return integral[0]
        except integrate.IntegrationWarning:
            integral = integrate.fixed_quad(integrand3a_sing,0,10,n=50, args=(expo, b,c))
            val = pi/(sqrt(b)) * cy_scipy.erfcx(sqrt(expo/b))
            return integral[0] + val

def integrand3b(double u, double expo, double b, double c, double a):
    ''' integrand evaluated at y-> infty'''
    cdef double fn = exp(-expo*pow(u,2))/(b*pow(u,2)+1 ) * (sin(2*a*c*u)/(2*a*c*u) + sin(c*u)/(c*u))
    return fn

def integrand3b_sing(np.ndarray[double, ndim=1] u, double expo, double b, double c, double a):
    '''handling integrationwarning exceptions at y->infty'''
    cdef Py_ssize_t xmax = u.shape[0]
    cdef Py_ssize_t x
    #make a memoryview for the numpy array - see https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html#efficient-indexing-with-memoryviews
    result = np.zeros(xmax, dtype=np.float)
    cdef double[:] result_view = result
    for x in range(xmax):
        result_view[x] = exp(-expo*pow(u[x],2))/(b*pow(u[x],2)+ 1) * (sin(2*a*c*u[x])/(2*a*c*u[x]) + sin(c*u[x])/(c*u[x]))
    return result

def I3b(double expo, double b, double c, double a):
    cdef tuple integral
    cdef double val = 0
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            integral = integrate.quad(integrand3b,0,10,args=(expo, b,c, a))
            return integral[0]
        except integrate.IntegrationWarning:
            integral = integrate.fixed_quad(integrand3b_sing,0,10,n=50, args=(expo, b,c, a))
            #val = pi/sqrt(b) * cy_scipy.erfcx(sqrt(expo/b))
            return integral[0] + val

'''Calculate and minimize energy for y->inf'''
def E_infty(np.ndarray[double, ndim=1] x, double n, double U):
    '''Hybrid calc energy evaluated at y>>1, normalized by KE
        x = [a_r, sig_r, a_R, sig_R]
    '''
    cdef double y = 100000.
    cdef double Am = pow(1-x[2],2) *pow(x[3],2) + pow(1-2*x[0],2) *pow(x[1],2)
    cdef double Ap = pow(1-x[2],2) *pow(x[3],2) + pow(1+2*x[0],2) *pow(x[1],2)
    cdef double b = 0.5*(pow(x[2],2) + 4*pow(x[0],2))
    cdef double KE = 3./2*(1/pow(x[1],2) + 1/pow(x[3],2))
    cdef double coul = U/(x[1]*y)
    cdef double e_ph = -(1-n)*U/(2*pi)* (I_cy_direct_inf(Am/4,b,(1-2*x[0])*y*x[1]) + I_cy_direct_inf(Ap/4,b,(1+2*x[0])*y*x[1]) + 2*I_cy_indirect_inf((Am+Ap)/8,b,y*x[1],x[0]))
    cdef double E = KE + e_ph + coul
    return E

def min_Einf(tuple args):
    '''Just minimize E_inf to compare with lg sigma optimization'''
    cdef double n = args[0]
    cdef double u = args[1]
    cdef tuple bnds_inf = ((1E-5,1),(1E-1, 1E3),(1E-5,1),(1E-1,1E3)) #a_r, sig_r, a_R, sig_R
    cdef np.ndarray[double, ndim=1] guess_inf 
    cdef double alpha = (1-n)*u/2
    if alpha <10: guess_inf = np.array([1.,1.,1.,1.])
    else: guess_inf = np.array([1E-5,0.2,1E-5,0.2])
    res_inf = minimize(E_infty,guess_inf,args=(n,u),bounds=bnds_inf)
    cdef np.ndarray[double, ndim=1] m_inf = res_inf.x
    cdef double E_inf = E_infty(m_inf,n,u)
    return n,u,m_inf[0],m_inf[1],m_inf[2], m_inf[3],E_inf

def E_infty2(np.ndarray[double, ndim=1] x, double n, double U):
    '''Hybrid calc energy evaluated at y>>1, normalized by KE
        x = [a_r, sig_r, a_R, sig_R]
    '''
    cdef double y = 1E7
    cdef double Am = pow(1-x[2],2) *pow(x[3],2) + pow(1-2*x[0],2) *pow(x[1],2)
    cdef double Ap = pow(1-x[2],2) *pow(x[3],2) + pow(1+2*x[0],2) *pow(x[1],2)
    cdef double b = 0.5*(pow(x[2],2) + 4*pow(x[0],2))
    cdef double KE = 3./2*(1/pow(x[1],2) + 1/pow(x[3],2))
    cdef double e_ph = -(1-n)*U/(2*pi)* (pi/(2*sqrt(b))* (cy_scipy.erfcx(sqrt(Am/(4*b))) + cy_scipy.erfcx(sqrt(Ap/(4*b))) ) 
            #+ I_cy_inf2(Am/4,b,(1-2*x[0])*y*x[1])+ I_cy_inf2(Ap/4,b,(1+2*x[0])*y*x[1]) 
            +2*I_cy_inf2((Am+Ap)/8,b,y*x[1]*2*x[0]) )
    cdef double E = KE + e_ph
    return E

def min_Einf2(tuple args):
    cdef double n = args[0]
    cdef double u = args[1]
    cdef tuple bnds_inf = ((1E-5,1),(1E-2, 1E3),(1E-5,1),(1E-2,1E3)) #a_r, sig_r, a_R, sig_R
    #cdef np.ndarray[double, ndim=1] guess_inf = np.array([0.5,1,0.5,1])
    cdef np.ndarray[double, ndim=1] guess_inf 
    cdef double alpha = (1-n)*u/2
    if alpha <10: guess_inf = np.array([1.,1.,1.,1.])
    else: guess_inf = np.array([1E-5,0.2,1E-5,0.2])
    res_inf = minimize(E_infty2,guess_inf,args=(n,u),bounds=bnds_inf)
    cdef np.ndarray[double, ndim=1] m_inf = res_inf.x
    cdef double E_inf = E_infty2(m_inf,n,u)
    return n,u,m_inf[0],m_inf[1],m_inf[2], m_inf[3],E_inf

def E_infty3(np.ndarray[double, ndim=1] x, double n, double U):
    '''Hybrid calc energy evaluated at y>>1, normalized by KE
        x = [a_r, sig_r, a_R, sig_R]
    '''
    cdef double y = 100000
    cdef double Am = pow(1-x[2],2) *pow(x[3],2) + pow(1-2*x[0],2) *pow(x[1],2)
    cdef double Ap = pow(1-x[2],2) *pow(x[3],2) + pow(1+2*x[0],2) *pow(x[1],2)
    cdef double b = 0.5*(pow(x[2],2) + 4*pow(x[0],2))
    cdef double KE = 3./2*(1/pow(x[1],2) + 1/pow(x[3],2))
    cdef double e_ph = -(1-n)*U/(2*pi)* (I3a(Am/4,b,(1-2*x[0])*y*x[1]) + I3a(Ap/4,b,(1+2*x[0])*y*x[1]) + 2*I3b((Am+Ap)/8,b,y*x[1],x[0]))
    cdef double E = KE + e_ph
    return E

def min_Einf3(tuple args):
    '''Just minimize E_inf to compare with lg sigma optimization'''
    cdef double n = args[0]
    cdef double u = args[1]
    cdef tuple bnds_inf = ((1E-5,1),(1E-2, 1E3),(1E-5,1),(1E-2,1E3)) #a_r, sig_r, a_R, sig_R
    cdef np.ndarray[double, ndim=1] guess_inf 
    cdef double alpha = (1-n)*u/2
    if alpha <10: guess_inf = np.array([1.,1.,1.,1.])
    else: guess_inf = np.array([1E-5,0.2,1E-5,0.2])
    res_inf = minimize(E_infty3,guess_inf,args=(n,u),bounds=bnds_inf)
    cdef np.ndarray[double, ndim=1] m_inf = res_inf.x
    cdef double E_inf = E_infty3(m_inf,n,u)
    return n,u,m_inf[0],m_inf[1],m_inf[2], m_inf[3],E_inf


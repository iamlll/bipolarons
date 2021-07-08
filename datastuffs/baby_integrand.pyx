import pyximport; pyximport.install()
import numpy as np
cimport numpy as np
from cpython cimport array
from libc.math cimport sqrt, pi, exp, sin, pow, erf, erfc, floor

def exp2(double u):
    return u*u
def pow2(double u):
    return pow(u,2)
def exptup2(tuple x):
    return x[0]*x[0]
def powtup2(tuple x):
    return pow(x[0],2)

def integrand(double u,double b,double c,tuple x):
    '''
    Inputs:
        u: integration variable
        x: numpy array 
    '''
    cdef double fn = exp(-pow(u,2))/(pow(u,2) +b) * (c*(1 + 2*exp(-pow(x[0],2))) + sin(c*u)/u + 8*exp(-pow(x[0],2)/2)*sin(c*u/2)/u)
    return fn

def integrand_sing(double u,double b,double c,tuple x):
    '''
    Modified integrand to deal with singularities near u = 0
    Inputs:
        u: integration variable
        x: numpy array 
    '''
    cdef double fn = exp(-pow(u,2))/(pow(u,2)+b) * ((sin(c*u)/u - c) + 4*exp(-pow(x[0],2)/2)* (2*sin(c*u/2)/u - c))
    return fn

def integrand_inf(double u, double b, double c):
    ''' integrand evaluated at y-> infty'''
    cdef double fn = exp(-pow(u,2))/(pow(u,2)+b) * (c+ sin(c*u)/u )
    return fn

from libc.stdio cimport printf
from cython.view cimport array as cvarray
def integrand_sing_inf(np.ndarray[double, ndim=1] u, double b, double c):
    '''handling IntegrationWarning exceptions at y->infty'''
    cdef Py_ssize_t xmax = u.shape[0]
    cdef Py_ssize_t x
    #make a memoryview for the numpy array - see https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html#efficient-indexing-with-memoryviews
    result = np.zeros(xmax, dtype=np.float)
    cdef double[:] result_view = result
    for x in range(xmax):
        result_view[x] = exp(-pow(u[x],2))/(pow(u[x],2)+b) * (sin(c*u[x])/u[x] - c)

    return result

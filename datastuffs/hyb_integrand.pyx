import numpy as np
cimport numpy as np
from scipy import optimize, integrate
#from scipy.special import erf, erfc
from cpython cimport array
from libc.math cimport sqrt, pi, exp, sin, pow, erf, erfc

def integrand(np.float64 u,np.float64 b,np.float64 c,np.ndarray[np.float64, ndim=2]x):
    '''
    Inputs:
        u: integration variable
        x: numpy array 
    '''
    cdef double cu = u
    cdef double cb = b
    cdef double cc = c
    cdef double[:] cx = x #convert numpy array to C array
    cdef double fn = exp(cu*cu)/(cu*cu+cb) * (c*(1 + 2*exp(-pow(cx[0],2))) + sin(cc*cu)/cu + 8*exp(-pow(cx[0],2)/2*sin(cc*cu/2)/cu))
    return fn
    
def integrate(np.float64 b,np.float64 c,np.ndarray[np.float64, ndim=2]x):
    cdef double cu, cb, cc = u,b,c
    cdef np.ndarray integral = integrate.quad(integrand,0,10,args=(cb,cc,x))
    return integral[0]

def integrand_sing(np.float64 u,np.float64 b,np.float64 c,np.ndarray[np.float64, ndim=2] x):
    '''
    Modified integrand to deal with singularities near u = 0
    Inputs:
        u: integration variable
        x: numpy array 
    '''
    cdef double cu,cb,cc = u,b,c
    cdef double[:] cx = x #convert numpy array to C array
    cdef fn = exp(cu*cu)/(cu*cu+cb) * ((sin(cc*cu)/cu-cc) + 4*exp(-pow(cx[0],2)/2)* (2*sin(cc*cu/2)/cu-cc))
    return fn

def integrate_sing(np.float64 b,np.float64 c,np.ndarray[np.float64, ndim=2] x):
    cdef double cb, cc = b, c
    cdef np.ndarray integral = integrate.quad(integrand_sing,0,10,args=(cb,cc,x)))
    cdef double val = erfc(sqrt(cb))
    if val == 0: 
        cdef double val2 = 0
    else: 
        cdef double val2 = exp(cb)
    cdef double ana = (1+2*exp(-pow(cx[0],2)/2) + exp(-pow(cx[0],2)))*pi*cc/sqrt(cb) * val2* val
    return integral[0] + ana


'''
def i1(b,c,x,option=0):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        if option == 0:
            try:
                integral = integrate.quad(lambda u: np.exp(-u**2)/(u**2+b) * (c*(1+2*np.exp(-x[0]**2)) + \
                    np.sin(c*u)/u + 8*np.exp(-x[0]**2/2)*np.sin(c*u/2)/u),0,10)
                return integral[0]
            except integrate.IntegrationWarning:
                #print ('Uh oh: option ' + str(option) + '\tb: ' + str(b) + '\tc: ' + str(c))
                integral = integrate.quad(lambda u: np.exp(-u**2)/(u**2+b) * ((np.sin(c*u)/u-c) + \
                    4*np.exp(-x[0]**2/2)*(2*np.sin(c*u/2)/u - c)),0,10)
                val = erfc(np.sqrt(b))
                if val == 0: val2 = 0
                else: val2 = np.exp(b)
                ana = (1+2*np.exp(-x[0]**2/2) + np.exp(-x[0]**2))*np.pi*c/np.sqrt(b) * val2* val
                return integral[0] + ana

            except RuntimeWarning:
                print ('Raised! option ' + str(option) + '\tb: ' + str(b) + '\tc: ' + str(c))
        else:
            try:
                integral = integrate.quad(lambda u: np.exp(-u**2)/(u**2+b) * (c+ np.sin(c*u)/u ),0,10)
                return integral[0]

            except integrate.IntegrationWarning:
                #print ('Uh oh: option ' + str(option) + '\tx: ' + str(x) + '\tb: ' + str(b) + '\tc: ' + str(c))
                integral = integrate.fixed_quad(lambda u: np.exp(-u**2)/(u**2+b) * (np.sin(c*u)/u-c),0,10,n=50)
                val = erfc(np.sqrt(b))
                if val == 0: val2 = 0
                else: val2 = np.exp(b)
                ana = np.pi*c/np.sqrt(b) * val* val2
                return integral[0] + ana
            except RuntimeWarning:
                print ('Raised! option ' + str(option) + '\tx: ' + str(x) + '\tb: ' + str(b) + '\tc: ' + str(c))
'''

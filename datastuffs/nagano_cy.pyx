import numpy as np
cimport numpy as np
from scipy import integrate
from scipy.optimize import minimize
from libc.math cimport fabs, sqrt, pi, exp, sin, cos, pow, erf, atan, log, floor, atan2, tan
#from libc.complex cimport cexp
import warnings
cimport scipy.special.cython_special as cy_scipy

'''integrand and double integral stuff'''
def Ang_Integrand(double u, double x, double y, double sig, double a):
    cdef double numer = exp(-pow(1-a,2)*pow(x,2)/4) * cos((1-a)/2* y*x*u) + exp(-(1+pow(a,2))*pow(x,2)/4) * cos((1+a)/2* y*x*u)
    cdef double denom = 1 + pow(a*x/sig,2) + exp(-pow(a*x,2)/2) * cos(a*y*x*u)
    return pow(numer,2)/denom
    
import mpmath as mpm
def Integrate_mpm(double y, double sig, double a):
    cdef double integral = mpm.quad(lambda x,u: Ang_Integrand(u,x,y,sig,a),[0,10],[-1,1])
    return integral

def I_cy_inf(double y, double sig, double a):
    cdef double result
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            result = Integrate_mpm(y,sig,a)
            return result
        except integrate.IntegrationWarning:
            print("problem at: (y,sig,a) = (%.2f,%.2f, %.2f)" %(y,sig,a))
            return 0.

'''Calculate and minimize energy for y->inf'''
def E_infty(np.ndarray[double, ndim=1] x, double y, double n, double U):
    '''Hybrid calc energy evaluated at y>>1, normalized by KE
        x = [a,sigma/l]
    '''
    cdef double KE = 3./pow(x[1],2)
    cdef double coul = U/(y * x[1])
    cdef double e_ph = -(1-n)*U/2.* 2/(pi*x[1]) * Integrate_mpm(y,x[1],x[0])
    cdef double E = KE + e_ph + coul
    return E

def min_Einf(tuple args):
    '''Just minimize E_inf to compare with lg sigma optimization'''
    cdef double n = args[0]
    cdef double u = args[1]
    cdef tuple bnds_inf = ((1E-5,1),(1E-2, 1E3),) #a,s
    cdef double y = 5000. #y fixed at 5000
    cdef np.ndarray[double, ndim=1] guess_inf = np.array([0.5,1.])
    res_inf = minimize(E_infty,guess_inf,args=(y,n,u),bounds=bnds_inf)
    cdef np.ndarray[double, ndim=1] m_inf = res_inf.x
    cdef double E_inf = E_infty(m_inf,y,n,u)
    return n,u,m_inf[0],m_inf[1],y,E_inf

'''Calculate and minimize energy for y->inf'''
def E_inf_fixed_a(np.ndarray[double, ndim=1] x, double a, double y, double n, double U):
    '''Hybrid calc energy evaluated at y>>1, normalized by KE, for fixed values of a and y
        x = [sigma/l]
    '''
    cdef double KE = 3./pow(x[0],2)
    cdef double coul = U/(y * x[0])
    cdef double e_ph = -(1-n)*U/2.* 2/(pi*x[0]) * Integrate_mpm(y,x[0],a)
    cdef double E = KE + e_ph + coul
    return E

def minE_fixed_a(tuple args):
    '''minimize energy for y -> infty with fixed value of a, y'''
    cdef double n = args[0]
    cdef double u = args[1]
    cdef double a = args[2]
    cdef tuple bnds_inf = ((1E-2, 1E2),) #s
    cdef double y = 500. #y fixed
    cdef np.ndarray[double, ndim=1] guess_inf = np.array([1.])
    res_inf = minimize(E_inf_fixed_a,guess_inf,args=(a,y,n,u),bounds=bnds_inf)
    cdef np.ndarray[double, ndim=1] m_inf = res_inf.x
    cdef double E_inf = E_inf_fixed_a(m_inf,a,y,n,u)
    return n,u,a,m_inf[0],y,E_inf

def E_inf_rR(np.ndarray[double, ndim=1] x, double n, double U):
    '''Hybrid calc energy for independent sigma_r, sigma_R evaluated at y>>1, normalized by KE
        x=[sig_r,sig_R,a]
    '''
    cdef double KE = 3./2*(1/pow(x[0],2) + 1/pow(x[1],2))
    cdef double e_ph = -2*sqrt(2)/x[2]*(1-n)*U/2.*cy_scipy.erfcx(sqrt((pow(x[0],2) + pow(1-x[2],2)*pow(x[1],2)) / (2*pow(x[2],2)) ))
    cdef double E = KE + e_ph
    return E

def min_Einf_rR(tuple args):
    cdef double n = args[0]
    cdef double u = args[1]
    cdef tuple bnds_inf = ((1E-2,1E3), (1E-2, 1E3),(1E-5,1)) #s_r, s_R, a
    cdef np.ndarray[double, ndim=1] guess_inf = np.array([1,1,0.5])
    res_inf = minimize(E_inf_rR,guess_inf,args=(n,u),bounds=bnds_inf)
    cdef np.ndarray[double, ndim=1] m_inf = res_inf.x
    cdef double E_inf = E_inf_rR(m_inf,n,u)
    return n,u,m_inf[2],m_inf[1],m_inf[0],E_inf

'''a=1 limit'''
def Integrand(double u, double z, double y, double s):
    cdef double numer = 1 + exp(-pow(z,2)/2)/ (1+ exp(-pow(y,2)/2)) *(cos(z*y*u) + exp(-pow(y,2)/2))
    cdef double denom = pow(z,2)/pow(s,2) + numer
    return pow(numer,2)/denom

def Integral(double y, double s):
    cdef double integral = mpm.quad(lambda z,u: Integrand(u,z,y,s),[0,10],[-1,1])
    return integral
    
def E_a_1(np.ndarray[double, ndim=1] x, double n, double U):
    '''Calculate and minimize energy for a=1 case
        Inputs:
            x = [y,s] nondimensionalized elec sep dist + size
    '''
    cdef double KE = 1/pow(x[1],2) * (3 - pow(x[0],2)/2* exp(-pow(x[0],2)/2) / (1 + exp(pow(x[0],2)/2)) )
    cdef double coul = U/x[1] * (erf(x[0]/sqrt(2)) /x[0] + sqrt(2/pi)*exp(-pow(x[0],2)/2)) / (1 + exp(-pow(x[0],2)/2))
    cdef double e_ph = -(1-n)*U/2.* 2/(pi* x[1]) * Integral(x[0],x[1])
    cdef double E = KE + e_ph + coul
    return E

def min_E_a_1(tuple args):
    '''Minimize energy wrt y and s to find bipolaron minima for various values of alpha'''
    cdef double n = args[0]
    cdef double u = args[1]
    cdef tuple bnds = ((1E-5,10), (1E-3, 1E2),) #y,s
    cdef np.ndarray[double, ndim=1] guess = np.array([1.,1.])
    res = minimize(E_a_1,guess,args=(n,u),bounds=bnds)
    cdef np.ndarray[double, ndim=1] m = res.x
    cdef double E = E_a_1(m,n,u)
    return n,u,1,m[1],m[0],E

#########################################################################
'''Single integral with envelope fxn approx for large y limit, a=1, using fact that arctan(tan(x)) = x. This result is not exact!!'''

def Envelope_Integral(double s):
    def Envelope_Integrand(double z, double s):
        return 1 -pow(z/s,2) + pow(z/s,4) / sqrt(pow(1+ pow(z/s,2),2) - exp(-pow(z,2)))
    cdef double integral = integrate.quad(Envelope_Integrand,0,2*pi,args=(s,))[0]
    return integral

def E_a_1_env(np.ndarray[double, ndim=1] x, double y, double n, double U):
    '''Energy evaluated at y>>1 (fixed), normalized by KE = hw
        Inputs:
            x = [s] nondimensionalized elec sep dist + size
    '''
    cdef double KE = 3./pow(x[0],2)
    cdef double coul = U/x[0] * 1./y
    cdef double e_ph = -(1-n)*U/2.* 2/x[0]*(1/y + 2/pi * Envelope_Integral(x[0]))
    cdef double E = KE + e_ph + coul
    return E

def min_E_a_1_env(tuple args):
    '''Minimize energy in Nagano formulation for a->1, fixed y-> infty'''
    cdef double n = args[0]
    cdef double u = args[1]
    cdef double y = 5000
    cdef tuple bnds = ((1E-2, 1E2),) #s
    cdef np.ndarray[double, ndim=1] guess = np.array([1.])
    res = minimize(E_a_1_env,guess,args=(y,n,u),bounds=bnds)
    cdef np.ndarray[double, ndim=1] m = res.x
    cdef double E_inf = E_a_1_env(m,y,n,u)
    return n,u,1,m[0],y,E_inf

#############################################################################################################################
'''Minimize energy at y->inf applying long wavelength cutoff z_c and adding on analytical short wavelength asymptote: relevant integrals'''
    
import mpmath as mpm
def Integral_ln_inf(double y, double s, double a, double z_c, int option):
    def Integrand1(double u, double x, double y, double s, double a):
        cdef double numer = exp(-pow(1-a,2)*pow(x,2)/4) * cos((1-a)/2* y*x*u) + exp(-(1+pow(a,2))*pow(x,2)/4) * cos((1+a)/2* y*x*u)
        cdef double denom = 1 + pow(a*x,2)* exp(-2*s) + exp(-pow(a*x,2)/2) * cos(a*y*x*u)
        return pow(numer,2)/denom

    def Integrand2(double x, double y, double s, double a): #a_c < a< 1
        return exp(-pow(1-a,2)*pow(x,2)/2) / (1 + pow(a*x,2)*exp(-2*s)) * (1+ sin((1-a)* y*x) / ((1-a)* y*x) )

    def Integrand3(double u, double z, double y, double s): #a=1
        cdef double numer = 1 + exp(-pow(z,2)/2) *cos(z*y*u) 
        cdef double denom = pow(z,2)*exp(-2*s) + numer
        return pow(numer,2)/denom

    cdef double integral
    if option == 1: #For a < a_c
        return 2*mpm.quad(lambda x,u: Integrand1(u,x,y,s,a),[0,z_c],[0,1])
        #return mpm.quad(lambda x,u: Integrand1(u,x,y,s,a),[0,z_c],[-1,1])
    elif option == 2: #For a_c <= a < 1
        return integrate.quad(Integrand2, z_c, np.inf, args=(y,s,a))[0] 
    elif option == 3: #For a=1
        return 2*mpm.quad(lambda z,u: Integrand3(u,z,y,s),[0,z_c],[0,1])
        #return mpm.quad(lambda z,u: Integrand3(u,z,y,s),[0,z_c],[-1,1])
    else: return 0.

def TestEnv(double y, double s, double a, int option):
    def Integrand(double u, double x, double y, double s, double a):
        cdef double numer = exp(-pow(1-a,2)*pow(x,2)/4) * cos((1-a)/2* y*x*u) + exp(-(1+pow(a,2))*pow(x,2)/4) * cos((1+a)/2* y*x*u)
        cdef double denom = 1 + pow(a*x,2)* exp(-2*s) + exp(-pow(a*x,2)/2) * cos(a*y*x*u)
        return pow(numer,2)/denom
    def Envelope(double x, double s, double a):
        return pow(exp(-pow(1-a,2)*pow(x,2)/4) + exp(-(1+pow(a,2))*pow(x,2)/4) ,2)/ (1 + pow(a*x,2)* exp(-2*s) + exp(-pow(a*x,2)/2) )
    #for a=1, replace angular integral by 0.5*(env + integrand|_{u=2pi/(1+a)})?
    def kIntegrand_a1(double x, double s):
        return 0.5*(pow(1 + exp(-pow(x,2)/2) ,2)/ (1 + pow(x,2)* exp(-2*s) + exp(-pow(x,2)/2) ) + pow(1 - exp(-pow(x,2)/2) ,2)/ (1 + pow(x,2)* exp(-2*s) - exp(-pow(x,2)/2) ))   
    if option == 1: return mpm.quad(lambda x,u: Integrand(u,x,y,s,a),[0,np.inf],[0,1])
    elif option == 2: return integrate.quad(Envelope, 0, np.inf, args=(s,a))[0]
    else: return integrate.quad(kIntegrand_a1,0,np.inf,args=(s,))[0]

def zIntegrand(double z, double y, double s, double a, int option):
    cdef double A = (1-exp(-0.5*pow(a*z,2)) + pow(a*z,2)*exp(-2*s)) / sqrt(1-exp(-pow(a*z,2)) + 2*pow(a*z,2)*exp(-2*s) + pow(a*z,4)*exp(-4*s))
    cdef double B = 0.5*a*y*z
    cdef double trig = pi* floor(z*B/pi) + atan2(1, 1/(A*tan(B))) #replaces arctan(A*tan(B))
    return exp(-0.5*(1-a)*pow(z,2)) + (exp(-pow(1-a,2)*pow(z,2)/2) + exp(-(1+pow(a,2))*pow(z,2)/2) + 2*exp(-0.5*(1-a)*pow(z,2)) * (1+pow(a*z,2)*exp(-2*s)) ) / (a*y*z* sqrt(1-exp(-pow(a*z,2)) + 2*pow(a*z,2)*exp(-2*s) + pow(a*z,4)*exp(-4*s))) * trig

def TestZInt(double y, double s, double a, double n, double U, double z_c):
    def Integrand(double z, double y, double s, double a):
        cdef double A = (1-exp(-0.5*pow(a*z,2)) + pow(a*z,2)*exp(-2*s)) / sqrt(1-exp(-pow(a*z,2)) + 2*pow(a*z,2)*exp(-2*s) + pow(a*z,4)*exp(-4*s))
        cdef double B = 0.5*a*y*z
        cdef double trig = pi* floor(z*B/pi) + atan2(1, 1/(A*tan(B))) #replaces arctan(A*tan(B))
        return exp(-0.5*(1-a)*pow(z,2)) + (exp(-pow(1-a,2)*pow(z,2)/2) + exp(-(1+pow(a,2))*pow(z,2)/2) + 2*exp(-0.5*(1-a)*pow(z,2)) * (1+pow(a*z,2)*exp(-2*s)) ) / (a*y*z* sqrt(1-exp(-pow(a*z,2)) + 2*pow(a*z,2)*exp(-2*s) + pow(a*z,4)*exp(-4*s))) * trig
    return -2*(1-n)*U*exp(-s)/pi* mpm.quad(lambda z: Integrand(z,y,s,a),[0,np.inf])

def Eph_inf_ln(double s, double a, double y, double n, double U, double z_c, double a_c):
    if a == 0.: 
        return -(1-n)*U*exp(-s) * (sqrt(2/pi) + 1/y) 
    elif (a>0 and a < a_c): return -(1-n)*U* exp(-s)/pi * Integral_ln_inf(y,s,a, np.inf,1) 
    elif (a_c <= a and a < 1): return -(1-n)*U* exp(-s)/pi* (Integral_ln_inf(y,s, a, z_c,1) + Integral_ln_inf(y, s,a,z_c,2))
    else: return - (1-n)*U * (1- 2*atan(z_c* exp(-s))/pi + exp(-s)/pi * Integral_ln_inf(y,s, 1.,z_c,3)) 

######################################################################################################################
'''
Energy and minimization for y->inf using logarithmic variable s = log(sig). 
Since E_inf seems only to favor a=0 and a=1, just compare these two values to get the min energy at electron sep dist y->inf
'''
def E_afix_inf(np.ndarray[double, ndim=1] x, double a, double y, double n, double U, double z_c, double a_c):
    '''Energy evaluated at y>>1 (fixed), normalized by KE = hw
        Inputs:
            x = [s] nondimensionalized elec size s = log(sig)
    '''
    cdef double KE = 3*exp(-2*x[0]) 
    cdef double coul = U*exp(-x[0]) * 1./y
    cdef double e_ph = Eph_inf_ln(x[0],a, y, n, U, z_c,a_c)
    return KE + e_ph

def min_E_afix_inf(tuple args):
    '''Minimize energy in Nagano formulation while keeping a fixed, fixed y-> infty, holding a fixed'''
    cdef double n = args[0]
    cdef double u = args[1]
    cdef double z_c = args[2]
    cdef double a_c = args[3]
    cdef double a = args[4]
    cdef double y = 500.
    cdef tuple bnds = ((-2.5,0),) #s
    cdef np.ndarray[double, ndim=1] guess = np.array([-1.])
    if (1-n)*u/2 > 9.5:
        bnds = ((-2.5,-0.5),) #s
        if (1-n)*u/2 > 15 and (1-n)*u/2 < 25:
            guess = np.array([-2.])
        elif (1-n)*u/2 >= 25:
            bnds = ((-5,-2),) #s
            guess = np.array([-4.]) #s
        res = minimize(E_afix_inf,guess,args=(a,y,n,u, z_c, a_c),bounds=bnds)
        return n,u,a,exp(res.x[0]),y,E_afix_inf(res.x,a,y,n,u,z_c, a_c)
    elif (1-n)*u/2 < 7:
        bnds = ((2,5),) #s
        guess = np.array([5.])
        res = minimize(E_afix_inf,guess,args=(1.,y,n,u, z_c, a_c),bounds=bnds)
        return n,u,1.,exp(res.x[0]),y,E_afix_inf(res.x,1.,y,n,u,z_c, a_c)
    else: #near the 1st order transition from weak/strong coupling
        res = minimize(E_afix_inf,guess,args=(a,y,n,u, z_c, a_c),bounds=bnds)
        bnds = ((2,5),) #s
        guess = np.array([5.])
        res2 = minimize(E_afix_inf,guess,args=(1.,y,n,u, z_c,a_c),bounds=bnds)
        if E_afix_inf(res.x,a,y,n,u,z_c, a_c) > E_afix_inf(res2.x,1.,y,n,u,z_c, a_c): 
            return n,u,1.,exp(res2.x[0]),y,E_afix_inf(res2.x,1.,y,n,u,z_c, a_c)
        else:
            return n,u,a, exp(res.x[0]),y,E_afix_inf(res.x,a,y,n,u,z_c, a_c)

#Treat a as a variational parameter
def E_avar_inf(np.ndarray[double, ndim=1] x, double y, double n, double U, double z_c, double a_c):
    '''Energy evaluated at y>>1 (fixed), normalized by KE = hw
        Inputs:
            x = [s,a] nondimensionalized elec size s = log(sig)
    '''
    cdef double KE = 3*exp(-2*x[0]) 
    cdef double coul = U*exp(-x[0]) * 1./y
    cdef double e_ph = Eph_inf_ln(x[0],x[1], y, n, U, z_c,a_c)
    return KE + e_ph

def min_E_avar_inf(tuple args):
    '''Minimize energy in Nagano formulation while keeping a as a variational param, fixed y-> infty'''
    cdef double n = args[0]
    cdef double u = args[1]
    cdef double z_c = args[2]
    cdef double a_c = args[3]
    cdef double y = args[4]
    cdef tuple bnds = ((-2.5,0),(0,1),) #s,a
    cdef np.ndarray[double, ndim=1] guess = np.array([-1.,0.002])
    if (1-n)*u/2 > 9.5:
        bnds = ((-2.5,-0.5),(0,1),) #s,a
        guess = np.array([-1,0.001])
        if (1-n)*u/2 > 15 and (1-n)*u/2 < 25:
            guess = np.array([-2.,1E-4])
        elif (1-n)*u/2 >= 25:
            bnds = ((-5,-2),(0,0.3),) #s, a
            guess = np.array([-4.,0.]) #s, a
        res = minimize(E_avar_inf,guess,args=(y,n,u, z_c, a_c),bounds=bnds)
        return n,u,res.x[1], exp(res.x[0]),y,E_avar_inf(res.x,y,n,u,z_c, a_c)
    elif (1-n)*u/2 < 7:
        bnds = ((2,5),) #s
        guess = np.array([5.])
        res = minimize(E_afix_inf,guess,args=(1.,y,n,u, z_c, a_c),bounds=bnds)
        return n,u,1.,exp(res.x[0]),y,E_afix_inf(res.x,1.,y,n,u,z_c, a_c)
    else: #near the 1st order transition from weak/strong coupling
        res = minimize(E_avar_inf,guess,args=(y,n,u, z_c, a_c),bounds=bnds)
        bnds = ((2,5),) #s
        guess = np.array([5.])
        res2 = minimize(E_afix_inf,guess,args=(1.,y,n,u, z_c,a_c),bounds=bnds)
        if E_avar_inf(res.x,y,n,u,z_c, a_c) > E_afix_inf(res2.x,1.,y,n,u,z_c, a_c): 
            return n,u,1.,exp(res2.x[0]),y,E_afix_inf(res2.x,1.,y,n,u,z_c, a_c)
        else:
            return n,u,res.x[1], exp(res.x[0]),y,E_avar_inf(res.x,y,n,u,z_c, a_c)

def min_E_inf(tuple args):
    '''Minimize energy in Nagano formulation while fixing a to be either 0 or 1 (this is what various E(a) plots seem to indicate), fixed y-> infty'''
    cdef double n = args[0]
    cdef double u = args[1]
    cdef double z_c = args[2]
    cdef double a_c = args[3]
    cdef double y = args[4]
    cdef tuple bnds = ((-2.5,0),) #s
    cdef np.ndarray[double, ndim=1] guess = np.array([-1.])
    if (1-n)*u/2 > 9.5:
        bnds = ((-2.5,-0.5),) #s
        guess = np.array([-1.])
        if (1-n)*u/2 > 15 and (1-n)*u/2 < 25:
            guess = np.array([-2.])
        elif (1-n)*u/2 >= 25:
            bnds = ((-5,-2),) #s
            guess = np.array([-4.]) #s
        res = minimize(E_afix_inf,guess,args=(0.,y,n,u, z_c, a_c),bounds=bnds)
        return n,u,0.,exp(res.x[0]),y,E_afix_inf(res.x,0.,y,n,u,z_c, a_c)
    elif (1-n)*u/2 < 7:
        bnds = ((2,5),) #s
        guess = np.array([5.])
        res = minimize(E_afix_inf,guess,args=(1.,y,n,u, z_c, a_c),bounds=bnds)
        return n,u,1.,exp(res.x[0]),y,E_afix_inf(res.x,1.,y,n,u,z_c, a_c)
    else: #near the 1st order transition from weak/strong coupling
        res = minimize(E_afix_inf,guess,args=(0.,y,n,u, z_c, a_c),bounds=bnds)
        bnds = ((2,5),) #s
        guess = np.array([5.])
        res2 = minimize(E_afix_inf,guess,args=(1.,y,n,u, z_c,a_c),bounds=bnds)
        if E_afix_inf(res.x,0.,y,n,u,z_c, a_c) > E_afix_inf(res2.x,1.,y,n,u,z_c, a_c): 
            return n,u,1.,exp(res2.x[0]),y,E_afix_inf(res2.x,1.,y,n,u,z_c, a_c)
        else:
            return n,u,0., exp(res.x[0]),y,E_afix_inf(res.x,0.,y,n,u,z_c, a_c)

########################################################################################################################
'''Various integrals for optimization'''

'''Minimize energy for finite y allowing a,y,sigma/l to vary'''    
import mpmath as mpm
def Integral1(double y, double sig, double a, double z_c):
    '''For a < a_c'''
    def Integrand(double u, double x, double y, double sig, double a):
        cdef double numer = exp(-pow(1-a,2)*pow(x,2)/4) * (cos((1-a)/2* y*x*u) + exp(-pow(y,2)/2) ) + exp(-(1+pow(a,2))*pow(x,2)/4) * (cos((1+a)/2* y*x*u)+ exp(-pow(y,2)/2) )
        cdef double denom = 1 + pow(a*x/sig,2) + exp(-pow(a*x,2)/2) / (1+ exp(-pow(y,2)/2) ) * (cos(a*y*x*u)+ exp(-pow(y,2)/2) )
        return pow(numer,2)/denom
    cdef double integral = mpm.quad(lambda x,u: Integrand(u,x,y,sig,a),[0,z_c],[-1,1])
    return integral

def Integral2(double y, double sig, double a, double z_c):
    '''For a_c <= a < 1'''
    def Integrand(double x, double y, double sig, double a):
        return exp(-pow(1-a,2)*pow(x,2)/2) / (1 + pow(a*x/sig,2)) * (1+ 2*exp(-pow(y,2)) + 4*exp(-pow(y,2)/2) *sin(0.5*(1-a)* y*x) / (0.5*(1-a)* y*x) + sin((1-a)* y*x) / ((1-a)* y*x) )
    cdef double integral = integrate.quad(Integrand, z_c, np.inf, args=(y,sig,a))[0] 
    return integral

def Integral3(double y, double s, double z_c):
    '''For a = 1'''
    def Integrand(double u, double x, double y, double s):
        cdef double numer = 1+ exp(-pow(x,2)/2) / (1+ exp(-pow(y,2)/2)) * (cos(y*x*u)+ exp(-pow(y,2)/2) )
        cdef double denom = pow(x,2)/pow(s,2) + numer
        return pow(numer,2)/denom
    cdef double integral = mpm.quad(lambda z,u: Integrand(u,z,y,s),[0,z_c],[-1,1])
    return integral

def Eph(double s, double a, double y, double n, double U, double z_c, double a_c):
    cdef double e_ph
    if a < a_c: e_ph = -(1-n)*U/(pi* s* pow(1+ exp(-pow(y,2)/2),2) ) * Integral1(y,s,a, np.inf) 
    elif (a_c <= a and a < 1): e_ph = -(1-n)*U/(pi* s* pow(1+ exp(-pow(y,2)/2),2))  * (Integral1(y,s, a, z_c) + Integral2(y, s,a,z_c))
    else: e_ph = - (1-n)*U * (1- 2*atan(z_c/s)/pi + 1/(pi* s) * Integral3(y,s, z_c)) 
    return e_ph

def Eph_warn(double s, double a, double y, double n, double U, double z_c, double a_c):
    cdef double e_ph
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            return Eph(s,a,y,n,U,z_c,a_c)
        except integrate.IntegrationWarning:
            print("problem at: (y,sig,a,n,U) = (%.2f,%.2f, %.2f,%.2f,%.2f)" %(y,s,a,n,U))
            return 0.

'''Try optimizing taking sigma/l -> e^s where s = ln(sigma/l) so the energy landscape is smoother'''
def Integral_ln(double y, double s, double a, double z_c, int option):
    def Integrand1(double u, double x, double y, double s, double a):
        cdef double numer = exp(-pow(1-a,2)*pow(x,2)/4) * (cos((1-a)/2* y*x*u) + exp(-pow(y,2)/2) ) + exp(-(1+pow(a,2))*pow(x,2)/4) * (cos((1+a)/2* y*x*u)+ exp(-pow(y,2)/2) )
        cdef double denom = 1 + pow(a*x,2)* exp(-2*s) + exp(-pow(a*x,2)/2) / (1+ exp(-pow(y,2)/2) ) * (cos(a*y*x*u)+ exp(-pow(y,2)/2) )
        return pow(numer,2)/denom

    def Integrand2(double x, double y, double s, double a):
        return exp(-pow(1-a,2)*pow(x,2)/2) / (1 + pow(a*x,2)*exp(-2*s)) * (1+ 2*exp(-pow(y,2)) + 4*exp(-pow(y,2)/2) *sin(0.5*(1-a)* y*x) / (0.5*(1-a)* y*x) + sin((1-a)* y*x) / ((1-a)* y*x) )
    
    def Integrand3(double u, double x, double y, double s):
        cdef double numer = 1+ exp(-pow(x,2)/2) / (1+ exp(-pow(y,2)/2)) * (cos(y*x*u)+ exp(-pow(y,2)/2) )
        cdef double denom = pow(x,2) * exp(-2*s) + numer
        return pow(numer,2)/denom

    if option == 1: #For a < a_c
        return 2*mpm.quad(lambda x,u: Integrand1(u,x,y,s,a),[0,z_c],[0,1])
    elif option == 2: #For a_c <= a < 1
        return integrate.quad(Integrand2, z_c, np.inf, args=(y,s,a))[0] 
    elif option == 3: #For a=1
        return 2*mpm.quad(lambda z,u: Integrand3(u,z,y,s),[0,z_c],[0,1])
    else: return 0.

def Eph_ln(double s, double a, double y, double n, double U, double z_c, double a_c):
    if a == 0.: 
        return -(1-n)*U*exp(-s) /pow(1+ exp(-pow(y,2)/2),2) * sqrt(2/pi) * (1 + 2*exp(-pow(y,2)) + 1/y* sqrt(pi/2) * (erf(y/sqrt(2)) + 8*exp(-0.5*pow(y,2)) * erf(y/(2*sqrt(2))) ) ) 
    elif (a>0 and a < a_c): return -(1-n)*U* exp(-s)/(pi* pow(1+ exp(-pow(y,2)/2),2) ) * Integral_ln(y,s,a, np.inf,1) 
    elif (a_c <= a and a < 1): return -(1-n)*U* exp(-s)/(pi* pow(1+ exp(-pow(y,2)/2),2))  * (Integral_ln(y,s, a, z_c,1) + Integral_ln(y, s,a,z_c,2))
    else: return - (1-n)*U * (1- 2*atan(z_c* exp(-s))/pi + exp(-s)/pi * Integral_ln(y,s, 1.,z_c,3)) 

##########################################################################################################################################################

def E_bip_ln(np.ndarray[double, ndim=1] x, double n, double U, double z_c, double a_c):
    '''bipolaron energy normalized by KE = hw; written in terms of s = ln (sig) (base e)
        Inputs:
            x = [y,s,a] nondimensionalized elec sep dist, size, el-ph boost param
    '''
    cdef double KE = exp(-2*x[1]) * (3 - exp(2*x[0])/2* exp(-exp(2*x[0])/2) / (1 + exp(exp(x[0]*2)/2)) )
    cdef double coul = U* exp(-x[1]) * (erf(exp(x[0])/sqrt(2)) *exp(-x[0]) + sqrt(2/pi)*exp(-exp(x[0]*2)/2)) / (1 + exp(-exp(x[0]*2)/2))
    cdef double e_ph = Eph_ln(x[1],x[2], exp(x[0]), n, U, z_c,a_c)
    return KE + e_ph + coul

def min_E_bip_ln(tuple args):
    '''Minimize energy in Nakano formulation allowing Y = ln(y), s = ln(sig), a to vary'''
    cdef double n = args[0]
    cdef double u = args[1]
    cdef double z_c = args[2]
    cdef double a_c = args[3]
    cdef tuple bnds = ((-3,2),(-1,5),(0,1),) #y, s, a
    cdef np.ndarray[double, ndim=1] guess = np.array([0.,2.,0.5])
    if (1-n)*u/2 > 8:
        bnds = ((-3,2),(-1,3),(0,0.3),) #y, s, a
        guess = np.array([0.,1.,0.001])
    elif (1-n)*u/2 < 3:
        bnds = ((-3,2),(2,5),(0.8,1),) #y,s,a
        guess = np.array([0.,5.,1.])
    res = minimize(E_bip_ln,guess,args=(n,u, z_c, a_c),bounds=bnds)
    cdef np.ndarray[double, ndim=1] m = res.x
    cdef double E = E_bip_ln(m,n,u,z_c, a_c)
    return n,u,m[2],exp(m[1]),exp(m[0]),E

'''bipolaron energy with a fixed'''
def E_bip_afix(np.ndarray[double, ndim=1] x, double a, double n, double U, double z_c, double a_c):
    '''bipolaron energy normalized by KE = hw with fixed a
        Inputs:
            x = [y,s] nondimensionalized elec sep dist + size
    '''
    cdef double KE = exp(-2*x[1]) * (3 - exp(2*x[0])/2* exp(-exp(2*x[0])/2) / (1 + exp(exp(x[0]*2)/2)) )
    cdef double coul = U* exp(-x[1]) * (erf(exp(x[0])/sqrt(2)) *exp(-x[0]) + sqrt(2/pi)*exp(-exp(x[0]*2)/2)) / (1 + exp(-exp(x[0]*2)/2))
    cdef double e_ph = Eph_ln(x[1],a, exp(x[0]), n, U, z_c,a_c)
    return KE + e_ph + coul

def min_E_bip_afix(tuple args):
    '''Minimize energy in Nakano formulation allowing y, s to vary, a kept fixed'''
    cdef double n = args[0]
    cdef double u = args[1]
    cdef double z_c = args[2]
    cdef double a_c = args[3]
    cdef double a = args[4]
    cdef tuple bnds = ((-3,2),(-1,5),) #y, s
    cdef np.ndarray[double, ndim=1] guess = np.array([0.,2.])
    if (1-n)*u/2 > 8:
        bnds = ((-3,2),(-1,3),) #y, s
        guess = np.array([0.,1.])
    elif (1-n)*u/2 < 3:
        bnds = ((-3,2),(2,5),) #y,s
        guess = np.array([0.,5.])
    res = minimize(E_bip_afix,guess,args=(a, n,u, z_c, a_c),bounds=bnds)
    cdef np.ndarray[double, ndim=1] m = res.x
    cdef double E = E_bip_afix(m,a, n,u,z_c, a_c)
    return n,u,a,m[1],m[0],E

def min_E_bip_ln2(tuple args):
    '''Minimize energy in Nakano formulation allowing Y = ln(y), s = ln(sig), a to vary. Set weak coupling result to a=1 so don't have to optimize a. For alpha near the phase transition need to check whether a=1 or something else'''
    cdef double n = args[0]
    cdef double u = args[1]
    cdef double z_c = args[2]
    cdef double a_c = args[3]
    cdef tuple bnds = ((-2,2),(-2.5,0),(0,1),) #y, s, a
    cdef np.ndarray[double, ndim=1] guess = np.array([0.,-1,0.1])
    if (1-n)*u/2 > 9:
        if n == 0 and (1-n)*u/2 > 25:
            bnds = ((-2,2),(-5,-2),(0,0.3),) #y, s, a
            guess = np.array([0.,-4.,0.001])
            res = minimize(E_bip_ln,guess,args=(n,u, z_c, a_c),bounds=bnds)
            return n,u,res.x[2],exp(res.x[1]),exp(res.x[0]),E_bip_ln(res.x,n,u,z_c, a_c)
        elif n > 0 and (1-n)*u/2 > 25:
            #need to compare y->inf strong coupling result with y = finite result and take the lower one
            bnds = ((-2,2),(-5,-2),(0,0.3),) #y, s, a
            guess = np.array([0.,-4.,0.001])
            res = minimize(E_bip_ln,guess,args=(n,u, z_c, a_c),bounds=bnds)
            bnds = ((-5,-2),) #s
            guess = np.array([-4.]) #s
            res2 = minimize(E_afix_inf,guess,args=(0.,500.,n,u, z_c, a_c),bounds=bnds) #treating y = 500 as "infty"
            if E_bip_ln(res.x,n,u,z_c, a_c) > E_afix_inf(res2.x,0.,500.,n,u,z_c, a_c): 
                return n,u,0.,exp(res2.x[0]),500.,E_afix_inf(res2.x,0.,500.,n,u,z_c, a_c)
            else:
                return n,u,res.x[2],exp(res.x[1]),exp(res.x[0]),E_bip_ln(res.x,n,u,z_c, a_c)
        else:             
            bnds = ((-2,2),(-2,0),(0,0.3),) #y, s, a
            guess = np.array([0.,-1,0.01])
            res = minimize(E_bip_ln,guess,args=(n,u, z_c, a_c),bounds=bnds)
            return n,u,res.x[2],exp(res.x[1]),exp(res.x[0]),E_bip_ln(res.x,n,u,z_c, a_c)
    elif (1-n)*u/2 < 7:
        bnds = ((-3,1),(2,5),) #y,s
        guess = np.array([0.,5.])
        res = minimize(E_bip_afix,guess,args=(n,u, z_c, a_c,1.),bounds=bnds)
        bnds = ((2,5),) #s
        guess = np.array([5.])
        res2 = minimize(E_afix_inf,guess,args=(1.,500.,n,u, z_c, a_c),bounds=bnds)
        if E_bip_afix(res.x,1.,n,u,z_c, a_c) > E_afix_inf(res2.x,1.,500.,n,u,z_c, a_c): 
            return n,u,1.,exp(res2.x[0]),500.,E_afix_inf(res2.x,1.,500.,n,u,z_c, a_c)
        else:
            return n,u,1.,exp(res.x[1]),exp(res.x[0]),E_bip_afix(res.x,1.,n,u,z_c, a_c)
    else:
        res = minimize(E_bip_ln,guess,args=(n,u, z_c, a_c),bounds=bnds)
        bnds = ((2,5),) #s
        guess = np.array([5.])
        res2 = minimize(E_afix_inf,guess,args=(1.,500.,n,u, z_c, a_c),bounds=bnds)
        if E_bip_ln(res.x,n,u,z_c, a_c) > E_afix_inf(res2.x,1.,500.,n,u,z_c, a_c): 
            return n,u,1.,exp(res2.x[0]),500.,E_afix_inf(res2.x,1.,500.,n,u,z_c, a_c)
        else:
            return n,u,res.x[2],exp(res.x[1]),exp(res.x[0]),E_bip_ln(res.x,n,u,z_c, a_c)

'''bipolaron energy with y fixed'''
def E_bip_yfix_ln(np.ndarray[double, ndim=1] x, double y, double n, double U, double z_c, double a_c):
    '''bipolaron energy normalized by KE = hw with fixed y
        Inputs:
            x = [a,s] nondimensionalized elec size
            s = log(sigma/l)
    '''
    cdef double KE = exp(-2*x[1]) * (3 - pow(y,2)/2* exp(-pow(y,2)/2) / (1 + exp(pow(y,2)/2)) )
    cdef double coul = U* exp(-x[1]) * (erf(y/sqrt(2)) /y + sqrt(2/pi)*exp(-pow(y,2)/2)) / (1 + exp(-pow(y,2)/2))
    cdef double e_ph = Eph_ln(x[1],x[0], y, n, U, z_c,a_c)
    return KE + e_ph + coul

def min_E_bip_yfix_ln(tuple args):
    '''Minimize energy in Nakano formulation allowing s and a to vary, y kept fixed'''
    cdef double n = args[0]
    cdef double u = args[1]
    cdef double z_c = args[2]
    cdef double a_c = args[3]
    cdef double y = args[4]
    cdef tuple bnds = ((0,1),(-2.5,0),) #a,s
    cdef np.ndarray[double, ndim=1] guess = np.array([0.1,-1.])

    if (1-n)*u/2 > 9:
        bnds = ((0,0.3),(-2,0),) #a, s
        guess = np.array([0.001,-1.])
        if (1-n)*u/2 > 25:
            bnds = ((0,0.3),(-5,-2),) #a, s
            guess = np.array([0.,-4.])
        res = minimize(E_bip_yfix_ln,guess,args=(y,n,u, z_c, a_c),bounds=bnds)
        return n,u,res.x[0],exp(res.x[1]),y,E_bip_yfix_ln(res.x,y,n,u,z_c, a_c)
    elif (1-n)*u/2 < 7:
        bnds = ((2,5),) #s
        guess = np.array([5.])
        res = minimize(E_bip_ayfix,guess,args=(y,1.,n,u, z_c, a_c),bounds=bnds)
        return n,u,1.,exp(res.x[0]),y,E_bip_ayfix(res.x,y,1.,n,u,z_c, a_c)
    else:
        res = minimize(E_bip_yfix_ln,guess,args=(y,n,u, z_c, a_c),bounds=bnds)
        bnds = ((2,5),) #s
        guess = np.array([5.])
        res2 = minimize(E_bip_ayfix,guess,args=(y,1.,n,u, z_c, a_c),bounds=bnds)
        if E_bip_yfix_ln(res.x,y,n,u,z_c, a_c) > E_bip_ayfix(res2.x,y,1.,n,u,z_c, a_c): 
            return n,u,1.,exp(res2.x[0]),y,E_bip_ayfix(res2.x,y,1.,n,u,z_c, a_c)
        else:
            return n,u,res.x[0],exp(res.x[1]),y,E_bip_yfix_ln(res.x,y,n,u,z_c, a_c)

'''bipolaron energy with y and a fixed'''
def E_bip_ayfix(np.ndarray[double, ndim=1] x, double y, double a, double n, double U, double z_c, double a_c):
    '''bipolaron energy normalized by KE = hw with fixed a
        Inputs:
            x = [s] nondimensionalized elec size
    '''
    cdef double KE = exp(-x[0]*2) * (3 - pow(y,2)/2* exp(-pow(y,2)/2) / (1 + exp(pow(y,2)/2)) )
    cdef double coul = U*exp(-x[0]) * (erf(y/sqrt(2)) /y + sqrt(2/pi)*exp(-pow(y,2)/2)) / (1 + exp(-pow(y,2)/2))
    cdef double e_ph = Eph_ln(x[0],a, y, n, U, z_c,a_c)
    return KE + e_ph + coul

def min_E_bip_ayfix(tuple args):
    '''Minimize energy in Nakano formulation allowing s=log (sig) to vary, a and y kept fixed'''
    cdef double n = args[0]
    cdef double u = args[1]
    cdef double z_c = args[2]
    cdef double a_c = args[3]
    cdef double a = args[4]
    cdef double y = args[5]
    cdef tuple bnds = ((-1,5),) #s
    cdef np.ndarray[double, ndim=1] guess = np.array([0.])
    res = minimize(E_bip_ayfix,guess,args=(y, a, n,u, z_c, a_c),bounds=bnds)
    cdef np.ndarray[double, ndim=1] m = res.x
    cdef double E = E_bip_ayfix(m,y, a, n,u,z_c, a_c)
    return n,u,a,m[0],y,E

def E_bip_sfix(np.ndarray[double, ndim=1] x, double s, double n, double U, double z_c, double a_c):
    '''bipolaron energy normalized by KE = hw; written in terms of s = ln (sig) (base e)
        Inputs:
            x = [y,a] nondimensionalized elec sep dist, size, el-ph boost param
    '''
    cdef double KE = exp(-2*s) * (3 - exp(2*x[0])/2* exp(-exp(2*x[0])/2) / (1 + exp(exp(x[0]*2)/2)) )
    cdef double coul = U* exp(-s) * (erf(exp(x[0])/sqrt(2)) *exp(-x[0]) + sqrt(2/pi)*exp(-exp(x[0]*2)/2)) / (1 + exp(-exp(x[0]*2)/2))
    cdef double e_ph = Eph_ln(s,x[1], exp(x[0]), n, U, z_c,a_c)
    return KE + e_ph + coul

def E_bip_asfix(np.ndarray[double, ndim=1] x, double s, double a, double n, double U, double z_c, double a_c):
    '''bipolaron energy normalized by KE = hw; written in terms of s = ln (sig) (base e)
        Inputs:
            x = [y] nondimensionalized elec sep dist
    '''
    cdef double KE = exp(-2*s) * (3 - exp(2*x[0])/2* exp(-exp(2*x[0])/2) / (1 + exp(exp(x[0]*2)/2)) )
    cdef double coul = U* exp(-s) * (erf(exp(x[0])/sqrt(2)) *exp(-x[0]) + sqrt(2/pi)*exp(-exp(x[0]*2)/2)) / (1 + exp(-exp(x[0]*2)/2))
    cdef double e_ph = Eph_ln(s,a, exp(x[0]), n, U, z_c,a_c)
    return KE + e_ph + coul

def min_E_bip_sfix(tuple args):
    '''Minimize energy in Nakano formulation allowing Y = ln(y), s = ln(sig), a to vary. Set weak coupling result to a=1 so don't have to optimize a. For alpha near the phase transition need to check whether a=1 or something else'''
    cdef double n = args[0]
    cdef double u = args[1]
    cdef double z_c = args[2]
    cdef double a_c = args[3]
    cdef double sig = args[4]
    cdef tuple bnds = ((-3,2),(0,1),) #y, a
    cdef np.ndarray[double, ndim=1] guess = np.array([0.,0.1])
    res = minimize(E_bip_sfix,guess,args=(log(sig),n,u, z_c, a_c),bounds=bnds)
    return n,u,res.x[1],sig,exp(res.x[0]),E_bip_sfix(res.x,log(sig),n,u,z_c, a_c)

################################################################################################################################
'''generate maps of E(a,sigma) at a couple fixed values of y=0, 10, "infty" to help better understand contradictions in E(a) and E(y) results'''
def E_bip_aysfix(tuple x):
    '''bipolaron energy normalized by KE = hw with fixed a
        Inputs:
            args = x = (s, y, a, n, U, z_c, a_c)
    '''
    cdef double KE = exp(-2*x[0]) * (3 - pow(x[1],2)/2* exp(-pow(x[1],2)/2) / (1 + exp(pow(x[1],2)/2)) )
    cdef double coul = x[4]* exp(-x[0]) * (erf(x[1]/sqrt(2)) *1./x[1] + sqrt(2/pi)*exp(-pow(x[1],2)/2)) / (1 + exp(-pow(x[1],2)/2))
    cdef double e_ph = Eph_ln(x[0],x[2], x[1], x[3], x[4], x[5],x[6])
    return x[3],x[4],x[2],exp(x[0]),x[1], KE + e_ph + coul

#################################################################################################################################
cdef extern from "complex.h":
    double complex cexp(double complex)
    double creal(double complex z)

def fr_special(tuple args):
    '''Phonon displacement function f(r) (FT of f_k) for 2 special cases: r || d, r perp d, d = electron separation vector (d || kz)

    Input:
        r: coordinate for position, in units of electron size sigma
        opt: 0 gives r perp d, 1 gives r || d
        s: sigma/l, nondim electron size
    '''
    def Integrand(double t,double x, double r,double a,double y,double s, int opt):
        cdef double numer = exp(-pow(1-a,2)*pow(x,2)/4) * (cos((1-a)/2* y*x*cos(t)) + exp(-pow(y,2)/2) ) + exp(-(1+pow(a,2))*pow(x,2)/4) * (cos((1+a)/2* y*x* cos(t))+ exp(-pow(y,2)/2) )
        cdef double denom = 1 + pow(a*x,2)/pow(s,2) + exp(-pow(a*x,2)/2) / (1+ exp(-pow(y,2)/2) ) * (cos(a*y*x* cos(t))+ exp(-pow(y,2)/2) )
        if opt == 1: return fabs(x)* cexp(1j*x*r* cos(t))* numer/denom * sin(t)
        if opt == 0: return fabs(x)* cexp(-1j*x*r* sin(t))* numer/denom * sin(t)

    cdef double r = args[0]
    cdef double a = args[1]
    cdef double y = args[2]
    cdef double s = args[3]
    cdef double alpha = args[4]
    
    return r, 1./(pow(s,2)* (1+exp(-pow(y,2))) ) * sqrt(4*pi*alpha)/pow(2*pi,2) * mpm.quad(lambda x,u: Integrand(u,x,r,a,y,s,1),[0,np.inf],[0,pi]), 1./(pow(s,2)* (1+exp(-pow(y,2))) ) * sqrt(4*pi*alpha)/pow(2*pi,2) *mpm.quad(lambda x,u: Integrand(u,x,r,a,y,s,0),[0,np.inf],[0,pi])

def nakf_k(double xx, double xy, double xz, double a,double y,double s, double alpha):
    '''Phonon displacement function f(kx,ky,kz)'''
    cdef double numer = exp(-pow(1-a,2)*(pow(xx,2)+pow(xy,2) + pow(xz,2))/4) * (cos((1-a)/2* y*xz) + exp(-pow(y,2)/2) ) + exp(-(1+pow(a,2))*(pow(xx,2)+pow(xy,2) + pow(xz,2))/4) * (cos((1+a)/2* y*xz)+ exp(-pow(y,2)/2) )
    cdef double denom = 1 + pow(a/s,2) + exp(-pow(a,2)*(pow(xx,2)+pow(xy,2) + pow(xz,2))/2) / (1+ exp(-pow(y,2)/2) ) * (cos(a*y*xz)+ exp(-pow(y,2)/2) )
    return sqrt(4*pi*alpha)/(1+exp(-pow(y,2))) * s * numer/denom * 1/ sqrt(pow(xx,2)+pow(xy,2) + pow(xz,2)) #setting V/l^3 = 1

def fr3D(tuple r, double a,double y,double s, double alpha):
    '''Phonon displacement function f(r) in real space - Fourier Transform of f_k'''
    def Integrand(double xx,double xy, double xz, double rx, double ry, double rz,double a,double y,double s):
        cdef double numer = exp(-pow(1-a,2)* (pow(xx,2) + pow(xy,2) + pow(xz,2))/4) * (cos((1-a)/2* y*xz) + exp(-pow(y,2)/2) ) + exp(-(1+pow(a,2))*(pow(xx,2) + pow(xy,2) + pow(xz,2))/4) * (cos((1+a)/2* y*xz)+ exp(-pow(y,2)/2) )
        cdef double denom = 1 + pow(a/s,2)*(pow(xx,2) + pow(xy,2) + pow(xz,2)) + exp(-pow(a,2)*(pow(xx,2) + pow(xy,2) + pow(xz,2))/2) / (1+ exp(-pow(y,2)/2) ) * (cos(a*y*xz)+ exp(-pow(y,2)/2) )
        return numer/denom *cexp(1j*(xx*rx + xy*ry + xz*rz))/sqrt((pow(xx,2) + pow(xy,2) + pow(xz,2)))
    return a,y,s,r[0],r[1],r[2],1./(pow(s,2)* (1+exp(-pow(y,2))) ) * sqrt(4*pi*alpha)/pow(2*pi,3) * mpm.quad(lambda kz,ky,kx: Integrand(kx,ky,kz,r[0],r[1],r[2],a,y,s),[-np.inf,np.inf],[-np.inf,np.inf],[-np.inf,np.inf])


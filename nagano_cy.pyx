import numpy as np
cimport numpy as np
from scipy import integrate
from scipy.optimize import minimize
from libc.math cimport fabs, sqrt, pi, exp, sin, cos, pow, erf, atan, log, floor, atan2, tan
#from libc.complex cimport cexp
import warnings
cimport scipy.special.cython_special as cy_scipy

############################################################################################################################################
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
    
def kIntegrand_a1(double x, double s):
   return 0.5*(pow(1 + exp(-pow(x,2)/2),2)/ (1 + pow(x,2)* exp(-2*s) + exp(-pow(x,2)/2) ) + pow(1 - exp(-pow(x,2)/2) ,2)/ (1 + pow(x,2)* exp(-2*s) - exp(-pow(x,2)/2) ))   

def TestEnv(double y, double s, double a, int option):
    '''Testing the envelope function for the nakano double integral assuming the cosines can be de-correlated (i.e. separating out the a-dependent terms and averaging over the     non a-dependent ones'''
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

def zIntegrand_blah(double z, double y, double s, double a):
    '''Testing just the z integral's integrand to make sure the branch cuts of the tangent fxn have been dealt with properly. This is for the envelope fxn approximation that totally ignores the argument of the arctangent and just turns arctan(A*tan(bx)) = bx'''
    cdef double A = (1-exp(-0.5*pow(a*z,2)) + pow(a*z,2)*exp(-2*s)) / sqrt(1-exp(-pow(a*z,2)) + 2*pow(a*z,2)*exp(-2*s) + pow(a*z,4)*exp(-4*s))
    cdef double B = 0.5*a*y*z
    cdef double trig = pi* floor(B/pi) + atan2(1, 1/(A*tan(B))) #replaces arctan(A*tan(B))
    return exp(-0.5*(1-a)*pow(z,2)) + (exp(-pow(1-a,2)*pow(z,2)/2) + exp(-(1+pow(a,2))*pow(z,2)/2) + 2*exp(-0.5*(1-a)*pow(z,2)) * (1+pow(a*z,2)*exp(-2*s)) ) / (a*y*z* sqrt(1-exp(-pow(a*z,2)) + 2*pow(a*z,2)*exp(-2*s) + pow(a*z,4)*exp(-4*s))) * trig

def TestZInt_blah(double y, double s, double a, double n, double U, double z_c):
    def Integrand(double z, double y, double s, double a):
        cdef double A = (1-exp(-0.5*pow(a*z,2)) + pow(a*z,2)*exp(-2*s)) / sqrt(1-exp(-pow(a*z,2)) + 2*pow(a*z,2)*exp(-2*s) + pow(a*z,4)*exp(-4*s))
        cdef double B = 0.5*a*y*z
        cdef double trig = pi* floor(z*B/pi) + atan2(1, 1/(A*tan(B))) #replaces arctan(A*tan(B))
        return exp(-0.5*(1-a)*pow(z,2)) + (exp(-pow(1-a,2)*pow(z,2)/2) + exp(-(1+pow(a,2))*pow(z,2)/2) + 2*exp(-0.5*(1-a)*pow(z,2)) * (1+pow(a*z,2)*exp(-2*s)) ) / (a*y*z* sqrt(1-exp(-pow(a*z,2)) + 2*pow(a*z,2)*exp(-2*s) + pow(a*z,4)*exp(-4*s))) * trig
    return -2*(1-n)*U*exp(-s)/pi* mpm.quad(lambda z: Integrand(z,y,s,a),[0,np.inf])

def zIntegrand_a1_yfin(double z, double y, double s, double opt):
    '''Testing just the z integral's integrand for a=1, finite (general) y to make sure the branch cuts of the tangent fxn have been dealt with properly'''
    cdef double Xm = pow(z,2)*exp(-2*s) + 1 + exp(-pow(z,2)/2) *(exp(-pow(y,2)/2)-1)/(exp(-pow(y,2)/2)+1)
    cdef double Xp = pow(z,2)*exp(-2*s) + 1 + exp(-pow(z,2)/2)
    cdef double tanarg = 0.5*y*z
    cdef double trig = pi* floor(tanarg/pi) + atan2(1, 1/(sqrt(Xm/Xp)*tan(tanarg)) ) #replaces arctan(A*tan(B))

    if opt == 1:
        #"reduced" integrand, integrating out everything that can be done analytically
        return 1. - pow(z,2) *exp(-2*s) + 2*pow(z,3)* exp(-4*s) /(y*sqrt(Xm*Xp)) * trig
    else:
        #full integrand
        return 1. - pow(z,2) *exp(-2*s) + 2*pow(z,3)* exp(-4*s) /(y*sqrt(Xm*Xp)) * trig + exp(-pow(z,2)/2)/(1 + exp(-pow(y,2)/2)) * (exp(-pow(y,2)/2) + sin(y*z)/(y*z))

def TestZInt_a1_yfin(double y, double s, double n, double U, int opt):
    if opt == 1:
        #remove oscillatory piece analytically
        return -(1-n)*U*exp(-s) * (1./(1+exp(-pow(y,2)/2)) * (sqrt(2/pi)*exp(-pow(y,2)/2) + 1/y* erf(y/sqrt(2)) ) + 2./pi* integrate.quad(zIntegrand_a1_yfin,0,np.inf,args=(y,s,1))[0] )
    if opt == 2:
        return -(1-n)*U*exp(-s) * 2./pi* integrate.quad(zIntegrand_a1_yfin,0,np.inf,args=(y,s,2))[0]

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
    return KE + e_ph + coul

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
    cdef double sig = args[5]
    cdef tuple bnds = ((-2.5,0),) #s
    cdef np.ndarray[double, ndim=1] guess = np.array([-1.])
    if (1-n)*u/2 >= 10:
        bnds = ((-2.5,-0.5),) #s
        guess = np.array([-1.])
        if (1-n)*u/2 > 15 and (1-n)*u/2 < 25:
            guess = np.array([-2.])
        elif (1-n)*u/2 >= 25:
            bnds = ((-5,-1.5),) #s
            guess = np.array([-4.]) #s
        res = minimize(E_afix_inf,guess,args=(0.,y,n,u, z_c, a_c),bounds=bnds)
        return n,u,0.,exp(res.x[0]),y,E_afix_inf(res.x,0.,y,n,u,z_c, a_c)
    elif (1-n)*u/2 < 7:
        bnds = ((sig-1,sig),) #s
        guess = np.array([sig])
        res = minimize(E_afix_inf,guess,args=(1.,y,n,u, z_c, a_c),bounds=bnds)
        return n,u,1.,exp(res.x[0]),y,E_afix_inf(res.x,1.,y,n,u,z_c, a_c)
    else: #near the 1st order transition from weak/strong coupling
        res = minimize(E_afix_inf,guess,args=(0.,y,n,u, z_c, a_c),bounds=bnds)
        bnds = ((sig-1,sig),) #s
        guess = np.array([sig])
        res2 = minimize(E_afix_inf,guess,args=(1.,y,n,u, z_c,a_c),bounds=bnds)
        if E_afix_inf(res.x,0.,y,n,u,z_c, a_c) > E_afix_inf(res2.x,1.,y,n,u,z_c, a_c): 
            return n,u,1.,exp(res2.x[0]),y,E_afix_inf(res2.x,1.,y,n,u,z_c, a_c)
        else:
            return n,u,0., exp(res.x[0]),y,E_afix_inf(res.x,0.,y,n,u,z_c, a_c)

def min_E_inf_sfix(tuple args):
    '''Minimize energy in Nagano formulation while fixing a to be either 0 or 1 (this is what various E(a) plots seem to indicate), fixed y-> infty'''
    cdef double n = args[0]
    cdef double u = args[1]
    cdef double z_c = args[2]
    cdef double a_c = args[3]
    cdef double s = args[4]
    cdef double y = args[5]
    cdef tuple X0 = tuple([s,y,0.,n,u,z_c,a_c])
    cdef tuple X1 = tuple([s,y,1.,n,u,z_c,a_c])

    if (1-n)*u/2 >= 10:
        return n,u,0.,exp(s),y,E_bip_aysfix(X0)[-1]
    elif (1-n)*u/2 < 7:
        return n,u,1.,exp(s),y,E_bip_aysfix(X1)[-1]
    else: #near the 1st order transition from weak/strong coupling
        if E_bip_aysfix(X0)[-1] > E_bip_aysfix(X1)[-1]: 
            return n,u,1.,exp(s),y,E_bip_aysfix(X1)[-1]
        else:
            return n,u,0.,exp(s),y,E_bip_aysfix(X0)[-1]

def min_E_inf_weak(tuple args):
    '''Minimize energy in Nagano formulation - WEAK COUPLING LIMIT (i.e. a=1), fixed y-> infty'''
    cdef double n = args[0]
    cdef double u = args[1]
    cdef double z_c = args[2]
    cdef double a_c = args[3]
    cdef double y = args[4]
    cdef double s = args[5]
    cdef tuple bnds = ((s-1,s),) #s
    cdef np.ndarray[double, ndim=1] guess = np.array([s])
    res = minimize(E_afix_inf,guess,args=(1.,y,n,u, z_c, a_c),bounds=bnds)
    return n,u,1.,exp(res.x[0]),y,E_afix_inf(res.x,1.,y,n,u,z_c, a_c)

def min_E_inf_strong(tuple args):
    '''Minimize energy in Nagano formulation - STRONG COUPLING LIMITwhile fixing a to be either 0 or 1 (this is what various E(a) plots seem to indicate), fixed y-> infty'''
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

########################################################################################################################
'''Various integrals for optimization'''

'''Minimize energy for finite y allowing a,y,sigma/l to vary'''    
import mpmath as mpm
'''Try optimizing taking sigma/l -> e^s where s = ln(sigma/l) so the energy landscape is smoother'''
def Integrand_ln(double z, double y, double s, double a):
    '''returns full z-integrand (integrate over u)'''
    def Integrand(double u, double x, double y, double s, double a):
        cdef double numer = exp(-pow(1-a,2)*pow(x,2)/4) * (cos((1-a)/2* y*x*u) + exp(-pow(y,2)/2) ) + exp(-(1+pow(a,2))*pow(x,2)/4) * (cos((1+a)/2* y*x*u)+ exp(-pow(y,2)/2) )
        cdef double denom = 1 + pow(a*x,2)* exp(-2*s) + exp(-pow(a*x,2)/2) / (1+ exp(-pow(y,2)/2) ) * (cos(a*y*x*u)+ exp(-pow(y,2)/2) )
        return pow(numer,2)/denom
    return 2*integrate.quad(Integrand, 0, 1, args=(z,y,s,a))[0]

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
    cdef double KE = exp(-2*x[1]) * (3 - 0.5*exp(2*x[0]) / (1 + exp(0.5*exp(x[0]*2))) )
    cdef double coul = U* exp(-x[1]) * (erf(exp(x[0])/sqrt(2)) *exp(-x[0]) + sqrt(2/pi)*exp(-0.5*exp(x[0]*2))) / (1 + exp(-0.5*exp(x[0]*2)))
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
    cdef double KE = exp(-2*x[1]) * (3 - exp(2*x[0])/2 * 1./ (1 + exp(exp(x[0]*2)/2)) )
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
    cdef double sig = args[5]
    cdef tuple bnds = ((-2,2),(-2.5,0),) #y, s
    cdef np.ndarray[double, ndim=1] guess = np.array([0.,-1.])
    if (1-n)*u/2 > 9:
        bnds = ((-2,1),(-2.5,0),) #y, s
        guess = np.array([0.,-1.])
        if (1-n)*u/2 >= 25:
            bnds = ((-2,1),(-5,-2),) #y, s
            guess = np.array([-1.,-4.])
        res = minimize(E_bip_afix,guess,args=(a, n,u, z_c, a_c),bounds=bnds)
        return n,u,a,exp(res.x[1]),exp(res.x[0]),E_bip_afix(res.x,a,n,u,z_c, a_c)
    elif (1-n)*u/2 < 7:
        bnds = ((-1,1),) #y
        guess = np.array([0.])
        res = minimize(E_bip_asfix,guess,args=(sig,1.,n,u, z_c, a_c),bounds=bnds)
        return n,u,1.,exp(sig),exp(res.x[0]),E_bip_asfix(res.x,sig,1.,n,u,z_c, a_c)
    else:
        res = minimize(E_bip_afix,guess,args=(a,n,u, z_c, a_c),bounds=bnds)
        bnds = ((-1,1),) #y
        guess = np.array([0.])
        res2 = minimize(E_bip_asfix,guess,args=(sig,1.,n,u, z_c, a_c),bounds=bnds)
        if E_bip_afix(res.x,a,n,u,z_c, a_c) > E_bip_asfix(res2.x,sig,1.,n,u,z_c, a_c):
            return n,u,1.,exp(sig),exp(res2.x[0]),E_bip_asfix(res2.x,sig,1.,n,u,z_c, a_c)
        else:
            return n,u,a,exp(res.x[1]),exp(res.x[0]),E_bip_afix(res.x,a,n,u,z_c, a_c)

def min_E_bip_ln2(tuple args):
    '''Minimize energy in Nakano formulation allowing Y = ln(y), s = ln(sig), a to vary. Set weak coupling result to a=1 so don't have to optimize a. For alpha near the phase transition need to check whether a=1 or something else'''
    cdef double n = args[0]
    cdef double u = args[1]
    cdef double z_c = args[2]
    cdef double a_c = args[3]
    cdef double sig = args[5]
    cdef tuple bnds = ((-2,2),(-2.5,0),(0,1),) #y, s, a
    cdef np.ndarray[double, ndim=1] guess = np.array([0.,-1,0.1])
    if (1-n)*u/2 > 9:
        if (1-n)*u/2 >= 30 and (1-n)*u/2 < 100:
            bnds = ((-2,1),(-5,-2),(0,0.3),) #y, s, a
            guess = np.array([0.,-2.5,0.001])
            res = minimize(E_bip_ln,guess,args=(n,u, z_c, a_c),bounds=bnds)
            return n,u,res.x[2],exp(res.x[1]),exp(res.x[0]),E_bip_ln(res.x,n,u,z_c, a_c)
        elif (1-n)*u/2 >= 100: 
            bnds = ((-2,1),(-5,-2),(0,0.3),) #y, s, a
            guess = np.array([0.,-4,0.])
            res = minimize(E_bip_ln,guess,args=(n,u, z_c, a_c),bounds=bnds)
            return n,u,res.x[2],exp(res.x[1]),exp(res.x[0]),E_bip_ln(res.x,n,u,z_c, a_c)
        else:             
            #this is for 9 < alpha < 30
            bnds = ((-2,1),(-2.5,0),(0,0.3),) #y, s, a
            guess = np.array([0.,-1,0.01])
            res = minimize(E_bip_ln,guess,args=(n,u, z_c, a_c),bounds=bnds)
            return n,u,res.x[2],exp(res.x[1]),exp(res.x[0]),E_bip_ln(res.x,n,u,z_c, a_c)
    elif (1-n)*u/2 < 7:
        bnds = ((-1,1),) #y
        guess = np.array([0.])
        res = minimize(E_bip_asfix,guess,args=(sig,1.,n,u, z_c, a_c),bounds=bnds)
        return n,u,1.,exp(sig),exp(res.x[0]),E_bip_asfix(res.x,sig,1.,n,u,z_c, a_c)
    else:
        #this is for 7 < alpha < 9
        res = minimize(E_bip_ln,guess,args=(n,u, z_c, a_c),bounds=bnds)
        bnds = ((-1,1),) #y
        guess = np.array([0.])
        res2 = minimize(E_bip_asfix,guess,args=(sig,1.,n,u, z_c, a_c),bounds=bnds)
        if E_bip_ln(res.x,n,u,z_c, a_c) > E_bip_asfix(res2.x,sig,1.,n,u,z_c, a_c):
            return n,u,1.,exp(sig),exp(res2.x[0]),E_bip_asfix(res2.x,sig,1.,n,u,z_c, a_c)
        else:
            return n,u,res.x[2],exp(res.x[1]),exp(res.x[0]),E_bip_ln(res.x,n,u,z_c, a_c)

def min_E_bip_strong(tuple args):
    '''Minimize energy in Nakano formulation allowing Y = ln(y), s = ln(sig), a to vary. Only look for strong coupling solution. '''
    cdef double n = args[0]
    cdef double u = args[1]
    cdef double z_c = args[2]
    cdef double a_c = args[3]
    
    cdef tuple bnds = ((-2,1),(-2.5,5),(0,1),) #y, s, a
    cdef np.ndarray[double, ndim=1] guess = np.array([0.,-1,0.1])
    if (1-n)*u/2 >= 30 and (1-n)*u/2 < 100:
        bnds = ((-2,1),(-5,-2),(0,0.3),) #y, s, a
        guess = np.array([0.,-2.5,0.001])
        res = minimize(E_bip_ln,guess,args=(n,u, z_c, a_c),bounds=bnds)
        return n,u,res.x[2],exp(res.x[1]),exp(res.x[0]),E_bip_ln(res.x,n,u,z_c, a_c)
    elif (1-n)*u/2 >= 100: 
        bnds = ((-2,1),(-5,-2),(0,0.3),) #y, s, a
        guess = np.array([0.,-4,0.])
        res = minimize(E_bip_ln,guess,args=(n,u, z_c, a_c),bounds=bnds)
        return n,u,res.x[2],exp(res.x[1]),exp(res.x[0]),E_bip_ln(res.x,n,u,z_c, a_c)
    elif (1-n)*u/2 > 9 and (1-n)*u/2 < 30:             
        #this is for alpha < 30
        bnds = ((-2,1),(-2.5,0),(0,0.3),) #y, s, a
        guess = np.array([0.,-1,0.01])
        res = minimize(E_bip_ln,guess,args=(n,u, z_c, a_c),bounds=bnds)
        return n,u,res.x[2],exp(res.x[1]),exp(res.x[0]),E_bip_ln(res.x,n,u,z_c, a_c)
    else:
        #this is for alpha < 9
        res = minimize(E_bip_ln,guess,args=(n,u, z_c, a_c),bounds=bnds)
        return n,u,res.x[2],exp(res.x[1]),exp(res.x[0]),E_bip_ln(res.x,n,u,z_c, a_c)

def min_E_bip_weak(tuple args):
    '''Minimize energy in Nagano formulation - WEAK COUPLING LIMIT (i.e. a=1), fixed y-> infty'''
    cdef double n = args[0]
    cdef double u = args[1]
    cdef double z_c = args[2]
    cdef double a_c = args[3]
    cdef double s = args[5]

    cdef tuple bnds = ((-1,1),) #y
    cdef np.ndarray[double, ndim=1] guess = np.array([0.])
    res = minimize(E_bip_asfix,guess,args=(s,1.,n,u, z_c, a_c),bounds=bnds)
    return n,u,1.,exp(s),exp(res.x[0]),E_bip_asfix(res.x,s,1.,n,u,z_c, a_c)

def E_bip_yfix_ln(np.ndarray[double, ndim=1] x, double y, double n, double U, double z_c, double a_c):
    '''bipolaron energy normalized by KE = hw with fixed y
        Inputs:
            x = [a,s] nondimensionalized elec size
            s = log(sigma/l)
    '''
    cdef double KE = exp(-2*x[1]) * (3 - pow(y,2)/2 / (1 + exp(pow(y,2)/2)) )
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

    cdef tuple bnds = ((-2.5,0),) #s
    cdef np.ndarray[double, ndim=1] guess = np.array([-1.])

    if (1-n)*u/2 > 9:
        if (1-n)*u/2 > 25:
            bnds = ((-5,-2),) #s
            guess = np.array([-4.])
        res = minimize(E_bip_ayfix,guess,args=(y,a,n,u, z_c, a_c),bounds=bnds)
        return n,u,a,exp(res.x[0]),y,E_bip_ayfix(res.x,y,a,n,u,z_c, a_c)
    elif (1-n)*u/2 < 7:
        bnds = ((2,5),) #s
        guess = np.array([5.])
        res = minimize(E_bip_ayfix,guess,args=(y,1.,n,u, z_c, a_c),bounds=bnds)
        return n,u,1.,exp(res.x[0]),y,E_bip_ayfix(res.x,y,1.,n,u,z_c, a_c)
    else:
        res = minimize(E_bip_ayfix,guess,args=(y,a,n,u, z_c, a_c),bounds=bnds)
        bnds = ((2,5),) #s
        guess = np.array([5.])
        res2 = minimize(E_bip_ayfix,guess,args=(y,1.,n,u, z_c, a_c),bounds=bnds)
        if E_bip_ayfix(res.x,y,a,n,u,z_c, a_c) > E_bip_ayfix(res2.x,y,1.,n,u,z_c, a_c): 
            return n,u,1.,exp(res2.x[0]),y,E_bip_ayfix(res2.x,y,1.,n,u,z_c, a_c)
        else:
            return n,u,a,exp(res.x[0]),y,E_bip_ayfix(res.x,y,a,n,u,z_c, a_c)

def E_bip_sfix(np.ndarray[double, ndim=1] x, double s, double n, double U, double z_c, double a_c):
    '''bipolaron energy normalized by KE = hw; written in terms of s = ln (sig) (base e)
        Inputs:
            x = [y,a] nondimensionalized elec sep dist, size, el-ph boost param
    '''
    cdef double KE = exp(-2*s) * (3 - exp(2*x[0])/2 * 1./ (1 + exp(exp(x[0]*2)/2)) )
    cdef double coul = U* exp(-s) * (erf(exp(x[0])/sqrt(2)) *exp(-x[0]) + sqrt(2/pi)*exp(-exp(x[0]*2)/2)) / (1 + exp(-exp(x[0]*2)/2))
    cdef double e_ph = Eph_ln(s,x[1], exp(x[0]), n, U, z_c,a_c)
    return KE + e_ph + coul

def E_bip_asfix(np.ndarray[double, ndim=1] x, double s, double a, double n, double U, double z_c, double a_c):
    '''bipolaron energy normalized by KE = hw; written in terms of s = ln (sig) (base e)
        Inputs:
            x = [y] nondimensionalized elec sep dist
    '''
    cdef double KE = exp(-2*s) * (3 - exp(2*x[0])/2* 1. / (1 + exp(exp(x[0]*2)/2)) )
    cdef double coul = U* exp(-s) * (erf(exp(x[0])/sqrt(2)) *exp(-x[0]) + sqrt(2/pi)*exp(-exp(x[0]*2)/2)) / (1 + exp(-exp(x[0]*2)/2))
    cdef double e_ph = Eph_ln(s,a, exp(x[0]), n, U, z_c,a_c)
    return KE + e_ph + coul

def min_E_bip_sfix(tuple args):
    '''Minimize energy in Nakano formulation allowing Y = ln(y), s = ln(sig), a to vary. Set weak coupling result to a=1 so don't have to optimize a. For alpha near the phase transition need to check whether a=1 or something else'''
    cdef double n = args[0]
    cdef double u = args[1]
    cdef double z_c = args[2]
    cdef double a_c = args[3]
    cdef double s = args[4]

    cdef tuple bnds = ((-2,1),(0,1),) #y, a
    cdef np.ndarray[double, ndim=1] guess = np.array([0.,0.1])

    if (1-n)*u/2 > 9:
        bnds = ((-2,1),(0,0.3),) #y, a
        if (1-n)*u/2 >= 30 and (1-n)*u/2 < 100:
            guess = np.array([0.,0.001])
        elif (1-n)*u/2 >= 100: 
            guess = np.array([0.,0.])
        else:             
            #this is for 9 < alpha < 30
            guess = np.array([0.,0.01])
        res = minimize(E_bip_sfix,guess,args=(s,n,u, z_c, a_c),bounds=bnds)
        return n,u,res.x[1],exp(s),exp(res.x[0]),E_bip_sfix(res.x,s,n,u,z_c, a_c)
    elif (1-n)*u/2 < 7:
        bnds = ((-1,1),) #y
        guess = np.array([0.])
        res = minimize(E_bip_asfix,guess,args=(s,1.,n,u, z_c, a_c),bounds=bnds)
        return n,u,1.,exp(s),exp(res.x[0]),E_bip_asfix(res.x,s,1.,n,u,z_c, a_c)
    else:
        #this is for 7 < alpha < 9
        res = minimize(E_bip_sfix,guess,args=(s,n,u, z_c, a_c),bounds=bnds)
        bnds = ((-1,1),) #y
        guess = np.array([0.])
        res2 = minimize(E_bip_asfix,guess,args=(s,1.,n,u, z_c, a_c),bounds=bnds)
        if E_bip_sfix(res.x,s,n,u,z_c, a_c) > E_bip_asfix(res2.x,s,1.,n,u,z_c, a_c):
            return n,u,1.,exp(s),exp(res2.x[0]),E_bip_asfix(res2.x,s,1.,n,u,z_c, a_c)
        else:
            return n,u,res.x[1],exp(s),exp(res.x[0]),E_bip_sfix(res.x,s,n,u,z_c, a_c)

################################################################################################################################
'''Compilation of wrapper functions for convenience'''

def min_E_nak(tuple args):
    '''Wrapper function for energy minimization in Nakano formulation'''

    cdef tuple resfin = min_E_bip_ln2(args)
    cdef tuple resinf = min_E_inf(args)
    #return n, U, all finite optimization quantities, all y->inf optimization quantities, and the relative binding energy (E_opt - E_inf)/|E_inf|
    cdef double Ebind = (resfin[-1] - resinf[-1])/abs(resinf[-1])
    return resfin + resinf[2:] + (Ebind,)

def min_E_nak_strong(tuple args):
    '''Wrapper function for strong-coupling limit energy minimization in Nakano formulation'''

    cdef tuple resfin = min_E_bip_strong(args)
    cdef tuple resinf = min_E_inf_strong(args)
    #return n, U, all finite optimization quantities, all y->inf optimization quantities, and the relative binding energy (E_opt - E_inf)/|E_inf|
    cdef double Ebind = (resfin[-1] - resinf[-1])/abs(resinf[-1])
    return resfin + resinf[2:] + (Ebind,)

def min_E_nak_weak(tuple args):
    '''Wrapper function for strong-coupling limit energy minimization in Nakano formulation'''

    cdef tuple resfin = min_E_bip_weak(args)
    cdef tuple resinf = min_E_inf_weak(args)
    #return n, U, all finite optimization quantities, all y->inf optimization quantities, and the relative binding energy (E_opt - E_inf)/|E_inf|
    cdef double Ebind = (resfin[-1] - resinf[-1])/abs(resinf[-1])
    return resfin + resinf[2:] + (Ebind,)

def min_E_sfix(tuple args):
    '''Wrapper function for energy minimization at fixed sigma values, to generate E(sigma) plots. Prob want to do same for regular optimization e.g. min_E_bip_ln2 + min_E_inf if this works'''
    cdef tuple resfin = min_E_bip_sfix(args)
    cdef tuple resinf = min_E_inf_sfix(args)
    #return n, U, all finite optimization quantities, all y->inf optimization quantities, and the relative binding energy (E_opt - E_inf)/|E_inf|
    cdef double Ebind = (resfin[-1] - resinf[-1])/abs(resinf[-1])
    return resfin + resinf[2:] + (Ebind,)

################################################################################################################################
'''generate maps of E(a,sigma) at a couple fixed values of y=0, 10, "infty" to help better understand contradictions in E(a) and E(y) results'''
def E_bip_aysfix(tuple x):
    '''bipolaron energy normalized by KE = hw with fixed a
        Inputs:
            args = x = (s, y, a, n, U, z_c, a_c)
    '''
    cdef double KE = exp(-2*x[0]) * (3 - pow(x[1],2)/2* 1./ (1 + exp(pow(x[1],2)/2)) )
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

#####################################################################################################################################
'''Verbist & Devreese '91 best oscillator-oscillator trial wavefunction (beta = 1) for comparison with Nakano solution in strong-coupling regime'''

def E_dev(np.ndarray[double, ndim=1] x, double n, double U, int opt):
    '''Devreese energy
        Inputs:
            x = [ln O,ln O1] nondimensionalized inverse (squared) electron sizes Omega, Omega1
    '''
    if opt == 1: #beta = 1 wfn
        return 0.5*(3*exp(x[0]) + 2*exp(x[1])) - 4.*(1-n)*U/sqrt(pi) * sqrt(exp(x[0]+x[1])/ (exp(x[0])+exp(x[1]))) * (1- 1./3 * exp(x[1])/(exp(x[0]) + exp(x[1])) + 1./12* exp(2*x[1])/pow(exp(x[0]) + exp(x[1]),2)) + 4*U/3 * sqrt(2*exp(x[0])/pi)
    else: #beta = 0 wfn
        return 3./2*(exp(x[0]) + exp(x[1])) - 4.*(1-n)*U/sqrt(pi) * sqrt(exp(x[0]+x[1])/ (exp(x[0])+exp(x[1]))) + U* sqrt(2*exp(x[0])/pi)

def min_E_dev(tuple args):
    '''Minimize Devreese energy'''
    cdef double n = args[0]
    cdef double u = args[1]
    cdef double opt = args[2]
    cdef tuple bnds = ((-10,10),(-10,10),) #O,O1
    cdef np.ndarray[double, ndim=1] guess = np.array([1.,1.])
    res = minimize(E_dev,guess,args=(n,u, opt,),bounds=bnds)
    cdef double Einf = -2*pow((1-n)*u/2,2)/(3*pi) #manually compare E_opt with 2*single polaron energy; Devreese calculated this to be -alpha^2/(3pi)
    return n,u,exp(res.x[0]),exp(res.x[1]),E_dev(res.x,n,u, opt), Einf, (E_dev(res.x,n,u, opt) - Einf)/abs(Einf) #last value is the binding energy

###############################################################################

'''Spherically symmetric soln, to compare to finite-y soln for Nakano, above'''
#minimizing wrt sigma/l = s, y = d/sigma
def sym_Ir(double k,double y):
    def Integrand(double r, double k, double y):
        return r*exp(-pow(r-y,2)) * sin(k*r/2)
    return integrate.quad(Integrand, 0, np.inf, args=(k,y))[0] 

def sym_Ik(double y, double const, int opt):
    def Integrand(double k, double y):
        return pow(sym_Ir(k,y)/k,2)
    def Integrand_inf(double k, double y, double const):
        return pow(sym_Ir(k,y)/(k*const),2)
    if opt == 0:
        return integrate.quad(Integrand, 0, np.inf, args=(y,))[0] 
    else:
        return integrate.quad(Integrand_inf, 0, np.inf, args=(y,const,))[0] 
  
def sym_E(np.ndarray[double, ndim=1] x, double n, double U):
    '''bipolaron energy normalized by KE = hw; written in terms of s = ln (sig) (base e)
        Inputs:
            x = [y,s] nondimensionalized elec sep dist, size
    '''
    cdef double const = sqrt(2/pi)*exp(x[0]) * exp(-exp(2*x[0])) + (1+exp(2*x[0]))*(1 + erf(exp(x[0])/sqrt(2)))
    cdef double KE = exp(-2*x[1])/2 * ((3+5*exp(2*x[0]) + sqrt(2/pi)*exp(x[0]) * exp(-exp(2*x[0])) + (exp(x[0])-1)*(exp(x[0])-3)* erf(exp(x[0])/sqrt(2)) ) /const + 3) 
    cdef double coul = U*exp(-x[1]) * (exp(-exp(2*x[0])) + (1+exp(x[0]))*erf(exp(x[0])/sqrt(2)))/const
    cdef double Eph = -32*(1-n)*U/pow(pi*const,2) * exp(-x[1]) * sym_Ik(exp(x[0]), const, 0)
    return KE + coul + Eph

def sym_E_inf(np.ndarray[double, ndim=1] x, double y, double n, double U):
    '''bipolaron energy normalized by KE = hw; written in terms of s = ln (sig) (base e)
        Inputs:
            x = [s] nondimensionalized elec sep dist, size
    '''
    cdef double const = sqrt(2/pi)*y * exp(-pow(y,2)) + (1+pow(y,2))*(1 + erf( y/sqrt(2)))
    cdef double KE = exp(-2*x[0])/2 * ((3+5*pow(y,2) + sqrt(2/pi)*y * exp(-pow(y,2)) + (y-1)*(y-3)* erf(y/sqrt(2)) ) /const + 3) 
    cdef double coul = U*exp(-x[0]) * (exp(-pow(y,2)) + (1+y)*erf(y/sqrt(2)))/const
    cdef double Eph = -32*(1-n)*U/pow(pi,2) * exp(-x[0]) * sym_Ik(y, const, 1)
    return KE + coul + Eph

def min_sym_E(tuple args):
    '''Minimize energy in Nakano formulation allowing Y = ln(y), s = ln(sig) to vary in the strong-coupling limit (a=0). Using a symmetric wfn'''
    cdef double n = args[0]
    cdef double u = args[1]
    cdef tuple bnds = ((-2,2),(-2.5,1),) #y, s 
    cdef np.ndarray[double, ndim=1] guess = np.array([0.,-1])
    res = minimize(sym_E,guess,args=(n,u),bounds=bnds)
    return n,u,0,exp(res.x[1]),exp(res.x[0]),sym_E(res.x,n,u), 0,0,0,-(1-n)*u, (sym_E(res.x,n,u) + 2*(1-n)*u/2)/ abs((1-n)*u)

def min_sym_E_inf(tuple args):
    '''Minimize energy in Nagano formulation while fixing a to be either 0 or 1 (this is what various E(a) plots seem to indicate), fixed y-> infty'''
    cdef double n = args[0]
    cdef double u = args[1]
    cdef double y = args[4]
    cdef tuple bnds = ((-2.5,0),) #s
    cdef np.ndarray[double, ndim=1] guess = np.array([-1.])
    res = minimize(sym_E_inf,guess,args=(y,n,u),bounds=bnds)
    return n,u,0,exp(res.x[0]),y,sym_E_inf(res.x,y,n,u)

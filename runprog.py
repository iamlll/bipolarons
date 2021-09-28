#import baby_integrand as bby
#import baby_integrand_py as beep
#import baby_integral as bee
#import var_a_cy as avar
import numpy as np
import timeit
#import pool_hyb as ph
from scipy.special import erfc
#import asym_U1 as asym
import nagano_cy as nag
#import indep_params as ip

a=0.5
x = np.array([5000., 6.])
t = tuple(x)
b = 2.
c = 5000.
c2 = 4.
u = 0.001
n = 0.

A= 100; b=.1; c=100;y = 5000
#print(nag.E_bip_ln(np.array([1.,1.,0.]), 0., 2., 10., 0.6))

#print(nag.TestZInt(y, np.log(.192), 0.02, 0.,39.52,np.inf))
#print(nag.Eph_inf_ln(np.log(.192),0.02,y, 0.,39.52,10.,0.6))
print(nag.Integral_ln_inf(500., np.exp(20), 1., 10., 3))

#print(nag.TestEnv(y, 145., 1., 1))
#print(nag.TestEnv(y, 145., 1., 3))
#print(avar.minimization(tuple(np.array([0.,32.]))))
#print(nag.TestZInt_a1_yfin(y, 1., 0., 10.,1))
#print(nag.TestZInt_a1_yfin(y, 1., 0., 10.,2))
#print(nag.Eph_inf_ln(1.,1.,y, 0.,10.,10.,2))
#print(nag.min_E_bip_ln2(tuple(np.array([0.08,80., 10.,0.6]))))
#print(nag.min_E_bip_ln2(tuple(np.array([0.05, 30., 10.,0.6]))))
#print(nag.min_E_bip_yfix_ln(tuple(np.array([0.05, 30., 10.,0.6, 1.4937278762288424]))))

#print(nag.min_E_bip_ln2(tuple(np.array([0.018947368,52.95151147, 10.,0.6]))))
#print(nag.min_E_bip_ln2(tuple(np.array([0.252631579, 169.9562482, 10.,0.6,500.]))))
#print(nag.min_E_inf(tuple(np.array([0.3,31.42858899,10.,0.6,500.]))))
#print(nag.E_bip_aysfix(tuple(np.array([np.log(0.2569211517936647), 1.4937278762288424,0.0338635503091855, 0.05, 30., 10.,0.6])))) #(s, y, a, n, U, z_c, a_c)
#print(nag.E_bip_aysfix(tuple(np.array([np.log(0.2),1E-5,0.,0.,40.,10.,0.6])))) #(s, y, a, n, U, z_c, a_c)
#print(nag.E_bip_aysfix(tuple(np.array([np.log(0.135335283),500.,0.,0.018947368,52.95151147,10.,0.6])))) #(s, y, a, n, U, z_c, a_c)

setupstr = "import numpy as np; \
            import baby_integrand as bby; \
            import baby_integral as bee; \
            import pool_hyb as ph; \
            import var_a_cy as avar; \
            import asym_U1 as asym; \
            import nagano_cy as nag; \
            import indep_params as ip; \
            a = 1.; x = np.array([5000.,6.]); t = tuple(x); \
            b=2.; c = 5000.; c2=4.; n=2E-4; y=5000;" \

#print(timeit.timeit(stmt="nag.TestZInt_a1_yfin(y, 1., 0., 10.,1)",setup=setupstr,number=5))
#print(timeit.timeit(stmt="nag.Eph_inf_ln(1.,1.,y, 0.,10.,10.,2)",setup=setupstr,number=5))

#print(timeit.timeit(stmt="nag.minE_fixed_a(tuple((0,1,1),))",setup=setupstr,number=3))
#print(timeit.timeit(stmt="nag.min_E_a_1(tuple((0,1),))",setup=setupstr,number=3))


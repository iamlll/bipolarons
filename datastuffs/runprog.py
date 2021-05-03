import baby_integrand as bby
import baby_integrand_py as beep
import baby_integral as bee
import var_a_cy as avar
import numpy as np
import timeit
import pool_hyb as ph
from scipy.special import erfc
import asym_U1 as asym
import nagano_cy as nag
import indep_params as ip

a=0.5
x = np.array([5000., 6.])
t = tuple(x)
b = 2.
c = 5000.
c2 = 4.
u = 0.001
n = 0.

A= 100; b=.1; c=100;y = 5000
print(nag.E_bip_ln(np.array([1.,1.,0.]), 0., 2., 10., 0.6))
#print(avar.minimization(tuple(np.array([0.,20.]))))
#print(nag.min_E_bip_ln(tuple(np.array([0.,20.,10.,0.6]))))
#print(nag.min_E_bip_yfix_log(tuple(np.array([0.,2.,10.,0.6,1]))))
#print(nag.E_bip_aysfix(tuple(np.array([0.3557006720401728,1.2863125731009495,0.06167790403402075,0.,20.,10.,0.6])))) #(s, y, a, n, U, z_c, a_c)
#print(nag.min_E_bip_yfix_log(tuple(np.array([0.,2.,0.,0.6,5.]))))
#print(nag.min_E_bip_yfix_log(tuple(np.array([0.,10.,10.,0.6,5.]))))

setupstr = "import numpy as np; import baby_integrand as bby; \
            #import baby_integral as bee; \
            #import pool_hyb as ph; \
            #import var_a_cy as avar; \
            #import asym_U1 as asym; \
            import nagano_cy as nag; \
            #import indep_params as ip; \
            a = 1.; x = np.array([5000.,6.]); t = tuple(x); \
            b=2.; c = 5000.; c2=4.; n=2E-4"
#print(timeit.timeit(stmt="nag.min_E_avar_inf(tuple(np.array([0.,10.,10.,0.6])))",setup=setupstr,number=1))
#print(timeit.timeit(stmt="nag.min_E_inf(tuple(np.array([0.,10.,10.,0.6])))",setup=setupstr,number=1))

#print(timeit.timeit(stmt="nag.minE_fixed_a(tuple((0,1,1),))",setup=setupstr,number=3))
#print(timeit.timeit(stmt="nag.min_E_a_1(tuple((0,1),))",setup=setupstr,number=3))


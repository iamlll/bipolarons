from setuptools import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np

setup(
    #package_dir={'datastuffs': ''},
    include_dirs = [np.get_include()],
    ext_modules = cythonize([
                            #"baby_integrand.pyx",      #Cython code file w/ baby_integrand() fxn
                            #"baby_integral.pyx",        #Cython code containing definitions for all integrals + minimization function
                            #"pool_hyb_cy.py",          #Python code containing python multiprocessing.pool implementation
                            #"var_a_cy.pyx",          #Python code containing python multiprocessing.pool implementation
                            #"pool_var_a_cy.py",
                            #"pool_lg_sig.py",
                            #"asym_U1.pyx",
                            "nagano_cy.pyx",
                            "nakanoplots.py",
                            #"indep_params.pyx",
                            ]),          
                            #annotate=True)              #generates html annotation file
)
#setup(ext_modules = cythonize("nagano_cy.pyx"))

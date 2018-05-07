#!/usr/bin/env python3

# Coefficients of the initial distribution are written to init.h5/coeffs

import numpy as np
import h5py

from pyboltz.initialize import BKWInitialDistribution, project_polar_laguerre
from pyboltz.basis import KSBasis


import matplotlib.pyplot as plt

# create a Polar-Laguerre basis object
basis = KSBasis(30)

# distribution centered at [1,0]
v0 = 1 + 0j
bkw = BKWInitialDistribution(v0, beta=2)
# other initial distributions types are:
# from pyboltz.initialize import SkewGaussian, Maxwellian
# Write your own class derived from MaxwellBaseFunction. Examples may be found in `pyboltz/initialize.py`.

# project to Polar-Laguerre basis
C = project_polar_laguerre(bkw, basis)

# plot long x-axis
x = np.linspace(-4, 4, 100)
F = basis.evaluatez(C, x)
plt.plot(x, F)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()

fh5 = h5py.File('init.h5', 'w')
fh5.create_dataset('coeffs', shape=C.shape, dtype=float, data=C.astype(float))
fh5.close()

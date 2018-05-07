#!/usr/bin/env python3

import h5py
from pyboltz.initialize import BKWInitialDistribution, project_polar_laguerre
from pyboltz.ks_basis import KSBasis

# define BKW initial distribution, located at z0=0+0j momentum and Temperature T=beta/2
bkw = BKWInitialDistribution(z0=0 + 0j, beta=2)

# Polar-Laguerre basis, with max. polynomial degree K = 40
basis = KSBasis(K=40)
C = project_polar_laguerre(
    bkw,
    basis,
    # the chosen initial distr. is radially symmetric,
    # exploit this by setting `rsym=True` (e.g. faster quadrature)
    rsym=True)

# save to HDF5 container
with h5py.File('coefficients.h5', 'w') as fh5:
    fh5.create_dataset('coeffs', shape=C.shape, data=C)

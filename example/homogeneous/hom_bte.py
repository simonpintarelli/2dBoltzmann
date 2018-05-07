#!/usr/bin/env python3

import h5py
from pyboltz.initialize import BKWInitialDistribution, project_polar_laguerre
from spectral_tools import Polar2Hermite
from pyboltz.ks_basis import KSBasis
from pyboltz.collision_tensor import CollisionTensor
from pyboltz.ode import rk4
from pyboltz.observables import Moments
import matplotlib.pyplot as plt
import numpy as np
import os

tpath = '/work/60G/20G/collision_tensor.h5'

N = 100  # number of time-steps
dt = 1e-2  # time-step length

if not os.path.exists(tpath):
    raise Exception('Tensor file not found.')

ct = CollisionTensor(tpath)
basis = ct.basis
K = basis.get_K()

p2h = Polar2Hermite(K)

print('using Polar-Laguerre basis of polynomial degree ', K)

# define BKW initial distribution, located at z0=0+0j momentum and Temperature T=beta/2
bkw = BKWInitialDistribution(z0=0 + 0j, beta=2)
# generate initial coefficients in the Polar-Laguerre basis
C = project_polar_laguerre(bkw, basis, rsym=True)

coeffs = np.zeros(shape=(len(C), N + 1))
coeffs[:, 0] = C

moments = Moments(K)

# integrate in time using RK4
t = 0
for i in range(N):
    t += dt
    coeffs[:, i + 1] = ct.time_step(rk4, coeffs[:, i], dt, proj=True)
    # transform to Hermite basis
    H = p2h.to_hermite(coeffs[:, i + 1].copy())
    # compute moments
    p = moments.compute(
        H,
        # mass, v_x^2, v_y^2
        [0, 0, 2],
        [0, 2, 0])
    print('time %4.2g, rho: %3.2g, energy: %3.2g' %
          (t, p[0], (p[1] + p[2]) / p[0]))

# plot final solution
vx = np.linspace(-4, 4, 40)
VX, VY = np.meshgrid(vx, vx)
F = basis.evaluatez(coeffs[:, -1], VX + 1j * VY)
plt.pcolormesh(VX, VY, F)
plt.xlabel('v_x')
plt.ylabel('v_y')
plt.title('f(v)')
plt.colorbar()
plt.show()

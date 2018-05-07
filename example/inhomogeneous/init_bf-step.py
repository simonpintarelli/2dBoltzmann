#!/usr/bin/env python3

# generate initial conditions for the space inhomogeneous case
#

from pyboltz.ks_basis import KSBasis
from pyboltz.initialize import Maxwellian, project_to_polar_basis
from pyboltz.quadrature import TensorQuadFactory
from numpy import loadtxt, array, zeros
import h5py, sys
from spectral_tools import ShiftPolar

if __name__ == '__main__':

    fname_nodes = sys.argv[1]

    basis = KSBasis(K=None, w=None, fname='spectral_basis.desc')

    idx, x, y = loadtxt(fname_nodes, unpack=True)
    L = len(idx)
    N = len(basis)

    print("Phys DoFs: ", L)
    print("Velo DoFs: ", N)

    quad_handler = TensorQuadFactory()

    C = zeros((L, N))
    T1 = 1.0
    m1 = Maxwellian(m=1.4, e=2 * T1, u=0.0)
    C1 = project_to_polar_basis(m1, basis, quad_handler, nptsr=100, nptsa=120)

    shift_obj = ShiftPolar()
    for l in range(L):
        shift_obj.shift(C[l, :], C1, -3.0, 0)

    with h5py.File('coefficients.h5', 'w') as f:
        C = C.reshape((-1, 1))
        f.create_dataset('coeffs', shape=C.shape, data=C)
        f.create_dataset('C1', shape=C1.shape, data=C1)

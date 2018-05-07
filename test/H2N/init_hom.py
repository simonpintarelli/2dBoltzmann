#!/usr/bin/env python2

# generate initial conditions for the space inhomogeneous case
#


from pyboltz.ks_basis import KSBasis
from pyboltz.basis import get_basis
from pyboltz.initialize import Maxwellian, project_to_polar_basis
from pyboltz.quadrature import TensorQuadFactory
from numpy import *
import h5py, sys, os
from libshiftPolar import *


if __name__ == '__main__':

    #basis = KSBasis(K=30, w=0.5)
    basis = get_basis()

    quad_handler = TensorQuadFactory()
    T1 = 2.0
    m1 = Maxwellian(m=1.0, e=2*T1, u=1.5)
    C = project_to_polar_basis(m1, basis, quad_handler, nptsr=100, nptsa=120)
    # C1orig = C1.copy()
    # shift_obj = ShiftPolar()
    # C2 = zeros_like(C1)


    # shift_obj.shift(C2, C1, -1.0, -1.0)
    # #shift_obj.shift(C1, C2, 1.0, 1.0)


    f = h5py.File('coefficients.h5', 'w')
    C = C.reshape((-1,1))
    f.create_dataset('coeffs', shape=C.shape, data=C)


    f.close()

#!/usr/bin/env python2

# generate initial conditions for the space inhomogeneous case
#

#import mpl_toolkits.axes_grid1 as axes_grid1
from numpy import *
from pyboltz.basis import get_basis
from pyboltz.ks_basis import KSBasis
from pyboltz.collision_tensor import CollisionTensor
import h5py, sys, os, argparse
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix, csc_matrix

from eigenhdf import *




if __name__ == '__main__':

    parser = argparse.ArgumentParser('')
    parser.add_argument('--pg',  help='/path/to/petrov-galerkin-tensor')
    parser.add_argument('--np',  help='/path/to/galerkin-tensor')

    args = parser.parse_args()

    f5out = h5py.File('new_tensor.h5', 'w')
    f5pg = h5py.File(args.pg, 'r')
    f5np = h5py.File(args.np, 'r')

    for k in f5np.keys():
        entry = load_sparse_h5(f5np, k)
        save_sparse_h5(f5out, k, entry)

    for k in f5pg.keys():
        if k in f5out:
            continue
        else:
            entry = load_sparse_h5(f5pg, k)
            save_sparse_h5(f5out, k, entry)



    f5out.close()
    f5pg.close()
    f5np.close()

#!/usr/bin/env python3

# generate initial conditions for the space inhomogeneous case
#

#import mpl_toolkits.axes_grid1 as axes_grid1
from numpy import *
from pyboltz.basis import get_basis
from pyboltz.ks_basis import KSBasis
from pyboltz.collision_tensor import CollisionTensor
import h5py, sys, os, argparse
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix, csc_matrix

from pyboltz.utility import *


def build_transfer_matrix(V, W):

    pairs = []

    bW = list(W.elements)
    bV = list(V.elements)
    for j, elem in enumerate(bW):
        i = bV.index(elem)
        pairs.append((i, j))

    data = array(pairs)
    return csc_matrix(
        coo_matrix(
            (ones(data.shape[0]), (data[:, 0], data[:, 1])),
            shape=(len(V), len(W))))


def get_permutation(T):
    """
    T: transfer matrix
    """
    T = lil_matrix(T)
    n = T.shape[0]

    perm = []
    for i in range(n):
        perm.append(nonzero(T[i, :].todense() == 1)[1][0])
    return perm


def apply_permutation(list_in, perm):

    return [list_in[p] for p in perm]


if __name__ == '__main__':

    parser = argparse.ArgumentParser('')
    parser.add_argument('-i', help='/path/to/source/tensor')
    parser.add_argument('-o', help='/path/to/output')
    parser.add_argument('-K', help='target degree')

    args = parser.parse_args()

    K = int(args.K)

    basis = KSBasis(K=K, w=0.5)

    ct = CollisionTensor(os.path.join(args.i, 'collision_tensor.h5'))
    print('Converting tensor for K=%d to %d' % (ct.basis.get_K(), K))

    T = build_transfer_matrix(ct.basis, basis)

    # create output directory
    try:
        os.makedirs(args.o)
    except:
        pass

    h5out_path = os.path.join(args.o, 'collision_tensor.h5')

    if os.path.isfile(h5out_path):
        raise Exception("Destination file already exists!")

    h5out = h5py.File(h5out_path, 'w')

    for i, slice in enumerate(ct.slices):
        inew = basis.get_idx(ct.basis.elements[i])
        if inew is None:
            continue
        else:
            sp = T.T * slice * T
            save_sparse_h5(h5out, str(int(inew)), sp)

    save_sparse_h5(h5out, 'mass_matrix', T.T * ct.M * T)

    basis.write_desc(os.path.join(args.o, 'spectral_basis.desc'))

    test_basis = KSBasis(K=0, w=-1)
    perm = get_permutation(T.T)
    test_basis.elements = apply_permutation(ct.test_basis.elements, perm)
    test_basis.write_desc(os.path.join(args.o, 'spectral_basis_test.desc'))

    h5out.close()

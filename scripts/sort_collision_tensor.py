#!/usr/bin/env python2

#
#      Conversion script to convert collision tensors
#      to lexicographical ordering in (k,j,sin/cos)
#

from numpy import *
from pyboltz.collision_tensor import *
from scipy.sparse import  lil_matrix, csr_matrix, coo_matrix
import os
import numpy as np
import copy
import h5py

import argparse

# KJT: lexicographical ordering in (k, j, sin/cos)
ORD = { 'KJT' :
        lambda x: (x[0].fr.k, x[0].fr.j, x[0].fa.sc == np.cos)}

def sort_basis(trial_basis, test_basis, ordering_type ='KJT'):
    """
    Keyword Arguments:
    trial_basis   -- Basis type
    test_basis    -- Basis type
    ordering_type -- (default 'KJT')

    Warning: Permutation is created based on `trial_basis` alone.
             Make sure `test_basis` and `trial_basis` have same enumeration
             in k,j,sin/cos.

    Returns:

    new_trial_basis
    """

    new_trial_basis = copy.copy(trial_basis)
    new_test_basis = copy.copy(test_basis)

    N = len(trial_basis.elements)
    # trial functions
    ll_trial = list(zip(trial_basis.elements, range(N)))
    ll_trial_sorted = sorted(ll_trial, key=ORD[ordering_type])
    # test functions
    ll_test = list(zip(test_basis.elements, range(N)))
    ll_test_sorted = sorted(ll_test, key=ORD[ordering_type])

    perm = [x[1] for x in ll_trial_sorted]

    new_trial_basis.elements = [x[0] for x in ll_trial_sorted]
    new_test_basis.elements = [x[0] for x in ll_test_sorted]

    return (new_trial_basis, new_test_basis, perm)



def write_sparse_matrix(h5file, name, M):
    #dtype = np.dtype([('r', '<u4'), ('c', '<u4'), ('v', '<f8')])
    dtype = np.dtype([('r', 'int32'), ('c', 'int32'), ('v', 'float64')])
    M = coo_matrix(M)

    data = np.zeros(len(M.row), dtype=dtype)
    data['r'] = M.row
    data['c'] = M.col
    data['v'] = M.data

    dset = h5file.create_dataset(name, dtype=dtype, shape=data.shape, data=data)
    dset.attrs.create('shape', data=M.shape, dtype=np.int32)



def write_hdf(ct, perm, fname):
    N = len(ct.basis)
    P = lil_matrix((N,N))
    for i in range(N):
        P[perm[i], i] = 1.0
    if P.nnz != N:
        raise Exception('Failure in P')

    P = csc_matrix(P)
    Pt = P.T

    if not P.nnz == N:
        print 'Error in permutation matrix'

    if os.path.exists(fname):
        raise Exception('%s: already exists' % fname)
    fh5 = h5py.File(fname, 'w')

    # write to HDF
    M = Pt * ct.M * P
    write_sparse_matrix(fh5, 'mass_matrix', M)

    if not M.nnz == ct.M.nnz:
       raise Exception

    for i in range(N):
        S = Pt * ct.slices[perm[i]] * P
        if S.nnz == 0:
            S[0,0] = 0.0
        write_sparse_matrix(fh5, str(i), S)

    fh5.close()
    print('Collision-Tensor successfully written to %s' % fname)



def write_desc(fname, basis):
    f = open(fname, 'w')
    if os.path.exists(fname):
        fname += '.new'
    outstr = '\n'.join(map(lambda x: str(x), basis.elements))
    f.write(outstr)
    f.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input_path',
                        help='/path/to/source/tensor',
                        default='./')
    parser.add_argument('-o', dest='output_path',
                        help='/path/to/output', default='./')
    args = parser.parse_args()
    cwd = os.getcwd()
    outpath = args.output_path
    if not os.path.isabs(outpath):
        outpath = os.path.join(cwd, outpath)

    inpath = args.input_path
    if not os.path.isabs(inpath):
        inpath = os.path.join(cwd, inpath)

    if os.path.samefile(outpath, inpath):
        raise Exception("src and dest path are identical. Abort!")

    # goto src path
    os.chdir(inpath)
    ct = CollisionTensor()

    new_trial_basis, new_test_basis, perm = sort_basis(ct.basis, ct.test_basis)

    # goto output path
    os.chdir(outpath)

    fname = 'collision_tensor.h5'
    if os.path.exists(fname):
        fname = 'collision_tensor.h5.new'
    write_hdf(ct, perm, fname)

    write_desc('spectral_basis_test.desc', new_test_basis)
    write_desc('spectral_basis.desc', new_trial_basis)

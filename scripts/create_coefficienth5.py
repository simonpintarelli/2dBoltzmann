#!/usr/bin/env python2




import h5py
from numpy import *
import numpy as np
import os, sys, glob
import argparse
import re


if __name__ == '__main__':

    parser = argparse.ArgumentParser('')
    parser.add_argument('input')
    parser.add_argument('--v2d', default='vertex2dofidx.dat')
    args = parser.parse_args()
    h5_input =  args.input

    v2d_path = args.v2d

    if not os.path.exists(v2d_path):
        print("File with vertex positions `vertex2dofidx.dat` doest not exist")
        sys.exit(1)

    v2dofs = loadtxt(v2d_path)
    nn = v2dofs.shape[0]
    print 'len: ', nn
    dof_index = zeros(nn).astype(np.uint32)
    for i in range(nn):
        dof_index[v2dofs[i,0]] = v2dofs[i,1]

    L = v2dofs.shape[0]
    fh5_input = h5py.File(h5_input, 'r')

    # load input file
    keys = fh5_input.keys()
    step = int(keys[0])
    print 'step id found in input file was: ', step
    C = array(fh5_input[keys[0]]['Values']).reshape(-1,1)

    print 'L = ', L
    print 'len(C) = ', len(C)

    N = len(C)/L
    assert(len(C)%L == 0)

    print C.shape
    fh5_input.close()

    O = zeros_like(C)

    for i in range(L):
        io = N*i
        ii = N*dof_index[i]
        O[io:io+N] = C[ii:ii+N]

    h5_output = 'coefficients'
    if os.path.exists('coefficients.h5'):
        n = len(glob.glob('coefficients[0-9]*.h5'))
        os.rename('coefficients.h5', 'coefficients%d.h5' % n)

    fh5_output = h5py.File('coefficients.h5', 'w')
    dset = fh5_output.create_dataset('coeffs', shape=O.shape, data=O)
    dset.attrs['timestep'] = step
    fh5_output.close()

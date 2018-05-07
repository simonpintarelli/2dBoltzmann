#!/usr/bin/env python2


import sys, re, yaml, argparse, os, glob, h5py
from copy import copy
from numpy import *
from lxml import etree as ET



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rpath', help='path/to/restart/data')
    parser.add_argument('-restart_coeff', dest='rcoeff', default='coefficients.h5')
    parser.add_argument('-a', '--frame-begin', default=None)

    args = parser.parse_args()


    # find restart solution_vector
    restart_coeff_file = os.path.join(args.rpath, args.rcoeff)
    if not os.path.isfile(restart_coeff_file):
        print 'Error. File: `', restart_coeff_file, '` not found!'
        raise Exception('File not found')

    with h5py.File(restart_coeff_file, 'r') as fh5:
        try:
            last_index = fh5['coeffs'].attrs['timestep']
        except:
            last_index = None

    if not os.path.isfile(os.path.join(args.rpath, 'vertex2dofidx.dat')):
        print 'Error. `vertex2dofidx.dat` not found in source dir'
        raise Exception('File not found')

    current_dir = os.getcwd()
    dst_config = yaml.load(open('config.yaml', 'r').read())
    rst_config = yaml.load(open(os.path.join(args.rpath,  'config.yaml'), 'r').read())

    dst_offset = int(dst_config['TimeStepping']['dump'])
    rst_offset = int(rst_config['TimeStepping']['dump'])

    # ------------------------------------------------------------
    # Coefficient data (`solution_vector%d.h5`)
    get_coeff_idx = lambda x: int(re.match('solution_vector([0-9]+).*(h5|hdf5)', os.path.basename(x)).group(1))
    restart_coeff_files = glob.glob(os.path.join(args.rpath, 'solution_vector[0-9]*.h5'))
    restart_coeff_files = sorted(restart_coeff_files , key=get_coeff_idx)
    #idx = array(map(get_coeff_idx, restart_coeff_files))
    if last_index is None:
        print 'Warning: Dataset coefficients.h5:/coeffs (source dir) has no attribute `timestep`. Restart assumed from last solution_vector in dst dir.'
        old_coeff_files = glob.glob('solution_vector[0-9]*.h5')
        last_index = int(get_coeff_idx(sorted(old_coeff_files, key=get_coeff_idx)[-1]))
    print 'Data continues from timestep-id: ', last_index

    restart_idxs = map(get_coeff_idx, restart_coeff_files)

    # create symlinks
    coeff_files_data = []
    for i,idx in enumerate(restart_idxs):
        if idx == 0:
            continue
        #os.symlink
        print 'Processing idx: ', idx
        dest_file = 'solution_vector' + str(idx + last_index) + '.h5'
        if os.path.isfile(dest_file):
            print 'Oooops: File `', dest_file, 'exists. Abort!'
            raise Exception('File not found')
        else:
            os.symlink(restart_coeff_files[i], dest_file)
            # idx is required to read from the right dataset in solution_vectorXXX.h5
            coeff_files_data.append({'file' : dest_file, 'idx' : idx})

    # create `restart.yaml` (metadata)
    yaml_restart_data = {}
    for elem in coeff_files_data:
        yaml_restart_data[elem['file']] = {'dset' :str(elem['idx']) + '/Values'}

    # create symlink to vertex2dofidx.dat
    v2d_path = os.path.join(args.rpath, 'vertex2dofidx.dat')
    v2d_link = 'v2d-restart.dat'
    if os.path.isfile(v2d_link):
        if not os.islink(v2d_link):
            raise Exception('File ' + v2d_link + ' is not a symlink and won\'t be overwritten.')

    os.symlink(v2d_path, v2d_link)

    f = open('restart.yaml', 'w')
    f.write(yaml.dump({'files' : yaml_restart_data,
                       'first index' : int(restart_idxs[0] + last_index),
                       'v2d' : v2d_link}))

    f.close()


    # ------------------------------------------------------------
    # VTK data (`solution-%d.h5`)

    # ... TODO

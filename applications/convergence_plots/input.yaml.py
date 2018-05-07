#!/usr/bin/env python

# Create yaml-config required by `pp_convergence`
# this script reads `input.yaml` and creates `input.final.yaml`

import sys, re, yaml, argparse, os, glob
from numpy import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('input_dir')
    # parser.add_argument('reference_dir')

    parser.add_argument('-c', default='input.yaml')

    args = parser.parse_args()

    config = yaml.load(open(args.c, 'r'))

    # input and reference path
    pinput = config['input']['path']
    pref = config['reference']['path']

    # parser.add_argument('--data-prefix', default=('solution_vector', '.h5'), type=tuple, nargs=2)
    input_config = yaml.load(open(os.path.join(pinput, 'config.yaml'), 'r'))
    reference_config = yaml.load(open(os.path.join(pref, 'config.yaml'), 'r'))

    config['input']['grid'] = input_config['Mesh']['file']
    config['reference']['grid'] = reference_config['Mesh']['file']

    input_config['TimeStepping']['dump']
    reference_config['TimeStepping']['dump']

    dt_ref = float(reference_config['TimeStepping']['dt'])
    dt_input = float(input_config['TimeStepping']['dt'])

    dump_ref = reference_config['TimeStepping']['dump']
    dump_inp = input_config['TimeStepping']['dump']

    print('solution_vector available every %d, %d steps (reference, input)' % (
        dump_ref, dump_inp))
    key = lambda x: int(re.match('.*solution_vector([0-9]*).*', x).group(1))
    ref_solutions = sorted(
        glob.glob(os.path.join(pref, 'solution_vector[0-9]*.h5')), key=key)
    input_solutions = sorted(
        glob.glob(os.path.join(pinput, 'solution_vector[0-9]*.h5')), key=key)

    times_ref = array(map(key, ref_solutions)) * dt_ref
    times_input = array(map(key, input_solutions)) * dt_input

    tol = 1e-8
    pairs = []  # indices at equal time (reference, input)
    j = 0
    for i, tref in enumerate(times_ref):
        for k in arange(j, len(times_input)):
            if abs(tref - times_input[k]) < tol:
                pairs.append((i, k))
                j = k + 1
                break

    # read `restart.yaml` configuration (if existent)
    ref_restart_config = None
    try:
        with open(os.path.join(pref, 'restart.yaml'), 'r') as fin:
            print('found restart.yaml in reference dir')
            ref_restart_config = yaml.load(fin.read())
    except:
        pass
    inp_restart_config = None
    try:
        with open(os.path.join(pinput, 'restart.yaml'), 'r') as fin:
            inp_restart_config = yaml.load(fin.read())
    except:
        pass

    timesteps = []
    # check
    for i, j in pairs:
        fref = os.path.split(ref_solutions[i])[-1]
        finput = os.path.split(input_solutions[j])[-1]

        # is symlink? if yes, this file comes from a restarted calculation
        if not os.path.islink(os.path.join(pref, fref)):
            idref = key(ref_solutions[i])
            dref = {'data': fref + ':/%d/Values' % idref}
        else:
            if os.path.basename(fref) in ref_restart_config['files']:
                dset = ref_restart_config['files'][fref]['dset']
                dref = {
                    'data': fref + ':/' + dset,
                    'v2d': ref_restart_config['v2d']
                }
            else:
                raise ValueError

        # is symlink? if yes, this file comes from a restarted calculation
        if not os.path.islink(os.path.join(pinput, finput)):
            idinput = key(input_solutions[j])
            dinp = {'data': finput + ':/%d/Values' % idinput}
        else:
            if os.path.basename(finput) in inp_restart_config['files']:
                dset = inp_restart_config['files'][finput]['dset']
                dinp = {
                    'data': finput + ':/' + dset,
                    'v2d': inp_restart_config['v2d']
                }
            else:
                raise ValueError

        t = times_ref[i]

        assert (abs(times_ref[i] - times_input[j]) < tol)

        timesteps.append({
            'reference': dref,
            'input': dinp,
            'time': '%.4e' % t
        })

    config['timesteps'] = timesteps

    with open('input.final.yaml', 'w') as f:
        f.write(yaml.dump(config))
        f.close()

    # FUZZY = 1e8
    # times_ref = times_ref * FUZZY
    # times_input = times_input * FUZZY

    #solution_vectors = sorted(list(set(ref_solutions).intersection(set(input_solutions))))

import sys, subprocess, re, os, glob
import matplotlib as mpl
from numpy.linalg import norm
from numpy import *
from matplotlib.pyplot import *
from StringIO import StringIO

## output file:
# ---------------------------------------------------------------------------------------
# ___ERRORS__
# #i            t              l2_squared     l2_m_squared   l2_u_squared   l2_e_squared
# ...data
#


def get_data(fname):
    # trim output file
    out = []
    with open(fname, 'r') as f:
        toggle = False
        for l in f.readlines():
            if not toggle:
                if re.match('__ERRORS__', l):
                    toggle = True
            else:
                out.append(l)
    return '\n'.join(out)


dirs = ['ref002', 'ref003']

data = []

for dir in dirs:
    kdirs = filter(lambda x: os.path.isdir(x), glob.glob(os.path.join(dir, 'K[0-9]*')))
    kdirs = map(lambda x: os.path.split(x)[-1], kdirs)
    for kdir in kdirs:
        path = os.path.join(dir, kdir, 'errors.out')
        if os.path.isfile(path):
            c = StringIO(get_data(path))
            arr = loadtxt(c)
            data.append({'ref' : dir, 'K' : int(kdir[1:]), 'data' : arr})
        else:
            print 'Path: ', path, 'not found'


for d in data:
    t = d['data'][:,1]
    l2 = d['data'][:,2]
    semilogy(t, l2)
show()

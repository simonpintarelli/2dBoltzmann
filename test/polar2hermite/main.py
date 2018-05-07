from numpy import *
from pyboltz.basis import get_basis
import numpy as np
import h5py
from matplotlib.pyplot import *
from eigenhdf import load_sparse_h5

spectral_basis = get_basis()
S = np.matrix(spectral_basis.make_mass_matrix())
Sinv = np.matrix(diag(1/diag(S)))
matrices = []
with h5py.File('p2h.hdf5', 'r') as fh5:
    matrices = []
    for name, obj in fh5.items():
        try:
            k = int(name, base=10)
            matrices.append({'k': k, 'M': np.array(obj)})
        except:
            if not obj.dtype == np.dtype('float64'):
                P = load_sparse_h5(fh5, name)

matrices = [x['M'] for x in sorted(matrices, key=lambda x: x['k'])]
shapes = [x.shape[0] for x in matrices]
offsets = hstack((0, np.cumsum(shapes)))
n = offsets[-1]
M = np.matrix(np.zeros((n, n)))
for i in range(len(offsets)-1):
    M[offsets[i]:offsets[i+1], offsets[i]:offsets[i+1]] = matrices[i]

I = Sinv*P.T*M.T*M*P
err = abs(I-eye(*I.shape)).sum()
print('error (cwise.sum): ', err)

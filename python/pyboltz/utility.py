import scipy.sparse
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix


def load_trilinos_sparse_matrix(M, shape=None):
    """load sparse matrix (trilinos format) from hdf5

    :param M:
    :param shape:
    :returns: sparse matrix in coo format
    :rtype: scipy.sparse.coo_matrix

    """
    return scipy.sparse.coo_matrix(
        (M['VAL'], (M['ROW'], M['COL'])), shape=shape)


def spargmax(M):
    n, m = M.shape
    max_entry = {'val': -np.inf, 'idx': ()}
    for i in range(n):
        try:
            row = np.array(M.getrow(i).todense()).flatten()
            ak = np.nanargmax(row)
            if row[ak] > max_entry['val']:
                max_entry['val'] = row[ak]
                max_entry['idx'] = (i, ak)
        except ValueError:
            # it can happen that there is a row with all nan's
            continue

    return max_entry['val'], max_entry['idx']


def spargmax_rel_error(M1, Mref):
    assert (M1.shape == Mref.shape)

    M1 = csr_matrix(M1)
    Mref = csr_matrix(Mref)
    diff = M1 - Mref
    diff.data = np.sqrt(diff.data**2)
    Mref.data = np.sqrt(Mref.data**2)

    return spargmax(csr_matrix(diff / Mref))


def from_sparse_repr(m, shape=None):
    """load Eigen sparse matrix form hdf5

    :param m: container
    :param shape: shape
    :returns: sparse matrix in coo format
    :rtype: scipy.sparse.coo_matrix

    """
    return coo_matrix(
        (m["v"], (m["r"], m["c"])), shape=shape, dtype=np.float64)


def load_sparse_matrix(M, shape=None):
    if hasattr(M, 'keys'):
        return scipy.sparse.coo_matrix(
            (M['VAL'], (M['ROW'], M['COL'])), shape=shape)
    else:
        return scipy.sparse.coo_matrix(
            (M['v'], (M['r'], M['c'])), shape=shape, dtype=np.float64)


def to_sparse_repr(mat):
    # from https://github.com/garrison/eigen3-hdf5
    mat = coo_matrix(mat)
    dtype = [('r', int), ('c', int), ('v', mat.dtype)]
    m = np.zeros(shape=(mat.nnz, ), dtype=dtype)
    m["r"] = mat.row
    m["c"] = mat.col
    m["v"] = mat.data
    return m


def save_sparse_h5(grp, name, data):
    # from https://github.com/garrison/eigen3-hdf5
    grp[name] = to_sparse_repr(data)
    grp[name].attrs["shape"] = data.shape

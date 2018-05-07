from pyboltz.real_basis import Basis as RealBasis
from pyboltz.ks_basis import KSBasis as KSBasis


def get_basis(fname='spectral_basis.desc'):
    ksbasis = KSBasis(K=None, w=None, fname=fname)
    rlbasis = RealBasis(fname=fname)

    if len(rlbasis) > 0 and len(ksbasis) == 0:
        return rlbasis
    elif len(ksbasis) > 0 and len(rlbasis) == 0:
        return ksbasis
    else:
        raise Exception("error in reading basis from descriptor file")


def make_basis(K):
    """
    Keyword Arguments:
    K -- polynomial degree
    """
    return KSBasis(K=K, w=0.5, fname=None)

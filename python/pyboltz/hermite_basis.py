import re, os
import numpy as np


from .normalized_hermite import hermval_weighted, hermval


def read_hermite_basis(fname='spectral_basis.desc'):
    """
    Keyword Arguments:
    fname -- (default 'spectral_basis.desc')
    """
    pattern = 'H_([0-9]+),\s+fw_(.*)\s+H_([0-9]+),\s+fw_(.*)'
    f = open(fname,'r')

    elements = []
    for l in f.readlines():
        match = re.match(pattern, l)
        if match:
            k1 = int(match.group(1))
            w1 = float(match.group(2))
            k2 = int(match.group(3))
            w2 = float(match.group(4))
            elements.append( {'k1' : k1, 'w1' : w1, 'k2' : k2, 'w2' : w2} )

    return elements



def hermite_evaluatez(C, Z, N, weighted=True):
    """
    Evaluate 2-d Hermite series with weight exp(-|z|^2/2)

    Keyword Arguments:
    C -- coefficients as 1d array
    Z -- evaluation points Z = X + 1j Y
    N -- max degree in one direction
    """

    X = np.real(Z)
    Y = np.imag(Z)

    degx = np.array([]).astype(np.int)
    degy = np.array([]).astype(np.int)

    for i in range(N+1):
        t = np.arange(i+1).astype(np.int)

        degx = np.hstack((degx,t))
        degy = np.hstack((degy,t[::-1]))

    #return degx, degy

    assert(len(degx) == ((N+1)*(N+2))/2)

    if weighted:
        Fx = hermval_weighted(X, np.diag(np.ones(N+1)))
        Fy = hermval_weighted(Y, np.diag(np.ones(N+1)))
    else:
        Fx = hermval(X, np.diag(np.ones(N+1)))
        Fy = hermval(Y, np.diag(np.ones(N+1)))

    return np.einsum(Fx[...,degx]*Fy[...,degy], [1, 2, 3], C, [3])

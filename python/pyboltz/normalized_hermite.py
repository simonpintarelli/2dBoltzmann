import numpy as np


def hermval_weighted(x, c):
    """
    Keyword Arguments:
    x      -- k-dimensional ndarray of evalution points
    c      -- p-dimensional ndarray of series coefficients in ascending order

    Evaluates:
         n
    y = Sum  c[n] H_n(x)
        i=0

    n = c.shape[0], n-1 is the max polynomial degree
    c[n]'s are stored along the 0-th dimension of c

    Returns:
    y  -- first k dimension: points, last p-1 dimension: series
    """

    h0 = np.pi**(-0.25)*np.exp(-x**2/2.0)
    h1 = np.pi**(-0.25)*np.sqrt(2)*np.exp(-x**2/2.0)*x

    deg = c.shape[0]

    # evaluate hermite polynomials
    H = np.zeros(x.shape + (deg,))
    H[..., 0] = h0
    H[..., 1] = h1
    for l in range(2, deg):
        H[..., l] = hermnext(l, x, H[...,l-1], H[...,l-2])

    # H is n-dimensional
    # dimensions 1 ... n-1 => x
    # dimension n : l
    ind = np.arange(H.ndim+c.ndim-1)
    Hind = list(map(int, ind[:H.ndim]))
    cind = list(map(int, ind[H.ndim-1:]))
    return np.einsum(H, Hind, c, cind)


def hermval(x, c):
    """
    Keyword Arguments:
    x      -- k-dimensional ndarray of evalution points
    c      -- p-dimensional ndarray of series coefficients

    Evaluates:
         n
    y = Sum  c[n] H_n(x)
        i=0

    n = c.shape[0], n-1 is the max polynomial degree
    c[n]'s are stored along the 0-th dimension of c

    Returns:
    y  -- first k dimension: points, last p-1 dimension: series
          ... now it is the other way around
    """

    h0 = np.pi**(-0.25)
    h1 = np.pi**(-0.25)*np.sqrt(2)*x

    deg = c.shape[0]

    # evaluate hermite polynomials
    H = np.zeros(x.shape + (deg,))
    H[..., 0] = h0
    H[..., 1] = h1
    for l in range(2, deg):
        H[..., l] = hermnext(l, x, H[..., l-1], H[..., l-2])

    # H is n-dimensional
    # dimensions 1 ... n-1 => x
    # dimension n : l
    ind = np.arange(H.ndim+c.ndim-1)
    Hind = list(map(int, ind[:H.ndim]))
    cind = list(map(int, ind[H.ndim-1:]))
    return np.einsum(H, Hind, c, cind)


def hermnext(n, x, vn, vnm):
    """
    Keyword Arguments:
    n   -- order
    x   -- ...
    vn  -- v[n]
    vnm -- v[n-1]

    returns H_{n+1}
    """
    return np.sqrt(2.0 / float(n))*x*vn - np.sqrt((n-1) / float(n)) * vnm


def eval_hermite(x, n):
    """
    Keyword Arguments:
    x -- point
    n -- degree
    """
    h0 = np.pi**(-0.25)
    if n == 0:
        return h0

    h1 = h0*np.sqrt(2)*x
    if n == 1:
        return h1

    for l in range(2, n+1):
        h0 = hermnext(l, h1, h0)
        tmp = h1
        h1 = h0
        h0 = tmp

    return h1



if __name__ == '__main__':
    from numpy.polynomial.hermite import hermgauss
    from numpy.linalg import norm

    x,w = hermgauss(80)

    C = np.diag(np.ones(10))
    H = hermval(x, C)

    # compute overlap integrals, e.g. verify orthogonality condition
    R = np.dot(H.T, H * w.reshape((-1, 1)))
    # should be the diagonal matrix
    II = np.diag(np.ones(R.shape[0]))

    np.set_printoptions(precision=3)
    ERR = R-II
    print('errors', R-II)
    print('total error: ', norm(ERR, ord='fro'))

import numpy as np
import matplotlib.mlab as mlab
from bquad import hermw


def bary_eval(yy, xs, lam, x):
    """
    Baryzentrische Interpolations Formel

    Keyword Arguments:
    yy  -- f_i
    xs  -- x_i
    lam -- lambda_i
    x   -- evaluation points
    """
    # Funktionswerte an der Stelle x
    y = np.zeros_like(x)

    xs = xs.reshape(-1, 1)
    x = x.reshape(1, -1)

    # finde Evaluationspunkte welche sehr nahe an den Stuetzstellen liegen
    lxx = np.product(xs - x, axis=0)
    tol = 1e-20
    # array of boolean's
    idx = abs(lxx) < tol

    idx_n = mlab.find(idx)
    # finde die zugehoerigen Funktionswerte
    yidx = np.zeros(len(idx_n), dtype=int)
    for i, ii in enumerate(idx_n):
        d = abs(np.squeeze(x)[ii] - xs)
        yidx_local = mlab.find(d == min(d))
        yidx[i] = yidx_local
    # an den Stuetzstellen settz man direkt die
    # bekannten Funktionswerte
    y[idx_n] = yy[yidx]

    # x, mit genuegend Abstand zu Stuetzstellen
    idx = np.logical_not(idx)  # invertiere idx
    xr = np.squeeze(x)[idx].reshape(1, -1)
    tmp = (np.squeeze(lam) * np.squeeze(yy)).reshape(-1, 1)
    y2 = (lxx[idx]).reshape(1, -1) * np.sum(tmp / (xs - xr), axis=0)

    y[idx.flatten()] = y2.squeeze()
    return y


def lam(xs):
    """
    Berechnet das baryzentrische Gewicht lambda_k

    Keyword Arguments:
    xs      -- Stuetzstellen , len(xs) = n
    """
    N = len(xs)
    indices = np.arange(N)

    def lamk(k):
        idx = indices != k
        return 1. / np.product(xs[idx] - xs[k])

    return np.array([lamk(k) for k in indices])


def interpolate_nodal(N):
    """
    Keyword Arguments:
    N -- nodal coefficients
    """

    assert(N.shape[-1] == N.shape[-2])
    original_shape = N.shape
    nx = np.prod(N.shape[:-2], dtype=int)
    K = N.shape[-1]
    N = N.reshape((nx,) + N.shape[-2:])

    _, w = hermw(1.0, K)
    sw = np.sqrt(w)
    F = np.array(N) / np.outer(sw, sw)

    return F.reshape(original_shape)


if __name__ == '__main__':
    from bquad import hermw
    from numpy import *
    from matplotlib.pyplot import *
    import matplotlib as mpl
    import itertools

    N = 5
    x, w = hermw(1.0, N)

    A = max(x)

    xe = linspace(-A, A, 1000)

    ll = lam(x)
    figure(figsize=(12, 12))
    plot(x, zeros_like(x), 'o')

    cyc = itertools.cycle(mpl.rcParams['axes.prop_cycle'])

    for i in arange(N//2):
        yl = zeros_like(x)
        yl[i] = 1

        yu = zeros_like(x)
        yu[-(i+1)] = 1

        lprop = next(cyc)
        yle = bary_eval(yl, x, ll, xe)
        yue = bary_eval(yu, x, ll, -xe)
        plot(xe, yle, color=lprop['color'])
        plot(xe, yue, color=lprop['color'], linestyle='--')


    grid(True)
    show()

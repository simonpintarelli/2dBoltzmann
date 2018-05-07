import numpy as np
import numpy.polynomial.laguerre as laguerre
from bquad import MaxwellQuad

from warnings import warn


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs)
        return cls._instances[cls]


class Trapz1DQuad:
    def __init__(self, npts):

        # self.pts = np.linspace(-np.pi,np.pi,npts)
        self.pts, h = np.linspace(0, 2 * np.pi, npts, retstep=True)
        self.wts = np.array([h] * npts)
        self.wts[0] = 0.5 * h
        self.wts[-1] = 0.5 * h


class MidpointQuad:
    def __init__(self, npts):

        # self.pts = np.linspace(-np.pi,np.pi,npts)
        self.pts, self.h = np.linspace(
            0, 2 * np.pi, npts, endpoint=False, retstep=True)
        if npts == 1:
            self.h = 2 * np.pi
        self.wts = np.ones_like(self.pts) * self.h


class GLxxAlphaQuad:
    """
    Gauss-Laguerre Quadrature rule for [0,\infty]
    with the weight r*mu^(alpha), mu = exp(-r*r),
    i.e. for the substitution x = alpha*r^2
    """

    def __init__(self, alpha, npts):

        warn("This class is deprecated")
        if alpha <= 0:
            raise Exception("negative alpha")

        self.npts = npts
        self.alpha = alpha

        # transform the ordinary GL quad
        lagg_pts, lagg_wts = laguerre.laggauss(npts)

        self.pts = np.sqrt(lagg_pts / alpha)
        self.wts = 0.5 * lagg_wts / alpha

    def apply(self, f):
        """
        integrate f
        """

        return sum(f(self.pts) * self.wts)


class TensorQuad:
    def __init__(self, Q1, Q2):
        """
        Keyword Arguments:
        """
        # self.Q1pts = Q1.pts
        # self.Q2pts = Q2.pts
        # self.wts = np.outer( Q2.wts, Q1.wts )
        self.Q1 = Q1
        self.Q2 = Q2
        W1, W2 = np.meshgrid(Q1.wts, Q2.wts)
        self.wts = W1 * W2

        self.Q1pts, self.Q2pts = np.meshgrid(Q1.pts, Q2.pts)

    def complex_pts(self):
        return self.Q2pts * np.exp(1j * self.Q1pts)


class QuadBuffer:
    def __init__(self, ):
        self.container = {}

    def __getitem__(self, key):
        if key in self.container:
            return self.container[key]
        else:
            raise KeyError()

    def __setitem__(self, key, val):
        self.container[key] = val


class TensorQuadFactory:
    __metaclass__ = Singleton

    def __init__(self, quadR=MaxwellQuad, quadA=MidpointQuad):
        self.buffer = QuadBuffer()
        self._qR_quad = quadR
        self._qA_quad = quadA

    def make(self, nptsa, nptsr, alpha):
        """
        Keyword Arguments:
        (Midpoint rule) x (Maxwell quadrature)
        self  --
        nptsa -- no. quad. points in angle
        nptsr -- no. quad. points in radius
        alpha -- exponential weight factor in exp(-a r^2)

        Q1pts => angle
        Q2pts => radial
        """
        key = (nptsa, nptsr, alpha)
        try:
            return self.buffer[key]
        except KeyError:
            qR = self._qR_quad(alpha, nptsr)
            qA = self._qA_quad(nptsa)
            quad = TensorQuad(qA, qR)
            self.buffer[key] = quad
            return quad

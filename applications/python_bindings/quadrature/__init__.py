from __future__ import absolute_import

# from .libbquad import QuadMaxwell, QuadHermiteW, GenQuadMaxwell
from . import libbquad as _libbquad
import numpy as np


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **
                                                                 kwargs)
        return cls._instances[cls]


class MaxwellQuadBase():
    __metaclass__ = Singleton

    def __init__(self):
        self.quadrules = {}

    def get(self, N, p=1):
        """retrurn quadrature rule

        :param N: # nodes
        :param p: weight parameter r^p exp(-r^2/2)
        :returns: nodes and weights
        :rtype: np.array
        """
        k = (N, p)

        if k in self.quadrules:
            return self.quadrules[k]
        else:
            if p == 1:
                self.quadrules[k] = _libbquad.QuadMaxwell(1.0, N)
            else:
                self.quadrules[k] = _libbquad.GenQuadMaxwell(N, p)
            return self.quadrules[k]


class HermiteWQuadBase():
    __metaclass__ = Singleton

    def __init__(self):
        self.quadrules = {}

    def get(self, N):

        if N in self.quadrules:
            return self.quadrules[N]
        else:
            #print('new quadrature computed')
            self.quadrules[N] = _libbquad.QuadHermiteW(1.0, N)
            return self.quadrules[N]


class MaxwellQuad():
    """
    Wrapper class around Maxwell quadrature,
    i.e. a quadrature rule with weight x^p exp(x*x) on [0, inf)

    :param float alpha:
    :param int N: num. points
    :param int p:
    """

    def __init__(self, alpha, N, p=1):
        self.base = MaxwellQuadBase()
        quad = self.base.get(N, p)
        self.pts = quad.points() / np.sqrt(alpha)
        self.wts = quad.weights() / alpha


def maxwell_quad(alpha, N):
    """TODO: Docstring for maxwell_quad.

    :alpha:   Weight exp(-alpha x^2)
    :N:       No. of quadrature points
    :returns: pts, wts

    """
    base = MaxwellQuadBase()
    quad = base.get(N)
    pts = quad.points() / np.sqrt(alpha)
    wts = quad.weights() / alpha

    return pts, wts


def hermw(alpha, N):
    """TODO: Docstring for hermw.

    :alpha:  Weight exp(-alpha x^2)
    :N:      Number of quadrature points
    :returns: pts, wts

    """
    quad = HermiteWQuadBase()
    pts = quad.get(N).points() / np.sqrt(alpha)
    wts = quad.get(N).weights() / np.sqrt(alpha)

    return pts, wts


def maxwellw(alpha, N):
    """
    Analogue to hermw for Maxwell quadrature.

    :alpha: weight exp(-alpha x^2)
    :N:     num quad. points
    :returns: pts, wts

    Note:
    wts are scaled by exp(alpha x**2), e.g. the integrand has to be evaluated with the exp factor.
    """
    base = MaxwellQuadBase()
    quad = base.get(N)
    pts = quad.points() / np.sqrt(alpha)
    wts = quad.weights() * (np.exp(alpha * pts**2) / alpha)

    return pts, wts

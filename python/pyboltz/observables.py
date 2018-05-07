from __future__ import absolute_import

import numpy as np
from bquad import MaxwellQuad, hermw
from .quadrature import TensorQuadFactory, TensorQuad
from .hermite_basis import hermite_evaluatez


class Mass:
    def __init__(self, basis, quad=None):

        if quad is None:
            # 90 pts are commonly used, and therefore it is likely to find it in
            # cache (of the singleton instance)
            quad = MaxwellQuad(0.5, max(90, basis.get_K()))

        if isinstance(quad, TensorQuad):
            print('old method')
            self.weights = np.zeros(len(basis))

            for i, elem in enumerate(basis.elements):
                if not elem.fa.l == 0:
                    continue
                self.weights[i] = np.sum(
                    (elem.evaluate(quad.Q1pts, quad.Q2pts, weighted=False) *
                     quad.wts).reshape(-1))
        elif isinstance(quad, MaxwellQuad):
            # backwards compatibility
            self.weights = np.zeros(len(basis))

            for i, elem in enumerate(basis.elements):
                if not elem.fa.l == 0:
                    continue
                self.weights[i] = 2 * np.pi * np.sum(
                    elem.fr.evaluate(quad.pts, weighted=False) * quad.wts)
        else:
            raise ValueError('wrong argument')

    def compute(self, F):
        return np.dot(self.weights, F)


class Momentum:
    """
    Computes :math:` (\rho \mathbf{u})(\mathbf(v))`
    """

    def __init__(self, basis, quad=None):
        self.weightsX = np.zeros(len(basis))
        self.weightsY = np.zeros(len(basis))

        if quad is None:
            quad = MaxwellQuad(0.5, max(90, basis.get_K()))

        if isinstance(quad, TensorQuad):
            for i, elem in enumerate(basis.elements):
                if not elem.fa.l == 1:
                    continue
                PHI = quad.Q1pts
                R = quad.Q2pts
                self.weightsX[i] = np.sum(
                    (elem.evaluate(PHI, R, weighted=False) * R * np.cos(PHI) *
                     quad.wts).reshape(-1))
                self.weightsY[i] = np.sum(
                    (elem.evaluate(PHI, R, weighted=False) * R * np.sin(PHI) *
                     quad.wts).reshape(-1))

        elif isinstance(quad, MaxwellQuad):

            for i, elem in enumerate(basis.elements):
                if not elem.fa.l == 1:
                    continue

                R = quad.pts
                val = np.pi * np.sum(
                    elem.fr.evaluate(R, weighted=False) * R * quad.wts)

                if elem.fa.sc is np.cos:
                    self.weightsX[i] = val
                    self.weightsY[i] = 0
                else:
                    assert (elem.fa.sc is np.sin)
                    self.weightsX[i] = 0
                    self.weightsY[i] = val

    def compute(self, F):
        return np.array([np.dot(self.weightsX, F), np.dot(self.weightsY, F)])


class Energy:
    """
    Computes :math:` (\rho E)(\mathbf(v))`
    """

    def __init__(self, basis, quad=None):

        if quad is None:
            quad = MaxwellQuad(0.5, max(90, basis.get_K()))

        if isinstance(quad, TensorQuad):
            self.weights = np.zeros(len(basis))

            for i, elem in enumerate(basis.elements):
                if not elem.fa.l == 0:
                    continue
                R = quad.Q2pts
                self.weights[i] = np.sum(
                    (elem.evaluate(quad.Q1pts, quad.Q2pts, weighted=False) * R
                     * R * quad.wts).reshape(-1))
        elif isinstance(quad, MaxwellQuad):
            self.weights = np.zeros(len(basis))

            for i, elem in enumerate(basis.elements):
                if not elem.fa.l == 0:
                    continue
                r = quad.pts
                self.weights[i] = 2 * np.pi * np.sum(
                    elem.fr.evaluate(r, weighted=False) * r**2 * quad.wts)
        else:
            raise ValueError('wrong argument')

    def compute(self, F):
        return np.dot(self.weights, F)


class Moments:
    def __init__(self, K):
        """
        Keyword Arguments:
        K  -- K parameter of associated Polar-Laguerre basis
        """
        self.K = K
        x, w = hermw(0.5, K)
        self.VX, self.VY = np.meshgrid(x, x)
        self.WX, self.WY = np.meshgrid(w, w)
        self.Z = self.VX + 1j * self.VY

    def compute(self, H, mx, my):
        """
        Keyword Arguments:
        H  -- Hermite coefficients in 1d array (ordering as in C++ hermite basis)
        mx -- list of integers, moments for vx
        my -- list of integers, moements for vy
        """

        assert(len(mx) == len(my))

        F = hermite_evaluatez(H, self.Z, self.K - 1, weighted=True)

        Xp = self.VX[..., np.newaxis]**np.array(mx).reshape((1, 1, len(mx)))
        Yp = self.VY[..., np.newaxis]**np.array(my).reshape((1, 1, len(my)))

        moments = np.sum(
            F[..., np.newaxis] * Xp * Yp * self.WX[..., np.newaxis] *
            self.WY[..., np.newaxis],
            axis=(0, 1))
        return moments


class Entropy:
    """
    Computes :math:` (\rho H)(\mathbf{v})`
    """

    def __init__(self, basis, quad_handler=TensorQuadFactory):
        """
        Keyword Arguments:
        basis --
        quad  --
        """

        self.basis = basis
        self.quad_handler = quad_handler()

    def compute(self, C, qpts=None):
        """
        Keyword Arguments:
        C    -- Polar-Laguerre coefficient array
        qpts -- (nptsa, nptsr) number of quadrature points in radial or angular direction.
                Defaults to None, then number of points is chosen to be (2K, 2K), where K
                is the polynomial degree of the basis

        Returns:
        Entropy
        """

        w = self.basis.elements[0].w
        K = self.basis.get_K()
        if qpts is None:
            quad = self.quad_handler.make(2 * K, 2 * K, w)
        else:
            quad = self.quad_handler.make(qpts[0], qpts[1], w)

        PHI = quad.Q1pts
        R = quad.Q2pts
        F = self.basis.evaluate(C, PHI, R, weighted=False)

        logf = np.log(np.abs(F)) - R**2 * w
        h = (-1) * np.sum((F * logf * quad.wts).reshape(-1))
        return h

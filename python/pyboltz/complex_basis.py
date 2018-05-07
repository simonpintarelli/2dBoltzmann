import numpy as np


def laguerre_eval_zero(deg, pts):
    return np.polynomial.laguerre.lagval(pts, np.append(np.zeros(deg), 1))


def laguerre_eval_one(deg, pts):
    return np.polynomial.laguerre.lagval(pts, np.ones(deg + 1))



class RadialBasis:
    def __init__(self, beta):
        self.beta = float(beta)

    def eval(self, k, r, weighted=True):
        if k % 2 == 0:
            B = laguerre_eval_zero(k / 2, r ** 2)
        else:
            B = np.sqrt(2.0) / np.sqrt(k + 1) * r * laguerre_eval_one((k - 1) / 2, r ** 2)
        if weighted:
            return B * np.exp(-r**2 / self.beta)
        else:
            return B

    def eval_all(self, C, r, weighted=True):
        result = np.zeros_like(r)
        for k, c in enumerate(C):
            result += np.real(c)*self.eval(k, r, weighted)

        return result



def laguerre_approx(K, L, beta, a, q, f):
    """
    Keyword Arguments:
    K    -- number of coefficients
    L    -- number of angular modes
    beta -- basis exponent
    a    -- f = poly(f)*exp(-a r^2)
    q    -- quadrature rule, a functional of alpha, the exponent
    f    -- function
    """
    C = np.zeros((K,L), dtype=np.complex)

    # compute the total exponent
    nu = 1-1./beta + a
    quad = q(nu)
    b = RadialBasis(beta)

    for k in range(0, K, 2):
        C[k,0] = 2*np.pi * sum(f(quad.pts) * b.eval(k, quad.pts, False) * quad.wts) / np.pi

    # # compute approximation error in l2-norm
    # || f - fapprox || = \int -2*f*fapprox + fapprox^2 + f^2 r dr
    #                                ^            ^        ^
    #     exponents:                nu12        nu11      nu22

    # factor out the largest common exponent
    gamma = min( min(a+1./beta, 2*a) , 2./beta )
    nu12 = a + 1./beta - gamma
    nu11 = 2./beta - gamma
    nu22 = 2*a-gamma

    quad = q(gamma)
    pts = quad.pts
    wts = quad.wts
    fapprox = b.eval_all(C[:,0], pts, weighted=False)
    fex = f(pts)
    mu = np.exp(-pts*pts)
    l2_err2 = np.sum((-2*fex*fapprox*mu**nu12 + fapprox**2*mu**nu11 + fex**2*mu**nu22)*wts)
    l2_norm = np.sqrt(np.sum(fex**2*mu**nu22*wts))

    return C, np.sqrt(np.abs(l2_err2))/l2_norm

class Basis:
    def __init__(self, nK, nL, beta):

        self.nK = nK
        self.nL = nL
        self.beta = beta


    def q_to_l(self, k, q):
        """

        Arguments:
        - `k`:
        - `q`:
        """
        l = 2 * q if q < self.nL / 2 else 2 * (q - self.nL)
        return l + (k % 2)


    def l_to_q(self, k, l):
        q = (l - (k % 2)) / 2
        return q if q >= 0 else q + self.nL


    def eval(self, l, k, r, phi, eval_weight=True):
        """
        eval( l, k, r, phi)
        """
        if k % 2 == 0:
            B = laguerre_eval_zero(k / 2, r ** 2) * np.exp(1j * float(l) * phi)
            if eval_weight == True:
                B *= np.exp(-r ** 2 / self.beta)
            return B
        else:
            B = np.sqrt(2.0) / np.sqrt(k + 1) * r * laguerre_eval_one(
                (k - 1) / 2, r ** 2) * np.exp(1j * float(l) * phi)
            if eval_weight == True:
                B *= np.exp(-r ** 2 / self.beta)
            return B


    def eval_cart(self, coeffs, X, Y, eval_weight=True):
        result = np.zeros(X.shape, dtype=complex)
        R = np.sqrt(X * X + Y * Y)
        PHI = np.arctan2(Y, X)

        for q in range(self.nL):
            for k in range(self.nK):
                l = self.q_to_l(k, q)
                result += self.eval(l, k, R, PHI, eval_weight) * coeffs[k, q]

        return result

    def eval_cartKL(self, K, L, coeffs, X, Y, eval_weight=True):
        """
        Evalutes all coefficients K x L
        Keyword Arguments:
        K           -- even integer
        L           -- even integer
        coeffs      -- !!! coeffs must have dimension self.K x self.L
        X           --
        Y           --
        eval_weight -- (default True)
        """
        assert coeffs.shape[0] == self.nK
        assert coeffs.shape[1] == self.nL

        result = np.zeros(X.shape, dtype=complex)
        R = np.sqrt(X * X + Y * Y)
        PHI = np.arctan2(Y, X)

        for k in range(K):
            for l in range(-L + (k % 2), L, 2):
                q = self.l_to_q(k, l)
                result += self.eval(l, k, R, PHI, eval_weight) * coeffs[k, q]

        return result

    def eval_KL(self, K, L, coeffs, R, PHI, eval_weight=True):
        """
        Evalutes all coefficients K x L
        Keyword Arguments:
        K           -- even integer
        L           -- even integer
        coeffs      -- !!! coeffs must have dimension self.K x self.L
        X           --
        Y           --
        eval_weight -- (default True)
        """
        assert coeffs.shape[0] == self.nK
        assert coeffs.shape[1] == self.nL

        result = np.zeros(R.shape, dtype=complex)

        for k in range(K):
            for l in range(-L + (k % 2), L, 2):
                q = self.l_to_q(k, l)
                result += self.eval(l, k, R, PHI, eval_weight) * coeffs[k, q]
        return result

    def eval_grid(self, coeffs, vmin, vmax, npts=100):
        # vxx = np.linspace(vmin,vmax,100)
        # U,V = np.meshgrid(vxx,vxx)
        U, V = np.mgrid[vmin:vmax:1j * npts, vmin:vmax:1j * npts]
        # for q in range(self.nL):
        #     for k in range(self.nK):
        #         l = self.q_to_l(k,q)
        #         result += self.eval(l,k, R, PHI)*coeffs[k,q]
        return U, V, self.eval_cart(coeffs, U, V, eval_weight=True)

    def eval_gridKL(self, K, L, coeffs, vmin, vmax, npts=100):
        # vxx = np.linspace(vmin,vmax,100)
        # U,V = np.meshgrid(vxx,vxx)
        U, V = np.mgrid[vmin:vmax:1j * npts, vmin:vmax:1j * npts]
        # for q in range(self.nL):
        #     for k in range(self.nK):
        #         l = self.q_to_l(k,q)
        #         result += self.eval(l,k, R, PHI)*coeffs[k,q]
        return U, V, self.eval_cartKL(K, L, coeffs, U, V, eval_weight=True)

import numpy as np
import re


def laguerre_eval_zero(deg, pts):
    return np.polynomial.laguerre.lagval(pts, np.append(np.zeros(deg), 1))


def laguerre_eval_one(deg, pts):
    return np.polynomial.laguerre.lagval(pts, np.ones(deg + 1))


class RadialBasisElem:
    def __init__(self, w, k):
        self.w = w
        self.k = k
        if k % 2 == 0:
            self.f = lambda x: laguerre_eval_zero(k/2, x*x)
        else:
            self.f = lambda x: (np.sqrt(2.0) / np.sqrt(k + 1)*x *
                                laguerre_eval_one((k-1)/2, x*x))

    def __str__(self):
        return 'RadialBasisElem(k=%d, w=%f)' % (self.k, self.w)

    def evaluate(self, x, weighted=True):
        B = self.f(x)
        if weighted:
            return B*np.exp(-x*x*self.w)
        else:
            return B


class SpectralBasisElem:
    def __init__(self, sincos, l, w, k):
        self.a = np.sin if sincos == 'sin' else np.cos
        self.sincos = sincos
        self.k = k
        self.l = l
        self.w = w
        self.radial_elem = RadialBasisElem(w, k)

    def evaluate(self, phi, r, weighted=True):
        return self.radial_elem.evaluate(r, weighted) * self.a(self.l*phi)

    def __str__(self):
        return '%s(%d), (fw_%.3f, k_%d)' % (self.sincos,
                                            self.l, self.w, self.k)


class Basis:

    def __init__(self, fname='spectral_basis.desc', K=None, L=None, beta=2):
        """
        Parsers the file fname containing the basis description

        Keyword Arguments:
        self  --
        fname -- basis descriptor (output from c++)
        """
        if K is None and L is None:
            self.init_from_file(fname)
        elif fname is None:
            self.init_from_params(K, L, beta)
        else:
            raise Exception("incompatible calling arguments")

    def init_from_params(self, K, L, beta):
        self.elements = []
        for k in range(0, K, 2):
            # add cos
            self.elements.append(SpectralBasisElem('cos', 0, 1./beta, k))
        for l in range(1, L+1):
            if l % 2 == 0:
                for k in range(0, K, 2):
                    self.elements.append(SpectralBasisElem('cos', l, 1./beta, k))
                    self.elements.append(SpectralBasisElem('sin', l, 1./beta, k))
            else:
                for k in range(1, K, 2):
                    self.elements.append(SpectralBasisElem('cos', l, 1./beta, k))
                    self.elements.append(SpectralBasisElem('sin', l, 1./beta, k))

    def init_from_file(self, fname):
        f = open(fname, 'r')
        pat = '([a-z]{3})_([0-9]+)\s+[(]beta_(.*),\sk_([0-9]+)[)]'
        pat_new = '([a-z]{3})_([0-9]+)\s+[(]fw_(.*),\sk_([0-9]+)[)]'
        self.elements = []
        for l in f.readlines():
            match = re.match(pat, l)
            if match:
                sincos = match.group(1)
                l = int(match.group(2))
                beta = float(match.group(3))
                k = int(match.group(4))
                self.elements.append(SpectralBasisElem(sincos, l, 1./beta, k))

            # new basis descriptor
            match = re.match(pat_new, l)
            if match:
                sincos = match.group(1)
                l = int(match.group(2))
                fw = float(match.group(3))
                k = int(match.group(4))
                self.elements.append(SpectralBasisElem(sincos, l, fw, k))

    def __len__(self):
        return len(self.elements)

    def get_elements(self):
        """
        return elements (list of spectral basis elements)
        i.e. [b_0(r, phi), ..., b_N(r, phi) ]
        """
        return self.elements

    def max_l(self):
        return max(self.elements, key=lambda x: x.l).l

    def max_k(self):
        return max(self.elements, key=lambda x: x.k).k

    def evaluate(self, C, phi, r, weighted=True):
        """

        Keyword Arguments:
        C    -- coefficients
        phi  -- ...
        r    -- ...

        Return Arguments:
        y    -- f(phi, r)
        """

        if len(C) != len(self.elements):
            raise Exception

        y = np.zeros_like(r)
        for i, elem in enumerate(self.elements):
            y += elem.evaluate(phi, r, weighted=weighted)*C[i]
        return y

    def ranges(self, predicate):
        """
        returns a boolean index vector with True whenever predicate(elem)
        evaluates to true

        Keyword Arguments:
        self      --
        predicate --
        """

        x = np.zeros(len(self)).astype(np.bool)
        for i, elem in enumerate(self.elements):
            x[i] = predicate(elem)

        return np.where(x)[0]

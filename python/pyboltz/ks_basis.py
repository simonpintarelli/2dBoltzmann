import numpy as np
from scipy.special import eval_genlaguerre, gamma
from scipy.sparse import csc_matrix, coo_matrix, lil_matrix
import re

def gammaqot(k, a):
    """
    Keyword Arguments:
    k -- integer
    a -- integer

    Returns: Gamma(k+1) / Gamma(k+a+1)
    """
    assert (k >= 0 and a >= 0)
    assert (type(k) is int and type(a) is int)
    return 1.0 / np.prod(np.arange(k + 1, k + a + 1, dtype=np.float64))


class RadialBasisElem:
    """
    Radial part of the Polar-Laguerre basis function:

    .. math::

        r^{2j + k \op{mod} 2} L_{floor(k/2)-1}^{2j + k \op{mod} 2} (r^2) e^{-w*r^2}

    Parameters:
    j : angular index
    k : radial index
    w : exp weight
    """

    def __init__(self, j, k, w=0.5):
        """
        Keyword Arguments:
        self --
        j    -- Angular index
        k    -- Radial index
        w    -- weight
        """

        self.j = j
        self.k = k
        self.w = w
        self.deg = (self.k - self.k % 2) // 2 - self.j
        self.alpha = 2 * self.j + (self.k % 2)

        # basis element is a polynomial with degree p
        self.p = 2 * self.deg + self.alpha

    def evaluate(self, r, weighted=True):
        return self.evaluatez(r, weighted)

    def evaluatez(self, z, weighted=True):
        """
        Parameters:
        -----------
        z : evaluation points from :math:`\mathbb{R}^2` stored as complex, e.g. :math:`z = x + i y`

        Returns:
        --------
        np.array
            function value at z
        """
        r = np.abs(z)

        scaling_old = np.sqrt(
            (1.0 * gamma(self.deg + 1)) / gamma(self.deg + self.alpha + 1))
        scaling = np.sqrt(gammaqot(self.deg, self.alpha))
        assert (abs(scaling_old - scaling) < 1e-9)
        f = (eval_genlaguerre(self.deg, self.alpha, r**2) * r**(self.alpha) * scaling)

        if weighted:
            return f * np.exp(-r**2 * self.w)
        else:
            return f

    def __str__(self):
        return '(fw_%.1f, k_%d, j_%d)' % (self.w, self.k, self.j)

    def __eq__(self, other):
        return self.j == other.j and self.k == other.k

    def __lt__(self, other):
        return (self.k, self.j) < (other.k, other.j)

    def __gt__(self, other):
        return (self.k, self.j) > (other.k, other.j)

    def __hash__(self):
        return hash((self.j, self.k, self.w, self.alpha))

    def evenq(self):
        return (self.alpha % 2 == 0)

    @staticmethod
    def get_max_r_deg(list_of_elements):
        """
        Keyword Arguments:
        list_of_elements --
        """

        return max(x.fr.p for x in list_of_elements)


class AngularBasisElem:
    def __init__(self, l, sc='c'):
        """
        Keyword Arguments:
        self --
        l    --
        sc   -- (default 'c') 's' for sin
        """
        self.l = l

        if sc == 'c':
            self.sc = np.cos
        elif sc == 's':
            self.sc = np.sin
        else:
            raise Exception("wrong argument")

    def evaluate(self, phi):
        """
        Keyword Arguments:
        self --
        phi  --
        """
        return self.sc(phi * self.l)

    def __str__(self):
        if self.sc == np.sin:
            return 'sin_%d' % self.l
        elif self.sc == np.cos:
            return 'cos_%d' % self.l
        else:
            raise Exception()

    def __eq__(self, other):
        return self.l == other.l and self.sc == other.sc

    def __hash__(self):
        return hash((self.l, self.sc))

    @staticmethod
    def get_max_l(list_of_elements):
        """
        Keyword Arguments:
        list_of_elements --
        """
        elem = max(list_of_elements, key=lambda x: x.fa.l)
        return elem.fa.l


class SpectralElem:
    fa_type = AngularBasisElem
    fr_type = RadialBasisElem

    def __init__(self, fa, fr):
        """
        Keyword Arguments:
        self --
        fa   -- angular basis function
        fr   -- radial basis function
        """
        assert (isinstance(fa, AngularBasisElem))
        assert (isinstance(fr, RadialBasisElem))
        self.fa = fa
        self.fr = fr
        self.w = self.fr.w

    def evaluate(self, phi, r, weighted=True):
        return self.fa.evaluate(phi) * self.fr.evaluate(r, weighted)

    def evaluatez(self, z, weighted=True):
        return self.fa.evaluate(np.angle(z)) * self.fr.evaluate(
            np.abs(z), weighted)

    def __str__(self):
        return str(self.fa) + '\t' + str(self.fr)

    def __eq__(self, other):
        return self.fa == other.fa and self.fr == other.fr

    def __hash__(self):
        return hash((hash(self.fr), hash(self.fa)))


class KSBasis:
    def __init__(self, K, w=0.5, fname=None):
        """
        Keyword Arguments:
        self --
        K    --
        w    --
        """

        self.elem_type = SpectralElem

        self.elements = []
        self.w = w

        if fname is None:
            self._init_from_values(K, w)
            # sort
            t = {np.sin: 's', np.cos: 'c'}
            self.elements = sorted(
                self.elements, key=lambda x: (x.fa.l, t[x.fa.sc], x.fr.k))
        else:
            self._init_from_desc(fname)
            self.w = None

    def _init_from_values(self, K, w):
        """

        """
        for k in range(K):
            # sin part
            for j in range(1 - k % 2, k // 2 + 1):
                l = 2 * j + k % 2
                self.elements.append(
                    self.elem_type(
                        AngularBasisElem(l, sc='s'), RadialBasisElem(j, k, w)))
            # cos part
            for j in range(0, k // 2 + 1):
                l = 2 * j + k % 2
                self.elements.append(
                    self.elem_type(
                        AngularBasisElem(l, sc='c'), RadialBasisElem(j, k, w)))
        self.elements = np.array(self.elements)

    def get_max_l(self):
        return self.elem_type.fa_type.get_max_l(self.elements)

    def get_max_r_deg(self):
        return self.elem_type.fr_type.get_max_r_deg(self.elements)

    def get_K(self):
        return self.elem_type.fr_type.get_max_r_deg(self.elements) + 1

    def _init_from_desc(self, fname):
        """
        initalize from descriptor file

        Keyword Arguments:
        self  --
        fname --
        """

        pattern = '([a-z]{3})_([0-9]+)\s+[(]fw_(.*),\s+k_([0-9]+),\s+j_([0-9]+)[)]'

        with open(fname, 'r') as f:
            for line in f.readlines():
                match = re.match(pattern, line)
                if match:
                    sincos = match.group(1)
                    l = int(match.group(2))
                    w = float(match.group(3))
                    k = int(match.group(4))
                    j = int(match.group(5))
                    if sincos == 'sin':
                        fa = AngularBasisElem(l, sc='s')
                    else:
                        fa = AngularBasisElem(l, sc='c')

                    fr = RadialBasisElem(j, k, w)
                    self.elements.append(SpectralElem(fa, fr))
            self.elements = np.array(self.elements)

    def evaluate(self, C, phi, r, weighted=True):
        assert (len(C) == len(self.elements))

        sum = 0
        for c, elem in zip(C, self.elements):
            sum += c * elem.evaluate(phi, r, weighted)

        return sum

    def make_mass_matrix(self):
        """

        """
        n = len(self)

        M = np.zeros((n, n))
        for i, elem in enumerate(self.elements):
            if elem.fa.l == 0:
                M[i, i] = np.pi
            else:
                M[i, i] = np.pi / 2
        return M

    def evaluatez(self, C, z, weighted=True):
        assert (len(C) == len(self.elements))

        phi = np.angle(z)
        r = np.abs(z)
        sum = 0
        for c, elem in zip(C, self.elements):
            sum += c * elem.evaluate(phi, r, weighted)

        return sum

    def evaluate_prod(self, C, Phi, R, weighted=True):
        sum = 0
        for c, elem in zip(C, self.elements):
            sum += np.outer(
                elem.fr.evaluate(R, weighted), c * elem.fa.evaluate(Phi))
        return sum

    def get_idx(self, elements):

        if not hasattr(elements, '__iter__'):
            elements = [elements]

        idx = []
        for elem in elements:
            matches = np.where(np.array(self.elements) == elem)
            if len(matches[0]) == 0:
                return None
            idx.append(matches[0][0])

        return np.array(idx).astype(np.uint)

    def __len__(self):
        """
        Keyword Arguments:
        self --
        """
        return len(self.elements)

    def write_desc(self, fname='spectral_basis.desc'):
        f = open(fname, 'w')
        outstr = '\n'.join(map(lambda x: str(x), self.elements))
        f.write(outstr)
        f.close()


def build_transfer_matrix(V, W):
    """
    transfer from W -> V

    Returns: scipy sparse matrix (csc_matrix)
    """
    pairs = []
    if type(V) is not list:
        bV = list(V.elements)
    else:
        bV = V

    if type(W) is not list:
        bW = list(W.elements)
    else:
        bW = W

    for j, elem in enumerate(bW):
        i = bV.index(elem)
        pairs.append((i, j))

    data = np.array(pairs)
    return csc_matrix(
        coo_matrix(
            (np.ones(data.shape[0]), (data[:, 0], data[:, 1])),
            shape=(len(V), len(W))))


def get_permutation(T):
    """
    T: transfer matrix
    """
    T = lil_matrix(T)
    n = T.shape[0]

    perm = []
    for i in range(n):
        perm.append(np.nonzero(T[i, :].todense() == 1)[1][0])
    return perm


def apply_permutation(list_in, perm):

    return [list_in[p] for p in perm]

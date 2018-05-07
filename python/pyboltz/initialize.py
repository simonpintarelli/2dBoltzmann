import numpy as np
from pyboltz.ks_basis import KSBasis

from scipy.linalg import solve


def diagonal_mass_matrix(basis, quad_handler=None):
    """
    computes < b_i, b_i>
    in l2-norm

    use only with beta=2 basis!!!

    Keyword Arguments:
    basis -- polar basis
    quad_handler
    """

    if isinstance(basis, KSBasis):
        return basis.make_mass_matrix()
    else:
        N = len(basis)
        M = np.eye(N, N)
        for i, elem in enumerate(basis.elements):
            if elem.fa.l == 0:
                M[i, i] = np.pi
            else:
                M[i, i] = np.pi / 2.
        return M


def project_polar_laguerre(f, basis, rsym=False):
    """
    Keyword Arguments:
    f     -- function object (must be of super type MaxwellBaseFunction)
    basis -- spectral_basis
    rsym  -- (default False) radial symmetric
    """
    from .quadrature import TensorQuadFactory
    from bquad import maxwell_quad

    N = len(basis)
    R = np.zeros(N)
    K = basis.get_K()
    fq = f.quad_evaluator()

    if rsym:
        pts, wts = maxwell_quad(alpha=f.w() + 0.5, N=max(2 * K, 90))
        FQ = fq(pts)

        for i, test_elem in enumerate(basis.elements):
            if test_elem.fa.l == 0:
                R[i] = 2 * np.pi * np.sum(
                    wts * FQ * test_elem.evaluatez(pts, weighted=False))
    else:
        quad_handler = TensorQuadFactory()
        quad = quad_handler.make(
            nptsa=max(100, 2 * K),
            nptsr=max(2 * K, 90),
            alpha=f.w() + basis.elements[0].w)
        z = quad.Q2pts * np.exp(1j * quad.Q1pts)
        FQ = fq(z)
        for i, test_elem in enumerate(basis.elements):
            R[i] = np.sum((FQ * test_elem.evaluate(
                quad.Q1pts, quad.Q2pts, weighted=False) *
                           quad.wts).reshape(-1))

    M = basis.make_mass_matrix()
    C = solve(M, R)

    return C


def project_to_polar_basis(f,
                           basis,
                           quad_handler,
                           nptsa,
                           nptsr,
                           selector=lambda x: True):
    """
    Keyword Arguments:
    f            -- an object with super type MaxwellBaseFunction
    basis        --
    quad_handler --
    nptsa        -- (default 60)
    nptsr        -- (default 60)
    """
    M = basis.make_mass_matrix()

    N = len(basis)
    R = np.zeros(N)

    fq = f.quad_evaluator()

    quad = quad_handler.make(
        nptsa=nptsa, nptsr=nptsr, alpha=f.w() + basis.elements[0].w)
    z = quad.Q2pts * np.exp(1j * quad.Q1pts)
    FQ = fq(z)
    for i, test_elem in enumerate(basis.elements):
        if not selector(test_elem):
            continue
        R[i] = np.sum((FQ * test_elem.evaluate(
            quad.Q1pts, quad.Q2pts, weighted=False) * quad.wts).reshape(-1))
    C = solve(M, R)

    return C


def project(Mlu,
            f,
            test_basis,
            quad_handler,
            nptsa,
            nptsr,
            selector=lambda x: True):
    """
    Keyword Arguments:
    Mlu          -- lu decomposition from mass matrix (there is one in CollisionTensor)
    f            -- function to approximate must be of super type `MaxwellBaseFunction`
    test_basis   -- test_functions
    quad_handler -- quadrature handler
    nptsa        -- number of quadrature points in angle
    nptsr        -- number of quadrature points in radius
    """
    R = np.zeros(len(test_basis))

    fq = f.quad_evaluator()
    quada = quad_handler.make(nptsa=nptsa, nptsr=nptsr, alpha=f.w() + 0.5)
    quadb = quad_handler.make(nptsa=nptsa, nptsr=nptsr, alpha=f.w())
    za = quada.Q2pts * np.exp(1j * quada.Q1pts)
    zb = quadb.Q2pts * np.exp(1j * quadb.Q1pts)

    FQa = fq(za)
    FQb = fq(zb)

    for i, test_elem in enumerate(test_basis.elements):
        if not selector(test_elem):
            continue
        if test_elem.fr.w == 0:
            R[i] = np.sum((FQb * test_elem.evaluate(
                quadb.Q1pts, quadb.Q2pts, weighted=False) *
                           quadb.wts).reshape(-1))
        else:
            R[i] = np.sum((FQa * test_elem.evaluate(
                quada.Q1pts, quada.Q2pts, weighted=False) *
                           quada.wts).reshape(-1))
    C = Mlu.solve(R)

    return C


def project_nodal(f, K, w=0.5):
    """
    Keyword Arguments:
    f -- f(x, y)
    K -- degree
    w -- weight: exp(-w x**2)

    Returns:
    C -- nodal coefficients
    """
    from bquad import hermw

    x, w = hermw(2 * w, K)
    X, Y = np.meshgrid(x, x)
    sw = np.sqrt(w)
    C = f(X, Y) * np.outer(sw, sw)
    return C


def project_nodal_array(F):
    """
    Keyword Arguments:
    F -- f(x, y) given at the Gauss-Hermite quadrature nodes
         if F has more than two dimension, the last two dimensions must be of
         size K x K, the transformation is then applied along the leading dimensions.

    Returns:
    C -- nodal coefficients
    """
    from bquad import hermw
    K = F.shape[-1]
    assert(F.shape[-2] == F.shape[-1])
    original_shape = F.shape
    nx = np.prod(original_shape[:-2], dtype=int)
    F = F.reshape((nx, K, K))
    w = 0.5
    x, w = hermw(2 * w, K)
    sw = np.sqrt(w)
    C = F * np.outer(sw, sw)
    return C.reshape(original_shape)


def gaussians_with_offset(phi, r):
    x = np.array(r * np.cos(phi)).astype(np.float128)
    y = np.array(r * np.sin(phi)).astype(np.float128)

    return np.exp(-(x - 1)**2 - (y - 1)**2) + np.exp(-(x + 1)**2 - (y + 1)**2)


def gaussians_center_with_offset(phi, r):
    x = np.array(r * np.cos(phi)).astype(np.float128)
    y = np.array(r * np.sin(phi)).astype(np.float128)

    return np.exp(-x**2 - y**2) + np.exp(-(x - 1)**2 - (y - 1)**2)


def crossed_streams(phi, r):
    x = np.array(r * np.cos(phi)).astype(np.float128)
    y = np.array(r * np.sin(phi)).astype(np.float128)

    return np.exp(-(x + 1)**2 - y**2) + np.exp(-(x - 1)**2 - (y - 1)**2)


def conforming_bkw(phi, r, beta=2):
    r = np.array(r).astype(np.float128)
    return 4. / beta * r * r * np.exp(-2 * r * r / beta)


class MaxwellBaseFunction(object):
    def __init__(self, decay_factor):
        """
        This base class represents functions
        :math:`f(x) = g(x) * exp(-decay_factor*x^2)`. When an object of this type is passed to an
        integration routine, the second factor :math:`exp(-decay_factor*x^2)` will be "moved" into the
        Gauss quadrature weights.

        Keyword Arguments:
        decay_factor --

        """
        self.decay_factor = decay_factor

    def w(self):
        return self.decay_factor


class FunctionWrapper(MaxwellBaseFunction):
    def __init__(self, basis, C, z0=0.0, decay_factor=0.5, scaling=1.0):
        """
        Keyword Arguments:
        self         --
        decay_factor --
        basis        --
        C            --
        z0           -- (default 0.0)
        scaling      -- (default 1.0)
        """
        self.basis = basis
        self.C = C
        self.z0 = z0
        if type(z0) is np.ndarray:
            raise TypeError()

        self.s = scaling
        super(FunctionWrapper, self).__init__(decay_factor)

    def __call__(self, z, weight=0.0):
        """
        Keyword Arguments:
        self   --
        z      --
        weight -- (default 0.0)
        """
        z = z.astype(np.complex128)
        return (
            self.s * self.basis.evaluatez(self.C, z - self.z0, weighted=False)
            * np.exp(-self.decay_factor * np.abs(z - self.z0)**2 +
                     weight * np.abs(z)**2))

    def quad_evaluator(self):
        return (lambda z: self.__call__(z, self.w()))


class Maxwellian(MaxwellBaseFunction):
    """
    A maxwellian function with momentum `u`, mass `m` and energy `e`

    This is:

    ..math:

    \frac{m e^{-|z-u|^2/e}}{\pi e}

    where :math:`e` denotes the energy and :math:`u` the momentum
    """

    def __init__(self, m, e, u):
        """
        Keyword Arguments:
        self --
        m    -- mass
        e    -- energy, note T=e/dim, e.g. e = T*dim
        u    -- momentum (as complex number)
        """
        dim = 2
        self.m = m
        self.e = e
        self.T = e / float(dim)
        self.u = u
        self.normalization = self.m / (2 * np.pi * self.T)
        super(Maxwellian, self).__init__(1.0 / e)

    def __call__(self, z, weight=0.0):
        """
        :weight: extra weight factor (required for quadrature rule)
        """
        z = z.astype(np.complex128)

        return self.normalization * np.exp(
            -np.abs(z - self.u)**2 / self.e + weight * abs(z)**2)

    def quad_evaluator(self):
        """
        Returns a suitable *weighted* lambda function for usage in quadrature rules
        """
        return (lambda z: self.__call__(z, self.w()))


class SkewGaussian(MaxwellBaseFunction):
    def __init__(self, D, phi0, z0):
        """
        Keyword Arguments:
        self --
        D    -- [d1, d2]
        phi0 -- rotation
        z0   --
        """

        w = np.mean(D)
        self.phi0 = phi0
        self.z0 = z0

        self.D = np.array([D[0], D[1]])
        assert (D[0] > 0 and D[1] > 0)

        super(SkewGaussian, self).__init__(w)

    def __call__(self, z, weight=0.0):

        Z = (z - self.z0) * np.exp(-1j * self.phi0).astype(np.complex128)
        X = np.real(Z)
        Y = np.imag(Z)
        return np.exp(
            -self.D[0] * X * X - self.D[1] * Y * Y + weight * abs(z)**2)

    def quad_evaluator(self):
        return (lambda z: self.__call__(z, self.w()))


class BKWInitialDistribution(MaxwellBaseFunction):
    """
    A beta-conforming initial distribution with analytically known solution
    for the Boltzmann equation. A solution is called beta conforming, if the
    equilibrium solution has the form C*exp(-(x-x0)**2/beta).
    This initial distribution will converge to 4*exp(-(x-x0)**2/beta).
    """

    def __init__(self, z0, beta):
        self.beta = beta
        self.z0 = z0
        super(BKWInitialDistribution, self).__init__(2.0 / beta)

    def __call__(self, z, weight=0.0):
        """
        :weight: extra weight factor (required for quadrature rule)
        """
        z = np.array(z).astype(np.complex128)
        return 4.0 / self.beta * np.exp(
            -2.0 * np.abs(z - self.z0)**2 / self.beta + weight * abs(z)**2
        ) * np.abs(z - self.z0)**2

    def quad_evaluator(self):
        """
        Returns a suitable *weighted* lambda function for usage in quadrature rules
        """
        return (lambda z: self.__call__(z, self.w()))


class DiscontinuousMaxwellian(MaxwellBaseFunction):
    def __init__(self, t1, t2):
        """
        Keyword Arguments:
        self --
        m    --
        e    --
        u    --
        """
        tmax = max(t1, t2)
        self.t1 = t1
        self.t2 = t2
        super(DiscontinuousMaxwellian, self).__init__(1.0 / (2 * tmax))

        self.m1 = Maxwellian(m=1, e=2 * t1, u=0)
        self.m2 = Maxwellian(m=1, e=2 * t2, u=0)

    def __call__(self, z, weight=0.0):
        z = z.astype(np.complex128)

        idx = np.real(z) > 0
        F = np.zeros(z.shape, dtype=np.float128)
        F[idx] = self.m1(z[idx], weight)
        idx = np.logical_not(idx)
        F[idx] = self.m2(z[idx], weight)

        return F

    def quad_evaluator(self):
        """
        Returns a suitable *weighted* lambda function for usage in quadrature rules
        """
        return (lambda z: self.__call__(z, self.w()))


class RadialSymmetricF(MaxwellBaseFunction):
    """
    Function of the form:
    f(v) = f_theta(arg(v)) * f_r(abs(v)) * exp(-w abs(v)**2)
    """

    def __init__(self, ftheta, fr, w=0.5, z0=0):
        self.z0 = z0
        self.ftheta = ftheta
        self.fr = fr
        super(RadialSymmetricF, self).__init__(w)

    def __call__(self, z, weight=0.0):
        """
        Keyword Arguments:
        self   --
        z      --
        weight -- (default 0.0)
        """
        r = np.abs(z - self.z0)
        theta = np.angle(z - self.z0)

        return self.ftheta(theta) * self.fr(r) * np.exp(
            -self.w() * r**2 + abs(z)**2 * weight)

    def quad_evaluator(self):
        """
        Returns a suitable *weighted* lambda function for usage in quadrature rules
        """
        return (lambda z: self.__call__(z, self.w()))


def exp_grid(phi, r, nx, ny):
    x = np.array(r * np.cos(phi)).astype(np.float128)
    y = np.array(r * np.sin(phi)).astype(np.float128)

    xp = np.linspace(-2, 2, nx)
    yp = np.linspace(-2, 2, ny)

    out = np.zeros_like(r)

    for i in range(nx):
        for j in range(ny):
            out += np.exp(-4 * (x - xp[i])**2 - 4 * (y - yp[j])**2)

    return out

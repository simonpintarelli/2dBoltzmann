import numpy as np

from pyboltz.initialize import MaxwellBaseFunction


class BKWSolution(MaxwellBaseFunction):
    """

    .. math::
    h(t, v) = \alpha f( \eta t , \gamma v + u)

    """

    def __init__(self, beta, t, z0=0):
        """
        Keyword Arguments:
        self --
        beta --
        t    --
        z0   -- (default 0) -> bkw solution centered at z0

        For equilibrium solution set t=np.inf
        """

        self.gamma = np.sqrt(2.0 / beta)
        self.alpha = 2 * np.pi
        self.eta = np.pi * beta
        self.z0 = z0

        zeta = 1.0 / 8.0
        t0 = np.log(2) / zeta
        if t == np.inf:
            ss = 1
        else:
            ss = 1 - np.exp(-zeta * (self.eta * t + t0))
        decay_factor = 1.0 / 2.0 / ss

        self.bkw_factor = lambda r: self.alpha / (2 * np.pi * ss) * (1.0 - (1 - ss) / (2.0 * ss) * (2 - r**2 / ss))
        self.bkw_weight = lambda r, rc, w: np.exp(-rc**2 / 2.0 / ss + (w * r**2 / self.gamma**2))

        super(BKWSolution, self).__init__(decay_factor * self.gamma**2)

    def __call__(self, z, weight=0.0):
        r = np.abs(z) * self.gamma
        rc = np.abs(z - self.z0) * self.gamma
        return self.bkw_weight(r, rc, weight) * self.bkw_factor(rc)

    def quad_evaluator(self):
        return (lambda z: self.__call__(z, self.w()))


class BKWPareschiRusso(MaxwellBaseFunction):
    """

    """

    def __init__(self, t, sigma=np.pi / 6):
        """
        For equilibrium solution set t=np.inf
        """
        self.t = t
        self.sigma = sigma
        if t==np.inf:
            self.S = 1
        else:
            self.S = 1 - 0.5 * np.exp(sigma**2 * t / 8)
        decay_factor = 1 / (2 * self.S * sigma**2)

        super(BKWPareschiRusso, self).__init__(decay_factor)

    def __call__(self, z, weight=0.0):
        r = np.abs(z)
        r2 = r**2
        S = self.S
        sigma2 = self.sigma**2
        return 1 / (2 * np.pi * S**2 * sigma2) * (
            2 * S - 1 + (1 - S) /
            (2 * S) * r2 / sigma2) * np.exp(-r2 / (2 * S * sigma2))

    def quad_evaluator(self):
        """
        Keyword Arguments:
        self --
        """

        return (lambda z: self.__call__(z, self.w()))

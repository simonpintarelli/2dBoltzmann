from __future__ import absolute_import

from .libCTensor import CTensor, CTensorDEigen, CTensorDBLAS
from pyboltz.ode import euler, rk4


def collision_time_step(tensor, y0, dt, scheme=euler, proj=True):
    """
    Keyword Arguments:
    tensor -- collision tensor, e.g. one of CTensor, CTensorDEigen, CTensorDBLAS
    y0   -- initial value
    dt   --  or dt/epsilon, epsilon: Knudsen number
    scheme --
    proj -- (default True)
    """

    y1 = scheme(lambda y: tensor.apply(y), y0, dt)

    if proj:
        return tensor.project(y1, y0)
    else:
        return y1

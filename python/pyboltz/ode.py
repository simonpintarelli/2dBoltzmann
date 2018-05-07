def rk4(f, y0, dt):
    """
    Keyword Arguments:
    f  --
    y0 --
    dt --
    """
    k1 = f(y0)
    k2 = f(y0 + 0.5 * dt * k1)
    k3 = f(y0 + 0.5 * dt * k2)
    k4 = f(y0 + dt * k3)

    y1 = y0 + dt / 6 * (k1 + 2 * (k2 + k3) + k4)

    return y1


def euler(f, y0, dt):
    """
    Keyword Arguments:
    f  -- 
    y0 -- 
    dt -- 
    """

    return y0 + dt*f(y0)
 

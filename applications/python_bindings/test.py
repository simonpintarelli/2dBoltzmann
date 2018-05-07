from bquad import hermw
from pyboltz.hermite_basis import hermval_weighted
from numpy import *
from matplotlib.pyplot import *


xq, wq = hermw(0.5, 4)



print sum(hermval_weighted(xq, array([1, 0, 0]))*wq)

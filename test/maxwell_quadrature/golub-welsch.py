#!/bin/env python2

from numpy import *
from matplotlib.pylab import *
from scipy.sparse import diags
from scipy.linalg import eig
from scipy.special import gamma
import subprocess
from numpy.linalg import norm

import sys

data = loadtxt(sys.stdin)

bi = data[1:,1]
ai = data[0:,2]
n = len(ai)

#A = diags((sqrt(bi[:-1]), ai, sqrt(bi[:-1])), (-1, 0, 1), shape=(n,n))
A = diags((sqrt(bi), ai, sqrt(bi)), (-1, 0, 1), shape=(n,n))

# moments
def m(n,p):
    return 0.5*gamma((n+p+1)*0.5)


print A.todense()

print 'eigenvalues'
l, v = eig(A.todense())

print l


# weights
print 'weights'
w = zeros(n)
for i in range(n):
    w[i] = v[0,i]**2*m(0,1)/norm(v[:,i])
print w

# integrate x^4 * w(x), {x,0, inf}
# w(x) = x* exp(-x**2)

f = lambda x: x**10

print 'integral = ', sum(f(l)*w)

scatter( real(l), imag(l))
savefig('x.png')

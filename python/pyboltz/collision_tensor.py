from __future__ import absolute_import

from pyboltz.basis import get_basis
from pyboltz.observables import Mass, Momentum, Energy
from .utility import from_sparse_repr
import h5py
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import splu
import os
import numpy as np


class L2Project:
    def __init__(self, basis):
        self.basis = basis
        mass = Mass(self.basis)
        energy = Energy(self.basis)
        momentum = Momentum(self.basis)

        self.H = np.zeros(shape=(len(self.basis), 4))
        self.H[:, 0] = mass.weights
        self.H[:, 1] = momentum.weightsX
        self.H[:, 2] = momentum.weightsY
        self.H[:, 3] = energy.weights
        HtH = np.dot(self.H.T, self.H)
        self.HtHinv = np.linalg.inv(HtH)

    def project(self, Ctilde, Cprev):
        """
        L2-projection of `Ctilde` to mass, energy and momentum of `Cprev`.

        Keyword Arguments:
        Ctilde -- coefficient vector at time k (unprojected)
        Cprev  -- coefficient vector at time k-1
        """

        ll = np.dot(self.HtHinv, np.dot(self.H.T, (Ctilde - Cprev)))
        return Ctilde - np.dot(self.H, ll)


class CollisionTensor:
    def __init__(self, fname='collision_tensor.h5'):
        dirname = os.path.dirname(fname)
        trial_desc_file = os.path.join(dirname, 'spectral_basis.desc')
        test_desc_file = os.path.join(dirname, 'spectral_basis_test.desc')

        if os.path.isfile(trial_desc_file):
            self.basis = get_basis(trial_desc_file)
        else:
            raise Exception(
                'trial descriptor not found\ndirname was: %s, fname: %s' %
                (dirname, fname))

        if os.path.isfile(test_desc_file):
            self.test_basis = get_basis(test_desc_file)
        else:
            raise Exception('test descriptor not found')

        f = h5py.File(fname)

        n = len(self.basis.elements)
        self.slices = [
            csr_matrix(from_sparse_repr(f[str(i)], (n, n))) for i in range(n)
        ]
        self.Minv = 1 / np.diag(self.basis.make_mass_matrix())

        self.project = L2Project(self.basis)

    def apply(self, x):
        y = np.zeros_like(x)
        for i, si in enumerate(self.slices):
            y[i] = np.dot(x, si * x)
        return self.Minv * y

    def __call__(self, x):
        return self.apply(x)

    def time_step(self, scheme, y0, dt, proj=True):
        """
        Keyword Arguments:
        scheme -- integration routine (rk4, Euler)
        y0     -- coefficient vector
        dt     -- timestep
        """
        y1 = scheme(self, y0, dt)
        if proj:
            return self.project.project(y1, y0)
        else:
            return y1

    def nnz(self):
        tot = 0
        for xi in self.slices:
            tot += xi.nnz

    def to_lil_matrix(self):
        for i, sli in enumerate(self.slices):
            self.slices[i] = lil_matrix(sli)

from numpy import *
import h5py
from matplotlib.pyplot import *
from pyboltz.initialize import SkewGaussian, project_to_polar_basis
from pyboltz.ks_basis import KSBasis
from pyboltz.quadrature import TensorQuadFactory



if __name__ == '__main__':

    basis = KSBasis(K=30, w=0.5, fname=None)
    basis.write_desc()


    m = SkewGaussian([1,0.5], pi/6, 2)


    quad_handler = TensorQuadFactory()

    C = project_to_polar_basis(m, basis, quad_handler, nptsa=120, nptsr=90)

    fh5 =  h5py.File('init.h5')
    fh5.create_dataset('coeffs', shape=C.shape, dtype=C.dtype, data=C)
    fh5.close()


    x = np.linspace(-1, 6, 200)
    y = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, x)
    Z = X +1j*Y
    Fref = m(Z)

    R, PHI = np.sqrt(X*X + Y*Y), np.arctan2(Y, X)
    F = basis.evaluate(C, PHI, R)

    figure()
    title('f(v)')
    pcolormesh(X, Y, F)
    savefig('fv.png')

    figure()
    title('log10 Errors')
    imshow(np.log10(np.abs(F-Fref)))
    colorbar()
    savefig('fv_err.png')
    show()

    # # test beta conforming BKW initial distribution
    # z0 = 1+1j
    # bkw = BKWInitialDistribution(z0, 1)

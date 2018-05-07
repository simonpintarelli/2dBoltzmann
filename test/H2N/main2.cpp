
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <iostream>

#include <Eigen/Dense>
#include <algorithm>

#include "aux/eigen2hdf.hpp"
#include "quadrature/qhermite.hpp"
#include "spectral/basis/spectral_basis_factory_hermite.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"
#include "spectral/basis/toolbox/spectral_basis.hpp"
#include "spectral/h2n_1d.hpp"
#include "spectral/hermite_to_nodal.hpp"
#include "spectral/polar_to_hermite.hpp"

//#include <deal.II/base/exceptions.h>

using namespace std;
using namespace boltzmann;

namespace po = boost::program_options;

int main(int argc, char* argv[])
{
  typedef Eigen::VectorXd vec_t;

  int K = atoi(argv[1]);

  // create basis files
  typedef typename SpectralBasisFactoryHN::basis_type hermite_basis_t;
  hermite_basis_t hermite_basis;
  SpectralBasisFactoryHN::create(hermite_basis, K, 2);
  cout << "Hermite basis size: " << hermite_basis.n_dofs() << endl;

  // hermite to nodal
  typedef Eigen::MatrixXd mat_t;
  Hermite2Nodal<hermite_basis_t> h2n(
      hermite_basis, K, [K](mat_t& m1, mat_t& m2) { H2N_1d<>::create(m1, m2, K); });
  Hermite2Nodal<hermite_basis_t> h2ng(
      hermite_basis, K, [K](mat_t& m1, mat_t& m2) { H2NG_1d::create(m1, m2, K, 1.0); });

  auto& M = h2n.get_h2n();

  auto& M2 = h2ng.get_h2n();

  auto diff = M - M2;
  diff = diff.select(abs(diff) < 1e-15, Eigen::MatrixXd::Zero(diff.rows(), diff.cols()), diff);

  cout << "difference matrix\n";
  cout << diff << endl;
  // cout << "M := H2N-matrix\n";
  // cout << "M.T * M\n";
  // cout << M.transpose()* M  << endl;

  // fname = "coefficients.h5";
  // if(!boost::filesystem::exists(fname)) {
  //   cout << endl << fname << " does not exist. Abort!\n";
  //   return 1;
  // }
  // vec_t cp(N); // polar coefficients
  // hid_t h5_init = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  // eigen2hdf::load(h5_init, "coeffs", cp);
  // H5Fclose(h5_init);

  // AssertDimension(polar_basis.n_dofs(), cp.size());

  hid_t h5f = H5Fcreate("test.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  eigen2hdf::save(h5f, "M", M);
  eigen2hdf::save(h5f, "Mg", M2);
  // eigen2hdf::save(h5f, "coeffs_nodal", cn);
  // eigen2hdf::save(h5f, "coeffs_hermite", ch);
  // eigen2hdf::save(h5f, "coeffs", cp2);
  H5Fclose(h5f);

  return 0;
}

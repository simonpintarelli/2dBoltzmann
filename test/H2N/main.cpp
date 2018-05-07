
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <cassert>
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

  typedef typename SpectralBasisFactoryKS::basis_type polar_basis_t;
  polar_basis_t polar_basis;

  string fname = "spectral_basis.desc";
  cout << "reading `" << fname << "` from disk.\n";
  SpectralBasisFactoryKS::create(polar_basis, fname);

  int K = spectral::get_max_k(polar_basis) + 1;
  int N = polar_basis.n_dofs();

  // create basis files
  typedef typename SpectralBasisFactoryHN::basis_type hermite_basis_t;
  hermite_basis_t hermite_basis;
  SpectralBasisFactoryHN::create(hermite_basis, K, 2);
  cout << "Hermite basis size: " << hermite_basis.n_dofs() << endl;
  cout << "Polar basis size: " << polar_basis.n_dofs() << endl;

  // polar 2 hermite
  Polar2Hermite<polar_basis_t, hermite_basis_t> P2H(polar_basis, hermite_basis);

  typedef Eigen::MatrixXd mat_t;
  // hermite to nodal
  Hermite2Nodal<hermite_basis_t> h2n(
      hermite_basis, K, [K](mat_t& m1, mat_t& m2) { H2N_1d<>::create(m1, m2, K); });

  // debug
  auto& M = h2n.get_h2n();
  cout << "M := H2N-matrix\n";
  cout << "M.T * M\n";
  Eigen::MatrixXd O = M.transpose() * M;

  double MtM_diff = (O - Eigen::MatrixXd::Identity(O.rows(), O.cols())).cwiseAbs().sum();
  if (MtM_diff > 1e-13 * O.rows() * O.cols()) {
    cout << "diff: " << MtM_diff << "\n";
    O = (O.array().cwiseAbs() > 1e-15).select(O, Eigen::ArrayXXd::Zero(O.rows(), O.cols()));
    cout << O << endl;
  }

  cout << "---------- Test: P->H->N->H->P  ----------"
       << "\n";
  fname = "coefficients.h5";
  vec_t cp(N);  // polar coefficients
  if (!boost::filesystem::exists(fname)) {
    cp.setOnes();
  } else {
    hid_t h5_init = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    eigen2hdf::load(h5_init, "coeffs", cp);
    H5Fclose(h5_init);
  }

  assert(polar_basis.n_dofs() == cp.size());

  // Transform Polar to Hermite basis
  vec_t ch = cp;
  P2H.to_hermite(ch, cp);

  // Transform Hermite to Nodal basis
  Eigen::MatrixXd cn;
  h2n.to_nodal(cn, ch);
  // Transform Nodal to Hermite basis
  h2n.to_hermite(ch.data(), cn);
  // Tranform Hermite to Polar basis
  vec_t cp2(N);
  P2H.to_polar(cp2, ch);

  /* hid_t h5f = H5Fcreate("out.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT); */
  /* eigen2hdf::save(h5f, "coeffs_nodal", cn); */
  /* eigen2hdf::save(h5f, "coeffs_hermite", ch); */
  /* eigen2hdf::save(h5f, "coeffs", cp2); */
  /* H5Fclose(h5f); */

  vec_t diff = (cp2 - cp).rowwise().squaredNorm();
  if (diff.sum() > 1e-10) {
    cout << "diff\n";
    cout << diff << endl;
  } else {
    cout << "sum(diff) " << diff.sum() << endl;
  }

  // int i,j;
  // cout << "max diff: " << diff.maxCoeff(&i, &j) << " at " << i << ", " << j << endl;

  return 0;
}

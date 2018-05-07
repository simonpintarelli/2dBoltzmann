/**
 * @file   main.cpp
 * @author Simon Pintarelli <simon@thinkpadX1>
 * @date   Wed Oct 21 17:37:10 2015
 *
 * @brief  Example for Polar->Nodal basis transformation and quadrature
 *         in the nodal basis
 *
 *
 */

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <iostream>

#include <Eigen/Dense>
#include <algorithm>

#include "quadrature/qhermite.hpp"

//#include "spectral/hermite_to_nodal.hpp"
#include "aux/eigen2hdf.hpp"
#include "post_processing/mass.hpp"
#include "spectral/basis/spectral_basis_factory_hermite.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"
#include "spectral/basis/toolbox/spectral_basis.hpp"
#include "spectral/polar_to_hermite.hpp"
#include "spectral/polar_to_nodal.hpp"


using namespace std;
using namespace boltzmann;

namespace po = boost::program_options;

// obviously wrong ... But it is not used, see below.
void test1(int K)
{
  typedef Eigen::VectorXd vec_t;

  typedef typename SpectralBasisFactoryKS::basis_type polar_basis_t;
  polar_basis_t polar_basis;
  SpectralBasisFactoryKS::create(polar_basis, K, K, 2, true);
  SpectralBasisFactoryKS::write_basis_descriptor(polar_basis, "spectral_basis.desc");

  Polar2Nodal<polar_basis_t> p2n;
  p2n.init(polar_basis, 0.5);

  Mass mass(polar_basis);

  std::vector<size_t> elems = {0, 1, 2, 3, 4, 5, 6};
  // quadrature
  QHermiteW quad(0.5, K);
  auto& w = quad.wts();
  auto& x = quad.pts();

  // Achtung mit den Knoten und der Skalierung der Gewichte
  QHermiteW quad1(1, K);
  auto& xh = quad1.pts();

  std::string fname = std::string("P2N") + std::to_string(K) + ".h5";
  hid_t file = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  for (size_t eidx : elems) {
    Eigen::VectorXd cp(polar_basis.n_dofs());
    cp.setZero();
    cp(eidx) = 1.0;

    const double mass_ref = mass.compute(cp.data());

    Eigen::MatrixXd cn(K, K);
    p2n.to_nodal(cn, cp);
    eigen2hdf::save(file, "cn" + std::to_string(eidx), cn);

    auto id = polar_basis.get_elem(eidx).id();

    double sum = 0;
    for (size_t q1 = 0; q1 < w.size(); ++q1) {
      for (size_t q2 = 0; q2 < w.size(); ++q2) {
        // (*) QUAD EXAMPLE
        sum += cn(q1, q2) * std::sqrt(w[q1] * w[q2]) *
               std::exp(-xh[q1] * xh[q1] / 2 - xh[q2] * xh[q2] / 2);
      }
    }
    cout << id.to_string() << "Sum: " << setprecision(5) << scientific << sum << endl;
    cout << id.to_string() << "Ref: " << setprecision(5) << scientific << mass_ref << "\n\n";
  }

  typedef Eigen::VectorXd vec_t;
  Eigen::Map<const vec_t> xq(quad.points_data(), K);
  Eigen::Map<const vec_t> wq(quad.weights_data(), K);

  eigen2hdf::save(file, "xq", xq);
  eigen2hdf::save(file, "wq", wq);

  auto& N2H = p2n.get_h2n()->get_n2h();
  auto& H2N = p2n.get_h2n()->get_h2n();

  eigen2hdf::save(file, "H2N", H2N);
  eigen2hdf::save(file, "N2H", N2H);

  H5Fclose(file);
}

// Polar-Laguerre coefficients with exponential decay
void test2(int K, const std::function<double(double)>& cfct)
{
  typedef Eigen::VectorXd vec_t;

  typedef typename SpectralBasisFactoryKS::basis_type polar_basis_t;
  polar_basis_t polar_basis;
  SpectralBasisFactoryKS::create(polar_basis, K, K, 2, true);
  SpectralBasisFactoryKS::write_basis_descriptor(polar_basis, "spectral_basis.desc");

  Polar2Nodal<polar_basis_t> p2n;
  p2n.init(polar_basis, 1.0);

  unsigned int N = polar_basis.n_dofs();
  Eigen::VectorXd cp(N);
  for (unsigned int i = 0; i < N; ++i) {
    cp(i) = cfct(float(i) / N);
  }
  Eigen::MatrixXd cn(K, K);
  p2n.to_nodal(cn, cp);
  Eigen::VectorXd cp2(N);
  p2n.to_polar(cp2, cn);

  auto diff = cp2;
  diff.setZero();
  for (unsigned int i = 0; i < N; ++i) {
    diff(i) = std::abs(cp2(i) - cp(i));
  }
  cout << " sum(err): " << diff.sum() << endl;
}

int main(int argc, char* argv[])
{
  if (argc < 2) {
    cout << "usage: " << argv[0] << " K\n";
    exit(1);
  }
  const int K = atoi(argv[1]);

  cout << "Test: P->N->P "
       << "\n";
  cout << "Exponential decaying Polar-Laguerre coefficients: cp[i] = exp(-20*i/N)"
       << "\n";
  test2(K, [](double i) { return std::exp(-20 * i); });

  cout << "Constant coefficients: cp[i] = 1.0"
       << "\n";
  test2(K, [](double i) { return 1.0; });
  return 0;
}

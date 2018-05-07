#include <gtest/gtest.h>
#include <boost/lexical_cast.hpp>
#include "quadrature/qhermite.hpp"
#include "spectral/polar_to_nodal.hpp"
#include "spectral/p2n_factory.hpp"

using namespace std;
using namespace boltzmann;

// Polar-Laguerre coefficients with exponential decay
void test2(int K, const std::function<double(double)>& cfct, double tol)
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
    // cp(i) = cfct(float(i)/N);
    cp(i) = 1;
  }
  Eigen::MatrixXd cn(K, K);
  p2n.to_nodal(cn, cp);
  Eigen::VectorXd cp2(N);
  p2n.to_polar(cp2, cn);

  Eigen::VectorXd diff(N);
  diff.setZero();

  for (unsigned int i = 0; i < N; ++i) {
    diff(i) = std::abs(cp2(i) - cp(i));
  }

  double max_diff = diff.cwiseAbs().maxCoeff();
  cout << "max_diff: " << max_diff << "\n";
  EXPECT_TRUE(max_diff < tol) << max_diff;
}

// Polar-Laguerre coefficients with exponential decay
void test3(int K, const std::function<double(double)>& cfct, double tol)
{
  typedef Eigen::VectorXd vec_t;

  typedef typename SpectralBasisFactoryKS::basis_type polar_basis_t;
  polar_basis_t polar_basis;
  SpectralBasisFactoryKS::create(polar_basis, K, K, 2, true);
  SpectralBasisFactoryKS::write_basis_descriptor(polar_basis, "spectral_basis.desc");

  const auto& p2n = P2NFactory<polar_basis_t>::GetInstance(polar_basis, 1.0);

  // Polar2Nodal<polar_basis_t> p2n;
  // p2n.init(polar_basis, 1.0);

  unsigned int N = polar_basis.n_dofs();
  Eigen::VectorXd cp(N);
  for (unsigned int i = 0; i < N; ++i) {
    // cp(i) = cfct(float(i)/N);
    cp(i) = 1;
  }
  Eigen::MatrixXd cn(K, K);
  p2n.to_nodal(cn, cp);
  Eigen::VectorXd cp2(N);
  p2n.to_polar(cp2, cn);

  Eigen::VectorXd diff(N);
  diff.setZero();

  for (unsigned int i = 0; i < N; ++i) {
    diff(i) = std::abs(cp2(i) - cp(i));
  }

  double max_diff = diff.cwiseAbs().maxCoeff();
  cout << "max_diff: " << max_diff << "\n";
  EXPECT_TRUE(max_diff < tol) << max_diff;
}

TEST(spectral, polar2nodal)
{
  const double tol = 1e-13;
  cout << "Tolerance: " << tol << "\n";
  std::vector<int> Ks = {6, 10, 20, 30, 40, 50, 70};
  for (auto K : Ks) {
    cout << "running test for K=" << boost::lexical_cast<string>(K) << "\n";
    // test2(K, [](double i){return std::exp(-1*i);}, tol);
    test2(K, [](double i) { return 1; }, tol);
  }
}

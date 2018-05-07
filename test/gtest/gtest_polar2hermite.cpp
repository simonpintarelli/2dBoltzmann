/**
   Transformation between Polar-Laguerre and Hermite basis
*/

// system includes
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <boost/lexical_cast.hpp>
#include <functional>
#include <iostream>

// own includes
#include "spectral/basis/spectral_basis_factory_hermite.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"
#include "spectral/polar_to_hermite.hpp"

using namespace std;
using namespace boltzmann;

TEST(spectral, polar2hermite)
{
  std::vector<int> Ks = {4, 10, 20, 25, 36, 50, 80};

  cout << "Testing for K=";
  for_each(Ks.begin(), Ks.end(), [](int x) { cout << x << " "; });
  cout << "\n";

  for (int K : Ks) {
    typedef typename SpectralBasisFactoryHN::basis_type hermite_basis_t;
    hermite_basis_t hermite_basis;
    SpectralBasisFactoryHN::create(hermite_basis, K, 2);

    typedef typename SpectralBasisFactoryKS::basis_type polar_basis_t;
    polar_basis_t polar_basis;
    SpectralBasisFactoryKS::create(polar_basis, K, K, 2, true /* sorted */);

    Polar2Hermite<polar_basis_t, hermite_basis_t> p2h(polar_basis, hermite_basis);

    Eigen::VectorXd cp(polar_basis.n_dofs());
    cp.setOnes();

    Eigen::VectorXd ch(hermite_basis.n_dofs());
    p2h.to_hermite(ch, cp);

    Eigen::VectorXd cp2(polar_basis.n_dofs());
    p2h.to_polar(cp2, ch);

    double err = (cp2 - cp).cwiseAbs2().maxCoeff();
    cout << "err: " << err << "\n";
    EXPECT_NEAR(err, 0, 1e-14) << "K=" << boost::lexical_cast<string>(K);
  }
}

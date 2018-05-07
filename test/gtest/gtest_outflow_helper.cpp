/**
   Transformation between Polar-Laguerre and Hermite basis
*/

// system includes
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <boost/lexical_cast.hpp>
#include <functional>
#include <iostream>
#include <vector>

// own includes
#include "matrix/bc/impl/outflow_helper.hpp"
#include "spectral/basis/spectral_basis_factory_hermite.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"
#include "spectral/nodal.hpp"
#include "spectral/polar_to_hermite.hpp"

using namespace std;


TEST(spectral, outflow)
{
  std::vector<int> Ks = {5, 10, 20, 30, 40, 42};
  for(int K : Ks) {
    Eigen::MatrixXd C(K, K);
    double T = 1;

    boltzmann::to_nodal(C, [T](double x, double y) { return std::exp(-(x * x + y * y) / 2 / T); });

    double csum = C.sum();

    boltzmann::QHermiteW quad(1.0, K);
    Eigen::VectorXd hx = quad.vpts<1>();
    Eigen::VectorXd hw = quad.vwts<1>();

    boltzmann::impl::outflow_helper outflow(hw, hx);

    // reference value: 2.50662827463100
    double rho_ref = 2.50662827463100;
    double rhop = outflow.compute(C);

    EXPECT_NEAR(rhop, rho_ref, 1e-10) << "outflow of maxwellian is wrong";
  }
}

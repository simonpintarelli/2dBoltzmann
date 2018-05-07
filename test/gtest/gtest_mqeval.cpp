#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <string>

#include "post_processing/macroscopic_quantities.hpp"
#include "post_processing/mass.hpp"
#include "post_processing/momentum.hpp"
#include "post_processing/energy.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"
#include "aux/eigen2hdf.hpp"

using namespace boltzmann;
using namespace std;



TEST(spectral, momentsfile)
{

  std::string filename = "mqeval.h5";
  MQEval mq_coeffs_;
  int K = 40;
  SpectralBasisFactoryKS::basis_type basis;
  SpectralBasisFactoryKS::create(basis, K);
  mq_coeffs_.init(basis);

  Eigen::VectorXd vec(basis.size());

  hid_t fh5 = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  eigen2hdf::load(fh5, "coeffs", vec);
  H5Fclose(fh5);

  auto evaluator = mq_coeffs_.evaluator();
  evaluator(vec);

  cout << setw(10) << "rho "
       << setw(20) << setprecision(8) << scientific << evaluator.m << "\n";

  cout << setw(10) << "e "
       << setw(20) << setprecision(8) << scientific << evaluator.e << "\n";

  cout << setw(10) << "v "
       << setw(20) << setprecision(8) << scientific << evaluator.v << "\n";

  cout << setw(10) << "v "
       << setw(20) << setprecision(8) << scientific << evaluator.v << "\n";
}

TEST(spectral, moments)
{

  std::string filename = "mqeval.h5";
  MQEval mq_coeffs;
  int K = 40;
  SpectralBasisFactoryKS::basis_type basis;
  SpectralBasisFactoryKS::create(basis, K);
  mq_coeffs.init(basis);

  auto cmass = mq_coeffs.cmass();
  auto cenergy = mq_coeffs.cenergy();

  Energy energy(basis);
  Mass mass(basis);

  Eigen::ArrayXd cenergy_ref(basis.size());
  cenergy_ref.setZero();
  Eigen::ArrayXd cmass_ref(basis.size());
  cmass_ref.setZero();

  for (auto entry : energy.entries()) {
    cenergy_ref(entry.first) = entry.second;
  }

  for (auto entry : mass.entries()) {
    cmass_ref(entry.first) = entry.second;
  }

  double diff_mass_max = (cmass_ref.segment(0, cmass.size()) - cmass).cwiseAbs().maxCoeff();
  double diff_energy_max = (cenergy_ref.segment(0, cenergy.size()) - cenergy).cwiseAbs().maxCoeff();

  EXPECT_NEAR(diff_mass_max, 0, 1e-11) << "max diff coeffs mass";
  EXPECT_NEAR(diff_energy_max, 0, 1e-11) << "max diff coeffs energy";
}

#include <Eigen/Sparse>
// system includes ------------------------------------------------------------
#include <algorithm>
#include <boost/program_options.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <cmath>
#include <random>

// own includes ---------------------------------------------------------------
#include "spectral/basis/spectral_basis_dimension_accessor.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"
#include "spectral/basis/spectral_elem_accessor.hpp"

#include "spectral/rotate_basis.hpp"

using namespace std;
using namespace boltzmann;
namespace po = boost::program_options;

const int dim = 2;

int main(int argc, char *argv[])
{
  int K;
  double beta;
  bool sorted;
  po::options_description options("options");
  options.add_options()
      ("help", "produce help message")
      ("K", po::value<int>(&K)->default_value(10))
      ("sorted", po::value<bool>(&sorted)->default_value(true))
      ("beta", po::value<double>(&beta)->default_value(2));
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, options), vm);
  po::notify(vm);

  if (vm.count("help")) {
    cout << options << "\n";
    return 1;
  }

  // ------------------------------------------------------------
  typedef SpectralBasisFactoryKS::basis_type basis_type;
  basis_type trial_basis;
  SpectralBasisFactoryKS::create(trial_basis, K, K, beta, sorted);
  // sort
  typedef typename basis_type::elem_t elem_t;

  // write to disk
  SpectralBasisFactoryKS::write_basis_descriptor(trial_basis, "spectral_basis.desc");
  basis_type test_basis;
  SpectralBasisFactoryKS::create_test(test_basis, K, K, beta, sorted);
  SpectralBasisFactoryKS::write_basis_descriptor(test_basis, "spectral_basis_test.desc");

  // for (int i = 0; i < K; ++i) {
  //   // loop over all elements in basis
  //   for (auto  it = trial_basis.begin(); it < trial_basis.end(); ++it) {

  //   }
  // }

  RotateBasis<basis_type> rotate_basis(trial_basis);

  rotate_basis.init();

  unsigned int L = 100;
  unsigned int N = trial_basis.n_dofs();

  Eigen::VectorXd x(L * N);
  Eigen::VectorXd y(L * N);
  Eigen::VectorXd y2(L * N);

  std::random_device rd;

  // Choose a random mean between 1 and 6
  // std::default_random_engine e1(rd());
  std::default_random_engine e1(1);
  std::uniform_real_distribution<double> uniform_dist(-10, 10);

  for (unsigned int i = 0; i < L * N; i++) {
    x[i] = uniform_dist(e1);
  }

  double dphi = 2;

  rotate_basis.apply(y.data(), x.data(), -dphi, L);
  rotate_basis.apply(y2.data(), y.data(), dphi, L);

  std::cout << "error: " << (y2 - x).norm() << std::endl;

  //  Eigen::VectorXd diff = (y2-x);

  // cout << diff
  //      << endl;

  return 0;
}

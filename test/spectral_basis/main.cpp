#include <Eigen/Sparse>
// system includes ------------------------------------------------------------
#include <algorithm>
#include <boost/program_options.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
// own includes ---------------------------------------------------------------
#include "spectral/basis/spectral_basis_dimension_accessor.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"
#include "spectral/basis/spectral_elem_accessor.hpp"

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
      ("K", po::value<int>(&K))
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

  return 0;
}

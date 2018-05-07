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
#include "spectral/basis/toolbox/spectral_basis.hpp"

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
      ("help", "produce help message")("K", po::value<int>(&K)->default_value(10))
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
  typedef typename basis_type::elem_t elem_t;

  // write to disk
  SpectralBasisFactoryKS::write_basis_descriptor(trial_basis, "spectral_basis.desc");
  basis_type test_basis;
  SpectralBasisFactoryKS::create_test(test_basis, K, K, beta, sorted);
  SpectralBasisFactoryKS::write_basis_descriptor(test_basis, "spectral_basis_test.desc");

#if __cplusplus >= 201402
  auto range = spectral::filter_freq(test_basis.begin(), test_basis.end(), 1);

  for (auto it = std::get<0>(range); it != std::get<1>(range); it++) {
    cout << it->id().to_string() << std::endl;
  }
#endif

  return 0;
}

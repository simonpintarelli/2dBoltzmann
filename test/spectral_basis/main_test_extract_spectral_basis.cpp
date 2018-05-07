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

int main(int argc, char* argv[])
{
  int K;
  double beta;
  bool sorted;
  po::options_description options("options");
  options.add_options()
      ("help", "produce help message")
      ("K", po::value<int>(&K)->default_value(10));
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, options), vm);
  po::notify(vm);

  if (vm.count("help")) {
    cout << options << "\n";
    return 1;
  }

  // ------------------------------------------------------------
  typedef SpectralBasisFactoryKS::basis_type basis_type;
  basis_type spectral_basis;
  typedef typename boost::mpl::at_c<typename basis_type::elem_t::types_t, 0>::type angular_t;
  typedef typename boost::mpl::at_c<typename basis_type::elem_t::types_t, 1>::type radial_t;
  typedef typename basis_type::DimAcc::template get_vec<radial_t> get_radial_basis_t;
  SpectralBasisFactoryKS::create(spectral_basis, K);
  // sort
  typedef typename basis_type::elem_t elem_t;
  typedef typename elem_t::Acc::template get<angular_t> get_xi;
  typedef typename elem_t::Acc::template get<radial_t> get_laguerre;

  /* for (auto elem : spectral_basis) { */
  /*   cout << elem.id().to_string() << "\n"; */
  /* } */

  get_radial_basis_t get_rad_basis;
  auto wtf = get_rad_basis(spectral_basis);

  cout << "---------- l=1 elements ----------"
       << "\n";
  auto range = spectral::filter_freq(spectral_basis.begin(), spectral_basis.end(), 1);
  for (auto it = std::get<0>(range); it != std::get<1>(range); ++it) {
    cout << it->id().to_string() << "\n";
  }

  // using boost iterator range
  for (auto&& elem : boost::make_iterator_range(std::get<0>(range), std::get<1>(range))) {
    cout << elem.id().to_string() << "\n";
  }

  return 0;
}

#include "quadrature/qhermite.hpp"
#include "spectral/basis/spectral_basis.hpp"
#include "spectral/basis/spectral_elem.hpp"
#include "spectral/basis/spectral_elem_accessor.hpp"
#include "spectral/basis/spectral_function/hermite_polynomial.hpp"

#include <iomanip>
#include <iostream>
#include <tuple>

#include <Eigen/Sparse>

using namespace boltzmann;

/*

class HermiteHX : public HermiteH
{
public:
  HermiteHX(int k) : HermiteH(k, 0.5) {}
  HermiteHX() {}
};

class HermiteHY : public HermiteH
{
public:
  HermiteHY(int k) : HermiteH(k, 0.5) {}
  HermiteHY() {}
};




int main(int argc, char *argv[])
{
  // typedef HermiteH HermiteHX;
  // typedef HermiteH HermiteHY;

  typedef SpectralElem<double, HermiteHX, HermiteHY>
    spectral_elem_t;

  typedef SpectralBasis<spectral_elem_t> basis_type;
  basis_type basis;

  int K = 10;

  for (int k = 0; k < K; ++k) {
    for (int s = 0; s <= k; ++s) {
      HermiteHX hx(s);
      HermiteHY hy(k-s);
      basis.add_elem(hx, hy);
    }
  }

  //
  std::cout << "completed: adding elements to basis \n";

  SpectralElemAccessor::get<HermiteHX> getX;
  SpectralElemAccessor::get<HermiteHY> getY;

  for (auto elem_it = basis.begin(); elem_it < basis.end(); ++elem_it) {
    std::cout << getX(*elem_it).get_id() << "\t"
              << getY(*elem_it).get_id() << std::endl;
  }

  return 0;
}

*/

#include "quadrature/qhermite.hpp"
#include "spectral/basis/basis_types.hpp"
#include "spectral/basis/spectral_basis.hpp"
#include "spectral/basis/spectral_basis_factory_hermite.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"
#include "spectral/basis/spectral_elem.hpp"
#include "spectral/basis/spectral_elem_accessor.hpp"
#include "spectral/basis/spectral_function/hermite_polynomial.hpp"
#include "spectral/eval_handlers.hpp"

#include <Eigen/Sparse>
#include <boost/program_options.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "spectral/polar_to_hermite.hpp"

#define PI 3.141592653589793238462643383279502884197

using namespace std;
using namespace boltzmann;

namespace po = boost::program_options;

int main(int argc, char *argv[])
{
  // this cannot work
  /* typedef HermiteH HermiteHX; */
  /* typedef HermiteH HermiteHY; */

  int K;
  po::options_description options("options");
  options.add_options()
      ("nK,K", po::value<int>(&K)->default_value(10), "K");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, options), vm);
  po::notify(vm);

  typedef typename SpectralBasisFactoryKS::basis_type polar_basis_t;
  polar_basis_t polar_basis;
  SpectralBasisFactoryKS::create(polar_basis, K, K, 2, true);
  SpectralBasisFactoryKS::write_basis_descriptor(polar_basis, "spectral_basis.desc");

  typedef typename SpectralBasisFactoryHN::basis_type hermite_basis_t;
  hermite_basis_t hermite_basis;
  SpectralBasisFactoryHN::create(hermite_basis, K, 2);
  SpectralBasisFactoryHN::write_basis_descriptor(hermite_basis, "hermite_basis.desc");

  typedef typename SpectralBasisFactoryHN::fa_type HX;
  typedef typename SpectralBasisFactoryHN::fa_type HY;

  // make evaluator handle
  std::vector<double> x{0.1, 0.5, 1, 10, 100};
  auto Heval2d = hermite_evaluator2d<>::make(hermite_basis, x);

  std::cout << "completed: adding elements to basis \n";

  SpectralElemAccessor::get<HermiteHX_t> getX;
  SpectralElemAccessor::get<HermiteHY_t> getY;

  for (auto elem_it = hermite_basis.begin(); elem_it < hermite_basis.end(); ++elem_it) {
    std::cout << getX(*elem_it).get_id() << "\t" << getY(*elem_it).get_id() << std::endl;
  }

  return 0;
}

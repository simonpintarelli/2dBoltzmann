#include "aux/eigen2hdf.hpp"
#include "aux/message.hpp"
#include "aux/timer.hpp"
#include "post_processing/mass.hpp"
#include "post_processing/momentum.hpp"
#include "quadrature/qhermite.hpp"
#include "spectral/basis/spectral_basis.hpp"
#include "spectral/basis/spectral_basis_factory_hermite.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"
#include "spectral/basis/spectral_elem.hpp"
#include "spectral/basis/spectral_elem_accessor.hpp"
#include "spectral/basis/spectral_function/hermite_polynomial.hpp"
#include "spectral/basis/toolbox/spectral_basis.hpp"

#include "spectral/polar_to_hermite.hpp"
#include "spectral/shift_hermite_2d.hpp"

#include <Eigen/Sparse>
#include <boost/program_options.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>

#define PI 3.141592653589793238462643383279502884197

using namespace std;
using namespace boltzmann;

namespace po = boost::program_options;

#ifdef EXTENDED_PRECISION
typedef long double numeric_t;
#else
typedef double numeric_t;
#endif
const int nrep = 1000;

int main(int argc, char *argv[])
{
  Timer<> timer;
  po::options_description options("options");
  options.add_options()
      ("help", "show help message");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, options), vm);
  po::notify(vm);

  if (vm.count("help")) {
    cout << options << "\n";
    return 0;
  }

  // read polar basis from file
  typedef typename SpectralBasisFactoryKS::basis_type polar_basis_t;
  polar_basis_t polar_basis;
  SpectralBasisFactoryKS::create(polar_basis, "spectral_basis.desc");
  //  SpectralBasisFactoryKS::write_basis_descriptor(polar_basis, "spectral_basis.desc");

  int max_deg = spectral::get_max_k(polar_basis);
  const unsigned int K = max_deg + 1;
  // create corresponding Hermite basis
  typedef typename SpectralBasisFactoryHN::basis_type hermite_basis_t;
  hermite_basis_t hermite_basis;
  SpectralBasisFactoryHN::create(hermite_basis, max_deg + 1, 2);
  SpectralBasisFactoryHN::write_basis_descriptor(hermite_basis, "hermite_basis.desc");

  if (hermite_basis.n_dofs() != polar_basis.n_dofs()) {
    throw runtime_error("Hermite basis does not match!");
    return 1;
  }

  cout << "size(polar basis) = " << polar_basis.n_dofs() << endl
       << "size(hermite basis) = " << hermite_basis.n_dofs();

  cout << "\n--------------------\n";
  cout << "Test 2: (P->H) -> (H->P) show coefficients\n";
  timer.start();
  Polar2Hermite<polar_basis_t, hermite_basis_t> P2H(polar_basis, hermite_basis);
  print_timer(timer.stop(), "init P2H");

  /*
   * load coefficients (polar basis) from HDF5
   */
  const unsigned int N = polar_basis.n_dofs();
  Eigen::VectorXd coeffs(N);
  hid_t h5_init = H5Fopen("init.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
  eigen2hdf::load(h5_init, "coeffs", coeffs);
  H5Fclose(h5_init);

  // compute bulk velocity
  Mass mass;
  mass.init(polar_basis);
  Momentum momentum;
  momentum.init(polar_basis);

  const double m = mass.compute(coeffs.data());
  auto u = momentum.compute(coeffs.data()) / m;
  cout << scientific << setprecision(8) << "mass: " << m << endl
       << "momentum: " << u(0) << ", " << u(1) << endl;

  // compute hermite coefficients
  Eigen::VectorXd buf(N);

  // --------------------------------------------------
  // Transform to Hermite basis
  // --------------------------------------------------
  {
    timer.start();
    int nrep = 100000;
    for (int i = 0; i < nrep; ++i) {
      P2H.to_hermite(buf, coeffs);
    }
    double t = timer.stop();

    print_timer(t / nrep, "P2H.to_hermite");
  }

  if (sizeof(numeric_t) == 16) {
    cout << "Using *extended precision*  in ShiftHermite\n";
  } else if (sizeof(numeric_t) == 8) {
    cout << "Using double precision in ShiftHermite\n";
  }

  // use (extended/double) precision for shifting ...
  std::vector<numeric_t> cH(buf.data(), buf.data() + N);
  ShiftHermite2D<hermite_basis_t, numeric_t> shift_hermite(hermite_basis);
  shift_hermite.init();

  // --------------------------------------------------
  // Shift hermite coefficients
  // --------------------------------------------------
  timer.start();
  for (int i = 0; i < nrep; ++i) {
    shift_hermite.shift(cH.data(), u(0), u(1));
  }
  double t_shift_hermite = timer.stop();
  print_timer(t_shift_hermite / nrep, "shift Hermite coefficients");
  cout << "t_shift_hermite: " << scientific << setprecision(10) << t_shift_hermite << endl;

  // // transform coefficients back to double
  // std::transform(cH.begin(), cH.end(), buf.begin(), [](numeric_t x) { return double(x); });

  // // -> Polar coordinates
  // std::vector<double> Cc(N, 0.0);
  // P2H.to_polar(Cc, buf);

  // const double mc = mass.compute(Cc.data());
  // auto uc = momentum.compute(Cc.data())/mc;
  // cout << "centered:\n";
  // cout << "mass: " << scientific  << setprecision(8) << mc << "\t(diff = " << std::abs(m-mc) <<
  // ")"
  //      << endl
  //      << "momentum: " << uc(0) << ", " << uc(1) << endl;

  // // write new coefficients to disk
  // hid_t h5_shifted = H5Fcreate("shifted.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  // Eigen::Map< Eigen::VectorXd> Cc_eigen(Cc.data(), Cc.size());
  // eigen2hdf::save(h5_shifted, "coeffs", Cc_eigen);
  // // export hermite coefficients
  // Eigen::Map< Eigen::VectorXd> cH_eigen(buf.data(), buf.size());
  // eigen2hdf::save(h5_shifted, "coeffs_hermite", cH_eigen);
  // H5Fclose(h5_shifted);

  // // do some cheap scattering
  // // ...

  // // Move back to original position
  // timer.start();
  // shift_hermite.shift(cH.data(), -u(0), -u(1));
  // print_timer(timer.stop(), "shift Hermite coefficients (back)");

  // // go back to polar coordinates
  // std::transform(cH.begin(), cH.end(), buf.begin(), [](numeric_t x) { return double(x); });
  // std::vector<double> Cc2(N, 0.0);
  // P2H.to_polar(Cc2, buf);

  // const double m1 = mass.compute(Cc2.data());
  // auto u1 = momentum.compute(Cc2.data())/m1;

  // cout << "move back:\n";
  // cout << "mass: " << scientific  << setprecision(8) << m1 << "\t(diff = " << std::abs(m-m1) <<
  // ")"
  //      << endl
  //      << "momentum: " << scientific  << setprecision(8) << u1(0) << ", " << u1(1) << "\t(diff =
  //      " << (u-u1).squaredNorm() << ")" << endl;
  return 0;}

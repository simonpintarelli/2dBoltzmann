#include "aux/eigen2hdf.hpp"
#include "aux/message.hpp"
#include "aux/timer.hpp"
#include "post_processing/macroscopic_quantities.hpp"
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
#include "spectral/utility/mass_matrix.hpp"

#include "spectral/polar_to_hermite.hpp"
#include "spectral/shift_hermite_2d.hpp"

#include <Eigen/Sparse>
#include <boost/program_options.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>

template <typename T>
struct show_name
{
};

#define PI 3.141592653589793238462643383279502884197

using namespace std;
using namespace boltzmann;

namespace po = boost::program_options;

#ifdef EXTENDED_PRECISION
typedef long double numeric_t;
#else
typedef double numeric_t;
#endif

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

  /*
   * load coefficients (polar basis) from HDF5
   */
  const unsigned int N = polar_basis.n_dofs();
  Eigen::VectorXd coeffs(N);
  hid_t h5_init = H5Fopen("init.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
  eigen2hdf::load(h5_init, "coeffs", coeffs);
  H5Fclose(h5_init);
  coeffs.setZero();
  coeffs[0] = 1;
  {
    cout << "input: ||cp||^2: " << coeffs.cwiseAbs2().sum() << endl;
    ofstream fout("cp.dat");
    fout << coeffs;
    fout.close();
  }

  // compute bulk velocity
  Mass mass;
  mass.init(polar_basis);
  Momentum momentum;
  momentum.init(polar_basis);

  {
    auto entries = momentum.entries();
    for (auto entry : entries) {
      cout << entry.first << " " << entry.second << "\n";
    }
  }

  MQEval mqtsc(polar_basis);
  auto mq_eval = mqtsc.evaluator();
  mq_eval(coeffs.data(), N);
  cout << "correct mass: " << mq_eval.m << endl;
  cout << "correct momentum: " << mq_eval.v.transpose() << endl;

  const double m = mass.compute(coeffs.data());
  Eigen::Vector2d u = momentum.compute(coeffs.data()) / m;
  cout << "\n----- input -----\n"
       << "\n";
  cout << scientific << setprecision(8) << "mass: " << m << endl
       << "momentum: " << u(0) << ", " << u(1) << endl;

  // compute hermite coefficients
  Polar2Hermite<polar_basis_t, hermite_basis_t> P2H(polar_basis, hermite_basis);
  // print_timer(timer.stop(), "init P2H");

  Eigen::VectorXd buf(N);
  P2H.to_hermite(buf, coeffs);

  if (sizeof(numeric_t) == 16) {
    cout << "Using *extended precision*  in ShiftHermite\n";
  } else if (sizeof(numeric_t) == 8) {
    cout << "Using double precision in ShiftHermite\n";
  }
  std::vector<numeric_t> cH(buf.data(), buf.data() + N);
  typedef Eigen::Array<numeric_t, Eigen::Dynamic, 1> array_t;
  Eigen::Map<const array_t> vec_cH(cH.data(), cH.size());
  cout << "||c_H||^2: " << vec_cH.cwiseAbs2().sum() << "\n";
  ShiftHermite2D<hermite_basis_t, numeric_t> shift_hermite(hermite_basis);
  shift_hermite.init();
  timer.start();
  shift_hermite.shift(cH.data(), u(0), u(1));
  //  print_timer(timer.stop(), "shift Hermite coefficients");

  // convert to double
  std::transform(cH.begin(), cH.end(), buf.data(), [](numeric_t x) { return double(x); });

  // -> Polar coordinates
  Eigen::VectorXd Cc(N);
  P2H.to_polar(Cc, buf);

  const double mc = mass.compute(Cc.data());
  Eigen::Vector2d uc = momentum.compute(Cc.data()) / mc;
  cout << "\n----- centered -----\n";
  cout << "mass: " << scientific << setprecision(8) << mc << "\t(diff = " << std::abs(m - mc) << ")"
       << endl
       << "momentum: " << uc(0) << ", " << uc(1) << endl;

  // write new coefficients to disk
  hid_t h5_shifted = H5Fcreate("shifted.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  Eigen::Map<Eigen::VectorXd> Cc_eigen(Cc.data(), Cc.size());
  eigen2hdf::save(h5_shifted, "coeffs", Cc_eigen);
  // export hermite coefficients
  Eigen::Map<Eigen::VectorXd> cH_eigen(buf.data(), buf.size());
  eigen2hdf::save(h5_shifted, "coeffs_hermite", cH_eigen);
  H5Fclose(h5_shifted);

  // do some cheap scattering
  // ...

  // Move back to original position
  timer.start();
  shift_hermite.shift(cH.data(), -u(0), -u(1));
  //  print_timer(timer.stop(), "shift Hermite coefficients (back)");

  // go back to polar coordinates
  std::transform(cH.begin(), cH.end(), buf.data(), [](numeric_t x) { return double(x); });
  Eigen::VectorXd Cc2(N);
  P2H.to_polar(Cc2, buf);

  const double m1 = mass.compute(Cc2.data());
  Eigen::Vector2d u1 = momentum.compute(Cc2.data()) / m1;
  // stop here

  auto M = make_mass_matrix(polar_basis, polar_basis);

  cout << "----- move to original pos. -----\n";
  cout << "mass: " << scientific << setprecision(8) << m1 << "\t(diff = " << std::abs(m - m1) << ")"
       << endl
       << "momentum: " << scientific << setprecision(8) << u1(0) << ", " << u1(1)
       << "\t(diff = " << (u - u1).squaredNorm() << ")" << endl;

  Eigen::Map<Eigen::VectorXd> coeffs2(Cc2.data(), Cc2.size());
  Eigen::VectorXd tmp = (coeffs - coeffs2).array().square();
  double shift_error = sqrt((M * tmp).sum());

  cout << "shift_error: " << scientific << setprecision(8) << shift_error << endl;

  return 0;
}

#include "quadrature/qhermite.hpp"
#include "spectral/basis/spectral_basis.hpp"
#include "spectral/basis/spectral_basis_factory_hermite.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"
#include "spectral/basis/spectral_elem.hpp"
#include "spectral/basis/spectral_elem_accessor.hpp"
#include "spectral/basis/spectral_function/hermite_polynomial.hpp"

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

/**
 * @brief Check coefficients P->H, H->P
 *
 * @param polar_basis
 * @param hermite_basis
 */
template <typename H, typename P>
void test1(const P& polar_basis, const H& hermite_basis)
{
  Polar2Hermite<P, H> P2H(polar_basis, hermite_basis);
  P2H.exportmat("p2h.hdf5");
  unsigned int N = polar_basis.n_dofs();

  std::vector<double> C(N, 1);

  //  C[N-1] = 1.0;
  std::vector<double> Ch(N, 0);
  std::vector<double> Cb(N, 0);

  P2H.to_hermite(Ch, C);
  P2H.to_polar(Cb, Ch);

  cout << setw(5) << "index" << setw(20) << "c" << setw(20) << "T(P->H) T(H->P) c" << setw(20)
       << "error" << endl;
  for (unsigned int i = 0; i < N; ++i) {
    cout << setw(5) << i << "\t" << setw(20) << Cb[i] << setw(20) << C[i] << setw(20)
         << setprecision(6) << scientific << std::abs(Cb[i] - C[i]) << endl;
  }
  cout << "finished\n";
}

/**
 * @brief Polar -> Hermite, evaluate at different points
 *
 * @param polar_basis
 * @param hermite_basis
 */
template <typename H, typename P>
void test2(const P& polar_basis, const H& hermite_basis)
{
  Polar2Hermite<P, H> P2H(polar_basis, hermite_basis);

  /*
   *  Initialize polar coefficients
   */
  const unsigned int N = polar_basis.n_dofs();
  // std::vector<double> C(N, 0);
  // auto elem = SpectralBasisFactoryKS::make_elem(0, K-1, TRIG::COS);
  // unsigned int idx = polar_basis.get_dof_index(elem.get_id());
  // exp(-0.5 r^2)
  // C[idx] = 1.0;

  // if (idx  >= 2)
  //   C[idx-2] = 1.0;
  std::vector<double> C(N, 1.0);

  /*
   * Compute hermite coefficients
   */
  std::vector<double> Ch(N);
  P2H.to_hermite(Ch, C);

  // // output
  // std::ofstream fout("transformed-coefficients");
  // for (size_t i = 0; i < Ch.size(); ++i) {
  //   fout << Ch[i] << endl;
  // }
  // fout.close();

  /*
   * CHECK: do the hermite and polar series expansion match at evaluation
   * points?
   */
  {
    auto evalH = [&](const std::vector<double>& c, double x, double y) {
      auto itc = c.begin();
      double val = 0;
      for (auto it = hermite_basis.begin(); it != hermite_basis.end(); ++it, ++itc) {
        val += it->evaluate_weighted(x, y) * (*itc);
      }
      return val;
    };

    auto evalB = [&](const std::vector<double>& c, double phi, double r) {
      auto itc = c.begin();
      double val = 0;
      for (auto it = polar_basis.begin(); it != polar_basis.end(); ++it, ++itc) {
        val += it->evaluate_weighted(phi, r) * (*itc);
      }
      return val;
    };

    double r = 0.1;
    const int n = 40;
    for (int i = 0; i < n; ++i) {
      double phi = 2 * PI * i / (n + 1);
      double x = r * cos(phi);
      double y = r * sin(phi);

      const double fB = evalB(C, phi, r);
      const double fH = evalH(Ch, x, y);
      cout << setprecision(6) << scientific << setw(20) << fB << "\t" << setw(20) << fH << "\t"
           << setw(20) << std::abs(fB - fH) << endl;
    }
    r = 2.5;
    cout << "r = " << r << endl;
    for (int i = 0; i < n; ++i) {
      double phi = 2 * PI * i / (n + 1);
      double x = r * cos(phi);
      double y = r * sin(phi);

      const double fB = evalB(C, phi, r);
      const double fH = evalH(Ch, x, y);
      cout << setprecision(6) << scientific << setw(20) << fB << "\t" << setw(20) << fH << "\t"
           << setw(20) << std::abs(fB - fH) << endl;
    }
  }
}

int main(int argc, char* argv[])
{
  int K;
  po::options_description options("options");
  options.add_options()
      ("help", "show help message")
      ("nK,K", po::value<int>(&K)->default_value(10), "K");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, options), vm);
  po::notify(vm);

  cout << "using \n\tK: " << K << "\n";

  if (vm.count("help")) {
    cout << options << "\n";
    return 0;
  }

  typedef typename SpectralBasisFactoryKS::basis_type polar_basis_t;
  polar_basis_t polar_basis;
  SpectralBasisFactoryKS::create(polar_basis, K, K, 2, true);
  SpectralBasisFactoryKS::write_basis_descriptor(polar_basis, "spectral_basis.desc");

  typedef typename SpectralBasisFactoryHN::basis_type hermite_basis_t;
  hermite_basis_t hermite_basis;
  SpectralBasisFactoryHN::create(hermite_basis, K, 2);
  SpectralBasisFactoryHN::write_basis_descriptor(hermite_basis, "hermite_basis.desc");

  cout << "size(polar basis) = " << polar_basis.n_dofs() << endl
       << "size(hermite basis) = " << hermite_basis.n_dofs();

  cout << "\n--------------------\n";
  cout << "Test 1: (P->H) -> (H->P) show coefficients\n";
  test1(polar_basis, hermite_basis);

  cout << "\n--------------------\n";
  cout << "Test 2: (P->H) and compare evaluation at point of c_H, c_P\n";
  test2(polar_basis, hermite_basis);

  return 0;
}

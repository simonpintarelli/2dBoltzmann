#include <fstream>
#include <iomanip>
#include <iostream>

//#include "quadrature/gauss_hermite_quadrature.hpp"
#include "quadrature/qhermite.hpp"
#include "quadrature/qhermitew.hpp"
#include "spectral/hermiten.hpp"
#include "spectral/hermitenw.hpp"

using namespace std;
using namespace boltzmann;

typedef double numeric_t;

int main(int argc, char* argv[])
{
  if (argc < 2) {
    cerr << "syntax: " << argv[0] << " N" << endl << "\tN: num. quad. points";
    exit(-1);
  }

  unsigned int N = atoi(argv[1]);

  QHermiteW quad(1.0, N);

  HermiteN<numeric_t> H(N);
  H.compute(quad.pts());

  auto& wts = quad.wts();
  auto& pts = quad.pts();

  ofstream fout("quad.dat");
  for (unsigned int i = 0; i < pts.size(); ++i) {
    fout << setw(30) << scientific << setprecision(20) << pts[i] << "\t" << setw(30) << scientific
         << setprecision(20) << wts[i] << endl;
  }
  fout.close();

  cout << "Errors of (h_i, h_i)_R\n";
  for (unsigned i = 0; i < N; ++i) {
    double val = 0;
    for (unsigned int q = 0; q < quad.size(); ++q) {
      val += H.get(i)[q] * H.get(i)[q] * quad.wts(q) * exp(-quad.pts(q) * quad.pts(q));
    }
    cout << setw(10) << i << setw(30) << scientific << setprecision(20) << std::abs(val - 1.0)
         << endl;
  }

  HermiteNW<numeric_t> HW(N);
  HW.compute(quad.pts());

  ofstream hwout("hermite-weighted.dat");
  for (unsigned int i = 0; i < N + 1; ++i) {
    for (unsigned int q = 1; q < quad.size(); ++q)
      hwout << setprecision(16) << scientific << HW.get(i)[q] << "\t";
    hwout << endl;
  }
  hwout.close();

  cout << "Errors of weighted (h_i, h_i)_R\n";
  for (unsigned i = 0; i < N; ++i) {
    double val = 0;
    for (unsigned int q = 0; q < quad.size(); ++q) {
      // const double wh = exp(0.5*quad.pts(q) * quad.pts(q));
      val += (HW.get(i)[q]) * (HW.get(i)[q]) * quad.wts(q);
    }
    cout << setw(10) << i << setw(30) << scientific << setprecision(20) << std::abs(val - 1.0)
         << endl;
  }

  return 0;
}

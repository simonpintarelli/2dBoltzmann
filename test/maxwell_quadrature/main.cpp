
#include <boost/lexical_cast.hpp>
#include <boost/math/constants/constants.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "quadrature/qmaxwell.hpp"

using namespace std;
using namespace boltzmann;


// template<typename MAP>
// void print(const MAP& m, std::ofstream& fout, string title)
// {
//   fout << "----- " << title << endl;
//   for (auto it = m.begin(); it != m.end(); ++it) {
//     fout << it->first.first << "\t"
//          << it->first.second
//          << "\t"
//          << setprecision(16) << it->second
//          << endl;
//   }
// }

// template<typename CONT>
// void print_basis(const CONT& cont, std::ofstream& fout)
// {
//   for (int i = 0; i < cont.size(); ++i) {
//     fout  << cont[i].get_id() << endl;
//   }
// }

// ----------------------------------------------------------------------
int main(int argc, char *argv[])
{
  const double beta = 2;

  if (argc < 2) {
    cerr << "info: " << argv[0] << " N" << endl << "q: No. quad. points\n";
    return 1;
  }
  int N = atoi(argv[1]);

  int digits = 128;
  QMaxwell qmaxwell(1, N, digits);

  std::ofstream fout("quadrule_order" + boost::lexical_cast<string>(N) + "_" +
                     boost::lexical_cast<string>(digits) + ".dat");
  for (int i = 0; i < qmaxwell.size(); ++i) {
    fout << setprecision(30) << qmaxwell.pts(i) << "\t" << setprecision(30) << qmaxwell.wts(i)
         << endl;
  }
  fout.close();

  // test integration
  cout << "evaluate integral: \\int_0^{2pi} \\int_0^\\infty e^{-r^} r \\dd r"
       << "\n";
  double sum = 0;
  for (int i = 0; i < N; ++i) {
    sum += qmaxwell.wts(i);
  }
  const double pi = boost::math::constants::pi<double>();
  sum *= 2 * pi;
  cout << "\terror:" << std::abs(sum - pi) << "\n";

  return 0;
}

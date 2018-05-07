#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>
#include "spectral/basis/spectral_basis_factory_ks.hpp"
#include "matrix/assembly/velocity_var_form.hpp"

using namespace std;
using namespace boltzmann;

namespace po = boost::program_options;

int
main(int argc, char *argv[])
{
  int K;
  po::options_description options("options");
  options.add_options()
      ("help", "produce help message")
      ("nK,K", po::value<int>(&K)->required(), "K");
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, options), vm);
    po::notify(vm);
    if (vm.count("help")) {
      cout << options << "\n";
      return 0;
    }
  } catch (std::exception& e) {
    cout << e.what() << "\n";
    return 1;
  }
  cout << "K: " << K << "\n";

  SpectralBasisFactoryKS::basis_type basis;
  SpectralBasisFactoryKS::create(basis, K);

  VelocityVarForm<2> ventries;
  ventries.init(basis);

  {
    cout << "writing s0.dat\n";
    auto s0 = ventries.get_s0m();
    std::ofstream fout("s0.dat");
    s0.print(fout);
    fout.close();
  }
  {
    cout << "writing s1.dat\n";
    auto s1 = ventries.get_s1m();
    std::ofstream fout("s1.dat");
    s1.print(fout);
    fout.close();
  }
  {
    cout << "writing t2.dat\n";
    auto t2 = ventries.get_t2m();
    std::ofstream fout("t2.dat");
    t2.print(fout);
    fout.close();
  }

  return 0;
}

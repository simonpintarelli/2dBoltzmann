#include <omp.h>
#include <Eigen/Sparse>
#include <boost/lexical_cast.hpp>
#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <functional>
#include <iomanip>
#include <iostream>
// own includes ------------------------------------------------------------
#include "aux/message.hpp"
#include "aux/timer.hpp"
//#include "base/numbers.hpp"

#include <mpi.h>

#include "spectral/basis/spectral_basis_factory_ks.hpp"
#include "collision_tensor/collision_tensor_galerkin.hpp"

using namespace std;
using namespace boltzmann;
namespace po = boost::program_options;


int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  int K;
  string tensor_file;

  po::options_description options("options");
  options.add_options()
      ("help", "produce help message")
      ("K", po::value<int>(&K), "K")
      ("tensor", po::value<string>(&tensor_file), "/path/to/tensor/h5")
      ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, options), vm);
  po::notify(vm);

  if (vm.count("help")) {
    cout << options << "\n";
    return 0;
  }

  if (!boost::filesystem::exists(tensor_file)) {
    throw std::runtime_error("path not found: " + tensor_file);
  }

  typedef typename SpectralBasisFactoryKS::basis_type basis_type;
  // trial space
  basis_type trial_basis;
  SpectralBasisFactoryKS::create(trial_basis, K);

  CollisionTensorGalerkin ct(trial_basis);
  ct.read_hdf5(tensor_file.c_str());

  MPI_Finalize();
  return 0;
}

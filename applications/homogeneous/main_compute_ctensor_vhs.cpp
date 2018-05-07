#include <omp.h>
#include <Eigen/Sparse>
#include <algorithm>
#include <boost/lexical_cast.hpp>
#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>
#include <functional>
#include <iomanip>
#include <iostream>
// own includes ------------------------------------------------------------
#include "bte_config.h"
#include "aux/message.hpp"
#include "aux/timer.hpp"
#include "base/numbers.hpp"
#include "collision_tensor/assembly/collision_tensor_exporter.hpp"
#include "collision_tensor/assembly/ct_factory.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"
#include "quadrature/qmaxwell.hpp"
#include "quadrature/qmidpoint.hpp"
#include "spectral/utility/utility.hpp"

using namespace std;
using namespace boltzmann;
namespace po = boost::program_options;

typedef QMaxwell QRadial;
typedef SpectralBasisFactoryKS basis_factory_t;

int main(int argc, char* argv[])
{
  double beta, dt, s, lambda;
  int K, nptsa, nptsr, nptsi;
  // bool augmented;
  string tensor_file;

  po::options_description options("options");
  options.add_options()("help", "produce help message")
      (",K", po::value<int>(&K), "K")
      (",l", po::value<double>(&lambda), "|v-v_s|^lambda")
      ("pg", "Petrov-Galerkin formulation")
      ("nptsr,r", po::value<int>(&nptsr)->default_value(21), "#quad. points in radius")
      ("nptsa,a", po::value<int>(&nptsa)->default_value(21), "#quad. points in angle")
      ("nptsi,i", po::value<int>(&nptsi)->default_value(81), "#quad. points for inner loop")
      ("eigen_threads", po::value<int>()->default_value(1), "number of threads used in eigen (nested!)");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, options), vm);
  po::notify(vm);

  if (vm.count("help")) {
    cout << options << "\n";
    return 0;
  }

  std::string version_id = GIT_SHA1;
  cout << "\nVersionID: " << version_id << "@" << GIT_BNAME << std::endl;

  const int dim = 2;

  cout << "Calling arguments:\n"
       << right << setw(15) << "lambda" << setw(12) << lambda << endl
       << right << setw(15) << "K" << setw(12) << K << endl
       << right << setw(15) << "#quad. points radius" << setw(12) << nptsr << endl
       << right << setw(15) << "#quad. points angle" << setw(12) << nptsa << endl
       << right << setw(15) << "#quad. points inner" << setw(12) << nptsi << endl
       << "\n\n\n";

  char* numthreadsenv;
  int n_total_threads = 1;
  numthreadsenv = getenv("OMP_NUM_THREADS");
  Eigen::initParallel();

  int eigen_threads = 1;
  if (vm.count("eigen_threads")) eigen_threads = vm["eigen_threads"].as<int>();
  Eigen::setNbThreads(eigen_threads);
  cout << "Eigen is using " << Eigen::nbThreads() << " threads \n";
  if (numthreadsenv != 0) n_total_threads = atoi(numthreadsenv);

  // Create test & trial basis
  typedef typename basis_factory_t::basis_type basis_type;
  // trial space
  basis_type trial_basis;
  basis_factory_t::create(trial_basis, K);
  basis_factory_t::write_basis_descriptor(trial_basis);
  // test space
  basis_type test_basis;
  if (vm.count("pg")) {
    cout << "Petrov-Galerkin\n";
    basis_factory_t::create_test(test_basis, K);
  } else {
    basis_factory_t::create(test_basis, K);
  }
  basis_factory_t::write_basis_descriptor(test_basis, "spectral_basis_test.desc");

  typedef collision_tensor_assembly::
      CollisionOperator<basis_type, KERNEL_TYPE::VHS, QMidpoint, QRadial>
          collision_operator_t;

  CollisionTensorExporter exporter("collision_tensor.h5");

  collision_operator_t collision_operator(
      test_basis, trial_basis, nptsa, nptsr, nptsi, lambda);

  std::vector<unsigned int> work;
  for (unsigned int i = 0; i < trial_basis.n_dofs(); ++i) {
    work.push_back(i);
  }

  collision_operator.compute(exporter, work);

  return 0;
}

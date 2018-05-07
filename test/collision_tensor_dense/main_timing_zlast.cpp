#include <Eigen/Dense>
#include <algorithm>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include "bte_config.h"
#include "aux/rdtsc_timer.hpp"
#include "collision_tensor/collision_tensor_galerkin.hpp"
#include "collision_tensor/dense/collision_tensor_zfirst.hpp"
#include "collision_tensor/dense/collision_tensor_zlast.hpp"
#include "collision_tensor/dense/multi_slices_factory.hpp"
#include "collision_tensor/dense/storage/vbcrs_sparsity.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"

namespace po = boost::program_options;
namespace bf = boost::filesystem;

#include <yaml-cpp/yaml.h>

using namespace std;
using namespace boltzmann;

typedef ct_dense::CollisionTensorZLast ct_dense_t;
typedef SpectralBasisFactoryKS basis_factory_t;
typedef SpectralBasisFactoryKS::basis_type basis_type;

int
main(int argc, char* argv[])
{
  // dummy call, internal storage of ct_dense is using MPI shmem, therefore:
  MPI_Init(&argc, &argv);

  std::string version_id = GIT_SHA1;
  cout << "VersionID: " << version_id << "@" << GIT_BNAME << std::endl;


  int nrep;
  if (argc < 2) {
    cerr << "usage: " << argv[0] << " nrep"
         << "\n";
    return 1;
  } else {
    nrep = atoi(argv[1]);
    cerr << "nrep: " << nrep << "\n";
  }

  // Load config
  if (!boost::filesystem::is_regular_file("config.yaml")) {
    cout << "config file not found\n";
    return 1;
  }
  YAML::Node config = YAML::LoadFile("config.yaml");
  const size_t K = config["SpectralBasis"]["deg"].as<size_t>();

  // create basis
  basis_type basis;
  SpectralBasisFactoryKS::create(basis, K);

  unsigned int N = basis.n_dofs();
  ct_dense_t ct_dense(basis);
  ct_dense.import_entries_mpishmem(config["Scattering"]["file"].as<std::string>());

  Eigen::VectorXd x(N);
  x.setRandom();

  RDTSCTimer timer;

  Eigen::VectorXd y1(N);
  for (int i = 0; i < nrep; ++i) {
    timer.start();
    ct_dense.apply(y1, x);
    auto tlap = timer.stop();
    timer.print(cout, tlap, "dense");
  }

  MPI_Finalize();
  return 0;
}

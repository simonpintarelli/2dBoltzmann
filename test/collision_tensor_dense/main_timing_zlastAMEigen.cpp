#include <Eigen/Dense>
#include <algorithm>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "bte_config.h"
#include "aux/rdtsc_timer.hpp"
#include "aux/timer.hpp"
#include "collision_tensor/collision_tensor_galerkin.hpp"
#include "collision_tensor/dense/collision_tensor_zlastAM_eigen.hpp"
#include "collision_tensor/dense/storage/vbcrs_sparsity.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"
#ifdef LIKWID
#include <likwid.h>
#endif

namespace po = boost::program_options;
namespace bf = boost::filesystem;

#include <yaml-cpp/yaml.h>

using namespace std;
using namespace boltzmann;

typedef ct_dense::CollisionTensorZLastAMEigen ct_dense_t;
typedef SpectralBasisFactoryKS basis_factory_t;
typedef SpectralBasisFactoryKS::basis_type basis_type;



int
main(int argc, char* argv[])
{
  // dummy call, internal storage of ct_dense is using MPI shmem, therefore:
  MPI_Init(&argc, &argv);

  std::string version_id = GIT_SHA1;
  cout << "VersionID: " << version_id << "@" << GIT_BNAME << std::endl;

  int nin, nrep, vblksize;
  if (argc < 4) {
    cerr << "usage: " << argv[0] << " nin nrep"
         << "\nnin: number of input vectors"
         << "\nnrep: repeat timings this many times"
         << "\nvblksize: vblksize (VBCRS sparsity)"
         << "\n";
    return 1;
  } else {
    nin = atoi(argv[1]);
    nrep = atoi(argv[2]);
    vblksize = atoi(argv[3]);
    cerr << "nin: " << nin << "\n";
    cerr << "nrep: " << nrep << "\n";
    cerr << "vblksize: " << vblksize << "\n";
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

  typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> array_t;

  Timer<> stimer;
  stimer.start();
  ct_dense_t ct_dense(basis, nin);
  stimer.print(std::cout, stimer.stop(), "tensor constructor");

  stimer.start();
  ct_dense.import_entries_mpishmem(config["Scattering"]["file"].as<std::string>(),
                                   vblksize);
  stimer.print(std::cout, stimer.stop(), "loading tensor");

  int npadded = ct_dense.padded_vector_length();
  array_t xb(npadded, nin);
  xb.setOnes();
  array_t yb(N, nin);
  cout << "doing timings..." << "\n";

#ifdef LIKWID
  LIKWID_MARKER_INIT;
  LIKWID_MARKER_REGISTER("CT");
  LIKWID_MARKER_START("CT");
#else
  RDTSCTimer timer;
#endif

  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 0; i < nrep; ++i) {
#ifndef LIKWID
    timer.start();
#endif
    ct_dense.apply(yb, xb);
#ifndef LIKWID
    auto clap = timer.stop();
    timer.print(cout, clap, "dense");
#endif
  }

  cout << "done" << "\n";

#ifdef LIKWID
  LIKWID_MARKER_STOP("CT");
  LIKWID_MARKER_CLOSE;
#endif

  MPI_Finalize();

  return 0;
}


#include <yaml-cpp/yaml.h>
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
#include <mpi.h>
#include "bte_config.h"
#include "aux/rdtsc_timer.hpp"
#include "aux/timer.hpp"
#include "collision_tensor/collision_tensor_galerkin_sparse.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"

#ifdef LIKWID
#include <likwid.h>
#endif


namespace po = boost::program_options;
namespace bf = boost::filesystem;


using namespace std;
using namespace boltzmann;

typedef SpectralBasisFactoryKS basis_factory_t;
typedef SpectralBasisFactoryKS::basis_type basis_type;


int
main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  std::string version_id = GIT_SHA1;
  cout << "VersionID: " << version_id << "@" << GIT_BNAME << std::endl;

  int nin, nrep;
  if (argc < 3) {
    cerr << "usage: " << argv[0] << " nin nrep"
         << "\nnin: number of input vectors"
         << "\nnrep: repeat timings this many times"
         << "\n";
    return 1;
  } else {
    nin = atoi(argv[1]);
    nrep = atoi(argv[2]);
    cerr << "nin: " << nin << "\n";
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

  typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> array_t;
  CollisionTensorGalerkinSparse ct(basis, nin);
  ct.read_hdf5(config["Scattering"]["file"].as<std::string>().c_str());

  array_t xb(N, nin);
  xb.setOnes();
  array_t yb(N, nin);

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
    ct.apply(yb, xb);
#ifndef LIKWID
    auto clap = timer.stop();
    timer.print(cout, clap, "sparse_blocked");
#endif
  }

#ifdef LIKWID
  LIKWID_MARKER_STOP("CT");
  LIKWID_MARKER_CLOSE;
#endif

  MPI_Finalize();
  return 0;
}

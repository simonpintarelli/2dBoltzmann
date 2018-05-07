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
#include <mpi.h>
#ifdef LIKWID
#include <likwid.h>
#endif

// own includes ---------------------------------------------------------------
#include "bte_config.h"
#include "collision_tensor/collision_tensor_galerkin.hpp"
#include "collision_tensor/dense/multi_slices_factory.hpp"
#include "aux/rdtsc_timer.hpp"
#include "aux/timer.hpp"
#include "collision_tensor/dense/storage/vbcrs_sparsity.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"

namespace po = boost::program_options;
namespace bf = boost::filesystem;

#include <yaml-cpp/yaml.h>

using namespace std;
using namespace boltzmann;

typedef SpectralBasisFactoryKS basis_factory_t;
typedef SpectralBasisFactoryKS::basis_type basis_type;

int main(int argc, char* argv[])
{
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

  // ----------------------------------------------------------------------
  // Load config
  if (!boost::filesystem::is_regular_file("config.yaml")) {
    cout << "config file not found\n";
    return 1;
  }
  YAML::Node config = YAML::LoadFile("config.yaml");
  const size_t K = config["SpectralBasis"]["deg"].as<size_t>();

  // ----------------------------------------------------------------------
  // create basis
  basis_type basis;
  SpectralBasisFactoryKS::create(basis, K);

  unsigned int n_velo_dofs = basis.n_dofs();

  Timer<> stimer;
  stimer.start();
  CollisionTensorGalerkin Q(basis);
  stimer.print(std::cout, stimer.stop(), "tensor constructor");

  stimer.start();
  Q.read_hdf5(config["Scattering"]["file"].as<std::string>().c_str());
  stimer.print(std::cout, stimer.stop(), "loading tensor");

  Eigen::VectorXd x(n_velo_dofs);
  x.setOnes();

#ifdef LIKWID
  LIKWID_MARKER_INIT;
  LIKWID_MARKER_REGISTER("CT");
  LIKWID_MARKER_START("CT");
#else
  RDTSCTimer timer;
#endif

  MPI_Barrier(MPI_COMM_WORLD);
  Eigen::VectorXd y2(n_velo_dofs);
  for (int i = 0; i < nrep; ++i) {
#ifndef LIKWID
    timer.start();
#endif
    Q.apply(y2.data(), x.data());
#ifndef LIKWID
    auto tlap = timer.stop();
    timer.print(cout, tlap, "sparse");
#endif
  }
#ifdef LIKWID
  LIKWID_MARKER_STOP("CT");
  LIKWID_MARKER_CLOSE;
#endif


  MPI_Finalize();
  return 0;
}

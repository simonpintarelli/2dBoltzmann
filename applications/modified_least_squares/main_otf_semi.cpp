#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/graph_coloring.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/types.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/numerics/vector_tools.h>

// system includes ------------------------------------------------------------
#include <EpetraExt_HDF5.h>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#ifdef GPROF
#include <gperftools/profiler.h>
#endif

// own includes ---------------------------------------------------------------
#include "bte_config.h"
#include "app/app.hpp"
#include "aux/debug_output.hpp"
#include "aux/message.hpp"
#include "aux/timer.hpp"
#include "base/logger.hpp"
//#include "collision_tensor/collision_tensor_factory.hpp"
#include "collision_tensor/collision_tensor.hpp"
#include "collision_tensor/collision_tensor_operator.hpp"
#include "export/epetra_row_matrix.hpp"
#include "export/export_dh.hpp"
#include "grid/grid_handler.hpp"
#include "init/import/load_coefficients.hpp"
#include "matrix/bc/boundary_conditions.hpp"
#include "matrix/bc/matrix_wrapper.hpp"
#include "matrix/dofs/dof_helper.hpp"
#include "matrix/dofs/dofindex_sets.hpp"
#include "matrix/system_matrix_handler.hpp"
#include "method/method.hpp"
#include "post_processing/xdmf_exporter.hpp"
#include "solver/solver_handler.hpp"
#include "spectral/basis/indexer.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"

using namespace std;
using namespace dealii;
using namespace boltzmann;
namespace bf = boost::filesystem;
namespace po = boost::program_options;

typedef SpectralBasisFactoryKS basis_factory_t;

const double PI = dealii::numbers::PI;

// define the method
const int dim = 2;
typedef App<2> app_t;
typedef Method<METHOD::MODLEASTSQUARES> method_t;

typedef TrilinosWrappers::MPI::Vector vector_t;
typedef TrilinosWrappers::Vector local_vector_t;
typedef TrilinosWrappers::SparseMatrix matrix_t;

int main(int argc, char* argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1 /* nthreads  */);
  const unsigned int process_id = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const unsigned int nprocs = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  ConditionalOStream pcout(std::cout, process_id == 0);

  // --------------------------------------------------------------------------------------------------
  // program options (read initial conditions from file)
  po::options_description options("options");
  options.add_options()
      ("help", "produce help message")
      ("init,i", po::value<string>()->default_value(""), "initial distribution, dset='coeffs'")
      ("export-dofs", "export dofhandler and exit")
      ("petrov-galerkin", "use petrov-galerkin method for collision tensor")
      ("condest", "estimate condition number")
      ("export-matrices", "export matrices as hdf5");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, options), vm);
    po::notify(vm);
    if (vm.count("help")) {
      pcout << options << "\n";
      return 0;
    }
  } catch (std::exception& e) {
    cout << e.what() << "\n";
    return 1;
  }

  auto& logger = Logger::GetInstance();
  if (process_id != 0) logger.detach_stdout();

  boltzmann::Timer<> timer;
  pcout << "using " << nprocs << " MPI x " << 1 << " OMP threads\n";
  pcout << "executable: " << argv[0] << endl << "Application info: " << app_t::info << endl;
  std::string version_id = GIT_SHA1;
  pcout << "VersionID: " << version_id << "@" << GIT_BNAME << std::endl;

  // ----------------------------------------
  // yaml config
  if (!boost::filesystem::is_regular_file("config.yaml")) {
    pcout << "config file not found\n";
    return 1;
  }
  YAML::Node config = YAML::LoadFile("config.yaml");
  pcout << "\n----------------------------------------\n";
  pcout << config << endl;
  pcout << "\n----------------------------------------\n";

  const size_t nK = config["SpectralBasis"]["deg"].as<size_t>();
  const double beta = 2;
  const double dt = config["TimeStepping"]["dt"].as<double>();
  // output frequency
  const size_t ntsteps = config["TimeStepping"]["N"].as<size_t>();
  // create spectral basis or load from file
  typedef basis_factory_t::basis_type basis_type;
  basis_type spectral_basis;
  basis_type spectral_test_basis;
  if (bf::exists("spectral_basis.desc")) {
    // read from file if possible
    // pcout << "reading basis from file" << std::endl;
    basis_factory_t::create(spectral_basis, "spectral_basis.desc");
  } else {
    bool sorted = true;
    basis_factory_t::create(spectral_basis, nK, nK, beta, sorted);
    if (process_id == 0) {
      basis_factory_t::write_basis_descriptor(spectral_basis, "spectral_basis.desc");
    }
  }

  if (bf::exists("spectral_basis_test.desc")) {
    basis_factory_t::create(spectral_test_basis, "spectral_basis_test.desc");
  } else {
    bool sorted = true;
    if (vm.count("petrov-galerkin"))
      basis_factory_t::create_test(spectral_test_basis, nK, nK, beta, sorted);
    else
      basis_factory_t::create(spectral_test_basis, nK, nK, beta, sorted);
    if (process_id == 0) {
      basis_factory_t::write_basis_descriptor(spectral_test_basis, "spectral_basis_test.desc");
    }
  }
  const size_t n_velo_dofs = spectral_basis.n_dofs();
  // --------------------------------------------------------------------------------------------------
  //
  // MESH
  timer.start();
  GridHandler<2> grid_handler;
  // grid_handler.init(config, GridPartitionerBC<2>(150, 4*n_velo_dofs,
  // nprocs));
  grid_handler.init(config, GridPartitionerBC<2>(1, 1, nprocs));
  const auto& dof_handler = grid_handler.dofhandler();
  print_timer(timer.stop(), "initialize mesh", pcout);
  if (process_id == 0) export_dh(dof_handler);  // export (DoF-Idx, Pos)
  if (vm.count("export-dofs")) {
    pcout << "\nvertex2dofix.dat/ dof.desc written to disk. Quit.\n";
    return 0;
  }

  // --------------------------------------------------------------------------------------------------
  // DoFs
  const size_t n_phys_dofs = dof_handler.n_dofs();

  const size_t n_dofs = n_velo_dofs * n_phys_dofs;
  pcout << "#DOFS: " << n_dofs << endl;
  pcout << "#VDOFS: " << n_velo_dofs << endl;
  Indexer<> indexer(n_phys_dofs, n_velo_dofs);
  DoFIndexSets dof_map(nprocs);
  dof_map.init(dof_handler, n_velo_dofs);
  pcout << "#DoFs = " << indexer.n_dofs() << endl;

  // --------------------------------------------------------------------------------------------------
  // Scattering
  bool has_scattering = (bool)config["Scattering"];
  typedef CollisionTensorOperatorBase collision_tensor_operator_t;
  collision_tensor_operator_t* QO = NULL;
  double kn = 1;  // Knudsen-number
  if (has_scattering) {
    std::string tensor_fname = config["Scattering"]["file"].as<std::string>().c_str();
    if (config["Scattering"]["Galerkin"] || !vm.count("petrov-galerkin")) {
      pcout << "CollisionTensor Galerkin\n";
      QO = new CollisionTensorOperatorG(
          dof_handler, spectral_basis, dof_map.locally_owned_dofs(process_id));
    } else {
      pcout << "CollisionTensor Petrov-Galerkin\n";
      QO = new CollisionTensorOperatorPG(
          dof_handler, spectral_basis, dof_map.locally_owned_dofs(process_id));
    }
    timer.start();
    QO->load_tensor(tensor_fname);
    print_timer(timer.stop(), "load collision_tensor", pcout);

    if (config["Scattering"]["kn"]) {
      kn = config["Scattering"]["kn"].as<double>();
    }
  }

  SystemMatrixHandler<method_t, app_t> system_matrix_handler(
      dof_handler, spectral_basis, indexer, dof_map, dt);

  auto local_dofs = dof_map.locally_owned_dofs(process_id);
  auto local_relevant_dofs = dof_map.locally_relevant_dofs();
  auto owned_phys_dofs = dof_map.locally_owned_phys_dofs(process_id);

#ifndef TIMING
  // ----------------------------------------------------------------------------------------------------
  // load initial distribution from file
  IndexSet ghost_dofs(local_relevant_dofs);
  ghost_dofs.subtract_set(local_dofs);
  vector_t mu_ghosted(local_dofs, ghost_dofs);
  vector_t mu(local_dofs);

  // this consumes a lot of memory ...
  {
    local_vector_t mu0(mu.size());
    if (vm["init"].as<string>().size() > 0) {
      pcout << "loading initial-distribution from file: " << vm["init"].as<string>() << endl;
      load_coefficients(mu0, vm["init"].as<string>(), dof_handler, indexer);
    } else {
      pcout << "setting initial distribution to zero"
            << "\n";
      std::fill(mu0.begin(), mu0.end(), 0);
    }
    mu = mu0;
  }

  unsigned int n_vtk = config["TimeStepping"]["export_vtk"]
                           ? config["TimeStepping"]["export_vtk"].as<unsigned int>()
                           : 1;

  XDMFH5Exporter xdmf_h5_exporter(dof_handler,
                                  spectral_basis,
                                  owned_phys_dofs,
                                  1 /* buffer size */,
                                  n_vtk /* output frequency  */);
  xdmf_h5_exporter(mu, 0);
#endif

  // ----------------------------------------------------------------------------------------------------
  SolverHandler solver_handler;
  auto& V = system_matrix_handler.get_lhs();

  // Boundary Conditions
  BoundaryConditions<method_t, app_t, impl::BdFacesManager> B(
      -1 * dt, /* -1 because it is moved to RHS*/
      dof_handler,
      spectral_basis,
      indexer,
      config);

  timer.start();
  solver_handler.init(config, V);
  print_timer(timer.stop(), "Preconditioner", pcout);

  auto& solver = solver_handler.get_solver();
  auto& P = solver_handler.get_preconditioner();
  auto& M = system_matrix_handler.get_rhs();

  vector_t inflow_rhs(local_dofs);
  vector_t bc(local_dofs);
  std::fill(inflow_rhs.begin(), inflow_rhs.end(), 0.0);
  B.assemble_rhs(inflow_rhs);
  inflow_rhs.compress(dealii::VectorOperation::add);

#ifdef HAVE_EPETRAEXT_HDF5
  if (vm.count("export-matrices")) {
    pcout << "Export matrices to hdf5\n";
    ExportEpetraHDF5 matrix_exporter(V.trilinos_matrix().Comm());
    matrix_exporter.write("A", V.trilinos_matrix());
    matrix_exporter.write("M", M.trilinos_matrix());
  }

  const auto& trilinos_vector = mu.trilinos_vector();
  EpetraExt::HDF5 epetra_exporter2(trilinos_vector.Comm());
  epetra_exporter2.Create("solution_vector" + boost::lexical_cast<string>(0) + ".h5");
  epetra_exporter2.Write(boost::lexical_cast<string>(0), trilinos_vector);
  epetra_exporter2.Flush();
  epetra_exporter2.Close();
  print_timer(timer.stop(), "EpetraExt IO", pcout);
#endif

  // ----------------------------------------------------------------------------------------------------
  // Timestepping
  pcout << "\n\n-------------------- TIME STEPPING --------------------\n\n";
  vector_t rhs(local_dofs);
  double current_time = 0;
  // dump full solution vector / export paraview data every xx timestep
  unsigned int n_dump =
      config["TimeStepping"]["dump"] ? config["TimeStepping"]["dump"].as<unsigned int>() : 0;

  for (unsigned int i = 1; i <= ntsteps; ++i) {
    timer.start();
    M.vmult(rhs, mu);
    print_timer(timer.stop(), "MV-product", pcout);
    timer.start();
    rhs.add(-1.0 * dt, inflow_rhs);
    B.apply(rhs, mu);
    solver.solve(V.trilinos_matrix(), mu, rhs, P);
    print_timer(timer.stop(), "timestep transport", pcout);

    if (has_scattering) {
      timer.start();
      QO->apply(mu, dt / kn);
      print_timer(timer.stop(), "timestep scattering", pcout);
    }

    current_time += dt;
    xdmf_h5_exporter(mu, current_time);
#ifdef HAVE_EPETRAEXT_HDF5
    if ((n_dump && (i % n_dump == 0)) || i == ntsteps) {
      timer.start();
      const auto& trilinos_vector = mu.trilinos_vector();
      EpetraExt::HDF5 epetra_exporter2(trilinos_vector.Comm());
      epetra_exporter2.Create("solution_vector" + boost::lexical_cast<string>(i) + ".h5");
      epetra_exporter2.Write(boost::lexical_cast<string>(i), trilinos_vector);
      epetra_exporter2.Flush();
      epetra_exporter2.Close();
      print_timer(timer.stop(), "EpetraExt IO", pcout);
    }
#endif
  }

  return 0;
}

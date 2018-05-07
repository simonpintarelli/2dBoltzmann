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
#include <algorithm>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/static_assert.hpp>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
// own includes ---------------------------------------------------------------
#include "bte_config.h"
#include "app/app.hpp"
#include "aux/debug_output.hpp"
#include "aux/message.hpp"
#include "collision_tensor/collision_tensor.hpp"
#include "export/epetra_row_matrix.hpp"
//#include "collision_tensor/collision_tensor_factory.hpp"
#include "collision_tensor/collision_tensor_operator.hpp"
#include "export/export_dh.hpp"
#include "grid/dof_mapper_periodic_distributed.hpp"
#include "grid/grid_handler_periodic.hpp"
#include "init/import/load_coefficients.hpp"
#include "matrix/assembly/assembly.hpp"
#include "matrix/assembly/velocity_var_form.hpp"
#include "matrix/dofs/dofindex_sets.hpp"
#include "matrix/dofs/periodic_utility.hpp"
#include "matrix/system_matrix_handler.hpp"
#include "method/method.hpp"
#include "post_processing/xdmf_exporter.hpp"
#include "solver/aztecOO_condest.hpp"
#include "solver/solver_handler.hpp"
#include "spectral/basis/indexer.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"

using namespace std;
using namespace dealii;
using namespace boltzmann;
namespace bf = boost::filesystem;
namespace po = boost::program_options;

typedef SpectralBasisFactoryKS basis_factory_t;

const int dim = 2;
const double PI = dealii::numbers::PI;

// define the method
typedef Method<METHOD::MODLEASTSQUARES> method_t;
typedef App<dim> app_t;

typedef TrilinosWrappers::MPI::Vector vector_t;
// typedef TrilinosWrappers::Vector vector_local_t;
typedef TrilinosWrappers::SparseMatrix matrix_t;

int
main(int argc, char* argv[])
{
  boltzmann::Timer<> timer;
  int nthreads;
  // --------------------------------------------------
  // program options (read initial conditions from file)
  string scratch_dir;
  po::options_description options("options");
  options.add_options()("help", "produce help message")(
      "export-matrices", "export matrices to hdf5 file")("export-dofs", "export dofs and exit")(
      "petrov-galerkin", "use Galerkin disc. for scattering")(
      "threads,t", po::value<int>(&nthreads)->default_value(1), "number of threads")(
      "init,i", po::value<string>()->default_value(""), "initial distribution, dset='coeffs'")(
      "condest", "estimate condition number");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, options), vm);
  po::notify(vm);
  if (vm.count("help")) {
    cout << options << "\n";
    return 0;
  }

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, nthreads);
  const unsigned int process_id = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const unsigned int nprocs = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  ConditionalOStream pcout(std::cout, process_id == 0);
  pcout << "using " << nprocs << " MPI x " << nthreads << " OMP threads\n";
  pcout << "executable: " << argv[0] << endl << "periodic boundary conditions" << endl;

  std::string version_id = GIT_SHA1;
  pcout << "VersionID: " << version_id << "@" << GIT_BNAME << std::endl;

  // ----------------------------------------
  // yaml config
  if (!boost::filesystem::is_regular_file("config.yaml")) {
    pcout << "config file not found\n";
    return 1;
  }
  YAML::Node config = YAML::LoadFile("config.yaml");

  // make sure convergence messages etc. ware written only by proc 0
  if (process_id != 0) dealii::deallog.depth_console(0);
  pcout << "\n----------------------------------------\n";
  pcout << config << endl;
  pcout << "\n----------------------------------------\n";

  const size_t nK = config["SpectralBasis"]["deg"].as<size_t>();
  const double beta = 2;
  const double dt = config["TimeStepping"]["dt"].as<double>();
  // output frequency
  const size_t ntsteps = config["TimeStepping"]["N"].as<size_t>();

  // ----------------------------------------
  // MESH
  timer.start();
  GridHandlerPeriodic<2> grid_handler;
  grid_handler.init(config, nprocs);
  print_timer(timer.stop(), "initialize mesh", pcout);
  const auto& dof_handler = grid_handler.dofhandler();
  if (process_id == 0) export_dh(dof_handler);
  // ----------------------------------------------------------------------
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
    if (process_id == 0)
      basis_factory_t::write_basis_descriptor(spectral_basis, "spectral_basis.desc");
  }

  if (bf::exists("spectral_basis_test.desc")) {
    basis_factory_t::create(spectral_test_basis, "spectral_basis_test.desc");
  } else {
    bool sorted = true;
    if (vm.count("petrov-galerkin"))
      basis_factory_t::create_test(spectral_test_basis, nK, nK, beta, sorted);
    else
      basis_factory_t::create(spectral_test_basis, nK, nK, beta, sorted);

    if (process_id == 0)
      basis_factory_t::write_basis_descriptor(spectral_test_basis, "spectral_basis_test.desc");
  }

  if (vm.count("export-dofs")) {
    pcout << "L: " << dof_handler.n_dofs() << "\n N: " << spectral_basis.n_dofs()
          << "\n L x N: " << dof_handler.n_dofs() * spectral_basis.n_dofs() << "\n";

    pcout << "\nvertex2dofix.dat/ dof.desc written to disk. Exit.\n";
    return 0;
  }

  // ----------------------------------------------------------------------------------------------------
  // DoFs
  const size_t n_phys_dofs = dof_handler.n_dofs();
  const size_t n_velo_dofs = spectral_basis.n_dofs();
  const size_t n_dofs = spectral_basis.n_dofs() * dof_handler.n_dofs();

  pcout << "#vertices on grid"
        << "\t" << n_phys_dofs << endl
        << "dim(V)"
        << "\t" << n_velo_dofs << endl;
  // regular indexer on full grid (used for post_processing and output)
  Indexer<> global_indexer(n_phys_dofs, n_velo_dofs);
  // indexer used for DoF enumeration
  typedef DoFMapperPeriodicDistributed<> dof_mapper_t;
  dof_mapper_t dof_mapper;
  dof_mapper.init(dof_handler, nprocs);
  IndexSet owned_phys_dofs = dof_mapper.owned_dofs(process_id);
  IndexSet relevant_phys_dofs = dof_mapper.relevant_dofs(process_id);

  DoFIndexSetsPeriodic index_sets(owned_phys_dofs, relevant_phys_dofs, n_velo_dofs, process_id);

  IndexSet owned_dofs = index_sets.locally_owned_dofs(process_id);
  IndexSet relevant_dofs = index_sets.locally_relevant_dofs();
  Indexer<dof_mapper_t> indexer(dof_mapper, n_phys_dofs, n_velo_dofs);
  pcout << "#DoFs = " << indexer.n_dofs() << endl;
  /// DoFIndexSets provides index_maps on distributed (unrestricted) grid
  // ----------------------------------------------------------------------
  // Velocity matrix entries
  VelocityVarForm<2> velocity_var_form;
  velocity_var_form.init(spectral_basis, beta);
  // --------------------------------------------------------------------------------------------------
  // Scattering
  bool has_scattering = (bool) config["Scattering"];
  CollisionTensorOperatorBase* QO = NULL;
  double kn = 1;  // Knudsen-number
  if (has_scattering) {
    std::string tensor_fname = config["Scattering"]["file"].as<std::string>().c_str();
    if (config["Scattering"]["Galerkin"] && !vm.count("petrov-galerkin")) {
      pcout << "CollisionTensor Galerkin\n";
      QO = new CollisionTensorOperatorG(
          dof_handler, spectral_basis, index_sets.locally_owned_dofs(process_id));
    } else if (config["Scattering"]["Petrov-Galerkin"] && vm.count("petrov-galerkin")) {
      pcout << "CollisionTensor Petrov-Galerkin\n";
      QO = new CollisionTensorOperatorPG(
          dof_handler, spectral_basis, index_sets.locally_owned_dofs(process_id));
    } else {
      // default
      pcout << "CollisionTensor Galerkin (DENSE)\n";
      QO = new CollisionTensorOperatorDense<>(
          dof_handler, spectral_basis, index_sets.locally_owned_dofs(process_id));
    }
    timer.start();
    QO->load_tensor(tensor_fname);
    print_timer(timer.stop(), "load collision_tensor", pcout);

    if (config["Scattering"]["kn"]) {
      kn = config["Scattering"]["kn"].as<double>();
    }
  }

  SystemMatrixHandler<method_t, app_t> system_matrix_handler(
      dof_handler, spectral_basis, indexer, index_sets, dt);

  const auto& A = system_matrix_handler.get_lhs();
  const auto& M = system_matrix_handler.get_rhs();

#ifdef HAVE_EPETRAEXT_HDF5
  if (vm.count("export-matrices")) {
    pcout << "Export matrices to hdf5\n";
    ExportEpetraHDF5 matrix_exporter(A.trilinos_matrix().Comm());
    matrix_exporter.write("A", A.trilinos_matrix());
    matrix_exporter.write("M", M.trilinos_matrix());
  }
#endif

  // estimate condition number
  if (vm.count("condest")) {
    int maxiters = 100;
    // estimate condest from BC
    AztecOOCondest condestCG;
    condestCG.initialize(A, "GMRES");
    double condCG = condestCG.compute(maxiters);
    pcout << "estimated condition number (GMRES): " << condCG << endl;
  }

  // ----------------------------------------------------------------------
  // Initial conditions
  // create f0 on all DoFs
  vector<double> f0_full_grid(n_velo_dofs * n_phys_dofs);
  pcout << "loading initial-distribution from file: " << vm["init"].as<string>() << endl;
  load_coefficients(f0_full_grid, vm["init"].as<string>(), dof_handler, global_indexer);
  // convert between periodic and non-periodic solution on phase space grid
  vector_t mu(owned_dofs);
  vector_t mu_ghosted(owned_dofs, relevant_dofs);
  // ----------------------------------------------------------------------
  // Output
  GridHelper grid_helper(n_velo_dofs, n_phys_dofs);
  grid_helper.to_restricted_grid(mu, f0_full_grid, global_indexer, indexer, owned_phys_dofs);

  /*
   *   hint: DoFIndexSets contains full grid indices
   */
  DoFIndexSets dof_map(nprocs);
  dof_map.init(dof_handler, n_velo_dofs);
  IndexSet ghosts = dof_map.locally_relevant_dofs();
  ghosts.subtract_set(dof_map.locally_owned_dofs(process_id));

  vector_t mu_f(dof_map.locally_owned_dofs(process_id));
  mu_ghosted = mu;
  auto dofs = dof_map.locally_owned_phys_dofs(process_id);
  unsigned int n_vtk = config["TimeStepping"]["export_vtk"]
                           ? config["TimeStepping"]["export_vtk"].as<unsigned int>()
                           : 1;
  XDMFH5Exporter xdmf_exporter(
      dof_handler, spectral_basis, dof_map.locally_owned_phys_dofs(process_id), 1, n_vtk);

  grid_helper.to_full_grid(mu_f, mu_ghosted, global_indexer, indexer, dofs);
  xdmf_exporter(mu_f, 0);
  // mu_f_ghosted = mu_f;
  //  output.run(mu_f_ghosted, 0, 0, global_indexer);
  // ----------------------------------------------------------------------------------------------------
  // Timestepping
  pcout << "\n\n-------------------- TIME STEPPING --------------------\n\n";
  vector_t rhs(owned_dofs);
  vector_t sc(owned_dofs);
  double current_time = 0;
  // dump full solution vector / export paraview data every xx timestep
  unsigned int n_dump =
      config["TimeStepping"]["dump"] ? config["TimeStepping"]["dump"].as<unsigned int>() : 0;

  SolverHandler solver_handler;
  timer.start();
  solver_handler.init(config, A);
  print_timer(timer.stop(), "Preconditioner", pcout);
  auto& solver = solver_handler.get_solver();
  auto& P = solver_handler.get_preconditioner();

#ifdef HAVE_EPETRAEXT_HDF5
  // write full solution to disk
  const auto& trilinos_vector = mu_f.trilinos_vector();
  EpetraExt::HDF5 epetra_exporter(trilinos_vector.Comm());
  epetra_exporter.Create("solution_vector" + boost::lexical_cast<string>(0) + ".h5");
  epetra_exporter.Write(boost::lexical_cast<string>(0), trilinos_vector);
  epetra_exporter.Flush();
  epetra_exporter.Close();
#endif

  for (unsigned int i = 1; i <= ntsteps; ++i) {
    timer.start();
    M.vmult(rhs, mu);
    print_timer(timer.stop(), "MV-product", pcout);
    timer.start();
    solver.solve(A, mu, rhs, P);
    auto ctr = solver.control();
    pcout << "Solver::GMRES " << ctr.last_step() << "\t" << ctr.last_value() << "\n";
    print_timer(timer.stop(), "timestep transport", pcout);

    if (has_scattering) {
      timer.start();
      QO->apply(mu, dt / kn);
      print_timer(timer.stop(), "timestep scattering", pcout);
    }

    current_time += dt;
    mu_ghosted = mu;
    grid_helper.to_full_grid(mu_f, mu_ghosted, global_indexer, indexer, dofs);
    xdmf_exporter(mu_f, current_time);

#ifdef HAVE_EPETRAEXT_HDF5
    if ((n_dump && (i % n_dump == 0)) || i == ntsteps) {
      timer.start();
      const auto& trilinos_vector = mu_f.trilinos_vector();
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

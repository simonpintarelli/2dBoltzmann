// deal.II includes ----------------------------------------------
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>

// system includes -----------------------------------------------
#include <hdf5.h>
#include <omp.h>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <ctime>
#include <fstream>
#include <iostream>
// from eigen unsupported
#include <Eigen/KroneckerProduct>

// own includes --------------------------------------------------
#include "aux/eigen2hdf.hpp"
#include "aux/message.hpp"
#include "aux/timer.hpp"
#include "grid_transfer.hpp"
#include "init/import/load_coefficients.hpp"
#include "post_processing/energy.hpp"
#include "post_processing/mass.hpp"
#include "post_processing/momentum.hpp"

#include "l2errors.hpp"
#include "spectral/basis/indexer.hpp"
// class SimpleGridHandler, Solution
#include "grid/grid_tools.hpp"
#include "outer_product_helper.hpp"
#include "solution_handler.hpp"
#include "spectral_transfer_matrix.hpp"

#include "export/data_out_hdf5.hpp"

using namespace boltzmann;
using namespace std;

const int dim = 2;

namespace bf = boost::filesystem;
namespace po = boost::program_options;

template <typename SPECTRAL_BASIS>
void make_overlap(std::vector<double>& S, const SPECTRAL_BASIS& spectral_basis)
{
  typedef typename SPECTRAL_BASIS::elem_t elem_t;

  // angular basis
  typedef typename std::tuple_element<0, typename SPECTRAL_BASIS::elem_t::container_t>::type
      angular_elem_t;
  typename elem_t::Acc::template get<angular_elem_t> acc_ang;

  // inverse mass matrix
  S.resize(spectral_basis.n_dofs());
  for (unsigned int j = 0; j < spectral_basis.n_dofs(); ++j) {
    auto& elem = spectral_basis.get_elem(j);
    if (acc_ang(elem).get_id().l == 0)
      S[j] = numbers::PI;
    else
      S[j] = numbers::PI / 2;
  }
}

typedef dealii::DoFHandler<dim> dh_t;
typedef dealii::Vector<double> vector_t;

int main(int argc, char* argv[])
{
  // #pragma omp parallel
  // {
  //   #pragma omp master
  //   {
  //     int num_threads = omp_get_num_threads();
  //     Eigen::setNbThreads(num_threads);
  //     cout << "Eigen is using " << Eigen::nbThreads() << " threads \n";
  //   }

  // }

  boltzmann::Timer<> timer;

  po::options_description options("options");
  options.add_options()
      ("config,c", po::value<string>()->required(), "config file")
      ("help,h", "help");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, options), vm);
  if (vm.count("help")) {
    cout << options << "\n";
    return 0;
  }

  po::notify(vm);

  string config_name = vm["config"].as<string>();

  if (!boost::filesystem::is_regular_file(config_name)) {
    cout << "config file not found\n";
    return 1;
  }

  // load yaml config
  YAML::Node config = YAML::LoadFile(config_name);

  string str_input_grid = config["input"]["grid"].as<string>();
  string str_input_path = config["input"]["path"].as<string>();

  string str_ref_grid = config["reference"]["grid"].as<string>();
  string str_ref_path = config["reference"]["path"].as<string>();

  auto cwd = bf::current_path();

  // initialize working directory
  bf::path working_dir = bf::path(str_input_path) / bf::path("convergence_plots");
  bf::create_directory(working_dir);

  // initialize logfile
  std::time_t result = std::time(nullptr);
  cout << argv[0] << "at: " << asctime(localtime(&result)) << ", executed in " << cwd.c_str()
       << endl
       << setw(14) << "Solution: " << (working_dir / bf::path(str_input_path)).c_str() << setw(14)
       << "Reference: " << (working_dir / bf::path(str_ref_path)).c_str() << endl;

  // ------------------------------------------------------------------------------------------
  // transfer matrix
  dealii::FE_Q<dim> fe(1);
  auto ref_grid_ptr = make_shared<SimpleGridHandler>(str_ref_path, str_ref_grid);
  auto grid_ptr = make_shared<SimpleGridHandler>(str_input_path, str_input_grid);

  const auto& ref_dh = ref_grid_ptr->get_dofhandler();
  const auto& input_dh = grid_ptr->get_dofhandler();

  cout << "Reference mesh: " << ref_dh.get_triangulation().n_used_vertices() << " vertices, "
       << ref_dh.get_triangulation().n_active_cells() << " cells." << endl;

  timer.start();
  GridTransfer<dim> grid_transfer;
  grid_transfer.init(ref_dh, input_dh);
  const auto& Tx = grid_transfer.get_transfer_matrix();
  print_timer(timer.stop(), "init GridTransfer");

  // output to hdf5
  hid_t file;
  file = H5Fcreate((working_dir / bf::path("transfer_matrix.h5")).c_str(),
                   H5F_ACC_TRUNC,
                   H5P_DEFAULT,
                   H5P_DEFAULT);
  eigen2hdf::save_sparse(file, "Tx", Tx);

  // spectral transfer matrix
  auto Tv =
      spectral_transfer_matrix(ref_grid_ptr->get_spectral_basis(), grid_ptr->get_spectral_basis());
  eigen2hdf::save_sparse(file, "Tv", Tv);

  // timer.start();
  // Eigen::SparseMatrix<double> T = Eigen::kroneckerProduct(Tx, Tv);
  // print_timer(timer.stop(), "kroneckerProduct");
  // timer.start();
  // eigen2hdf::save_sparse(file, "T", T);
  // print_timer(timer.stop(), "dump transfer matrix");

  // ------------------------------------------------------------------------------------------
  // load solution filenames from config
  // S contains (b_i (v), b_i (v))_R^2
  vector<double> S;
  make_overlap(S, ref_grid_ptr->get_spectral_basis());

  // load permutations from `vertex2dofidx.dat`
  std::vector<unsigned int> ref_perm(ref_grid_ptr->get_dofhandler().n_dofs());
  std::vector<unsigned int> input_perm(grid_ptr->get_dofhandler().n_dofs());
  // load permutation from ``
  load_permutation(ref_perm, str_ref_path);
  load_permutation(input_perm, str_input_path);

  auto ref_perm_tmp = v2d_permutation_vector(ref_dh);
  auto input_perm_tmp = v2d_permutation_vector(input_dh);

  Mass mass(ref_grid_ptr->get_spectral_basis());
  Momentum momentum(ref_grid_ptr->get_spectral_basis());
  Energy energy(ref_grid_ptr->get_spectral_basis());

  dealii::Vector<double> vmass(ref_dh.n_dofs());
  dealii::Vector<double> venergy(ref_dh.n_dofs());
  dealii::Vector<double> vux(ref_dh.n_dofs());  // momentum x
  dealii::Vector<double> vuy(ref_dh.n_dofs());  // momnetum y

  unsigned int nsteps = config["timesteps"].size();
  cout << "__ERRORS__\n";
  cout << setw(15) << "# i" << setw(15) << "t" << setw(15) << "l2_squared" << setw(15)
       << "l2_m_squared" << setw(15) << "l2_u_squared" << setw(15) << "l2_e_squared" << endl;

  for (unsigned int i = 0; i < nsteps; ++i) {
    string str_h5loc_ref = config["timesteps"][i]["reference"]["data"].as<string>();
    string str_h5loc_inp = config["timesteps"][i]["input"]["data"].as<string>();
    double time = config["timesteps"][i]["time"].as<double>();

    Eigen::VectorXd v_inp;  // approximate solution
    Eigen::VectorXd v_ref;  // reference solution

    load_solution_vector(v_inp, str_input_path, str_h5loc_inp);
    load_solution_vector(v_ref, str_ref_path, str_h5loc_ref);

    // transform to vertex ordering
    to_vertex_ordering(v_inp, input_perm);
    to_vertex_ordering(v_ref, ref_perm);

    // transform to active dofhandler ordering
    to_dof_ordering(v_inp, input_perm_tmp);
    to_dof_ordering(v_ref, ref_perm_tmp);

    // Eigen::VectorXd v_sol = T*v_inp;
    Eigen::VectorXd v_sol(v_ref.size());
    sparse_outer_product_multiply(v_sol, Tx, Tv, v_inp);

    Errors errors;
    double l2_error_sq =
        errors.compute(ref_dh, v_sol.data(), v_ref.data(), S, ref_grid_ptr->get_indexer());

    Eigen::VectorXd vdiff = v_sol - v_ref;
    mass.compute(vmass.begin(), vdiff.data(), ref_dh.n_dofs());
    energy.compute(venergy.begin(), vdiff.data(), ref_dh.n_dofs());
    momentum.compute(vux.begin(), vuy.begin(), vdiff.data(), ref_dh.n_dofs());
    std::for_each(vmass.begin(), vmass.end(), [](double v) { return std::abs(v); });
    std::for_each(venergy.begin(), venergy.end(), [](double v) { return std::abs(v); });
    std::transform(vux.begin(), vux.end(), vuy.begin(), vux.begin(), [](double x, double y) {
      return std::sqrt(x * x + y * y);
    });

    double l2diff_m = l2norm(ref_dh, vmass);
    double l2diff_u = l2norm(ref_dh, vux);
    double l2diff_e = l2norm(ref_dh, venergy);

    cout << setw(15) << i << setw(15) << time << setw(15) << scientific << setprecision(5)
         << l2_error_sq << setw(15) << scientific << setprecision(5) << l2diff_m << setw(15)
         << scientific << setprecision(5) << l2diff_u << setw(15) << scientific << setprecision(5)
         << l2diff_e << endl;

    // output
    {
      // dealii::DataOut<dim> data_out;
      dealii::DataOutHDF<dim> data_out;
      // data_out.attach_triangulation(ref_dh.get_triangulation());
      data_out.attach_dof_handler(ref_dh);
      const auto& cell_wise_error = errors.get_cell_wise_error();
      //      cout << "cell_wise_error.size = " << cell_wise_error.size() << endl;
      data_out.add_data_vector(cell_wise_error, "error^2");
      data_out.add_data_vector(vmass, "err_mass");
      data_out.add_data_vector(venergy, "err_energy");
      data_out.add_data_vector(vux, "err_abs(u)");
      //  data_out.add_data_vector(tmp_out, "test_direct");
      data_out.build_patches();

      dealii::DataOutBase::VtkFlags flags;
      data_out.set_flags(flags);
      // auto fname = "errors" + boost::lexical_cast<string>(i) + ".vtk";
      // std::ofstream vtk_output((working_dir / bf::path(fname)).c_str());
      // data_out.write_vtk(vtk_output);
      // vtk_output.close();
      typedef dealii::DataOutBase::DataOutFilterFlags data_out_filter_flags;
      dealii::DataOutBase::DataOutFilter data_out_filter(data_out_filter_flags(true, true));
      data_out.write_filtered_data(data_out_filter);
      string filename =
          (working_dir / bf::path("output" + boost::lexical_cast<string>(i) + ".h5")).c_str();
      string mesh_filename = (working_dir / bf::path("mesh.hdf5")).c_str();
      data_out.write_hdf5(filename, data_out_filter);
      auto xdmf_entry = data_out.create_xdmf_entry(data_out_filter, mesh_filename, filename, time);

      auto xdmf_file = working_dir / bf::path("solution.xdmf");
      std::ofstream fout(xdmf_file.c_str(), std::ios_base::out | std::ios_base::app);
      fout << xdmf_entry.get_xdmf_content(1) << std::endl;
      fout.close();

      if (i == 0) {
        data_out.write_mesh(mesh_filename, data_out_filter);
      }
    }
  }

  H5Fclose(file);

  return 0;
}

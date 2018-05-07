/**
 * @file   main_grid_transfer_test.cpp
 * @author  <simon@thinkpadX1>
 * @date   Tue Mar 31 15:27:16 2015
 *
 * @brief  Just for debugging purposes!
 *         GridTransfer (dh, dh) => should yield identity matrix
 *
 *
 */

// deal.II includes ----------------------------------------------
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/lac/vector.h>

// system includes -----------------------------------------------
#include <hdf5.h>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
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

#include "spectral/basis/indexer.hpp"

#include "l2errors.hpp"
// class SimpleGridHandler, Solution
#include "grid/grid_tools.hpp"
#include "solution_handler.hpp"
#include "spectral_transfer_matrix.hpp"

const int dim = 2;

namespace bf = boost::filesystem;
namespace po = boost::program_options;

using namespace boltzmann;
using namespace std;

const std::string spectral_basis_fname = "spectral_basis.desc";

typedef dealii::DoFHandler<dim> dh_t;
typedef dealii::Vector<double> vector_t;

int main(int argc, char* argv[])
{
  boltzmann::Timer<> timer;

  po::options_description options("options");
  options.add_options()
      ("config,c", po::value<string>()->required(), "config file")
      ("help,h", "help");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, options), vm);
  po::notify(vm);

  string config_name = vm["config"].as<string>();

  if (!boost::filesystem::is_regular_file(config_name)) {
    cout << "config file not found\n";
    return 1;
  }

  YAML::Node config = YAML::LoadFile(config_name);

  string str_input_grid = config["input"]["grid"].as<string>();
  string str_input_path = config["input"]["path"].as<string>();
  string str_input_solution = config["input"]["solution"].as<string>();

  string str_ref_grid = config["reference"]["grid"].as<string>();
  string str_ref_path = config["reference"]["path"].as<string>();
  string str_ref_solution = config["reference"]["solution"].as<string>();

  dealii::FE_Q<dim> fe(1);
  shared_ptr<SimpleGridHandler> ref_grid_ptr =
      make_shared<SimpleGridHandler>(str_ref_path, str_ref_grid);
  shared_ptr<SimpleGridHandler> grid_ptr =
      make_shared<SimpleGridHandler>(str_input_path, str_input_grid);

  const auto& ref_dh = ref_grid_ptr->get_dofhandler();
  const auto& input_dh = grid_ptr->get_dofhandler();

  GridTransfer<dim> grid_transfer;
  grid_transfer.init(ref_dh, ref_dh);
  const auto& Tx = grid_transfer.get_transfer_matrix();

  // output to hdf5
  hid_t file;
  file = H5Fcreate("transfer_matrix.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  eigen2hdf::save_sparse(file, "Tx", Tx);

  // // load solution data
  // typedef Solution<SimpleGridHandler> solution_t;
  // solution_t input_solution(grid_ptr, str_input_path, str_input_solution);
  // solution_t ref_solution(ref_grid_ptr, str_ref_path, str_ref_solution);

  // auto& mu_input = input_solution.get_solution();

  // spectral transfer matrix
  auto Tv =
      spectral_transfer_matrix(ref_grid_ptr->get_spectral_basis(), grid_ptr->get_spectral_basis());
  eigen2hdf::save_sparse(file, "Tv", Tv);

  // timer.start();
  // Eigen::SparseMatrix<double> T = Eigen::kroneckerProduct(Tx, Tv);
  // print_timer(timer.stop(), "make T");

  // timer.start();
  // eigen2hdf::save_sparse(file, "T", T);
  // print_timer(timer.stop(), "save T");

  // timer.start();
  // Eigen::VectorXd out = T*mu_input;
  // print_timer(timer.stop(), "interpolation to fine grid");

  // eigen2hdf::save(file, "coeffs", out);

  // auto vertex2dofidx = vertex_to_dof_index(ref_dh);
  // std::ofstream fout("v2d.dat");
  // for (auto it = vertex2dofidx.begin(); it != vertex2dofidx.end();
  //      ++it) {
  //   fout << it->first << "\t" << it->second << endl;
  // }
  // fout.close();

  // ------------------------------------------------------------------------------------------
  // TODO load coefficients

  // ------------------------------------------------------------------------------------------
  // compute errors, global, cell-wise

  // load basis file
  H5Fclose(file);

  return 0;
}

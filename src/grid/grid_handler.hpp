#pragma once

// own includes ---------------------------------------------------------
#include <grid/grid_tools.hpp>
// system includes ------------------------------------------------------
#include <yaml-cpp/yaml.h>
#include <boost/algorithm/string.hpp>
#include <cctype>
#include <fstream>
// deal.II includes -----------------------------------------------------
#include <deal.II/base/exceptions.h>

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include "grid_partitioner.hpp"
#include "grid_partitioner_bc.hpp"
#include "grid_handler_base.hpp"

// GridHandler
namespace boltzmann {

template <int dim>
class GridHandler : public GridHandlerBase<dim>
{
 private:
  typedef GridHandlerBase<dim> base_type;

 public:
  GridHandler();
  // void init(const dealii::ParameterHandler& param, int nprocs=1);
  /**
   *
   *
   * @param config
   * @param nvelo_dofs  required for load-balancing of METIS,
   *                    default: METIS ignores spectral basis
   *
   */
  void init(YAML::Node& config,
            const GridPartitioner<dim>& grid_partitioner = GridPartitioner<dim>());


 private:
  void init_square(int nref);
  void load_msh(std::string fname);

 protected:
  bool is_initialized;
  using base_type::fe;
  using base_type::triangulation_;
  using base_type::dofhandler_;
};

// ----------------------------------------------------------------------
template <int dim>
GridHandler<dim>::GridHandler()
    : base_type(),
      is_initialized(false)
{
  static_assert(dim == 2);
}

// ----------------------------------------------------------------------
template <int dim>
void
GridHandler<dim>::init(YAML::Node& config, const GridPartitioner<dim>& grid_partitioner)
{
  unsigned int nprocs = grid_partitioner.n_processes();

  if (config["Mesh"]["file"]) {
    this->load_msh(config["Mesh"]["file"].as<std::string>());
    this->dofhandler_.distribute_dofs(this->fe);

  } else if (config["Mesh"]["type"]) {
    int nref = config["Mesh"]["nref"].as<int>();
    std::string type = config["Mesh"]["type"].as<std::string>();
    if (std::strcmp("square", type.c_str()) == 0) {
      this->init_square(nref);
    } else {
      throw std::runtime_error("GridHandler: invalid yaml config");
    }
  } else {
    throw std::runtime_error("GridHandler: invalid yaml config");
  }

  if (nprocs > 1) {
    grid_partitioner.partition(dofhandler_, triangulation_);
  } else {
    dealii::DoFRenumbering::boost::Cuthill_McKee(dofhandler_);
  }
  is_initialized = true;
}


// ----------------------------------------------------------------------
template <int dim>
void
GridHandler<dim>::init_square(int nref)
{
  Assert(!is_initialized, dealii::ExcMessage("Attempt to initialize GridHandler twice"));
  bool colorize = true;
  dealii::GridGenerator::hyper_cube(triangulation_, 0, 1, colorize);
  this->triangulation_.refine_global(nref);
  this->dofhandler_.distribute_dofs(this->fe);
  dealii::DoFRenumbering::boost::Cuthill_McKee(dofhandler_);
  is_initialized = true;


  const unsigned int pid = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  if (pid == 0) {
    std::ofstream fout("square.msh");
    dealii::GridOut grid_out;
    grid_out.set_flags(dealii::GridOutFlags::Msh(true, true));
    grid_out.write_msh(triangulation_, fout);
    fout.close();
  }
}


// ------------------------------------------------------------------------
template <int dim>
void
GridHandler<dim>::load_msh(std::string fname)
{
  dealii::GridIn<dim> gridin;
  gridin.attach_triangulation(triangulation_);
  std::ifstream f(fname);
  gridin.read_msh(f);
}

}  // end namespace boltzmann

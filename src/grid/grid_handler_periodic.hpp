#pragma once

// deal.II includes
#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <yaml-cpp/yaml.h>
#include <cctype>
// own includes
#include "grid_handler_base.hpp"

namespace boltzmann {
// ----------------------------------------------------------------------

/**
 * @brief takes into account *all* ajdacent vertices
 *        when constructing the domain decomposition with METIS
 *        for *periodic* boundary conditions.
 *
 */
template <int dim>
class GridHandlerPeriodic : public GridHandlerBase<dim>
{
 private:
  typedef GridHandlerBase<dim> base_type;

 public:
  GridHandlerPeriodic();
  void init_square(int nref, int nprocs = 1);
  void init(dealii::ParameterHandler& param_handler, int nprocs = 1);
  void init(YAML::Node& config, int nprocs = 1);

 private:
  void make_graph();

 private:
  void load_msh(std::string fname);
  bool is_initialized;
  dealii::SparsityPattern graph;
  using base_type::triangulation_;
  using base_type::dofhandler_;
};

// ----------------------------------------------------------------------
template <int dim>
GridHandlerPeriodic<dim>::GridHandlerPeriodic()
    : GridHandlerBase<dim>()
    , is_initialized(false)
{
  static_assert(dim == 2, "dim must be 2, 3d code is not implemented");
}

// ----------------------------------------------------------------------
template <int dim>
void
GridHandlerPeriodic<dim>::init(dealii::ParameterHandler& param_handler, int nprocs)
{
  int n_refine = param_handler.get_integer("refine phys");
  std::string mesh = param_handler.get("mesh");
  if (!mesh.compare("")) {
    this->init_square(n_refine, nprocs);
  } else {
    throw std::runtime_error(
        "Grids for periodic boundary conditions should be generated and not loaded from file!");
    // load mesh grom file
    this->load_msh(mesh);
    this->dofhandler_.distribute_dofs(this->fe);
    if (nprocs > 1) {
      dealii::GridTools::partition_triangulation(nprocs, triangulation_);
      dealii::DoFRenumbering::subdomain_wise(dofhandler_);
    } else {
      dealii::DoFRenumbering::boost::Cuthill_McKee(dofhandler_);
    }
    is_initialized = true;
  }
}

// ----------------------------------------------------------------------
template <int dim>
void
GridHandlerPeriodic<dim>::init(YAML::Node& config, int nprocs)
{
  if (!config["Mesh"]["file"]) {
    int n_refine = config["Mesh"]["nref"].as<int>();
    this->init_square(n_refine, nprocs);
  } else {
    throw std::runtime_error(
        "Grids for periodic boundary conditions should be generated and not loaded from file!");
    std::string mesh = config["Mesh"]["file"].as<std::string>();
    // load mesh grom file
    this->load_msh(mesh);
    this->dofhandler_.distribute_dofs(this->fe);
    if (nprocs > 1) {
      dealii::GridTools::partition_triangulation(nprocs, triangulation_);
      dealii::DoFRenumbering::boost::Cuthill_McKee(dofhandler_);
      dealii::DoFRenumbering::subdomain_wise(dofhandler_);
    } else {
      dealii::DoFRenumbering::boost::Cuthill_McKee(dofhandler_);
    }
  }
  is_initialized = true;
}

// ----------------------------------------------------------------------
template <int dim>
void
GridHandlerPeriodic<dim>::init_square(int nref, int nprocs)
{
  AssertThrow(!is_initialized, dealii::ExcMessage("Attempt to initialize GridHandler twice"));
  bool colorize = true;
  dealii::GridGenerator::hyper_cube(triangulation_, 0, 1, colorize);
  triangulation_.refine_global(nref);
  dofhandler_.distribute_dofs(this->fe);
  if (nprocs > 1) {
    this->make_graph();
    dealii::GridTools::partition_triangulation(nprocs, graph, triangulation_);
    dealii::DoFRenumbering::boost::Cuthill_McKee(dofhandler_);
    dealii::DoFRenumbering::subdomain_wise(dofhandler_);
  } else {
    dealii::DoFRenumbering::boost::Cuthill_McKee(dofhandler_);
  }
  is_initialized = true;
}

// ----------------------------------------------------------------------
template <int dim>
void
GridHandlerPeriodic<dim>::make_graph()
{
  typedef long int lint;
  lint FUZZY = 1e6;
  typedef typename dealii::Triangulation<dim>::cell_iterator cell_t;
  //  typedef std::pair<cell_t, int> cell_face_pair_t;
  typedef std::map<lint, int> map_t;
  map_t hmap;
  map_t vmap;
  int nfaces = 4;
  dealii::DynamicSparsityPattern csp(triangulation_.n_active_cells());
  for (auto cell = triangulation_.begin_active(); cell != triangulation_.end(); ++cell) {
    for (int i = 0; i < nfaces; ++i) {
      auto neighbor = cell->neighbor(i);
      if (neighbor != triangulation_.end()) {
        csp.add(neighbor->index(), cell->index());
        csp.add(cell->index(), neighbor->index());
      }
    }
    if (cell->at_boundary()) {
      // vertical faces
      for (int f : {0, 1}) {
        if (cell->face(f)->at_boundary()) {
          auto center = cell->face(f)->center();
          // find horizontal neighbor
          auto it = vmap.find(lint(FUZZY * center[1]));
          if (it != vmap.end()) {
            // element found
            int index_other = it->second;
            csp.add(cell->index(), index_other);
            csp.add(index_other, cell->index());
            vmap.erase(it);
          } else {
            vmap.insert(it, std::make_pair(lint(center[1] * FUZZY), cell->index()));
          }
        }
      }
      // horizontal faces
      for (int f : {2, 3}) {
        if (cell->face(f)->at_boundary()) {
          auto center = cell->face(f)->center();
          // find horizontal neighbor
          auto it = hmap.find(lint(FUZZY * center[0]));
          if (it != hmap.end()) {
            // element found
            int index_other = it->second;
            csp.add(cell->index(), index_other);
            csp.add(index_other, cell->index());
            hmap.erase(it);
          } else {
            hmap.insert(it, std::make_pair(lint(center[0] * FUZZY), cell->index()));
          }
        }
      }
    }
  }
  // no single boundary faces
  AssertThrow(hmap.empty(), dealii::ExcMessage("unpaired horizontal face"));
  AssertThrow(vmap.empty(), dealii::ExcMessage("unpaired vertical face"));
  //  graph.reinit(triangulation_.n_active_cells());
  csp.compress();
  graph.copy_from(csp);
}

// ------------------------------------------------------------------------
template <int dim>
void
GridHandlerPeriodic<dim>::load_msh(std::string fname)
{
  dealii::GridIn<2> gridin;
  gridin.attach_triangulation(triangulation_);
  std::ifstream f(fname);
  gridin.read_msh(f);
  std::cout << "Number of active cells: " << triangulation_.n_active_cells() << std::endl;
  std::cout << "Total number of cells: " << triangulation_.n_cells() << std::endl;
}

}  // end namespace boltzmann

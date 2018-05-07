#pragma once

// deal.II includes ------------------------------------------------------------
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_pattern.h>
// system includes -------------------------------------------------------------
#include <boost/assert.hpp>
#include <unordered_map>

namespace boltzmann {

/**
 * @brief Taken from dealii::GridTools and slightly modified
 *        for the spectral formulation
 *
 */
void
partition_triangulation(const unsigned int n_partitions,
                        dealii::Triangulation<2, 2>& triangulation,
                        const unsigned int cell_weight = 1,
                        const unsigned int cell_weight_bc = 1);

/**
 *
 * @param sparsity Sparsity pattern
 * @param dh Dofhandler
 * @param dof_mapper provides mapping of DoFNumbers from full grid -> periodic grid
 */
template <int dim, typename DoFMapper>
void
make_periodic_connection_graph(dealii::SparsityPattern& sparsity,
                               const dealii::DoFHandler<dim>& dh,
                               const DoFMapper& dof_mapper)
{
  unsigned int restricted_size = dof_mapper.size();
  dealii::DynamicSparsityPattern csp(restricted_size);
  /// create sparsity pattern from dh
  dealii::DynamicSparsityPattern cphys(dh.n_dofs());
  dealii::DoFTools::make_sparsity_pattern(dh, cphys);
  dealii::SparsityPattern phys_sparsity(dh.n_dofs());
  phys_sparsity.copy_from(cphys);

  for (unsigned int ix = 0; ix < dh.n_dofs(); ++ix) {
    unsigned int ri = dof_mapper[ix];
    for (auto it_row = phys_sparsity.begin(ix); it_row != phys_sparsity.end(ix); ++it_row) {
      csp.add(ri, dof_mapper[*it_row]);
    }
  }

  csp.compress();
  sparsity.copy_from(csp);
}

// --------------------------------------------------------------------------------
/**
 * build a mapping (vertex index) -> (vertex DoF index)
 *
 * @param dh DoFHandler
 *
 * @return std::map<unsigned int, unsigned int>
 */
template <int dim>
std::unordered_map<unsigned int, unsigned int>
vertex_to_dof_index(const dealii::DoFHandler<dim>& dh)
{
  typedef unsigned int index_t;
  std::unordered_map<index_t, index_t> index_map;

  const auto& fe = dh.get_fe();
  unsigned int dofs_per_cell = fe.dofs_per_cell;

  for (auto cell = dh.begin_active(); cell < dh.end(); ++cell) {
    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
      index_map[cell->vertex_index(i)] = cell->vertex_dof_index(i, 0);
    }
  }

  return index_map;
}

/**
 * build a mapping (vertex index) -> (vertex DoF index)
 *
 * @param dh DoFHandler
 *
 * @return std::map<unsigned int, unsigned int>
 */
template <int dim>
std::unordered_map<unsigned int, unsigned int>
dof_to_vertex_index(const dealii::DoFHandler<dim>& dh)
{
  typedef unsigned int index_t;
  std::unordered_map<index_t, index_t> index_map;

  BOOST_ASSERT_MSG(dh.get_fe().n_components() == 1, "does not work for vector valued FE");

  const auto& fe = dh.get_fe();
  unsigned int dofs_per_cell = fe.dofs_per_cell;

  for (auto cell = dh.begin_active(); cell < dh.end(); ++cell) {
    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
      index_map[cell->vertex_dof_index(i, 0)] = cell->vertex_index(i);
    }
  }

  return index_map;
}

/**
 * @brief vertex index `i` -> dof_handler index `p[i]`
 *
 * @param dh Piecewise linear cont. FE DoFHandler
 *
 * @return vector<unsigned int> p
 */
template <int dim>
std::vector<unsigned int>
v2d_permutation_vector(const dealii::DoFHandler<dim>& dh)
{
  std::vector<unsigned int> p(dh.n_dofs());

  const auto& fe = dh.get_fe();
  unsigned int dofs_per_cell = fe.dofs_per_cell;

  for (auto cell = dh.begin_active(); cell < dh.end(); ++cell) {
    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
      p[cell->vertex_index(i)] = cell->vertex_dof_index(i, 0);
    }
  }

  return p;
}

}  // end namespace boltzmann

#pragma once

#include <boost/mpl/identity.hpp>
#include <deal.II/base/graph_coloring.h>
#include <deal.II/dofs/dof_handler.h>


namespace boltzmann {
namespace local_ {

template <typename DH>
struct make_graph_coloring_traits
{
};

template <int DIM>
struct make_graph_coloring_traits<dealii::DoFHandler<DIM> >
{
  typedef typename dealii::DoFHandler<DIM>::active_cell_iterator active_iterator_t;
};
}  // end namespace _local

template <typename DH, typename INDEXER>
std::vector<std::vector<typename local_::make_graph_coloring_traits<DH>::active_iterator_t> >
make_graph_coloring(const DH& dof_handler, const INDEXER& indexer)
{
  // number of dofs...
  const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;

  auto begin = dof_handler.begin_active();
  auto end = dof_handler.end();

  typedef decltype(begin) iterator_t;

  /*
   *   The function make_graph_coloring creates a colored graph of the FEM Matrix
   *
   *   Hint: indexer.to_global is used to take into account periodic boundary
   *   conditions (if present).
   */
  const std::function<std::vector<dealii::types::global_dof_index>(const iterator_t&)>
      get_conflict_indices = [&](const iterator_t& cell) {
        std::vector<dealii::types::global_dof_index> dof_idxs(dofs_per_cell);
        // apply indexer, set velocity index j = 0;
        cell->get_dof_indices(dof_idxs);

        const unsigned int j = 0;
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          dof_idxs[i] = indexer.to_global(dof_idxs[i], j);
        }
        return dof_idxs;
      };

  return dealii::GraphColoring::make_graph_coloring(begin, end, get_conflict_indices);
}

}  // end namespace boltzmann

#pragma once

// deal.II includes ------------------------------------------------------------
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
// system includes -------------------------------------------------------------
#include <algorithm>
#include <fstream>
#include <set>
#include <utility>
#include <vector>
// own includes ---------------------------------------------------------------
#include "aux/message.hpp"
#include "helpers.hpp"
#include "spectral/basis/global_indexer.hpp"
//#include "spectral/basis/spectral_function.hpp"
#include "matrix/assembly/velocity_var_form.hpp"
#include "matrix/dofs/dofindex_sets.hpp"
#include "matrix/sparsity_pattern/sparsity_pattern_base.hpp"

namespace boltzmann {

class SparsityPattern : public SparsityPatternBase
{
 private:
  typedef dealii::DoFHandler<2> dofhandler_t;
  typedef dealii::types::global_dof_index size_type;
  using SparsityPatternBase::sparsity_pattern_t;

 public:
  template <typename SPECTRAL_BASIS, typename INDEXER>
  void init(const SPECTRAL_BASIS& spectral_basis,
            const dofhandler_t& dof_handler,
            const sparsity_t& vsparsity,
            const INDEXER& indexer,
            const DoFIndexSetsBase& dof_map,
            const bool has_bc = true);

  virtual const sparsity_pattern_t& get_sparsity() const { return sparsity_pattern_; }

  virtual const dealii::SparsityPattern& get_L_sparsity() const { return L_sparsity_; }

  void init_trilinos_phys_sparsity(sparsity_pattern_t& sp, const DoFIndexSets& dof_map) const
  {
    unsigned int pid = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    const auto& local_idx = dof_map.locally_owned_phys_dofs(pid);
    sp.reinit(local_idx, phys_sparsity_);
  }

  const std::vector<bool>& get_boundary_vertex_indicator() const { return is_at_boundary; }

 private:
  sparsity_pattern_t sparsity_pattern_;
  //@{
  /// sparsity patterns in velocity dofs
  /// velocity nnz entries from differential operator L, stores (j1,j2)
  sparsity_t L_sparsity_;
  //@}
  dealii::SparsityPattern phys_sparsity_;
  // indicates boundary vertices
  std::vector<bool> is_at_boundary;
};

// ----------------------------------------------------------------------
template <typename SPECTRAL_BASIS, typename INDEXER>
void
SparsityPattern::init(const SPECTRAL_BASIS& polar_basis,
                      const dofhandler_t& dof_handler,
                      const sparsity_t& vsparsity,
                      const INDEXER& indexer,
                      const DoFIndexSetsBase& dof_partition,
                      const bool has_bc)
{
  unsigned int mpi_this_process = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  BAssertThrow(vsparsity.is_compressed(), "sparsity pattern must be compressed");

  // pepare sparsity patterns in velocity space
  L_sparsity_.copy_from(vsparsity);
  L_sparsity_.compress();

  // init_l_sparsity(polar_basis, velocity_var_form);

  const size_t N = polar_basis.n_dofs();
  const size_type L = dof_handler.n_dofs();
  // the sparsity pattern for the physical dofs
  dealii::DynamicSparsityPattern phys_block_sparsity(L, L);
  dealii::DoFTools::make_sparsity_pattern(dof_handler, phys_block_sparsity);
  // and now include the velocity degrees of freedom in the sparsity pattern
  phys_sparsity_.copy_from(phys_block_sparsity);
  phys_sparsity_.compress();

  // find vertices at boundary
  is_at_boundary = std::vector<bool>(dof_handler.n_dofs(), false);
  // build list of vertices at the boundary
  for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell) {
    if (cell->has_boundary_lines()) {
      for (unsigned int face_index = 0; face_index < dealii::GeometryInfo<2>::faces_per_cell;
           ++face_index) {  // iterate over faces of this cell
        if (cell->face(face_index)->at_boundary()) {
          for (unsigned int face_vertex = 0; face_vertex < 2; ++face_vertex) {
            is_at_boundary[cell->face(face_index)->vertex_dof_index(face_vertex, 0)] = true;
          }
        }
      }
    }
  }

  // count how many at boundary?
  unsigned int n_boundary = 0;
  std::for_each(is_at_boundary.begin(), is_at_boundary.end(), [&](bool v) {
    if (v) n_boundary += 1;
  });

  const auto& owned_dofs = dof_partition.locally_owned_dofs(mpi_this_process);
  const auto& relevant_x_dofs = dof_partition.locally_relevant_phys_dofs();
  const auto& relevant_dofs = dof_partition.locally_relevant_dofs();

  sparsity_pattern_.reinit(owned_dofs,    /* row parallel partitioning */
                           owned_dofs,    /* col parallel partitioning, dummy set */
                           relevant_dofs, /* writeable rows */
                           MPI_COMM_WORLD);
  std::vector<unsigned int> col_indices(N * phys_block_sparsity.max_entries_per_row());
  // Add nonzero entries from NON-boundary-DoFs
  for (size_t ix1 = 0; ix1 < L; ++ix1) {
    if (!relevant_x_dofs.is_element(indexer.to_restricted(ix1))) continue;
    //    if (subdomain_ids[ix1] != mpi_this_process) continue;
    for (size_t j1 = 0; j1 < N; ++j1) {
      // row index
      unsigned int ig1 = indexer.to_global(ix1, j1);
      // store nonzero col indices from current row into `col_indices`
      unsigned int rnnz = 0;
      for (auto ix2 = phys_sparsity_.begin(ix1); ix2 != phys_sparsity_.end(ix1); ++ix2) {
        if (has_bc && is_at_boundary[ix1] && is_at_boundary[ix2->column()]) {
          unsigned int col_idx = indexer.to_global(ix2->column(), 0);
          for (size_t j2 = 0; j2 < N; ++j2) {
            col_indices[rnnz + j2] = col_idx + j2;
          }
          rnnz += N;
        } else {
          for (auto j2 = L_sparsity_.begin(j1); j2 != L_sparsity_.end(j1); ++j2) {
            col_indices[rnnz++] = indexer.to_global(ix2->column(), j2->column());
          }
        }
      }
      sparsity_pattern_.add_entries(ig1, col_indices.data(), col_indices.data() + rnnz, true);
    }
  }
  sparsity_pattern_.compress();

  if (mpi_this_process == 0)
    std::cout << "\nglobal sparsity pattern #nnz: " << sparsity_pattern_.n_nonzero_elements()
              << std::endl;
}

}  // end namespace boltzmann

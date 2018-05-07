#pragma once

// deal.II includes -------------------------------------------------------
#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>

#ifdef DEBUG
#include <deal.II/base/conditional_ostream.h>
#endif

// system includes --------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace boltzmann {
namespace impl {

/**
 * @brief Collect local boundary faces and construct IndexSet (dofs on boundary)
 *
 */
class BdFacesManager
{
 private:
  constexpr const static int dimX = 2;
  typedef dealii::DoFCellAccessor<dealii::DoFHandler<dimX>, false> accessor_t;
  typedef int face_index_t;
  typedef std::tuple<accessor_t, face_index_t> faces_t;
  typedef std::vector<faces_t> vfaces_t;

 protected:
  BdFacesManager() = delete;

  template <typename DH, typename SPECTRAL_BASIS, typename INDEXER>
  BdFacesManager(const DH& dh, const SPECTRAL_BASIS& spectral_basis, const INDEXER& indexer);

  const vfaces_t& get_faces_list() const { return my_faces_; }

 private:
  vfaces_t my_faces_;

 protected:
  // boundary dofs (in global-xv enumeration)
  dealii::IndexSet relevant_dofs_;
};

template <typename DH, typename SPECTRAL_BASIS, typename INDEXER>
BdFacesManager::BdFacesManager(const DH& dh,
                               const SPECTRAL_BASIS& spectral_basis,
                               const INDEXER& indexer)
    : relevant_dofs_(indexer.n_dofs())
{
  const unsigned int pid = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  int dofs_per_face = dh.get_fe().dofs_per_face;
  std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_face);
  unsigned int N = spectral_basis.n_dofs();

  for (auto cell : dh.active_cell_iterators()) {
    if (cell->subdomain_id() != pid) continue;

    if (cell->at_boundary()) {
      // TODO: get number of faces from deal.II
      for (int face_idx = 0; face_idx < 4; ++face_idx) {
        if (cell->face(face_idx)->at_boundary()) {
          // add to work list
          my_faces_.push_back(std::make_tuple(*cell, face_idx));

          // add to IndexSet
          cell->face(face_idx)->get_dof_indices(local_dof_indices);
          for (auto ix : local_dof_indices) {
            auto iibegin = indexer.to_global(ix, 0);
            relevant_dofs_.add_range(iibegin, iibegin + N);
          }
        }
      }
    }
  }
  relevant_dofs_.compress();
}

}  // end namespace impl
}  // end namespace boltzmann

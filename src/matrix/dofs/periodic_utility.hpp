#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>

#include <iostream>

namespace boltzmann {
class GridHelper
{
 private:
  typedef dealii::types::global_dof_index size_type;

 public:
  GridHelper(const size_t velo_dofs_, const size_t phys_dofs_)
      : velo_dofs(velo_dofs_)
      , phys_dofs(phys_dofs_)
  { /* empty */
  }

  /**
   *
   *
   * @param dst
   * @param src
   * @param indexer_f     indexer on *full* phase space grid
   * @param indexer_r     indexer on *restriced* phase space grid (periodic enumeration)
   * @param owned_phys_dofs IndexSet indices from physical grid (NON-periodic!)
   *
   * @return
   */
  template <typename VEC_IN, typename VEC_OUT, typename INDEXER1, typename INDEXER2>
  void to_full_grid(VEC_OUT& dst,
                    const VEC_IN& src,
                    const INDEXER1& indexer_f,
                    const INDEXER2& indexer_r,
                    const dealii::IndexSet& owned_phys_dofs);

  /**
   *
   *
   * @param dst
   * @param indexer_f      indexer on *full* phase space grid
   * @param indexer_r      indexer on *restricted* phase space grid (periodic enumeration)
   * @param relevant_phys_dofs indices from periodic enumeration on *restricted*
   * *physical* grid
   *
   * @return
   */
  template <typename VEC_IN, typename VEC_OUT, typename INDEXER1, typename INDEXER2>
  void to_restricted_grid(VEC_OUT& dst,
                          const VEC_IN& src,
                          const INDEXER1& indexer_f,
                          const INDEXER2& indexer_r,
                          const dealii::IndexSet& relevant_phys_dofs);

 private:
  const size_t velo_dofs;
  const size_t phys_dofs;
};

// ----------------------------------------------------------------------
template <typename VEC_IN, typename VEC_OUT, typename INDEXER1, typename INDEXER2>
void
GridHelper::to_full_grid(VEC_OUT& dst,
                         const VEC_IN& src,
                         const INDEXER1& indexer_f,
                         const INDEXER2& indexer_r,
                         const dealii::IndexSet& owned_phys_dofs)
{
  for (unsigned int ixf = 0; ixf < phys_dofs; ++ixf) {
    if (owned_phys_dofs.is_element(ixf)) {
      // check if it is a relevant DoF!
      for (unsigned int j = 0; j < velo_dofs; ++j) {
        const int igf = indexer_f.to_global(ixf, j);
        const int igr = indexer_r.to_global(ixf, j);
        dst[igf] = src[igr];
      }
    }
  }
}

// ----------------------------------------------------------------------
template <typename VEC_IN, typename VEC_OUT, typename INDEXER1, typename INDEXER2>
void
GridHelper::to_restricted_grid(VEC_OUT& dst,
                               const VEC_IN& src,
                               const INDEXER1& indexer_f,
                               const INDEXER2& indexer_r,
                               const dealii::IndexSet& relevant_phys_dofs)
{
  //#pragma omp parallel for
  for (unsigned int ixf = 0; ixf < phys_dofs; ++ixf) {
    if (relevant_phys_dofs.is_element(indexer_r.to_restricted(ixf))) {
      for (unsigned int j = 0; j < velo_dofs; ++j) {
        const int igf = indexer_f.to_global(ixf, j);
        const int igr = indexer_r.to_global(ixf, j);
        dst[igr] = src[igf];
      }
    }
  }
}
}  // end namespace boltzmann

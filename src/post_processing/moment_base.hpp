#pragma once

#include <deal.II/base/index_set.h>
#include <deal.II/base/utilities.h>

// own includes ----------------------------------------------------------
#include "matrix/dofs/dofindex_sets.hpp"
#include "spectral/basis/global_indexer.hpp"


namespace boltzmann {
class MomentBase
{
 public:
  template <typename VEC_IN, typename VEC_OUT, typename INDEXER>
  void compute(VEC_OUT& dst,
               const VEC_IN& src,
               const dealii::IndexSet& relevant_phys_dofs,
               const INDEXER& indexer) const;

 protected:
  typedef double value_t;
  typedef std::pair<unsigned int, value_t> entry_t;
  unsigned int n_velo_dofs;
  std::vector<entry_t> contributions;

 public:
  const std::vector<entry_t>& entries() const { return contributions; }
};

// ----------------------------------------------------------------------
template <typename VEC_IN, typename VEC_OUT, typename INDEXER>
void
MomentBase::compute(VEC_OUT& dst,
                    const VEC_IN& src,
                    const dealii::IndexSet& relevant_phys_dofs,
                    const INDEXER& indexer) const
{
#pragma omp parallel for
  for (unsigned int ix = 0; ix < relevant_phys_dofs.size(); ++ix) {
    if (relevant_phys_dofs.is_element(ix)) {
      double sum = 0;
      for (unsigned int j = 0; j < contributions.size(); ++j) {
        unsigned int jx = contributions[j].first;
        sum += contributions[j].second * src[indexer.to_global(ix, jx)];
      }
      dst[ix] = sum;
    }
  }
}

}  // end namespace boltzmann

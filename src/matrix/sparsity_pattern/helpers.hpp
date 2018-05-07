#pragma once

#include <deal.II/lac/dynamic_sparsity_pattern.h>

namespace boltzmann {
namespace sparsity_helper {
template <typename ENTRIES_CONTAINER>

void
add_to_csp(dealii::DynamicSparsityPattern& csp, const ENTRIES_CONTAINER& obj)
{
  for (auto it = obj.begin(); it != obj.end(); ++it) {
    csp.add(it->row, it->col);
  }
}
}  // end namespace sparsity_helper
}  // end namespace boltzmann

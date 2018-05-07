#include "dof_helper.hpp"
#include <cassert>

#include <limits>

namespace boltzmann {
dealii::IndexSet replicate_index_set(const dealii::IndexSet& index_in, const unsigned int nrep)
{
  unsigned int n_phys_dofs = index_in.size();
  dealii::IndexSet index_out;
  index_out.set_size(n_phys_dofs * nrep);
  std::vector<bool> phys_indices(n_phys_dofs);
  index_in.fill_binary_vector(phys_indices);

  unsigned int i = 0;
  unsigned int set_begin = std::numeric_limits<unsigned int>::max();
  bool found_set = false;
  while (true) {
    const bool current_state = phys_indices[i];
    if (!found_set && (current_state == true)) {
      found_set = true;
      set_begin = i;
    } else if (found_set && (current_state == false)) {
      // add_range, adds the half open range [begin, end)
      assert(set_begin != std::numeric_limits<unsigned int>::max());
      index_out.add_range(set_begin * nrep, i * nrep);
      // set finished, go to next
      found_set = false;
    }
    // increment
    ++i;
    // termination
    if (i == n_phys_dofs) {
      // is there a last set pending?
      if (found_set && current_state == true) {
        assert(set_begin != std::numeric_limits<unsigned int>::max());
        index_out.add_range(set_begin * nrep, i * nrep);
      }
      break;
    }
  }
  index_out.compress();
  return index_out;
}

}  // end namespace boltzmann

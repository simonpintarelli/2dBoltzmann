#pragma once

#include <vector>
#include <deal.II/base/index_set.h>

namespace boltzmann {
/**
 * @brief this is used for graph coloring
 *
 * @param index_in
 * @param nrep      replicate every index nrep times, i.e. map a single index i to the range
 * [i*nrep, (i+1)*nrep)
 *
 * @return
 */
dealii::IndexSet
replicate_index_set(const dealii::IndexSet& index_in, const unsigned int nrep);

}  // end namespace boltzmann

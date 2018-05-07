#pragma once

#include <Eigen/Sparse>
#include <algorithm>

namespace boltzmann {

template <typename SPECTRAL_BASIS>
Eigen::SparseMatrix<double>
spectral_transfer_matrix(const SPECTRAL_BASIS& dst, const SPECTRAL_BASIS& src)
{
  unsigned int m = dst.n_dofs();
  unsigned int n = src.n_dofs();

  Eigen::SparseMatrix<double> T(m, n);

  for (unsigned int i = 0; i < m; ++i) {
    const auto& dst_elem = dst.get_elem(i);
    unsigned int j;
    try {
      j = src.get_dof_index(dst_elem.id());
    } catch (...) {
      continue;
    }
    T.insert(i, j) = 1;
  }

  return T;
}

}  // end namespace boltzmann

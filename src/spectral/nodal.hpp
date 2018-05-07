#pragma once

#include "quadrature/qhermitew.hpp"
#include <Eigen/Dense>


namespace boltzmann {

template <typename DERIVED, typename FUNC>
void
to_nodal(Eigen::DenseBase<DERIVED>& dst, const FUNC f)
{
  assert(dst.rows() == dst.cols());

  int K = dst.rows();
  QHermiteW quad(1.0, K);

  auto& x = quad.pts();
  auto& w = quad.wts();

  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
      // convection: x -> rows, y -> cols (for example, see Polar2Nodal)
      dst(i, j) = f(x[j], x[i]) * std::sqrt(w[i] * w[j]);
    }
  }
}

}  // end namespace boltzmann

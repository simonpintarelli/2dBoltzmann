#pragma once

#include <Eigen/Dense>

#include "spectral/basis/spectral_basis_factory_ks.hpp"
#include "spectral/basis/toolbox/spectral_basis.hpp"
#include "spectral/p2n_factory.hpp"

namespace boltzmann {

template <typename DERIVED, typename FUNC>
void
to_polar(Eigen::DenseBase<DERIVED>& dst,
         const FUNC& cart_fun,
         const SpectralBasisFactoryKS::basis_type basis)
{
  using basis_t = SpectralBasisFactoryKS::basis_type;

  int K = spectral::get_K(basis);
  const auto& P2N = P2NFactory<>::GetInstance(basis);

  dst.derived().resize(basis.size());

  QHermiteW quad(1., K);
  Eigen::MatrixXd Nd(K, K);
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
      double xi = quad.pts(i);
      double wi = quad.wts(i);
      double xj = quad.pts(j);
      double wj = quad.wts(j);
      Nd(i,j) = cart_fun(xi, xj) * std::sqrt(wi * wj);
    }
  }
  p2n.to_polar(dst, Nd);

}



}  // namespace boltzmann

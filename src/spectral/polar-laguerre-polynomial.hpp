#pragma once

#include <Eigen/Dense>
#include <type_traits>
#include <vector>

#include "ft_eval.hpp"
#include "pl_radial_eval.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"

namespace boltzmann {

template <typename NUMERIC>
class PolarLaguerreEvaluator
{
 public:
  typedef NUMERIC numeric_t;
  typedef SpectralBasisFactoryKS::basis_type basis_t;

 public:
  PolarLaguerreEvaluator(const basis_t& basis)
      : L_(basis)
  { /* empty */
  }

  template <typename D>
  void init(const Eigen::DenseBase<D>& phi, const Eigen::DenseBase<D>& r);

 private:
  enum class storage
  {
    rowMajor,
    colMajor
  };

 private:
  LaguerreNKS<numeric_t> L_;
  FT<NUMERIC> A_;
  const basis_t& basis_;
  enum storage storage_;
};

template <typename NUMERIC>
template <typename D>
PolarLaguerreEvaluator<NUMERIC>(const Eigen::DenseBase<D>& phi, const Eigen::DenseBase<D>& r)
{
  if (D::IsRowMajor)
    storage_ = storage::rowMajor;
  else if (D::IsColMajor)
    storage_ = storage::colMajor;

  int n = phi.rows();
  int m = phi.cols();
}

}  // boltzmann

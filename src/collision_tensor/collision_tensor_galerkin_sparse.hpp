#pragma once

#include <Eigen/Dense>
#include <boost/assert.hpp>
#include "collision_tensor_galerkin.hpp"

namespace boltzmann {

class CollisionTensorGalerkinSparse : public CollisionTensorGalerkin
{
 public:
  CollisionTensorGalerkinSparse(const basis_t& basis, int bufsize)
      : CollisionTensorGalerkin(basis)
      , buffer_(basis.size(), bufsize)
  {
    /* empty */
  }

  template <typename DERIVED1, typename DERIVED2>
  void apply(Eigen::DenseBase<DERIVED1>& out, const Eigen::DenseBase<DERIVED2>& in);

 public:
  mutable Eigen::MatrixXd buffer_;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


template <typename DERIVED1, typename DERIVED2>
void
CollisionTensorGalerkinSparse::apply(Eigen::DenseBase<DERIVED1>& out,
                                     const Eigen::DenseBase<DERIVED2>& in)
{
  for (unsigned int i = 0; i < slices_.size(); ++i) {
    buffer_ = slices_[i] * in.derived().matrix();
    for (int k = 0; k < in.cols(); ++k) {
      out(i, k) = buffer_.col(k).dot(in.col(k).matrix());
    }
  }
}


}  // namespace boltzmann

#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "spectral/basis/spectral_basis_factory_ks.hpp"

namespace boltzmann {

class CollisionTensorGalerkinBase
{
 protected:
  typedef Eigen::SparseMatrix<double> sparse_matrix_t;
  typedef std::shared_ptr<sparse_matrix_t> ptr_t;
  typedef Eigen::SparseLU<sparse_matrix_t> lu_t;
  typedef typename SpectralBasisFactoryKS::basis_type basis_t;
  typedef Eigen::MatrixXd matrix_t;
  typedef Eigen::Matrix4d m4_t;
  typedef Eigen::DiagonalMatrix<double, -1> diag_t;
  typedef Eigen::VectorXd vec_t;

 public:
  CollisionTensorGalerkinBase(const basis_t& basis);

  int get_N() const { return N_; }
  int get_K() const { return K_; }
  const basis_t& get_basis() const { return basis_; }

  void project(double* out, const double* in) const;

  template <typename DERIVED1, typename DERIVED2>
  void project(Eigen::DenseBase<DERIVED1>& out, const Eigen::DenseBase<DERIVED2>& in) const;

  template <typename DERIVED1, typename DERIVED2>
  void project_lambda(Eigen::DenseBase<DERIVED1>& out,
                      const Eigen::DenseBase<DERIVED2>& lambda) const;

  template <typename DERIVED1, typename DERIVED2>
  void get_lambda(Eigen::DenseBase<DERIVED1>& lambda, const Eigen::DenseBase<DERIVED2>& in) const;

 protected:
  const basis_t basis_;
  /// basis size
  const int N_;
  /// max polynomial degree
  const int K_;
  /// buffer
  mutable Eigen::VectorXd buf_;
  /// tensor entries
  matrix_t Ht_;
  m4_t HtHinv_;
  diag_t Sinv_;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <typename DERIVED1, typename DERIVED2>
void
CollisionTensorGalerkinBase::project(Eigen::DenseBase<DERIVED1>& out,
                                     const Eigen::DenseBase<DERIVED2>& in) const
{
  BOOST_ASSERT(out.cols() == in.cols());
  BOOST_ASSERT(out.rows() == in.rows());

  Eigen::MatrixXd lambda = HtHinv_ * Ht_ * (out.derived() - in.derived()).matrix();
  out.derived().matrix() -= Sinv_ * Ht_.transpose() * lambda;
}


template <typename DERIVED1, typename DERIVED2>
void
CollisionTensorGalerkinBase::project_lambda(Eigen::DenseBase<DERIVED1>& out,
                                            const Eigen::DenseBase<DERIVED2>& lambda) const
{
  out.derived().matrix() -= Sinv_ * Ht_.transpose() * lambda.derived().matrix();
}


template <typename DERIVED1, typename DERIVED2>
void
CollisionTensorGalerkinBase::get_lambda(Eigen::DenseBase<DERIVED1>& lambda,
                                        const Eigen::DenseBase<DERIVED2>& in) const
{
  lambda.derived().matrix() = HtHinv_ * Ht_ * in.derived().matrix();
}


}  // namespace boltzmann

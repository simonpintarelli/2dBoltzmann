#pragma once

#include <Eigen/Dense>

#include "quadrature/qmaxwell.hpp"
#include "spectral/lagrange_polynomial.hpp"


namespace boltzmann {
namespace impl_lsq {

class LSQ_BC_Base
{
 public:
  LSQ_BC_Base(int K);

  virtual void apply(Eigen::DenseBase<DERIVED>& dst, const Eigen::DenseBase<DERIVED>& src) const;

 protected:
  Eigen::MatrixXd L_;
};

LSQ_BC_Base::LSQ_BC_Base(int K)
{
  L_.resize(K, K);

  QMaxwell y_quad(1.0, K);
  QHermiteW qherm(1.0, K);

  Eigen::VectorXd y(K);
  for (int i = 0; i < K; ++i) {
    y(i) = -y_quad.pts(i);
  }

  lagrange_poly_simple lagpoly(qherm.pts().data(), qherm.pts().size());

  for (int i2 = 0; i2 < K; ++i2) {
    for (int j = 0; j < K; ++j) {
      double val = 0;
      for (int q = 0; q < K; ++q) {
        val += lagpoly.eval(j, y(q)) * lagpoly.eval(i2, y(q)) * y_quad.wts(q);
      }
      L_(i2, j) = val;
    }
  }
}

void
LSQ_BC_Base::apply(Eigen::DenseBase& dst, const Eigen::DenseBase& src) const
{
  dst = L_ * src;
}

}  // impl_lsq
}  // boltzmann

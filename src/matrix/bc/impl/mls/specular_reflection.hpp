#pragma once

#include <Eigen/Dense>
#include <vector>

#include "../flux_worker.hpp"

namespace boltzmann {
namespace impl_mls {

/**
 * @brief SpecularReflection
 *
 * velocity part (in Lagrange basis)
 *
 * upper hemisphere: outflow
 * lower hemisphere: inflow
 *
 */
class SpecularReflection : public boltzmann::impl::flux_worker
{
  using boltzmann::impl::flux_worker::vec_t;
  using boltzmann::impl::flux_worker::mat_t;

 public:
  SpecularReflection(const vec_t& hermite_weights, const vec_t& hermite_nodes)
      : w_(hermite_weights)
      , x_(hermite_nodes)
  {
    /* empty */
  }

  /**
   *
   *
   * @param out Ordering: out(i,j) = f(x_i, y_j)
   * \remark{Also see documentation of @ref H2N.}
   * @param in
   */
  virtual void apply(mat_t& out, const mat_t& in, const dealii::Point<2>& dummy) const;

 private:
  const vec_t w_;
  const vec_t x_;
};

void
SpecularReflection::apply(mat_t& out, const mat_t& in, const dealii::Point<2>& dummy) const
{
  AssertDimension(in.rows(), out.rows());
  AssertDimension(in.cols(), out.cols());
  AssertDimension(in.cols(), w_.size());

  int N = w_.size();

  int nhalf = N / 2;

  // reflection
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {
      if (i < nhalf)
        out(i, j) = in(N - 1 - i, j);
      else
        out(i, j) = in(i, j);
    }
  }

  // multiply with weights
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {
      out(i, j) *= x_[i];
    }
  }
}

}  // end namespace impl
}  // end namespace boltzmann

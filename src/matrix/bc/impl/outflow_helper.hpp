#pragma once

#include <Eigen/Dense>
#include <vector>

// quadrature rules
#include "quadrature/qhermite.hpp"
#include "quadrature/qmaxwell.hpp"
#include "spectral/lagrange_polynomial.hpp"


namespace boltzmann {
namespace impl {

/**
 * @brief stores integrals over Lagrange polynomials
 *        on the Hermite Quadrature nodes
 */
class outflow_helper
{
 private:
  typedef Eigen::VectorXd vec_t;

 public:
  outflow_helper(int K);

  /**
   *
   *
   * @param hw  Hermite weights
   * @param hx  Hermite nodes
   *
   * Note: both weights and nodes are wrt weight \f$ e^{-x^2} \f$
   */
  outflow_helper(const vec_t& hw, const vec_t& hx);

  /**
   * @brief returns \f$ \int_{\mathbb{R}^+} y l_i(y) e^{-y^2/2} dy\f$
   *
   * @param i the i-th lagrange polynomial
   *
   * @return
   */
  inline double get_y(unsigned int i) const
  {
    assert(i < ly_.size());
    return ly_[i];
  }

  /**
   * @brief returns \f$ \int_{\mathbb{R}} l_i(x) e^{-x^2/2} dx\f$
   *
   * @param i
   *
   * @return
   */
  inline double get_x(unsigned int i) const
  {
    assert(i < lx_.size());
    return lx_[i];
  }

  inline const vec_t& lx() const { return lx_; }

  inline const vec_t& ly() const { return ly_; }

  template <typename DERIVED>
  inline double compute(const Eigen::DenseBase<DERIVED>& src) const;

 private:
  vec_t ly_;
  vec_t lx_;
};

template <typename DERIVED>
inline double
outflow_helper::compute(const Eigen::DenseBase<DERIVED>& src) const
{
  assert(src.rows() == ly_.size());
  assert(src.cols() == lx_.size());

  int K = ly_.size();

  // compute outflow
  double rho_p = 0;
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
      rho_p += src(i, j) * get_x(j) * get_y(i);
    }
  }
  return rho_p;
}

}  // end namespace impl
}  // end namespace boltzmann

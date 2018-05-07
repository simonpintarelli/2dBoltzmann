#pragma once

#include "maxwell_quadrature.hpp"

namespace boltzmann {

class QMaxwellW : public Quadrature<1>
{
 private:
  typedef Quadrature<1> base_t;

 public:
  QMaxwellW(double alpha, int N)
      : QMaxwellW(alpha, N, 256)
  {
  }

  QMaxwellW() { /* default constructor */}

  /**
   * @brief maxwell quadrature
   *
   * Provides a quadrature rule for weight \f$ r e^{-alpha*r^2} \f$ on domain \f$ [0, \infty)
   * \f$. This is a wrapper for @see MaxwellQuadrature.
   *
   *
   * @param alpha factor in exp. weight
   * @param N number of quadrature points
   * @param ndigits multiprecision for Golub-Welsch
   *
   * @return
   */
  QMaxwellW(double alpha, int N, int ndigits);

 private:
  using base_t::pts_;
  using base_t::wts_;
};

inline QMaxwellW::QMaxwellW(double alpha, int N, int ndigits)
    : base_t(N)
{
  MaxwellQuadrature qbase(N, ndigits);

  for (int i = 0; i < N; ++i) {
    pts_[i] = qbase.pts(i) / std::sqrt(alpha);
    wts_[i] = (qbase.wts(i) * std::exp(alpha*pts_[i] * pts_[i]) )/ alpha;
  }
}

}  // boltzmann

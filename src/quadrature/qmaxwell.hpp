#pragma once

#include "maxwell_quadrature.hpp"


namespace boltzmann {

class QMaxwell : public Quadrature<1>
{
 private:
  typedef Quadrature<1> base_t;

 public:
  QMaxwell(double alpha, int N)
      : QMaxwell(alpha, N, 256)
  {
    /* empty */
  }

  QMaxwell() { /* default constructor */}

  /**
   * @brief Maxwell quadrature
   *
   * Provides a quadrature rule for weight \f$ r e^{-alpha*r^2} \f$ on domain \f$ [0, \infty)
   * \f$. This is a wrapper for \ref MaxwellQuadrature.
   *
   *
   * @param alpha factor in exp. weight
   * @param N number of quadrature points
   * @param ndigits multiprecision for Golub-Welsch
   *
   * @return
   */
  QMaxwell(double alpha, int N, int ndigits);

 private:
  using base_t::pts_;
  using base_t::wts_;
};

inline QMaxwell::QMaxwell(double alpha, int N, int ndigits)
    : base_t(N)
{
  MaxwellQuadrature qbase(N, ndigits);

  for (int i = 0; i < N; ++i) {
    pts_[i] = qbase.pts(i) / std::sqrt(alpha);
    wts_[i] = qbase.wts(i) / alpha;
  }
}

}  // boltzmann

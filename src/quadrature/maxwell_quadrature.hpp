#pragma once

#include "quadrature/quadrature_base.hpp"

namespace boltzmann {

/**
 * @brief Gauss-Quadrature on \f$[0,inf)\f$ with weight \f$ x^p \, exp(-x*x) \f$
 *
 * Reference:
 *  B. Shizgal, A Gaussian quadrature procedure for use in the solution of the
 *  Boltzmann equation and related problems https://doi.org/10.1016/0021-9991(81)90099-1
 *
 */
class MaxwellQuadrature : public Quadrature<1>
{
 private:
  typedef Quadrature<1> base_t;

 public:
  /**
   * @param N_ number of points
   * @param ndigits run Golub-Welsch with ndigits accuracy
   * @param p_ exponent
   */
  MaxwellQuadrature(int N_, int ndigits, int p_ = 1)
      : base_t(N_)
      , N(N_)
      , p(p_)
  {
    init(ndigits);
  }

  /**
   * @brief MaxwellQuadrature(N, ndigits=256, p_=1)
   *
   * @param N number of points
   */
  MaxwellQuadrature(int N)
      : MaxwellQuadrature(N, 256, 1)
  {
    /* empty */
  }

 protected:
  void init(int ndigits);

 private:
  using base_t::pts_;
  using base_t::wts_;
  int N;
  int p;
};

}  // boltzmann

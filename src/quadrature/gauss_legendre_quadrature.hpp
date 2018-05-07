#pragma once

#include "quadrature/quadrature_base.hpp"

namespace boltzmann {

/**
 * @brief Gauss-Quadrature on [0,inf) with weight x exp(-x*x)
 *        uses Golub-Welsch and a multiprecision eigenvalue solver
 */
class GaussLegendreQuadrature : public Quadrature<1>
{
 private:
  typedef Quadrature<1> base_t;

 public:
  /**
   *
   * @param N_ number of quadrature nodes
   * @param ndigits number of digits in multiprecision Golub-Welsch
   *
   * @return
   */
  GaussLegendreQuadrature(int N_, int ndigits)
      : base_t(N_)
      , N(N_)
  {
    init(ndigits);
  }

  GaussLegendreQuadrature(int N)
      : GaussLegendreQuadrature(N, 256)
  { /* empty */ }

 protected:
  void init(int ndigits);

 private:
  using base_t::pts_;
  using base_t::wts_;
  int N;
};

}  // namespace boltzmann

#pragma once

#include "quadrature/quadrature_base.hpp"

namespace boltzmann {

/**
 * @brief Gauss-Quadrature on [0,inf) with weight exp(-x*x)
 *        uses Golub-Welsch and a multiprecision eigenvalue solver
 */
class GaussHermiteQuadrature : public Quadrature<1>
{
 private:
  typedef Quadrature<1> base_t;

 public:

  /**
   * @param N number of quadrature nodes
   * @param ndigits number of digits used in multiprecision Golub-Welsch algorithm
   */
  GaussHermiteQuadrature(int N_, int ndigits)
      : base_t(N_)
      , N(N_)
  {
    init(ndigits);
  }

  GaussHermiteQuadrature(int N)
      : GaussHermiteQuadrature(N, 256)
  {
  }

 protected:
  void init(int ndigits);

 private:
  using base_t::pts_;
  using base_t::wts_;
  int N;
};

}  // boltzmann

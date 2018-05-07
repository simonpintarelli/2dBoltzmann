#pragma once

#include <cmath>

namespace boltzmann {

class L2Weight
{
 public:
  L2Weight(double beta_)
      : beta(beta_)
      , alpha(1 - 2 / beta)
  { /* empty */
  }

  double evaluate(double r) const;
  static double evaluate(double r, double alpha);
  double exponent() const { return alpha; }

 private:
  double beta;
  double alpha;
};

// ------------------------------------------------------------
inline double
L2Weight::evaluate(double r) const
{
  return std::exp(-alpha * r * r);
}

// ------------------------------------------------------------
inline double
L2Weight::evaluate(double r, double g)
{
  return std::exp(-g * r * r);
}

}  // end namespace boltzmann

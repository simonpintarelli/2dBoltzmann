#pragma once

#include "base/numbers.hpp"
#include "quadrature/quadrature_base.hpp"

namespace boltzmann {

/**
 * @brief Midpoint rule on [0, 2 Pi]
 */
class QMidpoint : public Quadrature<1>
{
 public:
  typedef Quadrature<1> base_type;
  QMidpoint(int npts);
};
}

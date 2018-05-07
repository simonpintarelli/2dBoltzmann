#pragma once

#include "base/numbers.hpp"
#include "enum/enum.hpp"

namespace boltzmann {
template <enum COLLISION_KERNEL>
class CollisionKernel
{
};

template <>
class CollisionKernel<MAXWELLIAN>
{
 public:
  double kernel() const { return 1. / (2 * numbers::PI); }

  double operator()(double dist, double theta) const { return 1. / (2 * numbers::PI); }
};
}

#include "velocity_radial_integrator.hpp"
#include <cmath>
#include <fstream>
#include <iostream>

namespace boltzmann {
// ----------------------------------------------------------------------
VelocityRadialIntegrator::iterator VelocityRadialIntegrator::begin_s0() const
{
  return map0_.cbegin();
}

// ----------------------------------------------------------------------
VelocityRadialIntegrator::iterator VelocityRadialIntegrator::end_s0() const
{
  return map0_.cend();
}

// ----------------------------------------------------------------------
VelocityRadialIntegrator::iterator VelocityRadialIntegrator::begin_t1() const
{
  return map1_.cbegin();
}

// ----------------------------------------------------------------------
VelocityRadialIntegrator::iterator VelocityRadialIntegrator::end_t1() const
{
  return map1_.cend();
}

// ----------------------------------------------------------------------
VelocityRadialIntegrator::iterator VelocityRadialIntegrator::begin_t2() const
{
  return map2_.cbegin();
}

// ----------------------------------------------------------------------
VelocityRadialIntegrator::iterator VelocityRadialIntegrator::end_t2() const { return map2_.cend(); }

}  // end namespace boltzmann

#include "velocity_angular_integrator.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <quadrature/trig_int.hpp>

using namespace std;

namespace boltzmann {

// -----------------------------------------------------------------------------
VelocityAngularIntegrator<2>::map_S0_t::const_iterator VelocityAngularIntegrator<2>::begin_s0()
    const
{
  return ms0.begin();
}

// -----------------------------------------------------------------------------
VelocityAngularIntegrator<2>::map_S0_t::const_iterator VelocityAngularIntegrator<2>::end_s0() const
{
  return ms0.end();
}

// -----------------------------------------------------------------------------
VelocityAngularIntegrator<2>::map_T1_t::const_iterator VelocityAngularIntegrator<2>::begin_t1()
    const
{
  return mt1.begin();
}

// -----------------------------------------------------------------------------
VelocityAngularIntegrator<2>::map_T1_t::const_iterator VelocityAngularIntegrator<2>::end_t1() const
{
  return mt1.end();
}

// -----------------------------------------------------------------------------
VelocityAngularIntegrator<2>::map_T2_t::const_iterator VelocityAngularIntegrator<2>::begin_t2()
    const
{
  return mt2.begin();
}

// -----------------------------------------------------------------------------
VelocityAngularIntegrator<2>::map_T2_t::const_iterator VelocityAngularIntegrator<2>::end_t2() const
{
  return mt2.end();
}

}  // end namespace boltzmann

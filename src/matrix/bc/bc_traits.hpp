#pragma once

#include <deal.II/dofs/dof_handler.h>
#include "enum/enum.hpp"
#include "impl/mls/diffusive_reflection_x.hpp"
#include "impl/mls/diffusive_reflection.hpp"
#include "impl/mls/inflow.hpp"
#include "impl/mls/specular_reflection.hpp"


namespace boltzmann {

template <enum METHOD>
class bc_traits
{
};

template <>
class bc_traits<METHOD::MODLEASTSQUARES>
{
 public:
  typedef impl_mls::SpecularReflection SpecularReflection;
  typedef impl_mls::DiffusiveReflection DiffusiveReflection;
  typedef impl_mls::Inflow Inflow;
  typedef impl_mls::DiffusiveReflectionX DiffusiveReflectionX;
};

template <>
class bc_traits<METHOD::LEASTSQUARES>
{
  // todo
};

}  // boltzmann

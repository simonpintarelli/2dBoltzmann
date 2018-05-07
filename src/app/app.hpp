#pragma once

#include "app_traits.hpp"
#include "enum/enum.hpp"


namespace boltzmann {

template <int DIM, enum BC_Type B = BC_Type::REGULAR>
struct App
{
  const static int dimX = DIM;
  const static int dimv = DIM;
  static const std::string info;
  const static enum BC_Type bc_type = B;
};

template <int DIM, enum BC_Type B>
const std::string App<DIM, B>::info = "APP-INFO";

/// \cond HIDDEN_SYMBOLS
template <int dim>
struct VOID
{
  static const std::string info;
};

template <int dim>
const std::string VOID<dim>::info = "Not defined";


}  // end namespace boltzmann

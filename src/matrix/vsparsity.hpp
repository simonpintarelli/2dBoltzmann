#pragma once

#include "enum/enum.hpp"
#include "matrix/sparsity_pattern/helpers.hpp"

#include <deal.II/lac/dynamic_sparsity_pattern.h>

namespace boltzmann {

template <enum METHOD>
struct vsparsity
{
};

template <>
struct vsparsity<METHOD::MODLEASTSQUARES>
{
  template <typename VELOCITY_VARFORM>
  static void make(dealii::SparsityPattern& lhs_sparsity,
                   dealii::SparsityPattern& rhs_sparsity,
                   int N,
                   const VELOCITY_VARFORM& var_form);
};

template <typename VELOCITY_VARFORM>
void
vsparsity<METHOD::MODLEASTSQUARES>::make(dealii::SparsityPattern& lhs_vsparsity,
                                         dealii::SparsityPattern& rhs_vsparsity,
                                         int N,
                                         const VELOCITY_VARFORM& var_form)
{
  dealii::DynamicSparsityPattern csp(N, N);
  sparsity_helper::add_to_csp(csp, var_form.get_s0());
  sparsity_helper::add_to_csp(csp, var_form.get_t2());
  lhs_vsparsity.copy_from(csp);
  lhs_vsparsity.compress();

  dealii::DynamicSparsityPattern csp2(N, N);
  sparsity_helper::add_to_csp(csp2, var_form.get_s0());
  sparsity_helper::add_to_csp(csp2, var_form.get_t1());
  rhs_vsparsity.copy_from(csp2);
  rhs_vsparsity.compress();
}

}  // boltzmann

#pragma once

#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

namespace boltzmann {

class SparsityPatternBase
{
 protected:
  typedef dealii::TrilinosWrappers::SparsityPattern sparsity_pattern_t;
  typedef dealii::SparsityPattern sparsity_t;

 public:
  virtual const sparsity_pattern_t& get_sparsity() const = 0;
  virtual const dealii::SparsityPattern& get_L_sparsity() const = 0;
};
}  // namespace boltzmann

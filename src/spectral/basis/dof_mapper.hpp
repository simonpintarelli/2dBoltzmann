#pragma once

namespace boltzmann {

/**
 * @brief Implementation for compatibility with DoFMapperPeriodic
 */
class DoFMapper
{
 private:
  typedef unsigned int index_t;

 public:
  DoFMapper(index_t L)
      : L_(L)
  { /* empty */ }

  inline index_t operator[](const index_t unrestriced_idx) const { return unrestriced_idx; }

  index_t n_dofs() const { return L_; }

 private:
  index_t L_;
};

}  // end namespace boltzmann

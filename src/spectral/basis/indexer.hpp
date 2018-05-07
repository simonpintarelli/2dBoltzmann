#pragma once

#include "dof_mapper.hpp"
#include <deal.II/dofs/dof_handler.h>

namespace boltzmann {
template <typename DOF_MAPPER = DoFMapper>
class Indexer
{
 private:
  typedef unsigned int index_t;

 public:
  typedef DOF_MAPPER dof_mapper_t;

 public:
  Indexer(const dof_mapper_t dof_mapper, unsigned int nphys_dofs, unsigned int nvelo_dofs)
      : dof_mapper_(dof_mapper)
      , n_phys_dofs(nphys_dofs)
      , n_velo_dofs(nvelo_dofs)
  { /* empty */
  }

  Indexer()
      : Indexer(0, 0)
  { /* empty */
  }

  Indexer(unsigned int nphys_dofs, unsigned int nvelo_dofs)
      : Indexer(dof_mapper_t(nphys_dofs), nphys_dofs, nvelo_dofs)
  { /* empty */
  }

  /**
   * @brief convert full *phys* grid index ix and velocity index j
   *        to global matrix index
   *
   * @param ix
   * @param j
   *
   * @return
   */
  index_t to_global(int ix, int j) const;

  /**
   * @brief convert full *phys* grid index ix to *restricted* index
   *
   * @param ix
   *
   * @return
   */
  index_t to_restricted(int ix) const;

  index_t n_dofs() const;
  index_t N() const { return n_velo_dofs; }
  index_t L() const { return n_phys_dofs; }

 private:
  dof_mapper_t dof_mapper_;
  int n_phys_dofs;
  int n_velo_dofs;
};

// ----------------------------------------------------------------------
template <typename DOF_MAPPER>
inline typename Indexer<DOF_MAPPER>::index_t
Indexer<DOF_MAPPER>::to_global(int ix, int j) const
{
  return dof_mapper_[ix] * n_velo_dofs + j;
}

// ----------------------------------------------------------------------
template <typename DOF_MAPPER>
inline typename Indexer<DOF_MAPPER>::index_t
Indexer<DOF_MAPPER>::to_restricted(int ix) const
{
  return dof_mapper_[ix];
}

// ----------------------------------------------------------------------
template <typename DOF_MAPPER>
unsigned int
Indexer<DOF_MAPPER>::n_dofs() const
{
  return dof_mapper_.n_dofs() * n_velo_dofs;
}

}  // end namespace boltzmann

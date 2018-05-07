#pragma once

// own includes -----------------------------------------------------------
#include "dof_mapper_periodic.hpp"

// system includes --------------------------------------------------------
#include <ostream>
#include <vector>
// debug
#include <iostream>
#include <stdexcept>

// deal.II includes -------------------------------------------------------
#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/utilities.h>

namespace boltzmann {

/**
 *
 * This class lives on the *physical* grid
 *
 * @param dh
 * @param nprocs
 */
template <typename Mapper_T = DoFMapperPeriodic>
class DoFMapperPeriodicDistributed
{
 public:
  typedef unsigned int index_t;
  typedef dealii::IndexSet index_set_t;

 public:
  void init(const dealii::DoFHandler<2>& dh, const int nprocs);

  /**
   * @brief Returns locally relevant DoFs (restricted enumeration)
   *        on subdomain i
   *
   * @param i Processor id
   *
   * @return
   */
  const index_set_t& relevant_dofs(int i) const;

  /**
   * @brief Returns locally owned DoFs (restriced enumeration)
   *
   * @param i Processor id
   *
   * @return
   */
  const index_set_t& owned_dofs(int i) const;

  /**
   * @brief Maps physical grid (full) DoF to periodic enumeration
   *
   * @param unrestriced_idx
   *
   * @return
   */
  index_t operator[](const index_t unrestriced_idx) const;

  /**
   * @brief Returns #DoFs on the *physical* periodic grid
   *
   *
   * @return
   */
  index_t n_dofs() const;

  void print_permutation(std::ostream& out)
  {
    for (unsigned int i = 0; i < permutation_.size(); ++i) {
      out << permutation_[i] << std::endl;
    }
  }

 private:
  Mapper_T periodic_mapper;
  /// permutation of dofs on restricted domain
  /// used as a prox to DoFMapperPeriodic
  std::vector<unsigned int> permutation_;
  //@{
  /// IndexSets on restricted grid for each subdomain
  std::vector<index_set_t> locally_relevant_dofs_;
  std::vector<index_set_t> locally_owned_dofs_;
  //@}
};

// ----------------------------------------------------------------------
template <typename Mapper_T>
const typename DoFMapperPeriodicDistributed<Mapper_T>::index_set_t&
DoFMapperPeriodicDistributed<Mapper_T>::relevant_dofs(int i) const
{
  return locally_relevant_dofs_[i];
}

// ----------------------------------------------------------------------
template <typename Mapper_T>
const typename DoFMapperPeriodicDistributed<Mapper_T>::index_set_t&
DoFMapperPeriodicDistributed<Mapper_T>::owned_dofs(int i) const
{
  return locally_owned_dofs_[i];
}

// ----------------------------------------------------------------------
template <typename Mapper_T>
typename DoFMapperPeriodicDistributed<Mapper_T>::index_t DoFMapperPeriodicDistributed<Mapper_T>::
operator[](const index_t unrestriced_idx) const
{
  return permutation_[periodic_mapper[unrestriced_idx]];
}

// ----------------------------------------------------------------------
template <typename Mapper_T>
typename DoFMapperPeriodicDistributed<Mapper_T>::index_t
DoFMapperPeriodicDistributed<Mapper_T>::n_dofs() const
{
  return periodic_mapper.size();
}

// ----------------------------------------------------------------------
template <typename Mapper_T>
void
DoFMapperPeriodicDistributed<Mapper_T>::init(const dealii::DoFHandler<2>& dh, const int nprocs)
{
  periodic_mapper.init(dh);

  std::vector<int> subd_assoc_restricted;

  const unsigned int n_full_dofs = dh.n_dofs();
  const unsigned int n_rest_dofs = periodic_mapper.size();
  std::vector<unsigned int> subd_assoc(n_full_dofs);
  dealii::DoFTools::get_subdomain_association(dh, subd_assoc);
  subd_assoc_restricted.resize(periodic_mapper.size(), -1);
  //
  const auto& dof_map = periodic_mapper.dof_map();

  for (unsigned int i = 0; i < n_full_dofs; ++i) {
    auto it = dof_map.find(i);
    if (it != dof_map.end()) {
      // this is a slave dof
      // use subdomain of master
      const unsigned int master_idx = it->second;
      subd_assoc_restricted[periodic_mapper[i]] = subd_assoc[master_idx];
    } else {
      // this is master dof
      subd_assoc_restricted[periodic_mapper[i]] = subd_assoc[i];
    }
  }
  // Renumbering (make the restriced DoFs contiguous across subdomains)
  // write result to permutation_
  permutation_.resize(n_rest_dofs);
  std::vector<unsigned int> offsets(nprocs, 0);
  for (unsigned int i = 0; i < n_rest_dofs; ++i) {
    unsigned int color = subd_assoc_restricted[i];
    assert(color >=0);
    assert(color < nprocs);
    permutation_[i] = offsets[color]++;
  }
  unsigned int offset = 0;
  for (int i = 0; i < nprocs; ++i) {
    unsigned int tmp = offsets[i];
    offsets[i] = offset;
    offset += tmp;
  }
  for (unsigned int i = 0; i < n_rest_dofs; ++i) {
    unsigned int color = subd_assoc_restricted[i];
    permutation_[i] += offsets[color];
  }

  locally_owned_dofs_.resize(nprocs);
  locally_relevant_dofs_.resize(nprocs);
  // build IndexSets of locally relevant and locally owned dofs
  for (int dom = 0; dom < nprocs; ++dom) {
    locally_owned_dofs_[dom].set_size(n_rest_dofs);
    locally_relevant_dofs_[dom].set_size(n_rest_dofs);
    auto index_set = dealii::DoFTools::dof_indices_with_subdomain_association(dh, dom);
    std::vector<bool> locally_relevant_phys_indices(n_full_dofs);
    index_set.fill_binary_vector(locally_relevant_phys_indices);
    for (unsigned int i = 0; i < n_full_dofs; ++i) {
      if (locally_relevant_phys_indices[i])
        locally_relevant_dofs_[dom].add_index(this->operator[](i));
      if (subd_assoc_restricted[this->operator[](i)] == dom)
        locally_owned_dofs_[dom].add_index(this->operator[](i));
    }
    locally_relevant_dofs_[dom].compress();
    locally_owned_dofs_[dom].compress();
  }

#ifdef DEBUG
  for (int i = 0; i < nprocs; ++i) {
    for (int j = i+1; j < nprocs; ++j) {
      for (auto it = locally_owned_dofs_[i].begin(); it != locally_owned_dofs_[i].end(); ++it) {
        if (locally_owned_dofs_[j].is_element(*it)) {
          throw std::runtime_error("ooops index" + std::to_string(*it) + " is contained in set " + std::to_string(j));
        }
      }
    }
  }
#endif  //DEBUG
}

}  // end namespace boltzmann

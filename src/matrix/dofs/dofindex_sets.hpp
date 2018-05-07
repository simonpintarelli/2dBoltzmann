#pragma once

// own includes ---------------------------------------------------------------
#include "matrix/dofs/dof_helper.hpp"
// deal.II includes -----------------------------------------------------------
#include <deal.II/base/index_set.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_tools.h>
// system includes ------------------------------------------------------------
#include <algorithm>
#include <boost/assert.hpp>


namespace boltzmann {
class DoFIndexSetsBase
{
 protected:
  typedef dealii::IndexSet index_set_t;

 public:
  virtual const index_set_t& locally_owned_dofs(size_t pid) const = 0;
  virtual const index_set_t& locally_relevant_dofs() const = 0;
  virtual const index_set_t& locally_relevant_phys_dofs() const = 0;
  virtual const index_set_t& locally_owned_phys_dofs(size_t i) const = 0;
  virtual const std::vector<unsigned int>& get_subdomain_ids() const = 0;
};

/**
 * @brief manages IndexSets and information about locally owned DoFs
 *        and locally relevant DoFs on D x V
 *        TODO: find a suitable name
 *
 * @param nprocs
 */
class DoFIndexSets : public DoFIndexSetsBase
{
 public:
  DoFIndexSets(int nprocs)
      : index_sets(nprocs)
  { /* empty */
  }

  template <typename DH>
  void init(const DH& dof_handler, size_t n_velo_dofs);

  /**
   *  Owned (x,v)-dofs
   *  @param pid processor id
   */
  virtual const index_set_t& locally_owned_dofs(size_t pid) const
  {
    BOOST_ASSERT(pid < index_sets.size());
    return index_sets[pid];
  }

  /**
   *  Relevant (x,v)-dofs
   *  @param pid processor id
   */
  virtual const index_set_t& locally_relevant_dofs() const
  {
    return locally_relevant_global_indices;
  }

  /**
   *  Relevant x-dofs by calling processor
   *  @param pid processor id
   */
  virtual const index_set_t& locally_relevant_phys_dofs() const { return relevant_x_dofs; }

  /**
   *  Owned x-dofs
   *  @param pid processor id
   */
  virtual const index_set_t& locally_owned_phys_dofs(size_t i) const
  {
    BOOST_ASSERT(i < phys_index_sets.size());
    return phys_index_sets[i];
  }

  /**
   * Subdomain association by DoF
   * @return
   */
  virtual const std::vector<unsigned int>& get_subdomain_ids() const { return subdomain_ids; }

 private:
  /// index_set for global dof on each processor
  std::vector<index_set_t> index_sets;
  /// subdomain_ids: stores subdomain id for each physical dof
  std::vector<unsigned int> subdomain_ids;
  /// locally owned phys dofs
  std::vector<index_set_t> phys_index_sets;
  /// locally relevant indices (phase space domain)
  index_set_t locally_relevant_global_indices;
  /// locally relevant x dofs
  index_set_t relevant_x_dofs;
};

// ------------------------------------------------------------
template <typename DH>
void
DoFIndexSets::init(const DH& dof_handler, size_t n_velo_dofs)
{
  const size_t n_phys_dofs = dof_handler.n_dofs();
  const size_t n_total_dofs = n_velo_dofs * n_phys_dofs;

  unsigned int nprocs = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  unsigned int pid = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  subdomain_ids.resize(dof_handler.n_dofs());
  index_sets.resize(nprocs);
  phys_index_sets.resize(nprocs);
  dealii::DoFTools::get_subdomain_association(dof_handler, subdomain_ids);

  for (unsigned int i = 0; i < nprocs; ++i) {
    // find local dof range
    auto itl = std::lower_bound(subdomain_ids.begin(), subdomain_ids.end(), i);
    auto itu = std::upper_bound(subdomain_ids.begin(), subdomain_ids.end(), i);
    const size_t lb = itl - subdomain_ids.begin();
    const size_t lu = itu - subdomain_ids.begin();
    index_sets[i] = index_set_t(n_total_dofs);
    index_sets[i].add_range(lb * n_velo_dofs, lu * n_velo_dofs);
    // physical dofs
    phys_index_sets[i] = index_set_t(n_phys_dofs);
    phys_index_sets[i].add_range(lb, lu);
  }

  for (unsigned int i = 0; i < index_sets.size(); ++i) {
    index_sets[i].compress();
    phys_index_sets[i].compress();
  }

  // build locally relevant_phys_indices **only for this process**
  relevant_x_dofs = dealii::DoFTools::dof_indices_with_subdomain_association(dof_handler, pid);
  // create locally relevant global indices
  locally_relevant_global_indices = replicate_index_set(relevant_x_dofs, n_velo_dofs);
}

// ----------------------------------------------------------------------------------------------
class DoFIndexSetsPeriodic : public DoFIndexSetsBase
{
 public:
  /**
   *
   * @param owned_dofs    owned physical degrees of freedom only (on periodic grid enumeration)
   * @param relevant_dofs relevant physical degrees of freedom only (on periodic grid enumeration)
   * @param n_velo_dofs   number of velocity degrees of freedom
   * @param mpi_this_proc processor id
   *
   * @return
   */
  DoFIndexSetsPeriodic(const index_set_t& owned_dofs,
                       const index_set_t& relevant_dofs,
                       unsigned int n_velo_dofs,
                       unsigned int mpi_this_proc = 0)
      : owned_phys_dofs(owned_dofs)
      , relevant_phys_dofs(relevant_dofs)
      , this_process(mpi_this_proc)
  {
    total_owned_dofs = replicate_index_set(owned_dofs, n_velo_dofs);
    total_relevant_dofs = replicate_index_set(relevant_dofs, n_velo_dofs);
  }

  virtual const index_set_t& locally_owned_dofs(size_t pid) const
  {
    if (pid != this_process)
      throw std::runtime_error("DoFIndexSetsPeriodic has only local processor data");
    return total_owned_dofs;
  }

  virtual const index_set_t& locally_relevant_dofs() const { return total_relevant_dofs; }

  virtual const index_set_t& locally_relevant_phys_dofs() const { return relevant_phys_dofs; }

  virtual const index_set_t& locally_owned_phys_dofs(size_t pid) const
  {
    if (pid != this_process)
      throw std::runtime_error("DoFIndexSetsPeriodic has only local processor data");
    return owned_phys_dofs;
  }

  virtual const std::vector<unsigned int>& get_subdomain_ids() const
  {
    throw std::runtime_error("Not implemented!");
  }

  index_set_t owned_phys_dofs;
  index_set_t relevant_phys_dofs;
  index_set_t total_owned_dofs;
  index_set_t total_relevant_dofs;
  const unsigned int this_process;
};

}  // end boltzmann

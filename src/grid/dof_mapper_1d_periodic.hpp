#pragma once

// deal.II includes ----------------------------------------
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/constraint_matrix.h>

// system  includes ----------------------------------------
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

namespace boltzmann {

/**
 * @brief Provides mapping between periodic and non periodic indexing on a
 *        given grid in the *physical* domain.
 *
 *        Periodic grid in x-direction
 *
 * @param dh
 */
class DoFMapper1DPeriodicX
{
 private:
  typedef unsigned int index_t;
  typedef std::set<index_t> index_set_t;
  typedef std::map<index_t, index_t> map_t;

 public:
  void init(const dealii::DoFHandler<2>& dh);

  /**
   * @brief full grid -> periodic enumeration
   *
   * @param unrestricted_idx (full grid id)
   *
   * @return restricted grid id
   */
  index_t operator[](const index_t unrestricted_idx) const;

  /**
   * @brief restriced -> full
   *
   * @param restricted_idx
   *
   * @return full_idx
   */
  index_t lookup(const index_t restricted_idx) const;

  const std::vector<index_t>& get_mapping() const { return mapping_; }

  unsigned int size() const { return size_; }

  /**
   *  @brief maps boundary_dof -> master_dof
   *
   * @return std::map<index_t, index_t>
   */
  const map_t& dof_map() const { return dof_map_; }

 private:
  unsigned int size_;
  /// full -> periodic renumbering
  std::vector<index_t> mapping_;
  ///
  std::vector<index_t> inverse_;
  /// maps boundary DoFs to their master DoF
  /// no entry means this DoF is original
  map_t dof_map_;
};

// ------------------------------------------------------------
/**
 * @brief periodic -> non-periodic
 *
 * @param restricted_idx
 *
 * @return
 */
DoFMapper1DPeriodicX::index_t
DoFMapper1DPeriodicX::lookup(const index_t restricted_idx) const
{
  return this->inverse_[restricted_idx];
}

// ------------------------------------------------------------
/**
 * @brief go from non-periodic dof-idx to periodic indexing
 *
 * @param unrestricted_idx  non-periodic dof-idx
 *
 * @return
 */
DoFMapper1DPeriodicX::index_t DoFMapper1DPeriodicX::operator[](const index_t unrestricted_idx) const
{
  return mapping_[unrestricted_idx];
}

// ------------------------------------------------------------
void
DoFMapper1DPeriodicX::init(const dealii::DoFHandler<2>& dh)
{
  dealii::ConstraintMatrix constraints;

  std::map<unsigned int, double> dof_locations;

  typedef dealii::DoFHandler<2> dh_t;

  for (typename dh_t::active_cell_iterator cell = dh.begin_active(); cell != dh.end(); ++cell)
    if (cell->at_boundary() && cell->face(1)->at_boundary()) {
      dof_locations[cell->face(1)->vertex_dof_index(0, 0)] = cell->face(1)->vertex(0)[1];
      dof_locations[cell->face(1)->vertex_dof_index(1, 0)] = cell->face(1)->vertex(1)[1];
    }

  for (typename dh_t::active_cell_iterator cell = dh.begin_active(); cell != dh.end(); ++cell)
    if (cell->at_boundary() && cell->face(0)->at_boundary()) {
      for (unsigned int face_vertex = 0; face_vertex < 2; ++face_vertex) {
        constraints.add_line(cell->face(0)->vertex_dof_index(face_vertex, 0));

        std::map<unsigned int, double>::const_iterator p = dof_locations.begin();
        for (; p != dof_locations.end(); ++p)
          if (std::fabs(p->second - cell->face(0)->vertex(face_vertex)[1]) < 1e-8) {
            constraints.add_entry(cell->face(0)->vertex_dof_index(face_vertex, 0), p->first, 1.0);
            // std::cout << "neighbor found!\n";
            break;
          }
        Assert(p != dof_locations.end(),
               dealii::ExcMessage("No corresponding degree of freedom was found!"));
      }
    }

  //  std::cout << "number of constraints: " << constraints.n_constraints() << std::endl;
  // ---- build permutation vectors ----
  int n_local_dofs = dh.n_dofs();

  dealii::IndexSet constrained_dofs(n_local_dofs);
  // what is get_local_lines doing???
  //  dealii::IndexSet constrained_dofs = constraints.get_local_lines();
  for (int i = 0; i < n_local_dofs; ++i) {
    if (constraints.is_constrained(i)) constrained_dofs.add_index(i);
  }

  // constrained_dofs.print(std::cout);
  //  std::cout << "--- end IndexSet ---\n";
  int n_constraint_dofs = constrained_dofs.n_elements();
  // std::cout << "inverse_.size = "<< n_local_dofs-n_constraint_dofs << std::endl;
  // std::cout << "n_local_dofs = "<< n_local_dofs << std::endl;
  // std::cout << "n_constraint_dofs = "<< n_constraint_dofs << std::endl;
  mapping_.resize(n_local_dofs);
  inverse_.resize(n_local_dofs - n_constraint_dofs);

  unsigned int counter = 0;
  for (int ix_local = 0; ix_local < n_local_dofs; ++ix_local) {
    if (constrained_dofs.is_element(ix_local)) {
      // pass
    } else {
      mapping_[ix_local] = counter++;
    }
  }

  for (int ix_local = 0; ix_local < n_local_dofs; ++ix_local) {
    if (constrained_dofs.is_element(ix_local)) {
      auto entries_ptr = constraints.get_constraint_entries(ix_local);
      mapping_[ix_local] = mapping_[(*entries_ptr)[0].first];
      AssertThrow(entries_ptr->size() == 1, dealii::ExcMessage("Constraint not unique!"));
      AssertThrow((*entries_ptr)[0].second == 1.0, dealii::ExcMessage("Invalid value!"));
      inverse_[mapping_[ix_local]] = ix_local;
      dof_map_[ix_local] = (*entries_ptr)[0].first;
    } else {
      inverse_[mapping_[ix_local]] = ix_local;
    }
  }

  size_ = counter;
}

}  // end boltzmann

#pragma once

// deal.II includes --------------------------------------------------------
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_accessor.h>

// system includes ---------------------------------------------------------
#include <stdexcept>
#include <unordered_map>
#include <vector>

// dealii includes ---------------------------------------------------------
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
// own includes ------------------------------------------------------------
#include "aux/hash_specializations.hpp"
#include "var_form/var_form.hpp"

namespace boltzmann {

class GroupFacesByNormals
{
 public:
  constexpr const static int dimX = 2;
  typedef dealii::DoFCellAccessor<dealii::DoFHandler<2>, false> accessor_t;
  typedef int face_index_t;
  /// FUZZY*normal_vector
  typedef std::tuple<long int, long int> key_t;
  typedef std::tuple<accessor_t, face_index_t> value_t;
  typedef std::vector<value_t> entries_t;
  typedef dealii::Tensor<1, 2> vec2_t;
  typedef std::unordered_map<key_t, entries_t> map_t;
  typedef std::unordered_map<key_t, vec2_t> normals_map_t;

 public:
  template <typename DoFHandler>
  GroupFacesByNormals(const DoFHandler& dof_handler,
                      int mpi_this_process = -1,
                      int boundary_id = -1);

  template <typename ITERATOR>
  void insert(vec2_t& normal, ITERATOR& it, face_index_t faceidx);

  constexpr const static double FUZZY = 1e8;

  const map_t& get_map() const { return map_; }
  const normals_map_t& get_normals() const { return normals_; }

 public:
  map_t map_;
  normals_map_t normals_;
};

// ----------------------------------------------------------------------
template <typename DoFHandler>
GroupFacesByNormals::GroupFacesByNormals(const DoFHandler& dof_handler,
                                         int mpi_this_process,
                                         int boundary_id)
{
  VarForm<2, dealii::QGauss> var_form(dof_handler.get_fe(), 2);
  auto& fe_face_values = var_form.get_fe_face_values();

  bool use_bd_ind = boundary_id == -1 ? false : true;
  bool has_mpi = mpi_this_process != -1 ? true : false;
  const int nfaces = dealii::GeometryInfo<dimX>::faces_per_cell;
  for (auto cell : dof_handler.active_cell_iterators()) {
    if (has_mpi)
      if (cell->subdomain_id() != (unsigned int)mpi_this_process) continue;

    if (cell->at_boundary()) {
      for (int face_idx = 0; face_idx < nfaces; ++face_idx) {
        if (cell->face(face_idx)->at_boundary()) {
          if (use_bd_ind)
            if (cell->face(face_idx)->boundary_id() != boundary_id) continue;
          fe_face_values.reinit(cell, face_idx);
          auto normal = fe_face_values.normal_vector(0);
          this->insert(normal, cell, face_idx);
        }
      }
    }
  }
}

// ----------------------------------------------------------------------
template <typename ITERATOR>
void
GroupFacesByNormals::insert(vec2_t& normal, ITERATOR& it, face_index_t faceidx)
{
  long int nx = FUZZY * normal[0];
  long int ny = FUZZY * normal[1];
  auto key = std::make_tuple(nx, ny);
  auto value = std::make_tuple(*it, faceidx);

  map_[key].emplace_back(value);
  normals_[key] = normal;
}


}  // end namespace boltzmann

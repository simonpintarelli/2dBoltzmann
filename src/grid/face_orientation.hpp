#pragma once

// deal.II includes
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_values.h>

#include <Eigen/Core>

// system includes
#include <map>
#include <vector>


namespace boltzmann {

class FaceOrientation2D
{
 protected:
  static const int dimX = 2;
  typedef dealii::DoFHandler<dimX> dof_handler_t;
  typedef typename dof_handler_t::active_cell_iterator cell_it;
  typedef dealii::types::global_dof_index size_type;

  /// key: <cell iterator, face_idx>
  typedef std::tuple<cell_it, int> key_t;
  typedef struct
  {
    Eigen::Vector2d normal_vector;
    /// DoF (vertex) indices right
    size_type phys_idx_r;
    /// DoF (vertex) indices left
    size_type phys_idx_l;
    double length;
  } face_data_t;

 public:
  typedef std::map<key_t, face_data_t> face_data_map_t;

 public:
  void init(const dof_handler_t& dof_handler);

  const face_data_map_t& get_faces_map() const { return face_data_map_; }

  const std::vector<face_data_t>& get_faces() const { return faces_; }

 protected:
  face_data_map_t face_data_map_;
  std::vector<face_data_t> faces_;
};

void
FaceOrientation2D::init(const dof_handler_t& dof_handler)
{
  // faces -> local fe dofs
  std::array<std::array<size_type, 2>, 4> faces2localdofs;
  faces2localdofs[0] = {0, 2};
  faces2localdofs[1] = {1, 3};
  faces2localdofs[2] = {0, 1};
  faces2localdofs[3] = {2, 3};

  const auto& fe = dof_handler.get_fe();
  dealii::UpdateFlags update_flags = dealii::update_values | dealii::update_JxW_values |
                                     dealii::update_quadrature_points |
                                     dealii::update_normal_vectors;
  dealii::QGauss<1> quad(1);
  dealii::FEFaceValues<dimX, dimX> fe_face_values(fe, quad, update_flags);

  const unsigned int dofs_per_cell = fe_face_values.dofs_per_cell;
  std::vector<size_type> local_dof_indices(dofs_per_cell);

  for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell) {
    cell->get_dof_indices(local_dof_indices);
    for (unsigned int face_idx = 0; face_idx < dealii::GeometryInfo<dimX>::faces_per_cell;
         ++face_idx) {
      if (cell->face(face_idx)->at_boundary()) {
        // *** Face is located at boundary ***
        fe_face_values.reinit(cell, face_idx);

        // relevant FE dofs
        const double nx = fe_face_values.normal_vector(0)[0];
        const double ny = fe_face_values.normal_vector(0)[1];
        face_data_t face_data;
        face_data.normal_vector << nx, ny;

        size_type phys_idx_l = local_dof_indices[faces2localdofs[face_idx][0]];
        size_type phys_idx_r = local_dof_indices[faces2localdofs[face_idx][1]];
        face_data.phys_idx_l = phys_idx_l;
        face_data.phys_idx_r = phys_idx_r;
        face_data.length = cell->face(face_idx)->measure();

        auto key = std::make_tuple(cell, face_idx);
        face_data_map_[key] = face_data;
      }
    }
  }

  for (auto it = face_data_map_.begin(); it != face_data_map_.end(); ++it) {
    faces_.push_back(it->second);
  }
}

}  // end namespace boltzmann

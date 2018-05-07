#pragma once

#include <deal.II/base/quadrature.h>
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_tools.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "grid/grid_tools.hpp"

namespace boltzmann {


template <int DIM>
class GridTransfer
{
 private:
  typedef dealii::types::global_dof_index index_t;
  typedef Eigen::SparseMatrix<double> sparse_matrix_t;
  typedef dealii::DoFHandler<DIM> dh_t;

 public:
  const sparse_matrix_t& get_transfer_matrix() { return T_; }
  void init(const dh_t& dh, const dh_t& dh_src);

 private:
  //  dealii::DoFHandler<DIM> dofhandler_;
  sparse_matrix_t T_;
};

template <int DIM>
void
GridTransfer<DIM>::init(const dh_t& dh, const dh_t& dh_src)
{
  auto v2d = vertex_to_dof_index(dh);
  auto v2d_src = vertex_to_dof_index(dh_src);
  unsigned int N = dh.n_dofs();
  unsigned int n = dh_src.n_dofs();
  T_.resize(N, n);
  const auto& vertices = dh.get_triangulation().get_vertices();
  const auto& vertices_source = dh_src.get_triangulation().get_vertices();
  std::set<index_t> todo_vertices;
  for (index_t i = 0; i < N; ++i) todo_vertices.insert(i);

  const double tol = 1e-9;
  // identifiy same vertices
  for (unsigned int j = 0; j < vertices_source.size(); ++j) {
    for (unsigned int i = 0; i < vertices.size(); ++i) {
      if (vertices_source[j].distance(vertices[i]) < tol) {
        T_.insert(v2d[i], v2d_src[j]) = 1;
        todo_vertices.erase(i);
      }
    }
  }

  typedef dealii::FEValues<DIM> fe_values_t;
  const auto& fe = dh_src.get_fe();

  const int dofs_per_cell = fe.dofs_per_cell;
  dealii::MappingQ1<DIM, DIM> map;
  std::vector<index_t> local_dof_indices(dofs_per_cell);

  // go through all remaining vertices and find
  // their parent cells to find the interpolation coefficients
  for (index_t i : todo_vertices) {
    const auto& p = vertices[i];
    auto cell = dealii::GridTools::find_active_cell_around_point(dh_src, p);
    cell->get_dof_indices(local_dof_indices);
    auto q = map.transform_real_to_unit_cell(cell, p);

    dealii::Quadrature<DIM> quad(q);
    dealii::UpdateFlags update_flags = (dealii::update_values | dealii::update_quadrature_points);
    fe_values_t fval(map, fe, quad, update_flags);

    fval.reinit(cell);
    for (int ix = 0; ix < dofs_per_cell; ++ix) {
      double v = fval.shape_value(ix, 0);
      if (std::abs(v) > 1e-10) T_.insert(v2d[i], local_dof_indices[ix]) = v;
    }
  }
}

}  // end namespace boltzmann

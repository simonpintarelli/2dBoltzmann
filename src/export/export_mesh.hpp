#pragma once

#include <Eigen/Dense>

#include "aux/eigen2hdf.hpp"
#include "grid/grid_tools.hpp"

namespace boltzmann {

/**
 * @brief Write mesh to hdf5
 *
 * @param dh DoF handler
 *
 */
template <typename DOFHANDLER>
void
export_mesh(const DOFHANDLER& dh, const char* fname = "mesh.hdf5")
{
  auto d2v = dof_to_vertex_index(dh);
  auto v2d = vertex_to_dof_index(dh);
  const auto& vertices = dh.get_triangulation().get_vertices();
  unsigned int L = vertices.size();
  const unsigned int DIM = DOFHANDLER::dimension;
  typedef Eigen::Matrix<double, Eigen::Dynamic, DIM> point_array_t;

  point_array_t X(L, DIM);

  // didx: dof index
  for (unsigned int didx = 0; didx < dh.n_dofs(); ++didx) {
    auto point = vertices[d2v[didx]];
    for (unsigned int d = 0; d < DIM; ++d) {
      X(didx, d) = point[d];
    }
  }

  const unsigned int dofs_per_cell = dh.get_fe().dofs_per_cell;
  typedef Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic> cell_array_t;
  cell_array_t M(dh.get_triangulation().n_active_quads(), dofs_per_cell);

  auto& tria = dh.get_triangulation();

  std::vector<unsigned int> ccw = {0, 1, 3, 2};

  unsigned int i = 0;
  for (auto cell : tria.active_cell_iterators()) {
    int counter = 0;
    for (auto j : ccw) {
      M(i, counter++) = v2d[cell->vertex_index(j)];
    }
    i++;
  }

  hid_t fh5 = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  eigen2hdf::save(fh5, "nodes", X);
  eigen2hdf::save(fh5, "cells", M);

  H5Fclose(fh5);
}

}  // boltzmann

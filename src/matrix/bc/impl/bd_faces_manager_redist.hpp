#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>
#include <set>
#include <vector>

#ifdef DEBUG
#include <fstream>
#endif

#include <deal.II/grid/grid_tools.h>


namespace boltzmann {
namespace impl {

/**
 * @brief collect faces on boundary and redistribute
 * them evenly among the processors
 *
 */
class BdFacesManagerDist
{
 private:
  constexpr const static int dimX = 2;
  typedef dealii::DoFCellAccessor<dealii::DoFHandler<dimX>, false> accessor_t;
  typedef int face_index_t;
  typedef std::tuple<accessor_t, face_index_t> face_t;
  typedef std::vector<face_t> vfaces_t;

 protected:
  BdFacesManagerDist() = delete;

 protected:
  template <typename DH, typename SPECTRAL_BASIS, typename INDEXER>
  BdFacesManagerDist(const DH& dh, const SPECTRAL_BASIS& spectral_basis, const INDEXER& indexer);

  const vfaces_t& get_faces_list() const { return my_faces_; }

 private:
  template <typename DH>
  face_t _find_first(const DH& dh) const;

  template <typename DH>
  bool _insert_next(vfaces_t& faces_vector, const DH& dh) const;

 private:
  vfaces_t my_faces_;

 protected:
  dealii::IndexSet relevant_dofs_;
};

// --------------------------------------------------------------------------------
template <typename DH, typename SPECTRAL_BASIS, typename INDEXER>
BdFacesManagerDist::BdFacesManagerDist(const DH& dh,
                                       const SPECTRAL_BASIS& spectral_basis,
                                       const INDEXER& indexer)
    : relevant_dofs_(indexer.n_dofs())
{
  const unsigned int nprocs = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  const unsigned int pid = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  int faces_per_cell = dealii::GeometryInfo<dimX>::faces_per_cell;

  vfaces_t all_faces;

  int dofs_per_face = dh.get_fe().dofs_per_face;
  std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_face);
  unsigned int N = spectral_basis.n_dofs();

  for (auto cell : dh.active_cell_iterators()) {
    if (cell->at_boundary()) {
      for (int face_idx = 0; face_idx < faces_per_cell; ++face_idx) {
        if (cell->face(face_idx)->at_boundary()) {
          // add to work list
          all_faces.push_back(std::make_tuple(*cell, face_idx));
        }
      }
    }
  }

  unsigned int nfaces = all_faces.size();
  // #ifdef DEBUG
  //   std::cout << "\nFound " << nfaces << " faces.\n";
  // #endif

  typedef Eigen::SparseMatrix<double, Eigen::RowMajor> sparse_matrix_t;
  sparse_matrix_t graph(nfaces, nfaces);

  std::vector<dealii::types::global_dof_index> my_dofs(dofs_per_face);

  const auto& tria = dh.get_triangulation();

  for (unsigned int i = 0; i < nfaces; ++i) {
    std::set<unsigned int> adjacent_elements;

    const auto& cell = std::get<0>(all_faces[i]);
    auto face_idx = std::get<1>(all_faces[i]);
    cell.face(face_idx)->get_dof_indices(my_dofs);

    auto find_adjacent_elems = [&](dealii::types::global_dof_index v) {
      auto v_adjacent_cells = dealii::GridTools::find_cells_adjacent_to_vertex(dh, v);

      for (auto& acell : v_adjacent_cells) {
        std::vector<dealii::types::global_dof_index> neighbor_dofs(dofs_per_face);
        if (!acell->at_boundary()) continue;
        for (int j = 0; j < faces_per_cell; ++j) {
          if (!acell->at_boundary(j)) continue;
          acell->face(j)->get_dof_indices(neighbor_dofs);
          if ((std::count(my_dofs.begin(), my_dofs.end(), neighbor_dofs[0]) > 0) ^
              (std::count(my_dofs.begin(), my_dofs.end(), neighbor_dofs[1]) > 0)) {
            face_t neighbor_face(*acell, j);
            auto it = std::find(all_faces.begin(), all_faces.end(), neighbor_face);
            if (it == all_faces.end()) {
              throw std::runtime_error("face not found!");
            }
            unsigned int pos = it - all_faces.begin();
            adjacent_elements.insert(pos);
          }
        }
      }
    };

    auto v0 = cell.face(face_idx)->vertex_index(0);
    find_adjacent_elems(v0);
    auto v1 = cell.face(face_idx)->vertex_index(1);
    find_adjacent_elems(v1);

    assert(adjacent_elements.size() == 2);

    for (auto col : adjacent_elements) {
      graph.insert(i, col) = 1;
    }
  }

  graph.makeCompressed();

  // // find neighbors
  // for (int j = 0; j < faces_per_cell; ++j) {
  //   std::vector<dealii::types::global_dof_index> neighbor_dofs(dofs_per_face);
  //   if (cell.at_boundary(j)) {
  //     // probably there is another adjacent edge on the same element
  //     cell.face(j)->get_dof_indices(neighbor_dofs);
  //     if ( (std::count(my_dofs.begin(), my_dofs.end(), neighbor_dofs[0]) > 0)
  //          ^
  //          (std::count(my_dofs.begin(), my_dofs.end(), neighbor_dofs[1]) > 0)) {
  //       face_t neighbor_face(cell, j);
  //       auto it = std::find(all_faces.begin(), all_faces.end(), neighbor_face);
  //       if (it == all_faces.end()) {
  //         throw std::runtime_error("face not found!");
  //       }
  //       unsigned int pos = it - all_faces.begin();
  //       adjacent_elements.push_back(pos);
  //     }
  //   } else if (!cell.neighbor(j)->at_boundary())
  //     continue;
  //   else {
  //     for (int k = 0; k < faces_per_cell; ++k) {

  //       if (!cell.neighbor(j)->at_boundary(k))
  //         continue;
  //       else {

  //         cell.neighbor(j)->face(k)->get_dof_indices(neighbor_dofs);
  //         // look for common vertex of the two edges..
  //         if ( (std::count(my_dofs.begin(), my_dofs.end(), neighbor_dofs[0]) > 0)
  //              ||
  //              (std::count(my_dofs.begin(), my_dofs.end(), neighbor_dofs[1]) > 0)) {
  //           // this edge is adjacent to the tuple (cell, face_idx) from the outermost loop
  //           // determine its position in all_faces ...
  //           face_t neighbor_face(*cell.neighbor(j), k);
  //           // this is slow, but a hash for DoFCellAccessor or a comparison function is
  //           // missing at the moment
  //           auto it = std::find(all_faces.begin(), all_faces.end(), neighbor_face);
  //           if (it == all_faces.end()) {
  //             throw std::runtime_error("face not found!");
  //           }
  //           unsigned int pos = it - all_faces.begin();
  //           adjacent_elements.push_back(pos);
  //         }
  //       }
  //     }
  //   }
  //}

  //     // print number of adjacent elements found ...
  // #ifdef DEBUG
  //     std::cout << "\n#adj. edges: " << adjacent_elements.size() << std::endl;

  //     if (adjacent_elements.size() < 2) {
  //       std::cout << "*\tError: only " << adjacent_elements.size() << "neighbors found\n"
  //                 << "occured at " << cell.face(face_idx)->barycenter() << "\n";

  //     } else if (adjacent_elements.size() > 2){
  //       throw std::runtime_error("\ncannot have more than two neighbor faces\n");
  //     }

  // #endif
  //     for(auto col : adjacent_elements) {
  //         graph.insert(i, col) = 1;
  //     }
  //   }

  //   graph.makeCompressed();
  // #ifdef DEBUG
  //   // export graph
  //   std::ofstream fout("edge_graph.dat");
  //   fout << graph;
  //   fout.close();
  // #endif
  // ------------------------------------
  // walk through graph and cluster edges
  std::set<unsigned int> remaining;
  for (unsigned int i = 0; i < nfaces; ++i) remaining.insert(i);

  std::vector<unsigned int> linear_ordering;
  int current = 0;
  linear_ordering.push_back(current);
  remaining.erase(current);

  while (!remaining.empty()) {
    bool found_next = false;
    for (typename sparse_matrix_t::InnerIterator it(graph, current); it; ++it) {
      if ((it.col() != current) && (remaining.find(it.col()) != remaining.end())) {
        current = it.col();
        remaining.erase(current);
        linear_ordering.push_back(current);
        // exit loop
        found_next = true;
        break;
      }
    }
    if (!found_next) {
      assert(!remaining.empty());
      // #ifdef DEBUG
      //       std::cout << "\ngrid is not simply connected\n";
      // #endif
      // got stuck, continue elsewhere
      current = *(remaining.begin());
      remaining.erase(current);
      linear_ordering.push_back(current);
    }
  }

  // #ifdef DEBUG
  //   std::cout << "\nlinear ordering now contains: " << linear_ordering.size() << " entries.\n";
  // #endif
  assert(linear_ordering.size() == all_faces.size());

  long int bsize = std::ceil(nfaces / (1. * nprocs));
  // insert my_faces
  for (int i = bsize * pid; i < std::min(bsize * (pid + 1), (long int)nfaces); ++i) {
    my_faces_.push_back(all_faces[i]);
  }

  // relevant dofs
  for (auto v : my_faces_) {
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_face);
    auto& cell = std::get<0>(v);
    int face_idx = std::get<1>(v);
    cell.face(face_idx)->get_dof_indices(local_dof_indices);
    for (auto ix : local_dof_indices) {
      auto iibegin = indexer.to_global(ix, 0);
      relevant_dofs_.add_range(iibegin, iibegin + N);
    }
  }

  relevant_dofs_.compress();
}

}  // end namespace impl
}  // end namespace boltzmann

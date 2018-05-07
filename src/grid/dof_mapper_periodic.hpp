#pragma once

// deal.II includes ----------------------------------------
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/tria.h>

// system  includes ----------------------------------------
#include <fstream>
#include <iostream>
#include <map>
#include <vector>


namespace boltzmann {

/**y
 * @brief Provides mapping between periodic and non periodic indexing on a
 *        given grid in the *physical* domain.
 *
 * @param dh
 */
class DoFMapperPeriodic
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

  const std::vector<index_t>& get_mapping() const { return mapping; }

  unsigned int size() const { return size_; }

  const map_t& dof_map() const { return dof_map_; }

 private:
  unsigned int size_;
  /// full -> periodic renumbering
  std::vector<index_t> mapping;
  ///
  std::vector<index_t> inverse;
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
DoFMapperPeriodic::index_t
DoFMapperPeriodic::lookup(const index_t restricted_idx) const
{
  return this->inverse[restricted_idx];
}

// ------------------------------------------------------------
/**
 * @brief go from non-periodic dof-idx to periodic indexing
 *
 * @param unrestricted_idx  non-periodic dof-idx
 *
 * @return
 */
DoFMapperPeriodic::index_t DoFMapperPeriodic::operator[](const index_t unrestricted_idx) const
{
  return mapping[unrestricted_idx];
}

// ------------------------------------------------------------
void
DoFMapperPeriodic::init(const dealii::DoFHandler<2>& dh)
{
  typedef std::map<index_t, double> map_t;
  map_t dof_locations_y;
  map_t dof_locations_x;

  for (dealii::DoFHandler<2>::active_cell_iterator cell = dh.begin_active(); cell != dh.end();
       ++cell) {
    auto collect_vertices = [&](map_t& dof_locations, const int face_id, const int c_id) {
      if (cell->at_boundary() && cell->face(face_id)->at_boundary()) {
        dof_locations[cell->face(face_id)->vertex_dof_index(0, 0)] =
            cell->face(face_id)->vertex(0)[c_id];
        dof_locations[cell->face(face_id)->vertex_dof_index(1, 0)] =
            cell->face(face_id)->vertex(1)[c_id];
      }
    };
    collect_vertices(dof_locations_y, 1, 1);
    collect_vertices(dof_locations_x, 3, 0);
  }

  typedef std::pair<index_t, index_t> index_pair_t;
  typedef std::set<index_pair_t> index_pair_set_t;

  index_pair_set_t edges;
  // contains all dof indices located on the boundary
  std::set<index_t> indices;
  // build edges
  for (dealii::DoFHandler<2>::active_cell_iterator cell = dh.begin_active(); cell != dh.end();
       ++cell) {
    // left and right boundary
    int face_id = 0;
    // coordinate id
    int c_id = 1;
    if (cell->at_boundary() && cell->face(face_id)->at_boundary()) {
      for (index_t face_vertex = 0; face_vertex < 2; ++face_vertex) {
        map_t::const_iterator p = dof_locations_y.begin();
        for (; p != dof_locations_y.end(); ++p) {
          if (std::fabs(p->second - cell->face(face_id)->vertex(face_vertex)[c_id]) < 1e-8) {
            const index_t ix1 = cell->face(face_id)->vertex_dof_index(face_vertex, 0);
            const index_t ix2 = p->first;
            if (ix1 < ix2) {
              edges.insert(std::make_pair(ix1, ix2));
              indices.insert(ix1);
            } else {
              edges.insert(std::make_pair(ix2, ix1));
              indices.insert(ix2);
            }
            break;
          }
        }
        if (p == dof_locations_y.end()) {
          std::cerr << "No corresponding degree of freedom was found!"
                    << "At coordinate y = " << p->second << std::endl;
          exit(-1);
        }
        // Assert( p != dof_locations_y.end(),
        //         dealii::ExcMessage( "No corresponding degree of freedom was found!"));
      }
    }
    face_id = 2;
    c_id = 0;
    if (cell->at_boundary() && cell->face(face_id)->at_boundary()) {
      for (index_t face_vertex = 0; face_vertex < 2; ++face_vertex) {
        map_t::const_iterator p = dof_locations_x.begin();
        for (; p != dof_locations_x.end(); ++p) {
          if (std::fabs(p->second - cell->face(face_id)->vertex(face_vertex)[c_id]) < 1e-8) {
            const index_t ix1 = cell->face(face_id)->vertex_dof_index(face_vertex, 0);
            const index_t ix2 = p->first;
            if (ix1 < ix2) {
              edges.insert(std::make_pair(ix1, ix2));
              indices.insert(ix1);
            } else {
              edges.insert(std::make_pair(ix2, ix1));
              indices.insert(ix2);
            }
            break;
          }
        }
        if (p == dof_locations_x.end()) {
          std::cerr << "No corresponding degree of freedom was found!"
                    << "At coordinate x = " << p->second << std::endl;
          exit(-1);
        }
        // Assert( p != dof_locations_x.end(),
        //         dealii::ExcMessage( "No corresponding degree of freedom was found!"));
      }
    }
  }
  // build cluster
  typedef std::set<index_t> index_set_t;
  std::function<void(index_t, index_set_t & neighbors)> find_clones = [&](index_t ix,
                                                                          index_set_t& clones) {
    index_pair_set_t::iterator it = edges.begin();
    while (true) {
      // finished?
      if (!(it != edges.end())) break;
      if (it->first == ix) {
        // add this edge to
        index_t next = it->second;
        edges.erase(it++);
        // recursion
        find_clones(next, clones);
        clones.insert(next);
        // remove this edge from the set and go to the next element
      } else {
        ++it;
      }
    }
  };

  std::map<index_t, index_set_t> clones_map;
  // iterate over boundary vertices and cluster them
  for (index_t ix : indices) {
    index_set_t& clones = clones_map[ix];
    // recursively traverse the graph add to
    // neighbors and delete the edges after they have been visited
    find_clones(ix, clones);
    // create a map
    for (index_t i : clones) {
      dof_map_[i] = ix;
    }
  }

  // mapping for periodic boundary condition has been constructed,
  // now renumber the dofs
  struct DOF
  {
    // helper class
    DOF(const index_t ix_)
        : ix(ix_)
    {
    }
    index_t ix;
  };

  std::nullptr_t nullp;

  typedef std::shared_ptr<DOF> ptr_DOF;
  std::vector<ptr_DOF> mapping_ptr(dh.n_dofs(), NULL);

  unsigned int index_counter = 0;
  for (unsigned int i = 0; i < dh.n_dofs(); ++i) {
    // slave node? (one that is remapped to another?)
    auto it = dof_map_.find(i);
    if (it != dof_map_.end()) {
      index_t master = it->second;
      // does it point to another DOF which is already initialized?
      if (mapping_ptr[master] != nullp) {
        mapping_ptr[i] = mapping_ptr[master];
      } else {
        // create this dof
        mapping_ptr[master] = ptr_DOF(new DOF(index_counter++));
      }
    } else {
      // it is a master node
      // initialize it if required
      if (mapping_ptr[i] == nullp) {
        mapping_ptr[i] = ptr_DOF(new DOF(index_counter++));
      }
    }
  }
  mapping.resize(dh.n_dofs());
  // store renumber into mapping
  for (unsigned int i = 0; i < dh.n_dofs(); ++i) {
    mapping[i] = mapping_ptr[i]->ix;
  }

  this->size_ = index_counter;

  // inverse mapping
  inverse.resize(this->size_);
  for (unsigned int i = 0; i < dh.n_dofs(); ++i) {
    inverse[mapping[i]] = i;
  }
}
} // end namespace boltzmann

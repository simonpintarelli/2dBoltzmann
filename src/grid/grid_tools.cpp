#include "grid_tools.hpp"

#include <deal.II/base/exceptions.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <boost/lexical_cast.hpp>

extern "C" {
#include <metis.h>
}

#include <map>
#include <vector>

using namespace dealii;

//   Code taken from deal.II and adapted such that METIS takes
//   the dense boundary blocks into account when partitioning the
//   grid.

namespace boltzmann {

namespace {

/**
 * Exception
 */
DeclException1(ExcInvalidNumberOfPartitions,
               int,
               << "The number of partitions you gave is " << arg1
               << ", but must be greater than zero.");
/**
 * Exception
 */
DeclException1(ExcNonExistentSubdomain,
               int,
               << "The subdomain id " << arg1 << " has no cells associated with it.");
/**
 * Exception
 */
DeclException0(ExcTriangulationHasBeenRefined);

/**
 * Exception
 */
DeclException1(
    ExcScalingFactorNotPositive, double, << "The scaling factor must be positive, but is " << arg1);
/**
 * Exception
 */
template <int N>
DeclException1(ExcPointNotFoundInCoarseGrid,
               Point<N>,
               << "The point <" << arg1 << "> could not be found inside any of the "
               << "coarse grid cells.");
/**
 * Exception
 */
template <int N>
DeclException1(ExcPointNotFound,
               Point<N>,
               << "The point <" << arg1 << "> could not be found inside any of the "
               << "subcells of a coarse grid cell.");

/**
 * Exception
 */
DeclException1(ExcVertexNotUsed,
               unsigned int,
               << "The given vertex " << arg1 << " is not used in the given triangulation");

DeclException2(ExcInvalidArraySize,
               int,
               int,
               << "The array has size " << arg1 << " but should have size " << arg2);

struct face_connectivity_of_cells
{
  typedef std::map<std::pair<unsigned int, unsigned int>, unsigned int> indexmap_t;

  static void get(const dealii::Triangulation<2, 2>& triangulation,
                  SparsityPattern& cell_connectivity,
                  indexmap_t& indexmap)
  {
    // as built in this function, we only consider face neighbors, which leads
    // to a fixed number of entries per row (don't forget that each cell couples
    // with itself, and that neighbors can be refined)
    cell_connectivity.reinit(
        triangulation.n_active_cells(),
        triangulation.n_active_cells(),
        GeometryInfo<2>::faces_per_cell * GeometryInfo<2>::max_children_per_face + 1);

    // create a map pair<lvl,idx> -> SparsityPattern index
    // TODO: we are no longer using user_indices for this because we can get
    // pointer/index clashes when saving/restoring them. The following approach
    // works, but this map can get quite big. Not sure about more efficient solutions.
    // std::map< std::pair<unsigned int,unsigned int>, unsigned int >
    //   indexmap;
    unsigned int index = 0;
    for (typename dealii::internal::ActiveCellIterator<2, 2, Triangulation<2, 2> >::type
             cell = triangulation.begin_active();
         cell != triangulation.end();
         ++cell, ++index)
      indexmap[std::pair<unsigned int, unsigned int>(cell->level(), cell->index())] = index;

    // next loop over all cells and their neighbors to build the sparsity
    // pattern. note that it's a bit hard to enter all the connections when a
    // neighbor has children since we would need to find out which of its
    // children is adjacent to the current cell. this problem can be omitted if
    // we only do something if the neighbor has no children -- in that case it
    // is either on the same or a coarser level than we are. in return, we have
    // to add entries in both directions for both cells
    index = 0;
    for (typename dealii::internal::ActiveCellIterator<2, 2, Triangulation<2, 2> >::type
             cell = triangulation.begin_active();
         cell != triangulation.end();
         ++cell, ++index) {
      cell_connectivity.add(index, index);
      for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f)
        if ((cell->at_boundary(f) == false) && (cell->neighbor(f)->has_children() == false)) {
          unsigned int other_index =
              indexmap
                  .find(std::pair<unsigned int, unsigned int>(cell->neighbor(f)->level(),
                                                              cell->neighbor(f)->index()))
                  ->second;
          cell_connectivity.add(index, other_index);
          cell_connectivity.add(other_index, index);
        }
    }

    // now compress the so-built connectivity pattern
    cell_connectivity.compress();
  }
};

// ----------------------------------------------------------------------
void partition(const SparsityPattern& sparsity_pattern,
               const unsigned int n_partitions,
               std::vector<unsigned int>& partition_indices,
               std::vector<idx_t> cell_weights = {})
{
  Assert(sparsity_pattern.n_rows() == sparsity_pattern.n_cols(), ExcNotQuadratic());
  Assert(sparsity_pattern.is_compressed(), SparsityPattern::ExcNotCompressed());

  Assert(n_partitions > 0, ExcInvalidNumberOfPartitions(n_partitions));
  Assert(partition_indices.size() == sparsity_pattern.n_rows(),
         ExcInvalidArraySize(partition_indices.size(), sparsity_pattern.n_rows()));

  // check for an easy return
  if (n_partitions == 1) {
    std::fill_n(partition_indices.begin(), partition_indices.size(), 0U);
    return;
  }

// Make sure that METIS is actually
// installed and detected
#ifndef DEAL_II_WITH_METIS
  AssertThrow(false, ExcMETISNotInstalled());
#else

  // generate the data structures for
  // METIS. Note that this is particularly
  // simple, since METIS wants exactly our
  // compressed row storage format. we only
  // have to set up a few auxiliary arrays
  idx_t n = static_cast<signed int>(sparsity_pattern.n_rows()),
        ncon = 1,                               // number of balancing constraints (should be >0)
      nparts = static_cast<int>(n_partitions),  // number of subdomains to create
      dummy;                                    // the numbers of edges cut by the
  // resulting partition

  // use default options for METIS
  idx_t options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(options);

  // one more nuisance: we have to copy our
  // own data to arrays that store signed
  // integers :-(
  std::vector<idx_t> int_rowstart(1);
  int_rowstart.reserve(sparsity_pattern.n_rows() + 1);
  std::vector<idx_t> int_colnums;
  int_colnums.reserve(sparsity_pattern.n_nonzero_elements());
  for (SparsityPattern::size_type row = 0; row < sparsity_pattern.n_rows(); ++row) {
    for (SparsityPattern::iterator col = sparsity_pattern.begin(row);
         col < sparsity_pattern.end(row);
         ++col)
      int_colnums.push_back(col->column());
    int_rowstart.push_back(int_colnums.size());
  }

  std::vector<idx_t> int_partition_indices(sparsity_pattern.n_rows());

  // Make use of METIS' error code.
  int ierr;

  // Select which type of partitioning to
  // create

  // Use recursive if the number of
  // partitions is less than or equal to 8
  // if (n_partitions <= 8)
  //   ierr = METIS_PartGraphRecursive(&n, &ncon, &int_rowstart[0], &int_colnums[0],
  //                                   cell_weights.data(), NULL, NULL,
  //                                   &nparts,NULL,NULL,&options[0],
  //                                   &dummy,&int_partition_indices[0]);

  // // Otherwise use kway
  // else
  ierr = METIS_PartGraphKway(&n,
                             &ncon,
                             &int_rowstart[0],
                             &int_colnums[0],
                             cell_weights.data(),
                             NULL,
                             NULL,
                             &nparts,
                             NULL,
                             NULL,
                             &options[0],
                             &dummy,
                             &int_partition_indices[0]);
  // ierr = METIS_PartGraphRecursive(&n, &ncon, &int_rowstart[0], &int_colnums[0],
  //                                 cell_weights.data(), NULL, NULL,
  //                                 &nparts,NULL,NULL,&options[0],
  //                                 &dummy,&int_partition_indices[0]);

  // If metis returns normally, an
  // error code METIS_OK=1 is
  // returned from the above
  // functions (see metish.h)
  if (ierr != 1) {
    throw std::runtime_error("METIS-Error: " + boost::lexical_cast<std::string>(ierr));
  }
  //  AssertThrow (ierr == 1, ExcMETISError (ierr));

  // now copy back generated indices into the
  // output array
  std::copy(int_partition_indices.begin(), int_partition_indices.end(), partition_indices.begin());
#endif
}

}  // end namespace

void partition_triangulation(const unsigned int n_partitions,
                             dealii::Triangulation<2, 2>& triangulation,
                             const unsigned int cell_weight,
                             const unsigned int cell_weight_bc)
{
  Assert(n_partitions > 0, ExcInvalidNumberOfPartitions(n_partitions));

  // check for an easy return
  if (n_partitions == 1) {
    for (typename dealii::internal::ActiveCellIterator<2, 2, Triangulation<2, 2> >::type cell =
             triangulation.begin_active();
         cell != triangulation.end();
         ++cell)
      cell->set_subdomain_id(0);
    return;
  }

  // we decompose the domain by first generating the connection graph of all
  // cells with their neighbors, and then passing this graph off to METIS.
  // finally defer to the other function for partitioning and assigning
  // subdomain idsdealii::
  dealii::SparsityPattern cell_connectivity;
  typename face_connectivity_of_cells::indexmap_t cell_enumeration;
  face_connectivity_of_cells::get(triangulation, cell_connectivity, cell_enumeration);

  // check for an easy return
  if (n_partitions == 1) {
    for (typename dealii::internal::ActiveCellIterator<2, 2, Triangulation<2, 2> >::type cell =
             triangulation.begin_active();
         cell != triangulation.end();
         ++cell)
      cell->set_subdomain_id(0);
    return;
  }

  std::vector<idx_t> cell_weights(triangulation.n_active_cells());
  for (auto cell : triangulation.active_cell_iterators()) {
    int nfaces = dealii::GeometryInfo<2>::faces_per_cell;
    int count = 0;
    for (int i = 0; i < nfaces; ++i)
      if (cell->face(i)->at_boundary()) count++;

    unsigned int index =
        cell_enumeration.find(std::make_pair(cell->level(), cell->index()))->second;
    if (count == 0) cell_weights[index] = cell_weight;
    if (count > 0) cell_weights[index] = cell_weight_bc;
  }

  // partition this connection graph and get back a vector of indices, one per
  // degree of freedom (which is associated with a cell)
  std::vector<unsigned int> partition_indices(triangulation.n_active_cells());
  partition(cell_connectivity, n_partitions, partition_indices, cell_weights);

  // finally loop over all cells and set the subdomain ids
  unsigned int index = 0;
  for (typename dealii::internal::ActiveCellIterator<2, 2, Triangulation<2, 2> >::type
           cell = triangulation.begin_active();
       cell != triangulation.end();
       ++cell, ++index)
    cell->set_subdomain_id(partition_indices[index]);

  // (end)
}

}  // end namespace boltzmann

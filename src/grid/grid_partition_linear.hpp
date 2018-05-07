#pragma once

#include <deal.II/base/mpi.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include "grid/grid_partitioner.hpp"

namespace boltzmann {

class GridPartitionerLinear : public GridPartitioner<2>
{
 public:
  GridPartitionerLinear()
      : GridPartitionerLinear(-1)
  { }

  GridPartitionerLinear(int nprocs)
  {
    if (nprocs == -1)
      nprocs_ = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    else
      nprocs_ = nprocs;
  }

  int n_processes() const { return nprocs_; }

  /**
   * Distribute cells by cell_id on processors...
   *
   * @param dof_handler
   * @param triangulation
   */
  virtual void partition(dealii::DoFHandler<2>& dof_handler,
                         dealii::Triangulation<2>& triangulation) const;

 protected:
  using GridPartitioner<2>::nprocs_;
};

void GridPartitionerLinear::partition(dealii::DoFHandler<2>& dof_handler,
                                      dealii::Triangulation<2>& triangulation) const
{
  const int DIM = 2;
  // partition triangulationp
  const unsigned int nelems = triangulation.n_active_quads();
  const unsigned int local_size = std::ceil(nelems / (1.0 * nprocs_));

  for (auto cell_it : triangulation.active_cell_iterators()) {
    // tested index returns the cell label defined by gmsh
    int i = cell_it->index();
    //    int subdomain_id = vertices[i][dir] / delta;
    int subdomain_id = i / local_size;

    AssertThrow(subdomain_id < nprocs_, dealii::ExcMessage("index error"));

    cell_it->set_subdomain_id(subdomain_id);
  }

  dealii::DoFRenumbering::boost::Cuthill_McKee(dof_handler);
  dealii::DoFRenumbering::subdomain_wise(dof_handler);
}

}  // end namespace boltzmann

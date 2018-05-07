#pragma once

#include <deal.II/base/mpi.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

namespace boltzmann {

template <int DIM>
class GridPartitioner
{
 public:
  GridPartitioner()
      : GridPartitioner(-1)
  {
  }
  GridPartitioner(int nprocs)
  {
    if (nprocs == -1)
      nprocs_ = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    else
      nprocs_ = nprocs;
  }

  int n_processes() const { return nprocs_; }

  virtual void partition(dealii::DoFHandler<DIM>& dof_handler,
                         dealii::Triangulation<DIM>& triangulation) const;

 protected:
  int nprocs_;
};

template <int DIM>
void
GridPartitioner<DIM>::partition(dealii::DoFHandler<DIM>& dof_handler,
                                dealii::Triangulation<DIM>& triangulation) const
{
  dealii::GridTools::partition_triangulation(nprocs_, triangulation);
  dealii::DoFRenumbering::boost::Cuthill_McKee(dof_handler);
  dealii::DoFRenumbering::subdomain_wise(dof_handler);
}

}  // end namespace boltzmann

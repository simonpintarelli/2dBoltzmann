#pragma once

#include "grid_partitioner.hpp"
#include "grid_tools.hpp"

namespace boltzmann {

template <int DIM>
class GridPartitionerBC : public GridPartitioner<DIM>
{
 public:
  /**
   *
   *
   * @param cell_weight
   * @param cell_weight_bc
   *
   * @return
   */
  GridPartitionerBC(unsigned int cell_weight, unsigned int cell_weight_bc, int nprocs = -1)
      : GridPartitioner<DIM>(nprocs)
      , cell_weight_(cell_weight)
      , cell_weight_bc_(cell_weight_bc)
  {
    dealii::LogStream::Prefix p("GridPartitionerBC");
    dealii::deallog << "cell_weight=" << cell_weight << std::endl;
    dealii::deallog << "cell_weight_bc=" << cell_weight_bc << std::endl;
  }

  virtual void partition(dealii::DoFHandler<DIM>& dof_handler,
                         dealii::Triangulation<DIM>& triangulation) const;

 private:
  unsigned int cell_weight_;
  unsigned int cell_weight_bc_;
  using GridPartitioner<DIM>::nprocs_;
};

template <int DIM>
void
GridPartitionerBC<DIM>::partition(dealii::DoFHandler<DIM>& dof_handler,
                                  dealii::Triangulation<DIM>& triangulation) const
{
  boltzmann::partition_triangulation(nprocs_, triangulation, cell_weight_, cell_weight_bc_);
  dealii::DoFRenumbering::boost::Cuthill_McKee(dof_handler);
  dealii::DoFRenumbering::subdomain_wise(dof_handler);
}

}  // end namespace boltzmann

#pragma once

// deal.II includes -----------------------------------------------------
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/tria.h>


namespace boltzmann {
template <int dim>
class GridHandlerBase
{
 public:
  GridHandlerBase()
      : fe(1)
      , dofhandler_(triangulation_)
  { /* empty */
  }

  const dealii::Triangulation<dim>& triangulation() const { return triangulation_; }

  const dealii::DoFHandler<dim>& dofhandler() const { return dofhandler_; }

 protected:
  dealii::FE_Q<dim> fe;
  dealii::Triangulation<dim> triangulation_;
  dealii::DoFHandler<dim> dofhandler_;
};

}  // end namespace boltzmann

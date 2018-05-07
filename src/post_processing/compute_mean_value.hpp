#pragma once

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <iostream>

namespace boltzmann {

template <int dim, typename InVector>
double
compute_mean_value(const dealii::DoFHandler<dim>& dh, const InVector& v)
{
  dealii::QGauss<dim> quad(2);
  return dealii::VectorTools::compute_mean_value(dh, quad, v, 0);
}

namespace mpi {

template <int dim>
std::vector<double>
compute_mean_value(const dealii::DoFHandler<dim>& dh, const Epetra_MultiVector& src)
{
  unsigned int pid = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  dealii::QGauss<dim> quad(2);
  dealii::UpdateFlags flags =
      dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points;
  dealii::FEValues<dim> fe_values(dh.get_fe(), quad, flags);

  int nvec = src.NumVectors();

  std::vector<double> lsum(nvec, 0.0);

  unsigned int dofs_per_cell = fe_values.dofs_per_cell;
  std::vector<unsigned int> local_dof_indices(dofs_per_cell);
  for (auto cell = dh.begin_active(); cell != dh.end(); ++cell) {
    if (cell->subdomain_id() == pid) {
      fe_values.reinit(cell);
      cell->get_dof_indices(local_dof_indices);
      for (int k = 0; k < nvec; ++k) {
        for (unsigned int ix = 0; ix < dofs_per_cell; ++ix) {
          int lid = src.Map().LID(int(local_dof_indices[ix]));
          BOOST_ASSERT(lid >= 0);
          // if (lid < 0) std::cout << "lid " << lid << " gid " << local_dof_indices[ix] << "\n";
          for (unsigned int q = 0; q < quad.size(); ++q) {
            lsum[k] += src[k][lid] * fe_values.shape_value(ix, q) * fe_values.JxW(q);
          }
        }
      }
    }
  }

  std::vector<double> sum(nvec, 0.0);
  MPI_Allreduce(lsum.data(), sum.data(), nvec, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sum;
}
}  // namespace mpi

}  // namespace boltzmann

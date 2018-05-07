#pragma once

// deal.II includes -------------------------------------------------------------
#include <deal.II/base/index_set.h>
#include <deal.II/base/point.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/numerics/data_out.h>

// system includes --------------------------------------------------------------
#include <boost/static_assert.hpp>
#include <iomanip>
#include <iostream>
#include <vector>
// own includes -----------------------------------------------------------------
#include "aux/filtered_data_out.hpp"
#include "export/export_vtk.hpp"
#include "post_processing/compute_mean_value.hpp"
#include "post_processing/energy.hpp"
#include "post_processing/mass.hpp"
#include "post_processing/momentum.hpp"

namespace boltzmann {
/**
 *
 *
 * @param domain
 *
 * @return
 */
template <int dim>
class Moments2VTK
{
 private:
  typedef dealii::TrilinosWrappers::Vector vector_t;  // this class is deprecated
  typedef dealii::TrilinosWrappers::MPI::Vector mpi_vector_t;
  typedef dealii::DoFHandler<dim> dh_t;

 public:
  /**
   *
   * @param dh DoFHandler
   * @param sb spectral basis
   * @param epetra_map_phys local dofs (physical domain)
   * @param local_dofs  local dofs (phase space domain)
   * @param local_relevant_dofs local relevant dofs (phase space domain)
   */
  template <typename SB>
  Moments2VTK(const dh_t& dh, const SB& sb)
      : dof_handler(dh)
      , data_out(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
  {
    BOOST_STATIC_ASSERT(dim == 2);
    energy.init(sb);
    momentum.init(sb);
    mass.init(sb);
    pid = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    this->locally_relevant_phys_dofs =
        dealii::DoFTools::dof_indices_with_subdomain_association(dh, pid);
    // initialize vectors with ghost cells
    v_mass.reinit(locally_relevant_phys_dofs);
    v_energy.reinit(locally_relevant_phys_dofs);
    v_momentumX.reinit(locally_relevant_phys_dofs);
    v_momentumY.reinit(locally_relevant_phys_dofs);

    if (pid == 0) {
      // clear file
      std::ofstream fout(fxdmf, std::ios_base::out);
      fout << "<!-- Problems reading this file? Encasing XDMF tags might be missing ... -->";
      fout.close();
    }
  }

  ~Moments2VTK();

  /**
   * @brief write results to parallel filesystem in HDF5-Format
   *
   * @param src           solution vector with ghost elements
   * @param timestep_no
   * @param present_time
   * @param indexer
   */
  template <typename VEC_IN, typename INDEXER>
  void run(const VEC_IN& src,
           const size_t timestep_no,
           const double present_time,
           const INDEXER& indexer);

  template <typename VEC_IN, typename INDEXER>
  void run_debug(const VEC_IN& src,
                 const size_t timestep_no,
                 const double present_time,
                 const INDEXER& indexer,
                 const mpi_vector_t& scores);

  const vector_t& get_mass() const { return v_mass; }

 private:
  const dh_t& dof_handler;
  /// tasks
  Energy energy;
  Momentum momentum;
  Mass mass;
  // internal storage
  mpi_vector_t v_mass;
  mpi_vector_t v_energy;
  mpi_vector_t v_momentumX;
  mpi_vector_t v_momentumY;
  /// locally relevant *phys* DoFs
  dealii::IndexSet locally_relevant_phys_dofs;
  unsigned int pid;
  // IO
  dealii::FilteredDataOut<dim> data_out;
  std::vector<dealii::XDMFEntry> xdmf_entries;
  std::string fxdmf = "solution.xdmf";
};

// ----------------------------------------------------------------------
template <int dim>
template <typename VEC_IN, typename INDEXER>
void
Moments2VTK<dim>::run(const VEC_IN& src,
                      const size_t timestep_no,
                      const double present_time,
                      const INDEXER& indexer)
{
  // collect ghost entries
  auto write_to_stdout = [&](std::string tag, double value) {
    std::cout << "moments::monitor\t" << std::setw(15) << tag << std::setw(15) << timestep_no
              << std::scientific << std::setprecision(20) << std::setw(30) << value << std::endl;
  };

  dealii::FilteredDataOut<dim> data_out(pid);
  dealii::DataOutBase::VtkFlags flags = dealii::DataOutBase::VtkFlags();
  const bool filter_duplicate_vertices = true;
  const bool xdmf_hdf5_output = true;
  typedef dealii::DataOutBase::DataOutFilterFlags data_out_filter_flags;
  dealii::DataOutBase::DataOutFilter data_out_filter(
      data_out_filter_flags(filter_duplicate_vertices, xdmf_hdf5_output));
  data_out.set_flags(flags);
  data_out.attach_dof_handler(dof_handler);
  energy.compute(v_energy, src, locally_relevant_phys_dofs, indexer);
  data_out.add_data_vector(v_energy, "E");
  double e = mpi::compute_mean_value(dof_handler, v_energy);
  if (pid == 0) write_to_stdout("Energy", e);
  // mass
  mass.compute(v_mass, src, locally_relevant_phys_dofs, indexer);
  data_out.add_data_vector(v_mass, "M");
  double rho = mpi::compute_mean_value(dof_handler, v_mass);
  if (pid == 0) write_to_stdout("Mass", rho);
  // momentum
  momentum.compute(v_momentumX, v_momentumY, src, locally_relevant_phys_dofs, indexer);
  double ux = mpi::compute_mean_value(dof_handler, v_momentumX);
  double uy = mpi::compute_mean_value(dof_handler, v_momentumY);
  if (pid == 0) {
    write_to_stdout("ux", ux);
    write_to_stdout("uy", uy);
  }
  // ------------------------------------------------------------
  // VTK output
  data_out.add_data_vector(v_momentumX, "Ux");
  data_out.add_data_vector(v_momentumY, "Uy");
  data_out.build_patches();

  std::string filename = "solution-" + dealii::Utilities::int_to_string(timestep_no, 6) + ".hdf5";
  data_out.write_filtered_data(data_out_filter);
  data_out.write_hdf5_parallel(
      data_out_filter, timestep_no == 0, "mesh.hdf5", filename, MPI_COMM_WORLD);

  // xdmf entry
  auto xdmf_entry = data_out.create_xdmf_entry(
      data_out_filter, "mesh.hdf5", filename, present_time, MPI_COMM_WORLD);
  // xdmf_entries.push_back(xdmf_entry);

  // TODO: for 20K timesteps takes > 0.5sec to write...
  if (pid == 0) {
    std::string xdmf_str = xdmf_entry.get_xdmf_content(1 /* indent level */);
    std::ofstream fout(fxdmf, std::ios_base::out | std::ios_base::app);
    fout << xdmf_str << std::endl;
    fout.close();
  }

  //  data_out.write_xdmf_file(xdmf_entries, "solution.xdmf", MPI_COMM_WORLD);
}

// ----------------------------------------------------------------------
template <int dim>
template <typename VEC_IN, typename INDEXER>
void
Moments2VTK<dim>::run_debug(const VEC_IN& src,
                            const size_t timestep_no,
                            const double present_time,
                            const INDEXER& indexer,
                            const mpi_vector_t& scores)
{
  // collect ghost entries
  auto write_to_stdout = [&](std::string tag, double value) {
    std::cout << "moments::monitor\t" << std::setw(15) << tag << std::setw(15) << timestep_no
              << std::scientific << std::setprecision(20) << std::setw(30) << value << std::endl;
  };

  dealii::FilteredDataOut<dim> data_out(pid);
  dealii::DataOutBase::VtkFlags flags = dealii::DataOutBase::VtkFlags();
  const bool filter_duplicate_vertices = true;
  const bool xdmf_hdf5_output = true;
  typedef dealii::DataOutBase::DataOutFilterFlags data_out_filter_flags;
  dealii::DataOutBase::DataOutFilter data_out_filter(
      data_out_filter_flags(filter_duplicate_vertices, xdmf_hdf5_output));
  data_out.set_flags(flags);
  data_out.attach_dof_handler(dof_handler);
  energy.compute(v_energy, src, locally_relevant_phys_dofs, indexer);
  data_out.add_data_vector(v_energy, "E");
  double e = mpi::compute_mean_value(dof_handler, v_energy);
  if (pid == 0) write_to_stdout("Energy", e);
  // mass
  mass.compute(v_mass, src, locally_relevant_phys_dofs, indexer);
  data_out.add_data_vector(v_mass, "M");
  double rho = mpi::compute_mean_value(dof_handler, v_mass);
  if (pid == 0) write_to_stdout("Mass", rho);
  // momentum
  momentum.compute(v_momentumX, v_momentumY, src, locally_relevant_phys_dofs, indexer);
  double ux = mpi::compute_mean_value(dof_handler, v_momentumX);
  double uy = mpi::compute_mean_value(dof_handler, v_momentumY);
  if (pid == 0) {
    write_to_stdout("ux", ux);
    write_to_stdout("uy", uy);
  }
  // ------------------------------------------------------------
  // VTK output
  data_out.add_data_vector(v_momentumX, "Ux");
  data_out.add_data_vector(v_momentumY, "Uy");
  data_out.add_data_vector(scores, "Scores");
  data_out.build_patches();

  std::string filename = "solution-" + dealii::Utilities::int_to_string(timestep_no, 6) + ".hdf5";
  data_out.write_filtered_data(data_out_filter);
  data_out.write_hdf5_parallel(
      data_out_filter, timestep_no == 0, "mesh.hdf5", filename, MPI_COMM_WORLD);

  // xdmf entry
  auto xdmf_entry = data_out.create_xdmf_entry(
      data_out_filter, "mesh.hdf5", filename, present_time, MPI_COMM_WORLD);
  // xdmf_entries.push_back(xdmf_entry);

  // TODO: for 20K timesteps takes > 0.5sec to write...
  if (pid == 0) {
    std::string xdmf_str = xdmf_entry.get_xdmf_content(1 /* indent level */);
    std::ofstream fout(fxdmf, std::ios_base::out | std::ios_base::app);
    fout << xdmf_str << std::endl;
    fout.close();
  }

  //  data_out.write_xdmf_file(xdmf_entries, "solution.xdmf", MPI_COMM_WORLD);
}

// ----------------------------------------------------------------------
template <int dim>
Moments2VTK<dim>::~Moments2VTK()
{
  //  data_out.write_xdmf_file(xdmf_entries, "solution.xdmf", MPI_COMM_WORLD);
}
}

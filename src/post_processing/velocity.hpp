#pragma once

// deal.II includes ------------------------------------------------------------
//#include <deal.II/base/exceptions.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
// system includes ------------------------------------------------------------
#include <hdf5.h>
#include <boost/filesystem.hpp>
#include <fstream>
#include <list>
#include <vector>

namespace dealii {
DeclException0(PointNotInsideDomain);
}  // end namespace dealii

namespace boltzmann {
namespace local_ {

/**
 * @brief tracer object for velocity coefficients, sits at `x` in space
 *        (coefficients are averaged over physical neighbors)
 *
 */
template <int dim>
class VelocityPointRecorder
{
 private:
  typedef dealii::DoFHandler<dim> dofhandler_t;
  typedef dealii::Point<dim> point_t;

 public:
  /**
   * @brief initialize for velocity recording
   *        returns subodmain id of the cell where x is located in.
   *
   * @param x    global point in space
   * @param dh   dofhandler
   *
   * @return
   */
  int init(const point_t& x, const dofhandler_t& dh, const int n_velo_dofs);

  template <typename IN_VEC>
  void compute(const IN_VEC& src);

  const std::vector<double>& get_data() const { return result; }

  unsigned int n_coeffs() { return result.size(); }

 private:
  typedef dealii::types::global_dof_index size_type;
  /// physical dof indices
  std::vector<size_type> indices;
  std::vector<double> weights;
  std::vector<double> result;
};

// ----------------------------------------------------------------------
template <int dim>
int
VelocityPointRecorder<dim>::init(const point_t& x, const dofhandler_t& dh, const int n_velo_dofs)
{
  typedef typename dofhandler_t::cell_iterator cell_t;
  cell_t cell = dh.begin_active();
  for (; cell != dh.end(); ++cell) {
    if (cell->point_inside(x)) break;
  }
  AssertThrow(cell != dh.end(), dealii::PointNotInsideDomain());

  int n_dofs_per_cell = dh.get_fe().dofs_per_cell;
  auto& fe = dh.get_fe();
  this->indices.resize(n_dofs_per_cell);
  this->weights.resize(n_dofs_per_cell);
  this->result.resize(n_velo_dofs);
  cell->get_dof_indices(this->indices);
  dealii::MappingQ1<dim> mapping;
  point_t xref = mapping.transform_real_to_unit_cell(cell, x);
  for (int i = 0; i < n_dofs_per_cell; ++i) {
    this->weights[i] = fe.shape_value(i, xref);
  }

  return cell->subdomain_id();
}

// ---------------------------------------------------------------------
template <int dim>
template <typename IN_VEC>
void
VelocityPointRecorder<dim>::compute(const IN_VEC& src)
{
  const unsigned int n_velo_dofs = result.size();
  for (unsigned int i = 0; i < n_velo_dofs; ++i) {
    // reset
    result[i] = 0;
    for (unsigned int k = 0; k < indices.size(); ++k) {
      unsigned int gid = indices[k] * n_velo_dofs + i;
      result[i] += src[gid] * weights[k];
    }
  }
}

}  // end namespace local_

template <int dim>
class VelocityRecorder
{
 private:
  typedef dealii::Point<dim> point_t;

 public:
  VelocityRecorder()
      : is_initialized(false)
      , output_dir("")
  { /* empty */
  }

  ~VelocityRecorder();

  void init(const std::vector<point_t>& recording_points,
            const dealii::DoFHandler<dim>& dof_handler,
            const int n_velo_dofs);

  void set_output_dir(const std::string& path);

  template <typename VEC_IN>
  void write_txt(const VEC_IN& src, int frame_id);

  template <typename VEC_IN>
  void write_hdf(const VEC_IN& src, int frame_id);

 private:
  typedef local_::VelocityPointRecorder<dim> rec_t;

 private:
  bool is_initialized;
  /// set output directory
  std::string output_dir;
  int my_pid;
  std::list<std::pair<point_t, rec_t> > recs;
  /// this is a pointer because it is dynamically allocated
  //  ptr_h5_file_t ptr_h5_file;
  hid_t h5_file;
};

// ----------------------------------------------------------------------
template <int dim>
VelocityRecorder<dim>::~VelocityRecorder()
{
  H5Fclose(h5_file);
}

// ----------------------------------------------------------------------
template <int dim>
void
VelocityRecorder<dim>::init(const std::vector<point_t>& recording_points,
                            const dealii::DoFHandler<dim>& dof_handler,
                            const int n_velo_dofs)
{
  this->my_pid = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  for (auto& xi : recording_points) {
    rec_t rec;
    int sd;
    try {
      sd = rec.init(xi, dof_handler, n_velo_dofs);
    } catch (dealii::PointNotInsideDomain) {
      continue;
    }
    if (sd == this->my_pid) recs.push_back(std::make_pair(xi, rec));
  }

  // write metadata

  // build filesystem
  boost::filesystem::path fpath(output_dir);
  std::string fname = "velocity_" + dealii::Utilities::int_to_string(this->my_pid, 3) + ".h5";
  boost::filesystem::path pfname(fname);
  boost::filesystem::path ppath2fname = fpath / pfname;
  // create file
  h5_file = H5Fcreate(ppath2fname.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  herr_t status;
  hid_t space, dset;
  hsize_t dims[2] = {recs.size(), dim};
  // create dataspace
  space = H5Screate_simple(2, dims, NULL);
  // create dataset
  dset = H5Dcreate2(
      h5_file, "/locations", H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // prepare buffer
  boost::multi_array<double, 2> buffer(boost::extents[recs.size()][dim]);
  int i = 0;
  for (auto it = recs.begin(); it != recs.end(); ++it, ++i) {
    const auto& x = it->first;
    for (int d = 0; d < dim; ++d) {
      buffer[i][d] = x[d];
    }
  }

  status = H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer.data());

  is_initialized = true;

  // close resources
  H5Sclose(space);
  H5Dclose(dset);
}

// ----------------------------------------------------------------------
template <int dim>
void
VelocityRecorder<dim>::set_output_dir(const std::string& path)
{
  AssertThrow(boost::filesystem::is_directory(path),
              dealii::ExcMessage("VelocityRecorder::set_output_dir, path does not exist\n"));

  output_dir = path;
}

// ----------------------------------------------------------------------
template <int dim>
template <typename VEC_IN>
void
VelocityRecorder<dim>::write_txt(const VEC_IN& src, int frame_id)
{
  if (is_initialized && recs.size() > 0) {
    // Assumption: locally_relevant dofs have been communicated!
    // build filesystem
    boost::filesystem::path fpath(output_dir);
    std::string fname = "velocity_" + dealii::Utilities::int_to_string(frame_id, 4) + "." +
                        dealii::Utilities::int_to_string(this->my_pid, 3) + ".dat";
    boost::filesystem::path pfname(fname);
    boost::filesystem::path ppath2fname = fpath / pfname;

    std::ofstream fout(ppath2fname.string());
    for (auto& v : recs) {
      const auto& x = v.first;
      auto& data_obj = v.second;
      data_obj.compute(src);
      fout << x << "\t";
      const auto& coefficients = data_obj.get_data();
      for (unsigned int i = 0; i < coefficients.size(); ++i) {
        fout << coefficients[i] << " ";
      }
      fout << "\n";
    }
    fout.close();
  }
}

// --------------------------------------------------------------------------
template <int dim>
template <typename VEC_IN>
void
VelocityRecorder<dim>::write_hdf(const VEC_IN& src, int frame_id)
{
  if (is_initialized && recs.size() > 0) {
    // write coefficients into temporary buffer
    boost::multi_array<double, 2> buffer(
        boost::extents[recs.size()][recs.front().second.n_coeffs()]);
    int i = 0;
    for (auto it = recs.begin(); it != recs.end(); ++it, ++i) {
      it->second.compute(src);
      const auto& data = it->second.get_data();
      for (unsigned int k = 0; k < data.size(); ++k) {
        buffer[i][k] = data[k];
      }
    }

    // pass temporary buffer to hdf
    hsize_t dims[] = {recs.size(), buffer.shape()[1]};

    herr_t status;
    hid_t space, dset;
    std::string dset_name = boost::lexical_cast<std::string>(frame_id);
    // create dataspace
    space = H5Screate_simple(2, dims, NULL);
    // create dataset
    dset = H5Dcreate2(h5_file,
                      dset_name.c_str(),
                      H5T_NATIVE_DOUBLE,
                      space,
                      H5P_DEFAULT,
                      H5P_DEFAULT,
                      H5P_DEFAULT);

    status = H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer.data());

    is_initialized = true;

    // close resources
    H5Sclose(space);
    H5Dclose(dset);
    H5Fflush(h5_file, H5F_SCOPE_GLOBAL);
  }
}
}  // end namespace boltzmann

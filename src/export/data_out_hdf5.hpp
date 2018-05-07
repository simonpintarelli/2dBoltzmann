#pragma once

#include <hdf5.h>
#include <string>

#include <deal.II/base/data_out_base.h>
#include <deal.II/numerics/data_out.h>

namespace dealii {
template <int dim>
class DataOutHDF : public DataOut<dim>
{
 public:
  void write_hdf5(const std::string& filename,
                  const DataOutBase::DataOutFilter& data_filter = DataOutBase::DataOutFilter());

  XDMFEntry create_xdmf_entry(const DataOutBase::DataOutFilter& data_filter,
                              const std::string& h5_mesh_filename,
                              const std::string& h5_solution_filename,
                              const double cur_time) const;

  void write_mesh(const std::string& mesh_filename, const DataOutBase::DataOutFilter& data_filter);

 private:
};

template <int dim>
XDMFEntry
DataOutHDF<dim>::create_xdmf_entry(const DataOutBase::DataOutFilter& data_filter,
                                   const std::string& h5_mesh_filename,
                                   const std::string& h5_solution_filename,
                                   const double cur_time) const
{
  unsigned int node_cell_count[2];

  AssertThrow(dim == 2 || dim == 3, ExcMessage("XDMF only supports 2 or 3 dimensions."));

  node_cell_count[0] = data_filter.n_nodes();
  node_cell_count[1] = data_filter.n_cells();

  // std::cout << "DataOutHDF::create_xdmf_entry, nodes: " << node_cell_count[0]
  //           << std::endl
  //           << "cells: " << node_cell_count[1]
  //           << std::endl;

  XDMFEntry entry(h5_mesh_filename,
                  h5_solution_filename,
                  cur_time,
                  node_cell_count[0],
                  node_cell_count[1],
                  dim);
  unsigned int n_data_sets = data_filter.n_data_sets();

  // The vector names generated here must match those generated in the HDF5 file
  for (unsigned int i = 0; i < n_data_sets; ++i) {
    entry.add_attribute(data_filter.get_data_set_name(i), data_filter.get_data_set_dim(i));
  }

  return entry;
}

template <int dim>
void
DataOutHDF<dim>::write_hdf5(const std::string& filename,
                            const DataOutBase::DataOutFilter& data_filter)
{
  std::vector<double> node_data_vec;
  std::vector<unsigned int> cell_data_vec;

  hsize_t node_cell_count[2];
  hid_t h5_solution_file_id;
  h5_solution_file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  node_cell_count[0] = data_filter.n_nodes();
  node_cell_count[1] = data_filter.n_cells();

  // std::cout << "DataOutHDF, node_cell_count, nodes: " << node_cell_count[0]
  //           << std::endl
  //           << "cells: " << node_cell_count[1]
  //           << std::endl;

  // HDF5 output
  for (unsigned int i = 0; i < data_filter.n_data_sets(); ++i) {
    unsigned int pt_data_vector_dim = data_filter.get_data_set_dim(i);
    std::string vector_name = data_filter.get_data_set_name(i);

    // create the dataspace for the point data;
    hsize_t node_ds_dim[2];
    hsize_t count[2];
    hsize_t cell_ds_dim[2];

    node_ds_dim[0] = node_cell_count[0];
    node_ds_dim[1] = pt_data_vector_dim;

    hid_t pt_data_dataspace = H5Screate_simple(2, node_ds_dim, NULL);

#if H5Gcreate_vers == 1
    hid_t pt_data_dataset = H5Dcreate(h5_solution_file_id,
                                      vector_name.c_str(),
                                      H5T_NATIVE_DOUBLE,
                                      pt_data_dataspace,
                                      H5P_DEFAULT);
#else
    hid_t pt_data_dataset = H5Dcreate(h5_solution_file_id,
                                      vector_name.c_str(),
                                      H5T_NATIVE_DOUBLE,
                                      pt_data_dataspace,
                                      H5P_DEFAULT,
                                      H5P_DEFAULT,
                                      H5P_DEFAULT);
#endif
    AssertThrow(pt_data_dataset >= 0, ExcIO());

    count[0] = node_cell_count[0];
    count[1] = pt_data_vector_dim;

    hid_t pt_data_memory_dataspace = H5Screate_simple(2, count, NULL);
    AssertThrow(pt_data_memory_dataspace >= 0, ExcIO());

    hid_t pt_data_file_dataspace = H5Dget_space(pt_data_dataset);
    AssertThrow(pt_data_file_dataspace >= 0, ExcIO());

    herr_t status = H5Dwrite(pt_data_dataset,
                             H5T_NATIVE_DOUBLE,
                             pt_data_memory_dataspace,
                             pt_data_file_dataspace,
                             H5P_DEFAULT,
                             data_filter.get_data_set(i));

    AssertThrow(status >= 0, ExcIO());

    status = H5Sclose(pt_data_dataspace);
    AssertThrow(status >= 0, ExcIO());
    status = H5Sclose(pt_data_memory_dataspace);
    AssertThrow(status >= 0, ExcIO());
    status = H5Sclose(pt_data_file_dataspace);
    AssertThrow(status >= 0, ExcIO());
    // Close the dataset
    status = H5Dclose(pt_data_dataset);
    AssertThrow(status >= 0, ExcIO());
  }

  herr_t status = H5Fclose(h5_solution_file_id);
  AssertThrow(status >= 0, ExcIO());
}

template <int dim>
void
DataOutHDF<dim>::write_mesh(const std::string& mesh_filename,
                            const DataOutBase::DataOutFilter& data_filter)
{
  std::vector<double> node_data_vec;
  std::vector<unsigned int> cell_data_vec;

  hsize_t node_cell_count[2];
  hsize_t node_ds_dim[2];
  hsize_t count[2];
  hsize_t cell_ds_dim[2];

  herr_t status;

  node_cell_count[0] = data_filter.n_nodes();
  node_cell_count[1] = data_filter.n_cells();

  // Overwrite any existing files (change this to an option?)
  hid_t h5_mesh_file_id = H5Fcreate(mesh_filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  AssertThrow(h5_mesh_file_id >= 0, ExcIO());

  // Create the dataspace for the nodes and cells
  node_ds_dim[0] = node_cell_count[0];
  node_ds_dim[1] = dim;
  hid_t node_dataspace = H5Screate_simple(2, node_ds_dim, NULL);
  AssertThrow(node_dataspace >= 0, ExcIO());

  cell_ds_dim[0] = node_cell_count[1];
  cell_ds_dim[1] = GeometryInfo<dim>::vertices_per_cell;
  hid_t cell_dataspace = H5Screate_simple(2, cell_ds_dim, NULL);
  AssertThrow(cell_dataspace >= 0, ExcIO());

// Create the dataset for the nodes and cells
#if H5Gcreate_vers == 1
  hid_t node_dataset =
      H5Dcreate(h5_mesh_file_id, "nodes", H5T_NATIVE_DOUBLE, node_dataspace, H5P_DEFAULT);
#else
  hid_t node_dataset = H5Dcreate(h5_mesh_file_id,
                                 "nodes",
                                 H5T_NATIVE_DOUBLE,
                                 node_dataspace,
                                 H5P_DEFAULT,
                                 H5P_DEFAULT,
                                 H5P_DEFAULT);
#endif
  AssertThrow(node_dataset >= 0, ExcIO());
#if H5Gcreate_vers == 1
  hid_t cell_dataset =
      H5Dcreate(h5_mesh_file_id, "cells", H5T_NATIVE_UINT, cell_dataspace, H5P_DEFAULT);
#else
  hid_t cell_dataset = H5Dcreate(h5_mesh_file_id,
                                 "cells",
                                 H5T_NATIVE_UINT,
                                 cell_dataspace,
                                 H5P_DEFAULT,
                                 H5P_DEFAULT,
                                 H5P_DEFAULT);
#endif
  AssertThrow(cell_dataset >= 0, ExcIO());

  // Close the node and cell dataspaces since we're done with them
  status = H5Sclose(node_dataspace);
  AssertThrow(status >= 0, ExcIO());
  status = H5Sclose(cell_dataspace);
  AssertThrow(status >= 0, ExcIO());

  // Create the data subset we'll use to read from memory
  count[0] = node_cell_count[0];
  count[1] = dim;
  hid_t node_memory_dataspace = H5Screate_simple(2, count, NULL);
  AssertThrow(node_memory_dataspace >= 0, ExcIO());

  // Select the hyperslab in the file
  hid_t node_file_dataspace = H5Dget_space(node_dataset);
  AssertThrow(node_file_dataspace >= 0, ExcIO());

  // And repeat for cells
  count[0] = node_cell_count[1];
  count[1] = GeometryInfo<dim>::vertices_per_cell;
  // offset[0] = global_node_cell_offsets[1];
  // offset[1] = 0;
  hid_t cell_memory_dataspace = H5Screate_simple(2, count, NULL);
  AssertThrow(cell_memory_dataspace >= 0, ExcIO());

  hid_t cell_file_dataspace = H5Dget_space(cell_dataset);
  AssertThrow(cell_file_dataspace >= 0, ExcIO());
  // status = H5Sselect_hyperslab(cell_file_dataspace, H5S_SELECT_SET, offset, NULL, count, NULL);
  // AssertThrow(status >= 0, ExcIO());

  // And finally, write the node data
  data_filter.fill_node_data(node_data_vec);
  status = H5Dwrite(node_dataset,
                    H5T_NATIVE_DOUBLE,
                    node_memory_dataspace,
                    node_file_dataspace,
                    H5P_DEFAULT,
                    &node_data_vec[0]);
  AssertThrow(status >= 0, ExcIO());
  node_data_vec.clear();

  // And the cell data
  data_filter.fill_cell_data(0, cell_data_vec);
  status = H5Dwrite(cell_dataset,
                    H5T_NATIVE_UINT,
                    cell_memory_dataspace,
                    cell_file_dataspace,
                    H5P_DEFAULT,
                    &cell_data_vec[0]);
  AssertThrow(status >= 0, ExcIO());
  cell_data_vec.clear();

  // Close the file dataspaces
  status = H5Sclose(node_file_dataspace);
  AssertThrow(status >= 0, ExcIO());
  status = H5Sclose(cell_file_dataspace);
  AssertThrow(status >= 0, ExcIO());

  // Close the memory dataspaces
  status = H5Sclose(node_memory_dataspace);
  AssertThrow(status >= 0, ExcIO());
  status = H5Sclose(cell_memory_dataspace);
  AssertThrow(status >= 0, ExcIO());

  // Close the datasets
  status = H5Dclose(node_dataset);
  AssertThrow(status >= 0, ExcIO());
  status = H5Dclose(cell_dataset);
  AssertThrow(status >= 0, ExcIO());

  status = H5Fclose(h5_mesh_file_id);
  AssertThrow(status >= 0, ExcIO());
}

}  // end namespace dealii

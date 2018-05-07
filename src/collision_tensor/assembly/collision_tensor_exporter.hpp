#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <boost/lexical_cast.hpp>
#include <string>
#include "hdf5.h"
// own includes ----------------------------------------------------------------------
#include "aux/eigen2hdf.hpp"

#include "spectral/basis/basis_descriptor.hpp"

namespace boltzmann {

/**
 * @brief Helper class for hdf export of collision_tensor
 *
 */
class CollisionTensorExporter
{
 private:
  typedef Eigen::SparseMatrix<double> sparse_matrix_t;

 public:
  CollisionTensorExporter(const char* fname);
  // /**
  //  * @brief add meta-data to hdf output file
  //  *
  //  * @param info
  //  */
  // void write_description(const BasisDescriptor& info);

  virtual ~CollisionTensorExporter();

  /**
   * TODO
   *
   * @param j
   * @param mat
   */
  void write_slice(int j, const sparse_matrix_t& mat);
  void write_mass_matrix(const sparse_matrix_t& mat);
  //  void add_info();

  hid_t file;
};

// --------------------------------------------------------------------------------
CollisionTensorExporter::CollisionTensorExporter(const char* fname)
{
  file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
}

CollisionTensorExporter::~CollisionTensorExporter() { H5Fclose(file); }

// // --------------------------------------------------------------------------------
// inline void CollisionTensorExporter::write_description(const BasisDescriptor& info)
// {
//   H5::DSetCreatPropList plist;
//   hsize_t dims[] = {1};
//   H5::DataSpace dspace(1, dims);
//   H5::DataSet dset = file.createDataSet("info",
//                                         H5::PredType::NATIVE_INT, dspace, plist);
//   H5::Attribute K = dset.createAttribute("K",
//                                          H5::PredType::NATIVE_INT, dspace);
//   K.write(H5::PredType::NATIVE_INT, &info.K);
//   H5::Attribute L = dset.createAttribute("L",
//                                          H5::PredType::NATIVE_INT, dspace);
//   L.write(H5::PredType::NATIVE_INT, &info.L);
//   H5::Attribute beta = dset.createAttribute("beta",
//                                             H5::PredType::NATIVE_DOUBLE, dspace);
//   beta.write(H5::PredType::NATIVE_DOUBLE, &info.beta);
// }

// ---------------------------------------------------------------------------------
inline void
CollisionTensorExporter::write_slice(int j, const sparse_matrix_t& mat)
{
  eigen2hdf::save_sparse(this->file, boost::lexical_cast<std::string>(j), mat);
  H5Fflush(this->file, H5F_SCOPE_GLOBAL);
}

// ----------------------------------------------------------------------------------
inline void
CollisionTensorExporter::write_mass_matrix(const sparse_matrix_t& mat)
{
  eigen2hdf::save_sparse(this->file, "mass_matrix", mat);
  H5Fflush(this->file, H5F_SCOPE_GLOBAL);
}

}  // end namespace boltzmann

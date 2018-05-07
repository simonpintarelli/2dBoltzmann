#include <iostream>
//#include <boost/lexical_cast.hpp>
#include <cassert>
#include <exception>
#include <string>
//#include <deal.II/base/exceptions.h>
#include <boost/filesystem.hpp>
#include "aux/exceptions.h"

#include "collision_tensor.hpp"
//#include "aux/eigen3-hdf5-sparse.hpp"
#include "aux/eigen2hdf.hpp"

using namespace std;

namespace boltzmann {

// --------------------------------------------------------------------------------------
CollisionTensor::CollisionTensor(int N)
    : N_(N)
    , vtmp(N)
    , slices_(N)
{
  /* empty */
}

// --------------------------------------------------------------------------------------
void CollisionTensor::add(ptr_t& slice, unsigned int j)
{
  if (j < N_)
    slices_[j] = slice;
  else {
    slices_.push_back(slice);
    ++N_;
  }
  vtmp.resize(N_);
}

// --------------------------------------------------------------------------------------
const CollisionTensor::sparse_matrix_t& CollisionTensor::get(int j) { return *(this->slices_[j]); }

// --------------------------------------------------------------------------------------
void CollisionTensor::set_mass_matrix(sparse_matrix_t& m)
{
  // mass_matrix_.swap(m);
  mass_matrix_ = m;
  lu_.compute(mass_matrix_);
  assert(lu_.info() != 0);
}

// // --------------------------------------------------------------------------------------
// void CollisionTensor::export_hdf5(const char* fname, const BasisDescriptor& info) const
// {
//   H5::H5File file(fname, H5F_ACC_TRUNC);
//   for (unsigned int j = 0; j < slices_.size(); ++j) {
//     EigenHDF5::save_sparse(file, boost::lexical_cast<std::string>(j), *slices_[j]);
//   }
//   H5::DSetCreatPropList plist;
//   hsize_t dims[] = {1};
//   H5::DataSpace dspace(1, dims);
//   H5::DataSet dset = file.createDataSet("info", H5::PredType::NATIVE_INT, dspace, plist);
//   H5::Attribute K = dset.createAttribute("K", H5::PredType::NATIVE_INT, dspace);
//   K.write(H5::PredType::NATIVE_INT, &info.K);
//   H5::Attribute L = dset.createAttribute("L", H5::PredType::NATIVE_INT, dspace);
//   L.write(H5::PredType::NATIVE_INT, &info.L);
//   H5::Attribute beta = dset.createAttribute("beta", H5::PredType::NATIVE_DOUBLE, dspace);
//   beta.write(H5::PredType::NATIVE_DOUBLE, &info.beta);
// }

// --------------------------------------------------------------------------------------
void CollisionTensor::read_hdf5(const char* fname, const int N)
{
  if (!boost::filesystem::exists(fname)) {
    throw std::runtime_error("collision tensor file not found");
  }
  N_ = N;
  slices_.resize(N);
  vtmp.resize(N);
  hid_t file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
  BAssertThrow(file > 0, "Could not open collision tensor file `" + std::string(fname) + "`");

  for (int i = 0; i < N; ++i) {
    slices_[i] = ptr_t(new sparse_matrix_t(N, N));
    //    EigenHDF5::load_sparse(file , boost::lexical_cast<string>(i), *slices_[i]);
    eigen2hdf::load_sparse(file, std::to_string(i), *slices_[i]);
    // make sure this matrix is stored in compressed format
    slices_[i]->makeCompressed();
  }

  mass_matrix_ = sparse_matrix_t(N, N);
  //  EigenHDF5::load_sparse(file, "mass_matrix", mass_matrix_);
  eigen2hdf::load_sparse(file, "mass_matrix", mass_matrix_);
  mass_matrix_.makeCompressed();
  lu_.compute(mass_matrix_);
  BAssertThrow((N == slices_[0]->rows()) && (N == slices_[0]->cols()),
               "Basis does not match collision tensor");

  H5Fclose(file);
}
}  // end boltzmann

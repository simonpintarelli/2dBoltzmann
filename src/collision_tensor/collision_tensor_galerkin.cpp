#include <boost/lexical_cast.hpp>
#include <cassert>
#include <exception>
#include <iostream>
#include <string>
// own includes
#include <boost/filesystem.hpp>
#include "aux/eigen2hdf.hpp"
#include "aux/exceptions.h"
#include "collision_tensor_galerkin.hpp"
//#include "post_processing/macroscopic_quantities.hpp"
//#include "spectral/utility/mass_matrix.hpp"

using namespace std;

namespace boltzmann {

// --------------------------------------------------------------------------------------
const CollisionTensorGalerkin::sparse_matrix_t&
CollisionTensorGalerkin::get(int j) const
{
  return slices_[j];
}

// --------------------------------------------------------------------------------------
void
CollisionTensorGalerkin::read_hdf5(const char* fname)
{
  if (!boost::filesystem::exists(fname)) {
    throw std::runtime_error("collision tensor file not found");
  }
  slices_.resize(N_);
  buf_.resize(N_);
  hid_t file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);

  for (unsigned int i = 0; i < N_; ++i) {
    slices_[i] = sparse_matrix_t(N_, N_);
    //    EigenHDF5::load_sparse(file , boost::lexical_cast<string>(i), *slices_[i]);
    eigen2hdf::load_sparse(file, boost::lexical_cast<string>(i), slices_[i]);
    // make sure this matrix is stored in compressed format
    slices_[i].makeCompressed();
  }

  BAssertThrow((N_ == static_cast<unsigned int>(slices_[0].rows())) &&
                   (N_ == static_cast<unsigned int>(slices_[0].cols())),
               "Basis does not match collision tensor");

  H5Fclose(file);
}


unsigned long
CollisionTensorGalerkin::nnz() const
{
  unsigned long nz = 0;
  for (const auto sl : slices_) {
    nz += sl.nonZeros();
  }
  return nz;
}

}  // namespace boltzmann

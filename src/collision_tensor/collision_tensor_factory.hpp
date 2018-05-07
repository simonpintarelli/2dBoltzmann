#pragma once

// system includes -------------------------------------------------------------
#include <iostream>
#include <type_traits>

// own includes ----------------------------------------------------------------
#include "spectral/basis/spectral_basis_factory_ks.hpp"
#include "traits/spectral_traits.hpp"

#include "aux/eigen2hdf.hpp"
#include "collision_tensor/collision_tensor.hpp"

// system includes -------------------------------------------------------------
#include <boost/filesystem.hpp>

namespace boltzmann {

template <typename ELEM>
class CollisionTensorFactory
{
 public:
  typedef CollisionTensor collision_tensor_t;

 private:
  typedef Eigen::SparseMatrix<double> matrix_t;

 private:
  typedef typename SpectralFactoryTraits<ELEM>::basis_factory_t basis_factory_t;

 public:
  /**
   * @brief read collision tensor from file
   *
   * Note: new_basis must be a subset of the original basis in which the
   *       tensor was computed.
   *
   * @param tensor         destination
   * @param trial_basis    destination trial basis
   * @param tensor_file    path/to/collision/tensor/hdf5-File
   * @param tensor_basis_descriptor path/to/collsion/tensor/basis/descriptor
   *                                (trial basis)
   */
  template <typename BASIS>
  static void load_from_file(collision_tensor_t& tensor,
                             const BASIS& trial_basis,
                             std::string tensor_file = "collision_tensor.h5",
                             std::string tensor_basis_descriptor = "ct_basis.desc");
};

template <typename ELEM>
template <typename BASIS>
void
CollisionTensorFactory<ELEM>::load_from_file(collision_tensor_t& tensor,
                                             const BASIS& trial_basis,
                                             std::string tensor_file,
                                             std::string tensor_basis_descriptor)
{
  // !! Attention: all (from tensor and new ones) basis descriptors need to have the same ordering
  // in k,l
  static_assert(std::is_same<typename BASIS::elem_t, ELEM>::value, "type mismatch");

  hid_t file = H5Fopen(tensor_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  if (!boost::filesystem::exists(tensor_basis_descriptor)) {
    std::cerr << "cannot find file " << tensor_basis_descriptor << std::endl;
  }
  // read the reference basis from file
  BASIS ref_basis;
  basis_factory_t::create(ref_basis, tensor_basis_descriptor);

  unsigned int Nref = ref_basis.n_dofs();
  unsigned int N = trial_basis.n_dofs();

  //  - check: test_basis is a subset of ref_basis
  //  - create mapping jref2jnew index old -> index new
  std::vector<int> jref2jnew(Nref, -1);
  for (auto it = trial_basis.begin(); it != trial_basis.end(); ++it) {
    unsigned int jnew = it - trial_basis.begin();
    auto refit = ref_basis.get_iter(it->get_id());
    if (refit == ref_basis.end()) {
      // new basis contains element that are not
      // contained in old basis, => cannot proceed
      std::cerr << "Error! Basis elem not found!\n" << it->get_id().to_string() << std::endl;
      exit(-1);
    } else {
      unsigned int jref = refit - ref_basis.begin();
      jref2jnew[jref] = jnew;
    }
  }

  // fill new tensor with entries from old tensor
  for (auto it_master = trial_basis.begin(); it_master != trial_basis.end(); ++it_master) {
    // get corresponding index in ref_basis
    unsigned int jref = ref_basis.get_dof_index(it_master->get_id());
    int jnew = jref2jnew[jref];
    if (jnew != -1) {  // read slice into ref_slice
      matrix_t ref_slice(Nref, Nref);
      eigen2hdf::load_sparse(file, boost::lexical_cast<std::string>(jref), ref_slice);
      // allocate new slice
      typedef std::shared_ptr<matrix_t> ptr_t;
      ptr_t slice = ptr_t(new matrix_t(N, N));
      // copy relevant entries from ref_slice into slice
      for (int k = 0; k < ref_slice.outerSize(); ++k)
        for (matrix_t::InnerIterator it(ref_slice, k); it; ++it) {
          it.value();
          int iref = it.row();  // row index
          int jref = it.col();  // col index (here it is equal to k)
          it.index();           // inner index, here it is equal to it.row()
          if ((jref2jnew[jref] != -1) && (jref2jnew[iref] != -1)) {
            slice->insert(jref2jnew[iref], jref2jnew[jref]) = it.value();
          }
        }
      tensor.add(slice, jnew);
    }
  }

  matrix_t mass_matrix(N, N);
  matrix_t ref_mass_matrix(Nref, Nref);
  eigen2hdf::load_sparse(file, "mass_matrix", ref_mass_matrix);

  for (int k = 0; k < ref_mass_matrix.outerSize(); ++k) {
    for (matrix_t::InnerIterator it(ref_mass_matrix, k); it; ++it) {
      it.value();
      int iref = it.row();  // row index
      int jref = it.col();  // col index (here it is equal to k)
      it.index();           // inner index, here it is equal to it.row()
      if ((jref2jnew[jref] != -1) && (jref2jnew[iref] != -1)) {
        mass_matrix.insert(jref2jnew[iref], jref2jnew[jref]) = it.value();
      }
    }
  }

  tensor.set_mass_matrix(mass_matrix);

  H5Fclose(file);
}
}  // end namespace boltzmann

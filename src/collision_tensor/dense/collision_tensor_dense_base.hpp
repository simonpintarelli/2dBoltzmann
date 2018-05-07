#pragma once

#include <boost/align/aligned_allocator.hpp>
#include <boost/mpl/at.hpp>
#include <iostream>
#include <unordered_map>
#include "../collision_tensor_galerkin_base.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"
#include "storage/vbcrs_sparsity.hpp"
#include "spectral/basis/toolbox/spectral_basis.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif


namespace boltzmann {
namespace ct_dense {
// ---------------------------------------------------------------------------
class CollisionTensorDenseBase : public CollisionTensorGalerkinBase
{
 protected:
  typedef double numeric_t;
  // 256bit alignment (assumes 64bit float)
  const int _align = VBCRSSparsity<>::align;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> mat_t;
  typedef Eigen::VectorXd vec_t;

 public:
  typedef typename SpectralBasisFactoryKS::basis_type basis_t;

 public:
  CollisionTensorDenseBase(const basis_t& basis)
      : CollisionTensorGalerkinBase(basis)
  {
    /* empty */
  }

  void import_entries_mpishmem(const std::string& filename,
                               int vbcrs_min_blk_size=1,
                               MPI_Comm comm = MPI_COMM_WORLD);

  unsigned long nnz() const;

 protected:
  virtual void import_entries(const std::vector<std::shared_ptr<Eigen::SparseMatrix<double>> >& slices) = 0;

 protected:
  // allocated by MPI_Win_allocate_shared
  double* entries_ = NULL;
  /// inverse of mass matrix
  std::vector<numeric_t, boost::alignment::aligned_allocator<double, 32> > sinv_;
  std::vector<VBCRSSparsity<>> sparsity_patterns_;

  MPI_Comm shmcomm_;
  MPI_Win win_;
};

}  // end namespace ct_dense

}  // end namespace boltzmann

#include <cblas.h>
#include <immintrin.h>
#include <x86intrin.h>
#include <iostream>

#include "aux/eigen2hdf.hpp"
#include "aux/exceptions.h"
#include "collision_tensor_dense_base.hpp"
#include "spectral/utility/mass_matrix.hpp"
#include "collision_tensor/dense/multi_slices_factory.hpp"
#include "collision_tensor/dense/cluster_vbcrs_sparsity.hpp"

namespace boltzmann {
namespace ct_dense {

void CollisionTensorDenseBase::import_entries_mpishmem(const std::string& filename,
                                                       int vbcrs_min_blk_size,
                                                       MPI_Comm comm)
{
  typedef VBCRSSparsity<>::mem_size_t memsize_t;
  typedef long unsigned int size_t;
  static_assert(sizeof(size_t) == 8, "need 64 bit integer");

  // create data layout for dense collision tensors
  typename multi_slices_factory::container_t multi_slices;
  multi_slices_factory::create(multi_slices, get_basis());
  // initialize a vector of vbcrs_sparsity
  std::vector<VBCRSSparsity<>> vbcrs_sparsity_patterns_tmp(2 * K_ - 1);
  int i = 0;
  for (auto& mslice : multi_slices) {
    auto& vbcrs = vbcrs_sparsity_patterns_tmp[i++];
    vbcrs.init(mslice.second.data(), K_);
    size_t msize = vbcrs.memsize();
  }

  std::vector<VBCRSSparsity<>> vbcrs_sparsity_patterns;
  cluster_vbcrs_sparsity::cluster(vbcrs_sparsity_patterns,
                                  vbcrs_sparsity_patterns_tmp,
                                  multi_slices,
                                  get_basis(),
                                  vbcrs_min_blk_size);

  // compute number of required bytes to store collision tensor entries
  memsize_t memsize = 0;
  for (unsigned int i = 0; i < vbcrs_sparsity_patterns.size(); ++i) {
    memsize += vbcrs_sparsity_patterns[i].memsize();
  }

  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm_);
  int shmrank;
  MPI_Comm_rank(shmcomm_, &shmrank);
  sparsity_patterns_ = vbcrs_sparsity_patterns;

  typedef Eigen::SparseMatrix<double> sparse_matrix_t;
  std::vector<std::shared_ptr<sparse_matrix_t>> slices;

  // memsize + VBCRSSparsity<>::align => make sure that pointer can be aligned
  double* mem = NULL;
  MPI_Win_allocate_shared((memsize + VBCRSSparsity<>::align) * sizeof(double),
                          sizeof(double),
                          MPI_INFO_NULL,
                          shmcomm_,
                          &mem,
                          &win_);
  // get pointer
  MPI_Aint sz;
  int dispunit;
  double* baseptr = NULL;
  MPI_Win_shared_query(win_, MPI_PROC_NULL, &sz, &dispunit, &baseptr);
  MPI_Barrier(shmcomm_);
  // align pointer to 32 byte boundary
  entries_ = (double*)(((char*)baseptr) + (32 - size_t(baseptr) % 32));

  if (shmrank == 0) {
    int mpi_shm_comm_size;
    MPI_Comm_size(shmcomm_, &mpi_shm_comm_size);
    std::cout << "CollisionTensorDenseBase mpi shared mem group size: " << mpi_shm_comm_size  << "\n";
    std::cout << "N: " << N_ << "\n";
    slices.resize(N_);
    // load collision tensor from disk and import entries
    hid_t h5f = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    BAssertThrow(h5f >= 0, "failed to open collision_tensor file" + std::string(filename));
    for (int i = 0; i < N_; ++i) {
      // std::cout << "loading slice " << i << "\n";
      slices[i] = std::shared_ptr<sparse_matrix_t>(new sparse_matrix_t(N_, N_));
      eigen2hdf::load_sparse(h5f, boost::lexical_cast<std::string>(i), *slices[i]);
    }
    H5Fclose(h5f);
    // import slices
  }
  this->import_entries(slices);
  slices.clear();

  MPI_Barrier(shmcomm_);
}

unsigned long CollisionTensorDenseBase::nnz() const
{
  unsigned long nnz = 0;
  for(const auto& sp : sparsity_patterns_) {
    nnz += sp.nnz();
  }
  return nnz;
}

}  // ct_dense
}  // boltzmann

#include "collision_tensor_zlastAM_storage.hpp"
#include <Eigen/Dense>
#include "aux/eigen2hdf.hpp"
#include "aux/exceptions.h"
#include "collision_tensor_dense_base.hpp"
#include "spectral/utility/mass_matrix.hpp"


namespace boltzmann {
namespace ct_dense {

CollisionTensorZLastAMstorage::CollisionTensorZLastAMstorage(const basis_t& basis, int bufsize)
    : CollisionTensorDenseBase(basis)
    , bufsize_(bufsize)
{
  static_assert(EIGEN_DEFAULT_ALIGN_BYTES == 32, "EIGEN_DEFAULT_ALIGN_BYTES is not 32");
  static_assert(EIGEN_MAX_ALIGN_BYTES == 32, "EIGEN_MAX_ALIGN_BYTES is not 32");
}


void CollisionTensorZLastAMstorage::resize_buffer(int bufsize)
{
  int padded_size = (*voffsets_.rbegin());
  if (buffer_.size() / padded_size != bufsize) {
    buffer_.resize(bufsize*padded_size);
    bufsize_ = bufsize;
    std::fill(buffer_.begin(), buffer_.end(), 0.0);
  }
}


int CollisionTensorZLastAMstorage::get_buffer_size() const
{
  int padded_size = (*voffsets_.rbegin());
  int bufsize = buffer_.size() / padded_size;
  return bufsize;
}


void
CollisionTensorZLastAMstorage::import_entries(
    const std::vector<std::shared_ptr<Eigen::SparseMatrix<double>>>& slices)
{
  auto aligned_size = [](size_t size) {
    return size + (VBCRSSparsity<>::align - (size % VBCRSSparsity<>::align));
  };

  static_assert(!mat_t::IsRowMajor, "the underlying sparsity pattern assumes col-major sub-blocks");

  /*   It makes no difference to use Eigen::Unaliged, but using Eigen::Aligned
   *   will trigger an assert, if the given pointer is not properly aligned
   */
  // typedef Eigen::Map<mat_t, Eigen::Aligned, Eigen::OuterStride<> > vmat_t;
  typedef Eigen::Map<mat_t, Eigen::Unaligned, Eigen::OuterStride<>> vmat_t;

  typedef VBCRSSparsity<>::mem_size_t memsize_t;

  double* data = entries_;
  BOOST_ASSERT(data != NULL);

  memsize_t memoffset = 0;
  unsigned int z_offset = 0;
  // iterate over (l,t) pairs (block slices)
  for (unsigned int lt = 0; lt < sparsity_patterns_.size(); ++lt) {
    const auto& vbcrs = sparsity_patterns_[lt];
    for (unsigned int z = 0; z < vbcrs.dimz(); ++z) {
      unsigned int slice_id = z_offset + z;
      // iterate over block rows
      for (unsigned int i = 0; i < vbcrs.nblock_rows(); ++i) {
        const size_t row_offset = vbcrs.get_row_info(i).offset;
        const size_t row_extent = vbcrs.get_row_info(i).extent;
        for (unsigned int blkidx = vbcrs.row_begin(i); blkidx < vbcrs.row_end(i); ++blkidx) {
          memsize_t padded_row_size = aligned_size(row_extent);
          auto& subblock = vbcrs.get_block(blkidx);
          const size_t col_offset = subblock.offset;
          const size_t col_extent = subblock.extent;
          memsize_t sblocksize = padded_row_size * col_extent;
          // copy blocks for all k-values to the given (l1,t1), (l2,t2) values.
          if (slices.size() > 0) {
            // only proc = 0 reads tensor entries
            vmat_t mat_dst(
                data + memoffset, row_extent, col_extent, Eigen::OuterStride<>(padded_row_size));
            mat_dst = (*slices[slice_id]).block(row_offset, col_offset, row_extent, col_extent);
          }
          // increment data pointer
          memoffset += sblocksize;
        }
      }
    }
    // update z-offset
    z_offset += vbcrs.dimz();
  }
  BOOST_ASSERT(z_offset == N_);

  // the vector *voffsets_* contains the offsets for the aligned (padded) input
  // vector offset_map_ is used to map unpadded offsets to the padded offsets
  // (the padded offsets are not stored in the sparsity structure)
  auto& vbcrs = sparsity_patterns_[0];
  voffsets_ = std::vector<int>(vbcrs.nblock_rows() + 1);
  // build voffsets_
  int voffset = 0;
  int unpadded_offset = 0;
  for (int i = 0; i < vbcrs.nblock_rows(); ++i) {
    int row_extent = vbcrs.get_row_info(i).extent;
    voffsets_[i] = voffset;
    // this is needed because vbcrs stores the unpadded col_offset
    offset_map_[unpadded_offset] = voffset;
    voffset += aligned_size(row_extent);
    unpadded_offset += row_extent;
  }
  voffsets_[vbcrs.nblock_rows()] = voffset;
  offset_map_[unpadded_offset] = voffset;

  this->resize_buffer(bufsize_);
  // buffer_.setZero();
}

}  // namespace ct_dense
}  // namespace boltzmann

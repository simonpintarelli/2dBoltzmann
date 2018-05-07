#pragma once

#include "collision_tensor_dense_base.hpp"
#include "collision_tensor_zlastAM_storage.hpp"
#include <cblas.h>
#include <immintrin.h>
#include <x86intrin.h>
#include <Eigen/Dense>
#include <boost/align/aligned_allocator.hpp>
#include <boost/assert.hpp>
#include <vector>


namespace boltzmann {
namespace ct_dense {

class CollisionTensorZLastAM : public CollisionTensorZLastAMstorage
{
 public:
  typedef std::vector<double, boost::alignment::aligned_allocator<double, 32>> aligned_vector;

 public:
  CollisionTensorZLastAM(const basis_t &basis, int bufsize)
      : CollisionTensorZLastAMstorage(basis, bufsize)
  {
    /* empty */
  }

  /**
   * @param[out] out   output array (col major, unpadded)
   * @param[in]  in    input array (col major, padded)
   * @param[in]  imax  apply slices up to and including index imax, if imax<=0, apply all
   */
  template <typename DERIVED1, typename DERIVED2>
  void apply(Eigen::DenseBase<DERIVED1> &out,
             const Eigen::DenseBase<DERIVED2> &in,
             int imax=0) const;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};

template <typename DERIVED1, typename DERIVED2>
void
CollisionTensorZLastAM::apply(Eigen::DenseBase<DERIVED1> &out,
                              const Eigen::DenseBase<DERIVED2> &in,
                              int imax) const
{
  auto aligned_size = [](size_t size) {
    return size + (VBCRSSparsity<>::align - (size % VBCRSSparsity<>::align));
  };

  if (imax <= 0) {
    imax = N_;
  }

  typedef VBCRSSparsity<>::mem_size_t memsize_t;

  static_assert(!DERIVED1::IsRowMajor, "expect column major");
  static_assert(!DERIVED2::IsRowMajor, "expect column major");

  BOOST_ASSERT(out.cols() == in.cols());
  BOOST_ASSERT(buffer_.size() >= in.cols() * in.rows());

  double *buf = buffer_.data();
  const double *data = entries_;
  BOOST_ASSERT(buffer_.size() >= *(voffsets_.rbegin()));
  int Npadded = *(voffsets_.rbegin());

  BOOST_ASSERT(out.rows() == N_);
  BOOST_ASSERT(in.rows() == Npadded);

  memsize_t memoffset = 0;
  unsigned int z_offset = 0;
  for (unsigned int lt = 0; lt < sparsity_patterns_.size(); ++lt) {
    const auto &vbcrs = sparsity_patterns_[lt];

    for (unsigned int z = 0; z < vbcrs.dimz(); ++z) {
      // set buffer to zero (cblas_dgemv will add results (beta=1.0))
      std::fill(buffer_.begin(), buffer_.end(), 0.0);
      for (unsigned int i = 0; i < vbcrs.nblock_rows(); ++i) {
        size_t row_offset = vbcrs.get_row_info(i).offset;
        size_t row_extent = vbcrs.get_row_info(i).extent;
        if(row_offset > imax) continue;

        memsize_t padded_row_size = aligned_size(row_extent);
        memsize_t padded_row_offset = voffsets_[i];
        // set buf in current row to zero
        for (unsigned int blkidx = vbcrs.row_begin(i); blkidx < vbcrs.row_end(i); ++blkidx) {
          auto &subblock = vbcrs.get_block(blkidx);
          size_t col_offset = subblock.offset;
          if(col_offset > imax)
            continue;

          size_t col_extent = subblock.extent;
          // get padded column offset
          auto it = offset_map_.find(col_offset);
          BOOST_ASSERT(it != offset_map_.end());
          size_t padded_col_offset = it->second;
          const double *x1 = in.derived().data() + padded_col_offset;
          memsize_t sblocksize = padded_row_size * col_extent;
          const double *M = data + memoffset;
          cblas_dgemm(CblasColMajor,
                      CblasNoTrans,            /* trans A */
                      CblasNoTrans,            /* trans B */
                      row_extent,              /* m */
                      in.cols(),               /* n */
                      col_extent,              /* k */
                      1.0,                     /* alpha */
                      M,                       /* A* */
                      padded_row_size,         /* lda */
                      x1,                      /* B* */
                      Npadded,                 /* ldb */
                      1.0,                     /* beta */
                      buf + padded_row_offset, /* C */
                      Npadded /* ldc */);

          // increment data pointer
          memoffset += sblocksize;
        }  // end iterate over local row
      }    // end iterate over rows
      // compute dot product
      size_t slice_id = z_offset + z;
      double sinv = Sinv_.diagonal()(slice_id);
      for (int kk = 0; kk < in.cols(); ++kk) {
        out(slice_id, kk) =
            sinv *
            cblas_ddot(Npadded, in.derived().data() + kk * Npadded, 1, buf + kk * Npadded, 1);
      }
    }  // end iterate over z
    // update z-offset
    z_offset += vbcrs.dimz();
  }
}

}  // namespace ct_dense
}  // namespace boltzmann

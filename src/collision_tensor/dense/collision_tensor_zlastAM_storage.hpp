#pragma once

#include <cblas.h>
#include <immintrin.h>
#include <x86intrin.h>
#include <Eigen/Dense>
#include <boost/align/aligned_allocator.hpp>
#include <boost/assert.hpp>
#include <vector>

#include "collision_tensor_dense_base.hpp"

namespace boltzmann {
namespace ct_dense {

class CollisionTensorZLastAMstorage : public CollisionTensorDenseBase
{
 public:
  typedef std::vector<double, boost::alignment::aligned_allocator<double, 32>> aligned_vector;

 public:
  /**
   * @brief constructor ....
   *
   * @param basis Polar-Laguerre basis
   * @param bufsize number of input vectors (for example local spatial DoFs)
   *
   */
  CollisionTensorZLastAMstorage(const basis_t &basis, int bufsize);

  virtual void import_entries(
      const std::vector<std::shared_ptr<Eigen::SparseMatrix<double>>> &slices);

  template <typename DERIVED1, typename DERIVED2>
  inline void pad(Eigen::DenseBase<DERIVED1> &out, const Eigen::DenseBase<DERIVED2> &in) const;

  template <typename DERIVED1, typename DERIVED2>
  inline void unpad(Eigen::DenseBase<DERIVED1> &out, const Eigen::DenseBase<DERIVED2> &in) const;

  inline int padded_vector_length() const;

  /**
   * @brief resize buffer
   *
   * @param number of input vectors (for example local spatial DoFs)
   */
  void resize_buffer(int bufsize);

  int get_buffer_size() const;

 protected:
  int bufsize_;
  mutable aligned_vector buffer_;

  //@{
  ///  helpers for padded input vectors
  std::vector<int> voffsets_;
  std::unordered_map<int, int> offset_map_;
  //@}
};


template <typename DERIVED1, typename DERIVED2>
inline void
CollisionTensorZLastAMstorage::pad(Eigen::DenseBase<DERIVED1> &out,
                                   const Eigen::DenseBase<DERIVED2> &in) const
{
  BOOST_ASSERT(voffsets_.size() > 0);
  static_assert(!DERIVED1::IsRowMajor, "expecting row major storage");
  static_assert(!DERIVED2::IsRowMajor, "expecting row major storage");

  int nvectors = in.cols();
  int Npadded = this->padded_vector_length();
  const double *ptr_in = in.derived().data();
  auto &vbcrs = sparsity_patterns_[0];
  if (out.cols() != in.cols() || out.rows() != Npadded) {
    out.derived().resize(Npadded, in.cols());
    // argh, need to set to zero
    out.derived().setZero();
  }
  double *ptr_out = out.derived().data();

  for (int k = 0; k < nvectors; ++k) {
    int unpadded_offset = 0;
    for (int i = 0; i < voffsets_.size() - 1; ++i) {
      int row_extent = vbcrs.get_row_info(i).extent;
      std::copy(ptr_in + unpadded_offset + k * N_,
                ptr_in + unpadded_offset + k * N_ + row_extent,
                ptr_out + k * Npadded + voffsets_[i]);
      unpadded_offset += row_extent;
    }
  }
}


template <typename DERIVED1, typename DERIVED2>
inline void
CollisionTensorZLastAMstorage::unpad(Eigen::DenseBase<DERIVED1> &out,
                                     const Eigen::DenseBase<DERIVED2> &in) const
{
  BOOST_ASSERT(voffsets_.size() > 0);
  static_assert(!DERIVED1::IsRowMajor, "expecting row major storage");
  static_assert(!DERIVED2::IsRowMajor, "expecting row major storage");

  int nvectors = in.cols();
  out.resize(N_, nvectors);
  int Npadded = this->padded_vector_length();
  const double *ptr_in = in.derived().data();
  double *ptr_out = out.derived().data();
  auto &vbcrs = sparsity_patterns_[0];

  for (int k = 0; k < nvectors; ++k) {
    int unpadded_offset = 0;
    for (int i = 0; i < voffsets_.size() - 1; ++i) {
      int row_extent = vbcrs.get_row_info(i).extent;
      std::copy(ptr_in + voffsets_[i] + k * Npadded,
                ptr_in + voffsets_[i] + k * Npadded + row_extent,
                ptr_out + k * N_ + unpadded_offset);
      unpadded_offset += row_extent;
    }
  }
}


inline int
CollisionTensorZLastAMstorage::padded_vector_length() const
{
  return (*voffsets_.rbegin());
}


}  // namespace ct_dense
}  // namespace boltzmann

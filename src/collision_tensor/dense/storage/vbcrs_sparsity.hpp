#pragma once

#include <boost/assert.hpp>
#include <iostream>
#include <map>
#include <ostream>
#include <tuple>
#include <vector>

#include "enum/enum.hpp"
#include "aux/filtered_range.hpp"

namespace boltzmann {
namespace ct_dense {

/**
 * @brief Variable sized block compressed row storage format. E.g. CSR made up of variable sized
 * blocks. This class stores only locations of non-zero blocks and no actual values/entries.
 *
 * @tparam ALIGN Align to ALIGN units (e.g. doubles, floats, ...), this is not alignment in bytes.
 *
 * This is a compressed version of \ref MultiSlice.
 *
 * Notation: y = A*x
 * Thus x_offset, x_extent refer to columns and y_offset, y_extent to rows.
 *
 */
template <int ALIGN = 4>
class VBCRSSparsity
{
 public:
  typedef unsigned int size_t;
  typedef long unsigned int mem_size_t;
  struct block_t
  {
    block_t() {}
    block_t(size_t offset_, size_t extent_)
        : offset(offset_)
        , extent(extent_)
    {
    }
    size_t offset;
    size_t extent;
  };
  typedef int block_idx;

 public:
  /// align subblocks to `align` doubles
  static size_t align;

 public:
  VBCRSSparsity() { /* empty */}

  template <typename INDEX_TYPE, typename SUBBLOCK>
  void init(const std::multimap<INDEX_TYPE, SUBBLOCK>& multi_slices, const int K);

  template <typename INDEX_TYPE, typename SUBBLOCK, typename SUPER_BLOCK>
  void init(const std::multimap<INDEX_TYPE, SUBBLOCK>& multi_slice,
            const std::vector<SUPER_BLOCK>& super_blocks);

  /**
   * @param idx block row index
   *
   * for use in \ref get_block
   *
   * @return begin of row
   */
  size_t row_begin(size_t idx) const;

  /**
   * @param idx block row index
   *
   * for use in \ref get_block
   *
   * @return end of row
   */
  size_t row_end(size_t idx) const;

  /**
   * @param block_idx
   *
   * to iterate over a block row use \ref row_begin, \ref row_end as input
   *
   * @return column offset, column width
   */
  const block_t& get_block(block_idx block_idx) const;

  /**
   * @param idx block row index
   *
   * @return row offset, row height (as struct{offset, extent}).
   */
  const block_t& get_row_info(size_t idx) const;

  /*
   * number of storage units (assuming each block is aligned to 4 units)
   *
   * 4 units because we store 64bit floating point numbers and align to 256 bits (AVX)
   */
  mem_size_t memsize() const;
  size_t dimz() const;
  /// number of rows / columns
  size_t nblock_rows() const;
  /// total number of blocks
  size_t nblocks() const;
  /// number of nonzero entries
  size_t nnz() const;

  void save(std::ostream& out) const;

 private:
  /// length = number of block rows + 1
  std::vector<block_idx> row_ptr_;
  /// length = number of block rows
  std::vector<block_t> block_rows_;  // row offset, row height (for each block row)
  /// length = nnz
  std::vector<block_t> block_columns_;  // column offset, column width (for each block)
  /// length = nnz+1
  // std::vector<mem_size_t> mem_offsets_;  // memory offset
  mem_size_t memsize_;
  size_t dimz_;
  size_t nnz_;
  bool is_initialized_ = false;
};

template <int ALIGN>
unsigned int VBCRSSparsity<ALIGN>::align = ALIGN;

template <int ALIGN>
template <typename INDEX_TYPE, typename SUBBLOCK>
void
VBCRSSparsity<ALIGN>::init(const std::multimap<INDEX_TYPE, SUBBLOCK>& multi_slice, const int K)
{
  BOOST_ASSERT(is_initialized_ == false);
  block_idx block_count = 0;
  int row_id = 0;
  // size_t row_offset = 0;
  mem_size_t mem_offset = 0;
  nnz_ = 0;
  block_rows_.resize(2 * K - 1);
  row_ptr_.resize(2 * K);
  for (int k = 0; k < K; ++k) {
    for (auto t : {TRIG::COS, TRIG::SIN}) {
      // sin(0*phi) == 0 is of course not contained in the basis
      if (k == 0 && t == TRIG::SIN) continue;
      // row_ptr points to the first block in this row
      row_ptr_[row_id] = block_count;
      auto key = std::make_tuple(k, t);
      if (multi_slice.find(key) == multi_slice.end()) {
        block_rows_[row_id] = block_t(0, 0);
        // this row is empty
      } else {
        // key = (angular index, sin or cos)
        auto range = multi_slice.equal_range(key);
        block_rows_[row_id] = block_t(range.first->second.offset_x, range.first->second.size_x);
        dimz_ = range.first->second.size_z;
        for (auto row_it = range.first; row_it != range.second; ++row_it) {
          auto& sblock = row_it->second;
          block_columns_.push_back(block_t(sblock.offset_y, sblock.size_y));
          // leading dimension (lda) (e.g. row in col-major matrix storage)
          // of a sub-block must be 128bit aligned!
          mem_size_t lsize =
              (sblock.size_x + (align - (sblock.size_x % align))) * sblock.size_y * sblock.size_z;
          nnz_ += sblock.size_x * sblock.size_y * sblock.size_z;
          mem_offset += lsize;
          block_count++;
        }
      }
      row_id++;
      // std::cout << std::get<0>(key) << " " << std::get<1>(key) << "\n";
    }
  }
  row_ptr_[row_id] = block_count;
  // one past last byte
  // mem_offsets_.push_back(mem_offset);
  memsize_ = mem_offset;
  // one past last row
  BOOST_ASSERT(row_id == 2 * K - 1);
  // BOOST_ASSERT(mem_offsets_.size() == block_count + 1);
  BOOST_ASSERT((*row_ptr_.rbegin()) == block_count);
  BOOST_ASSERT(multi_slice.size() == this->nblocks());
  is_initialized_ = true;
}

template <int ALIGN>
template <typename INDEX_TYPE, typename SUBBLOCK, typename SUPER_BLOCK>
void
VBCRSSparsity<ALIGN>::init(const std::multimap<INDEX_TYPE, SUBBLOCK>& multi_slice,
                           const std::vector<SUPER_BLOCK>& super_blocks)
{
  // INDEX_TYPE := tuple<k, t>
  // k: polynomial degree
  // t: enum TRIG (eg. SIN or COS)

  nnz_ = 0;
  block_rows_.resize(super_blocks.size());
  row_ptr_.resize(super_blocks.size() + 1);
  block_idx block_count = 0;
  std::map<size_t, size_t> sblock_offset_to_idx;
  size_t offset = 0;
  for (unsigned int i = 0; i < super_blocks.size(); ++i) {
    const auto& sblock = super_blocks[i];
    size_t loffset = sblock.index_first;
    size_t extent = sblock.extent;
    block_rows_[i] = block_t(loffset, extent);
    offset += extent;
    sblock_offset_to_idx[offset] = i;
  }

  // for each entry in super_blocks find super_block_row_id (row id)
  // and super_block_col_id (column id)
  typedef int row_idx_t;
  typedef int col_idx_t;
  typedef std::pair<row_idx_t, col_idx_t> block_key_t;
  std::set<block_key_t> block_nnz_ids;
  for (const auto& elem : multi_slice) {
    int row_offset = elem.second.offset_x;
    int col_offset = elem.second.offset_y;
    auto it_row = sblock_offset_to_idx.upper_bound(row_offset);
    BOOST_ASSERT(it_row != sblock_offset_to_idx.end());
    auto it_col = sblock_offset_to_idx.upper_bound(col_offset);
    BOOST_ASSERT(it_col != sblock_offset_to_idx.end());
    size_t scol_idx = it_row->second;
    size_t srow_idx = it_col->second;
    block_nnz_ids.insert(std::make_pair(srow_idx, scol_idx));
    dimz_ = elem.second.size_z;
  }

  // for (auto elem : block_nnz_ids) {
  //   std::cout << "block_nnz: " << elem.first << ", " << elem.second << "\n";
  // }

  // todo: fill row_ptr and block_columns_
  mem_size_t mem_offset = 0;
  size_t block_counter = 0;
  for (unsigned int srow_idx = 0; srow_idx < super_blocks.size(); ++srow_idx) {
    row_ptr_[srow_idx] = block_counter;
    size_t row_extent = super_blocks[srow_idx].extent;
    auto range = filtered_range(
        block_nnz_ids.begin(), block_nnz_ids.end(), [srow_idx](const block_key_t& key) {
          return key.first == srow_idx;
        });
    std::vector<block_key_t> elems(std::get<0>(range), std::get<1>(range));
    // debug/test output
    //std::cout << "on block " << srow_idx << " found " << elems.size() << " elements\n";
    for (const block_key_t& elem : elems) {
      //std::cout << elem.first << ", " << elem.second << "\n";
      size_t scol_idx = elem.second;
      const auto& sblock = super_blocks[scol_idx];
      int col_begin = sblock.index_first;
      size_t col_extent = sblock.extent;
      block_columns_.push_back(block_t(col_begin, col_extent));
      block_counter++;
      mem_size_t lsize = (row_extent + (align - (row_extent % align))) * col_extent * dimz_;
      nnz_ += row_extent*col_extent*dimz_;
      mem_offset += lsize;
    }
  }
  // last row_ptr_ points one past the last block
  row_ptr_[super_blocks.size()] = block_counter;
  memsize_ = mem_offset;

  BOOST_ASSERT(block_nnz_ids.size() == this->nblocks());
  is_initialized_ = true;
}

template <int ALIGN>
inline typename VBCRSSparsity<ALIGN>::size_t
VBCRSSparsity<ALIGN>::row_begin(size_t idx) const
{
  BOOST_ASSERT(idx < row_ptr_.size());
  return row_ptr_[idx];
}

template <int ALIGN>
inline typename VBCRSSparsity<ALIGN>::size_t
VBCRSSparsity<ALIGN>::row_end(size_t idx) const
{
  BOOST_ASSERT(idx + 1 < row_ptr_.size());
  return row_ptr_[idx + 1];
}

template <int ALIGN>
inline const typename VBCRSSparsity<ALIGN>::block_t&
VBCRSSparsity<ALIGN>::get_block(block_idx block_idx) const
{
  BOOST_ASSERT(block_idx < block_columns_.size());
  return block_columns_[block_idx];
}

template <int ALIGN>
inline const typename VBCRSSparsity<ALIGN>::block_t&
VBCRSSparsity<ALIGN>::get_row_info(size_t idx) const
{
  BOOST_ASSERT(idx < block_rows_.size());
  return block_rows_[idx];
}

template <int ALIGN>
inline typename VBCRSSparsity<ALIGN>::mem_size_t
VBCRSSparsity<ALIGN>::memsize() const
{
  // return *(mem_offsets_.rbegin());
  return memsize_;
}

template <int ALIGN>
inline typename VBCRSSparsity<ALIGN>::size_t
VBCRSSparsity<ALIGN>::dimz() const
{
  return dimz_;
}

template <int ALIGN>
inline typename VBCRSSparsity<ALIGN>::size_t
VBCRSSparsity<ALIGN>::nblock_rows() const
{
  BOOST_ASSERT(block_rows_.size());
  // there is one pointer past the end, thus -1
  return block_rows_.size();
}


template <int ALIGN>
inline void
VBCRSSparsity<ALIGN>::save(std::ostream& out) const
{
  out << "row_begin col_begin row_extent col_extent\n";
  for(size_t row_idx = 0; row_idx < this->nblock_rows(); ++row_idx) {
    size_t row_offset = get_row_info(row_idx).offset;
    size_t row_extent = get_row_info(row_idx).extent;
    for(size_t ptr = row_begin(row_idx); ptr < row_end(row_idx); ++ptr) {
      size_t col_offset = get_block(ptr).offset;
      size_t col_extent = get_block(ptr).extent;
      out << row_offset << " "
          << col_offset << " "
          << row_extent << " "
          << col_extent << "\n";
    }
  }
}

template <int ALIGN>
inline typename VBCRSSparsity<ALIGN>::size_t
VBCRSSparsity<ALIGN>::nblocks() const
{
  return (*row_ptr_.rbegin());
}


template<int ALIGN>
inline typename VBCRSSparsity<ALIGN>::size_t
VBCRSSparsity<ALIGN>::nnz() const
{
  return nnz_;
}

}  // ct_dense
}  // end namespace boltzmann

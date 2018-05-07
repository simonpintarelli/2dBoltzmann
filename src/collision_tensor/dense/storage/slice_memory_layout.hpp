#pragma once

#include <array>
#include <map>
#include <vector>

namespace boltzmann {

namespace slicememlayout {
/**
 *  @brief Block-CRS memory layout.
 *
 *  Hint: see also \ref MultiSlice
 *
 */
struct Block2D
{
  typedef unsigned int size_type;
  /**
   *
   * @param offsets_
   * @param extents_
   * @param offset    local memory offset
   *
   */
  Block2D(std::array<size_type, 2> offsets_, std::array<size_type, 2> extents_, size_type offset)
      : offsets(offsets_)
      , extents(extents_)
      , local_mem_offset(offset)
  {
    size = offsets[0] * offsets[1];
  }

  std::array<size_type, 2> offsets;
  std::array<size_type, 2> extents;
  size_type local_mem_offset;
  size_type size;
};
}  // slicememlayout

// --------------------------------------------------------------------------------
template <typename BLOCK = slicememlayout::Block2D>
class SliceMemoryLayout
{
 public:
  typedef BLOCK Block;
  typedef typename BLOCK::size_type size_type;

  // protected:

 public:
  typedef Block entry_t;
  typedef std::vector<size_type> vrow_indices_t;
  typedef std::vector<entry_t> vblock_t;

 public:
  SliceMemoryLayout() {}

  template <typename INDEX_TYPE, typename SUBBLOCK>
  void init(const std::multimap<INDEX_TYPE, SUBBLOCK>& multi_slices, const int K);
  size_type size() const { return total_size_; }
  size_type nblocks() const { return blocks_.size(); }
  const vrow_indices_t& row_indices() const { return row_indices_; }
  const vblock_t& blocks() const { return blocks_; }

  std::tuple<size_type, size_type> row_ind(size_type k) const
  {
    return std::make_tuple(row_indices_[k], row_indices_[k + 1]);
  }

 protected:
  // length K+1
  vrow_indices_t row_indices_;
  vblock_t blocks_;
  size_type total_size_ = 0;
  int _align = 2;  // align on 2*sizeof(double) (e.g. 128bit)
};

// -------------------------------------------------------------------------------------
template <typename BLOCK>
template <typename INDEX_TYPE, typename SUBBLOCK>
void
SliceMemoryLayout<BLOCK>::init(const std::multimap<INDEX_TYPE, SUBBLOCK>& multi_slices, const int K)
{
  row_indices_.resize(K + 1);

  typedef SUBBLOCK subblock_t;
  typedef std::array<size_type, 2> vindex_t;

  size_type mem_offset = 0;
  size_type block_counter = 0;
  for (int k = 0; k < K; ++k) {
    auto iterators = multi_slices.equal_range(k);
    // set row_index
    row_indices_[k] = block_counter;
    for (auto it = iterators.first; it != iterators.second; ++it) {
      auto& v = it->second;

      vindex_t offsets = {(unsigned int)v.offset_x, (unsigned int)v.offset_y};
      // no. of rows, columns
      vindex_t extents = {(unsigned int)v.size_x, (unsigned int)v.size_y};
      entry_t entry(offsets, extents, mem_offset);
      blocks_.push_back(std::move(entry));
      // align in v.size_x since this is the leading dimension for col-major storage
      mem_offset += (v.size_x + v.size_x % _align) * v.size_y * v.size_z;
      mem_offset += mem_offset;
      block_counter++;
    }
  }
  row_indices_[K] = block_counter;
  total_size_ = mem_offset;
}

}  // end namespace boltzmann

#pragma once

#include <array>
#include <vector>

#include <blaze/Blaze.h>

namespace boltzmann {
class Slice
{
 public:
  typedef unsigned int size_type;

  typedef blaze::DynamicMatrix<double, blaze::columnMajor> blaze_matrix_t;

 protected:
  struct Block
  {
    Block(std::array<size_type, 2> offsets_, std::array<size_type, 2> extents_)
        : offsets(offsets_)
        , extents(extents_)
    {
      mat.resize(extents[0], extents[1], false);
    }

    std::array<size_type, 2> offsets;
    std::array<size_type, 2> extents;
    blaze_matrix_t mat;
    //    size_type local_mem_offset;
    //    size_type size;
  };

 public:
  typedef Block entry_t;
  typedef std::vector<size_type> vrow_indices_t;
  typedef std::vector<Block> vblock_t;

 public:
  Slice() {}

  template <typename INDEX_TYPE, typename SUBBLOCK>
  void init(const std::multimap<INDEX_TYPE, SUBBLOCK>& multi_slice, const int K);

  //  size_type size() const { return total_size_; }
  size_type nblocks() const { return blocks_.size(); }

  const vrow_indices_t& row_indices() const { return row_indices_; }
  vblock_t& blocks() { return blocks_; }

  std::tuple<size_type, size_type> row_ind(size_type k) const
  {
    return std::make_tuple(row_indices_[k], row_indices_[k + 1]);
  }

 protected:
  // length K+1
  vrow_indices_t row_indices_;
  vblock_t blocks_;
  //  size_type total_size_=0;
};

// -------------------------------------------------------------------------------------
template <typename INDEX_TYPE, typename SUBBLOCK>
void
Slice::init(const std::multimap<INDEX_TYPE, SUBBLOCK>& multi_slice, const int K)
{
  row_indices_.resize(K + 1);

  typedef SUBBLOCK subblock_t;
  typedef std::array<size_type, 2> vindex_t;

  // size_type mem_offset = 0;
  size_type block_counter = 0;
  for (int k = 0; k < K; ++k) {
    auto iterators = multi_slice.equal_range(k);
    // set row_index
    row_indices_[k] = block_counter;
    for (auto it = iterators.first; it != iterators.second; ++it) {
      auto& v = it->second;

      vindex_t offsets = {v.offset_x, v.offset_y};
      vindex_t extents = {v.size_x, v.size_y};

      entry_t entry(offsets, extents);
      blocks_.push_back(std::move(entry));
      // mem_offset += v.size_x * v.size_y;
      block_counter++;
    }
  }
  row_indices_[K] = block_counter;

  // std::cout << "block_counter: " << block_counter << std::endl;;
}

}  // end namespace boltzmann

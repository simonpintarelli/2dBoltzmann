#pragma once

#include "aux/hash_specializations.hpp"
#include "enum/enum.hpp"
#include "subblock.hpp"

#include <boost/lexical_cast.hpp>
#include <map>
#include <unordered_map>
#include <utility>

namespace boltzmann {
namespace ct_dense {

/**
 * @brief Stores blocks with identical angular index (l-index) _row-wise_.  This class
 *        represents some sort of uncompressed block-CRS format for variable sized
 *        sub-blocks.
 *
 *        This class is an intermediate for \ref SliceMemoryLayout.
 */
class MultiSlice
{
 public:
  typedef unsigned int size_type;
  typedef int index_type;

 public:
  typedef SubBlock<index_type> subBlock_t;
  typedef std::tuple<index_type, enum TRIG> key_t;

 protected:
  // index_type: angular index
  // it is a multimap because there may be more than one block, e.g. maximal two blocks
  //
  typedef std::multimap<key_t, subBlock_t> container_t;

 public:
  typedef subBlock_t entry_t;

 public:
  MultiSlice() {}

  MultiSlice(enum TRIG t, index_type l, size_type N)
      : t_(t)
      , l_(l)
      , N_(N)
  { /* empty */
  }

  void add_block(index_type l1,
                 enum TRIG t1,
                 index_type l2,
                 enum TRIG t2,
                 size_type offset_x,
                 size_type offset_y,
                 size_type size_x,
                 size_type size_y,
                 size_type size_z);

  size_type size() const { return size_; }
  size_type nblocks() const { return entries_.size(); }
  const container_t& data() const { return entries_; }
  void finalize();
  std::string get_id() const;

 protected:
  enum TRIG t_;
  int l_;
  size_type N_;
  size_type size_;
  container_t entries_;
};

// ---------------------------------------------------------------------------
inline void
MultiSlice::add_block(index_type l1,
                      enum TRIG t1,
                      index_type l2,
                      enum TRIG t2,
                      size_type offset_x,
                      size_type offset_y,
                      size_type size_x,
                      size_type size_y,
                      size_type size_z)
{
  //  key_t key(l1,l2);
  assert(int(t1) < 2);
  key_t key(l1, t1);

  subBlock_t sb;
  sb.offset_x = offset_x;
  sb.offset_y = offset_y;
  sb.size_x = size_x;
  sb.size_y = size_y;
  sb.size_z = size_z;
  // entries_[key] = sb;
  entries_.insert(std::make_pair(key, sb));
}

// ---------------------------------------------------------------------------
inline void
MultiSlice::finalize()
{
  size_type total_size = 0;
  for (auto& elem : entries_) {
    // test-function dimension (i.e. size_z) is not taken into account
    total_size += elem.second.size();
  }
  size_ = total_size;
}

// ---------------------------------------------------------------------------
inline std::string
MultiSlice::get_id() const
{
  return boost::lexical_cast<std::string>(l_) + " " + boost::lexical_cast<std::string>(t_);
}

}  // ct_dense
}  // end namespace boltzmann

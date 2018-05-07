#pragma once

#include <functional>
#include <vector>
#include "aux/filtered_range.hpp"
#include "collision_tensor/dense/storage/vbcrs_sparsity.hpp"
#include "enum/enum.hpp"
#include "spectral/basis/toolbox/spectral_basis.hpp"

namespace boltzmann {

class cluster_vbcrs_sparsity
{
 public:
  typedef SpectralBasisFactoryKS::elem_t elem_t;
  typedef typename boost::mpl::at_c<typename elem_t::types_t, 0>::type fa_type;
  typename elem_t::Acc::template get<fa_type> fa_accessor;

  /// a block of basis functions
  /// with identical polynomial degree l and trigonometric index t
  struct block_t
  {
    int l;
    enum TRIG t;
    int index_first;
    int index_last;
  };

  /// a cluster of consecutive block_t's
  struct super_block_t
  {
    int extent = 0;
    int index_first = std::numeric_limits<int>::max();
    int index_last = -1;
    std::vector<block_t> elems;
    void insert(const block_t& block)
    {
      elems.push_back(block);
      extent += block.index_last - block.index_first;
      index_first = std::min(block.index_first, index_first);
      index_last = std::max(block.index_last, index_last);
    }
  };

 public:
  template <typename VBSPARSITY_OUT, typename VBSPARSITY_IN, typename BASIS, typename MULTI_SLICES>
  static void cluster(std::vector<VBSPARSITY_OUT>& dst,
                      const std::vector<VBSPARSITY_IN>& src,
                      const MULTI_SLICES multi_slices,
                      const BASIS& basis,
                      int min_blk_size)
  {
    typedef typename BASIS::elem_t elem_t;
    typedef typename boost::mpl::at_c<typename elem_t::types_t, 0>::type fa_type;
    typename elem_t::Acc::template get<fa_type> fa_accessor;

    auto cmp = [&fa_accessor](const elem_t& e, int l, enum TRIG t) {
      auto id = fa_accessor(e).get_id();
      return (id.l == l && TRIG(id.t) == t);
    };

    const int K = spectral::get_K(basis);

    // find blocks with identical (k, t)
    // k: polynomial degree
    // t: SIN/COS
    std::vector<block_t> blocks;
    int offset = 0;
    for (int l = 0; l < K; ++l) {
      for (auto t : {TRIG::COS, TRIG::SIN}) {
        auto range_z =
            filtered_range(basis.begin(), basis.end(), std::bind(cmp, std::placeholders::_1, l, t));
        std::vector<elem_t> elemsz(std::get<0>(range_z), std::get<1>(range_z));
        if (elemsz.size() == 0) continue;
        int size = elemsz.size();
        block_t block = {l, t, offset, offset + size};
        blocks.push_back(block);
        offset += size;
      }
    }

    // cluster blocks, such that number of elements is > min_blk_size
    std::vector<super_block_t> super_blocks;
    while (!blocks.empty()) {
      auto elem = blocks.back();
      int extent = elem.index_last - elem.index_first;
      auto& last_super_block = super_blocks.back();
      if (!super_blocks.empty() && last_super_block.extent < min_blk_size) {
        last_super_block.insert(elem);
      } else {
        super_block_t sblock;
        sblock.insert(elem);
        super_blocks.push_back(sblock);
      }
      // remove last elem
      blocks.pop_back();
    }

    std::sort(super_blocks.begin(),
              super_blocks.end(),
              [](const super_block_t& a, const super_block_t& b) {
                return a.index_first < b.index_first;
              });

    dst.resize(src.size());

    BOOST_ASSERT(src.size() == multi_slices.size());

    int i = 0;
    for (auto& mslice : multi_slices) {
      auto& vbcrs_blocked = dst[i];
      const auto& vbcrs = src[i];
      vbcrs_blocked.init(mslice.second.data(), super_blocks);
      ++i;
    }
  }
};


}  // namespace boltzmann

#pragma once

#include <boost/mpl/at.hpp>
#include <cassert>
#include <cstdio>
#include <functional>
#include <iostream>
#include "aux/filtered_range.hpp"
#include "spectral/basis/toolbox/spectral_basis.hpp"
#include "storage/multi_slice.hpp"


namespace boltzmann {
namespace ct_dense {

class multi_slices_factory
{
 public:
  typedef MultiSlice::index_type index_type;
  // typedef std::tuple<enum TRIG, index_type> key_t;
  /// key: (angular frequency, _sin_ or _cos_)
  typedef std::tuple<index_type, enum TRIG> key_t;
  typedef unsigned int size_type;
  typedef std::map<key_t, MultiSlice> container_t;

 public:
  template <typename BASIS>
  static void create(container_t& data, const BASIS& basis);
};

template <typename BASIS>
void
multi_slices_factory::create(container_t& data, const BASIS& basis)
{
  typedef std::map<key_t, MultiSlice> container_t;
  typedef typename BASIS::elem_t elem_t;
  typedef typename boost::mpl::at_c<typename elem_t::types_t, 0>::type fa_type;
  typedef typename boost::mpl::at_c<typename elem_t::types_t, 1>::type fr_type;
  // typename elem_t::Acc::template get<fr_type> fr_accessor;
  typename elem_t::Acc::template get<fa_type> fa_accessor;

  const index_type L = spectral::get_max_l(basis);
  const size_type N = basis.n_dofs();

  for (auto elem = basis.begin(); elem != basis.end(); elem++) {
    auto ang_elem = fa_accessor(*elem);

    const int l = ang_elem.get_id().l;
    const enum TRIG t = (TRIG)ang_elem.get_id().t;
    // key_t key(t, l);
    key_t key(l, t);

    if (data.find(key) != data.end()) continue;  // this element is already done

    typedef std::vector<std::pair<index_type, index_type> > line_t;
    // line 1 (first line)
    line_t line1;
    for (index_type l1 = 0; l1 < l + 1; ++l1) {
      assert(l - l1 >= 0);
      line1.push_back(std::make_pair(l1, l - l1));
    }
    // line 2 (lower diagonal)
    line_t line2;
    for (index_type l1 = l + 1; l1 <= L; ++l1) {
      line2.push_back(std::make_pair(l1, l1 - l));
    }
    // line 3 (upper diagonal)
    line_t line3;

    if (l > 0) {
      // hint: if l==0, line2 and line3 are the same
      for (index_type l1 = 1; l1 <= L - l; ++l1) {
        line3.push_back(std::make_pair(l1, l1 + l));
      }
    }

    auto cmp = [&fa_accessor](const elem_t& e, index_type l, enum TRIG t) {
      auto id = fa_accessor(e).get_id();
      return (id.l == l && TRIG(id.t) == t);
    };
    // ---------- add lines to data, compute offsets and block sizes ----------
    // find all elements with given (l, t)-values
    auto range_z =
        filtered_range(basis.begin(), basis.end(), std::bind(cmp, std::placeholders::_1, l, t));
    std::vector<elem_t> elemsz(std::get<0>(range_z), std::get<1>(range_z));
    size_type size_z = elemsz.size();

    MultiSlice current_mslice(t, l, N);

    auto add_line = [&](const line_t& line) {
      // process lines
      for (auto pair : line) {
        // l1 <-> row
        const index_type l1 = pair.first;
        // l2 <-> column
        const index_type l2 = pair.second;
        // enum TRIG t1;
        // enum TRIG t2;

        auto add_range = [&](const std::vector<elem_t>& x, const std::vector<elem_t>& y) {
          if (x.size() == 0 || y.size() == 0) return;  // nothing to do
          size_type offset_x = basis.get_dof_index(x.begin()->get_id());
          size_type size_x = basis.get_dof_index(x.rbegin()->get_id()) - offset_x + 1;
          assert(size_x == x.size());
          size_type offset_y = basis.get_dof_index(y.begin()->get_id());
          size_type size_y = basis.get_dof_index(y.rbegin()->get_id()) - offset_y + 1;
          assert(size_y == y.size());
          assert(size_x > 0 && size_y > 0);
          enum TRIG t1 = TRIG(fa_accessor(x[0]).get_id().t);
          enum TRIG t2 = TRIG(fa_accessor(y[0]).get_id().t);
          current_mslice.add_block(l1, t1, l2, t2, offset_x, offset_y, size_x, size_y, size_z);
        };

        if (t == TRIG::COS) {
          // Note: range should be contiguous in ordering of elements (this is not ensured here!!)
          // add range1
          auto range1x = filteredv(
              basis.begin(), basis.end(), std::bind(cmp, std::placeholders::_1, l1, TRIG::COS));
          auto range1y = filteredv(
              basis.begin(), basis.end(), std::bind(cmp, std::placeholders::_1, l2, TRIG::COS));
          // aka add block (l1, cos) x (l2, cos) to multi_slice
          add_range(range1x, range1y);

          auto range2x = filteredv(
              basis.begin(), basis.end(), std::bind(cmp, std::placeholders::_1, l1, TRIG::SIN));
          auto range2y = filteredv(
              basis.begin(), basis.end(), std::bind(cmp, std::placeholders::_1, l2, TRIG::SIN));
          add_range(range2x, range2y);
        } else if (t == TRIG::SIN) {
          // add range1
          auto range1x = filteredv(
              basis.begin(), basis.end(), std::bind(cmp, std::placeholders::_1, l1, TRIG::COS));
          auto range1y = filteredv(
              basis.begin(), basis.end(), std::bind(cmp, std::placeholders::_1, l2, TRIG::SIN));
          add_range(range1x, range1y);
          // add range2
          auto range2x = filteredv(
              basis.begin(), basis.end(), std::bind(cmp, std::placeholders::_1, l1, TRIG::SIN));
          auto range2y = filteredv(
              basis.begin(), basis.end(), std::bind(cmp, std::placeholders::_1, l2, TRIG::COS));
          add_range(range2x, range2y);
        }
      }
    };

    add_line(line1);
    add_line(line2);
    add_line(line3);

    current_mslice.finalize();

#ifdef DEBUG
    std::printf("MultiSlice finished\n\tStats: (l=%3d, t=%3d), size: %6d, blocks: %3d\n",
                l,
                int(t),
                current_mslice.size(),
                current_mslice.nblocks());
#endif
    // insert mslice into map
    data[key] = current_mslice;
  }
}
}  // ct_dense
}  // end namespace boltzmann

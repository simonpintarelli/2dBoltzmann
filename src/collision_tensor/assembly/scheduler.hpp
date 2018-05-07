#pragma once

// system includes ---------------------------------------------------------
#include <algorithm>
#include <array>
#include <list>
#include <set>
#include <vector>
// own includes ------------------------------------------------------------
#include "spectral/basis/spectral_elem.hpp"
#include "spectral/basis/spectral_elem_accessor.hpp"
#include "spectral/basis/spectral_function.hpp"


namespace boltzmann {
/**
 * @brief helper class for load-balancing during collision tensor assembly
 */
class Scheduler
{
 private:
  typedef unsigned int elem_index;

 public:
  typedef std::set<elem_index> local_work_t;

  /// a vector of j-values (pairs) of equal work count
  typedef std::vector<local_work_t> vector_t;

  template <typename SPECTRAL_BASIS>
  vector_t init(const SPECTRAL_BASIS& spectral_basis);
};

template <typename SPECTRAL_BASIS>
typename Scheduler::vector_t
Scheduler::init(const SPECTRAL_BASIS& spectral_basis)
{
  typedef std::pair<elem_index, int> pair_t;
  std::list<pair_t> Ls;

  // accessor
  typedef typename SPECTRAL_BASIS::elem_t elem_t;
  ///
  typedef typename std::tuple_element<0, typename elem_t::container_t>::type angular_elem_t;
  typename elem_t::Acc::template get<angular_elem_t> xir_accessor;

  // loop over basis and collect l-indices
  for (auto it = spectral_basis.begin(); it != spectral_basis.end(); ++it) {
    unsigned int elem_index = it - spectral_basis.begin();
    const auto& xir = xir_accessor(*it);
    int l = xir.get_id().l;
    Ls.push_back(std::make_pair(elem_index, l));
  }

  // sort vec_L in l
  Ls.sort([](const pair_t& p1, const pair_t& p2) { return p1.second < p2.second; });

  vector_t unit_work;
  while (Ls.size() > 1) {
    elem_index tail = Ls.back().first;
    elem_index head = Ls.front().first;
    local_work_t local_work;
    local_work.insert(tail);
    local_work.insert(head);
    unit_work.push_back(local_work);

    Ls.pop_back();
    Ls.pop_front();
  }

  if (Ls.size() == 1) {
    local_work_t local_work;
    elem_index tail = Ls.back().first;
    local_work.insert(tail);
    unit_work.push_back(local_work);
    Ls.pop_back();
  } else if (Ls.size() == 0) {
  } else {
    throw 0;
  }

  return unit_work;
}

}  // end namespace boltzmann

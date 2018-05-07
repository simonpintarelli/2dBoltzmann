#pragma once

#include <boost/iterator/filter_iterator.hpp>
#include <boost/mpl/identity.hpp>

#include <functional>
#include <iterator>
#include <tuple>


namespace boltzmann {

template <typename ITERATOR, typename CMP>
std::tuple<boost::filter_iterator<CMP, ITERATOR>, boost::filter_iterator<CMP, ITERATOR> >
filtered_range(const ITERATOR& begin,
               const typename boost::mpl::identity<ITERATOR>::type& end,
               const CMP& f)
{
  // typedef  boost::filter_iterator<std::function<bool(const ELEM&)>, ITERATOR >
  //   iterator_t;

  return std::make_tuple(boost::make_filter_iterator(f, begin, end),
                         boost::make_filter_iterator(f, end, end));
}

template <typename ITERATOR, typename CMP>
std::vector<typename std::iterator_traits<ITERATOR>::value_type>
filteredv(const ITERATOR& begin,
          const typename boost::mpl::identity<ITERATOR>::type& end,
          const CMP& f)
{
  typedef boost::filter_iterator<CMP, ITERATOR> iterator_t;

  typedef typename std::iterator_traits<ITERATOR>::value_type elem_t;

  std::vector<elem_t> ret(boost::make_filter_iterator(f, begin, end),
                          boost::make_filter_iterator(f, end, end));
  return ret;
}

}  // end namespace boltzmann

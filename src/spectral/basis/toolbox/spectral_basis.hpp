#pragma once

// system includes ------------------------------------------------------------
#include <Eigen/Sparse>
#include <boost/iterator/filter_iterator.hpp>
#include <boost/mpl/identity.hpp>
#include <functional>
#include <type_traits>
#include <iterator>
#include <tuple>

// own includes ------------------------------------------------------------
#include "spectral/basis/spectral_basis.hpp"
#include "spectral/basis/spectral_elem_accessor.hpp"
#include "spectral/basis/spectral_basis_factory_hermite.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"


namespace boltzmann {

namespace spectral {

// ----------------------------------------------------------------------
template <typename BASIS>
unsigned int
get_max_l(const BASIS& basis)
{
  typedef typename std::tuple_element<0, typename BASIS::elem_t::container_t>::type elem_t;
  // radial basis
  typename BASIS::elem_t::Acc::template get<elem_t> get_xir;
  unsigned int maxL = 0;
  for (auto it = basis.begin(); it != basis.end(); ++it) {
    unsigned int l = get_xir(*it).get_id().l;
    if (l > maxL) maxL = l;
  }

  return maxL;
}

// ----------------------------------------------------------------------
/**
 * @Brief Return the max polynomial degree for the Polar-Laguerre basis
 *
 * @param basis
 *
 * @return
 */
template <typename BASIS>
typename std::enable_if<std::is_same<typename BASIS::elem_t::container_t,
                                     typename ::boltzmann::SpectralBasisFactoryKS::elem_t::container_t
                                     >::value,
                        unsigned int>::type
get_max_k(const BASIS& basis)
{
  typedef typename std::tuple_element<1, typename BASIS::elem_t::container_t>::type elem_t;
  typename BASIS::elem_t::Acc::template get<elem_t> get_phi;
  unsigned int maxK = 0;
  for (auto it = basis.begin(); it != basis.end(); ++it) {
    unsigned int k = get_phi(*it).get_id().k;
    if (k > maxK) maxK = k;
  }
  return maxK;
}


// ----------------------------------------------------------------------
/**
 * @Brief Return the max polynomial degree for the Polar-Laguerre basis
 *
 * @param basis
 *
 * @return
 */
template <typename BASIS>
typename std::enable_if<std::is_same<typename BASIS::elem_t::container_t,
                                     typename ::boltzmann::SpectralBasisFactoryKS::elem_t::container_t
                                     >::value,
                        unsigned int>::type
get_K(const BASIS& basis)
{
  typedef typename std::tuple_element<1, typename BASIS::elem_t::container_t>::type elem_t;
  typename BASIS::elem_t::Acc::template get<elem_t> get_phi;
  unsigned int maxK = 0;
  for (auto it = basis.begin(); it != basis.end(); ++it) {
    unsigned int k = get_phi(*it).get_id().k;
    if (k > maxK) maxK = k;
  }
  return maxK+1;
}


// ----------------------------------------------------------------------
/**
 * @Brief Return the max polynomial degree for the Hermite basis
 *
 * @param basis
 *
 * @return
 */
template <typename BASIS>
typename std::enable_if<std::is_same<typename BASIS::elem_t::container_t,
                                     typename ::boltzmann::SpectralBasisFactoryHN::elem_t::container_t
                                     >::value,
                        unsigned int>::type
get_K(const BASIS& basis)
{
  typedef typename std::tuple_element<0, typename BASIS::elem_t::container_t>::type hx_t;
  typedef typename std::tuple_element<1, typename BASIS::elem_t::container_t>::type hy_t;
  typename BASIS::elem_t::Acc::template get<hx_t> get_hx;
  unsigned int maxKx = 0;
  for (auto it = basis.begin(); it != basis.end(); ++it) {
    unsigned int k = get_hx(*it).get_id().k;
    if (k > maxKx) maxKx = k;
  }

  typename BASIS::elem_t::Acc::template get<hy_t> get_hy;
  unsigned int maxKy = 0;
  for (auto it = basis.begin(); it != basis.end(); ++it) {
    unsigned int k = get_hx(*it).get_id().k;
    if (k > maxKy) maxKy = k;
  }

  return std::max(maxKx, maxKy)+1;
}


#ifdef USE_CXX14
/**
 * @brief return boost::filter_iterator with basis functions that have angular frequency \f$ l \f$.
 *
 * @param begin
 * @param end
 * @param l    angular frequency
 *
 * @return
 */
template <typename ITERATOR>
auto
filter_freq(const ITERATOR& begin,
            const typename boost::mpl::identity<ITERATOR>::type& end,
            unsigned int l)
{
  typedef typename std::iterator_traits<ITERATOR>::value_type elem_t;

  typedef typename std::tuple_element<0, typename elem_t::container_t>::type aelem_t;

  auto f = [l](const elem_t& elem) {
    typename elem_t::Acc::template get<aelem_t> get_xir;
    return get_xir(elem).get_id().l == l;
  };

  return std::make_tuple(boost::make_filter_iterator(f, begin, end),
                         boost::make_filter_iterator(f, end, end));
}

/**
 * @brief return boost::filter_iterator with basis functions that total polynomial degree \f$ k \f$.
 *
 * @param begin
 * @param end
 * @param k    polynomial degree
 *
 * @return
 */
template <typename ITERATOR>
auto
filter_deg(const ITERATOR& begin,
           const typename boost::mpl::identity<ITERATOR>::type& end,
           unsigned int k)
{
  typedef typename std::iterator_traits<ITERATOR>::value_type elem_t;

  typedef typename std::tuple_element<1, typename elem_t::container_t>::type aelem_t;

  auto f = [k](const elem_t& elem) {
    typename elem_t::Acc::template get<aelem_t> get_lag;
    return get_lag(elem).get_id().k == k;
  };

  return std::make_tuple(boost::make_filter_iterator(f, begin, end),
                         boost::make_filter_iterator(f, end, end));
}
#endif  // USE_CXX14

}  // end namespace spectral_basis
}  // end namespace boltzmann

#pragma once

#include <deal.II/base/tensor.h>
// system includes ------------------------------------------------------------
#include <boost/mpl/if.hpp>
#include <boost/type_traits.hpp>
#include <complex>


namespace boltzmann {

template <int rank, int dimX>
struct tensor_traits
{
  template <typename NUMBER>
  using value_type = dealii::Tensor<rank, dimX, NUMBER>;
};

template <int dimX>
struct tensor_traits<0, dimX>
{
  template <typename NUMBER>
  using value_type = NUMBER;
};

template <typename F>
struct extract_value_type
{
  typedef typename F::value_type type;
};

// ----------------------------------------------------------------------
// tensor number traits
template <typename T>
struct tensor_number_traits
{
  typedef typename boost::mpl::eval_if_c<boost::is_scalar<T>::value || boost::is_complex<T>::value,
                                         boost::mpl::identity<T>,
                                         extract_value_type<T> >::type type;
};

}  // end namespace boltzmann

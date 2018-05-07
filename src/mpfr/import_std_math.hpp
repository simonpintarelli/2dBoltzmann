#pragma once

#include <boost/mpl/identity.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <type_traits>

#include <cmath>

// own includes ---------------------------------------------------------
#include "traits/type_traits.hpp"

namespace math {

// ----------------------------------------------------------------------
// SQRT
template <typename T>
inline constexpr typename std::enable_if<std::is_floating_point<T>::value, T>::type
sqrt(const T& t)
{
  return std::sqrt(t);
}

template <typename T>
inline constexpr typename std::enable_if<!std::is_floating_point<T>::value, T>::type
sqrt(const T& t)
{
  return boost::multiprecision::sqrt(t);
}

// ----------------------------------------------------------------------
// EXP
template <typename T>
inline constexpr typename std::enable_if<std::is_floating_point<T>::value, T>::type
exp(const T& t)
{
  return std::exp(t);
}

template <typename T>
inline constexpr typename std::enable_if<!std::is_floating_point<T>::value, T>::type
exp(const T& t)
{
  return boost::multiprecision::exp(t);
}

// ----------------------------------------------------------------------
// LOG
template <typename T>
inline constexpr typename std::enable_if<std::is_floating_point<T>::value, T>::type
log(const T& t)
{
  return std::log(t);
}

template <typename T>
inline constexpr typename std::enable_if<!std::is_floating_point<T>::value, T>::type
log(const T& t)
{
  return boost::multiprecision::log(t);
}

// ----------------------------------------------------------------------
// POW
template <typename T>
inline constexpr typename std::enable_if<std::is_floating_point<T>::value, T>::type
pow(const T& base, const typename boost::mpl::identity<T>::type& exp)
{
  return std::pow(base, exp);
}

template <typename T>
inline constexpr typename std::enable_if<!std::is_floating_point<T>::value, T>::type
pow(const T& base, const typename boost::mpl::identity<T>::type& exp)
{
  return boost::multiprecision::pow(base, exp);
}

// ----------------------------------------------------------------------
// ABS
template <typename T>
inline constexpr typename std::enable_if<std::is_floating_point<T>::value, T>::type
abs(const T& t)
{
  return std::abs(t);
}

template <typename T>
inline constexpr typename std::enable_if<!std::is_floating_point<T>::value, T>::type
abs(const T& t)
{
  return boost::multiprecision::abs(t);
}

}  // end namespace math

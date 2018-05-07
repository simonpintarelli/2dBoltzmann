#pragma once

#include <deal.II/base/tensor.h>
#include <type_traits>


namespace dealii {
// ----------------------------------------------------------------------
template <int dim, typename NUMBER>
inline Tensor<1, dim, NUMBER>
outer_product(NUMBER alpha, const Tensor<1, dim, NUMBER>& t)
{
  return alpha * t;
}

// ----------------------------------------------------------------------
template <int dim, typename NUMBER>
Tensor<1, dim, NUMBER>
outer_product(const Tensor<1, dim, NUMBER>& t, NUMBER alpha)
{
  return alpha * t;
}

// ----------------------------------------------------------------------
template <int dim, typename NUMBER>
inline Tensor<1, dim, NUMBER>
outer_product(const Tensor<1, dim, NUMBER>& t, const Tensor<0, dim, NUMBER>& alpha)
{
  return t * alpha;
}

// ----------------------------------------------------------------------
template <int dim, typename NUMBER>
Tensor<1, dim, NUMBER> inline outer_product(const Tensor<0, dim, NUMBER>& alpha,
                                            const Tensor<1, dim, NUMBER>& t)
{
  return t * alpha;
}

// ----------------------------------------------------------------------
template <int dim, typename NUMBER>
inline void outer_product(Tensor<1, dim, NUMBER>& dst,
                          const Tensor<0, dim, NUMBER> alpha,
                          const Tensor<1, dim, NUMBER>& t)
{
  // hack to extract alpha as a number...

  dst = NUMBER(alpha) * t;
}

template <int dim, typename NUMBER>
inline void outer_product(Tensor<0, dim, NUMBER>& dst,
                          const Tensor<0, dim, NUMBER> alpha,
                          const Tensor<0, dim, NUMBER>& t)
{
  dst = alpha * t;
}

#if (DEAL_II_VERSION_MAJOR >= 8 && DEAL_II_VERSION_MINOR <= 3)
// ----------------------------------------------------------------------
template <int dim, typename NUMBER>
inline void outer_product(Tensor<1, dim, NUMBER>& dst,
                          const Tensor<1, dim, NUMBER>& t,
                          const Tensor<0, dim, NUMBER>& alpha)
{
  dst = alpha[0] * t;
}

// ----------------------------------------------------------------------
template <int dim, class NUMBER>
inline NUMBER
scalar_product(const Tensor<1, dim, NUMBER>& t1, const Tensor<1, dim, NUMBER>& t2)
{
  NUMBER sum = 0;
  for (int i = 0; i < dim; ++i) sum += t1[i] * t2[i];

  return sum;
}

// ----------------------------------------------------------------------
template <int dim, class NUMBER>
inline NUMBER
scalar_product(const Tensor<0, dim, NUMBER>& t1, const Tensor<0, dim, NUMBER>& t2)
{
  return t1 * t2;
}
#endif  // DEAL_II_VERSION_MAJOR >= 8 && DEAL_II_VERSION_MINOR <= 3

// ----------------------------------------------------------------------
template <class NUMBER>
inline typename std::enable_if<std::is_scalar<NUMBER>::value, NUMBER>::type
scalar_product(const NUMBER& t1, const NUMBER& t2)
{
  return t1 * t2;
}

}  // end namespace dealii

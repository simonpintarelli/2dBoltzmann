#pragma once

// system includes ---------------------------------------------------------
#include <cmath>
#include <complex>
#include <iostream>
#include <map>
#include <vector>

// own includes ------------------------------------------------------------
#include "enum/enum.hpp"
#include "gain_cache.hpp"
#include "spectral/basis/spectral_elem.hpp"
#include "spectral/basis/spectral_elem_accessor.hpp"
#include "traits/type_traits.hpp"

namespace boltzmann {
namespace collision_tensor_assembly {

template <typename AngularBasisElem,
          typename RadialBasisElem,
          typename WEIGHT,
          COLLISION_KERNEL KERNEL = MAXWELLIAN>
class GainEvaluatorComplex
{
 private:
  typedef typename numeric_super_type<typename AngularBasisElem::numeric_t,
                                      typename RadialBasisElem::numeric_t>::type numeric_t;

  typedef SpectralElem<numeric_t, AngularBasisElem, RadialBasisElem> elem_t;
  typedef typename elem_t::id_t elem_id_t;
  typedef WEIGHT weight_t;

  typedef std::unordered_map<std::tuple<int, int>, double> gain_cache_t;

 public:
  GainEvaluatorComplex(const elem_t& elem_, const weight_t& weight_, const int nqpts = 81);

  /**
   *
   * @param r0     |v+v*|/2
   * @param r      |v-v*|/2
   *
   * @return
   */
  double compute(double r0, double r);

 private:
  const elem_t& elem;
  const weight_t& weight;
  /// #quad. points
  const int npts;
  /// internal variables
  gain_cache_t cache;
  // std::map<typename elem_t::id_t, double> cache;
};

// ----------------------------------------------------------------------
template <typename AngularBasisElem,
          typename RadialBasisElem,
          typename WEIGHT,
          COLLISION_KERNEL KERNEL>
GainEvaluatorComplex<AngularBasisElem, RadialBasisElem, WEIGHT, KERNEL>::GainEvaluatorComplex(
    const elem_t& elem_, const weight_t& weight_, const int nqpts)
    : elem(elem_)
    , weight(weight_)
    , npts(nqpts + (nqpts + 1) % 2) /* must be odd */
{
  /* empty */
}

// ----------------------------------------------------------------------
template <typename AngularBasisElem,
          typename RadialBasisElem,
          typename WEIGHT,
          COLLISION_KERNEL KERNEL>
inline double
GainEvaluatorComplex<AngularBasisElem, RadialBasisElem, WEIGHT, KERNEL>::compute(double r0,
                                                                                 double r)
{
  /* to avoid branching problems in std::arg, npts must be odd! */
  assert(npts % 2 == 1);
  typedef unsigned long long key_t;
  const double FUZZY = 1e8;
  auto key = std::make_tuple(key_t(r0 * FUZZY), key_t(r * FUZZY));
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  } else {
    typename elem_t::Acc::template get<RadialBasisElem> R_getter;
    typename elem_t::Acc::template get<AngularBasisElem> X_getter;
    // evaluate
    const auto& radf = R_getter(elem);
    const auto& angf = X_getter(elem);
    const double exponent = weight.exponent() + radf.w();
    // w = h
    const double w = 2 * numbers::PI / (1.0 * npts);
    double val = 0;
    for (int i = 0; i < npts; ++i) {
      const double t = 2 * numbers::PI * i / (1.0 * npts);
      const double x = r0 + std::cos(t) * r;
      const double y = std::sin(t) * r;
      const double rw = std::sqrt(x * x + y * y);
      // avoid problems with branch cut => npts must be odd
      double phi = std::atan2(y, x);
      /// exp -l => cos(l ...)
      val += std::real(angf.evaluate(phi)) * radf.evaluate(rw) * weight.evaluate(rw, exponent) * w;
    }
    cache[key] = val;
    return val;
  }
}

}  // end namespce collision_tensor_assembly
}  // end namespace boltzmann

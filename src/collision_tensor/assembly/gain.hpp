#pragma once

// system includes ---------------------------------------------------------
#include <cmath>
#include <iostream>
#include <map>
#include <vector>
// own includes ------------------------------------------------------------
#include "gain_cache.hpp"
#include "spectral/basis/spectral_elem.hpp"
#include "spectral/basis/spectral_elem_accessor.hpp"
#include "base/numbers.hpp"

namespace boltzmann {
namespace collision_tensor_assembly {

/**
 *  @brief Evaluator for the inner integral \f$ \mathcal{I}^+(v, v_*) \f$
 *
 *  Compatible with sperarable kernels of the form
 *  \f$ B( | v-v_*|, \cos(\theta)) = |v-v_*|^\lambda b(\cos \theta) \f$
 */
template <typename SPECTRAL_BASIS>
class GainEvaluator
{
 private:
  typedef typename SPECTRAL_BASIS::elem_t elem_t;
  typedef typename std::tuple_element<0, typename elem_t::container_t>::type angular_elem_t;
  typedef typename std::tuple_element<1, typename elem_t::container_t>::type radial_elem_t;

 public:
  GainEvaluator(const elem_t& elem_, const double w = 0, const int nqpts = 81);

  /**
   *
   * Computes \f$ \int_{0}^{2 \pi} \Psi_{k,l} (\mathbf{v}') \mathrm{d} \sigma \f$
   *
   * @param r0   \f$|v+v*|/2 \f$
   * @param r    \f$|v-v*|/2 \f$
   *
   * @return
   */
  double compute(const double r0, const double r);

  /**
   * @brief show infos about the entry cache
   *
   */
  void print_info() const;

 private:
  const elem_t& elem;
  /// inner product weight exp(-w*r^2), zero by default
  const double w;
  /// #quad. points
  const int npts;
  /// Entry cache
  GainCache cache;
  int l;
};

// ----------------------------------------------------------------------
template <typename SPECTRAL_BASIS>
GainEvaluator<SPECTRAL_BASIS>::GainEvaluator(const elem_t& elem_, const double w_, const int nqpts)
    : elem(elem_)
    , w(w_)
    , npts(nqpts + (nqpts + 1) % 2) /* must be odd */
{
  // get l
  this->l = boost::fusion::at_key<angular_elem_t>(elem_.get_id()).l;
}

// ----------------------------------------------------------------------
template <typename SPECTRAL_BASIS>
inline double
GainEvaluator<SPECTRAL_BASIS>::compute(const double r0, const double r)
{
  typedef unsigned long long key_t;
  /* to avoid branching problems in std::arg, npts must be odd! */
  assert(npts % 2 == 1);
  const double FUZZY = 1e8;
  auto key = std::make_tuple(key_t(r0 * FUZZY), key_t(r * FUZZY));
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  } else {
    typename elem_t::Acc::template get<radial_elem_t> R_getter;
    // evaluate
    const auto& radf = R_getter(elem);
    const double exponent = w + radf.w();
    // w = h
    const double h = 2 * numbers::PI / (1.0 * npts);
    double val = 0.;
    for (int i = 0; i < npts; ++i) {
      const double t = i * h;
      const double x = r0 + r * std::cos(t);
      const double y = r * std::sin(t);
      const double rw2 = x * x + y * y;
      const double rw = std::sqrt(rw2);
      // avoid problems with branch cut => npts must be odd
      const double phi = std::atan2(y, x);
      val += std::cos(this->l * phi) * radf.evaluate(rw) * std::exp(-rw2 * exponent) * h;
    }
    cache[key] = val;
    return val;
  }
}

// ----------------------------------------------------------------------
template <typename SPECTRAL_BASIS>
void
GainEvaluator<SPECTRAL_BASIS>::print_info() const
{
  std::cout << " --- Entry Cache Info --- \n"
            << "\tCache size: " << cache.size()
            //            << "\t#Buckets: " << cache.bucket_count()
            << "\n\n";
  // for (unsigned int i = 0; i < cache.bucket_count(); ++i) {
  //   std::cout << "bucket[" << i << "].size = " << cache.bucket_size(i)
  //             << std::endl;
  // }
}

}  // end namespce collision_tensor_assembly
}  // end namespace boltzmann

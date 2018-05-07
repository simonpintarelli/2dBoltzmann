#pragma once

#include <Eigen/Core>
#include <deal.II/base/index_set.h>
#include <cmath>

#include "base/numbers.hpp"
#include "enum/enum.hpp"
#include "quadrature/qmidpoint.hpp"
#include "quadrature/quad_handler1d.hpp"
#include "quadrature/tensor_product_quadrature.hpp"
#include "spectral/basis/spectral_elem_accessor.hpp"
#include "spectral/basis/toolbox/spectral_basis.hpp"


namespace boltzmann {
class Momentum
{
 private:
  typedef Eigen::Vector2d value_t;

 public:
  Momentum() {}

  template <typename SPECTRAL_BASIS>
  Momentum(const SPECTRAL_BASIS& spectral_basis)
  {
    init(spectral_basis);
  }

  template <typename SPECTRAL_BASIS>
  void init(const SPECTRAL_BASIS& spectral_basis);

  /**
   * @brief compute momentum,
   *
   * @param dst
   * @param src
   * @param count #physical dofs
   */
  template <typename NUMERIC_T>
  void compute(NUMERIC_T* dstx, NUMERIC_T* dsty, const NUMERIC_T* src, int count) const;

  template <typename VEC_IN, typename VEC_OUT, typename INDEXER>
  void compute(VEC_OUT& dstx,
               VEC_OUT& dsty,
               const VEC_IN& src,
               const dealii::IndexSet& relevant_dofs,
               const INDEXER& indexer) const;

  template <typename NUMERIC_T>
  value_t compute(const NUMERIC_T* src) const;

 private:
  typedef std::pair<unsigned int, value_t> entry_t;

 private:
  unsigned int n_velo_dofs;
  std::vector<entry_t> contributions;

 public:
  const std::vector<entry_t>& entries() const { return contributions; }
};

// ------------------------------------------------------------
template <typename SPECTRAL_BASIS>
void
Momentum::init(const SPECTRAL_BASIS& spectral_basis)
{
  n_velo_dofs = spectral_basis.n_dofs();

  typedef typename std::tuple_element<1, typename SPECTRAL_BASIS::elem_t::container_t>::type
      radial_elem_t;

  typedef typename std::tuple_element<0, typename SPECTRAL_BASIS::elem_t::container_t>::type
      angular_elem_t;

  auto& QR_handler = QuadHandler<MaxwellQuadrature>::GetInstance();
  typedef QuadAdaptor<MaxwellQuadrature> qr_adapt;

  unsigned int nqR = spectral::get_max_k(spectral_basis) + 2;
  Eigen::VectorXd pts_qR(nqR);
  Eigen::VectorXd wts_qR(nqR);

  qr_adapt::apply(pts_qR.data(), wts_qR.data(), QR_handler.get(nqR), 0.5);

  typedef typename SPECTRAL_BASIS::elem_t elem_t;
  for (auto it = spectral_basis.begin(); it != spectral_basis.end(); ++it) {
    typename elem_t::Acc::template get<radial_elem_t> getter;
    typename elem_t::Acc::template get<angular_elem_t> geta;

    const auto& rr = getter(*it);

    if (geta(*it).get_id().l != 1) continue;

    // make sure we are using the right basis
    const double w = rr.w();
    if (abs(0.5 - w) > 1e-10) throw std::runtime_error("Error: wrong weight in Momentum::init");

    value_t sum;
    sum[0] = 0.0;
    sum[1] = 0.0;
    if (geta(*it).get_id().t == TRIG::SIN) {
      // sin
      for (unsigned int q = 0; q < pts_qR.size(); ++q) {
        const double r = pts_qR[q];
        sum[TRIG::SIN] += rr.evaluate(r) * r * wts_qR[q];
      }
    } else if (geta(*it).get_id().t == TRIG::COS) {
      for (unsigned int q = 0; q < pts_qR.size(); ++q) {
        const double r = pts_qR[q];
        sum[TRIG::COS] += rr.evaluate(r) * r * wts_qR[q];
      }
    }

    const unsigned int i = it - spectral_basis.begin();
    contributions.push_back(std::make_pair(i, numbers::PI * sum));
  }
}

// // ------------------------------------------------------------
template <typename NUMERIC_T>
void
Momentum::compute(NUMERIC_T* dstx, NUMERIC_T* dsty, const NUMERIC_T* src, int count) const
{
#pragma omp parallel for
  for (int i = 0; i < count; ++i) {
    const NUMERIC_T* local_src = src + i * n_velo_dofs;
    value_t sum;
    sum[0] = 0;
    sum[1] = 0;
    for (unsigned int j = 0; j < contributions.size(); ++j) {
      sum += contributions[j].second * local_src[contributions[j].first];
    }
    dstx[i] = sum[0];
    dsty[i] = sum[1];
  }
}

// ------------------------------------------------------------
template <typename NUMERIC_T>
Momentum::value_t
Momentum::compute(const NUMERIC_T* src) const
{
  value_t sum;
  sum[0] = 0;
  sum[1] = 0;
  for (unsigned int j = 0; j < contributions.size(); ++j) {
    sum += contributions[j].second * src[contributions[j].first];
  }
  return sum;
}

// // ------------------------------------------------------------
template <typename VEC_IN, typename VEC_OUT, typename INDEXER>
void
Momentum::compute(VEC_OUT& dstx,
                  VEC_OUT& dsty,
                  const VEC_IN& src,
                  const dealii::IndexSet& relevant_dofs,
                  const INDEXER& indexer) const
{
#pragma omp parallel for
  for (unsigned int ix = 0; ix < relevant_dofs.size(); ++ix) {
    if (relevant_dofs.is_element(ix)) {
      value_t sum;
      sum[0] = 0;
      sum[1] = 0;
      for (unsigned int j = 0; j < contributions.size(); ++j) {
        unsigned int jx = contributions[j].first;
        sum += contributions[j].second * src[indexer.to_global(ix, jx)];
      }
      dstx[ix] = sum[0];
      dsty[ix] = sum[1];
    }
  }
}

}  // end namespace boltzmann

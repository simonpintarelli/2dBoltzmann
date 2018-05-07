#pragma once

#include <cmath>

#include "base/numbers.hpp"
#include "enum/enum.hpp"
#include "moment_base.hpp"
#include "quadrature/quad_handler1d.hpp"
#include "quadrature/tensor_product_quadrature.hpp"
#include "spectral/basis/spectral_elem_accessor.hpp"
#include "spectral/basis/toolbox/spectral_basis.hpp"

namespace boltzmann {

class Mass : public MomentBase
{
 public:
  Mass() {}

  template <typename SPECTRAL_BASIS>
  Mass(const SPECTRAL_BASIS& spectral_basis)
  {
    init(spectral_basis);
  }

  template <typename SPECTRAL_BASIS>
  void init(const SPECTRAL_BASIS& spectral_basis);

  /**
   * @brief compute density,
   *
   * @param dst
   * @param src
   * @param count #physical dofs
   */
  void compute(double* dst, const double* src, unsigned int count = 0) const;

  double compute(const double* src) const;

  template <typename VEC_IN, typename VEC_OUT, typename INDEXER>
  void compute(VEC_OUT& dst,
               const VEC_IN& src,
               const dealii::IndexSet& relevant_phys_dofs,
               const INDEXER& indexer) const
  {
    // some g++ cannot see this function inherited from MomentBase
    MomentBase::compute(dst, src, relevant_phys_dofs, indexer);
  }

 private:
  using MomentBase::entry_t;
  using MomentBase::n_velo_dofs;
  using MomentBase::contributions;
};

// ------------------------------------------------------------
template <typename SPECTRAL_BASIS>
void
Mass::init(const SPECTRAL_BASIS& spectral_basis)
{
  typedef typename std::tuple_element<1, typename SPECTRAL_BASIS::elem_t::container_t>::type
      radial_elem_t;

  typedef typename std::tuple_element<0, typename SPECTRAL_BASIS::elem_t::container_t>::type
      angular_elem_t;

  auto& QR_handler = QuadHandler<MaxwellQuadrature>::GetInstance();
  typedef QuadAdaptor<MaxwellQuadrature> qr_adapt;

  unsigned int nqR = spectral::get_max_k(spectral_basis) + 2;
  std::vector<double> pts_qR(nqR);
  std::vector<double> wts_qR(nqR);

  n_velo_dofs = spectral_basis.n_dofs();

  qr_adapt::apply(pts_qR.data(), wts_qR.data(), QR_handler.get(nqR), 0.5);

  typedef typename SPECTRAL_BASIS::elem_t elem_t;
  for (auto it = spectral_basis.begin(); it != spectral_basis.end(); ++it) {
    typename elem_t::Acc::template get<radial_elem_t> getter;
    typename elem_t::Acc::template get<angular_elem_t> geta;

    if (geta(*it).get_id().l != 0) continue;

    const auto& rr = getter(*it);
    const double w = rr.w();

    // make sure we are using the right basis
    if (abs(0.5 - w) > 1e-10) throw std::runtime_error("Error: wrong element weight in Mass::init");

    // TODO: magic numbers
    double sum = 0;
    for (unsigned int q = 0; q < pts_qR.size(); ++q) {
      const double r = pts_qR[q];
      sum += rr.evaluate(r) * wts_qR[q];
    }
    const unsigned int i = it - spectral_basis.begin();
    contributions.push_back(std::make_pair(i, 2 * numbers::PI * sum));
  }
}

// ------------------------------------------------------------
void
Mass::compute(double* dst, const double* src, unsigned int count) const
{
  typedef double NUMERIC_T;
#pragma omp parallel for
  for (unsigned int i = 0; i < count; ++i) {
    const NUMERIC_T* local_src = src + i * n_velo_dofs;
    double sum = 0;
    for (unsigned int j = 0; j < contributions.size(); ++j) {
      sum += contributions[j].second * local_src[contributions[j].first];
    }
    dst[i] = sum;
  }
}

// ------------------------------------------------------------
double
Mass::compute(const double* src) const
{
  typedef double NUMERIC_T;
  NUMERIC_T m = NUMERIC_T(0);
  for (unsigned int i = 0; i < contributions.size(); ++i) {
    m += contributions[i].second * src[contributions[i].first];
  }

  return m;
}
}  // end namespace boltzmann

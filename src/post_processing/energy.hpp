#pragma once

#include <deal.II/base/numbers.h>
// // system includes --------------------------------------------------------
#include <cmath>

#include "base/numbers.hpp"
#include "enum/enum.hpp"
#include "moment_base.hpp"
#include "quadrature/qmidpoint.hpp"
#include "quadrature/quad_handler1d.hpp"
#include "quadrature/quadrature_handler.hpp"
#include "spectral/basis/spectral_elem_accessor.hpp"
#include "spectral/basis/toolbox/spectral_basis.hpp"

namespace boltzmann {

class Energy : public MomentBase
{
 public:
  Energy() {}

  // ------------------------------------------------------------
  template <typename SPECTRAL_BASIS>
  Energy(const SPECTRAL_BASIS& spectral_basis)
  {
    init(spectral_basis);
  }

  // ------------------------------------------------------------
  template <typename SPECTRAL_BASIS>
  void init(const SPECTRAL_BASIS& spectral_basis);

  /**
   * @brief compute energy
   *
   * @param dst
   * @param src
   * @param count #physical dofs
   */
  void compute(double* dst, const double* src, int count) const;

  // ------------------------------------------------------------
  double compute(const double* src) const;

  // ------------------------------------------------------------
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
Energy::init(const SPECTRAL_BASIS& spectral_basis)
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

    if (geta(*it).get_id().l != 0) continue;

    const auto& rr = getter(*it);
    const double w = rr.w();

    if (std::abs(w - 0.5) > 1e-10)
      throw std::runtime_error("Error: wrong element weight in Energy::init");

    double sum = 0;
    for (unsigned int q = 0; q < pts_qR.size(); ++q) {
      const double r = pts_qR[q];
      sum += rr.evaluate(r) * r * r * wts_qR[q];
    }
    const unsigned int i = it - spectral_basis.begin();
    contributions.push_back(std::make_pair(i, 2 * numbers::PI * sum));
  }
}

// ------------------------------------------------------------
void
Energy::compute(double* dst, const double* src, int count) const
{
#pragma omp parallel for
  for (int i = 0; i < count; ++i) {
    const double* local_src = src + i * n_velo_dofs;
    double sum = 0;
    for (unsigned int j = 0; j < contributions.size(); ++j) {
      sum += contributions[j].second * local_src[contributions[j].first];
    }
    dst[i] = sum;
  }
}

// ------------------------------------------------------------
double
Energy::compute(const double* src) const
{
  double sum = 0.0;
  for (unsigned int j = 0; j < contributions.size(); ++j) {
    sum += contributions[j].second * src[contributions[j].first];
  }
  return sum;
}

}  // end namespace boltzmann

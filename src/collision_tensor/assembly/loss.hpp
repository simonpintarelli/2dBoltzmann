#pragma once

#include <map>
#include "aux/simple_sparse_matrix.hpp"
#include "base/numbers.hpp"
#include "quadrature/qmaxwell.hpp"
#include "spectral/basis/spectral_basis_dimension_accessor.hpp"

namespace boltzmann {
namespace collision_tensor_assembly {

template <typename SPECTRAL_BASIS>
class Loss
{
 private:
  typedef SimpleSparseMatrix<double> entry_map_t;

  typedef typename SPECTRAL_BASIS::elem_t elem_t;
  typedef typename std::tuple_element<0, typename elem_t::container_t>::type angular_elem_t;
  typedef typename std::tuple_element<1, typename elem_t::container_t>::type radial_elem_t;

 public:
  void init(const SPECTRAL_BASIS* trial_basis, const entry_map_t* s0);

  double compute(int j, int j1, int j2) const;

 private:
  typedef typename radial_elem_t::id_t key_t;
  // Radial overlap Integrals
  std::map<key_t, double> rho_phi_k2;
  const SPECTRAL_BASIS* basis_;
  const entry_map_t* s0_; /* test_basis, trial_basis overlap */
};

template <typename SPECTRAL_BASIS>
void
Loss<SPECTRAL_BASIS>::init(const SPECTRAL_BASIS* trial_basis, const entry_map_t* s0)
{
  // set pointers to basis and mass matrix entries
  basis_ = trial_basis;
  s0_ = s0;

  typename SPECTRAL_BASIS::DimAcc::template get_vec<radial_elem_t> get_basis;
  const auto& rr = get_basis(*trial_basis);

  typedef double beta_key;
  typedef std::shared_ptr<QMaxwell> qptr_t;
  std::map<beta_key, qptr_t> quad_map;
  for (auto it = rr.begin(); it != rr.end(); ++it) {
    const double w = it->w();
    auto qit = quad_map.find(w);
    if (qit == quad_map.end()) {
      /// TODO: hardcoded number of quadrature points
      quad_map[w] = qptr_t(new QMaxwell(w, 100));
    }
  }

  for (unsigned int i = 0; i < rr.size(); ++i) {
    const auto& b = rr[i];
    double val = 0;
    const auto& quad = *(quad_map[b.w()]);
    for (unsigned int q = 0; q < quad.size(); ++q) {
      val += b.evaluate(quad.pts(q)) * quad.wts(q);
    }
    rho_phi_k2[b.get_id()] = 2 * numbers::PI * val;
  }
}

// ------------------------------------------------------------
template <typename SPECTRAL_BASIS>
inline double
Loss<SPECTRAL_BASIS>::compute(int j, int j1, int j2) const
{
  double sj1j;
  auto it = s0_->find(std::make_pair(j, j1));
  if (it == s0_->end())
    return 0.;
  else
    sj1j = it->val;
  const auto& id2 = basis_->get_elem(j2).get_id();
  const auto& idk2 = boost::fusion::at_key<radial_elem_t>(id2);
  if (boost::fusion::at_key<angular_elem_t>(id2).l != 0) return 0.;
  auto itm = rho_phi_k2.find(idk2);
  if (itm == rho_phi_k2.end()) return 0.;
  double rho2 = itm->second;
  return sj1j * rho2;
}
}  // end collision_tensor_assembly
}  // end boltzmann

#pragma once

// system includes -----------------------------------------------
#include <omp.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <algorithm>
#include <boost/lexical_cast.hpp>
#include <boost/multi_array.hpp>
#include <functional>
#include <iomanip>
#include <iostream>
// own includes --------------------------------------------------
#include "aux/message.hpp"
#include "aux/timer.hpp"
#include "collision_tensor/assembly/gain.hpp"
#include "collision_tensor/assembly/loss.hpp"
#include "collision_tensor/assembly/scheduler.hpp"
#include "enum/enum.hpp"
#include "matrix/assembly/velocity_var_form.hpp"
#include "quadrature/quadrature_handler.hpp"
#include "spectral/basis/toolbox/spectral_basis.hpp"

// #include "collision_tensor/assembly/outer.hpp"
#include "spectral/laguerren_ks.hpp"
#include "spectral/pl_radial_eval.hpp"


namespace boltzmann {
namespace collision_tensor_assembly {

namespace local_ {
class SparseMatrixWrapper
{
 public:
  SparseMatrixWrapper(unsigned int n)
      : matrix_(n, n)
  { /* empty */ }

  void insert(unsigned int i, unsigned int j, double v)
  {
    constexpr const double tol = 1e-11;
    if (std::abs(v) > tol) matrix_.insert(i, j) = v;
  }

  const Eigen::SparseMatrix<double>& get() { return matrix_; }

 private:
  Eigen::SparseMatrix<double> matrix_;
};

// ----------------------------------------------------------------------
template <typename S, typename M>
void
make_mass_matrix(M& m, const S& s)
{
  for (auto it = s.begin(); it != s.end(); ++it) {
    const int j1 = it->row;
    const int j2 = it->col;
    const double val = it->val;
    m.insert(j1, j2) = val;
  }
  m.makeCompressed();
}

}  // end namespace local_

// ----------------------------------------------------------------------
class CollisionOperatorBase
{
 public:
  CollisionOperatorBase() { /* empty */}

 protected:
  template <typename SPECTRAL_BASIS, typename QUAD>
  void init(const SPECTRAL_BASIS& test_basis,
            const SPECTRAL_BASIS& trial_basis,
            const QUAD& quad,
            const double beta = 2.0);

  template <typename EXPORTER>
  void export_mass_matrix(EXPORTER& exporter, std::size_t n);

 protected:
  typedef boost::multi_array<double, 2> array_t;
  /// shared across threads
  array_t basis_functions;
  VelocityVarForm<2> var_form;

  std::vector<std::size_t> Loffsets;
};

// ----------------------------------------------------------------------
template <typename SPECTRAL_BASIS, typename QUAD>
void
CollisionOperatorBase::init(const SPECTRAL_BASIS& test_basis,
                            const SPECTRAL_BASIS& trial_basis,
                            const QUAD& quad,
                            const double beta)
{
  const unsigned int n_dofs = trial_basis.n_dofs();
  const unsigned int n_quad_points = quad.size();

  typedef typename SPECTRAL_BASIS::elem_t elem_t;

  // angular basis
  typedef typename std::tuple_element<0, typename SPECTRAL_BASIS::elem_t::container_t>::type
      angular_elem_t;
  typename elem_t::Acc::template get<angular_elem_t> acc_ang;

  unsigned int K = spectral::get_max_k(trial_basis);
  LaguerreNKS<double> L_rad(K);
  PLRadialEval<decltype(L_rad), elem_t> L_poly_eval(L_rad);

  unsigned int nqa = quad.dims()[0];
  unsigned int nqr = quad.dims()[1];

  std::vector<double> pts_r(nqr);
  for (unsigned int i = 0; i < nqr; ++i) {
    pts_r[i] = quad.pts(i)[1];
  }

  L_rad.compute(pts_r.data(), nqr, 1 / beta);

  basis_functions.resize(boost::extents[n_dofs][n_quad_points]);

  for (auto elem_it = trial_basis.begin(); elem_it < trial_basis.end(); ++elem_it) {
    unsigned int idx = elem_it - trial_basis.begin();
    // TODO: use LaguerreNKS for evaluation of basis functions
    for (unsigned int iqa = 0; iqa < nqa; ++iqa) {
      const double phi = quad.pts(iqa * nqr)[0];
      const double fa = acc_ang(*elem_it).evaluate(phi);
      for (unsigned int iqr = 0; iqr < nqr; ++iqr) {
        unsigned int q = iqa * nqr + iqr;
        const double r = quad.pts(iqr)[1];
        basis_functions[idx][q] =
            L_poly_eval(*elem_it, iqr) * fa * (quad.wts(q) * std::exp(r * r / beta));
      }
    }
  }

  var_form.init(test_basis, trial_basis, beta);

  // compute l-index offsets for basis
  unsigned int L = spectral::get_max_l(trial_basis);

  Loffsets.resize(L + 2);
  std::vector<unsigned int> Lvalues;

  typedef typename SPECTRAL_BASIS::elem_t elem_t;
  typedef typename std::tuple_element<0, typename elem_t::container_t>::type angular_elem_t;
  typename elem_t::Acc::template get<angular_elem_t> get_xir;
  for_each(trial_basis.begin(), trial_basis.end(), [&](const elem_t& e) {
    Lvalues.push_back(get_xir(e).get_id().l);
  });
  for (unsigned int i = 0; i < L + 2; ++i) {
    Loffsets[i] = std::lower_bound(Lvalues.begin(), Lvalues.end(), i) - Lvalues.begin();
  }
}

// ----------------------------------------------------------------------
template <typename EXPORTER>
void
CollisionOperatorBase::export_mass_matrix(EXPORTER& exporter, std::size_t n)
{
  Eigen::SparseMatrix<double> M(n, n);
  local_::make_mass_matrix(M, var_form.get_s0());
  exporter.write_mass_matrix(M);
}

// ----------------------------------------------------------------------
template <typename BASIS_TYPE, enum KERNEL_TYPE, typename QANGLE, typename QRADIAL>
class CollisionOperator { /* abstract template */ };

// ----------------------------------------------------------------------
// Implementation:

}  // end namespace collision_tensor_assembly
}  // end namespace boltzmann

// Maxwellian molecules
#include "impl/ct_factory_maxwellian.hxx"
// general variable hard spheres kernel
#include "impl/ct_factory_vhs.hxx"

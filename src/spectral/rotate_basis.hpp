#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <iostream>
#include <utility>
#include <vector>

#include "aux/filtered_range.hpp"
#include "enum/enum.hpp"
#include "spectral/basis/toolbox/spectral_basis.hpp"


namespace boltzmann {

// ----------------------------------------------------------------------
template <typename SPECTRAL_BASIS>
class RotateBasis
{
 private:
  typedef SPECTRAL_BASIS basis_t;
  typedef typename basis_t::index_t index_t;

 public:
  RotateBasis(const SPECTRAL_BASIS& basis)
      : basis_(basis)
  {
    this->init();
  }

  void init();

  /**
   * @brief rotate in counter clockwise direction
   *
   * @param out
   * @param in
   * @param phi angle
   * @param L number of repetitions
   */
  template <typename NUMERIC>
  void apply(NUMERIC* out, const NUMERIC* in, const double phi, const int L = 1) const;

 private:
  const basis_t& basis_;
  Eigen::SparseMatrix<double> R_;

  // internal variable
  mutable double theta_;

  typedef int a_freq;  // angular frequency
  std::unordered_map<a_freq, std::vector<std::pair<index_t, index_t> > > v_pairs_;

  // those elements who do not depend on the angular index `l` need just to be copied
  std::vector<index_t> v_copy_;

  int l_max;
};

// ----------------------------------------------------------------------
template <typename SPECTRAL_BASIS>
void
RotateBasis<SPECTRAL_BASIS>::init()
{
  int N = basis_.n_dofs();
  R_.resize(N, N);
  l_max = spectral::get_max_l(basis_);

  typedef typename basis_t::elem_t elem_t;
  typedef typename basis_t::index_t index_t;

  typedef typename boost::mpl::at_c<typename elem_t::types_t, 1>::type radial_elem_t;
  typedef typename boost::mpl::at_c<typename elem_t::types_t, 0>::type ang_elem_t;

  typename elem_t::Acc::template get<radial_elem_t> get_rad;
  typename elem_t::Acc::template get<ang_elem_t> get_ang;

  {
    // im cos(0*phi) elements do not change, append their
    // indices to the v_copy_ array.
    std::function<bool(const elem_t&)> pred = [&](const elem_t& e) {
      return (get_ang(e).get_id().l == 0);
    };
    auto range = filtered_range(basis_.begin(), basis_.end(), pred);

    auto begin = std::get<0>(range);
    auto end = std::get<1>(range);
    for (auto it = begin; it != end; ++it) {
      v_copy_.push_back(basis_.get_dof_index(it->get_id()));
    }
  }

  for (int l = 1; l <= l_max; ++l) {
    std::function<bool(const elem_t&)> pred = [&](const elem_t& e) {
      return (get_ang(e).get_id().l == l);
    };
    auto range = filtered_range(basis_.begin(), basis_.end(), pred);
    auto begin = std::get<0>(range);
    auto end = std::get<1>(range);

    for (auto it = begin; it != end; ++it) {
      auto ang_elem = get_ang(*it);
      auto rad_elem = get_rad(*it);

      typedef std::vector<std::pair<index_t, index_t> > vec_t;

      if (ang_elem.get_id().t == SIN) {
        index_t i_sin = basis_.get_dof_index(it->get_id());
        ang_elem_t cos_elem(COS, l);
        elem_t elem2(cos_elem, rad_elem);
        index_t i_cos = basis_.get_dof_index(elem2.get_id());

        auto search = v_pairs_.find(l);
        if (search == v_pairs_.end()) {
          v_pairs_[l] = vec_t({std::make_pair(i_cos, i_sin)});
        } else {
          v_pairs_[l].push_back(std::make_pair(i_cos, i_sin));
        }
      }
    }
  }
}

// ----------------------------------------------------------------------
template <typename SPECTRAL_BASIS>
template <typename NUMERIC>
void
RotateBasis<SPECTRAL_BASIS>::apply(NUMERIC* out,
                                   const NUMERIC* in,
                                   const double phi,
                                   const int L) const
{
  typedef NUMERIC numeric_t;
  const unsigned int N = basis_.n_dofs();

  Eigen::VectorXd vsin(l_max + 1);
  Eigen::VectorXd vcos(l_max + 1);

  for (int i = 0; i <= l_max; ++i) {
    vsin[i] = std::sin(-i * phi);
    vcos[i] = std::cos(i * phi);
  }

  //#pragma omp parallel for
  for (int ix = 0; ix < L; ++ix) {
    numeric_t* p_out = out + ix * N;
    const numeric_t* p_in = in + ix * N;

    // walk through v_copy_
    for (auto& i : v_copy_) {
      p_out[i] = p_in[i];
    }

    // apply the rotation
    for (auto it = v_pairs_.begin(); it != v_pairs_.end(); ++it) {
      int l = it->first;

      const double rcos = vcos[l];
      const double rsin = vsin[l];

      for (auto it_v = it->second.begin(); it_v != it->second.end(); ++it_v) {
        index_t i_cos = it_v->first;
        index_t i_sin = it_v->second;

        p_out[i_sin] = p_in[i_sin] * rcos - p_in[i_cos] * rsin;
        p_out[i_cos] = p_in[i_sin] * rsin + p_in[i_cos] * rcos;
      }
    }
  }
}

}  // end namespace

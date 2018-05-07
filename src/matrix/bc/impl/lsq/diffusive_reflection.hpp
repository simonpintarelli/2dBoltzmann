#pragma once

#include <boost/math/constants/constants.hpp>
#include <cmath>
#include "../outflow_helper.hpp"
#include "lsq_bc_base.hpp"
#include "quadrature/qhermitew.hpp"


namespace boltzmann {
namespace impl_lsq {

class DiffusiveReflection : public LSQ_BC_Base
{
 public:
  typedef Eigen::MatrixXd mat_t;

 public:
  DiffusiveReflection(int K, double vt, double Tw);

  inline void apply(mat_t& dst, const mat_t& src) const;

 private:
  double vt_;
  double Tw_;

  Eigen::MatrixXd Mw_;
  Eigen::MatrixXd D_;
  Eigen::MatrixXd C_;
  boltzmann::impl::outflow_helper outflow_helper_;
};


DiffusiveReflection::DiffusiveReflection(int K, double vt, double Tw)
    : LSQ_BC_Base(K)
    , vt_(vt)
    , Tw_(Tw)
{
  Mw_.resize(K, K);
  D_.resize(K, K);
  C_.resize(K, K);


  double PI = boost::math::constants::pi<double>();

  boltzmann::QHermiteW hermite_quad(1.0, K);
  auto x = hermite_quad.pts();
  auto w = hermite_quad.wts();

  double Tw32 = std::pow(Tw, 1.5);
  double p = std::sqrt(1 / 2 / PI);

  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
      Mw_(i, j) =
          Tw32 * p * std::exp(-(x[i] * x[i] + x[j] * x[j]) / 2 / Tw) * std::sqrt(w[i] * w[j]);
    }
  }
}


inline void
DiffusiveReflection::apply(mat_t& dst, const mat_t& src) const
{
  double rho_p = outflow_helper_.compute(src);

}


}  // impl_lsq
}  // boltzmann

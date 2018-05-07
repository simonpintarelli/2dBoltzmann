#pragma once

#include <Eigen/Dense>
#include <vector>
#include "../flux_worker.hpp"
#include "../outflow_helper.hpp"
#include "aux/eigen2hdf.hpp"
#include "base/numbers.hpp"

namespace boltzmann {
namespace impl_mls {

class DiffusiveReflection : public boltzmann::impl::flux_worker
{
 public:
  /**
   * @param hw Hermite quad. weights (from @see QHermiteW)
   * @param hx Hermite quad. nodes
   * @param vt tangential velocity (moving wall)
   * @param Tw wall temperature
   * @param rho scaling factor
   *
   */
  DiffusiveReflection(const vec_t& hw, const vec_t& hx, double vt, double Tw, double rho = 1.0);

  virtual void apply(mat_t& out,
                     const mat_t& in,
                     const dealii::Point<2>& dummy = dealii::Point<2>(0, 0)) const;

 private:
  boltzmann::impl::outflow_helper outflow_;
  const vec_t w_;
  const vec_t x_;
  double rho_;

  // Eigen::MatrixXd rho_minus_;
  Eigen::MatrixXd Mw_;
};

// --------------------------------------------------------------------------------
DiffusiveReflection::DiffusiveReflection(
    const vec_t& hw, const vec_t& hx, double vt, double Tw, double rho)
    : outflow_(hw, hx)
    , w_(hw)
    , x_(hx)
    , rho_(rho)
{
  unsigned int K = hx.size();
  Mw_.resize(K, K);

  unsigned int khalf = K / 2;

  // normalization factor
  const double fMw = rho_ / std::sqrt(2 * numbers::PI) / std::pow(Tw, 1.5);

  // initialize maxwellian
  for (unsigned int i = 0; i < K; ++i) {
    for (unsigned int j = 0; j < K; ++j) {
      if (i < khalf) {  // inflow boundary
        double x2h = std::pow(x_[i], 2) + std::pow(x_[j] - vt, 2);
        Mw_(i, j) = fMw * std::exp(0.5 * x2h * (-1. / Tw)) * std::sqrt(w_[i] * w_[j]) * x_[i];
      } else {  // outflow
        Mw_(i, j) = 0;
      }
    }
  }

}

// --------------------------------------------------------------------------------
void
DiffusiveReflection::apply(mat_t& out, const mat_t& in, const dealii::Point<2>& dummy) const
{
  int K = out.cols();

  // compute outflow
  double rhom = 0;
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
      rhom += in(i, j) * outflow_.get_y(i) * outflow_.get_x(j);
    }
  }

  int khalf = K / 2;

  out.setZero();

  // outflow
  for (int i = khalf; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
      out(i, j) = in(i, j) * x_[i];
    }
  }

  // inflow M_w
  out += Mw_ * rhom;
}

}  // namespace impl_mls
}  // end namespace boltzmann

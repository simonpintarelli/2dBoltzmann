#include "outflow_helper.hpp"
#include "quadrature/qhermitew.hpp"

using namespace boltzmann::impl;

outflow_helper::outflow_helper(const vec_t& hw, const vec_t& hx)
    : ly_(hw.size())
    , lx_(hw.size())
{
  unsigned int K = hw.size();

  QHermiteW hermite_quad(0.5, K);
  QMaxwell maxwell_quad(0.5, K);

  auto& h2x = hermite_quad.pts();
  auto& h2w = hermite_quad.wts();

  // Note: the Maxwell quadrature weights are not scaled by exp(x^2)!!!
  auto& m2x = maxwell_quad.pts();
  auto& m2w = maxwell_quad.wts();

  lagrange_poly_simple lp(hx.data(), K);

  for (unsigned int i = 0; i < K; ++i) {
    double val = 0;
    for (unsigned int q = 0; q < K; ++q) {
      val += lp.eval(i, h2x[q]) / std::sqrt(hw[i]) *
             std::exp(hx[i] * hx[i] / 2 - h2x[q] * h2x[q] / 2) * h2w[q];
    }
    lx_[i] = val;
  }

  for (unsigned int i = 0; i < K; ++i) {
    double val = 0;
    for (unsigned int q = 0; q < K; ++q) {
      val += lp.eval(i, m2x[q]) / std::sqrt(hw[i]) * (std::exp(hx[i] * hx[i] / 2) * m2w[q]);
    }
    ly_[i] = val;
  }
}

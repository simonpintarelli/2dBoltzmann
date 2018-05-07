#pragma once

#include <array>
#include <quadrature/gauss_legendre_quadrature.hpp>
#include <quadrature/maxwell_quadrature.hpp>
#include <quadrature/tensor_product_quadrature.hpp>


namespace boltzmann {

/**
 * @brief quadrature rule required for in/out-flow DEPRECATED
 *
 */
class quad_half_sphere
{
 public:
  typedef std::array<double, 2> coord_type;
  typedef GaussLegendreQuadrature quad_ang_t;
  typedef MaxwellQuadrature quad_rad_t;

 public:
  quad_half_sphere(unsigned int na, unsigned int nr)
      : quad_(GaussLegendreQuadrature(na), MaxwellQuadrature(nr)) __attribute__((deprecated))
  { /* empty */
  }

  quad_half_sphere(const quad_ang_t& qA, const quad_rad_t& qR)
      : quad_(qA, qR)
  { /* empty */
  }

  /**
   *
   * @param theta_begin build quadrature [theta_begin, theta_begin+pi] (inflow/outflow)
   * @param weight exponential weight
   * @param nx normal vector
   * @param ny normal vector
   * @param vx wall velocity
   * @param vy wall velocity
   */
  void reinit(
      double theta_begin, double weight, double nx, double ny, double vx = 0.0, double vy = 0.0);

  unsigned int size() const { return pts_.size(); }

  /// returns quadrature nodes in polar coordinates
  coord_type pts(unsigned int i) const
  {
    assert(i < pts_.size());
    return pts_[i];
  }

  const coord_type* pts_data() const { return pts_.data(); }
  const coord_type* pts_cart_data() const { return pts_cart_.data(); }
  const double* wts_data() const { return wts_.data(); }

  /// returns quadraturs nodes in cartesian coordinates
  coord_type pts_cart(unsigned int i) const
  {
    assert(i < pts_cart_.size());
    return pts_cart_[i];
  }

  double wts(unsigned int i) const
  {
    assert(i < wts_.size());
    return wts_[i];
  }

 private:
  typedef TensorProductQuadratureC<GaussLegendreQuadrature, MaxwellQuadrature> tensor_quad_t;

 private:
  /// Tensor product of Gauleg Q. on [-1, 1], Maxwell Q. with weight exp(-r^2)
  tensor_quad_t quad_;
  typedef std::complex<double> cdouble;

  /// {phi, r}
  std::vector<coord_type> pts_;
  /// quad. nodes in cartesian coord.
  std::vector<coord_type> pts_cart_;
  /// scaled weights
  std::vector<double> wts_;
};

void
quad_half_sphere::reinit(
    double theta_begin, double weight, double nx, double ny, double vx, double vy)
{
  const auto& ref_pts = quad_.pts();
  const auto& ref_wts = quad_.wts();

  unsigned int npts = quad_.size();
  std::vector<cdouble> ptsc;
  ptsc.resize(npts);
  wts_.resize(npts);
  pts_.resize(npts);
  pts_cart_.resize(npts);
  const auto ii = cdouble(0, 1.0);

  // prepare weights
  std::transform(ref_wts.begin(), ref_wts.end(), wts_.begin(), [&](const double w) {
    return w / 2.0 * numbers::PI / weight;
  });
  // scale pts_, wts_ to given intervals / modify weights accordingly
  std::transform(ref_pts.begin(), ref_pts.end(), ptsc.begin(), [&](const std::array<double, 2>& x) {
    const double phi = x[0];  // phi Gauleg quadrature on [-1,1]
    const double r = x[1];
    return r / std::sqrt(weight) * std::exp(ii * ((phi + 1.) / 2. * numbers::PI + theta_begin));
  });

  // apply to inflow/outflow wrt wall velocity and normal vector,
  // i.e. *translate and rotate*
  const cdouble Rphi = std::exp(ii * std::atan2(ny, nx));
  std::transform(ptsc.begin(), ptsc.end(), ptsc.begin(), [&](const cdouble& x) {
    const double vwn = nx * vx + ny * vy;
    return Rphi * x + vwn * (nx + ii * ny);
  });
  // compute polar coordinates
  std::transform(ptsc.begin(), ptsc.end(), pts_.begin(), [](const cdouble& x) {
    return coord_type({std::arg(x), std::abs(x)});
  });

  // cartesian coordinates
  std::transform(ptsc.begin(), ptsc.end(), pts_cart_.begin(), [](const cdouble& x) {
    return coord_type({std::real(x), std::imag(x)});
  });
}

}  // end namespace boltzmann

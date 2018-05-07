#pragma once

#include <array>
#include <boost/mpl/if.hpp>
#include <vector>
#include <Eigen/Dense>
#include <type_traits>

namespace boltzmann {
template <int DIM>
class Quadrature
{
 public:
  const static unsigned int dim = DIM;
  typedef typename boost::mpl::if_c<DIM == 1, double, std::array<double, DIM> >::type coord_type;

 protected:
  typedef std::vector<coord_type> v_pts_t;
  typedef std::vector<double> v_wts_t;
  typedef Eigen::Map<const Eigen::VectorXd> mvec_t;

 public:
  Quadrature(int size)
      : pts_(size)
      , wts_(size)
  { /* empty */
  }

  Quadrature(const v_pts_t& pts, const v_wts_t& wts)
      : pts_(pts)
      , wts_(wts)
  { /* empty */
  }

  Quadrature()
      : pts_(0)
      , wts_(0)
  { /* empty */
  }

  const coord_type* points_data() const { return pts_.data(); }
  const double* weights_data() const { return wts_.data(); }

  const double& wts(unsigned int i) const { return wts_[i]; }
  const std::vector<double>& wts() const { return wts_; }
  const std::vector<coord_type>& pts() const { return pts_; }
  const coord_type& pts(unsigned int i) const { return pts_[i]; }
  template<int D>
  typename std::enable_if<D == 1, const mvec_t>::type vpts() const
  {
    return Eigen::Map<const Eigen::VectorXd>(pts_.data(), pts_.size());
  }

  template<int D>
  typename std::enable_if<D == 1, mvec_t>::type vwts() const
  {
    return Eigen::Map<const Eigen::VectorXd>(wts_.data(), wts_.size());
  }

  unsigned int size() const { return pts_.size(); }

 protected:
  v_pts_t pts_;
  v_wts_t wts_;
};

} // namespace boltzmann

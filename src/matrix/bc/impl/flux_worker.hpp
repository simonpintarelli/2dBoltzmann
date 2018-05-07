#pragma once

#include <deal.II/base/point.h>
#include <Eigen/Dense>
#include <vector>

namespace boltzmann {
namespace impl {

class flux_worker
{
 protected:
  // typedef std::vector<double> vec_t;
  typedef Eigen::VectorXd vec_t;
  typedef Eigen::MatrixXd mat_t;

 public:
  virtual void apply(mat_t& out,
                     const mat_t& in,
                     const dealii::Point<2>& x = dealii::Point<2>(0, 0)) const = 0;
};

}  // end namespace impl
}  // end namespace boltzmann

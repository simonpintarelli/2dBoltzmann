
// deal.II includes ----------------------------------------------------------------------
#include <deal.II/base/quadrature.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>

// system includes -----------------------------------------------------------------------
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <boost/multi_array.hpp>

#ifndef _L2ERRORS_H_
#define _L2ERRORS_H_

namespace boltzmann {

class Errors
{
 private:
  //  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  //  typedef boost::multi_array<double, 2> matrix_t;
  typedef dealii::DoFHandler<2> dh_t;
  typedef dealii::Vector<double> vec_t;

 public:
  /**
   * Compute L2-distance for tensor-product discretization
   *
   * @param dh DoFHandler
   * @param y1 coefficients
   * @param y2 coefficients
   * @param SN spectral-basis overlap coefficients
   * @param INDEXER map(physical dof, velocity dof) -> global dof
   *
   * @return l2 error
   */
  template <typename VECTOR, typename INDEXER>
  double compute(
      const dh_t& dh, const double* y1, const double* y2, const VECTOR& SN, const INDEXER& indexer);

  /**
   * @brief same as compute, but return also norm of y2 (to compute relative error)
   *
   * @param dh
   * @param y1
   * @param y2
   * @param SN
   * @param indexer
   *
   * @return
   */
  template <typename VECTOR, typename INDEXER>
  std::array<double, 2> compute2(
      const dh_t& dh, const double* y1, const double* y2, const VECTOR& SN, const INDEXER& indexer);

  const vec_t& get_cell_wise_error() const { return cell_wise_error; }

 private:
  dealii::Vector<double> cell_wise_error;
};

template <typename VECTOR, typename INDEXER>
double
Errors::compute(
    const dh_t& dh, const double* y1, const double* y2, const VECTOR& SN, const INDEXER& indexer)
{
  dealii::QGauss<2> quad(2);

  const dealii::UpdateFlags update_flags =
      (dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
  auto& fe = dh.get_fe();
  dealii::FEValues<2> fe_values(fe, quad, update_flags);

  int dofs_per_cell = fe_values.dofs_per_cell;
  int n_qpoints = quad.size();

  // number of velocity dofs
  const int N = SN.size();
  cell_wise_error.reinit(dh.get_triangulation().n_active_cells());
  std::fill(cell_wise_error.begin(), cell_wise_error.end(), 0.0);

  double error = 0;
  std::vector<unsigned int> local_indices(dofs_per_cell);

  typedef Eigen::Map<const Eigen::VectorXd> vvec_t;

  auto vcontrib = [&](vvec_t& v1, vvec_t& v2) {
    double sum = 0;
    for (int i = 0; i < N; ++i) {
      sum += v1(i) * v2(i) * SN[i];
    }
    return sum;
  };

  double global_error = 0;
  for (auto cell : dh.active_cell_iterators()) {
    double cell_error = 0;
    cell->get_dof_indices(local_indices);
    fe_values.reinit(cell);

    double area = 0;
    for (int q = 0; q < n_qpoints; ++q) {
      area += fe_values.JxW(q);
    }

    for (int ix1 = 0; ix1 < dofs_per_cell; ++ix1) {
      // index to global block
      unsigned int ig1 = indexer.to_global(local_indices[ix1], 0);
      vvec_t v1(y1 + ig1, N);
      vvec_t v2(y2 + ig1, N);

      // off diagonal contribution
      for (int ix2 = 0; ix2 < dofs_per_cell; ++ix2) {
        unsigned int ig2 = indexer.to_global(local_indices[ix2], 0);
        vvec_t v1p(y1 + ig2, N);
        vvec_t v2p(y2 + ig2, N);

        double overlap = 0;
        for (int q = 0; q < n_qpoints; ++q) {
          overlap +=
              fe_values.shape_value(ix1, q) * fe_values.shape_value(ix2, q) * fe_values.JxW(q);
        }

        double verr = vcontrib(v1, v1p) + vcontrib(v2, v2p) - 2 * vcontrib(v1, v2p);
        cell_error += overlap * verr;
      }
    }
    cell_wise_error[cell->index()] = cell_error / area;
    global_error += cell_error;
  }

  return global_error;
}

/**
 * Compute L2-distance for tensor-product discretization
 *
 * @param dh DoFHandler
 * @param y1 coefficients
 * @param y2 coefficients
 * @param SN spectral-basis overlap coefficients
 * @param INDEXER map(physical dof, velocity dof) -> global dof
 *
 * @return l2 error |y1-y2|, |y2|
 */
template <typename VECTOR, typename INDEXER>
std::array<double, 2>
Errors::compute2(
    const dh_t& dh, const double* y1, const double* y2, const VECTOR& SN, const INDEXER& indexer)
{
  dealii::QGauss<2> quad(2);

  const dealii::UpdateFlags update_flags =
      (dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
  auto& fe = dh.get_fe();
  dealii::FEValues<2> fe_values(fe, quad, update_flags);

  int dofs_per_cell = fe_values.dofs_per_cell;
  int n_qpoints = quad.size();

  // number of velocity dofs
  const int N = SN.size();
  cell_wise_error.reinit(dh.get_triangulation().n_active_cells());
  std::fill(cell_wise_error.begin(), cell_wise_error.end(), 0.0);

  double error = 0;
  double norm = 0;
  std::vector<unsigned int> local_indices(dofs_per_cell);

  typedef Eigen::Map<const Eigen::VectorXd> vvec_t;

  auto vcontrib = [&](vvec_t& v1, vvec_t& v2) {
    double sum = 0;
    for (int i = 0; i < N; ++i) {
      sum += v1(i) * v2(i) * SN[i];
    }
    return sum;
  };

  double global_error = 0;
  for (auto cell : dh.active_cell_iterators()) {
    double cell_error = 0;
    cell->get_dof_indices(local_indices);
    fe_values.reinit(cell);

    double area = 0;
    for (int q = 0; q < n_qpoints; ++q) {
      area += fe_values.JxW(q);
    }

    for (int ix1 = 0; ix1 < dofs_per_cell; ++ix1) {
      // index to global block
      unsigned int ig1 = indexer.to_global(local_indices[ix1], 0);
      vvec_t v1(y1 + ig1, N);
      vvec_t v2(y2 + ig1, N);

      // off diagonal contribution
      for (int ix2 = 0; ix2 < dofs_per_cell; ++ix2) {
        unsigned int ig2 = indexer.to_global(local_indices[ix2], 0);
        vvec_t v1p(y1 + ig2, N);
        vvec_t v2p(y2 + ig2, N);

        double overlap = 0;
        for (int q = 0; q < n_qpoints; ++q) {
          overlap +=
              fe_values.shape_value(ix1, q) * fe_values.shape_value(ix2, q) * fe_values.JxW(q);
        }

        double verr = vcontrib(v1, v1p) + vcontrib(v2, v2p) - 2 * vcontrib(v1, v2p);
        cell_error += overlap * verr;
      }
    }

    // compute |y2|
    for (int ix = 0; ix < dofs_per_cell; ++ix) {
      unsigned int ig = indexer.to_global(local_indices[ix], 0);
      double sum = 0;
      for (int q = 0; q < n_qpoints; ++q) {
        const double f = fe_values.shape_value(ix, q);
        sum += f * f * fe_values.JxW(q);
      }
      vvec_t v2(y1 + ig, N);
      sum *= vcontrib(v2, v2);
      norm += sum;
    }

    cell_wise_error[cell->index()] = cell_error / area;
    global_error += cell_error;
  }
  return std::array<double, 2>({global_error, norm});
}

template <typename DH, typename VECTOR>
double
l2norm(const DH& dh, const VECTOR& vec)
{
  assert(dh.n_dofs() == vec.size());
  dealii::QGauss<2> quad(2);

  const dealii::UpdateFlags update_flags =
      (dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
  auto& fe = dh.get_fe();
  dealii::FEValues<2> fe_values(fe, quad, update_flags);
  int dofs_per_cell = fe_values.dofs_per_cell;
  int n_qpoints = quad.size();

  double sum = 0;
  std::vector<unsigned int> local_indices(dofs_per_cell);

  for (auto cell : dh.active_cell_iterators()) {
    double cell_error = 0;
    cell->get_dof_indices(local_indices);
    fe_values.reinit(cell);

    for (int ix1 = 0; ix1 < dofs_per_cell; ++ix1) {
      const double c = vec[local_indices[ix1]];
      for (int q = 0; q < n_qpoints; ++q) {
        const double f = c * fe_values.shape_value(ix1, q);
        sum += f * f * fe_values.JxW(q);
      }
    }
  }

  return sum;
}

}  // end namespace boltzmann

#endif /* _L2ERRORS_H_ */

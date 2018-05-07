#pragma once

// deal.II includes --------------------
#include <deal.II/base/function.h>

// my own includes
#include "var_form/polar_var_form.hpp"

namespace boltzmann {

/**
 * @brief spatial integrals for weighted least squares formulation
 *
 *
 * \f$ \operatorname{L} := v \cdot \nabla_x + \sigma(x)\f$
 *
 */
template <int dimX, typename APP>
class LeastSquaresVarForm : public PolarXVarForm<dimX>
{
 public:
  /**
   * @param fe : fe space
   */
  template <typename FE>
  LeastSquaresVarForm(const FE& fe)
      : PolarXVarForm<dimX>(fe)
  { /* empty */
  }

 public:
  /**
   * @brief transport matrix \f$ \left ( \operatorname{R} b_i , \epsilon(x) \operatorname{L} b_j
   * \right)_{L^2} \f$
   * where \f$ \operatorname{R} = \operatorname{L} \f$
   * @param cell
   *
   *
   * Query results with members: PolarXVarForm::S0(), PolarXVarForm::S1(), PolarXVarForm::T1(),
   * PolarXVarForm::T2()
   *
   * Note that for thea bsorption free case, i.e. \f$\sigma(x) = 0\f$, only T2() has nonzero
   * contribution.
   *
   */
  template <typename cell_iterator>
  void calc_transport_cell(const cell_iterator& cell);

  /**
   * @brief Stabilized identity \f$ \left ( \operatorname{R} b_i,
   *        \epsilon(x) b_j \right )_{L^2} \f$
   *        where \f$ \operatorname{R} = v \cdot \nabla_x + \sigma(x)\f$
   *
   *
   * @param cell
   *
   *
   * Query results with members: PolarXVarForm::S0(), PolarXVarForm::S1()
   *
   *
   */
  template <typename cell_iterator>
  void calc_identity(const cell_iterator& cell);

  /**
   * @brief Stabilized identity
   *        \f$
   *        \left ( \operatorname{R} b_i, b_j \right )_{L_2(\Omega)} +
   *        \left ( b_i, \operatorname{R} b_j \right )_{L_2(\Omega)}
   *        \f$
   *        where \f$ \operatorname{R} = v \cdot \nabla_x \f$
   * DEPRECATED
   *
   * @param cell
   *
   *
   */
  template <typename cell_iterator>
  void calc_identity_sym(const cell_iterator& cell) __attribute__((deprecated));

  /**
   * @brief just overlap (without stabilization term)
   *        hint: this uses the weighted l2 scalar product
   *        \f$ \left(b_i, b_j\right )_L^2 \f$
   *
   * Query results with members: PolarXVarForm::S0()
   *
   */
  template <typename cell_iterator>
  void calc_raw_identity(const cell_iterator& cell);

  template <typename cell_iterator>
  void calc_boundary(const cell_iterator& cell);

  static const std::string info;
};

template <int dimX, typename APP>
const std::string LeastSquaresVarForm<dimX, APP>::info = "Weighted least squares";

// --------------------------------------------------------------------------------
template <int dimX, typename APP>
template <typename cell_iterator>
inline void
LeastSquaresVarForm<dimX, APP>::calc_transport_cell(const cell_iterator& cell)
{
  // update fevalues
  this->init_cell(cell);
  // clear S0_mat, S1_mat, T1_mat, T2_mat
  PolarXVarForm<dimX>::clear_storage();

  const int n_qpoints = this->quad.size();
  // make sure the arrays have the right size
  const int dofs_per_cell = this->fe_values.dofs_per_cell;

  for (int ix1 = 0; ix1 < dofs_per_cell; ++ix1) {
    for (int ix2 = 0; ix2 < dofs_per_cell; ++ix2) {
      for (int q = 0; q < n_qpoints; ++q) {
        auto shape_ix1 = this->fe_values.shape_value(ix1, q);
        auto shape_ix2 = this->fe_values.shape_value(ix2, q);
        auto grad_ix1 = this->fe_values.shape_grad(ix1, q);
        auto grad_ix2 = this->fe_values.shape_grad(ix2, q);
        double weight = this->fe_values.JxW(q);

#if DEAL_II_VERSION_MAJOR >= 8 && DEAL_II_VERSION_MINOR <= 3
        typename PolarXVarForm<dimX>::T2_t T2tmp;
        outer_product(T2tmp, grad_ix1, grad_ix2 * weight);
        this->T2_mat[ix1][ix2] += T2tmp;
#else
        this->T2_mat[ix1][ix2] += outer_product(grad_ix1, grad_ix2 * weight);
#endif  // DEAL_II_VERSION_MAJOR >= 8 && DEAL_II_VERSION_MINOR <= 3
        // mass contributions
      }
    }
  }
}

// --------------------------------------------------------------------------------
template <int dimX, typename APP>
template <typename cell_iterator>
inline void
LeastSquaresVarForm<dimX, APP>::calc_identity(const cell_iterator& cell)
{
  // update fevalues
  this->init_cell(cell);
  // clear S0_mat, S1_mat, T1_mat, T2_mat
  PolarXVarForm<dimX>::clear_storage();

  const int n_qpoints = this->quad.size();

  const int dofs_per_cell = this->fe_values.dofs_per_cell;
  for (int ix1 = 0; ix1 < dofs_per_cell; ++ix1) {
    // test
    for (int ix2 = 0; ix2 < dofs_per_cell; ++ix2) {
      // trial
      for (int q = 0; q < n_qpoints; ++q) {
        auto shape_ix1 = this->fe_values.shape_value(ix1, q);
        auto shape_ix2 = this->fe_values.shape_value(ix2, q);
        auto grad_ix1 = this->fe_values.shape_grad(ix1, q);
        double weight = this->fe_values.JxW(q);
        // mass contributions
        this->S1_mat[ix1][ix2] += grad_ix1 * (shape_ix2 * weight);
      }
    }
  }
}

// --------------------------------------------------------------------------------
template <int dimX, typename APP>
template <typename cell_iterator>
inline void
LeastSquaresVarForm<dimX, APP>::calc_identity_sym(const cell_iterator& cell)
{
  // update fevalues
  this->init_cell(cell);
  // clear S0_mat, S1_mat, T1_mat, T2_mat
  PolarXVarForm<dimX>::clear_storage();

  const int n_qpoints = this->quad.size();
  const int dofs_per_cell = this->fe_values.dofs_per_cell;

  for (int ix1 = 0; ix1 < dofs_per_cell; ++ix1) {
    // test
    for (int ix2 = 0; ix2 < dofs_per_cell; ++ix2) {
      // trial
      for (int q = 0; q < n_qpoints; ++q) {
        auto shape_ix1 = this->fe_values.shape_value(ix1, q);
        auto shape_ix2 = this->fe_values.shape_value(ix2, q);
        auto& grad_ix1 = this->fe_values.shape_grad(ix1, q);
        auto& grad_ix2 = this->fe_values.shape_grad(ix2, q);
        // auto grad_ix2 = this->fe_values.shape_grad(ix2,q);
        double weight = this->fe_values.JxW(q);
        // mass contributions
        this->S1_mat[ix1][ix2] += grad_ix1 * (shape_ix2 * weight) + grad_ix2 * (shape_ix1 * weight);
      }
    }
  }
}

// --------------------------------------------------------------------------------
template <int dimX, typename APP>
template <typename cell_iterator>
inline void
LeastSquaresVarForm<dimX, APP>::calc_raw_identity(const cell_iterator& cell)
{
  // update fevalues
  this->init_cell(cell);
  // clear S0_mat, S1_mat, T1_mat, T2_mat
  PolarXVarForm<dimX>::clear_storage();

  const int n_qpoints = this->quad.size();
  // make sure the arrays have the right size

  const int dofs_per_cell = this->fe_values.dofs_per_cell;
  for (int ix1 = 0; ix1 < dofs_per_cell; ++ix1) {
    // test
    for (int ix2 = 0; ix2 < dofs_per_cell; ++ix2) {
      // trial
      for (int q = 0; q < n_qpoints; ++q) {
        auto shape_ix1 = this->fe_values.shape_value(ix1, q);
        auto shape_ix2 = this->fe_values.shape_value(ix2, q);
        double weight = this->fe_values.JxW(q);
        // mass contributions
        this->S0_mat[ix1][ix2] += shape_ix1 * shape_ix2 * weight;
      }
    }
  }
}

// -------------------------------------------------------------------------------
template <int dimX, typename APP>
template <typename cell_iterator>
inline void
LeastSquaresVarForm<dimX, APP>::calc_boundary(const cell_iterator& cell)
{
  // update fevalues
  // clear S0_mat, S1_mat, T1_mat, T2_mat
  PolarXVarForm<dimX>::clear_storage();

  if (cell->at_boundary()) return;

  const int faces_per_cell = dealii::GeometryInfo<dimX>::faces_per_cell;
  const int n_qpoints = this->face_quad.size();
  const int dofs_per_cell = this->fe_values.dofs_per_cell;

  for (int face_idx = 0; face_idx < faces_per_cell; ++face_idx) {
    this->init_face(cell, face_idx);
    for (int ix1 = 0; ix1 < dofs_per_cell; ++ix1) {
      // test
      for (int ix2 = 0; ix2 < dofs_per_cell; ++ix2) {
        // trial
        for (int q = 0; q < n_qpoints; ++q) {
          auto shape_ix1 = this->fe_face_values.shape_value(ix1, q);
          auto shape_ix2 = this->fe_face_values.shape_value(ix2, q);
          double weight = this->fe_values.JxW(q);
          // mass contributions
          this->S1_mat[ix1][ix2] +=
              (this->fe_face_values.normal_vector(q) * shape_ix1 * shape_ix2 * weight);
        }
      }
    }
  }
}

}  // end namespace boltzmann

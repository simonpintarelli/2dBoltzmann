#pragma once

// deal.II includes -------------------------------------------------------
#include <deal.II/base/tensor.h>
// system includes --------------------------------------------------------
#include <boost/multi_array.hpp>
#include <iostream>
// own includes -----------------------------------------------------------
#include "traits/tensor_type_traits.hpp"
#include "var_form.hpp"


namespace boltzmann {
/**
 * @brief common base class for Least squares and SUPG formulation
 *
 */
template <int dimX, int rank>
class PolarXVarFormRhs : public VarForm<dimX, dealii::QGauss>
{
 private:
  typedef VarForm<dimX, dealii::QGauss> base_class;

 public:
  static const int fe_dim;
  static const int n_quad_points;

  //@{
  /* Storage typedefs **/
  typedef typename tensor_traits<rank + 1, dimX>::template value_type<double> Tx_t;
  typedef typename tensor_traits<rank, dimX>::template value_type<double> Sx_t;
  // typedef dealii::Tensor<rank+1, dimX> Tx_t;
  // typedef dealii::Tensor<rank  , dimX> Sx_t;
  /// arrays of dimension dofs_per_cell
  typedef boost::multi_array<Tx_t, 1> Tx_vec_t;
  typedef boost::multi_array<Sx_t, 1> Sx_vec_t;
  //@}
  void clear_storage();

  const Tx_vec_t& T() const { return Tx_vec; }
  const Sx_vec_t& S() const { return Sx_vec; }

 public:
  template <typename FE>
  PolarXVarFormRhs(const FE& fe);

 protected:
  /// storage for stabilization term
  Tx_vec_t Tx_vec;
  /// storage for regular term
  Sx_vec_t Sx_vec;
};

template <int dimX, int rank>
const int PolarXVarFormRhs<dimX, rank>::fe_dim = 1;
/// hardcoded value n_quad_points!
template <int dimX, int rank>
const int PolarXVarFormRhs<dimX, rank>::n_quad_points = 3;

template <int dimX, int rank>
template <typename FE>
PolarXVarFormRhs<dimX, rank>::PolarXVarFormRhs(const FE& fe)
    : base_class(fe, n_quad_points)
{
  int n_dofs_per_cell = fe.dofs_per_cell;
  Tx_vec.resize(boost::extents[n_dofs_per_cell]);
  Sx_vec.resize(boost::extents[n_dofs_per_cell]);
}

template <int dimX, int rank>
void
PolarXVarFormRhs<dimX, rank>::clear_storage()
{
  const int n_dofs_per_cell = this->fe_values.dofs_per_cell;
  for (int i = 0; i < n_dofs_per_cell; ++i) {
    Tx_vec[i] = 0;
    Sx_vec[i] = 0;
  }
}
}  // end namespace boltzmann

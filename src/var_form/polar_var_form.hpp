#pragma once

// deal.II includes -------------------------------------------------------
#include <deal.II/base/tensor.h>
// system includes --------------------------------------------------------
#include <boost/multi_array.hpp>
#include <iostream>
// own includes -----------------------------------------------------------
#include "var_form.hpp"

namespace boltzmann {

template <int DIMX>
class PolarXVarForm : public VarForm<DIMX, dealii::QGauss>
{
 private:
  typedef VarForm<DIMX, dealii::QGauss> base_class;

 public:
  static const int fe_dim;
  static const int n_quad_points;

  //@{
  /* Storage typedefs **/
  typedef dealii::Tensor<2, DIMX> T2_t;
  typedef dealii::Tensor<1, DIMX> T1_t;
  typedef boost::multi_array<T2_t, 2> T2_mat_t;
  typedef boost::multi_array<T1_t, 2> T1_mat_t;
  typedef boost::multi_array<double, 2> D_mat_t;
  //@}

 public:
  template <typename FE>
  PolarXVarForm(const FE& fe);
  void clear_storage();
  const T1_mat_t& T1() const { return T1_mat; }
  const T2_mat_t& T2() const { return T2_mat; }
  const D_mat_t& S0() const { return S0_mat; }
  const T1_mat_t& S1() const { return S1_mat; }

 protected:
  //@{
  /** local matrix storage */
  T2_mat_t T2_mat;
  T1_mat_t T1_mat;
  T1_mat_t S1_mat;
  D_mat_t S0_mat;
  //@}
};

template <int DIMX>
const int PolarXVarForm<DIMX>::fe_dim = 1;
template <int DIMX>
const int PolarXVarForm<DIMX>::n_quad_points = 2;

template <int DIMX>
template <typename FE>
PolarXVarForm<DIMX>::PolarXVarForm(const FE& fe)
    : base_class(fe, n_quad_points)
{
  int n_dofs_per_cell = fe.dofs_per_cell;
  T2_mat.resize(boost::extents[n_dofs_per_cell][n_dofs_per_cell]);
  T1_mat.resize(boost::extents[n_dofs_per_cell][n_dofs_per_cell]);
  S1_mat.resize(boost::extents[n_dofs_per_cell][n_dofs_per_cell]);
  S0_mat.resize(boost::extents[n_dofs_per_cell][n_dofs_per_cell]);
}

template <int DIMX>
void
PolarXVarForm<DIMX>::clear_storage()
{
  // const int n_dofs_per_cell = this->fe_values.dofs_per_cell;
  assert(this->fe_values.dofs_per_cell == 4);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      T2_mat[i][j] = 0;
      T1_mat[i][j] = 0;
      S1_mat[i][j] = 0;
      S0_mat[i][j] = 0;
    }
  }
}

}  // end namespace boltzmann

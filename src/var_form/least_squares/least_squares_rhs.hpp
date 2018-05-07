#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

#include "aux/tensor_helpers.hpp"
#include "var_form/polar_var_form_rhs.hpp"


namespace boltzmann {

template <int dimX, typename APP, typename SOURCE = typename APP::source_function_t>
class RhsVarForm : public PolarXVarFormRhs<dimX, SOURCE::rank>
{
 private:
  typedef PolarXVarFormRhs<dimX, SOURCE::rank> base_class;
  typedef SOURCE source_xv_t;
  typedef typename source_xv_t::SX_t source_function_t;

 public:
  // is S(x,v) = s(x) . s(v) ?s(x),s(v) scalar or vector valued?
  static const int rank = source_function_t::rank;

 public:
  template <typename FE>
  RhsVarForm(const FE& fe)
      : base_class(fe)
  { /* empty */
  }

  template <typename cell_iterator>
  void calc(const cell_iterator& cell);

 private:
  //@{
  /// functions
  source_function_t source_;
  //@}
  //@{
  /// working storage
  std::vector<typename base_class::Sx_t> source_values_;
  //@}
};

/**
 * @brief Weighted least squares var form in X-domain
 \f[
 \int_D s^x(x) \otimes \nabla \alpha(x) \epsilon(x) \mathrm{d} x
 + \int_D s^x \times \alpha(x) \sigma(x) \epsilon(x) \mathrm{d} x
 \f]
 Note that \f$ s^x (x) \f$ can be \f$ \mathbb{R}^2 \rightarrow \mathbb{R} \f$
 or \f$ \mathbb{R}^2 \rightarrow \mathbb{R}^2 \f$
 *
 *
 * @param cell
 *
 * @return
 */
template <int dimX, typename APP, typename SOURCE>
template <typename cell_iterator>
void
RhsVarForm<dimX, APP, SOURCE>::calc(const cell_iterator& cell)
{
  // update fevalues
  this->init_cell(cell);
  // clear
  base_class::clear_storage();
  const int nqpoints = this->quad.size();
  const int dofs_per_cell = this->fe_values.dofs_per_cell;
  // resize storage
  source_.value_list(this->fe_values.get_quadrature_points(), source_values_);

  for (int ix = 0; ix < dofs_per_cell; ++ix) {
    for (int q = 0; q < nqpoints; ++q) {
      auto shape = this->fe_values.shape_value(ix, q);
      auto grad = this->fe_values.shape_grad(ix, q);
      double weight = this->fe_values.JxW(q);
      /// mass part
      /// stabilization part i.e multiplied with grad
      typename base_class::Tx_t Tx_tmp;
      outer_product(Tx_tmp, source_values_[q], grad);

      this->Tx_vec[ix] += Tx_tmp * weight;
    }
  }
}

}  // end namespace boltzmann

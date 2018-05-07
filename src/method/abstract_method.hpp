#pragma once

namespace boltzmann {
template <template <int, class> class VARFORM,
          template <int, class, class> class RHS_VARFORM,
          typename FE,
          typename SPECTRAL_BASIS = void>
class AbstractMethod
{
 public:
  //@{ public typedefs
  // template alias for var_form
  template <int dim, typename APP>
  using var_form_t = VARFORM<dim, APP>;

  template <int dim, typename APP, typename SOURCE = typename APP::source_function_t>
  using var_form_rhs_t = RHS_VARFORM<dim, APP, SOURCE>;
  typedef FE fe_t;
  typedef SPECTRAL_BASIS spectral_basis_t;
  //@}
};

}  // end namespace boltzmann

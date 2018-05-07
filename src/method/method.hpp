#pragma once

// deal.II includes --------------------------------------------------------
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_q.h>

// own includes ------------------------------------------------------------
#include "abstract_method.hpp"
#include "enum/enum.hpp"
#include "spectral/basis/spectral_basis.hpp"
#include "spectral/basis/spectral_elem.hpp"
#include "var_form/least_squares/least_squares.hpp"
#include "var_form/least_squares/least_squares_rhs.hpp"


namespace boltzmann {

// short cut for least squares method
class MLSMethod : public AbstractMethod<LeastSquaresVarForm,
                                        RhsVarForm,
                                        dealii::FE_Q<2>,
                                        SpectralBasis<SpectralElem<double, XiR, LaguerreKS> > >
{
  const static std::string info;
};
const std::string MLSMethod::info = "MLS";

template <enum METHOD>
class Method
{
};

template <>
class Method<METHOD::MODLEASTSQUARES> : public MLSMethod
{
 public:
  const static enum METHOD lsq_type;
};

const METHOD Method<METHOD::MODLEASTSQUARES>::lsq_type = METHOD::MODLEASTSQUARES;

}  // end namespace boltzmann

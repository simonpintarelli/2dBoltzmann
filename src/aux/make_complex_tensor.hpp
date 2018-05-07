#pragma once

#include <deal.II/base/tensor.h>
#include <complex>

namespace boltzmann {
namespace helper {

template <int dim>
dealii::Tensor<2, dim, std::complex<double> >
make_complex_tensor(const dealii::Tensor<2, dim, double>& tensor)
{
  dealii::Tensor<2, dim, std::complex<double> > ret;
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < dim; ++j) {
      ret[i][j] = std::complex<double>(tensor[i][j], 0.);
    }
  return ret;
}

template <int dim>
dealii::Tensor<1, dim, std::complex<double> >
make_complex_tensor(const dealii::Tensor<1, dim, double>& tensor)
{
  dealii::Tensor<1, dim, std::complex<double> > ret;
  for (int i = 0; i < dim; ++i) ret[i] = std::complex<double>(tensor[i], 0.);
  return ret;
}

template <int dim>
dealii::Tensor<0, dim, std::complex<double> >
make_complex_tensor(const dealii::Tensor<0, dim, double>& tensor)
{
  dealii::Tensor<0, dim, std::complex<double> > ret;
  ret = std::complex<double>(tensor, 0.);
  return ret;
}

std::complex<double>
make_complex_tensor(double a)
{
  return std::complex<double>(a, 0);
}

}  // end namespace helper

}  // end namespace Namespace

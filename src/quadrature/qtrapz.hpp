#pragma once

#include <deal.II/base/tensor.h>
#include <complex>
#include <functional>
#include "aux/tensor_helpers.hpp"


namespace boltzmann {

template <int dim>
class QTrapz;
namespace _ {

template <class T>
struct NumberTraits;

template <int dim, int rank, class NUMBER>
struct NumberTraits<dealii::Tensor<dim, rank, NUMBER> >
{
  typedef NUMBER numeric_t;
};

template <>
struct NumberTraits<double>
{
  typedef double numeric_t;
};

template <>
struct NumberTraits<std::complex<double> >
{
  typedef std::complex<double> numeric_t;
};

}  // end namespace

template <>
class QTrapz<1>
{
 public:
  QTrapz(int npts, double a, double b);

  template <class NUMBER>
  NUMBER compute(const std::function<NUMBER(double)>& f) const;

  const std::vector<double>& get_weights() { return weights_; }

  const std::vector<double>& get_points() { return points_; }

 private:
  const double a;
  const double b;
  const int npts;
  std::vector<double> points_;
  std::vector<double> weights_;
};

/**
 * trapezoidal quadrature rule, works also with tensor valued functions
 *
 *
 * @return
 */
template <class NUMBER>
NUMBER
qtrapz1d(const std::function<NUMBER(double)>& f, double a, double b, int npts)
{
  typedef typename _::NumberTraits<NUMBER>::numeric_t numeric_t;
  const double h = (b - a) / double(npts - 1);

  NUMBER sum = 0;
  for (int i = 0; i < npts - 1; ++i) {
    const double x1 = a + h * (i);
    const double x2 = a + h * (i + 1);
    sum += f(x1) + f(x2);
  }
  sum *= numeric_t(0.5 * h);
  return sum;
}

QTrapz<1>::QTrapz(int npts, double a, double b)
    : a(a)
    , b(b)
    , npts(npts)
    , points_(npts)
    , weights_(npts)
{
  double h = (b - a) / (npts - 1);
  for (int i = 0; i < npts; ++i) {
    if (i == 0 || i == npts - 1)
      weights_[i] = 0.5 * h;
    else
      weights_[i] = h;
    points_[i] = a + i * h;
  }
}

}  // end namespace boltzmann

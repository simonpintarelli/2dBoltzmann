/**
 * @file   evaluator_wrapper.C
 * @author  <simon@thinkpadX1>
 * @date   Fri Jan 31 16:32:11 2014
 *
 * @brief  Expose the evaluator class directly to python, i.e. read a
 * triangulation from disk
 *
 *
 */

// system includes --------------------------------------------------------
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#ifdef PYTHON
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/tuple.hpp>
namespace bp = boost::python;
namespace np = boost::python::numpy;
#endif

#include <complex>
#include <fstream>
#include <iostream>
#include <string>
// own includes -----------------------------------------------------------
#include "quadrature/gauss_legendre_quadrature.hpp"
#include "quadrature/maxwell_quadrature.hpp"
#include "quadrature/qhermitew.hpp"
#include "quadrature/qmaxwell.hpp"

using namespace std;
using namespace boltzmann;


// ------------------------------------------------------------------------
class QuadMaxwell
{
  // private:
  //  typedef std::complex<double> cdouble;
 public:
  QuadMaxwell(double alpha, int N);

  np::ndarray points() const;
  np::ndarray weights() const;

 private:
  QMaxwell qmax;
};

// ------------------------------------------------------------------------
QuadMaxwell::QuadMaxwell(double alpha, int N)
    : qmax(alpha, N)
{
}

// ----------------------------------------------------------------------
np::ndarray QuadMaxwell::points() const
{
  const double* data = qmax.points_data();
  np::ndarray arr = np::from_data(data,
                                  np::dtype::get_builtin<double>(),
                                  bp::make_tuple(qmax.size()),
                                  bp::make_tuple(sizeof(double)),
                                  bp::object());
  return arr;
}

// ----------------------------------------------------------------------
np::ndarray QuadMaxwell::weights() const
{
  const double* data = qmax.weights_data();
  np::ndarray arr = np::from_data(data,
                                  np::dtype::get_builtin<double>(),
                                  bp::make_tuple(qmax.size()),
                                  bp::make_tuple(sizeof(double)),
                                  bp::object());
  return arr;
}

// ----------------------------------------------------------------------
class GenQuadMaxwell
{
 public:
  GenQuadMaxwell(int N, int p)
      : quad(N, 256, p)
  { /* empty */
  }

  np::ndarray points() const;
  np::ndarray weights() const;

 private:
  MaxwellQuadrature quad;
};

// ----------------------------------------------------------------------
np::ndarray GenQuadMaxwell::points() const
{
  const double* data = quad.points_data();
  np::ndarray arr = np::from_data(data,
                                  np::dtype::get_builtin<double>(),
                                  bp::make_tuple(quad.size()),
                                  bp::make_tuple(sizeof(double)),
                                  bp::object());
  return arr;
}

// ----------------------------------------------------------------------
np::ndarray GenQuadMaxwell::weights() const
{
  const double* data = quad.weights_data();
  np::ndarray arr = np::from_data(data,
                                  np::dtype::get_builtin<double>(),
                                  bp::make_tuple(quad.size()),
                                  bp::make_tuple(sizeof(double)),
                                  bp::object());
  return arr;
}

// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// Hermite quadrature
class QuadHermiteW
{
 public:
  QuadHermiteW(double alpha, int N)
      : quad(alpha, N)
  {
  }

  np::ndarray points() const;
  np::ndarray weights() const;

 private:
  boltzmann::QHermiteW quad;
};

// ---------------------------------------------------------------------
np::ndarray QuadHermiteW::points() const
{
  const double* data = quad.points_data();
  np::ndarray arr = np::from_data(data,
                                  np::dtype::get_builtin<double>(),
                                  bp::make_tuple(quad.size()),
                                  bp::make_tuple(sizeof(double)),
                                  bp::object());
  return arr;
}

// ----------------------------------------------------------------------
np::ndarray QuadHermiteW::weights() const
{
  const double* data = quad.weights_data();
  np::ndarray arr = np::from_data(data,
                                  np::dtype::get_builtin<double>(),
                                  bp::make_tuple(quad.size()),
                                  bp::make_tuple(sizeof(double)),
                                  bp::object());
  return arr;

}

// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
class QuadGauleg
{
 public:
  QuadGauleg(int N)
      : quad(N)
  { /* empty */
  }

  np::ndarray points() const;
  np::ndarray weights() const;

 private:
  GaussLegendreQuadrature quad;
};

// ---------------------------------------------------------------------
np::ndarray QuadGauleg::points() const
{
  const double* data = quad.points_data();
  np::ndarray arr = np::from_data(data,
                                  np::dtype::get_builtin<double>(),
                                  bp::make_tuple(quad.size()),
                                  bp::make_tuple(sizeof(double)),
                                  bp::object());
  return arr;
}

// ----------------------------------------------------------------------
np::ndarray QuadGauleg::weights() const
{
  const double* data = quad.weights_data();
  np::ndarray arr = np::from_data(data,
                                  np::dtype::get_builtin<double>(),
                                  bp::make_tuple(quad.size()),
                                  bp::make_tuple(sizeof(double)),
                                  bp::object());
  return arr;
}

#ifdef PYTHON
// #if PY_VERSION_HEX >= 0x03000000
// int init_numpy() { import_array(); }
// #else
// void init_numpy() { import_array(); }
// #endif

BOOST_PYTHON_MODULE(libbquad)
{
  using namespace boost::python;
  // import for handling of numpy arrays
  np::initialize();
  //  docstring_options local_docstring_options(true, true, false);
  class_<QuadMaxwell, boost::noncopyable>("QuadMaxwell", init<double, int>())
      .def("points", &QuadMaxwell::points, "get points")
      .def("weights", &QuadMaxwell::weights, "get weights");

  class_<GenQuadMaxwell, boost::noncopyable>("GenQuadMaxwell", init<int, int>())
      .def("points", &GenQuadMaxwell::points, "get points")
      .def("weights", &GenQuadMaxwell::weights, "get weights");

  class_<QuadHermiteW, boost::noncopyable>("QuadHermiteW", init<double, int>())
      .def("points", &QuadHermiteW::points, "get points")
      .def("weights", &QuadHermiteW::weights, "get weights");

  class_<QuadGauleg, boost::noncopyable>("QuadGauleg", init<int>())
      .def("points", &QuadGauleg::points, "get points")
      .def("weights", &QuadGauleg::weights, "get weights");
}
#endif

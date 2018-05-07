#include <Eigen/Sparse>
#include <boost/program_options.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/tuple.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include "aux/boostnpy.hpp"

#include "spectral/laguerren_ks.hpp"

namespace bp = boost::python;

/**
 * @brief Wrapper class for \ref LaguerreNKS
 *
 * Evaluator for Polar-Laguerre basis functions. Evaluation includes \f$exp(-r^2/2)\f$.
 * See test/laguerren_ks for an example how to use it.
 *
 */
class LaguerreKS_wrapper
{
 private:
  typedef double numeric_t;

 public:
  LaguerreKS_wrapper()
      : laguerreks_(0)
  {
  }
  LaguerreKS_wrapper(unsigned int K, const np::ndarray& x);
  np::ndarray get(unsigned int k, unsigned int alpha) const;

 private:
  boltzmann::LaguerreNKS<numeric_t> laguerreks_;
  int npts_;
};

LaguerreKS_wrapper::LaguerreKS_wrapper(unsigned int K, const np::ndarray& x)
    : laguerreks_(K)
{
  int dim = x.get_nd();
  assert(dim == 1);
  const Py_intptr_t* shape = x.get_shape();
  const numeric_t* x_ptr = reinterpret_cast<const numeric_t*>(x.get_data());
  laguerreks_.compute(x_ptr, shape[0], 0.5);
  npts_ = shape[0];
}

np::ndarray LaguerreKS_wrapper::get(unsigned int k, unsigned int alpha) const
{
  const numeric_t* dptr = laguerreks_.get(k, alpha);
  np::ndarray np_arr = np::from_data(reinterpret_cast<const void*>(dptr),
                                     np::dtype::get_builtin<numeric_t>(),
                                     bp::make_tuple(npts_),
                                     bp::make_tuple(sizeof(numeric_t)),
                                     bp::object(*this));
  return np_arr;
}

#ifdef PYTHON
BOOST_PYTHON_MODULE(libLaguerreKS)
{
  using namespace boost::python;
  np::initialize();
  class_<LaguerreKS_wrapper>("LaguerreKS",
                             "Evaluator for Polar-Laguerre basis functions. Evaluation includes "
                             "\f$exp(-r^2/2)\f$.\nSee test/laguerren_ks for a usage example.",
                             init<>())
      .def(init<unsigned int, const np::ndarray&>(args("K", "x")))
      .def("get",
           &LaguerreKS_wrapper::get,
           "returns the values of Polar-Laguerre radial basis fcts at positions x",
           args("k", "a"));
}

#endif

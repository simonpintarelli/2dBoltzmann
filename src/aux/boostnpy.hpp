#ifndef BOOSTNPY_H
#define BOOSTNPY_H

#ifdef USE_DEPRECATED_BOOST_NPY
#include <boost/numpy.hpp>
#else
#include <boost/python/numpy.hpp>
#endif  // USE_DEPRECATED_BOOST_NPY

#ifdef USE_DEPRECATED_BOOST_NPY
namespace np = boost::numpy;
#else
namespace np = boost::python::numpy;
#endif  // USE_DEPRECATED_BOOST_NPY

#endif /* BOOSTNPY_H */

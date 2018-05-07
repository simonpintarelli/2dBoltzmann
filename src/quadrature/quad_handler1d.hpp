#pragma once

#include <array>
#include <map>
#include <memory>
#include <mutex>
#include "aux/singleton.hpp"

#include "gauss_legendre_quadrature.hpp"
#include "maxwell_quadrature.hpp"

namespace boltzmann {

template <typename T>
class QuadAdaptor
{
};

// ----------------------------------------------------------------------
template <>
struct QuadAdaptor<MaxwellQuadrature>
{
  inline static double pts(double p, double alpha) { return p / std::sqrt(alpha); }

  inline static double wts(double w, double alpha) { return w / alpha; }

  inline static void apply(double* vpts, double* vwts, const MaxwellQuadrature& quad, double alpha)
  {
    for (unsigned int i = 0; i < quad.size(); ++i) {
      vpts[i] = pts(quad.pts(i), alpha);
      vwts[i] = wts(quad.wts(i), alpha);
    }
  }
};

// ----------------------------------------------------------------------
template <>
struct QuadAdaptor<GaussLegendreQuadrature>
{
  inline static double pts(double p, double a, double b)
  {
    double d = b - a;
    assert(d > 0);
    return d * (p + 1.0) / 2.0 + a;
  }

  inline static double wts(double w, double a, double b)
  {
    double d = b - a;
    assert(d > 0);
    return w / 2.0 * d;
  }

  /**
   *
   *
   * @param vpts target quadrature nodes
   * @param vwts target quadrature weights
   * @param quad GaussLegendreQuadrature on [-1,1]
   * @param a    interval begin
   * @param b    interval end
   *
   * @return
   */
  inline static void apply(
      double* vpts, double* vwts, const GaussLegendreQuadrature& quad, double a, double b)
  {
    for (unsigned int i = 0; i < quad.size(); ++i) {
      vpts[i] = pts(quad.pts(i), a, b);
      vwts[i] = wts(quad.wts(i), a, b);
    }
  }
};

// ----------------------------------------------------------------------
template <typename QUADRATURE_TYPE>
class QuadHandler : public CSingleton<QuadHandler<QUADRATURE_TYPE>>
{
 public:
  typedef QUADRATURE_TYPE quad_t;
  typedef int key_t;
  typedef std::shared_ptr<quad_t> ptr_t;

 public:
  const quad_t& get(int n)
  {
    auto it = storage_.find(n);

    if (it != storage_.end()) {
      return *(it->second);
    } else {
      _mutex.lock();
      ptr_t p(new quad_t(n));
      storage_[n] = p;
      _mutex.unlock();
      return *p;
    }
  }

 private:
  std::map<key_t, ptr_t> storage_;
  QuadHandler() {}
  friend CSingleton<QuadHandler>;
  std::mutex _mutex;
};

}  // end namespace boltzmann

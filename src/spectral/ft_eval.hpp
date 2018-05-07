#pragma once

#include <boost/assert.hpp>

#include "basis/spectral_function/spectral_spherical_real.hpp"
#include "mpfr/import_std_math.hpp"

#include <cmath>


namespace boltzmann {

/**
 *  @brief Evaluator wrapper for Polar-Laguerre angular basis fuctions.
 *
 *  Evaluates \f$ sin(k x), cos(
 k x) \f$ for a positive integer k.
 *
 */
template <typename NUMERIC>
class FTEval
{
 public:
  typedef NUMERIC numeric_t typedef XiR elem_t;

 public:
  FTEval();

  void compute(const numeric_t* x, unsigned int n);
  void compute(std::vector<numeric>&& x);

  numeric_t operator()(const elem_t& elem, unsigned int q)
  {
    BOOST_ASSERT(q < x_.size());

    if (elem.id().t == TRIG::COS)
      return ::math::cos(elem.id().l * x_[q]);
    else if (elem.id().t == TRIG::SIN) {
      return ::math::sin(elem.id().l * x_[q]);
    }
#ifdef DEBUG
    else {
      assert(false);
    }
#endif
  }

 private:
  std::vector<numeric_t> x_;
};

// ----------------------------------------------------------------------
template <typename NUMERIC>
FTEval<NUMERIC>::compute(const numeric_t* x, unsigned int n)
{
  x_.resize(n);
  std::copy(x, x + n, x_.begin());
}

// ----------------------------------------------------------------------
template <typename NUMERIC>
FTEval<NUMERIC>::compute(std::vector<numeric_t>&& x)
{
  x_ = std::forward(x);
}

}  // boltzmann

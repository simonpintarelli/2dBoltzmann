#pragma once

#include <Eigen/Core>
#include <cassert>
#include <vector>


namespace boltzmann {
// homogeneous
namespace hg {
template <typename NUMERIC_T = double>
class RK4
{
 public:
  typedef NUMERIC_T numeric_t;

 private:
  typedef Eigen::VectorXd vec_t;

 public:
  RK4(int N_)
      : N(N_)
      , v_k1(N_)
      , v_k2(N_)
      , v_k3(N_)
      , v_k4(N_)
  { /* empty */
  }

  void apply(NUMERIC_T* dst,
             const NUMERIC_T* src,
             const std::function<void(NUMERIC_T* dst, const NUMERIC_T* src)>& f,
             double dt);

 private:
  int N;
  vec_t v_k1;
  vec_t v_k2;
  vec_t v_k3;
  vec_t v_k4;
};

// ----------------------------------------------------------------------
template <typename NUMERIC_T>
void
RK4<NUMERIC_T>::apply(NUMERIC_T* dst,
                      const NUMERIC_T* src,
                      const std::function<void(NUMERIC_T* dst, const NUMERIC_T* src)>& f,
                      double dt)
{
  assert(src != dst);

  typedef Eigen::Map<vec_t> map_vec_t;
  typedef Eigen::Map<const vec_t> map_const_vec_t;

  map_const_vec_t v_in(src, N);
  map_vec_t v_dst(dst, N);

  f(v_k1.data(), v_in.data());

  v_dst = v_in + 0.5 * dt * v_k1;
  f(v_k2.data(), v_dst.data());

  v_dst = v_in + 0.5 * dt * v_k2;
  f(v_k3.data(), v_dst.data());

  v_dst = v_in + dt * v_k3;
  f(v_k4.data(), v_dst.data());

  v_dst = v_in + dt / 6 * (v_k1 + 2 * v_k2 + 2 * v_k3 + v_k4);
}

}  // end hg
}  // end boltzmann

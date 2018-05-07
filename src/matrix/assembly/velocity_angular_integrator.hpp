#pragma once

// deal.II includes ------------------------------------------------------------
#include <deal.II/base/numbers.h>
#include <deal.II/base/tensor.h>
//#include <deal.II/base/quadrature_lib.h>
// system includes -------------------------------------------------------------
#include <map>
// own includes ----------------------------------------------------------------
#include <quadrature/trig_int.hpp>


namespace boltzmann {

namespace local_ {
// ----------------------------------------------------------------------
inline bool
is_zero(const dealii::Tensor<2, 2, double>& t2)
{
  const double tol = 1e-16;
  return std::abs(t2[0][0]) < tol && std::abs(t2[0][1]) < tol && std::abs(t2[1][0]) < tol &&
         std::abs(t2[1][1]) < tol;
}

// ----------------------------------------------------------------------
inline bool
is_zero(const dealii::Tensor<1, 2, double>& t1)
{
  const double tol = 1e-16;
  return std::abs(t1[0]) < tol && std::abs(t1[1]) < tol;
}

}  // end namespace local_

template <int DIM>
class VelocityAngularIntegrator;

/**
 * @brief helper class for system matrix assembly
 *
 */
template <>
class VelocityAngularIntegrator<2>
{
 public:
  typedef double numeric_t;
  typedef dealii::Tensor<2, 2, numeric_t> T2_t;
  typedef dealii::Tensor<1, 2, numeric_t> T1_t;
  typedef std::pair<unsigned int, unsigned int> key_t;

 private:
  typedef std::map<key_t, numeric_t> map_S0_t;
  typedef std::map<key_t, T1_t> map_T1_t;
  typedef std::map<key_t, T2_t> map_T2_t;

 public:
  VelocityAngularIntegrator() { /* empty */}
  template <typename BASIS>
  void init(const BASIS& angular_basis);

  map_S0_t::const_iterator begin_s0() const;
  map_S0_t::const_iterator end_s0() const;
  map_T1_t::const_iterator begin_t1() const;
  map_T1_t::const_iterator end_t1() const;
  map_T2_t::const_iterator begin_t2() const;
  map_T2_t::const_iterator end_t2() const;

  const map_S0_t& get_s0() const { return ms0; }
  const map_T1_t& get_s1() const { return mt1; }
  const map_T1_t& get_t1() const { return mt1; }
  const map_T2_t& get_t2() const { return mt2; }

  //  void write_to_file(std::string fname) const;
 private:
  map_S0_t ms0;
  map_T1_t mt1;
  map_T2_t mt2;
};

// ---------------------------------------------------------------------------
template <typename BASIS>
void
VelocityAngularIntegrator<2>::init(const BASIS& angular_basis)
{
  auto make_t1 = [](int l1, int t1, int l2, int t2) {
    T1_t m1;
    m1[0] = trig_int(COS, 1, {(TRIG)t1, (TRIG)t2}, {l1, l2});
    m1[1] = trig_int(SIN, 1, {(TRIG)t1, (TRIG)t2}, {l1, l2});
    return m1;
  };

  auto make_t2 = [](int l1, int t1, int l2, int t2) {
    T2_t m2;
    for (int tp = 0; tp < 2; ++tp) {
      for (int t = 0; t < 2; ++t) {
        m2[tp][t] = trig_int((TRIG)tp, 1, {(TRIG)t, (TRIG)t1, (TRIG)t2}, {1, l1, l2});
      }
    }
    return m2;
  };

  for (auto it1 = angular_basis.begin(); it1 != angular_basis.end(); ++it1) {
    int l1 = it1->get_id().l;
    int t1 = it1->get_id().t;
    unsigned int ix1 = it1 - angular_basis.begin();
    for (auto it2 = angular_basis.begin(); it2 != angular_basis.end(); ++it2) {
      int l2 = it2->get_id().l;
      int t2 = it2->get_id().t;
      unsigned int ix2 = it2 - angular_basis.begin();
      // S0
      double s0 = trig_int((TRIG)t1, l1, {(TRIG)t2}, {l2});
      if (std::abs(s0) > 1e-16) ms0[std::make_pair(ix1, ix2)] = s0;
      // T1
      T1_t m1 = make_t1(l1, t1, l2, t2);
      if (!local_::is_zero(m1)) mt1[std::make_pair(ix1, ix2)] = m1;
      // T2
      T2_t m2 = make_t2(l1, t1, l2, t2);
      if (!local_::is_zero(m2)) mt2[std::make_pair(ix1, ix2)] = m2;
    }
  }
}

}  // end namespace boltzmann

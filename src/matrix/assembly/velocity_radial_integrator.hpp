#pragma once

// system includes --------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <map>
#include <utility>
#include <vector>
#include <memory>

#include "quadrature/qmaxwell.hpp"

#ifdef DEBUG
#include <fstream>
#include <iomanip>
#endif


namespace boltzmann {

/**
 * @brief radial integrator for system matrix assembly
 *
 */
class VelocityRadialIntegrator
{
 public:
  typedef std::pair<int, int> index_t;

 private:
  typedef std::map<index_t, double> map_t;

 public:
  typedef map_t value_type;
  typedef typename map_t::const_iterator iterator;

 public:
  template <typename RadialBasis, typename Weight>
  void init(const RadialBasis& test_basis,
            const RadialBasis& trial_basis,
            const Weight& weight,
            int nqpts = 61);

  template <typename RadialBasis, typename Weight>
  void init(const RadialBasis& basis, const Weight& weight, int nqpts = 61);

  /// \f$\int r  \phi_{k1}(r) \phi_{k2}(r) \mu \;r \mathrm{d}r  \f$
  double get_T1(int k1, int k2) const;

  /// \f$ \int r^2 \, \phi_{k1}r \phi_{k2}(r) \mu \; r \mathrm{d}r \f$
  double get_T2(int k1, int k2) const;

  /// \f$ \int \phi_{k1}r \phi_{k2}(r) \mu \; r \mathrm{d}r \f$
  double get_S0(int k1, int k2) const;

  iterator begin_s0() const;
  iterator end_s0() const;
  iterator begin_t1() const;
  iterator end_t1() const;
  iterator begin_t2() const;
  iterator end_t2() const;

  const map_t& get_s0() const { return map0_; }
  const map_t& get_s1() const { return map1_; }
  const map_t& get_t1() const { return map1_; }
  const map_t& get_t2() const { return map2_; }

  void info() const;

 private:
  //@{
  /// TODO replace by Eigen Sparse Matrix
  /// storage S0
  map_t map0_;
  /// storage T1
  map_t map1_;
  /// storage T2
  map_t map2_;
  //@}

  /// max. degree of Laguerre polynomial
  int nK;

  std::vector<double> pts;
  std::vector<double> wts;
};

// --------------------------------------------------------------------------------
template <typename RadialBasis, typename Weight>
void
VelocityRadialIntegrator::init(const RadialBasis& test_basis,
                               const RadialBasis& trial_basis,
                               const Weight& weight,
                               int nqpts)
{
  const double tol = 1e-10;
  typedef std::shared_ptr<QMaxwell> ptr_t;
  std::map<double, ptr_t> quad_map;
  const double alpha = weight.exponent();

  for (auto it1 = test_basis.begin(); it1 != test_basis.end(); ++it1) {
    const int j1 = it1 - test_basis.begin();

    for (auto it2 = trial_basis.begin(); it2 != trial_basis.end(); ++it2) {
      const int j2 = it2 - trial_basis.begin();

      const double nu = alpha + it1->w() + it2->w();
      if (quad_map.find(nu) == quad_map.end()) quad_map[nu] = ptr_t(new QMaxwell(nu, nqpts));
      const auto& quad = *quad_map[nu];
      double t1 = 0;
      double t2 = 0;
      double s0 = 0;
      for (unsigned int q = 0; q < quad.size(); ++q) {
        s0 += it1->evaluate(quad.pts(q)) * it2->evaluate(quad.pts(q)) * quad.wts(q);
        t1 += it1->evaluate(quad.pts(q)) * it2->evaluate(quad.pts(q)) * quad.pts(q) * quad.wts(q);
        t2 += it1->evaluate(quad.pts(q)) * it2->evaluate(quad.pts(q)) * quad.pts(q) * quad.pts(q) *
              quad.wts(q);
      }
      if (std::abs(s0) > tol) map0_[std::make_pair(j1, j2)] = s0;
      if (std::abs(t1) > tol) map1_[std::make_pair(j1, j2)] = t1;
      if (std::abs(t2) > tol) map2_[std::make_pair(j1, j2)] = t2;
    }
  }
}

// -------------------------------------------------------------------------------
template <typename RadialBasis, typename Weight>
void
VelocityRadialIntegrator::init(const RadialBasis& basis, const Weight& weight, int nqpts)
{
  this->init(basis, basis, weight, nqpts);
}

// ----------------------------------------------------------------------
inline void
VelocityRadialIntegrator::info() const

{
  std::cout << "size(map0_) = " << map0_.size() << std::endl
            << "size(map1_) = " << map1_.size() << std::endl
            << "size(map2_) = " << map2_.size() << std::endl;
}

// --------------------------------------------------------------------------------
inline double
VelocityRadialIntegrator::get_S0(int k1, int k2) const
{
  auto it = map0_.find(std::make_pair(k1, k2));
  if (it == map0_.end()) {
    return 0;
  }
  return it->second;
}

// --------------------------------------------------------------------------------
inline double
VelocityRadialIntegrator::get_T1(int k1, int k2) const
{
  auto it = map1_.find(std::make_pair(k1, k2));
  if (it == map1_.end()) {
    return 0;
  }
  return it->second;
}

// --------------------------------------------------------------------------------
inline double
VelocityRadialIntegrator::get_T2(int k1, int k2) const
{
  auto it = map2_.find(std::make_pair(k1, k2));
  if (it == map2_.end()) {
    return 0;
  }

  return it->second;
}

}  // end namespace boltzmann

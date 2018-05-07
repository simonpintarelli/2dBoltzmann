#pragma once

#include <deal.II/base/tensor.h>
#include <functional>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "aux/filtered_range.hpp"
#include "aux/hash_specializations.hpp"
#include "aux/simple_sparse_matrix.hpp"
#include "matrix/assembly/velocity_angular_integrator.hpp"
#include "matrix/assembly/velocity_radial_integrator.hpp"
#include "matrix/assembly/weight.hpp"
#include "spectral/basis/spectral_basis_dimension_accessor.hpp"


namespace boltzmann {
namespace local_ {
/**
 * @brief Compute and cache overlap integrals in radial direction
 */
struct radial_entry_computer_t
{
  // -----------------------------------------------------------------
  radial_entry_computer_t(int nqpts_ = 90)
      : nqpts(nqpts_)
  {
    /* empty */
  }

  /**
   *  @brief I = \f$\int_\bbR b_1(r) k(r), b_2(r) r \ud r \f$
   *
   *  @tparam E spectral element
   *  @param k additional function argument
   *  @param b1 1st basis function
   *  @param b2 2nd basis function
   *  @return I integral
   */
  template <typename E>
  double compute(const std::function<double(double)>& k,
                 const E& b1,
                 const E& b2,
                 const double a = 0 /* inner product weight exp(-a r**2) */)
  {
    const double FUZZY = 1e7;
    const double nu = a + b1.w() + b2.w();
    const long int key = FUZZY * nu;
    if (quad_cache.find(key) == quad_cache.end()) quad_cache[key] = ptr_t(new QMaxwell(nu, nqpts));
    const auto& quad = *(quad_cache[key]);

    double sum = 0;
    for (unsigned int q = 0; q < quad.size(); ++q) {
      sum += b1.evaluate(quad.pts(q)) * b2.evaluate(quad.pts(q)) * k(quad.pts(q)) * quad.wts(q);
    }

    return sum;
  }

 private:
  const int nqpts;
  typedef std::shared_ptr<QMaxwell> ptr_t;
  std::map<long int, ptr_t> quad_cache;
};
}  // end namespace local_

template <int DIM>
class VelocityVarForm
{ };

/**
 * @brief Storage for velocity domain matrix entries
 *
 */
template <>
class VelocityVarForm<2>
{
 private:
  static const int DIM = 2;
  typedef unsigned int index_t;
  typedef dealii::Tensor<1, DIM, double> t1_t;
  typedef dealii::Tensor<2, DIM, double> t2_t;

  typedef SimpleSparseMatrix<double> s0_entries_t;
  typedef SimpleSparseMatrix<t1_t> t1_entries_t;
  typedef SimpleSparseMatrix<t2_t> t2_entries_t;

 public:
  template <typename BASIS>
  void init(const BASIS& test_basis, const BASIS& trial_basis, const double beta = 2);

  template <typename BASIS>
  void init(const BASIS& trial_basis, const double beta = 2);

 private:
  s0_entries_t s0_;
  t1_entries_t t1_;
  t2_entries_t t2_;

  /// helpers
  VelocityAngularIntegrator<2> vai;

  constexpr const static double TOL = 1e-11; // magic number

 public:
  auto get_s0() const -> decltype(s0_.get_vec()) { return s0_.get_vec(); }
  auto get_s1() const -> decltype(t1_.get_vec()) { return t1_.get_vec(); }
  auto get_t1() const -> decltype(t1_.get_vec()) { return t1_.get_vec(); }
  auto get_t2() const -> decltype(t2_.get_vec()) { return t2_.get_vec(); }

  const SimpleSparseMatrix<double>& get_s0m() const { return s0_; }
  const SimpleSparseMatrix<t1_t>& get_s1m()   const { return t1_; }
  const SimpleSparseMatrix<t1_t>& get_t1m()   const { return t1_; }
  const SimpleSparseMatrix<t2_t>& get_t2m()   const { return t2_; }

  /**
   *  @brief \f$\langle \psi_1, \psi_2\rangle\f$
   *
   *  Detailed description
   *
   *  @return Returns a vector [(entry_t...)] which contains the non-zero
   *    entries of the sparse matrix arising from the overlap integral above.
   *    entry_t has members row, col and val. row and col correspond to the
   *    enumeration of DoFs in the spectral basis.

   */
  auto s0() const -> const decltype(s0_) & { return s0_; }

  /**
   *  @brief \f$\langle v_i \psi_1, \psi_2\rangle\f$
   *
   *  Detailed description
   *
   *  @return Returns a vector [(entry_t...)] which contains the non-zero
   *    entries of the sparse matrix arising from the overlap integral above.
   *    entry_t has members row, col and val. row and col correspond to the
   *    enumeration of DoFs in the spectral basis.
   */
  auto s1() const -> const decltype(t1_) & { return t1_; }

  /**
   *   @brief same as s1
   */
  auto t1() const -> const decltype(t1_) & { return t1_; }

  /**
   *   @brief \f$\langle v_i v_j \psi_1, \psi_2 \rangle\f$
   *
   *  Detailed description
   *
   *  @return Returns a vector [(entry_t...)] which contains the non-zero
   *    entries of the sparse matrix arising from the overlap integral above.
   *    entry_t has members row, col and val. row and col correspond to the
   *    enumeration of DoFs in the spectral basis.
   */
  auto t2() const -> const decltype(t2_) & { return t2_; }

  const VelocityAngularIntegrator<2>& get_vai() const { return vai; }

  void print_info() const;
};

// --------------------------------------------------------------------------------
template <typename BASIS>
void
VelocityVarForm<2>::init(const BASIS& test_basis, const BASIS& trial_basis, const double beta)
{
  typedef typename BASIS::elem_t elem_t;

  // angular basis
  typedef typename std::tuple_element<0, typename BASIS::elem_t::container_t>::type angular_elem_t;
  // radial basis
  typedef typename std::tuple_element<1, typename BASIS::elem_t::container_t>::type radial_elem_t;
  /// vector of basis functions in dim1
  typedef typename BASIS::DimAcc::template get_vec<angular_elem_t> a1_t;
  typedef typename BASIS::DimAcc::template get_vec<radial_elem_t> a2_t;

  typedef typename radial_elem_t::id_t rid_t;
  typedef typename radial_elem_t::numeric_t numeric_t;

  // element accessors for radial and angular parts
  typename elem_t::Acc::template get<radial_elem_t> acc_rad;
  typename elem_t::Acc::template get<angular_elem_t> acc_ang;

  /// they correspond to rows and columns in R1
  const auto& test_angular_basis = a1_t()(test_basis);
  //  const auto& test_radial_basis  = a2_t()(test_basis);

  const auto& trial_angular_basis = a1_t()(trial_basis);
  // const auto& trial_radial_basis  = a2_t()(trial_basis);

  // matrix assembly
  vai.init(trial_angular_basis);
  // regular L2-inner product
  L2Weight weight(beta);

  // ------------------------------
  // Compute Radial Entries
  local_::radial_entry_computer_t radial_entry_computer;
  // create a cache for the radial entries
  std::unordered_map<std::tuple<rid_t, rid_t>, numeric_t> cache;

  typedef std::pair<unsigned int, unsigned int> index_pair_t;

  // ----------------------------------------------------------------------
  // S0
  s0_.reinit(trial_basis.n_dofs());
  cache.clear();
  auto k0 = [](double r __attribute__((unused))) { return 1; };
  // iterate over nonzero entries in s0
  const auto& s0A = vai.get_s0();
  for (auto itA = s0A.begin(); itA != s0A.end(); ++itA) {
    // get ids from anuglar basis
    unsigned int iA1 = itA->first.first;
    unsigned int iA2 = itA->first.second;
    const auto& idA1 = test_angular_basis[iA1].get_id();
    const auto& idA2 = trial_angular_basis[iA2].get_id();

    std::function<bool(const elem_t&)> pred1 = [&](const elem_t& e) {
      return acc_ang(e).get_id() == idA1;
    };

    std::function<bool(const elem_t&)> pred2 = [&](const elem_t& e) {
      return acc_ang(e).get_id() == idA2;
    };

    auto test_range_rad = filtered_range(test_basis.begin(), test_basis.end(), pred1);
    auto trial_range_rad = filtered_range(trial_basis.begin(), trial_basis.end(), pred2);

    // iterate over test functions
    for (auto itR1 = std::get<0>(test_range_rad); itR1 != std::get<1>(test_range_rad); ++itR1) {
      const auto& br1 = acc_rad(*itR1);  // extract radial part
      const auto& id1 = br1.get_id();

      // iterate over trial functions
      for (auto itR2 = std::get<0>(trial_range_rad); itR2 != std::get<1>(trial_range_rad); ++itR2) {
        const auto& br2 = acc_rad(*itR2);  // extract radial part
        const auto& id2 = br2.get_id();
        // create key
        auto key = std::make_tuple(id1, id2);
        // lookup
        auto cit = cache.find(key);
        // found?
        double vr;
        if (cit != cache.end()) {
          vr = cit->second;
        } else {
          vr = radial_entry_computer.compute(k0, br1, br2, 1 - 2.0 / beta);
          cache[key] = vr;
        }

        // insert value into entries structure
        if (std::abs(vr) > TOL) {
          unsigned int gidx1 = test_basis.get_dof_index(itR1->get_id());
          unsigned int gidx2 = trial_basis.get_dof_index(itR2->get_id());
          s0_.insert(gidx1, gidx2, (itA->second) * vr);
        }
      }
    }
  }
  s0_.compress();

  // ----------------------------------------------------------------------
  // T1
  t1_.reinit(trial_basis.n_dofs());
  cache.clear();
  auto k1 = [](double r) { return r; };
  // iterate over nonzero entries in t1
  const auto& t1A = vai.get_t1();
  for (auto itA = t1A.begin(); itA != t1A.end(); ++itA) {
    // get ids from anuglar basis
    unsigned int iA1 = itA->first.first;
    unsigned int iA2 = itA->first.second;
    const auto& idA1 = test_angular_basis[iA1].get_id();
    const auto& idA2 = trial_angular_basis[iA2].get_id();

    std::function<bool(const elem_t&)> pred1 = [&](const elem_t& e) {
      return acc_ang(e).get_id() == idA1;
    };
    std::function<bool(const elem_t&)> pred2 = [&](const elem_t& e) {
      return acc_ang(e).get_id() == idA2;
    };

    auto test_range_rad = filtered_range(test_basis.begin(), test_basis.end(), pred1);
    auto trial_range_rad = filtered_range(trial_basis.begin(), trial_basis.end(), pred2);

    // iterate over test functions
    for (auto itR1 = std::get<0>(test_range_rad); itR1 != std::get<1>(test_range_rad); ++itR1) {
      const auto& br1 = acc_rad(*itR1);  // extract radial part
      const auto& id1 = br1.get_id();

      // iterate over trial functions
      for (auto itR2 = std::get<0>(trial_range_rad); itR2 != std::get<1>(trial_range_rad); ++itR2) {
        const auto& br2 = acc_rad(*itR2);  // extract radial part
        const auto& id2 = br2.get_id();
        // create key
        auto key = std::make_tuple(id1, id2);
        // lookup
        auto cit = cache.find(key);
        // found?
        double vr;
        if (cit != cache.end()) {
          vr = cit->second;
        } else {
          vr = radial_entry_computer.compute(k1, br1, br2, 1 - 2.0 / beta);
          cache[key] = vr;
        }

        // insert value into entries structure
        if (std::abs(vr) > TOL) {
          unsigned int gidx1 = test_basis.get_dof_index(itR1->get_id());
          unsigned int gidx2 = trial_basis.get_dof_index(itR2->get_id());
          t1_.insert(gidx1, gidx2, (itA->second) * vr);
        }
      }
    }
  }
  t1_.compress();

  // ----------------------------------------------------------------------
  // T2
  t2_.reinit(trial_basis.n_dofs());
  cache.clear();
  auto k2 = [](double r) { return r * r; };
  const auto& t2A = vai.get_t2();
  for (auto itA = t2A.begin(); itA != t2A.end(); ++itA) {
    // get ids from anuglar basis
    unsigned int iA1 = itA->first.first;
    unsigned int iA2 = itA->first.second;
    const auto& idA1 = test_angular_basis[iA1].get_id();
    const auto& idA2 = trial_angular_basis[iA2].get_id();

    std::function<bool(const elem_t&)> pred1 = [&](const elem_t& e) {
      return acc_ang(e).get_id() == idA1;
    };
    std::function<bool(const elem_t&)> pred2 = [&](const elem_t& e) {
      return acc_ang(e).get_id() == idA2;
    };

    auto test_range_rad = filtered_range(test_basis.begin(), test_basis.end(), pred1);
    auto trial_range_rad = filtered_range(trial_basis.begin(), trial_basis.end(), pred2);

    // iterate over test functions
    for (auto itR1 = std::get<0>(test_range_rad); itR1 != std::get<1>(test_range_rad); ++itR1) {
      const auto& br1 = acc_rad(*itR1);  // extract radial part
      const auto& id1 = br1.get_id();

      // iterate over trial functions
      for (auto itR2 = std::get<0>(trial_range_rad); itR2 != std::get<1>(trial_range_rad); ++itR2) {
        const auto& br2 = acc_rad(*itR2);  // extract radial part
        const auto& id2 = br2.get_id();
        // create key
        auto key = std::make_tuple(id1, id2);
        // lookup
        auto cit = cache.find(key);
        // found?
        double vr;
        if (cit != cache.end()) {
          vr = cit->second;
        } else {
          vr = radial_entry_computer.compute(k2, br1, br2, 1 - 2.0 / beta);
          cache[key] = vr;
        }

        // insert value into entries structure
        if (std::abs(vr) > TOL) {
          unsigned int gidx1 = test_basis.get_dof_index(itR1->get_id());
          unsigned int gidx2 = trial_basis.get_dof_index(itR2->get_id());
          t2_.insert(gidx1, gidx2, (itA->second) * vr);
        }
      }
    }
  }
  t2_.compress();
}

// ------------------------------------------------------------
template <typename BASIS>
void
VelocityVarForm<2>::init(const BASIS& trial_basis, const double beta)
{
  this->init(trial_basis, trial_basis, beta);
}

// -------------------------------------------------------------
inline void
VelocityVarForm<2>::print_info() const
{
  std::cout << "s0_.size()"
            << "\t" << s0_.size() << std::endl
            << "t1_.size()"
            << "\t" << t1_.size() << std::endl
            << "t2_.size()"
            << "\t" << t2_.size() << std::endl;
}

}  // end namespace boltzmann

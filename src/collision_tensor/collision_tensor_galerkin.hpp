#pragma once

// system includes ------------------------------------------------------------
#include <iostream>
#include <vector>
#include "aux/exceptions.h"

#include "collision_tensor_galerkin_base.hpp"


namespace boltzmann {
class CollisionTensorGalerkin : public CollisionTensorGalerkinBase
{
 public:
  CollisionTensorGalerkin(const basis_t& basis)
      : CollisionTensorGalerkinBase(basis)
      , slices_(basis.n_dofs())
  { /* empty */ }

  void apply(double* out, const double* in) const;
  void apply(double* out, const double* in, const int L) const;
  void apply_adaptive(double* out, const double* in, int nmax) const;

  const std::vector<sparse_matrix_t>& slices() const { return slices_; }
  const sparse_matrix_t& get(int j) const;

  /**
   * @brief read tensor from file
   *
   * @param fname Filename
   */
  void read_hdf5(const char* fname);

  unsigned long nnz() const;

 protected:
  std::vector<sparse_matrix_t> slices_;
};

// ------------------------------------------------------------
inline void
CollisionTensorGalerkin::apply(double* out, const double* in) const
{
  typedef Eigen::Map<const vec_t> cvec_t;  // constant vector
  typedef Eigen::Map<vec_t> mvec_t;        // mutable vector

  mvec_t vout(out, N_);
  cvec_t vin(in, N_);

#pragma omp parallel for
  for (int i = 0; i < N_; ++i) {
    vout(i) = vin.dot(slices_[i] * vin);
  }

  vout = Sinv_ * vout;
}

// ------------------------------------------------------------
inline void
CollisionTensorGalerkin::apply_adaptive(double* out, const double* in, int nmax) const
{
  typedef Eigen::Map<const vec_t> cvec_t;  // constant vector
  typedef Eigen::Map<vec_t> mvec_t;        // mutable vector

  mvec_t vout(out, N_);
  cvec_t vin(in, nmax);

#pragma omp parallel for
  for (int i = 0; i < N_; ++i) {
    vout(i) = vin.dot(slices_[i].topLeftCorner(nmax, nmax) * vin);
  }
  vout = Sinv_ * vout;
}

// ----------------------------------------------------------------------
inline void
CollisionTensorGalerkin::apply(double* out, const double* in, const int L) const
{
  typedef Eigen::Map<Eigen::VectorXd> vec_t;
  typedef Eigen::Map<const Eigen::VectorXd> const_vec_t;

  for (int j = 0; j < N_; j++) {
    for (int l = 0; l < L; ++l) {
      const_vec_t vin(in + N_ * l, N_);
      out[N_ * l + j] = vin.dot(slices_[j] * vin);
    }
  }
  for (int l = 0; l < L; ++l) {
    vec_t vout(out + N_ * l, N_);
    vout = Sinv_ * vout;
  }
}

}  // namespace boltzmann

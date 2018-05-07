#pragma once

// system includes ------------------------------------------------------------
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>
#include "aux/exceptions.h"

namespace boltzmann {
class CollisionTensor
{
 private:
  typedef Eigen::SparseMatrix<double> sparse_matrix_t;
  typedef std::shared_ptr<sparse_matrix_t> ptr_t;
  typedef Eigen::SparseLU<sparse_matrix_t> lu_t;

 public:
  CollisionTensor(int N);
  CollisionTensor()
      : N_(0)
  {
  }

  void apply(double* out, const double* in) const;
  void apply(double* out, const double* in, double* buffer) const;

  /**
   *
   *
   * @param out
   * @param in
   * @param L        #local phys. DoFs
   * @param buffer   external buffer of length N*L
   * @param local_bd_indicator true: is at boundary, do not apply scattering,
   *                           false: inner vertex, apply scattering
   */
  void apply(double* out,
             const double* in,
             const unsigned int L,
             double* buffer,
             const std::vector<bool>& local_bd_indicator = std::vector<bool>()) const;

  void apply_adaptive(double* out, const double* in, int nmax) const;
  void add(ptr_t& slice, unsigned int j);
  const sparse_matrix_t& get(int j);

  void set_mass_matrix(sparse_matrix_t& m);
  const lu_t& get_invm() const { return lu_; }
  const sparse_matrix_t& mass_matrix() const { return mass_matrix_; }

  const std::vector<ptr_t>& slices() const { return slices_; }

  /**
   * @brief read tensor from file
   *
   * @param fname Filename
   * @param N     #DoFs
   */
  void read_hdf5(const char* fname, const int N);

 private:
  /// basis size
  unsigned int N_;
  /// buffer
  mutable Eigen::VectorXd vtmp;
  /// tensor entries
  std::vector<ptr_t> slices_;
  /// mass matrix
  sparse_matrix_t mass_matrix_;
  /// inverse of mass matrix
  lu_t lu_;
};

// ------------------------------------------------------------
inline void
CollisionTensor::apply(double* out, const double* in) const
{
  assert(mass_matrix_.rows() > 0);
  typedef Eigen::Map<Eigen::VectorXd> vec_t;
  typedef Eigen::Map<const Eigen::VectorXd> const_vec_t;
  const_vec_t v_in(in, N_);
  vec_t v_out(out, N_);

#pragma omp parallel for
  for (unsigned int j = 0; j < N_; ++j) {
    vtmp[j] = v_in.dot((*slices_[j]) * v_in);
  }

  // this is not parallel
  v_out = lu_.solve(vtmp);
}

// ------------------------------------------------------------
inline void
CollisionTensor::apply_adaptive(double* out, const double* in, int nmax) const
{
  assert(mass_matrix_.rows() > 0);
  typedef Eigen::Map<Eigen::VectorXd> vec_t;
  typedef Eigen::Map<const Eigen::VectorXd> const_vec_t;
  const_vec_t v_in(in, nmax);
  vec_t v_out(out, N_);

// is this load-balanced? probably not
#pragma omp parallel for
  for (unsigned int j = 0; j < N_; ++j) {
    vtmp[j] = v_in.dot((*slices_[j]).topLeftCorner(nmax, nmax) * v_in);
  }

  // this is not parallel
  v_out = lu_.solve(vtmp);
}

// ------------------------------------------------------------
inline void
CollisionTensor::apply(double* out, const double* in, double* buffer) const
{
  assert(mass_matrix_.rows() > 0);
  typedef Eigen::Map<Eigen::VectorXd> vec_t;
  typedef Eigen::Map<const Eigen::VectorXd> const_vec_t;
  const_vec_t v_in(in, N_);
  vec_t vbuf(buffer, N_);
  vec_t v_out(out, N_);
#pragma omp parallel for schedule(guided)
  for (unsigned int j = 0; j < N_; ++j) {
    vbuf[j] = v_in.dot((*slices_[j]) * v_in);
  }
  v_out = lu_.solve(vbuf);
}

// ------------------------------------------------------------
inline void
CollisionTensor::apply(double* out,
                       const double* in,
                       const unsigned int L,
                       double* buffer,
                       const std::vector<bool>& local_bd_indicator) const
{
  assert(mass_matrix_.rows() > 0);
  typedef Eigen::Map<Eigen::VectorXd> vec_t;
  typedef Eigen::Map<const Eigen::VectorXd> const_vec_t;

  vec_t vbuf(buffer, N_ * L);

  if (local_bd_indicator.size() == 0) {
#pragma omp parallel
    {
#pragma omp for schedule(guided)
      for (unsigned int j = 0; j < N_; ++j) {
        for (unsigned int l = 0; l < L; ++l) {
          const_vec_t v_in(in + N_ * l, N_);
          vbuf[N_ * l + j] = v_in.dot((*slices_[j]) * v_in);
        }
      }

#pragma omp for schedule(static)
      for (unsigned int l = 0; l < L; ++l) {
        vec_t vout(out + N_ * l, N_);
        const_vec_t vbuf(buffer + N_ * l, N_);
        vout = lu_.solve(vbuf);
      }
    }
  } else if (local_bd_indicator.size() == L) {
#pragma omp parallel
    {
#pragma omp for schedule(guided)
      for (unsigned int j = 0; j < N_; ++j) {
        for (unsigned int l = 0; l < L; ++l) {
          if (local_bd_indicator[l]) continue;
          const_vec_t v_in(in + N_ * l, N_);
          vbuf[N_ * l + j] = v_in.dot((*slices_[j]) * v_in);
        }
      }

#pragma omp for schedule(static)
      for (unsigned int l = 0; l < L; ++l) {
        if (local_bd_indicator[l]) continue;
        vec_t vout(out + N_ * l, N_);
        const_vec_t vbuf(buffer + N_ * l, N_);
        vout = lu_.solve(vbuf);
      }
    }

  }
#ifdef DEBUG
  else {
    BAssertThrow(false, "something went wrong with bd_indicator");
  }
#endif
}

}  // end boltzmann

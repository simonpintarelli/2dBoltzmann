#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/trilinos_vector.h>

#include "collision_tensor.hpp"
#include "collision_tensor_galerkin.hpp"
#include "dense/collision_tensor_zlastAM.hpp"
#include "dense/collision_tensor_zlastAM_eigen.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"

namespace boltzmann {
// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
class CollisionTensorOperatorBase
{
 protected:
  typedef typename SpectralBasisFactoryKS::basis_type spectral_basis_t;
  typedef dealii::TrilinosWrappers::MPI::Vector trilinos_vector_t;

 protected:
  CollisionTensorOperatorBase(MPI_Comm& communicator)
      : comm(communicator)
  { /* empty */
  }

 public:
  /// explicit Euler step
  virtual void apply(trilinos_vector_t& out, double dt) const = 0;
  /// load tensor from HDF5-file
  virtual void load_tensor(std::string fname) = 0;
  void set_truncation_threshold(double tre);


 protected:
  MPI_Comm comm;
  double truncate_treshold_ = 0;
  bool use_treshold_ = false;

};

// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
class CollisionTensorOperatorPG : public CollisionTensorOperatorBase
{
 public:
  CollisionTensorOperatorPG(const dealii::DoFHandler<2>& dh,
                            const spectral_basis_t& basis,
                            const dealii::IndexSet& parallel_partitioning,
                            MPI_Comm communicator = MPI_COMM_WORLD)
      : CollisionTensorOperatorBase(communicator)
      , n_phys_dofs(dh.n_dofs())
      , n_velo_dofs(basis.n_dofs())
      , Q(basis.n_dofs())
      , vtmp(parallel_partitioning, communicator)
      , local_buffer(parallel_partitioning.n_elements())
  { /* empty */ }

  virtual void apply(trilinos_vector_t& out, double dt) const;
  virtual void load_tensor(std::string fname);

 private:
  const unsigned int n_phys_dofs;
  const unsigned int n_velo_dofs;
  CollisionTensor Q;
  mutable trilinos_vector_t vtmp;
  mutable std::vector<double> local_buffer;
};

// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
class CollisionTensorOperatorG : public CollisionTensorOperatorBase
{
 public:
  CollisionTensorOperatorG(const dealii::DoFHandler<2>& dh,
                           const spectral_basis_t& basis,
                           const dealii::IndexSet& parallel_partitioning,
                           MPI_Comm communicator = MPI_COMM_WORLD)
      : CollisionTensorOperatorBase(communicator)
      , n_phys_dofs(dh.n_dofs())
      , n_velo_dofs(basis.n_dofs())
      , n_local_phys_dofs_(parallel_partitioning.n_elements() / basis.n_dofs())
      , Q(basis)
      , vtmp(parallel_partitioning, communicator)
  {
    lambda.resize(4, n_local_phys_dofs_);
    lambda_prev.resize(4, n_local_phys_dofs_);
  }

  virtual void apply(trilinos_vector_t& out, double dt) const;
  virtual void load_tensor(std::string fname);

 private:
  const unsigned int n_phys_dofs;
  const unsigned int n_velo_dofs;
  const unsigned int n_local_phys_dofs_;
  CollisionTensorGalerkin Q;
  mutable trilinos_vector_t vtmp;
  /// conservation of momentum
  typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> carray_t;
  mutable carray_t lambda;
  mutable carray_t lambda_prev;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};

#ifdef USE_MPI
// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
template <typename CT_DENSE = ct_dense::CollisionTensorZLastAM>
class CollisionTensorOperatorDense : public CollisionTensorOperatorBase
{
 public:
  CollisionTensorOperatorDense(const dealii::DoFHandler<2>& dh,
                               const spectral_basis_t& basis,
                               const dealii::IndexSet& parallel_partitioning,
                               int vblksize = 1,
                               MPI_Comm communicator = MPI_COMM_WORLD)
      : CollisionTensorOperatorBase(communicator)
      , basis_(basis)
      , n_phys_dofs_(dh.n_dofs())
      , n_velo_dofs_(basis.n_dofs())
      , vblksize_(vblksize)
      , n_local_phys_dofs_(parallel_partitioning.n_elements() / basis.n_dofs())
      , Q_(basis, parallel_partitioning.n_elements() / basis.n_dofs())
      , vtmp(parallel_partitioning, communicator)
  {
    /* empty */
  }

 private:
  const spectral_basis_t& basis_;
  const unsigned int n_phys_dofs_;
  const unsigned int n_velo_dofs_;
  const int vblksize_;
  const int n_local_phys_dofs_;
  CT_DENSE Q_;

 private:
  mutable trilinos_vector_t vtmp;
  typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> in_array_t;
  /// padded input array
  mutable in_array_t padded_in;
  /// conservation of momentum
  mutable in_array_t lambda;
  mutable in_array_t lambda_prev;

 public:
  virtual void apply(trilinos_vector_t& out, double dt) const override;
  virtual void load_tensor(std::string fname) override;


 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <typename CT_DENSE>
void
CollisionTensorOperatorDense<CT_DENSE>::apply(trilinos_vector_t& out, double dt) const
{
  BOOST_ASSERT(out.local_size()/n_velo_dofs_ == n_local_phys_dofs_);
  BOOST_ASSERT(vtmp.local_size()/n_velo_dofs_ == n_local_phys_dofs_);

  const double* data = out.begin();

  Eigen::Map<const in_array_t> vin(data, n_velo_dofs_, n_local_phys_dofs_);
  int imax = -1;
  if (use_treshold_) {
    for (int i = n_velo_dofs_ -1; i >= 0; --i) {
      if ((vin.row(i).abs() > truncate_treshold_).any()) {
        imax = i;
        break;
      }
    }
  }
  // pad & copy to padded array
  Q_.pad(padded_in, vin);
  Eigen::Map<in_array_t> vout(out.begin(), n_velo_dofs_, n_local_phys_dofs_);
  // get lambda (moment conservation)
  Q_.get_lambda(lambda_prev, vin);
  // apply collision tensor write directly to output array
  Eigen::Map<in_array_t> vtmp_eigen(vtmp.begin(), n_velo_dofs_, n_local_phys_dofs_);
  Q_.apply(vtmp_eigen, padded_in, use_treshold_ ? imax : -1);
  // explicit Euler timestep
  out.sadd(1.0, dt, vtmp);
  // get contribution to lambda from next timestep
  Q_.get_lambda(lambda, vout);
  lambda -= lambda_prev;
  // conserve moments
  Q_.project_lambda(vout, lambda);
}

template <typename CT_DENSE>
void
CollisionTensorOperatorDense<CT_DENSE>::load_tensor(std::string fname)
{
  Q_.import_entries_mpishmem(fname, vblksize_);

  // initialize buffer arrays used during ::apply
  int npadded = Q_.padded_vector_length();
  padded_in.resize(npadded, n_local_phys_dofs_);
  // this needs to be done only once, the parts used to
  // store the input vector are overwritten, and the elements in between
  // are multiplied by zero during Q_.apply, just make sure they are not NaN's by chance.
  padded_in.setZero();

  // temporary arrays for conservation of momentum
  lambda_prev.resize(4, n_local_phys_dofs_);
  lambda.resize(4, n_local_phys_dofs_);
}


#endif

}  // namespace boltzmann

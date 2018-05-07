#include <boost/assert.hpp>
#include <boost/filesystem.hpp>

#include "collision_tensor_operator.hpp"
#include "dense/multi_slices_factory.hpp"
#include "spectral/basis/toolbox/spectral_basis.hpp"

namespace boltzmann {

void CollisionTensorOperatorBase::set_truncation_threshold(double tre)
{
  if (tre >= 0) {
    truncate_treshold_ = tre;
    use_treshold_ = true;
  } else {
    use_treshold_ = false;
  }
}

// -------------------------------------------------------------------------------------
void
CollisionTensorOperatorPG::apply(trilinos_vector_t& out, double dt) const
{
  typedef typename trilinos_vector_t::iterator ptr_t;
  typedef typename trilinos_vector_t::const_iterator c_ptr_t;

  c_ptr_t in_begin = out.begin();
  c_ptr_t in_end = out.end();

  ptr_t buf_begin = vtmp.begin();
  ptr_t buf_end = vtmp.end();

  BOOST_ASSERT_MSG((in_end - in_begin) == (buf_end - buf_begin), "size mismatch");

  unsigned int local_size = in_end - in_begin;
  BOOST_ASSERT_MSG((local_size % n_velo_dofs) == 0, "block size mismatch");
  unsigned int local_phys_size = local_size / n_velo_dofs;

  Q.apply(buf_begin, in_begin, local_phys_size, local_buffer.data());
  out.sadd(1.0, dt, vtmp);
}

// -------------------------------------------------------------------------------------
void
CollisionTensorOperatorPG::load_tensor(std::string fname)
{
  int proc_id = dealii::Utilities::MPI::this_mpi_process(comm);
  if (proc_id == 0 && !boost::filesystem::exists(fname)) {
    throw std::runtime_error("Collision tensor file not found!");
  }

  Q.read_hdf5(fname.c_str(), n_velo_dofs);
}

// -------------------------------------------------------------------------------------
void
CollisionTensorOperatorG::apply(trilinos_vector_t& out, double dt) const
{
  typedef typename trilinos_vector_t::iterator ptr_t;

  ptr_t out_begin = out.begin();
  ptr_t out_end = out.end();

  ptr_t buf_begin = vtmp.begin();
  ptr_t buf_end = vtmp.end();

  BOOST_ASSERT_MSG((out_end - out_begin) == (buf_end - buf_begin), "size mismatch");
  Eigen::Map<const carray_t> vin(out_begin, n_velo_dofs, n_local_phys_dofs_);
  Q.get_lambda(lambda_prev, vin);

  unsigned int local_size = out_end - out_begin;
  BOOST_ASSERT_MSG((local_size % n_velo_dofs) == 0, "block size mismatch");
  unsigned int local_phys_size = local_size / n_velo_dofs;


  // apply collision tensor
  Q.apply(buf_begin, out_begin, local_phys_size);
  out.sadd(1.0, dt, vtmp);

  Eigen::Map<carray_t> vout(out_begin, n_velo_dofs, n_local_phys_dofs_);
  Q.get_lambda(lambda, vout);
  lambda = -lambda_prev;
  Q.project_lambda(vout, lambda);
}

// -------------------------------------------------------------------------------------
void
CollisionTensorOperatorG::load_tensor(std::string fname)
{
  int proc_id = dealii::Utilities::MPI::this_mpi_process(comm);
  if (proc_id == 0 && !boost::filesystem::exists(fname)) {
    throw std::runtime_error("Collision tensor file not found!");
  }

  Q.read_hdf5(fname.c_str());
}


}  // namespace boltzmann

#include "collision_tensor_galerkin_base.hpp"
#include "post_processing/macroscopic_quantities.hpp"
#include "spectral/utility/mass_matrix.hpp"

namespace boltzmann {
// --------------------------------------------------------------------------------------
CollisionTensorGalerkinBase::CollisionTensorGalerkinBase(const basis_t& basis)
    : basis_(basis)
    , N_(basis.n_dofs())
    , K_(boltzmann::spectral::get_K(basis))
    , buf_(basis.n_dofs())
{
  /* empty */
  MQEval Mq(basis);
  // l2 projection
  Ht_.resize(4, N_);
  Ht_.setZero();
  Ht_.block(0, 0, 1, Mq.cmass().rows()) = Mq.cmass().transpose();
  Ht_.block(1, 0, 1, Mq.cenergy().rows()) = Mq.cenergy().transpose();
  Ht_.block(2, 0, 1, Mq.cux().rows()) = Mq.cux().transpose();
  Ht_.block(3, 0, 1, Mq.cuy().rows()) = Mq.cuy().transpose();
  auto s = make_mass_vdiag(basis);
  vec_t sinv = s.array().inverse();
  //  Eigen::DiagonalMatrix<double, -1> Sinv(sinv);
  Sinv_ = diag_t(sinv);
  HtHinv_ = (Ht_ * Sinv_ * Ht_.transpose()).inverse();
  //  std::cout << "HtHinv_\n\n" << HtHinv_ << std::endl;
}

// ------------------------------------------------------------
void
CollisionTensorGalerkinBase::project(double* out, const double* in) const
{
  typedef Eigen::Map<const vec_t> cvec_t;  // constant vector
  typedef Eigen::Map<vec_t> mvec_t;        // mutable vector

  cvec_t vin(in, N_);
  mvec_t vout(out, N_);

  Eigen::Vector4d lambda = HtHinv_ * Ht_ * (vout - vin);
  vout -= Sinv_ * Ht_.transpose() * lambda;
}

}  // namespace boltzmann

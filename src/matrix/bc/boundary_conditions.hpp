#pragma once

// deal.II includes -------------------------------------------------------
#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>

#ifdef DEBUG
#include <deal.II/base/conditional_ostream.h>
#endif

// system includes --------------------------------------------------------
#include <yaml-cpp/yaml.h>
#include <boost/multi_array.hpp>
#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
// trilinos includes ------------------------------------------------------
#include <Epetra_Export.h>
#include <Epetra_Import.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>

// own includes -----------------------------------------------------------
#include "bc_traits.hpp"
#include "grid/dof_mapper_periodic_distributed.hpp"
#include "impl/bd_faces_manager_redist.hpp"
#include "impl/bd_faces_manager_simple.hpp"
#include "quadrature/qhermitew.hpp"
#include "quadrature/qmaxwell.hpp"
#include "spectral/basis/dof_mapper.hpp"
#include "spectral/basis/indexer.hpp"
#include "spectral/basis/spectral_basis_factory_hermite.hpp"
#include "spectral/hermite_to_nodal.hpp"
#include "spectral/polar_to_hermite.hpp"
#include "spectral/rotate_basis.hpp"
#include "base/numbers.hpp"


namespace boltzmann {
namespace local_ {
// helper struct to keep quadrature rules for inflow rhs
struct inflow_bc_quad
{
  /**
   *
   * @param npts quad. degree
   * @param Tw   inflow temperature
   * @param w    basis weight, e.g. exp(-r^2/2) => w=0.5
   *
   *
   * @return
   */
  inflow_bc_quad(int npts, double Tw, double w = 0.5);

  inflow_bc_quad() {}

  QHermiteW x_quad;
  QMaxwell y_quad;
  // hermite polynomials

  typedef HermiteNW<double> hermw_t;
  std::shared_ptr<hermw_t> hermwx;
  std::shared_ptr<hermw_t> hermwy;
};

inflow_bc_quad::inflow_bc_quad(int npts, double Tw, double w)
    : x_quad(w + 1. / (2 * Tw), npts)
    , y_quad(w + 1. / (2 * Tw), npts)
    , hermwx(std::make_shared<HermiteNW<double> >(npts))
    , hermwy(std::make_shared<HermiteNW<double> >(npts))
// 256 digits (mpfr accuracy)
{
  hermwx->compute(x_quad.pts());

  std::vector<double> y(npts);
  for (int i = 0; i < npts; ++i) {
    y[i] = -y_quad.pts(i);
  }

  hermwy->compute(y);
}
}  // end namespace local_

// -------------------------------------------------------------------------------------
template <typename METHOD, typename APP, typename BD_FACES_MANAGER>
class BoundaryConditions : public BD_FACES_MANAGER
{
 private:
  typedef BD_FACES_MANAGER bd_faces_manager_t;
  typedef dealii::types::global_dof_index size_type;
  typedef dealii::DoFHandler<2> dh_t;
  typedef typename METHOD::spectral_basis_t spectral_basis_t;
  typedef typename traits::DoFMapper<APP::bc_type>::type mapper_t;
  typedef Indexer<mapper_t> indexer_t;
  typedef dealii::TrilinosWrappers::MPI::Vector deal_vector_t;

  // boundary id
  typedef unsigned int bid_t;

 public:
  BoundaryConditions(double dt,
                     const dh_t& dh,
                     const spectral_basis_t& spectral_basis,
                     const indexer_t& indexer,
                     const YAML::Node& config);

  void apply(deal_vector_t& out, const deal_vector_t& in) const;

  void apply(Epetra_MultiVector& out, const Epetra_MultiVector& in) const;

  /**
   *  @brief Assemble inflow type boundary conditions into the right hand side
   *
   */
  template <typename VECTOR>
  void assemble_rhs(VECTOR& dst) const;

 private:
  typedef SpectralBasisFactoryHN::basis_type hermite_basis_t;
  typedef std::shared_ptr<impl::flux_worker> ptr_flux_worker;

 private:
  double dt_;
  const dh_t& dh_;
  const spectral_basis_t& spectral_basis_;
  const indexer_t& indexer_;
  const YAML::Node& config_;

  //@{
  /// spectral basis transformation / rotation operators
  RotateBasis<spectral_basis_t> R_;
  hermite_basis_t hermite_basis_;
  typedef Polar2Hermite<spectral_basis_t, hermite_basis_t> P2H_t;
  typedef Hermite2Nodal<hermite_basis_t> H2N_t;
  std::shared_ptr<P2H_t> ptr_P2H_;
  std::shared_ptr<H2N_t> ptr_H2N_;
  //@}

  std::map<bid_t, ptr_flux_worker> flux_workers_;
  std::set<bid_t> periodic_;

  //@{
  /*  Buffers for Lagrange coefficients
   *  1p: output from flux_worker for vertex 1
   *  2p: output from flux_worker for vertex 2
   */
  mutable Eigen::MatrixXd L1p_;  // output f (1. trial) for flux worker
  mutable Eigen::MatrixXd L1_;   // input f  (1. tiral function) ..
  mutable Eigen::MatrixXd L2p_;  // output f  (2. trial function) ..
  mutable Eigen::MatrixXd L2_;
  mutable Eigen::MatrixXd L_;
  //@}

  /// buffer for hermite coefficients
  /// ybuffer
  mutable Eigen::VectorXd ybuf_;

  // polynomial degree
  int K_ = -1;

  // TODO Epetra Multi vector, Import, Export, Maps
  // imports from base class
  using bd_faces_manager_t::relevant_dofs_;
  using bd_faces_manager_t::get_faces_list;
  // epetra vectors
  mutable std::shared_ptr<Epetra_Vector> x_ghosted_;
  mutable std::shared_ptr<Epetra_Vector> y_ghosted_;
  mutable std::shared_ptr<Epetra_Import> importer_;
  mutable std::shared_ptr<Epetra_Export> exporter_;
};

// --------------------------------------------------------------------------------
template <typename METHOD, typename APP, typename BD_FACES_MANAGER>
BoundaryConditions<METHOD, APP, BD_FACES_MANAGER>::BoundaryConditions(
    double dt,
    const dh_t& dh,
    const spectral_basis_t& spectral_basis,
    const indexer_t& indexer,
    const YAML::Node& config)
    : bd_faces_manager_t(dh, spectral_basis, indexer)
    , dt_(dt)
    , dh_(dh)
    , spectral_basis_(spectral_basis)
    , indexer_(indexer)
    , config_(config)
    , R_(spectral_basis)
{
  int K_ = config_["SpectralBasis"]["deg"].as<int>();

  // initialize hermite basis
  SpectralBasisFactoryHN::create(hermite_basis_, K_, 2);
  static QHermiteW hermite_quad(1.0, K_);

  Eigen::VectorXd hw = hermite_quad.vwts<1>();
  Eigen::VectorXd hx = hermite_quad.vpts<1>();

  // resize buffers
  L1_.resize(K_, K_);
  L2_.resize(K_, K_);
  L1p_.resize(K_, K_);
  L2p_.resize(K_, K_);
  L_.resize(K_, K_);
  ybuf_.resize(spectral_basis_.n_dofs());

  typedef Eigen::MatrixXd mat_t;
  ptr_P2H_ = std::make_shared<P2H_t>(spectral_basis_, hermite_basis_);
  ptr_H2N_ = std::make_shared<H2N_t>(
      hermite_basis_, K_, [K_](mat_t& m1, mat_t& m2) { H2N_1d<>::create(m1, m2, K_); });

  auto node = config_["BoundaryDescriptors"];
  // TODO: process config
  for (YAML::const_iterator it = node.begin(); it != node.end(); ++it) {
    auto entry = it->second;
    id_t id = it->first.as<id_t>();
    std::string type = entry["type"].as<std::string>();

    typedef bc_traits<METHOD::lsq_type> bc_traits_t;
    if (std::strcmp(type.c_str(), "diffusive reflection") == 0) {
      double T = entry["T"].as<double>();
      double rho = 1.0;
      if (entry["rho"]) {
        rho = entry["rho"].as<double>();
      }
      if (entry["vt"]) {
        double vt = entry["vt"].as<double>();
        flux_workers_[id] =
            std::make_shared<typename bc_traits_t::DiffusiveReflection>(hw, hx, vt, T, rho);
      } else {
        flux_workers_[id] =
            std::make_shared<typename bc_traits_t::DiffusiveReflection>(hw, hx, 0, T, rho);
      }
    } else if (std::strcmp(type.c_str(), "diffusive reflection x") == 0) {
      std::string Tx = entry["Tx"].as<std::string>();
      if (entry["vt"]) {
        double vt = entry["vt"].as<double>();
        flux_workers_[id] =
            std::make_shared<typename bc_traits_t::DiffusiveReflectionX>(hw, hx, vt, Tx);
      } else {
        flux_workers_[id] =
            std::make_shared<typename bc_traits_t::DiffusiveReflectionX>(hw, hx, 0, Tx);
      }
    }
    else if (std::strcmp(type.c_str(), "specular reflection") == 0) {
      flux_workers_[id] = std::make_shared<typename bc_traits_t::SpecularReflection>(hw, hx);
    } else if (std::strcmp(type.c_str(), "inflow") == 0) {
      flux_workers_[id] = std::make_shared<typename bc_traits_t::Inflow>(hw, hx);

    } else if (std::strcmp(type.c_str(), "periodic") == 0) {
      periodic_.insert(id);
    } else {
      AssertThrow(false, dealii::ExcMessage("Unkown boundary condition entry"));
    }
  }

  auto epetra_map = relevant_dofs_.make_trilinos_map(MPI_COMM_WORLD, true);

  x_ghosted_ = std::make_shared<Epetra_Vector>(epetra_map);
  y_ghosted_ = std::make_shared<Epetra_Vector>(epetra_map);
}

// --------------------------------------------------------------------------------
template <typename METHOD, typename APP, typename BD_FACES_MANAGER>
template <typename VECTOR>
void
BoundaryConditions<METHOD, APP, BD_FACES_MANAGER>::assemble_rhs(VECTOR& dst) const
{
  /* inflow boundary conditions need to be assembled into the rhs vector. this is done in this
     routine */
  const int K = spectral::get_max_k(spectral_basis_);
  std::map<double, local_::inflow_bc_quad> inflow_bc_quad_map;
  auto& bd_conf = config_["BoundaryDescriptors"];
  const int npts = 2 * K;

  // --------------------------------------------------
  // prepare quadrature rules
  for (YAML::const_iterator it = bd_conf.begin(); it != bd_conf.end(); ++it) {
    auto entry = it->second;
    id_t id = it->first.as<id_t>();

    std::string type = entry["type"].as<std::string>();
    if (std::strcmp(type.c_str(), "inflow") == 0) {
      // currently inflow function g(v), can be of Maxwellian type only
      if (std::strcmp(entry["func"].as<std::string>().c_str(), "maxwellian") == 0) {
        double Tw = entry["T"].as<double>();
        // basis exp weight
        const double basis_alpha = 0.5;
        inflow_bc_quad_map[Tw] = local_::inflow_bc_quad(npts, Tw, basis_alpha);
      }
    }
  }

  // --------------------------------------------------
  // prepare deal.II stuff
  const int dimX = 2;
  unsigned int N = spectral_basis_.n_dofs();
  dealii::UpdateFlags update_flags = dealii::update_values | dealii::update_JxW_values |
                                     dealii::update_quadrature_points |
                                     dealii::update_normal_vectors;

  dealii::QGauss<dimX - 1> quad(2);
  int n_qpoints = quad.size();

  auto& fe = dh_.get_fe();
  const int dofs_per_cell = fe.dofs_per_cell;
  dealii::FEFaceValues<2> fe_face_values(fe, quad, update_flags);
  std::vector<size_type> local_dof_indices(fe.dofs_per_cell);

  const auto& faces_list = this->get_faces_list();

  typedef typename VECTOR::size_type size_type;
  std::vector<size_type> global_indices(N);

  Eigen::VectorXd II(N);
  Eigen::VectorXd IIp(N);    // buffer to add into trilinos vector
  Eigen::VectorXd IIloc(N);  // buffer to add into trilinos vector

  // iterate over faces at boundary
  for (const auto& mypair : faces_list) {
    const auto& cell = std::get<0>(mypair);
    int face_idx = std::get<1>(mypair);
    id_t bd_ind = cell.face(face_idx)->boundary_id();

    // check if bc is periodic.
    std::string type = config_["BoundaryDescriptors"][bd_ind]["type"].as<std::string>();
    if (periodic_.find(bd_ind) != periodic_.end() || std::strcmp(type.c_str(), "inflow") != 0)
      // bc is periodic: nothing to do
      continue;
    // --------------------------------------------------
    // load parameter
    auto params = config_["BoundaryDescriptors"][bd_ind];

    std::string func = params["func"].as<std::string>();
    if (std::strcmp(func.c_str(), "zero") == 0) {
      // nothing to do
      continue;
    }
    // check the inflow function
    BAssertThrow(std::strcmp(func.c_str(), "maxwellian") == 0,
                 "BoundaryConditions::assemble_rhs, do not know how to handle boundary type.\n" +
                     "Type was: " + func);

    double Tw = params["T"].as<double>();

    auto iter = inflow_bc_quad_map.find(Tw);
    assert(iter != inflow_bc_quad_map.end());
    const auto& quad_helper = iter->second;
    Eigen::Vector2d v0 = {0, 0};
    if (params["v"]) {
      double vx = params["v"][0].as<double>();
      double vy = params["v"][1].as<double>();
      v0 = {vx, vy};
    }

    double rho;
    if (params["rho"])
      rho = params["rho"].as<double>();
    else
      rho = 1;

    typedef dealii::DoFCellAccessor<dealii::DoFHandler<2>, false> accessor_t;
    typedef dealii::TriaIterator<accessor_t> tria_iterator_t;

    cell.get_dof_indices(local_dof_indices);
    fe_face_values.reinit(tria_iterator_t(cell), face_idx);

    const double nx = fe_face_values.normal_vector(0)[0];
    const double ny = fe_face_values.normal_vector(0)[1];
    // winkel zwischen y-achse und n
    const double alpha = numbers::PI / 2 - std::atan2(ny, nx);

    std::array<double, 4> B;
    B.fill(0);
    for (unsigned int l1 = 0; l1 < fe.dofs_per_cell; ++l1) {
      double val = 0;
      for (int q = 0; q < n_qpoints; ++q) {
        val += fe_face_values.shape_value(l1, q) * fe_face_values.JxW(q);
      }
      B[l1] = val;
    }

    const double v02 = v0.squaredNorm();
    // hermite basis accessors
    typedef typename hermite_basis_t::elem_t elem_t;
    typedef typename boost::mpl::at_c<typename elem_t::types_t, 0>::type hx_t;
    typedef typename boost::mpl::at_c<typename elem_t::types_t, 1>::type hy_t;
    typename elem_t::Acc::template get<hx_t> get_hx;
    typename elem_t::Acc::template get<hy_t> get_hy;

    auto hermwx = quad_helper.hermwx;
    auto hermwy = quad_helper.hermwy;
    const auto& xpts = quad_helper.x_quad.pts();
    const auto& xwts = quad_helper.x_quad.wts();
    const auto& ypts = quad_helper.y_quad.pts();
    const auto& ywts = quad_helper.y_quad.wts();

    Eigen::Rotation2D<double> rot2d(alpha);
    Eigen::Matrix2d rotm = rot2d.toRotationMatrix();
    auto v0h = rotm * v0;
    /*
     *  HINT: about Hermite quad. rule: (used in x-direction) The weights are
     *     multiplied by e^(x^2/2), which accounts for the factor e^(-x^2/2)
     *     which is included in the evaluation of the Hermite polynomials (e.g.
     *     Hermite functions in this case)
     */
    for (unsigned int j = 0; j < N; ++j) {
      // iterate over test functions
      const auto& elem = hermite_basis_.get_elem(j);
      const unsigned int jx = get_hx(elem).get_id().k;
      const unsigned int jy = get_hy(elem).get_id().k;
      double val = 0;
      // 2d quadrature
      for (int qx = 0; qx < npts; ++qx) {
        for (int qy = 0; qy < npts; ++qy) {
          Eigen::Vector2d vh = {xpts[qx], -1.0 * ypts[qy]};
          // inflow => -1.0
          val -= hermwx->get(jx)[qx] * hermwy->get(jy)[qy] * std::exp(v0h.dot(vh) / Tw) *
                 (std::exp(-xpts[qx] * xpts[qx] / 2 / Tw) * xwts[qx]) *
                 (std::exp(vh[1] * vh[1] / 2) * ywts[qy]);
        }
      }
      II(j) = std::exp(-v02 / 2 / Tw) * val * rho / (2 * numbers::PI * Tw);
    }
    // transform to polar
    ptr_P2H_->to_hermite_T(IIp, II);
    // rotate back
    R_.apply(IIloc.data(), IIp.data(), -alpha);

    for (unsigned int l1 = 0; l1 < fe.dofs_per_cell; ++l1) {
      if (std::abs(B[l1]) > 1e-15) {
        for (unsigned int j = 0; j < N; ++j) {
          global_indices[j] = indexer_.to_global(local_dof_indices[l1], j);
        }
        II = IIloc * B[l1];
        dst.add(N, global_indices.data(), II.data());
      }
    }
  }

  dst.compress(dealii::VectorOperation::add);
}

// --------------------------------------------------------------------------------
template <typename METHOD, typename APP, typename BD_FACES_MANAGER>
void
BoundaryConditions<METHOD, APP, BD_FACES_MANAGER>::apply(deal_vector_t& out,
                                                         const deal_vector_t& in) const
{
  this->apply(out.trilinos_vector(), in.trilinos_vector());
}

// --------------------------------------------------------------------------------
template <typename METHOD, typename APP, typename BD_FACES_MANAGER>
void
BoundaryConditions<METHOD, APP, BD_FACES_MANAGER>::apply(Epetra_MultiVector& out,
                                                         const Epetra_MultiVector& in) const
{
#ifdef DEBUG
  out.SetTracebackMode(2);
  in.SetTracebackMode(2);
  x_ghosted_->SetTracebackMode(2);
  y_ghosted_->SetTracebackMode(2);
#endif

  const int dimX = 2;
  // ATTENTION: this makes assumptions on deal.II internals
  static size_type faces_vertex[4][2] = {{0, 2}, {1, 3}, {0, 1}, {2, 3}};

  static dealii::Tensor<2, 2> B;

#ifdef DEBUG
  const unsigned int pid = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  dealii::ConditionalOStream pcout(std::cout, pid == 0);
#endif

  if (importer_.use_count() == 0) {
#ifdef DEBUG
    pcout << "Importer not set. Initializing\n";
#endif
    auto& ghosted_map = x_ghosted_->Map();
    auto& src_map = out.Map();
    importer_ = std::make_shared<Epetra_Import>(ghosted_map, src_map);
  }

  if (exporter_.use_count() == 0) {
#ifdef DEBUG
    pcout << "Exporter not set. Initializing\n";
#endif
    auto& ghosted_map = x_ghosted_->Map();
    auto& src_map = out.Map();
    exporter_ = std::make_shared<Epetra_Export>(ghosted_map, src_map);
  }

  // Import nonlocal elements
  int ier;
  ier = x_ghosted_->Import(in, *importer_.get(), Insert);
  AssertThrow(ier == 0, dealii::ExcTrilinosError(ier));
  // get pointer to x_ghosted_
  const double* data_in = x_ghosted_->Values();
  // reset y_ghosted_
  std::fill(y_ghosted_->Values(), y_ghosted_->Values() + y_ghosted_->MyLength(), 0);

  double* data_out = y_ghosted_->Values();

  unsigned int N = spectral_basis_.n_dofs();
  dealii::UpdateFlags update_flags = dealii::update_values | dealii::update_JxW_values |
                                     dealii::update_quadrature_points |
                                     dealii::update_normal_vectors;

  dealii::QGauss<dimX - 1> quad(2);
  int n_qpoints = quad.size();

  auto& fe = dh_.get_fe();
  dealii::FEFaceValues<2> fe_face_values(fe, quad, update_flags);
  std::vector<size_type> local_dof_indices(fe.dofs_per_face);

  // transform to lagrange coefficients
  unsigned int local_size = in.MyLength();
  const auto& faces_list = this->get_faces_list();
  // rotated Polar coeffs
  Eigen::VectorXd cP_rot(N);
  // Hermite coeffs
  Eigen::VectorXd cH(N);

  for (const auto& mypair : faces_list) {
    const auto& cell = std::get<0>(mypair);
    int face_idx = std::get<1>(mypair);
    id_t bd_ind = cell.face(face_idx)->boundary_id();

    // this is a periodic boundary and thus requires no further computations
    // (since it is built into the enumeration of DoFs already)
    if (periodic_.find(bd_ind) != periodic_.end()) continue;

    auto fentry = flux_workers_.find(bd_ind);
    if (fentry == flux_workers_.end()) {
      AssertThrow(false, dealii::ExcMessage("unknown boundary type"));
    }

    typedef dealii::DoFCellAccessor<dealii::DoFHandler<2>, false> accessor_t;
    typedef dealii::TriaIterator<accessor_t> tria_iterator_t;

    cell.face(face_idx)->get_dof_indices(local_dof_indices);

    fe_face_values.reinit(tria_iterator_t(cell), face_idx);

#ifdef DEBUG
    // checking dof indexing
    std::vector<size_type> ldof_indices(fe.dofs_per_cell);
    cell.get_dof_indices(ldof_indices);
    size_type l0 = cell.face(face_idx)->vertex_dof_index(0, 0);
    size_type l1 = cell.face(face_idx)->vertex_dof_index(1, 0);

    AssertThrow(l0 == ldof_indices[faces_vertex[face_idx][0]],
                dealii::ExcMessage("something with the local dofs is wrong"));
    AssertThrow(l1 == ldof_indices[faces_vertex[face_idx][1]],
                dealii::ExcMessage("something with the local dofs is wrong"));
    AssertThrow(l0 == local_dof_indices[0],
                dealii::ExcMessage("something with local dofs is wrong"));
    AssertThrow(l1 == local_dof_indices[1],
                dealii::ExcMessage("something with local dofs is wrong"));
#endif
    B *= 0;
    for (int i = 0; i < dimX; ++i) {
      int l1 = faces_vertex[face_idx][i];
      for (int j = 0; j < dimX; ++j) {
        int l2 = faces_vertex[face_idx][j];
        for (unsigned int q = 0; q < quad.size(); ++q) {
          B[i][j] += (fe_face_values.shape_value(l1, q) * fe_face_values.shape_value(l2, q) *
                      fe_face_values.JxW(q));
        }
      }
    }

    Assert(std::abs(B[0][1]) > 0, dealii::ExcMessage("something went wrong"));
    Assert(std::abs(B[1][1]) > 0, dealii::ExcMessage("something went wrong"));
    Assert(std::abs(B[0][0]) > 0, dealii::ExcMessage("something went wrong"));

    const double nx = fe_face_values.normal_vector(0)[0];
    const double ny = fe_face_values.normal_vector(0)[1];
    // winkel zwischen y-achse und n
    const double alpha = numbers::PI / 2 - std::atan2(ny, nx);

    typedef dealii::TrilinosWrappers::types::int_type index_t;
    index_t ig1 =
        x_ghosted_->Map().LID(static_cast<index_t>(indexer_.to_global(local_dof_indices[0], 0)));

    Assert(ig1 != -1, dealii::ExcMessage("global index not found in x_ghosted_!"));

    index_t ig2 =
        x_ghosted_->Map().LID(static_cast<index_t>(indexer_.to_global(local_dof_indices[1], 0)));
    Assert(ig2 != -1, dealii::ExcMessage("global index not found in x_ghosted_!"));

    // *** flux ***
    R_.apply(cP_rot.data(), data_in + ig1, alpha);
    ptr_P2H_->to_hermite(cH, cP_rot);
    ptr_H2N_->to_nodal(L1_, cH);
    fentry->second->apply(L1p_, L1_, cell.face(face_idx)->vertex(0));

    R_.apply(cP_rot.data(), data_in + ig2, alpha);
    ptr_P2H_->to_hermite(cH, cP_rot);
    ptr_H2N_->to_nodal(L2_, cH);
    fentry->second->apply(L2p_, L2_, cell.face(face_idx)->vertex(1));

    // *** Assemble into y_ghosted_ ***
    // lfe_idx1-test function => global index ig1, ig1+N
    L_ = B[0][0] * L1p_ + B[0][1] * L2p_;
    ptr_H2N_->to_hermite(cH.data(), L_);               // to_hermite = to_nodal^T
    ptr_P2H_->to_hermite_T(cP_rot, cH);  // to_hermite_T = to_hermite^T
    // rotate back
    R_.apply(ybuf_.data(), cP_rot.data(), -alpha);
    // write to ghosted vector
    Eigen::Map<Eigen::VectorXd> vout1(data_out + ig1, N);
    vout1 += ybuf_;

    // lfe_idx2-test function => global index ig2, ig2+N
    L_ = B[1][0] * L1p_ + B[1][1] * L2p_;
    ptr_H2N_->to_hermite(cH, L_);               // to_hermite = to_nodal^T
    ptr_P2H_->to_hermite_T(cP_rot, cH);  // to_hermite_T = to_hermite^T
    // rotate back
    R_.apply(ybuf_.data(), cP_rot.data(), -alpha);

    // write to ghosted vector
    Eigen::Map<Eigen::VectorXd> vout2(data_out + ig2, N);
    vout2 += ybuf_;
  }

  // scale y_ghosted_
  y_ghosted_->Scale(dt_);
  // *** export y_ghosted_ into global solution vector ***
  ier = out.Export(*y_ghosted_.get(), *exporter_.get(), Epetra_CombineMode::Epetra_AddLocalAlso);
  Assert(ier == 0, dealii::ExcMessage("Export to global solution vector failed."));
}

}  // end namespace boltzmann

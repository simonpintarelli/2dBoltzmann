#pragma once

#include <type_traits>
#include <unordered_map>
#include <boost/assert.hpp>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <app/app.hpp>
#include <matrix/assembly/velocity_var_form.hpp>
#include <matrix/dofs/dofindex_sets.hpp>
#include <matrix/sparsity_pattern/sparsity_pattern_base.hpp>
#include "aux/tensor_helpers.hpp"
#include "matrix/sparsity_pattern/sparsity_pattern.hpp"
#include "vsparsity.hpp"

namespace boltzmann {

template <typename METHOD, typename APP>
class SystemMatrixHandler
{
 public:
  constexpr const static int dim = 2;

 public:
  typedef dealii::TrilinosWrappers::SparseMatrix matrix_t;
  typedef dealii::TrilinosWrappers::MPI::Vector vector_t;
  typedef dealii::TrilinosWrappers::SparsityPattern sparsity_pattern_t;
  typedef dealii::DoFHandler<dim> dof_handler_t;

 private:
  typedef typename METHOD::spectral_basis_t spectral_basis_t;

 public:
  template <typename INDEXER>
  SystemMatrixHandler(const dof_handler_t& dof_handler,
                      const spectral_basis_t& spectral_basis,
                      const INDEXER& indexer,
                      const DoFIndexSetsBase& dof_map,
                      const double dt);

  const matrix_t& get_lhs() const;
  const matrix_t& get_rhs() const;

 private:
  void finalize();

 protected:
  typedef typename METHOD::template var_form_t<dim, APP> var_form_t;

 protected:
  matrix_t LHS_matrix;
  matrix_t RHS_matrix;
  bool finalized = false;

  VelocityVarForm<2> ventries_;
  //@{
  /// Velocity sparsity pattern for matrices appearing on the left, right
  dealii::SparsityPattern vsparsity_lhs_;
  dealii::SparsityPattern vsparsity_rhs_;
  //@}

  //@{
  // global sparsity patterns
  boltzmann::SparsityPattern sparsity_lhs_;
  boltzmann::SparsityPattern sparsity_rhs_;
  //@}
};

// ----------------------------------------------------------------------
template <typename METHOD, typename APP>
template <typename INDEXER>
SystemMatrixHandler<METHOD, APP>::SystemMatrixHandler(const dof_handler_t& dof_handler,
                                                      const spectral_basis_t& spectral_basis,
                                                      const INDEXER& indexer,
                                                      const DoFIndexSetsBase& dof_map,
                                                      const double dt)
{
  ventries_.init(spectral_basis);
  vsparsity<METHOD::lsq_type>::make(
      vsparsity_lhs_, vsparsity_rhs_, spectral_basis.size(), ventries_);

  sparsity_lhs_.init(spectral_basis, dof_handler, vsparsity_lhs_, indexer, dof_map, false);
  LHS_matrix.reinit(sparsity_lhs_.get_sparsity());

  sparsity_rhs_.init(spectral_basis, dof_handler, vsparsity_rhs_, indexer, dof_map, false);
  RHS_matrix.reinit(sparsity_rhs_.get_sparsity());

  const int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
  var_form_t var_form_id(dof_handler.get_fe());
  var_form_t var_form_transport(dof_handler.get_fe());
  var_form_t var_form_rhs(dof_handler.get_fe());

  const unsigned int mpi_this_process = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const unsigned int N = spectral_basis.n_dofs();

  const double dt2 = dt * dt;

  typedef typename matrix_t::size_type size_type;

  for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell) {
    if (cell->subdomain_id() == mpi_this_process) {
      std::vector<size_type> local_dof_indices(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);
      var_form_id.calc_raw_identity(cell);
      var_form_transport.calc_transport_cell(cell);
      var_form_rhs.calc_identity(cell);

      const auto& IDx_loc = var_form_id.S0();
      const auto& T2x_loc = var_form_transport.T2();
      const auto& T1x_loc = var_form_rhs.S1();

      const auto& VS0 = ventries_.s0();
      const auto& VT1 = ventries_.t1();
      const auto& VT2 = ventries_.t2();

      // iterate over velocity test functions
      for (unsigned int j1 = 0; j1 < N; ++j1) {
        typedef std::unordered_map<unsigned int, unsigned int> map_t;
        // prepare column indices for velocity sparsity pattern
        auto prepare_cidx = [](
            map_t& J2Idx, const dealii::SparsityPattern& vsparsity, unsigned int j1) {
          // non zero velocity entries on row j1
          const unsigned int nentries = vsparsity.end(j1) - vsparsity.begin(j1);
          std::vector<unsigned int> J2(nentries);
          auto it = vsparsity.begin(j1);
          for (unsigned int i = 0; i < nentries; ++i) J2[i] = it++->column();
          // map column indices to array
          unsigned int counter = 0;
          std::for_each(J2.begin(), J2.end(), [&](unsigned int v) { J2Idx[v] = counter++; });
        };

        // std::unordered_map j2 (col index) -> local accumulator
        map_t mJ2_lhs, mJ2_rhs;
        prepare_cidx(mJ2_lhs, vsparsity_lhs_, j1);
        prepare_cidx(mJ2_rhs, vsparsity_rhs_, j1);

        auto get_idx = [](const map_t& m, unsigned int j2) {
          auto it = m.find(j2);
          assert(it != m.end());
          return it->second;
        };

        // global column indices
        std::vector<size_type> gcol_idx_lhs(mJ2_lhs.size());
        std::vector<size_type> gcol_idx_rhs(mJ2_rhs.size());
        // local accumulators
        std::vector<double> entriesLHS(mJ2_lhs.size());
        std::vector<double> entriesRHS(mJ2_rhs.size());

        // TODO: factor this out, it depends on the method!
        for (int i1_loc = 0; i1_loc < dofs_per_cell; ++i1_loc) {
          for (int i2_loc = 0; i2_loc < dofs_per_cell; ++i2_loc) {
            std::fill(entriesLHS.begin(), entriesLHS.end(), 0.0);
            std::fill(entriesRHS.begin(), entriesRHS.end(), 0.0);

            for (auto it_s0 = VS0.row_begin(j1); it_s0 < VS0.row_end(j1); ++it_s0) {
              const double v = dealii::scalar_product(IDx_loc[i1_loc][i2_loc], it_s0->val);
              unsigned int ilhs = get_idx(mJ2_lhs, it_s0->col);
              unsigned int irhs = get_idx(mJ2_rhs, it_s0->col);
              entriesLHS[ilhs] += v;
              entriesRHS[irhs] += v;
            }

            for (auto it_t1 = VT1.row_begin(j1); it_t1 < VT1.row_end(j1); ++it_t1) {
              unsigned int irhs = get_idx(mJ2_rhs, it_t1->col);
              entriesRHS[irhs] += dt * dealii::scalar_product(T1x_loc[i1_loc][i2_loc], it_t1->val);
            }

            for (auto it_t2 = VT2.row_begin(j1); it_t2 < VT2.row_end(j1); ++it_t2) {
              unsigned int ilhs = get_idx(mJ2_lhs, it_t2->col);
              entriesLHS[ilhs] += dt2 * dealii::scalar_product(T2x_loc[i1_loc][i2_loc], it_t2->val);
            }

            // prepare indices: global indices
            unsigned int i1 = local_dof_indices[i1_loc];
            unsigned int i2 = local_dof_indices[i2_loc];

            // this could be further improved...
            {
              unsigned int counter = 0;
              for (auto it = vsparsity_lhs_.begin(j1); it != vsparsity_lhs_.end(j1); ++it)
                gcol_idx_lhs[counter++] = indexer.to_global(i2, it->column());
#ifdef DEBUG
              size_type cmax = *std::max_element(gcol_idx_lhs.begin(), gcol_idx_lhs.end());
              BOOST_ASSERT(cmax < LHS_matrix.m());
#endif
            }
            {
              unsigned int counter = 0;
              for (auto it = vsparsity_rhs_.begin(j1); it != vsparsity_rhs_.end(j1); ++it)
                gcol_idx_rhs[counter++] = indexer.to_global(i2, it->column());
#ifdef DEBUG
              size_type cmax = *std::max_element(gcol_idx_rhs.begin(), gcol_idx_rhs.end());
              BOOST_ASSERT(cmax < RHS_matrix.m());
#endif
            }
            // global row index
            unsigned int gi = indexer.to_global(i1, j1);
            BOOST_ASSERT(gi < LHS_matrix.n());
            BOOST_ASSERT(gi < RHS_matrix.n());
            // TrilinosWrappers::SparseMatrix::add(row,n_cols, *col_indices, *values,
            // elide_zero_values col_indices_are_sorted)
            LHS_matrix.add(gi, mJ2_lhs.size(), gcol_idx_lhs.data(), entriesLHS.data(), true, true);
            RHS_matrix.add(gi, mJ2_rhs.size(), gcol_idx_rhs.data(), entriesRHS.data(), true, true);
          }
        }
      }
    }
  }
  finalize();
}

// ----------------------------------------------------------------------
template <typename METHOD, typename APP>
void
SystemMatrixHandler<METHOD, APP>::finalize()
{
  LHS_matrix.compress(dealii::VectorOperation::add);
  RHS_matrix.compress(dealii::VectorOperation::add);
  finalized = true;
}

// ----------------------------------------------------------------------
template <typename METHOD, typename APP>
const typename SystemMatrixHandler<METHOD, APP>::matrix_t&
SystemMatrixHandler<METHOD, APP>::get_lhs() const
{
  if (!finalized) throw std::runtime_error("not finalized!");
  return LHS_matrix;
}

// ----------------------------------------------------------------------
template <typename METHOD, typename APP>
const typename SystemMatrixHandler<METHOD, APP>::matrix_t&
SystemMatrixHandler<METHOD, APP>::get_rhs() const
{
  if (!finalized) throw std::runtime_error("not finalized!");
  return RHS_matrix;
}

}  // end namespace boltzmann

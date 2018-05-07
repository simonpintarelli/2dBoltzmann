#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <AztecOO_ConditionNumber.h>
#include <Epetra_CrsMatrix.h>


namespace boltzmann {
class AztecOOCondest
{
 public:
  typedef dealii::TrilinosWrappers::SparseMatrix matrix_t;

 public:
  void initialize(const matrix_t& matrix, std::string solver_type);

  void initialize(const ::Epetra_Operator& matrix, std::string solver_type);
  double compute(int maxiters, double tol = 1e-2);

 private:
  ::AztecOOConditionNumber cond_;
};

// ----------------------------------------------------------------------
void
AztecOOCondest::initialize(const matrix_t& matrix, std::string solver_type)
{
  const auto& trilinos_matrix = matrix.trilinos_matrix();

  this->initialize(matrix.trilinos_matrix(), solver_type);
}

// ----------------------------------------------------------------------
void
AztecOOCondest::initialize(const ::Epetra_Operator& matrix, std::string solver_type)
{
  int krylovSubspaceSize = 100;
  bool printSolve = true;

  if (solver_type.compare("GMRES") == 0)
    cond_.initialize(
        matrix, ::AztecOOConditionNumber::SolverType::GMRES_, krylovSubspaceSize, printSolve);
  else if (solver_type.compare("CG") == 0)
    cond_.initialize(
        matrix, ::AztecOOConditionNumber::SolverType::CG_, krylovSubspaceSize, printSolve);
  else {
    throw std::runtime_error("unknown SolverType");
  }
}

// ----------------------------------------------------------------------
double
AztecOOCondest::compute(int maxiters, double tol)
{
  cond_.computeConditionNumber(maxiters, tol);
  return cond_.getConditionNumber();
}

}  // end namespace boltzmann

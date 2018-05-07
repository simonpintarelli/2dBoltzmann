#pragma once

#include <Amesos.h>
#include <AztecOO.h>
#include <Epetra_LinearProblem.h>
#include <Epetra_Operator.h>
#include <string>

#include "base/logger.hpp"


namespace boltzmann {

class GMRES
{
 public:
  GMRES(const Epetra_Operator& A, Epetra_MultiVector& x, const Epetra_MultiVector& b);

  void do_solve(const Epetra_Operator& precond,
                int max_steps,
                double tol,
                int restart = 30,
                bool solver_details = true);

  const AztecOO& get_solver() const { return solver; }

 private:
  std::shared_ptr<Epetra_LinearProblem> linear_problem;
  AztecOO solver;
};

GMRES::GMRES(const Epetra_Operator& A, Epetra_MultiVector& x, const Epetra_MultiVector& b)
{
  linear_problem.reset();
  linear_problem.reset(new Epetra_LinearProblem(
      const_cast<Epetra_Operator*>(&A), &x, const_cast<Epetra_MultiVector*>(&b)));
}

void
GMRES::do_solve(
    const Epetra_Operator& precond, int max_steps, double tol, int restart, bool solver_details)
{
  int ierr;

  auto& logger = Logger::GetInstance();
  logger.push_prefix("transport");
  logger.push_prefix("soler");

  solver.SetProblem(*linear_problem);

  solver.SetAztecOption(AZ_solver, AZ_gmres);
  solver.SetAztecOption(AZ_kspace, restart);

  ierr = solver.SetPrecOperator(const_cast<Epetra_Operator*>(&precond));
  if (ierr != 0)
    throw std::runtime_error("setPrecOperator failed with error code: " + std::to_string(ierr) +
                             "\n");

  solver.SetAztecOption(AZ_output, solver_details ? AZ_all : AZ_none);
  solver.SetAztecOption(AZ_conv, AZ_noscaled);

  // ... and then solve!
  ierr = solver.Iterate(max_steps, tol);
  // if (ierr != 0) throw std::runtime_error("solver failed with error code: " +
  // std::to_string(ierr) + "\n");
  int niter = solver.NumIters();
  double res = solver.TrueResidual();

  logger << "GMRES" << niter << " " << res << "\n";

  logger.pop_prefix();
  logger.pop_prefix();
}

}  // end namespace boltzmann

#pragma once

#include <Epetra_Map.h>
#include <Epetra_MpiComm.h>
#include <Epetra_RowMatrix.h>
#include <Epetra_Vector.h>
#include <Ifpack.h>
#include <Teuchos_ParameterList.hpp>


namespace boltzmann {

class PreconditionILU
{
 public:
  template <typename MATRIX>
  PreconditionILU(const MATRIX& matrix,
                  unsigned int ilu_fill = 0,
                  double ilu_atol = 0,
                  double ilu_rtol = 1,
                  unsigned int overlap = 0);

  const Epetra_Operator& get() const { return *preconditioner; }

 private:
  std::shared_ptr<Epetra_Operator> preconditioner;
};

template <typename MATRIX>
PreconditionILU::PreconditionILU(const MATRIX& matrix,
                                 unsigned int ilu_fill,
                                 double ilu_atol,
                                 double ilu_rtol,
                                 unsigned int overlap)
{
  preconditioner.reset();
  preconditioner.reset(Ifpack().Create("ILU", const_cast<Epetra_CrsMatrix*>(&matrix), overlap));

  Ifpack_Preconditioner* ifpack = static_cast<Ifpack_Preconditioner*>(preconditioner.get());
  if (ifpack == 0) {
    throw std::runtime_error("Trilinos could not create this preconditioner");
  }

  int ierr;

  Teuchos::ParameterList parameter_list;
  parameter_list.set("fact: level-of-fill", static_cast<int>(ilu_fill));
  parameter_list.set("fact: absolute threshold", ilu_atol);
  parameter_list.set("fact: relative threshold", ilu_rtol);
  parameter_list.set("schwarz: combine mode", "Add");

  ierr = ifpack->SetParameters(parameter_list);
  if (ierr != 0) throw std::runtime_error("Trilinos error");

  ierr = ifpack->Initialize();
  if (ierr != 0) throw std::runtime_error("Trilinos error");

  ierr = ifpack->Compute();
  if (ierr != 0) throw std::runtime_error("Trilinos error");
}

}  // end namespace boltzmann

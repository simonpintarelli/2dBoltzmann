#pragma once

#include <Epetra_MultiVector.h>
#include <Epetra_Operator.h>

namespace boltzmann {

/*** Wrapper for use in AztecOOConditionNumber ***/
class SystemMatrix : public ::Epetra_Operator
{
 public:
  SystemMatrix(const ::Epetra_CrsMatrix& A, const ::Epetra_CrsMatrix& B, double gamma = 1.0)
      : A_(A)
      , B_(B)
      , gamma_(gamma)
  { /* empty */
  }

  int Apply(const ::Epetra_MultiVector& X, ::Epetra_MultiVector& Y) const;
  int ApplyInverse(const ::Epetra_MultiVector& X, ::Epetra_MultiVector& Y) const;
  double NormInf() const;
  const char* Label() const;
  int SetUseTranspose(bool f);
  bool UseTranspose() const;
  bool HasNormInf() const;
  const ::Epetra_Comm& Comm() const;
  const ::Epetra_Map& OperatorDomainMap() const;
  const ::Epetra_Map& OperatorRangeMap() const;

  void set_gamma(double gamma) { gamma_ = gamma; }

 private:
  const ::Epetra_CrsMatrix& A_;
  const ::Epetra_CrsMatrix& B_;
  double gamma_;
};

int
SystemMatrix::Apply(const ::Epetra_MultiVector& X, ::Epetra_MultiVector& Y) const
{
  Epetra_MultiVector V_(X);
  int retA = A_.Apply(X, V_);
  int retB = B_.Apply(X, Y);

  int ret = Y.Update(gamma_, V_, 1.0);

  return ret;
}

int
SystemMatrix::ApplyInverse(const ::Epetra_MultiVector& X, ::Epetra_MultiVector& Y) const
{
  throw std::runtime_error("not implemented");
}

double
SystemMatrix::NormInf() const
{
  throw std::runtime_error("not implemented");
}

bool
SystemMatrix::HasNormInf() const
{
  return false;
}

const char*
SystemMatrix::Label() const
{
  return A_.Label();
}

int
SystemMatrix::SetUseTranspose(bool f)
{
  std::runtime_error("not implemented");
}

bool
SystemMatrix::UseTranspose() const
{
  return false;
}

const ::Epetra_Comm&
SystemMatrix::Comm() const
{
  return A_.Comm();
}

const ::Epetra_Map&
SystemMatrix::OperatorDomainMap() const
{
  return A_.OperatorDomainMap();
}

const ::Epetra_Map&
SystemMatrix::OperatorRangeMap() const
{
  return A_.OperatorRangeMap();
}

}  // end namespace boltzmann

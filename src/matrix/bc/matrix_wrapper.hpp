#pragma once

#include <Epetra_CrsMatrix.h>
#include <Epetra_MultiVector.h>
#include <Epetra_Operator.h>
#include <deal.II/base/conditional_ostream.h>
#include "aux/message.hpp"
#include "aux/timer.hpp"
#include "boundary_conditions.hpp"

namespace boltzmann {
namespace otf_bc {

template <typename METHOD, typename APP>
class SystemMatrix : public ::Epetra_Operator
{
 private:
  typedef ::boltzmann::BoundaryConditions<METHOD, APP, impl::BdFacesManager> B_t;

 public:
  SystemMatrix(const ::Epetra_CrsMatrix& A, const B_t& B)
      : A_(A)
      , B_(B)
      , pcout(std::cout, ::dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    /* empty */
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

 private:
  const ::Epetra_CrsMatrix& A_;
  const B_t& B_;

  mutable Timer<> timer;
  mutable ::dealii::ConditionalOStream pcout;
};

template <typename METHOD, typename APP>
int
SystemMatrix<METHOD, APP>::Apply(const ::Epetra_MultiVector& X, ::Epetra_MultiVector& Y) const
{
  //  timer.start();
  int retA = A_.Apply(X, Y);
  //  print_timer(timer.stop(), "A*x", pcout);

  // timer.start();
  B_.apply(Y, X);
  // print_timer(timer.stop(), "B*x", pcout);

  return retA;
}

template <typename METHOD, typename APP>
int
SystemMatrix<METHOD, APP>::ApplyInverse(const ::Epetra_MultiVector& X,
                                        ::Epetra_MultiVector& Y) const
{
  throw std::runtime_error("not implemented");
}

template <typename METHOD, typename APP>
double
SystemMatrix<METHOD, APP>::NormInf() const
{
  throw std::runtime_error("not implemented");
}

template <typename METHOD, typename APP>
bool
SystemMatrix<METHOD, APP>::HasNormInf() const
{
  return false;
}

template <typename METHOD, typename APP>
const char*
SystemMatrix<METHOD, APP>::Label() const
{
  return A_.Label();
}

template <typename METHOD, typename APP>
int
SystemMatrix<METHOD, APP>::SetUseTranspose(bool f)
{
  std::runtime_error("not implemented");
  return -1;
}

template <typename METHOD, typename APP>
bool
SystemMatrix<METHOD, APP>::UseTranspose() const
{
  return false;
}

template <typename METHOD, typename APP>
const ::Epetra_Comm&
SystemMatrix<METHOD, APP>::Comm() const
{
  return A_.Comm();
}

template <typename METHOD, typename APP>
const ::Epetra_Map&
SystemMatrix<METHOD, APP>::OperatorDomainMap() const
{
  return A_.OperatorDomainMap();
}

template <typename METHOD, typename APP>
const ::Epetra_Map&
SystemMatrix<METHOD, APP>::OperatorRangeMap() const
{
  return A_.OperatorRangeMap();
}

}  // end namespace otf_bc
}  // end namespace boltzmann

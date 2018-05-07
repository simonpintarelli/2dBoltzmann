#pragma once

#include <EpetraExt_HDF5.h>
#include <Epetra_Comm.h>

#include <string>


#ifdef HAVE_EPETRAEXT_HDF5
namespace boltzmann {

class ExportEpetraHDF5
{
 public:
  ExportEpetraHDF5(const Epetra_Comm& comm, const std::string& fname = "matrices.h5")
      : exporter(comm)
  {
    exporter.Create(fname);
  }

  template <typename T>
  void write(const std::string& group_name, const T& data)
  {
    exporter.Write(group_name, data);
  }

 private:
  EpetraExt::HDF5 exporter;
};
}  // end namespace boltzmann

#endif

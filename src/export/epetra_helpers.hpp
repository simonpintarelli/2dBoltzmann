#pragma once

#include <hdf5.h>
#include <stdexcept>

#include <Epetra_MultiVector.h>

namespace epetra_helpers {

class HDF5
{
 public:
  HDF5(const std::string& fname);

  void CreateGroup(const std::string& GroupName);

  void Write(const std::string& GroupName, const Epetra_MultiVector& X);

  void Write(const std::string& GroupName, const std::string& DataSetName, int what);
  void Write(const std::string& GroupName, const std::string& DataSetName, double what);
  void Write(const std::string& GroupName, const std::string& DataSetName, const std::string& data);

  bool IsContained(const std::string& Name);

  void Close();

  ~HDF5()
  {
    if (is_open_) Close();
  }

 private:
  hid_t file_id_;
  hid_t plist_id_;
  bool is_open_ = false;
};

}  // epetra_helpers

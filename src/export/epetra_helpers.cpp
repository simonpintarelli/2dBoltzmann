/**
 * Modified Version of EpetraExt_HDF5
 *
 * @param loc_id
 * @param name
 * @param opdata
 *
 * @return
 */

#include "epetra_helpers.hpp"

#include <Epetra_BlockMap.h>
#include <Epetra_MultiVector.h>
#include <Teuchos_RCP.hpp>
#include <boost/assert.hpp>

#define CHECK_HID(hid_t)                                                      \
  {                                                                           \
    if (hid_t < 0) throw(Exception(__FILE__, __LINE__, "hid_t is negative")); \
  }

namespace epetra_helpers {

struct FindDataset_t
{
  std::string name;
  bool found;
};

static herr_t FindDataset(hid_t loc_id, const char* name, void* opdata)
{
  std::string& token = ((FindDataset_t*)opdata)->name;
  if (token == name) ((FindDataset_t*)opdata)->found = true;

  return (0);
}

HDF5::HDF5(const std::string& fname)
{
  plist_id_ = H5Pcreate(H5P_FILE_ACCESS);
  // create the file collectively and release property list identifier.

  BOOST_VERIFY(H5Pset_fapl_mpio(plist_id_, MPI_COMM_WORLD, MPI_INFO_NULL) == 0);

  file_id_ = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id_);

  H5Pclose(plist_id_);
  is_open_ = true;
}

void HDF5::Close()
{
  if (is_open_) {
    H5Fclose(file_id_);
    is_open_ = false;
  }
}

bool HDF5::IsContained(const std::string& Name)
{
  FindDataset_t data;
  data.name = Name;
  data.found = false;

  // int idx_f =
  H5Giterate(file_id_, "/", NULL, FindDataset, (void*)&data);

  return (data.found);
}

void HDF5::Write(const std::string& GroupName, const Epetra_MultiVector& X)
{
  hid_t group_id, dset_id;
  hid_t filespace_id, memspace_id;
  herr_t status;

  // need a linear distribution to use hyperslabs
  Teuchos::RCP<Epetra_MultiVector> LinearX;

  //  if (X.Map().LinearMap())
  LinearX = Teuchos::rcp(const_cast<Epetra_MultiVector*>(&X), false);
  // else
  //   {
  //     throw std::runtime_error("not implemented");
  //     // Epetra_Map LinearMap(X.GlobalLength(), X.Map().IndexBase(), Comm_);
  //     // LinearX = Teuchos::rcp(new Epetra_MultiVector(LinearMap, X.NumVectors()));
  //     // Epetra_Import Importer(LinearMap, X.Map());
  //     // LinearX->Import(X, Importer, Insert);
  //   }

  int NumVectors = X.NumVectors();
  int GlobalLength = X.GlobalLength();

  // Whether or not we do writeTranspose or not is
  // handled by one of the components of q_dimsf, offset and count.
  // They are determined by indexT
  int indexT = 0;
  // if (writeTranspose) indexT = 1;

  hsize_t q_dimsf[] = {static_cast<hsize_t>(GlobalLength), static_cast<hsize_t>(GlobalLength)};
  q_dimsf[indexT] = NumVectors;

  filespace_id = H5Screate_simple(2, q_dimsf, NULL);

  if (!IsContained(GroupName)) CreateGroup(GroupName);

  group_id = H5Gopen(file_id_, GroupName.c_str(), H5P_DEFAULT);

  // Create the dataset with default properties and close filespace_id.
  dset_id = H5Dcreate(
      group_id, "Values", H5T_NATIVE_DOUBLE, filespace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Create property list for collective dataset write.
  plist_id_ = H5Pcreate(H5P_DATASET_XFER);
#ifdef HAVE_MPI
  BOOST_VERIFY(H5Pset_dxpl_mpio(plist_id_, H5FD_MPIO_COLLECTIVE) == 0);
#endif

  BOOST_ASSERT_MSG(LinearX->Map().MinElementSize() == LinearX->Map().MaxElementSize(),
                   "Uniform element size required");
  int element_size = LinearX->Map().MinElementSize();

  // std::cout << "element_size: " << element_size << "\n";
  // std::cout << "LinearX->MyLength(): " << LinearX->MyLength() << "\n";

  // Select hyperslab in the file.
  hsize_t offset[] = {
      element_size * static_cast<hsize_t>(LinearX->Map().GID(0) - X.Map().IndexBase()),
      element_size * static_cast<hsize_t>(LinearX->Map().GID(0) - X.Map().IndexBase())};
  hsize_t stride[] = {1, 1};
  hsize_t count[] = {static_cast<hsize_t>(LinearX->MyLength()),
                     static_cast<hsize_t>(LinearX->MyLength())};
  hsize_t block[] = {1, 1};

  // write vectors one by one
  for (int n = 0; n < NumVectors; ++n) {
    // Select hyperslab in the file.
    offset[indexT] = n;
    count[indexT] = 1;

    filespace_id = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, offset, stride, count, block);

    // Each process defines dataset in memory and writes it to the hyperslab in the file.
    hsize_t dimsm[] = {static_cast<hsize_t>(LinearX->MyLength())};
    memspace_id = H5Screate_simple(1, dimsm, NULL);

    // Write hyperslab
    BOOST_VERIFY(H5Dwrite(dset_id,
                          H5T_NATIVE_DOUBLE,
                          memspace_id,
                          filespace_id,
                          plist_id_,
                          LinearX->operator[](n)) == 0);
  }
  H5Gclose(group_id);
  H5Sclose(memspace_id);
  H5Sclose(filespace_id);
  H5Dclose(dset_id);
  H5Pclose(plist_id_);

  Write(GroupName, "GlobalLength", GlobalLength);
  Write(GroupName, "NumVectors", NumVectors);
  Write(GroupName, "__type__", "Epetra_MultiVector");
}

// ==========================================================================
void HDF5::Write(const std::string& GroupName, const std::string& DataSetName, int what)
{
  if (!IsContained(GroupName)) CreateGroup(GroupName);

  hid_t filespace_id = H5Screate(H5S_SCALAR);
  hid_t group_id = H5Gopen(file_id_, GroupName.c_str(), H5P_DEFAULT);
  hid_t dset_id = H5Dcreate(group_id,
                            DataSetName.c_str(),
                            H5T_NATIVE_INT,
                            filespace_id,
                            H5P_DEFAULT,
                            H5P_DEFAULT,
                            H5P_DEFAULT);

  herr_t status = H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, filespace_id, H5P_DEFAULT, &what);
  BOOST_ASSERT(status == 0);

  // Close/release resources.
  H5Dclose(dset_id);
  H5Gclose(group_id);
  H5Sclose(filespace_id);
}

// ==========================================================================
void HDF5::Write(const std::string& GroupName, const std::string& DataSetName, double what)
{
  if (!IsContained(GroupName)) CreateGroup(GroupName);

  hid_t filespace_id = H5Screate(H5S_SCALAR);
  hid_t group_id = H5Gopen(file_id_, GroupName.c_str(), H5P_DEFAULT);
  hid_t dset_id = H5Dcreate(group_id,
                            DataSetName.c_str(),
                            H5T_NATIVE_DOUBLE,
                            filespace_id,
                            H5P_DEFAULT,
                            H5P_DEFAULT,
                            H5P_DEFAULT);

  herr_t status = H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, filespace_id, H5P_DEFAULT, &what);
  BOOST_VERIFY(status == 0);

  // Close/release resources.
  H5Dclose(dset_id);
  H5Gclose(group_id);
  H5Sclose(filespace_id);
}

void HDF5::CreateGroup(const std::string& GroupName)
{
  hid_t group_id = H5Gcreate(file_id_, GroupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Gclose(group_id);
}

// ==========================================================================
void HDF5::Write(const std::string& GroupName,
                 const std::string& DataSetName,
                 const std::string& data)
{
  if (!IsContained(GroupName)) CreateGroup(GroupName);

  hsize_t len = 1;

  hid_t group_id = H5Gopen(file_id_, GroupName.c_str(), H5P_DEFAULT);

  hid_t dataspace_id = H5Screate_simple(1, &len, NULL);

  hid_t atype = H5Tcopy(H5T_C_S1);
  H5Tset_size(atype, data.size() + 1);

  hid_t dataset_id = H5Dcreate(
      group_id, DataSetName.c_str(), atype, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  BOOST_VERIFY(H5Dwrite(dataset_id, atype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.c_str()) == 0);
  BOOST_VERIFY(H5Tclose(atype) == 0);
  BOOST_VERIFY(H5Dclose(dataset_id) == 0);
  BOOST_VERIFY(H5Sclose(dataspace_id) == 0);
  BOOST_VERIFY(H5Gclose(group_id) == 0);
}

}  // EpetraHelpers

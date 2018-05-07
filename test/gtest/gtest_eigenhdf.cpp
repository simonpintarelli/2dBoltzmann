/**
 *   @file gtest_eigenhdf.cpp
 *   @brief write to and read from hdf5 files via eigen2hdf wrapper
 *
 *  Detailed description
 *
 *  Note: eigen2hdf always uses 2 dimensional arrays, also for
 *        vectors, this is not the case for h5py.
 */

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "aux/eigen2hdf.hpp"

// write and read from hdf5 files using wrapper functions
// testing for type double (std::complex<double> is missing)

TEST(eigenhdf, row_col_vector)
{
  typedef Eigen::Array<double, 1, Eigen::Dynamic> col_array_t;
  typedef Eigen::Array<double, Eigen::Dynamic, 1> row_array_t;
  const char* fname = "gtest_eigenhdf_row_vector.h5";

  hid_t h5_file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  ASSERT_TRUE(h5_file > 0);

  col_array_t X(100);
  X.setRandom();
  eigen2hdf::save(h5_file, "X", X);

  row_array_t Y(100);
  Y.setRandom();
  eigen2hdf::save(h5_file, "Y", Y);

  herr_t info = H5Fclose(h5_file);
  ASSERT_TRUE(info == 0);

  // -- re-open -- //
  h5_file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
  ASSERT_TRUE(h5_file > 0);

  col_array_t Xn;
  eigen2hdf::load(h5_file, "X", Xn);
  EXPECT_EQ(Xn.rows(), X.rows());
  EXPECT_EQ(Xn.cols(), X.cols());

  row_array_t Yn;
  eigen2hdf::load(h5_file, "Y", Yn);
  EXPECT_EQ(Yn.rows(), Y.rows());
  EXPECT_EQ(Yn.cols(), Y.cols());

  EXPECT_TRUE(((Yn - Y).abs() == 0).any());
  EXPECT_TRUE(((Xn - X).abs() == 0).any());

  info = H5Fclose(h5_file);
  ASSERT_TRUE(info == 0);
}

TEST(eigenhdf, c_array)
{
  typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> c_array_t;
  const char* fname = "gtest_eigenhdf_col_vector.h5";

  hid_t h5_file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  ASSERT_TRUE(h5_file > 0);

  c_array_t X(100, 10);
  X.setRandom();
  eigen2hdf::save(h5_file, "X", X);

  c_array_t Y(20, 100);
  Y.setRandom();
  eigen2hdf::save(h5_file, "Y", Y);

  herr_t info = H5Fclose(h5_file);
  ASSERT_TRUE(info == 0);

  // -- re-open -- //
  h5_file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
  ASSERT_TRUE(h5_file > 0);

  c_array_t Xn;
  eigen2hdf::load(h5_file, "X", Xn);
  EXPECT_EQ(Xn.rows(), X.rows());
  EXPECT_EQ(Xn.cols(), X.cols());

  c_array_t Yn;
  eigen2hdf::load(h5_file, "Y", Yn);
  EXPECT_EQ(Yn.rows(), Y.rows());
  EXPECT_EQ(Yn.cols(), Y.cols());

  EXPECT_TRUE(((Yn - Y).abs() == 0).any());
  EXPECT_TRUE(((Xn - X).abs() == 0).any());

  info = H5Fclose(h5_file);
  ASSERT_TRUE(info == 0);
}

TEST(eigenhdf, sparse_matrix)
{
  typedef Eigen::SparseMatrix<double, Eigen::RowMajor> sp_cmat_t;

  int n = 10;
  int m = 144;

  sp_cmat_t K(n, m);

  for (int i = 0; i < n; ++i) {
    int j = i + 1;
    if (j < m) K.insert(i, j) = i + j;
    j = i - 1;
    if (j >= 0 && j < m) K.insert(i, j) = i - j;
  }
  K.makeCompressed();

  const char* fname = "gtest_eigenhdf_sparse.h5";
  hid_t h5_file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  eigen2hdf::save_sparse(h5_file, "S", K);
  hid_t info = H5Fclose(h5_file);
  ASSERT_TRUE(info == 0);

  // -- open --
  h5_file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
  sp_cmat_t Kl;

  eigen2hdf::load_sparse(h5_file, "S", Kl);

  ASSERT_TRUE(Kl.isCompressed());

  ASSERT_EQ(Kl.rows(), K.rows());
  ASSERT_EQ(Kl.cols(), K.cols());

  EXPECT_TRUE((K - Kl).sum() == 0);

  info = H5Fclose(h5_file);
  ASSERT_TRUE(info == 0);

  // -- open as col major matrix
  typedef Eigen::SparseMatrix<double, Eigen::ColMajor> sp_fmat_t;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> array_t;

  h5_file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
  sp_fmat_t Kf;

  eigen2hdf::load_sparse(h5_file, "S", Kf);

  ASSERT_TRUE(Kf.isCompressed());

  ASSERT_EQ(Kf.rows(), K.rows());
  ASSERT_EQ(Kf.cols(), K.cols());

  array_t K2 = K;
  array_t Kf2 = Kf;

  EXPECT_TRUE((K2 - Kf2).cwiseAbs().sum() == 0);

  info = H5Fclose(h5_file);
  ASSERT_TRUE(info == 0);
}

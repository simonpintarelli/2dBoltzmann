#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/tuple.hpp>
#include <memory>
#include <stdexcept>
#include <string>
#include <iostream>

#include "aux/exceptions.h"
#include "aux/timer.hpp"
#include "aux/hexify.hpp"
#include "collision_tensor/collision_tensor_galerkin.hpp"
#include "collision_tensor/dense/collision_tensor_zlastAM_eigen.hpp"
#include "collision_tensor/dense/collision_tensor_zlastAM.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"

namespace bp = boost::python;
namespace np = boost::python::numpy;
namespace bf = boost::filesystem;

bool is_c_order(np::ndarray::bitflag flag) {
  return ( (flag & np::ndarray::bitflag::C_CONTIGUOUS) != 0x0 );

};

bool is_f_order(np::ndarray::bitflag flag) {
  return ( (flag & np::ndarray::bitflag::F_CONTIGUOUS) != 0x0 );
};

template<typename TENSOR_PTR>
np::ndarray project_coeffs(TENSOR_PTR tensor_, const np::ndarray& Cn, const np::ndarray& Cp)
{
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> row_matrix_t;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> col_matrix_t;

  // check dimensions
  int rows = Cp.shape(0);
  int cols = 1;
  if (Cp.get_nd() == 2) {
    cols = Cp.shape(1);
  }
  if (!(rows == tensor_->get_basis().size())) {
    int N = tensor_->get_basis().size();
    throw std::runtime_error("input array has wrong number of columns, expected" +
                             std::to_string(N) + " got " + std::to_string(rows));
  }
  int cn_rows = Cn.shape(0);
  int cn_cols = 1;
  if (Cn.get_nd() == 2) {
    cn_cols = Cn.shape(1);
  }
  if( ! ((cn_rows == rows) && (cn_cols == cols)) ) {
    throw std::runtime_error("input arrays must have identical shapes!");
  }
  BAssertThrow(Cp.get_dtype().get_itemsize() == sizeof(double), "Type length mismatch");

  col_matrix_t out(rows, cols);
  // copy Cn to out
  np::ndarray::bitflag cn_flags = Cn.get_flags();
  const double* cn_ptr = reinterpret_cast<const double*>(Cn.get_data());
  if(is_c_order(cn_flags)) {
    Eigen::Map<const row_matrix_t> mCn_row(cn_ptr, rows, cols);
    out = mCn_row;
  } else if (is_f_order(cn_flags)) {
    Eigen::Map<const col_matrix_t> mCn_col(cn_ptr, rows, cols);
    out = mCn_col;
  } else {
    throw std::runtime_error(std::string("oops, got bad np.flag: ") + std::to_string(cn_flags));
  }

  // see https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flags.html
  // and http://www.boost.org/doc/libs/1_66_0/libs/python/doc/html/numpy/reference/ndarray.html
  // for documentation about numpy array flags
  np::ndarray::bitflag cp_flags = Cp.get_flags();
  if (is_c_order(cp_flags)) {
    const double* in_ptr = reinterpret_cast<const double*>(Cp.get_data());
    Eigen::Map<const row_matrix_t> mCp_row(in_ptr, rows, cols);
    tensor_->project(out, mCp_row);
  } else if (is_f_order(cp_flags)) {
    const double* in_ptr = reinterpret_cast<const double*>(Cp.get_data());
    Eigen::Map<const col_matrix_t> mCp(in_ptr, rows, cols);
    tensor_->project(out, mCp);
  } else {
    throw std::runtime_error(std::string("oops, got bad np.flag: ") + std::to_string(cp_flags) + "\n" +
                             "see "
                             "http://www.boost.org/doc/libs/1_66_0/libs/python/doc/html/numpy/"
                             "reference/ndarray.htm;");
  }

  // see below how to make a column major numpy array from c++ :)
  np::ndarray arrOut =
      np::from_data(out.data(),
                    np::dtype::get_builtin<double>(),
                    bp::make_tuple(rows, cols),
                    bp::make_tuple(sizeof(double), /* increment to go to the next row */
                                   sizeof(double) * rows /* increment to go to the next column */),
                    bp::object());
  return arrOut.copy();
}

class CollisionTensorSparseWrapper
{
 public:
  /**
   * @param pathname path to collision tensor HDF5 file
   *
   */
  CollisionTensorSparseWrapper(std::string pathname = "collision_tensor.h5");
  np::ndarray apply(const np::ndarray& C) const;

  /**
   *  @brief project to same mass, momentum and energy
   *
   *  @param Cn (unprojected) Polar-Laguerre coefficients for next step
   *  @param Cp Polar-Laguerre coefficients from previous step (used to extract m, u, e)
   */
  np::ndarray project(const np::ndarray& Cn, const np::ndarray& Cp) const;

  void use_timer(bool flag);

 private:
  typedef boltzmann::CollisionTensorGalerkin tensor_t;
  std::shared_ptr<tensor_t> tensor_;
  bool print_timer_ = false;

};


CollisionTensorSparseWrapper::CollisionTensorSparseWrapper(std::string pathname)
{
  if (!bf::exists(pathname)) {
    throw std::runtime_error("CollisionTensor HDF file not found.");
  }

  auto path = bf::path(pathname);
  auto src_path = path.parent_path();
  typedef boltzmann::SpectralBasisFactoryKS::basis_type spectral_basis_t;

  auto basis_desc_path = src_path / bf::path("spectral_basis.desc");
  spectral_basis_t basis;
  boltzmann::SpectralBasisFactoryKS::create(basis, bf::canonical(basis_desc_path).string());
  tensor_ = std::allocate_shared<tensor_t>(Eigen::aligned_allocator<tensor_t>(), basis);

  tensor_->read_hdf5(pathname.c_str());
}


np::ndarray
CollisionTensorSparseWrapper::apply(const np::ndarray& C) const
{
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> row_matrix_t;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> col_matrix_t;
  boltzmann::Timer<std::chrono::microseconds> timer;

  // check dimensions
  int rows = C.shape(0);
  int cols = 1;
  if (C.get_nd() == 2) {
    cols = C.shape(1);
  }
  if (!(rows == tensor_->get_basis().size())) {
    int N = tensor_->get_basis().size();
    throw std::runtime_error("input array has wrong number of columns, expected" +
                             std::to_string(N) + " got " + std::to_string(rows));
  }

  col_matrix_t out(rows, cols);

  BAssertThrow(C.get_dtype().get_itemsize() == sizeof(double), "Type length mismatch");

  // see https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flags.html
  // and http://www.boost.org/doc/libs/1_66_0/libs/python/doc/html/numpy/reference/ndarray.html
  // for documentation about numpy array flags
  np::ndarray::bitflag flags = C.get_flags();
  if (is_c_order(flags)) {
    const double* in_ptr = reinterpret_cast<const double*>(C.get_data());
    Eigen::Map<const row_matrix_t> mC_row(in_ptr, rows, cols);
    col_matrix_t mC = mC_row;
    timer.start();
    for (int i = 0; i < cols; ++i) {
      tensor_->apply(out.col(i).data(), mC.col(i).data());
    }
    auto tlap = timer.stop();
    if(print_timer_) {
      timer.print(std::cout, tlap, "ctsparse");
      std::cout << std::flush;
    }
  } else if (is_f_order(flags)) {
    const double* in_ptr = reinterpret_cast<const double*>(C.get_data());
    Eigen::Map<const col_matrix_t> mC(in_ptr, rows, cols);
    timer.start();
    for (int i = 0; i < cols; ++i) {
      tensor_->apply(out.col(i).data(), mC.col(i).data());
    }
    auto tlap = timer.stop();
    if(print_timer_) {
      timer.print(std::cout, tlap, "ctsparse");
      std::cout << std::flush;
    }
  } else {
    throw std::runtime_error(std::string("oops, got bad np.flag: ") + hexify(flags) + "\n" +
                             "see "
                             "http://www.boost.org/doc/libs/1_66_0/libs/python/doc/html/numpy/"
                             "reference/ndarray.htm;");
  }

  // create np::ndarray from out and return
  np::ndarray np_arr = np::from_data(reinterpret_cast<void*>(out.data()),
                                     np::dtype::get_builtin<double>(),
                                     bp::make_tuple(rows, cols),
                                     bp::make_tuple(sizeof(double), /* increment to go the next row */
                                                    sizeof(double)*rows /* increment to go to the next column */),
                                     bp::object());
  return np_arr.copy();
}


np::ndarray
CollisionTensorSparseWrapper::project(const np::ndarray& Cn, const np::ndarray& Cp) const
{
  return project_coeffs(tensor_, Cn, Cp);
}

void CollisionTensorSparseWrapper::use_timer(bool flag)
{
  print_timer_ = flag;
}


/**
 *  @tparam TENSOR should be CollisionTensorZLastAM or CollisionTensorZLastAMEigen
 *
 */
template<typename TENSOR>
class CollisionTensorDenseWrapper
{
 public:
  CollisionTensorDenseWrapper(std::string pathname = "collision_tensor.h5", int bufsize=1);

  np::ndarray apply(const np::ndarray& C, int imax=0);
  np::ndarray project(const np::ndarray& Cn, const np::ndarray& Cp);

  int getBufsize() const;
  void setBufsize(int bufsize);
  void use_timer(bool flag);

 private:
  typedef TENSOR tensor_t;
  typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> in_array_t;

 private:
  std::shared_ptr<tensor_t> tensor_;
  in_array_t padded_in;
  bool print_timer_ = false;

};


template<typename TENSOR>
CollisionTensorDenseWrapper<TENSOR>::CollisionTensorDenseWrapper(std::string pathname, int bufsize)
{
  if (!bf::exists(pathname)) {
    throw std::runtime_error("CollisionTensor HDF file not found.");
  }
  auto path = bf::path(pathname);
  auto src_path = path.parent_path();
  typedef boltzmann::SpectralBasisFactoryKS::basis_type spectral_basis_t;

  boltzmann::Timer<> timer;
  auto basis_desc_path = src_path / bf::path("spectral_basis.desc");
  spectral_basis_t basis;
  boltzmann::SpectralBasisFactoryKS::create(basis, bf::canonical(basis_desc_path).string());
  timer.start();
  tensor_ = std::allocate_shared<tensor_t>(Eigen::aligned_allocator<tensor_t>(), basis, bufsize);
  timer.print(std::cout, timer.stop(), "tensor constructor");
  std::cout << std::flush;
  int vblksize = 1;
  timer.start();
  tensor_->import_entries_mpishmem(pathname, vblksize);
  timer.print(std::cout, timer.stop(), "loading tensor HDF");
  std::cout << std::flush;
}


template<typename TENSOR>
int CollisionTensorDenseWrapper<TENSOR>::getBufsize() const
{
  if(tensor_) {
    return tensor_->get_buffer_size();
  } else {
    return 0;
  }
}


template<typename TENSOR>
void CollisionTensorDenseWrapper<TENSOR>::setBufsize(int bufsize)
{
  if(tensor_) {
    tensor_->resize_buffer(bufsize);
  } else {
    throw std::runtime_error("invalid pointer");
  }
}


template<typename TENSOR>
np::ndarray CollisionTensorDenseWrapper<TENSOR>::apply(const np::ndarray& C, int imax)
{
#ifdef DEBUG
  if(is_c_order(C.get_flags())) {
    std::cout << "CTensorD found c ordered input" << "\n";
  }
  if(is_f_order(C.get_flags())) {
    std::cout << "CTensorD found fortran ordered input" << "\n";
  }
  if(!is_f_order(C.get_flags())) {
    throw std::runtime_error("must use column-major array");
  }
#endif

  boltzmann::Timer<std::chrono::microseconds> timer;

  int rows = C.shape(0);
  int cols = 1;
  if (C.get_nd() == 2) {
    cols = C.shape(1);
  }
  this->setBufsize(cols);

  BAssertThrow(rows == tensor_->get_basis().size(), "size mismatch");

  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> col_matrix_t;
  const double* in_data = reinterpret_cast<const double*>(C.get_data());
  Eigen::Map<const col_matrix_t> C_eigen(in_data, rows, cols);
  tensor_->pad(padded_in, C_eigen);

  col_matrix_t out(rows, cols);
  timer.start();
  tensor_->apply(out, padded_in, imax);
  auto tlap = timer.stop();
  if (print_timer_) {
    timer.print(std::cout, tlap, "ctdense");
    std::cout << std::flush;
  }

  np::ndarray out_arr = np::from_data(reinterpret_cast<void*>(out.data()),
                                      np::dtype::get_builtin<double>(),
                                      bp::make_tuple(rows, cols),
                                      bp::make_tuple(
                                          /* increment to go the next row */
                                          sizeof(double),
                                          /* increment to go to the next column */
                                          sizeof(double)*rows),
                                      bp::object());
  return out_arr.copy();
}


template<typename TENSOR>
void CollisionTensorDenseWrapper<TENSOR>::use_timer(bool flag)
{
  print_timer_ = flag;
}


template<typename TENSOR>
np::ndarray CollisionTensorDenseWrapper<TENSOR>::project(const np::ndarray& Cn, const np::ndarray& Cp)
{
  return project_coeffs(tensor_, Cn, Cp);
}

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(ctdense_apply_overloads, apply, 1, 2)

#ifdef PYTHON
BOOST_PYTHON_MODULE(libCTensor)
{
  using namespace boost::python;
  typedef CollisionTensorDenseWrapper<boltzmann::ct_dense::CollisionTensorZLastAM> ctblas;
  typedef CollisionTensorDenseWrapper<boltzmann::ct_dense::CollisionTensorZLastAMEigen> cteigen;

  // import for handling of numpy arrays
  np::initialize();
  //  docstring_options local_docstring_options(true, true, false);
  class_<CollisionTensorSparseWrapper, boost::noncopyable>("CTensor", init<std::string>())
      .def("apply", &CollisionTensorSparseWrapper::apply, "apply collision operator")
      .def("project", &CollisionTensorSparseWrapper::project, "m,u,e conservation")
      .def("use_timer", &CollisionTensorSparseWrapper::use_timer, "time apply call");
  class_<ctblas, boost::noncopyable>("CTensorDBLAS", init<std::string, int>())
      .def("apply", &ctblas::apply, ctdense_apply_overloads(args("C", "imax"), "apply collision tensor"))
      .def("project", &ctblas::project, "m,u,e conservation")
      .def("getBufsize", &ctblas::getBufsize, "...")
      .def("setBufsize", &ctblas::setBufsize, "...")
      .def("use_timer", &ctblas::use_timer, "time apply call");
  class_<cteigen, boost::noncopyable>("CTensorDEigen", init<std::string, int>())
      .def("apply", &cteigen::apply, ctdense_apply_overloads(args("C", "imax"), "apply collision tensor"))
      .def("project", &cteigen::project, "m,u,e conservation")
      .def("getBufsize", &cteigen::getBufsize, "...")
      .def("setBufsize", &cteigen::setBufsize, "...")
      .def("use_timer", &cteigen::use_timer, "time apply call");
}
#endif

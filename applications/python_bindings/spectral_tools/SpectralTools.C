// system includes ========================================
#include <Eigen/Sparse>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/python.hpp>
#include <boost/python/tuple.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
// own includes ===========================================
#include "aux/boostnpy.hpp"
#include "aux/exceptions.h"
#include "aux/message.hpp"
#include "aux/timer.hpp"
#include "quadrature/qhermite.hpp"
#include "spectral/basis/spectral_basis.hpp"
#include "spectral/basis/spectral_basis_factory_hermite.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"
#include "spectral/basis/spectral_elem.hpp"
#include "spectral/basis/spectral_elem_accessor.hpp"
#include "spectral/basis/spectral_function/hermite_polynomial.hpp"
#include "spectral/basis/toolbox/spectral_basis.hpp"
#include "spectral/polar_to_hermite.hpp"
#include "spectral/polar_to_nodal.hpp"
#include "spectral/shift_hermite_2d.hpp"

namespace bp = boost::python;

using namespace std;
using namespace boltzmann;

class P2H_wrapper
{
 public:
  P2H_wrapper() {}
  P2H_wrapper(int K) { this->init(K); }
#ifdef PYTHON
  void to_polar(np::ndarray& O, const np::ndarray& C);
  void to_hermite(np::ndarray& O, const np::ndarray& C);

  np::ndarray to_polar_ret(const np::ndarray& C);
  np::ndarray to_hermite_ret(const np::ndarray& C);

#endif

 private:
  void init_from_file(const std::string& fname);
  void init(int K);

 private:
  typedef typename SpectralBasisFactoryKS::basis_type polar_basis_t;
  typedef typename SpectralBasisFactoryHN::basis_type hermite_basis_t;
  typedef Polar2Hermite<polar_basis_t, hermite_basis_t> P2H_t;

 private:
  // read polar basis from file
  polar_basis_t polar_basis;
  hermite_basis_t hermite_basis;

  std::shared_ptr<P2H_t> ptr_p2h;
};


// ------------------------------------------------------------------------------------------
void P2H_wrapper::init_from_file(const std::string& fname)
{
  SpectralBasisFactoryKS::create(polar_basis, fname);
  int max_deg = spectral::get_max_k(polar_basis);
  const unsigned int K = max_deg + 1;
  // create corresponding Hermite basis
  typedef typename SpectralBasisFactoryHN::basis_type hermite_basis_t;
  // std::cout  << "K = " << K << std::endl;
  SpectralBasisFactoryHN::create(hermite_basis, K);
  // SpectralBasisFactoryHN::write_basis_descriptor(hermite_basis, "hermite_basis.desc");

  if (hermite_basis.n_dofs() != polar_basis.n_dofs()) {
    throw runtime_error(
        "Hermite basis does not match!, K=" + boost::lexical_cast<std::string>(K) +
        ", size(polar_basis) =" + boost::lexical_cast<std::string>(polar_basis.n_dofs()));
    exit(1);
  }

  ptr_p2h = std::shared_ptr<P2H_t>(new P2H_t(polar_basis, hermite_basis));
}

// ------------------------------------------------------------------------------------------
void P2H_wrapper::init(int K_)
{
  SpectralBasisFactoryKS::create(polar_basis, K_);
  int max_deg = spectral::get_max_k(polar_basis);
  const unsigned int K = max_deg + 1;
  BAssertThrow(K == K_, "oops");
  // create corresponding Hermite basis
  typedef typename SpectralBasisFactoryHN::basis_type hermite_basis_t;
  // std::cout  << "K = " << K << std::endl;
  SpectralBasisFactoryHN::create(hermite_basis, max_deg + 1, 2);
  //SpectralBasisFactoryHN::write_basis_descriptor(hermite_basis, "hermite_basis.desc");

  if (hermite_basis.n_dofs() != polar_basis.n_dofs()) {
    throw runtime_error(
        "Hermite basis does not match!, K=" + boost::lexical_cast<std::string>(K) +
        ", size(polar_basis) =" + boost::lexical_cast<std::string>(polar_basis.n_dofs()));
    exit(1);
  }

  ptr_p2h = std::shared_ptr<P2H_t>(new P2H_t(polar_basis, hermite_basis));
}

// ------------------------------------------------------------
void P2H_wrapper::to_polar(np::ndarray& O, const np::ndarray& C)
{
  const double* src = reinterpret_cast<double*>(C.get_data());
  double* dst = reinterpret_cast<double*>(O.get_data());

  Eigen::Map<Eigen::VectorXd> vdst(dst, polar_basis.size());
  Eigen::Map<const Eigen::VectorXd> vsrc(src, polar_basis.size());

  ptr_p2h->to_polar(vdst, vsrc);
}


np::ndarray P2H_wrapper::to_polar_ret(const np::ndarray& H)
{
  np::ndarray out = np::empty(bp::make_tuple(polar_basis.size()),
                              np::dtype::get_builtin<double>());
  double* dst = reinterpret_cast<double*>(out.get_data());
  int max_deg = spectral::get_max_k(polar_basis);
  const unsigned int K = max_deg + 1;

  BAssertThrow(H.get_shape()[0] == polar_basis.size(), "size mismatch");

  if((np::ndarray::C_CONTIGUOUS & H.get_flags()) != 0x0) {
    const double* src = reinterpret_cast<const double*>(H.get_data());
    Eigen::Map<Eigen::VectorXd> vdst(dst, polar_basis.size());
    Eigen::Map<const Eigen::VectorXd> vsrc(src, polar_basis.size());

    ptr_p2h->to_polar(vdst, vsrc);
  } else {
    throw std::runtime_error("expect C-major ordering");
  }

  return out;
}

// ------------------------------------------------------------
void P2H_wrapper::to_hermite(np::ndarray& O, const np::ndarray& C)
{
  const double* src = reinterpret_cast<double*>(C.get_data());
  double* dst = reinterpret_cast<double*>(O.get_data());
  Eigen::Map<Eigen::VectorXd> vdst(dst, polar_basis.size());
  Eigen::Map<const Eigen::VectorXd> vsrc(src, polar_basis.size());

  ptr_p2h->to_hermite(vdst, vsrc);
}


np::ndarray P2H_wrapper::to_hermite_ret(const np::ndarray& C)
{
  BAssertThrow(C.get_shape()[0] == polar_basis.size(), "size mismatch");
  int max_deg = spectral::get_max_k(polar_basis);
  const unsigned int K = max_deg + 1;

  np::ndarray out = np::zeros(bp::make_tuple(polar_basis.size()),
                              np::dtype::get_builtin<double>());
  double* dst = reinterpret_cast<double*>(out.get_data());

  const double* src = reinterpret_cast<const double*>(C.get_data());

  Eigen::Map<Eigen::VectorXd> vdst(dst, polar_basis.size());
  Eigen::Map<const Eigen::VectorXd> vsrc(src, polar_basis.size());

  ptr_p2h->to_hermite(vdst, vsrc);

  return out;
}

// ------------------------------------------------------------------------------------------

class P2N_wrapper
{
 public:
  P2N_wrapper() {}
  P2N_wrapper(int K) { this->init(K); }
#ifdef PYTHON
  void to_nodal(np::ndarray& O, const np::ndarray& C);
  void to_polar(np::ndarray& O, const np::ndarray& C);

  np::ndarray to_polar_ret(const np::ndarray& C);
  np::ndarray to_nodal_ret(const np::ndarray& C);
#endif

 private:
  void init(int K);

 private:
  typedef typename SpectralBasisFactoryKS::basis_type polar_basis_t;
  typedef Polar2Nodal<polar_basis_t> P2N_t;

 private:
  polar_basis_t polar_basis;
  int K_;
  double w_;
  P2N_t p2n;
};

void P2N_wrapper::init(int K)
{
  SpectralBasisFactoryKS::create(polar_basis, K);
  int max_deg = spectral::get_max_k(polar_basis);
  K_ = max_deg + 1;
  w_ = 0.5;
  p2n.init(polar_basis, 1.0);
}

#ifdef PYTHON
void P2N_wrapper::to_nodal(np::ndarray& O, const np::ndarray& C)
{
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> row_matrix_t;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> col_matrix_t;

  int ndim = O.get_nd();
  int ndim_in = C.get_nd();
  const Py_intptr_t* shapes = O.get_shape();
  const Py_intptr_t* shapes_in = C.get_shape();
  BAssertThrow(ndim == 2, "dimension mismatch");
  BAssertThrow(shapes[0] == K_ && shapes[1] == K_, "shape mismatach");
  if (ndim_in == 2) BAssertThrow(shapes_in[0] == 1 || shapes_in[1] == 1, "shape mismatch");

  // Polar2Nodal assumes ColMajor storage for the src array
  // np::ndarray::bitflag src_flags = C.get_flags();
  const double* src = reinterpret_cast<const double*>(C.get_data());
  Eigen::Map<const Eigen::VectorXd> vsrc(src, polar_basis.size());

  double* dst = reinterpret_cast<double*>(O.get_data());
  np::ndarray::bitflag flags = O.get_flags();

  if ((flags & np::ndarray::F_CONTIGUOUS) != 0x0) {
    // row-major matrix
    Eigen::Map<col_matrix_t> mO(dst, K_, K_);
    p2n.to_nodal(mO, vsrc);
  } else if ((flags & np::ndarray::C_CONTIGUOUS) != 0x0) {
    Eigen::Map<row_matrix_t> mO(dst, K_, K_);
    p2n.to_nodal(mO, vsrc);
  } else {
    BOOST_VERIFY(false);
  }
}


np::ndarray P2N_wrapper::to_nodal_ret(const np::ndarray& C)
{
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> row_matrix_t;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> col_matrix_t;


  // Polar2Nodal assumes ColMajor storage for the src array
  // np::ndarray::bitflag src_flags = C.get_flags();
  BAssertThrow(C.get_nd() == 1, "input must be a 1D-array");
  BAssertThrow(C.shape(0) == polar_basis.size(), "size mismatch");
  const double* src = reinterpret_cast<const double*>(C.get_data());
  Eigen::Map<const Eigen::VectorXd> vsrc(src, polar_basis.size());

  np::ndarray Out = np::empty(bp::make_tuple(K_, K_),
                              np::dtype::get_builtin<double>());
  double* dst = reinterpret_cast<double*>(Out.get_data());

  np::ndarray::bitflag flags = Out.get_flags();

  if ((flags & np::ndarray::F_CONTIGUOUS) != 0x0) {
    // col-major matrix
    Eigen::Map<col_matrix_t> mO(dst, K_, K_);
    p2n.to_nodal(mO, vsrc);
  } else if ((flags & np::ndarray::C_CONTIGUOUS) != 0x0) {
    Eigen::Map<row_matrix_t> mO(dst, K_, K_);
    p2n.to_nodal(mO, vsrc);
  } else {
    BOOST_VERIFY(false);
  }

  return Out;
}


// ------------------------------------------------------------
void P2N_wrapper::to_polar(np::ndarray& O, const np::ndarray& C)
{
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> row_matrix_t;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> col_matrix_t;

  int ndim = O.get_nd();
  BAssertThrow(ndim == 1, "dimension mismatch");
  const Py_intptr_t* shapes = O.get_shape();
  BAssertThrow(shapes[0] == polar_basis.n_dofs(), "wrong basis size");

  BAssertThrow(C.get_nd() == 2, "dimension mismatch");
  BAssertThrow(C.get_shape()[0] == K_ && C.get_shape()[1] == K_, "shape mismatch");

  const double* src = reinterpret_cast<const double*>(C.get_data());
  double* dst = reinterpret_cast<double*>(O.get_data());
  Eigen::Map<Eigen::VectorXd> vdst(dst, polar_basis.size());

  np::ndarray::bitflag flags = C.get_flags();
  if ((flags & np::ndarray::F_CONTIGUOUS) != 0x0) {
    // col-major matrix
    Eigen::Map<const col_matrix_t> mO(src, K_, K_);
    p2n.to_polar(vdst, mO);
  } else if ((flags & np::ndarray::C_CONTIGUOUS) != 0x0) {
    Eigen::Map<const row_matrix_t> mO(src, K_, K_);
    p2n.to_polar(vdst, mO);
  } else {
    BOOST_VERIFY(false);
  }
}


// ------------------------------------------------------------
np::ndarray P2N_wrapper::to_polar_ret(const np::ndarray& C)
{
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> row_matrix_t;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> col_matrix_t;

  BAssertThrow(C.get_nd() == 2, "dimension mismatch");
  BAssertThrow(C.shape(0) == K_, "size mismatch");
  BAssertThrow(C.shape(1) == K_, "size mismatch");


  const double* src = reinterpret_cast<const double*>(C.get_data());
  np::ndarray Out = np::empty(bp::make_tuple(polar_basis.size()),
                          np::dtype::get_builtin<double>());

  double* dst = reinterpret_cast<double*>(Out.get_data());
  Eigen::Map<Eigen::VectorXd> vdst(dst, polar_basis.size());

  np::ndarray::bitflag flags = C.get_flags();
  if ((flags & np::ndarray::F_CONTIGUOUS) != 0x0) {
    // col-major matrix
    Eigen::Map<const col_matrix_t> mO(src, K_, K_);
    p2n.to_polar(vdst, mO);
  } else if ((flags & np::ndarray::C_CONTIGUOUS) != 0x0) {
    Eigen::Map<const row_matrix_t> mO(src, K_, K_);
    p2n.to_polar(vdst, mO);
  } else {
    BOOST_VERIFY(false);
  }
  return Out;
}




#endif
// ======================================== ShiftPolar ========================================

template <typename numeric_t>
class ShiftPolar
{
 public:
  ShiftPolar();
  ShiftPolar(int K) { this->init(K); }
#ifdef PYTHON
  void shift_numpy(np::ndarray& O, const np::ndarray& C, double x, double y);
  np::ndarray shift_numpy_ret(const np::ndarray& C, double x, double y);
#endif
  void shift(double* dst, const double* src, double x, double y);

 private:
  void init_from_file(const std::string& fname);
  void init(int K);

 private:
  typedef typename SpectralBasisFactoryKS::basis_type polar_basis_t;
  typedef typename SpectralBasisFactoryHN::basis_type hermite_basis_t;
  typedef Polar2Hermite<polar_basis_t, hermite_basis_t> P2H_t;
  typedef ShiftHermite2D<hermite_basis_t, numeric_t> shift_hermite_2d_t;

 private:
  // read polar basis from file
  polar_basis_t polar_basis;
  hermite_basis_t hermite_basis;

  std::shared_ptr<P2H_t> ptr_p2h;
  std::shared_ptr<shift_hermite_2d_t> ptr_shift_hermite_2d;

  std::vector<double> buf;
};

// ------------------------------------------------------------------------------------------
template <typename numeric_t>
ShiftPolar<numeric_t>::ShiftPolar()
{
  this->init_from_file("spectral_basis.desc");
}

// ------------------------------------------------------------------------------------------
template <typename numeric_t>
void ShiftPolar<numeric_t>::init_from_file(const std::string& fname)
{
  SpectralBasisFactoryKS::create(polar_basis, fname);
  int max_deg = spectral::get_max_k(polar_basis);
  const unsigned int K = max_deg + 1;
  // create corresponding Hermite basis
  typedef typename SpectralBasisFactoryHN::basis_type hermite_basis_t;
  // std::cout  << "K = " << K << std::endl;
  SpectralBasisFactoryHN::create(hermite_basis, max_deg + 1, 2);
  // SpectralBasisFactoryHN::write_basis_descriptor(hermite_basis, "hermite_basis.desc");

  if (hermite_basis.n_dofs() != polar_basis.n_dofs()) {
    throw runtime_error(
        "Hermite basis does not match!, K=" + boost::lexical_cast<std::string>(K) +
        ", size(polar_basis) =" + boost::lexical_cast<std::string>(polar_basis.n_dofs()));
    exit(1);
  }

  /* std::cout << "size(polar basis) = " << polar_basis.n_dofs() */
  /*           << std::endl */
  /*           << "size(hermite basis) = " << hermite_basis.n_dofs(); */
  /* cout << "\n--------------------\n"; */
  /* cout << "Test 2: (P->H) -> (H->P) show coefficients\n"; */
  ptr_p2h = std::shared_ptr<P2H_t>(
      new Polar2Hermite<polar_basis_t, hermite_basis_t>(polar_basis, hermite_basis));

  ptr_shift_hermite_2d = std::shared_ptr<shift_hermite_2d_t>(new shift_hermite_2d_t(hermite_basis));
  ptr_shift_hermite_2d->init();
}

// ------------------------------------------------------------------------------------------
template <typename numeric_t>
void ShiftPolar<numeric_t>::init(int K_)
{
  SpectralBasisFactoryKS::create(polar_basis, K_);
  int max_deg = spectral::get_max_k(polar_basis);
  const unsigned int K = max_deg + 1;
  // create corresponding Hermite basis
  typedef typename SpectralBasisFactoryHN::basis_type hermite_basis_t;
  SpectralBasisFactoryHN::create(hermite_basis, max_deg + 1);
  if (hermite_basis.n_dofs() != polar_basis.n_dofs()) {
    throw runtime_error(
        "Hermite basis does not match!, K=" + boost::lexical_cast<std::string>(K) +
        ", size(polar_basis) =" + boost::lexical_cast<std::string>(polar_basis.n_dofs()));
    exit(1);
  }

  ptr_p2h = std::shared_ptr<P2H_t>(
      new Polar2Hermite<polar_basis_t, hermite_basis_t>(polar_basis, hermite_basis));

  ptr_shift_hermite_2d = std::shared_ptr<shift_hermite_2d_t>(new shift_hermite_2d_t(hermite_basis));
  ptr_shift_hermite_2d->init();
}

#ifdef PYTHON
// ------------------------------------------------------------
template <typename numeric_t>
void ShiftPolar<numeric_t>::shift_numpy(np::ndarray& O, const np::ndarray& C, double x, double y)
{
  const double* src = reinterpret_cast<double*>(C.get_data());
  double* dst = reinterpret_cast<double*>(O.get_data());
  this->shift(dst, src, x, y);
}


template <typename numeric_t>
np::ndarray ShiftPolar<numeric_t>::shift_numpy_ret(const np::ndarray& C, double x, double y)
{
  np::ndarray O = np::empty(bp::make_tuple(polar_basis.size()),
                          np::dtype::get_builtin<double>());

  const double* src = reinterpret_cast<double*>(C.get_data());
  double* dst = reinterpret_cast<double*>(O.get_data());
  this->shift(dst, src, x, y);
  return O;
}

#endif

// ------------------------------------------------------------
template <typename numeric_t>
void ShiftPolar<numeric_t>::shift(double* dst, const double* src, double x, double y)
{
  int N = polar_basis.n_dofs();
  std::vector<numeric_t> buf1(polar_basis.n_dofs());
  std::vector<double> cH_double(polar_basis.n_dofs());
  typedef Eigen::Map<const Eigen::VectorXd> cmap_t;
  typedef Eigen::Map<Eigen::VectorXd> map_t;

  {
    map_t vdst(cH_double.data(), N);
    ptr_p2h->to_hermite(vdst, cmap_t(src, N));
  }

  std::transform(
      cH_double.begin(), cH_double.end(), buf1.begin(), [](double x) { return numeric_t(x); });

  ptr_shift_hermite_2d->shift(buf1.data(), x, y);

  std::transform(
      buf1.begin(), buf1.end(), cH_double.begin(), [](numeric_t x) { return double(x); });

  {
    map_t vdst(dst, N);
    ptr_p2h->to_polar(vdst, cmap_t(cH_double.data(), N));
  }
}


#ifdef PYTHON

BOOST_PYTHON_MODULE(libSpectralTools)
{
  using namespace boost::python;
  // import for handling of numpy arrays
  //  import_array();
  np::initialize();
  //  numeric::array::set_module_and_type("numpy", "ndarray");
  class_<P2H_wrapper, boost::noncopyable>("Polar2Hermite", init<>())
      .def(init<int>(args("K: degree"), "Polar <-> Hermite transform"))
      .def("to_polar", &P2H_wrapper::to_polar, "DST, SRC")
      .def("to_polar", &P2H_wrapper::to_polar_ret, "SRC")
      .def("to_hermite", &P2H_wrapper::to_hermite, "DST, SRC")
      .def("to_hermite", &P2H_wrapper::to_hermite_ret, "SRC");
  class_<P2N_wrapper, boost::noncopyable>("Polar2Nodal", init<>())
      .def(init<int>(args("K: degree"), "Polar <-> Nodal transform"))
      .def("to_nodal", &P2N_wrapper::to_nodal_ret, "SRC")
      .def("to_nodal", &P2N_wrapper::to_nodal, "DST, SRC")
      .def("to_polar", &P2N_wrapper::to_polar_ret, "SRC")
      .def("to_polar", &P2N_wrapper::to_polar, "DST, SRC");
  class_<ShiftPolar<double>, boost::noncopyable>(
      "ShiftPolar", init<>("Shift transform. reads basis from `spectral_basis.desc`."))
      .def(init<int>(args("K: degree"), "Translate Polar-Laguerre coefficients."))
      .def("shift",
           &ShiftPolar<double>::shift_numpy,
           "DST, SRC returns Polar-Laguerre coefficients of f(x+x0, y+y0)")
      .def("shift",
           &ShiftPolar<double>::shift_numpy_ret,
           "DST returns Polar-Laguerre coefficients of f(x+x0, y+y0)");


}
#else
int main(int argc, char* argv[])
{
  // P2H_wrapper<double> shift_obj;

  // std::vector<double> in(5000);
  // std::vector<double> out(5000);

  // shift_obj.shift(out.data(), in.data(), 1, 1);

  return 0;
}
#endif

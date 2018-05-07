#pragma once

#include <Eigen/Dense>
#include <type_traits>
#include <boost/assert.hpp>

namespace boltzmann {
namespace ct_dense {

enum class ct_input_order
{
  rowMajor = Eigen::RowMajor,
  colMajor = Eigen::ColMajor
};

class ct_input_base
{
 public:
  ct_input_base()
      : n_(0),
        N_(0)
  {/* empty */ }

 protected:
  /// number of input vectors
  int n_;
  /// length
  int N_;
};


template<enum ct_input_order ORDER>
class ct_input
{ };


template<>
class ct_input<ct_input_order::rowMajor> : public ct_input_base
{
 public:
  using mat_t = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  // using map_t = Eigen::Map<mat_t>;

 public:
  template<typename DERIVED, typename CTENSOR>
  void read(const CTENSOR& ct, const Eigen::DenseBase<DERIVED>& src);

  template<typename DERIVED, typename CTENSOR>
  void write(const CTENSOR& ct, Eigen::DenseBase<DERIVED>& dst);

  inline auto get(int offset, int extent);

  inline auto col(int i);

  void resize(int rows, int cols);

 private:
  mat_t data_;
};

template<typename DERIVED, typename CTENSOR>
void
ct_input<ct_input_order::rowMajor>::read(const CTENSOR& ct, const Eigen::DenseBase<DERIVED>& src)
{
  // row major implementation
  data_.resize(src.rows(), src.cols());
  data_ = src;
  N_ = ct.get_N();
  n_ = data_.cols();
}


template<typename DERIVED, typename CTENSOR>
void
ct_input<ct_input_order::rowMajor>::write(const CTENSOR& ct, Eigen::DenseBase<DERIVED>& dst)
{
  // row major implementation
  dst.derived().resize(data_.rows(), data_.cols());
  dst.derived() = data_;
}


inline auto
ct_input<ct_input_order::rowMajor>::get(int offset, int extent)
{
  return data_.block(offset, 0, extent, n_);
}


inline auto
ct_input<ct_input_order::rowMajor>::col(int i)
{
  return data_.col(i);
}


inline void
ct_input<ct_input_order::rowMajor>::resize(int rows, int cols)
{
  N_ = rows;
  n_ = cols;
  data_.resize(rows, cols);
}

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
template<>
class ct_input<ct_input_order::colMajor> : public ct_input_base
{
 public:
  using mat_t = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  using map_t = Eigen::Map<mat_t>;

 public:
  template<typename DERIVED, typename CTENSOR>
  void read(const CTENSOR& ct, const Eigen::DenseBase<DERIVED>& src);

  template<typename DERIVED, typename CTENSOR>
  void write(const CTENSOR& ct, Eigen::DenseBase<DERIVED>& dst);

  inline auto get(int offset, int extent);

  inline auto col(int i);

  inline void resize(int rows, int cols);

 private:
  mat_t data_;
};


template<typename DERIVED, typename CTENSOR>
void
ct_input<ct_input_order::colMajor>::read(const CTENSOR& ct, const Eigen::DenseBase<DERIVED>& src)
{
  // row major implementation
  ct.pad(data_, src);
  N_ = ct.get_N();
  n_ = data_.cols();
}


template<typename DERIVED, typename CTENSOR>
void
ct_input<ct_input_order::colMajor>::write(const CTENSOR& ct, Eigen::DenseBase<DERIVED>& dst)
{
  // row major implementation
  ct.unpad(dst, data_);
}


inline auto
ct_input<ct_input_order::colMajor>::get(int offset, int extent)
{
  return data_.block(offset, 0, extent, n_);
}


inline auto
ct_input<ct_input_order::colMajor>::col(int i)
{
  return data_.col(i);
}


inline void
ct_input<ct_input_order::colMajor>::resize(int rows, int cols)
{
  data_.resize(rows, cols);
}



}  // ct_dense
}  // boltzmann

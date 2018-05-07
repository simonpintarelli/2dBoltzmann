#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <map>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>


namespace boltzmann {

template <typename T>
class SimpleSparseMatrix
{
 public:
  typedef T value_type;
  typedef unsigned int index_t;

  struct entry_t
  {
    entry_t(const T& val_, index_t row_, index_t col_)
        : val(val_)
        , row(row_)
        , col(col_){};

    entry_t() {}

    T val;
    index_t row;
    index_t col;
  };

  typedef entry_t* ptr_t;
  typedef const entry_t* ptr_const_t;

 public:
  SimpleSparseMatrix(index_t N)
      : row_ptrs_(N + 1, std::nullptr_t())
      , N_(N)
      , is_compressed_(false)
      , _rows_(N)
  {
  }

  SimpleSparseMatrix()
      : SimpleSparseMatrix(0){};

  void reinit(index_t N);

  void insert(index_t i, index_t j, value_type&& v);

  /// find index, if found returns pointer to entry, else pointer to end
  ptr_const_t find(const std::pair<index_t, index_t>& key) const;
  /// begin of data_
  ptr_const_t begin() const;
  /// end of data_
  ptr_const_t end() const;

  ptr_const_t row_begin(index_t i) const;
  ptr_const_t row_end(index_t i) const;
  ptr_const_t data() const;
  index_t size() const;
  void print(std::ostream& out) const;

  /// needed for backwards compatibility
  const std::vector<entry_t>& get_vec() const { return data_; }

  void compress();

 protected:
  std::vector<ptr_t> row_ptrs_;
  std::vector<entry_t> data_;
  std::vector<index_t> columns_;
  index_t N_;
  bool is_compressed_;

 private:
  /// temporary storage
  std::vector<std::map<index_t, value_type> > _rows_;
};

// ----------------------------------------------------------------------
template <typename T>
void
SimpleSparseMatrix<T>::reinit(index_t N)
{
  row_ptrs_.resize(N + 1, std::nullptr_t());
  N_ = N;
  is_compressed_ = false;
  _rows_.resize(N, typename decltype(_rows_)::value_type());
}

// ----------------------------------------------------------------------
template <typename T>
void
SimpleSparseMatrix<T>::insert(index_t i, index_t j, value_type&& v)
{
  if (i < N_ && j < N_)
    _rows_[i][j] = v;
  else
    throw std::runtime_error("out of bounds");
}

// ----------------------------------------------------------------------
template <typename T>
void
SimpleSparseMatrix<T>::compress()
{
  unsigned int nentries = 0;
  for (index_t row = 0; row < N_; ++row) {
    nentries += _rows_[row].size();
  }

  data_.resize(nentries);
  columns_.resize(nentries);
  // insert elements form _rows_ into data and update row_ptr_ and columns
  ptr_t row_ptr = data_.data();
  unsigned int counter = 0;
  for (index_t row = 0; row < N_; ++row) {
    row_ptrs_[row] = row_ptr + counter;
    for (auto it = _rows_[row].begin(); it != _rows_[row].end(); ++it) {
      // auto key = std::make_pair(row, it->first);
      // data_[counter] = std::make_pair(key, it->second);
      data_[counter] = entry_t(it->second, row, it->first);
      counter++;
    }
    _rows_[row].clear();
  }
  _rows_ = decltype(_rows_)();
  row_ptrs_[N_] = data_.data() + data_.size();
  is_compressed_ = true;
}

// ----------------------------------------------------------------------
template <typename T>
inline typename SimpleSparseMatrix<T>::ptr_const_t
SimpleSparseMatrix<T>::begin() const
{
  return data_.data();
}

// ----------------------------------------------------------------------
template <typename T>
inline typename SimpleSparseMatrix<T>::ptr_const_t
SimpleSparseMatrix<T>::end() const
{
  return data_.data() + data_.size();
}

// ----------------------------------------------------------------------
template <typename T>
inline typename SimpleSparseMatrix<T>::ptr_const_t
SimpleSparseMatrix<T>::find(const std::pair<index_t, index_t>& key) const
{
  assert(is_compressed_);
  auto i = key.first;
  auto j = key.second;

  for (auto it = row_begin(i); it < row_end(i); ++it) {
    if (it->col == j) return it;
  }
  return end();
}

// ----------------------------------------------------------------------
template <typename T>
inline typename SimpleSparseMatrix<T>::ptr_const_t
SimpleSparseMatrix<T>::row_begin(index_t i) const
{
  assert(is_compressed_);
  return row_ptrs_[i];
}

// ----------------------------------------------------------------------
template <typename T>
inline typename SimpleSparseMatrix<T>::ptr_const_t
SimpleSparseMatrix<T>::row_end(index_t i) const
{
  assert(is_compressed_);
  return row_ptrs_[i + 1];
}

// ----------------------------------------------------------------------
template <typename T>
inline typename SimpleSparseMatrix<T>::ptr_const_t
SimpleSparseMatrix<T>::data() const
{
  assert(is_compressed_);
  return data_.data();
}

// ----------------------------------------------------------------------
template <typename T>
unsigned int
SimpleSparseMatrix<T>::size() const
{
  return data_.size();
}


template<typename T>
void
SimpleSparseMatrix<T>::print(std::ostream& out) const
{
  for(auto it = this->begin(); it < this->end(); ++it) {
    out << it->row << " "
        << it->col << " "
        << std::setprecision(10) << std::scientific << it->val
        << std::endl;
  }
}

}  // end namespace boltzmann

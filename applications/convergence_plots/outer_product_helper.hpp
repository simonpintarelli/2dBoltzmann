#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

template <typename SPARSE_MAT1, typename SPARSE_MAT2, typename VEC_SRC, typename VEC_DST>
void
sparse_outer_product_multiply(VEC_DST& dst_vec,
                              const SPARSE_MAT1& M1,
                              const SPARSE_MAT2& M2,
                              const VEC_SRC& src_vec)
{
  // apply M2
  typedef Eigen::VectorXd vec_t;
  typedef Eigen::Map<vec_t> vvec_t;
  typedef Eigen::Map<const vec_t> const_vvec_t;

  unsigned int Nx = M2.cols();
  unsigned int Ny = M2.rows();
  unsigned int Ly = dst_vec.size() / Ny;
  unsigned int Lx = src_vec.size() / Nx;
  assert(src_vec.size() % Nx == 0);
  assert(dst_vec.size() % Ny == 0);

  typedef typename SPARSE_MAT1::InnerIterator InnerIterator1;
  typedef typename SPARSE_MAT2::InnerIterator InnerIterator2;

  for (unsigned int i = 0; i < Lx; ++i) {
    vvec_t ldest(dst_vec.data() + i * Ny, Ny);
    const_vvec_t lsrc(src_vec.data() + i * Nx, Nx);

    ldest = M2 * lsrc;
  }

  // copy
  vec_t tmp = dst_vec;
  dst_vec *= 0;

  // apply M1
  for (int k = 0; k < M1.outerSize(); ++k) {
    for (InnerIterator1 it(M1, k); it; ++it) {
      vvec_t ldst(dst_vec.data() + it.row() * Ny, Ny);
      const_vvec_t lsrc(tmp.data() + it.col() * Ny, Ny);
      ldst += it.value() * lsrc;
    }
  }
}

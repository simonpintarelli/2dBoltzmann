#include <cstdio>
#include <iostream>

#include "collision_tensor/dense/collision_tensor.hpp"
#include "collision_tensor/dense/multi_slices_factory.hpp"
#include "collision_tensor/dense/storage/slice_memory_layout.hpp"

// own includes ---------------------------------------------------------------
#include "spectral/basis/spectral_basis_dimension_accessor.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"
#include "spectral/basis/spectral_elem_accessor.hpp"

using namespace boltzmann;

int main(int argc, char *argv[])
{
  if (argc < 2) {
    std::cout << "usage: " << argv[0] << " K\n";
    exit(1);
  }

  int K = atoi(argv[1]);

  // ------------------------------------------------------------
  typedef SpectralBasisFactoryKS::basis_type basis_type;
  basis_type basis;
  // typedef typename basis_type::DimAcc::template get_vec<LaguerreRR> get_radial_basis_t;
  SpectralBasisFactoryKS::create(basis, K);

  typename multi_slices_factory::container_t multi_slices;
  multi_slices_factory::create(multi_slices, basis);

  ct_dense::CollisionTensor ct;

  std::cout << "CollisionTensor consumes: " << (ct.nentries() * 8) / 1e6 << "MB" << std::endl;

  return 0;
}

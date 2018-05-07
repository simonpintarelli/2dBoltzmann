#include <fstream>
#include <iostream>

#include "spectral/basis/spectral_basis_factory_ks.hpp"
#include "spectral_transfer_matrix.hpp"

using namespace std;
using namespace boltzmann;

int main(int argc, char *argv[])
{
  typedef typename SpectralBasisFactoryKS::basis_type basis_type;
  basis_type B1;
  basis_type B2;

  SpectralBasisFactoryKS::create(B1, 10, 10, 2, true);
  SpectralBasisFactoryKS::create(B2, 50, 50, 2, true);

  auto T12 = spectral_transfer_matrix(B1, B2);
  auto T21 = spectral_transfer_matrix(B2, B1);

  std::ofstream fout12("T12");
  fout12 << T12;
  fout12.close();

  std::ofstream fout21("T21");
  fout21 << T21;
  fout21.close();

  return 0;
}

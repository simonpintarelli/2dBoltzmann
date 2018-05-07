// system includes
#include <gtest/gtest.h>
// own includes
#include "spectral/basis/spectral_basis_dimension_accessor.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"
#include "spectral/basis/spectral_elem_accessor.hpp"

using namespace boltzmann;

TEST(spectral, basis_factory)
{
  std::vector<int> Ks = {1, 11, 15, 20, 23, 50, 60, 100};
  for (int K : Ks) {
    const char* fname = "spectral_basis.desc.tmp";
    typedef boltzmann::SpectralBasisFactoryKS bfactory_t;
    typedef bfactory_t::basis_type basis_type;
    basis_type spectral_basis;
    bfactory_t::create(spectral_basis, K);
    bfactory_t::write_basis_descriptor(spectral_basis, fname);
    basis_type basis_from_file;
    bfactory_t::create(basis_from_file, fname);

    // check that spectral_basis and basis_from_file are the same
    EXPECT_TRUE(basis_from_file.n_dofs() == spectral_basis.n_dofs()) << "basis size";
    int N = spectral_basis.n_dofs();
    for (int i = 0; i < N; ++i) {
      EXPECT_TRUE(spectral_basis.get_elem(i).id() == basis_from_file.get_elem(i).id())
          << "element " << i;
      auto elem = spectral_basis.get_elem(i);
      int ir = basis_from_file.get_dof_index(elem.id());
      EXPECT_TRUE(i == ir) << "lookup index";
    }
  }
}

TEST(spectral, basisOpEqual)
{
  using basis_type = SpectralBasisFactoryKS::basis_type;

  auto b10 = SpectralBasisFactoryKS::create(10);
  auto b20 = SpectralBasisFactoryKS::create(20);

  bool not_equal_b10_b20 = (b10 == b20);
  bool equal_b10 = (b10 == b10);
  bool equal_b20 = (b20 == b20);

  EXPECT_TRUE(!not_equal_b10_b20);
  EXPECT_TRUE(equal_b10);
  EXPECT_TRUE(equal_b20);
}

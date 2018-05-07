#include <gtest/gtest.h>
#include "quadrature/qhermitew.hpp"
#include "spectral/basis/basis_types.hpp"
#include "spectral/basis/spectral_basis_factory_hermite.hpp"
#include "spectral/eval_handlers.hpp"

using namespace boltzmann;

TEST(spectral, hermite_evaluator)
{
  int K = 40;

  hermite_basis_t hermite_basis;
  SpectralBasisFactoryHN::create(hermite_basis, K);

  int N = 50;
  QHermiteW quad(1, N);

  auto elem1 = hermite_basis.get_elem(10);
  auto elem2 = hermite_basis.get_elem(30);

  auto& x = quad.pts();
  auto& w = quad.wts();

  auto Heval2d = hermite_evaluator2d<>::make(hermite_basis, x);
  double v = 0;
  for (int q1 = 0; q1 < N; ++q1) {
    for (int q2 = 0; q2 < N; ++q2) {
      v += Heval2d(elem1, q1, q2) * Heval2d(elem2, q1, q2) * w[q1] * w[q2];
    }
  }

  EXPECT_NEAR(v, 0, 1e-12) << "hermite overlap integral";
}

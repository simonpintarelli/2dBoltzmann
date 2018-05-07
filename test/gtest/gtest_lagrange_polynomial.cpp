#include <gtest/gtest.h>
#include <boost/lexical_cast.hpp>
#include "quadrature/qhermite.hpp"
#include "spectral/lagrange_polynomial.hpp"

using namespace std;
using namespace boltzmann;

TEST(spectral, LagrangePolySimple)
{
  std::vector<int> ks({10, 20, 30, 40, 80});
  for (int K : ks) {
    QHermite qh(1.0, K);

    lagrange_poly_simple lp(qh.points_data(), qh.size());

    for (int j = 0; j < qh.size(); ++j) {
      for (int i = 0; i < qh.size(); ++i) {
        double val = lp.eval(j, qh.pts(i));
        if (i == j)
          EXPECT_NEAR(val, 1, 1e-12);
        else
          EXPECT_NEAR(val, 0, 1e-12);
      }
    }
  }
}

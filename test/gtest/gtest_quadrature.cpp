#include <gtest/gtest.h>
#include <boost/math/constants/constants.hpp>
#include <functional>
#include <vector>
#include <iostream>
#include <iomanip>

#include "quadrature/qhermite.hpp"
#include "quadrature/qhermitew.hpp"
#include "quadrature/qmaxwell.hpp"
#include "quadrature/qmaxwellw.hpp"
#include "spectral/hermiten.hpp"
#include "spectral/hermitenw.hpp"

using namespace std;
using namespace boltzmann;

TEST(quadrature, maxwell)
{
  int digits = 128;

  static const double PI = boost::math::constants::pi<double>();
  std::vector<int> Ns = {1, 5, 10, 21, 40, 60, 80};

  for (auto N : Ns) {
    QMaxwell qmaxwell(1, N, digits);

    // test integration \int \exp{-r^2} r \dd r = pi
    double sum = 0;
    for (int i = 0; i < N; ++i) {
      sum += qmaxwell.wts(i);
    }
    sum *= 2 * PI;
    EXPECT_NEAR(sum, PI, 1e-12) << boost::lexical_cast<string>(N)
                                << ": integrate: e^{r^2} r, on [0,\\infty)";

    // test integration \int r \exp{-r^2} r \dd r = 1/2 pi^3/2
    sum = 0;
    for (int i = 0; i < N; ++i) {
      sum += qmaxwell.wts(i) * qmaxwell.pts(i);
    }
    sum *= 2 * PI;
    EXPECT_NEAR(sum, 0.5 * std::pow(PI, 1.5), 1e-12)
        << boost::lexical_cast<string>(N) << "integrate: e^{r^2} r^2, on [0,\\infty)";
  }
}

TEST(quadrature, maxwellw)
{
  int digits = 128;

  static const double PI = boost::math::constants::pi<double>();
  std::vector<int> Ns = {1, 5, 10, 21, 40, 60, 80};

  for (auto N : Ns) {
    QMaxwellW qmaxwell(1, N, digits);

    // test integration \int \exp{-r^2} r \dd r = pi
    double sum = 0;
    for (int i = 0; i < N; ++i) {
      double x = qmaxwell.pts(i);
      sum += qmaxwell.wts(i) * std::exp(-x*x);
    }
    sum *= 2 * PI;
    EXPECT_NEAR(sum, PI, 1e-12) << boost::lexical_cast<string>(N)
                                << ": integrate: e^{r^2} r, on [0,\\infty)";

    // test integration \int r \exp{-r^2} r \dd r = 1/2 pi^3/2
    sum = 0;
    for (int i = 0; i < N; ++i) {
      double x = qmaxwell.pts(i);
      sum += qmaxwell.wts(i) * qmaxwell.pts(i) * std::exp(-x*x);
    }
    sum *= 2 * PI;
    EXPECT_NEAR(sum, 0.5 * std::pow(PI, 1.5), 1e-12)
        << boost::lexical_cast<string>(N) << "integrate: e^{r^2} r^2, on [0,\\infty)";
  }
}





/**
 *  @brief Test Gauss-Hermite quadrature to integrate 1, x, x^2
 *
 *  Detailed description
 *
 *  @param param
 *  @return return type
 */
TEST(quadrature, hermite_quad_only)
{
  std::vector<int> Ns = {2, 5, 10, 21, 40, 60, 80, 120};
  int digits = 128;
  cout << "Testing n=" << std::for_each(Ns.begin(), Ns.end(), [](int x) { cout << x << " "; })
       << "\n";

  static const double PI = boost::math::constants::pi<double>();

  for (auto N : Ns) {
    QHermite quad(1.0, /* alpha */
                  N,   /*  num. quad. poins */
                  digits);

    {
      double sum = 0;
      for (int i = 0; i < N; ++i) {
        sum += quad.wts(i);
      }
      EXPECT_NEAR(sum, std::sqrt(PI), 1e-12)
          << boost::lexical_cast<string>(N) << "integrate e^{-x^2} on (-\\infty, \\infty)";
    }
    {
      double sum = 0;
      for (int i = 0; i < N; ++i) {
        sum += quad.wts(i) * quad.pts(i);
      }
      EXPECT_NEAR(sum, 0, 1e-12) << boost::lexical_cast<string>(N)
                                 << "integrate x e^{-x^2} on (-\\infty, \\infty)";
    }
    {
      double sum = 0;
      for (int i = 0; i < N; ++i) {
        sum += quad.wts(i) * quad.pts(i) * quad.pts(i);
      }
      EXPECT_NEAR(sum, std::sqrt(PI) / 2, 1e-12)
          << boost::lexical_cast<string>(N) << "integrate x^2 e^{-x^2} on (-\\infty, \\infty)";
    }
  }
}

/**
 *  @brief Test hermite quadrature rule by computing overlap integrals of Hermite polynomials.
 *
 *
 */
TEST(quadrature, hermite_full)
{
  // hint: quadrature rule for N=1 does not exist!
  std::vector<int> Ns = {2, 5, 10, 21, 40, 60, 80, 120};
  cout << "Testing n=" << std::for_each(Ns.begin(), Ns.end(), [](int x) { cout << x << " "; })
       << "\n";
  typedef double numeric_t;

  for (auto N : Ns) {
    QHermiteW quad(1.0, N);
    auto& wts = quad.wts();
    auto& pts = quad.pts();

    HermiteNW<numeric_t> HW(N);
    HW.compute(quad.pts());

    for (unsigned i = 0; i < N; ++i) {
      double val = 0;
      for (unsigned int q = 0; q < quad.size(); ++q) {
        // const double wh = exp(0.5*quad.pts(q) * quad.pts(q));
        val += (HW.get(i)[q]) * (HW.get(i)[q]) * quad.wts(q);
      }
      EXPECT_NEAR(val, 1, 1e-12)
          << "(h_i, h_i) "
          << "i=" << boost::lexical_cast<string>(i) << ", N=" << boost::lexical_cast<string>(N)
          << " (evaluatued Hermite polynomials _with_ expt weight aka Hermite functions)";
    }
  }
}

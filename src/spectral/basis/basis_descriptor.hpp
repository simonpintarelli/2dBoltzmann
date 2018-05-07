#pragma once

namespace boltzmann {
struct BasisDescriptor
{
  BasisDescriptor(int K_, int L_, double beta_)
      : K(K_)
      , L(L_)
      , beta(beta_)
  {
  }
  BasisDescriptor()
      : K(0)
      , L(0)
      , beta(0)
  {
  }
  const int K;
  const int L;
  const double beta;

  static BasisDescriptor DEFAULT()
  {
    BasisDescriptor b;
    return b;
  }
};

}  // end boltzmann

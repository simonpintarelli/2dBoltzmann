#pragma once

#include <mutex>
#include <tuple>
#include <utility>

#include "spectral/basis/toolbox/spectral_basis.hpp"
#include "spectral/polar_to_nodal.hpp"
#include "aux/singleton_collection.hpp"

namespace boltzmann {


template <typename PolarBasis = SpectralBasisFactoryKS::basis_type>
class P2NFactory : public SingletonCollection<P2NFactory<PolarBasis>,
                                              Polar2Nodal<PolarBasis>>
{
 public:
  typedef Polar2Nodal<PolarBasis> value_t;
  typedef PolarBasis basis_t;

 public:
  value_t& make(const basis_t& basis, double a = 1.0);

 private:
  /// key is (basis size, exp-weight)
  typedef std::tuple<int, double> key_t;
  std::map<key_t, value_t> p2n_;
  std::mutex mutex_;
};


template <typename PolarBasis>
typename P2NFactory<PolarBasis>::value_t&
P2NFactory<PolarBasis>::make(const basis_t& basis, double a)
{
  int K = spectral::get_K(basis);
  key_t key(K, a);
  auto it = p2n_.find(key);
  if (it == p2n_.end()) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = p2n_.emplace(
        std::piecewise_construct, std::forward_as_tuple(key), std::forward_as_tuple(basis, a));
    return it.first->second;
  } else {
    return it->second;
  }
}


}  // namespace boltzmann

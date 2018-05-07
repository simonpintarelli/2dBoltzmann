#pragma once

#include <boost/functional/hash.hpp>
#include <map>
#include <unordered_map>

// own includes ------------------------------------------------------------
#include "aux/hash_specializations.hpp"

namespace boltzmann {
namespace collision_tensor_assembly {

class GainCache
    : public std::unordered_map<std::tuple<unsigned long long, unsigned long long>, double>
{
};

}  // end namespace collision_tensor_assembly
}  // end namespace boltzmann

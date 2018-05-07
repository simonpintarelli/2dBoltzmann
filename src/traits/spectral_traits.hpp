#pragma once

#include "spectral/basis/spectral_elem.hpp"


namespace boltzmann {

// forward declarations
class SpectralBasisFactoryKS;
class XiR;
class LaguerreKS;

template <typename ELEM>
struct SpectralFactoryTraits
{
};

template <>
struct SpectralFactoryTraits<SpectralElem<double, XiR, LaguerreKS>>
{
  typedef SpectralBasisFactoryKS basis_factory_t;
};

}  // boltzmann

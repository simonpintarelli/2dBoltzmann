#pragma once

#include "enum/enum.hpp"


namespace boltzmann {

class DoFMapper;
class DoFMapperPeriodic;
class DoFMapper1DPeriodicX;
template <class>
class DoFMapperPeriodicDistributed;

namespace traits {

template <enum BC_Type>
struct DoFMapper
{
  /* empty */
};

template <>
struct DoFMapper<BC_Type::XPERIODIC>
{
  typedef DoFMapperPeriodicDistributed<DoFMapper1DPeriodicX> type;
};

template <>
struct DoFMapper<BC_Type::REGULAR>
{
  typedef boltzmann::DoFMapper type;
};

}  // end namespace traits
}  // end namespace boltzmann

#pragma once

namespace boltzmann {
namespace ct_dense {

template <typename size_type = unsigned int>
struct SubBlock
{
  SubBlock() { /* empty */}
  size_type offset_x = 0;  // row
  size_type offset_y = 0;  // column
  size_type size_x = 0;    // row
  size_type size_y = 0;    // column
  size_type size_z = 0;    // thickness
  inline size_type size() const { return size_x * size_y; }
};

}  // ct_dense
}  // end namespace boltzmann

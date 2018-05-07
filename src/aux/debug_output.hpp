#pragma once

#include <deal.II/lac/vector.h>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/lexical_cast.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>


namespace boltzmann {
// ------------------------------------------------------------
template <typename T>
void
Dwrite_to_file(const T& obj, std::string fname)
{
#ifdef DEBUG
  std::ofstream fout(fname.c_str());
  obj.print(fout);
  fout.close();
#endif
}

// ------------------------------------------------------------
template <typename T>
void
write_to_file(const T& obj, std::string fname)
{
  std::ofstream fout(fname.c_str());
  obj.print(fout);
  fout.close();
}

}  // end namespace boltzmann

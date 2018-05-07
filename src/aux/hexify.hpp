#pragma once

#include <iomanip>
#include <sstream>


template<typename T>
std::string hexify(T t)
{
  std::stringbuf buf;
  std::ostream os(&buf);

  os << "0x" << std::setfill('0') << std::setw(sizeof(T)*2) << std::hex << t;
  return buf.str().c_str();
}

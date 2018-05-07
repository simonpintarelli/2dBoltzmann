#pragma once

#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>


namespace boltzmann {

namespace local_ {
template<typename T>
struct duration_string
{

};

template<>
struct duration_string<std::chrono::milliseconds>
{
  constexpr static char const* label="[ms]";
};
//inline const std::string duration_string<std::chrono::milliseconds>::label = "[ms]";

template<>
struct duration_string<std::chrono::microseconds>
{
  constexpr static char const* label="[us]";
};

template<>
struct duration_string<std::chrono::nanoseconds>
{
  constexpr static char const* label="[ns]";
};

}  // local_

template<typename T=std::chrono::milliseconds>
class Timer
{
 public:
  void start() { this->t = std::chrono::high_resolution_clock::now(); }

  long long int stop()
  {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<T>(now - this->t).count();
  }

  void print(std::ostream& out, long long int tlap, const std::string& label)
  {
    out << std::setw(17) << std::left << ("TIMINGS::" + label) << ": " << std::setw(15)
        << std::scientific << tlap << local_::duration_string<T>::label << "\n";
  }

 private:
  typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_point;
  time_point t;
};

}  // end boltzmann

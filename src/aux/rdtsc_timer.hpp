#pragma once

#include <stdint.h>
#include <sys/time.h>
#include <time.h>
#include <cassert>

#include <iomanip>
#include <memory>


namespace boltzmann {

extern inline unsigned long long __attribute__((always_inline)) rdtsc()
{
  unsigned int hi, lo;
  __asm__ __volatile__(
      "xorl %%eax, %%eax\n\t"
      "cpuid\n\t"
      "rdtsc"
      : "=a"(lo), "=d"(hi)
      : /* no inputs */
      : "rbx", "rcx");
  return ((unsigned long long)hi << 32ull) | (unsigned long long)lo;
}

/**
 * @brief Measures in CPU cycles
 *
 */
class RDTSCTimer
{
 public:
  RDTSCTimer()
      : started(false)
      , tbegin(0)
  {
  }

  inline void start()
  {
    started = true;
    tbegin = rdtsc();
  }

  inline uint64_t stop()
  {
    uint64_t tend = rdtsc();
    assert(started);
    started = false;
    return tend - tbegin;
  }

  void print(std::ostream& out, uint64_t tlap, const std::string& label)
  {
    out << std::setw(17) << std::left << ("TIMINGS::" + label) << ": " << std::setw(10)
        << std::scientific << std::setprecision(4) << tlap / 1e9 << " [Gcycles]\n";
  }

 private:
  bool started;
  uint64_t tbegin;
};

}  // boltzmann

#pragma once

#include <iomanip>
#include <iostream>
#include <string>


template <typename OSTREAM = std::ostream>
static void
print_timer(double t, std::string tag, OSTREAM& out = std::cout)
{
  out << std::setw(30) << std::left << "Timing task: <" + tag + "> "
      << "took " << t << " ms" << std::endl;
}

#include <deal.II/base/function_parser.h>
#include <deal.II/base/point.h>
#include <iostream>
#include <map>
#include <unordered_map>
#include <boost/functional/hash.hpp>

using namespace dealii;


namespace std {
template<>
class hash<dealii::Point<2>>
{
 public:
  std::size_t operator()(const dealii::Point<2>& x) const
  {
    std::size_t current = std::hash<double>()(x[0]);
    boost::hash_combine(current, std::hash<double>()(x[1]));

    return current;
  }

};
}  // std


int main(int argc, char *argv[])
{
  // set up problem:
  std::string variables = "x,y";
  std::string expression = "cos(pi*x)+sqrt(y)";
  std::map<std::string,double> constants;
  constants["pi"] = 3.14159265358979323846264338328;
  // FunctionParser with 2 variables and 1 component:
  FunctionParser<2> fp(1);
  fp.initialize(variables,
                expression,
                constants);
  // Point at which we want to evaluate the function
  Point<2> point(1.0, 4.0);
  // evaluate the expression at 'point':
  double result = fp.value(point);
  std::cout << "result " << result << "\n";
  std::unordered_map<Point<2>, double> huch;

  return 0;
}

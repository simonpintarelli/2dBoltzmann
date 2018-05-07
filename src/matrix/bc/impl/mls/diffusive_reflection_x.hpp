#pragma once

#include <deal.II/base/function_parser.h>
#include <deal.II/base/point.h>

#include "matrix/bc/impl/mls/diffusive_reflection.hpp"

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


namespace boltzmann {
namespace impl_mls {

class DiffusiveReflectionX : public boltzmann::impl::flux_worker
{
 public:
  DiffusiveReflectionX(const vec_t& hw, const vec_t& hx, double vt, const std::string& Tx)
      : hw_(hw),
        hx_(hx),
        vt_(vt),
        Tx_(1 /* output dim */)
  {
    std::map<std::string, double> constants;
    constants["pi"] = 3.14159265358979323846264338328;
    std::string variables = "x,y";
    Tx_.initialize(variables, Tx, constants);
  }

  virtual void apply(mat_t& out, const mat_t& in, const dealii::Point<2>& x) const;

 private:
  vec_t hw_;
  vec_t hx_;
  double vt_;
  dealii::FunctionParser<2> Tx_;
  mutable std::unordered_map<dealii::Point<2>, DiffusiveReflection> values_;
};


void DiffusiveReflectionX::apply(mat_t& out, const mat_t& in, const dealii::Point<2>& x) const
{
  auto it = values_.find(x);
  if (it != values_.end()) {
    it->second.apply(out, in);
  } else {
    double Tloc = Tx_.value(x);
    auto it = values_.emplace(std::make_pair(x, DiffusiveReflection(hw_, hx_, vt_, Tloc)));
    it.first->second.apply(out, in);
  }
}



}  // impl_mls
}  // boltzmann

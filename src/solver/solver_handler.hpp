#pragma once

// system includes -----------------------------------------------------------------
#include <yaml-cpp/yaml.h>
#include <memory>
#include <stdexcept>

// deal.II includes ----------------------------------------------------------------
#include <deal.II/base/parameter_handler.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_solver.h>

#include "base/logger.hpp"

namespace boltzmann {

class SolverHandler
{
 public:
  typedef dealii::TrilinosWrappers::SolverBase solver_t;
  typedef dealii::TrilinosWrappers::PreconditionBase precondition_t;

 protected:
  typedef std::shared_ptr<solver_t> solver_ptr_t;
  typedef std::shared_ptr<precondition_t> prec_ptr_t;

 public:
  SolverHandler() {}

  template <typename MATRIX>
  void init(YAML::Node config, const MATRIX& matrix);

  solver_t& get_solver();
  precondition_t& get_preconditioner();

 protected:
  /// Linear solver
  solver_ptr_t solver_;
  /// Preconditioner
  prec_ptr_t prec_;

 private:
  dealii::SolverControl control_;
  bool is_initialized_ = false;
};

// --------------------------------------------------------------------------------
template <typename MATRIX>
void
SolverHandler::init(YAML::Node config, const MATRIX& matrix)
{
  auto node = config["Solver"];

  std::string type = node["type"].as<std::string>();
  auto& logger = Logger::GetInstance();
  logger.push_prefix("transport");
  logger.push_prefix("SolverHandler");

  // initialize solver
  if (std::strcmp("gmres", type.c_str()) == 0) {
    // prepare GMRES-SOLVER
    bool log_result = node["log result"].as<bool>();
    bool log_history = node["log history"].as<bool>();
    int restart = node["restart"].as<int>();
    int maxiter = node["maxiter"].as<int>();
    typedef dealii::TrilinosWrappers::SolverGMRES gmres_solver_t;
    typedef typename gmres_solver_t::AdditionalData additional_data;
    control_.set_max_steps(maxiter);
    control_.log_history(log_history);
    control_.log_result(log_result);
    if (node["tol"]) {
      double tol = node["tol"].as<double>();
      control_.set_tolerance(tol);
    }
    solver_ = solver_ptr_t(new gmres_solver_t(control_, additional_data(true, restart)));
    logger << " using GMRES\n";
    logger << "log_history: " << log_history;
    logger << "log_result: " << log_result;

  } else if (std::strcmp("cg", type.c_str()) == 0) {
    // prepare CG-SOLVER
    typedef dealii::TrilinosWrappers::SolverCG cg_solver_t;
    bool log_result = node["log result"].as<bool>();
    bool log_history = node["log history"].as<bool>();
    double tol = node["tol"].as<double>();
    control_.log_history(log_history);
    control_.log_result(log_result);
    control_.set_tolerance(tol);
    solver_ = solver_ptr_t(new cg_solver_t(control_));
    logger << " using CG\n";
    logger << "log_history: " << log_history;
    logger << "log_result: " << log_result;
  } else {
    throw std::runtime_error("invalid solver type");
  }

  // initialize preconditioner
  if (config["Preconditioner"]) {
    std::string type = config["Preconditioner"].as<std::string>();
    if (std::strcmp("ilu", type.c_str()) == 0) {
      // ILU Preconditioner
      typedef dealii::TrilinosWrappers::PreconditionILU P_t;
      prec_ = prec_ptr_t(new P_t);
      std::dynamic_pointer_cast<P_t>(prec_)->initialize(matrix);
      logger << " Precondition ILU\n";
    } else if (std::strcmp("amg", type.c_str()) == 0) {
      // AMG Preconditioner
      typedef dealii::TrilinosWrappers::PreconditionAMG P_t;
      // TODO add this to input file or infer from problem
      bool is_elliptic = true;
      static typename P_t::AdditionalData additional_data(is_elliptic);
      prec_ = prec_ptr_t(new P_t);
      std::dynamic_pointer_cast<P_t>(prec_)->initialize(matrix, additional_data);
      logger << " Precondition AMG\n";
    } else if (std::strcmp("blockwisedirect", type.c_str()) == 0) {
      // AMG Preconditioner
      typedef dealii::TrilinosWrappers::PreconditionBlockwiseDirect P_t;
      // TODO add this to input file or infer from problem
      prec_ = prec_ptr_t(new P_t);
      std::dynamic_pointer_cast<P_t>(prec_)->initialize(matrix);
      logger << " Precondition BlockwiseDirect\n";
    } else if (std::strcmp("ic", type.c_str()) == 0) {
      // AMG Preconditioner
      typedef dealii::TrilinosWrappers::PreconditionIC P_t;
      prec_ = prec_ptr_t(new P_t);
      std::dynamic_pointer_cast<P_t>(prec_)->initialize(matrix);
      logger << " Precondition IC\n";
    } else if (std::strcmp("chebyshev", type.c_str()) == 0) {
      typedef dealii::TrilinosWrappers::PreconditionChebyshev P_t;
      prec_ = prec_ptr_t(new P_t);
      std::dynamic_pointer_cast<P_t>(prec_)->initialize(matrix);
      logger << " Precondition Chebyshev\n";
    } else if (std::strcmp("none", type.c_str()) == 0) {
      prec_ = prec_ptr_t(new dealii::TrilinosWrappers::PreconditionIdentity);
      logger << " Precondition Identity\n";
    } else {
      throw std::runtime_error("invalid preconditioner type");
    }
  } else {
    // Preconditioner section is optional, use none if not specified
    prec_ = prec_ptr_t(new dealii::TrilinosWrappers::PreconditionIdentity);
    logger << " Precondition Identity\n";
  }

  is_initialized_ = true;
}

// --------------------------------------------------------------------------------
typename SolverHandler::solver_t&
SolverHandler::get_solver()
{
  if (!is_initialized_) throw std::runtime_error("SolverHandler: not initialized!");
  return *solver_;
}

// --------------------------------------------------------------------------------
typename SolverHandler::precondition_t&
SolverHandler::get_preconditioner()
{
  if (!is_initialized_) throw std::runtime_error("SolverHandler: not initialized!");
  return *prec_;
}

}  // end namespace dealii

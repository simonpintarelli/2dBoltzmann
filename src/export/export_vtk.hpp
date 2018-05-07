#pragma once

#include <boost/lexical_cast.hpp>
#include <fstream>
#include <iostream>
#include <string>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>

namespace boltzmann {

// ----------------------------------------------------------------------
class VtkExporter
{
 public:
  VtkExporter(const dealii::DoFHandler<2>& dofhandler)
      : dofhandler(dofhandler)
  { /* empty */
  }

  // ------------------------------------------------------------
  void operator()(const dealii::Vector<double>& solution,
                  const std::string fname = "nx.vtk",
                  const dealii::DataOutBase::VtkFlags flags = dealii::DataOutBase::VtkFlags()) const
  {
    typedef dealii::DataOut<2> out_t;
    out_t data_out;
    data_out.set_flags(flags);
    data_out.attach_dof_handler(dofhandler);
    data_out.add_data_vector(solution, "nx");
    data_out.build_patches();
    std::ofstream fout(fname.c_str());
    data_out.write_vtk(fout);
    fout.close();
  }

  dealii::DataOutBase::VtkFlags get_vtk_flags(double time, unsigned int cycle) const
  {
    return dealii::DataOutBase::VtkFlags(time, cycle);
  }

  // ------------------------------------------------------------
  std::string get_fname(unsigned int frameid, std::string prefix = "nx_") const
  {
    return prefix + boost::lexical_cast<std::string>(frameid) + ".vtk";
  }

 private:
  const dealii::DoFHandler<2>& dofhandler;
};

}  // end namespace boltzmann

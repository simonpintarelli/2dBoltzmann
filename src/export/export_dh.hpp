#pragma once

#include <fstream>
#include <iomanip>

#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>

#include "grid/grid_tools.hpp"


namespace boltzmann {

template <int dim>
inline void
export_dh(const dealii::DoFHandler<dim>& dh, std::string filename = "dof.desc")
{
  // const unsigned int dofs_per_cell = dh.get_fe().dofs_per_cell;

  const auto& tria = dh.get_triangulation();
  const auto& vertices = tria.get_vertices();

  std::ofstream fout(filename);

  for (unsigned int i = 0; i < vertices.size(); ++i) {
    fout << std::setw(15) << i << "\t" << vertices[i] << std::endl;
  }

  fout.close();

  {
    auto vertex2dofidx = vertex_to_dof_index(dh);
    std::ofstream fout("vertex2dofidx.dat");
    for (auto it = vertex2dofidx.begin(); it != vertex2dofidx.end(); ++it) {
      fout << it->first << "\t" << it->second << std::endl;
    }
    fout.close();
  }
}

}  // end namespace boltzmann

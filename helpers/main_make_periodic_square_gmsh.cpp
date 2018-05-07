// ---------------------------------------------------------------------
//
// Copyright (C) 1998 - 2013 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>
#include <boost/lexical_cast.hpp>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

using namespace dealii;

template <int dim>
void create_grid()
{
  //  SolutionTransfer<dim> soltrans(dof_handler);
  int nref = 16;
  // test a): pure refinement
  for (int i = 0; i < nref; ++i) {
    // Triangulation<dim> tria(Triangulation<dim>::allow_anisotropic_smoothing);
    Triangulation<dim> tria;

    Point<dim> p0(0, 0);
    const double hloc = std::pow(0.5, i);
    Point<dim> p1(hloc, 1);

    /*
     *       3
     *   *-------*
     *   |       |
     * 0 |       | 1
     *   |       |
     *   *-------*
     *       2
     *
     */

    bool colorize = true;
    GridGenerator::hyper_rectangle(tria, p0, p1, colorize);
    // std::vector<unsigned char> bdids=tria.get_boundary_ids();
    // for (auto v: bdids) {
    //   std::cout << "bd ind: " << (int)v << std::endl;
    // }

    std::cout << i << "\t" << hloc << std::endl;

    for (int k = 0; k < i; ++k) {
      typename Triangulation<dim>::active_cell_iterator cell = tria.begin_active(),
                                                        endc = tria.end();

      for (; cell != endc; ++cell) cell->set_refine_flag(RefinementCase<dim>::cut_y);

      tria.prepare_coarsening_and_refinement();
      tria.execute_coarsening_and_refinement();
    }

    GridOut grid_out;

    // write faces and lines (with boundary indicators)
    grid_out.set_flags(GridOutFlags::Msh(true, true));

    char buf[256];
    std::sprintf(buf, "pipe1d_%03d.msh", i);
    std::cout << "fname= " << buf << std::endl;
    std::ofstream fout(buf);
    grid_out.write_msh(tria, fout);
  }
}

int main() { create_grid<2>(); }

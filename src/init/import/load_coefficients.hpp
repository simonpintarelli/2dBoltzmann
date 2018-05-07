#pragma once

#include "aux/eigen2hdf.hpp"
#include "grid/grid_tools.hpp"

#include <deal.II/dofs/dof_handler.h>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>


namespace boltzmann {

/**

 * @brief Read distribution function \f$ f(x,v) \f$ from HDF5-file. *
 *
 * @param dst          destination vector *
 * @param fname        h5-Filename, (dset-Name must be `coeffs`)
 * @param dof_handler
 * @param L            # physical DoFs
 * @param N            # velocity DoFs
 */
template <typename VECTOR, typename INDEXER>
void
load_coefficients(VECTOR& dst,
                  const std::string& fname,
                  const dealii::DoFHandler<2>& dof_handler,
                  const INDEXER& indexer)
{
  auto index_map = vertex_to_dof_index(dof_handler);

  if (!boost::filesystem::exists(fname)) {
    dealii::deallog << "cannot open file input coeffs file" << fname << std::endl;
    dealii::deallog << "setting dst to zero" << std::endl;
    for (unsigned int i = 0; i < dst.size(); ++i) {
      dst[i] = 0;
    }
  } else {
    hid_t h5_init = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    Eigen::VectorXd buffer;
    eigen2hdf::load(h5_init, "coeffs", buffer);

    unsigned int n = buffer.rows();
    if (n != dst.size()) {
      throw std::runtime_error(std::string("load_coefficients: dimension mismatch ") + "required " +
                               std::to_string(dst.size()) + ", provided " +
                               std::to_string(n));
    }
    unsigned int N = indexer.N();
    for (unsigned int i = 0; i < indexer.L(); ++i) {
      unsigned int ix = index_map[i];
      unsigned int gidx = indexer.to_global(ix, 0);
      for (unsigned int j = 0; j < indexer.N(); ++j) {
        dst[gidx + j] = buffer[i * N + j];
      }
    }

    H5Fclose(h5_init);
  }
}

}  // end namespace boltzmann

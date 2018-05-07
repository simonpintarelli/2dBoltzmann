#include <Eigen/Dense>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/filesystem.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <utility>

#include <deal.II/base/exceptions.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_in.h>

#include "spectral/basis/spectral_basis_factory_ks.hpp"
//#include "init/import/load_coefficients.hpp"
#include "aux/eigen2hdf.hpp"
#include "spectral/basis/indexer.hpp"

#ifndef _SOLUTION_HANDLER_H_
#define _SOLUTION_HANDLER_H_

namespace boltzmann {

// ------------------------------------------------------------------------------------------
class SimpleGridHandler
{
 private:
  typedef SpectralBasisFactoryKS basis_factory_t;

 public:
  typedef typename basis_factory_t::basis_type basis_type;
  typedef dealii::DoFHandler<2> dh_t;
  typedef Indexer<> indexer_t;

 public:
  SimpleGridHandler(const std::string& path,
                    const std::string& grid_fname,
                    const std::string& basis_descriptor_file = "spectral_basis.desc");

  const basis_type& get_spectral_basis() const { return basis; }
  const dh_t& get_dofhandler() const { return dh; }
  const indexer_t& get_indexer() const { return indexer; }

 private:
  dealii::FE_Q<2> fe;
  dealii::Triangulation<2> tria;
  dh_t dh;
  basis_type basis;
  indexer_t indexer;
};

SimpleGridHandler::SimpleGridHandler(const std::string& path,
                                     const std::string& grid_fname,
                                     const std::string& basis_descriptor_fname)
    : fe(1)
    , dh(tria)
{
  typedef boost::filesystem::path bpath;

  auto grid_path = bpath(path) / bpath(grid_fname);
  auto basis_path = bpath(path) / bpath(basis_descriptor_fname);

  if (!boost::filesystem::exists(basis_path)) {
    std::cerr << "File `" << basis_path.c_str() << "` does not exist. Abort!\n";
    exit(-1);
  }
  basis_factory_t::create(basis, basis_path.c_str());

  if (!boost::filesystem::exists(grid_path)) {
    std::cerr << "File `" << grid_path.c_str() << "` not found. Abort!\n";
    exit(-1);
  }

  dealii::GridIn<2> grid_in;
  grid_in.attach_triangulation(tria);
  std::ifstream f(grid_path.c_str());
  grid_in.read_msh(f);
  f.close();
  dh.distribute_dofs(fe);
  indexer = Indexer<>(dh.n_dofs(), basis.n_dofs());
}

// ------------------------------------------------------------------------------------------

void
load_permutation(std::vector<unsigned int>& perm,
                 const std::string& path,
                 const std::string& fname = "vertex2dofidx.dat")
{
  typedef boost::filesystem::path bpath;
  auto descriptor_fname = bpath(path) / bpath(fname);

  std::string f = descriptor_fname.c_str();

  if (!boost::filesystem::exists(descriptor_fname))
    throw std::runtime_error("File `" + f + "` does not exist.");

  std::ifstream ifile;
  ifile.open(descriptor_fname.c_str());

  while (!ifile.eof()) {
    int p, i;
    ifile >> i;
    ifile >> p;
    perm.at(i) = p;
  }
}

std::string
load_solution_vector(Eigen::VectorXd& dst, const std::string& path, const std::string& fname_h5loc)
{
  typedef std::vector<std::string> split_vector_t;
  split_vector_t split_vector;
  boost::algorithm::split(split_vector,
                          fname_h5loc,
                          boost::algorithm::is_any_of(":"),
                          boost::algorithm::token_compress_on);

  std::string fname = split_vector[0];
  std::string h5loc = "";
  if (split_vector.size() > 1) h5loc = split_vector[1];

  typedef boost::filesystem::path bpath;
  auto solution_path = bpath(path) / bpath(fname);
  hid_t h5_init = H5Fopen(solution_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (h5_init < 0) {
    throw std::runtime_error("Error opening HDF: " + std::string(solution_path.c_str()));
  }

  eigen2hdf::load(h5_init, h5loc, dst);
  H5Fclose(h5_init);

  return std::string(solution_path.c_str());
}

// void invert_permutation_blocked(Eigen::VectorXd& src, std::vector<unsigned int> perm)
// {
//   unsigned int block_size = src.size() / perm.size();

//   if(src.size() % block_size)
//     throw std::runtime_error("In `invert_permutation_blocked`, perm is obviously incompatible
//     with src!");

//   unsigned int L = src.size() / block_size;

//   typedef Eigen::VectorXd vec_t;
//   vec_t tmp(src.size());
//   for (unsigned int i = 0; i < L; ++i) {
//     Eigen::Map<vec_t> vdst(tmp.data() + perm[i]*n_velo_dofs, n_velo_dofs);
//     Eigen::Map<const vec_t> vsrc(src.data() + i*n_velo_dofs, n_velo_dofs);
//     vdst = vsrc;
//   }

//   src = std::move(tmp);
// }

void
to_vertex_ordering(Eigen::VectorXd& src, std::vector<unsigned int> perm)
{
  unsigned int block_size = src.size() / perm.size();

  if (src.size() % block_size)
    throw std::runtime_error(
        "In `invert_permutation_blocked`, perm is obviously incompatible with src!");

  unsigned int L = src.size() / block_size;

  typedef Eigen::VectorXd vec_t;
  vec_t tmp(src.size());
  for (unsigned int i = 0; i < L; ++i) {
    Eigen::Map<vec_t> vdst(tmp.data() + i * block_size, block_size);
    Eigen::Map<const vec_t> vsrc(src.data() + perm[i] * block_size, block_size);
    vdst = vsrc;
  }

  src = std::move(tmp);
}

void
to_dof_ordering(Eigen::VectorXd& src, std::vector<unsigned int> perm)
{
  unsigned int block_size = src.size() / perm.size();

  if (src.size() % block_size)
    throw std::runtime_error(
        "In `invert_permutation_blocked`, perm is obviously incompatible with src!");

  unsigned int L = src.size() / block_size;

  typedef Eigen::VectorXd vec_t;
  vec_t tmp(src.size());
  for (unsigned int i = 0; i < L; ++i) {
    Eigen::Map<vec_t> vdst(tmp.data() + perm[i] * block_size, block_size);
    Eigen::Map<const vec_t> vsrc(src.data() + i * block_size, block_size);
    vdst = vsrc;
  }

  src = std::move(tmp);
}

// // ------------------------------------------------------------------------------------------
// template<typename GH>
// class Solution
// {
// private:
//   typedef std::shared_ptr<GH> gh_ptr_t;
//   typedef Eigen::VectorXd vec_t;
// public:
//   Solution(gh_ptr_t gh_ptr_,
//            const std::string& path,
//            const std::string& fname_h5loc);

//   void apply_permutation(const std::vector<unsigned int>& perm);

//   const GH& get_grid()        const { return *gh_ptr; }
//   const vec_t& get_solution() const { return data; }

// private:
//   gh_ptr_t gh_ptr;
//   vec_t data;
//   unsigned int n_velo_dofs;
// };

// // ------------------------------------------------------------------------------------------
// template<typename GH>
// Solution<GH>::Solution(gh_ptr_t gh_ptr_,
//                        const std::string& path,
//                        const std::string& fname_h5loc)
//   : gh_ptr(gh_ptr_),
//     n_velo_dofs(gh_ptr_->get_spectral_basis().n_dofs())
// {

//   typedef std::vector<std::string> split_vector_t;
//   split_vector_t split_vector;
//   boost::algorithm::split(split_vector, fname_h5loc, boost::algorithm::is_any_of(":"),
//   boost::algorithm::token_compress_on);

//   std::string fname = split_vector[0];
//   std::string h5loc = "";
//   if (split_vector.size() > 1)
//     h5loc = split_vector[1];

//   typedef boost::filesystem::path bpath;
//   auto solution_path = bpath(path) / bpath(fname);
//   hid_t h5_init = H5Fopen(solution_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
//   eigen2hdf::load(h5_init, h5loc, data);
//   H5Fclose(h5_init);
//   // data.resize(gh_ptr->get_dofhandler().n_dofs() * gh_ptr->get_spectral_basis().n_dofs());
//   // load_coefficients(data, solution_path.c_str(), gh_ptr->get_dofhandler(),
//   gh_ptr->get_indexer());
// }

// // ------------------------------------------------------------------------------------------
// template<typename GH>
// void Solution<GH>::
// apply_permutation(const std::vector<unsigned int>& perm)
// {
//   unsigned int L = data.size() / n_velo_dofs;
//   AssertThrow(perm.size() == L, dealii::ExcDimensionMismatch(perm.size(), L));

//   vec_t tmp(data.size());
//   for (unsigned int i = 0; i < L; ++i) {
//     Eigen::Map<vec_t> vdst(tmp.data() + perm[i]*n_velo_dofs, n_velo_dofs);
//     Eigen::Map<const vec_t> vsrc(data.data() + i*n_velo_dofs, n_velo_dofs);
//     vdst = vsrc;
//   }

//   data = std::move(tmp);
// }

}  // end namespace boltzmann

#endif /* _SOLUTION_HANDLER_H_ */

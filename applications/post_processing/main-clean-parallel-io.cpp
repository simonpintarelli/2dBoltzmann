#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>

#include <Eigen/Dense>

#include <H5Cpp.h>
#include <algorithm>
#include <array>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <iostream>
#include <string>
#include <unordered_map>

#include <cstdlib>
#include <fstream>
#include <regex>

#include "aux/hash_specializations.hpp"

namespace bf = boost::filesystem;
namespace po = boost::program_options;

const int DIM = 2;
/// solution vector basename
const std::string sname = "solution-";
const std::string MESH_NAME = "mesh.hdf5";

typedef unsigned int index_t;
typedef std::array<double, DIM> vnode_t;
typedef std::array<index_t, DIM> vcell_t;

/**
 * @brief Find hdf5-files called solution-vector%05d.h5 in directory \p dir.
 *
 * @param dir Data directory
 *
 * @return std::vector<string> \p filenames with relative paths
 */
std::vector<std::string> find_solution_vectors(const std::string& dir)
{
  std::vector<std::string> filenames;
  bf::path p(dir);
  try {
    if (bf::exists(p)) {           // does p actually exist?
      if (bf::is_regular_file(p))  // is p a regular file?
        throw std::runtime_error("not a directory!");
      else if (bf::is_directory(p)) {  // is p a directory?
        // cout << p << " is a directory containing:\n";
        // iterate over files
        for (auto it = bf::directory_iterator(p); it != bf::directory_iterator(); ++it) {
          std::string fname = it->path().filename().c_str();
          if (std::regex_match(fname, std::regex(sname + ".*"))) {
            filenames.emplace_back(it->path().c_str());
          }
        }
      } else
        std::cout << p << " exists, but is neither a regular file nor a directory\n";
    } else
      std::cout << p << " does not exist\n";

  } catch (const bf::filesystem_error& ex) {
    std::cout << ex.what() << '\n';
  }

  std::sort(filenames.begin(), filenames.end());
  return filenames;
}

// ------------------------------------------------------------------------------------------
/// \cond HIDDEN_SYMBOLS
template <int DIM>
class SimpleQuadMesh
{
 public:
  typedef std::array<double, DIM> vnode_t;
  typedef std::array<unsigned int, (1 << DIM)> vcell_t;

 public:
  SimpleQuadMesh() {}

  void load_hdf5(const std::string& fname);
  void load_tria(const dealii::Triangulation<DIM>& tria);

  void write_hdf5(const std::string& fname) const;

  const std::vector<vnode_t> get_nodes() const { return nodes_; }
  const std::vector<vcell_t> get_cells() const { return cells_; }

 private:
  std::vector<vnode_t> nodes_;
  std::vector<vcell_t> cells_;
};

template <int DIM>
void SimpleQuadMesh<DIM>::load_hdf5(const std::string& fname)
{
  H5::H5File mesh_h5(fname, H5F_ACC_RDONLY);
  int num_objs = mesh_h5.getNumObjs();

  auto dset_nodes = mesh_h5.openDataSet("nodes");
  auto dset_cells = mesh_h5.openDataSet("cells");
  auto nodes_space = dset_nodes.getSpace();
  auto cell_space = dset_cells.getSpace();

  hsize_t dims[2];

  // load vertices
  nodes_space.getSimpleExtentDims(dims, NULL);
  nodes_.resize(dims[0]);
  H5::DataSpace memspace(2, dims);
  nodes_.resize(dims[0]);
  dset_nodes.read((void*)nodes_.data(), H5::PredType::NATIVE_DOUBLE, memspace, nodes_space);

  // load cells
  cell_space.getSimpleExtentDims(dims, NULL);
  cells_.resize(dims[0]);
  memspace = H5::DataSpace(2, dims);
  cells_.resize(dims[0]);
  dset_cells.read((void*)cells_.data(), H5::PredType::NATIVE_UINT, memspace, cell_space);
}

template <int DIM>
void SimpleQuadMesh<DIM>::load_tria(const dealii::Triangulation<DIM>& tria)
{
  cells_ = std::vector<vcell_t>();
  const unsigned int vertices_per_cell = dealii::GeometryInfo<DIM>::vertices_per_cell;
  // prepare data
  std::vector<vcell_t> cells;
  for (auto cell : tria.active_cell_iterators()) {
    vcell_t cell_loc;
    cell_loc[0] = cell->vertex_index(0);
    cell_loc[1] = cell->vertex_index(1);
    cell_loc[2] = cell->vertex_index(3);
    cell_loc[3] = cell->vertex_index(2);

    cells_.emplace_back(cell_loc);
  }

  auto vertices = tria.get_vertices();
  auto used_vertices = tria.get_used_vertices();
  for (bool i : used_vertices)
    if (i == 0) throw std::runtime_error("");
  nodes_ = std::vector<vnode_t>(vertices.size());
  for (unsigned int i = 0; i < vertices.size(); ++i) {
    for (unsigned int k = 0; k < DIM; ++k) {
      nodes_[i][k] = vertices[i][k];
    }
  }
}

template <int DIM>
void SimpleQuadMesh<DIM>::write_hdf5(const std::string& dir) const
{
  auto path = bf::path(dir) / bf::path("clean-mesh.hdf5");
  H5::H5File h5_mesh(path.c_str(), H5F_ACC_TRUNC);

  unsigned int ncells = cells_.size();
  unsigned int nvertices = nodes_.size();

  // ------------------------------------------------------------
  // write nodes
  H5::DSetCreatPropList plist;
  hsize_t dims[] = {nvertices, DIM};
  H5::DataSpace dspace_nodes(2, dims);
  H5::DataSet dset_nodes =
      h5_mesh.createDataSet("nodes", H5::PredType::NATIVE_DOUBLE, dspace_nodes, plist);
  dset_nodes.write(nodes_.data(), H5::PredType::NATIVE_DOUBLE);

  // ------------------------------------------------------------
  // write cells
  unsigned int vertices_per_cell = dealii::GeometryInfo<DIM>::vertices_per_cell;
  hsize_t dims_cells[] = {cells_.size(), vertices_per_cell};
  H5::DataSpace dspace_cells(2, dims_cells);
  auto dset_cells = h5_mesh.createDataSet("cells", H5::PredType::NATIVE_UINT, dspace_cells, plist);

  dset_cells.write(cells_.data(), H5::PredType::NATIVE_UINT);
}

// ------------------------------------------------------------------------------------------
/**
 * @brief build permuation vpos1[i] = vpos2[\p perm[i]]
 *
 * @param vpos1  vertex pos. (unique)
 * @param vpos2  vertex pos. (probably not unique)
 *
 * @return perm
 */
template <size_t DIM>
std::vector<index_t> permutation_vector(
    const std::vector<std::array<double, DIM> >& vertices_target,
    const std::vector<std::array<double, DIM> >& vertices_source)
{
  typedef std::vector<std::array<double, DIM> > dvec_t;

  typedef long long int lint;
  constexpr const double FUZZY = 1e8;

  typedef std::array<lint, DIM> vint_t;
  // std::hash specialization is missing for array.
  //  typedef std::unordered_map<vint_t, index_t> map_t;
  typedef std::map<vint_t, index_t> map_t;

  auto build_map = [&](map_t& map, const dvec_t& dvec) {
    for (unsigned int i = 0; i < dvec.size(); ++i) {
      vint_t x;
      for (unsigned int k = 0; k < DIM; ++k) x[k] = dvec[i][k] * FUZZY;
      map[x] = i;
    }
  };

  // vertex pos -> vertex index (in vertices_source)
  map_t mv2;
  build_map(mv2, vertices_source);
  std::vector<index_t> out(mv2.size());
  for (unsigned int i = 0; i < vertices_target.size(); ++i) {
    vint_t x;
    for (unsigned int k = 0; k < DIM; ++k) x[k] = vertices_target[i][k] * FUZZY;
    auto it = mv2.find(x);
    if (it == mv2.end()) throw std::runtime_error("In `permuation_vector`: vertex not found!");
    out[i] = it->second;
  }

  return out;
}

/**
 * @brief Permute solution vectors s.t. they match vertex enumeration from
 *        \p triangulation.
 *
 * @param dirname         /path/to/data/directory
 * @param target_mesh     constructed from dealii::Triangulation
 */
void process_dir(const std::string& dirname, const SimpleQuadMesh<DIM>& target_mesh)
{
  // read mesh
  SimpleQuadMesh<DIM> source_mesh;
  auto fname = bf::path(dirname) / bf::path(MESH_NAME);
  if (!bf::exists(fname)) throw std::runtime_error("Mesh not found in `" + dirname + "``");
  source_mesh.load_hdf5(fname.c_str());

  // build permuation vector
  auto perm = permutation_vector(target_mesh.get_nodes(), source_mesh.get_nodes());

  unsigned int source_length = source_mesh.get_nodes().size();
  unsigned int target_length = perm.size();

  // iterate over solution-vectors ...
  auto solution_vectors = find_solution_vectors(dirname);
  for (unsigned int i = 0; i < solution_vectors.size(); ++i) {
    // open source files
    auto source_path = bf::path(solution_vectors[i]);
    H5::H5File h5solution(source_path.c_str(), H5F_ACC_RDONLY);
    hsize_t num_objs = h5solution.getNumObjs();

    // Prepare output file
    std::smatch match;  // std::string match
    std::string outfile;
    std::regex source_regex(sname + "([0-9]*).*");
    std::string tmp = source_path.filename().c_str();
    if (!std::regex_search(tmp, match, source_regex)) {
      throw std::runtime_error("Ooops unrecognized filename pattern");
    } else {
      outfile = "clean-solution-" + match[1].str() + ".h5";
    }
    auto out_path = bf::path(dirname) / bf::path(outfile);
    H5::H5File h5output(out_path.c_str(), H5F_ACC_TRUNC);

    // iterate over datasets in source
    for (hsize_t i = 0; i < num_objs; ++i) {
      auto dset_name = h5solution.getObjnameByIdx(i);
      auto dset = h5solution.openDataSet(dset_name);
      auto dspace = dset.getSpace();
      hsize_t ndim = dspace.getSimpleExtentNdims();
      std::vector<hsize_t> dims(ndim);
      dspace.getSimpleExtentDims(dims.data(), NULL);
      H5::DataSpace memspace(ndim, dims.data());
      if (dims[0] != source_length) throw std::runtime_error("dimension mismatch");

      if (ndim == 2) {
        // allocate memory for input
        std::vector<double> data(dims[0]);
        // read data
        dset.read((void*)data.data(), H5::PredType::NATIVE_DOUBLE, memspace, dspace);
        // apply permuation
        std::vector<double> target_data(target_length);
        for (unsigned int j = 0; j < target_length; ++j) {
          target_data[j] = data[perm[j]];
        }

        // write target_data to hdf5
        hsize_t dims[2];
        dims[0] = target_data.size();
        dims[1] = 1;
        H5::DataSpace outspace(1, dims);
        auto dset_out = h5output.createDataSet(dset_name, H5::PredType::NATIVE_DOUBLE, outspace);
        dset_out.write((void*)target_data.data(), H5::PredType::NATIVE_DOUBLE);
      } else {
        throw std::runtime_error("dset is not a vector");
      }
    }
  }
  target_mesh.write_hdf5(dirname);
}

int main(int argc, char* argv[])
{
  std::vector<std::string> dirs;
  std::string gmsh;
  po::options_description options("options");
  options.add_options()
      ("help", "produce help message")
      ("dir", po::value<std::vector<std::string> >(&dirs), "input directories")
      ("gmsh", po::value<std::string>(&gmsh), "path/to/gmsh/msh");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, options), vm);
  po::notify(vm);
  if (vm.count("help")) {
    std::cout << options << "\n";
    return 0;
  }

  dealii::Triangulation<DIM> triangulation;
  dealii::GridIn<DIM> gridin;
  gridin.attach_triangulation(triangulation);

  std::ifstream f(gmsh);
  gridin.read_msh(f);
  SimpleQuadMesh<DIM> target_mesh;
  target_mesh.load_tria(triangulation);

  // process directories
  // - write cleaned mesh (same enumeration of gmsh & no duplicates)
  // - write solution vectors (remove duplicate entries & gmsh enumeration)
  for (auto dir : dirs) {
    process_dir(dir, target_mesh);
  }

  return 0;
}

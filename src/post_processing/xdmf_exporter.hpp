#pragma once

#include <EpetraExt_HDF5.h>
#include <Epetra_Map.h>
#include <Epetra_MultiVector.h>
#include <map>
#include <memory>
#include <sstream>

#include "export/epetra_helpers.hpp"
#include "export/export_mesh.hpp"
#include "post_processing/macroscopic_quantities.hpp"


namespace boltzmann {
namespace local_ {
std::string
indent(const unsigned int indent_level)
{
  std::string res = "";
  for (unsigned int i = 0; i < indent_level; ++i) res += "  ";
  return res;
}

/**
 * @brief xdmf open tag (header)
 *
 *
 * @return  std::string
 */
std::string
xdmf_open()
{
  std::stringstream ss;
  ss << "<?xml version=\"1.0\" ?>\n";
  ss << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
  ss << "<Xdmf Version=\"2.0\">\n";
  ss << "  <Domain>\n";
  ss << "    <Grid Name=\"CellTime\" GridType=\"Collection\" CollectionType=\"Temporal\">\n";
  return ss.str();
}

/**
 * @brief xdmf close tag (footer)
 *
 *
 * @return std::string
 */
std::string
xdmf_close()
{
  std::stringstream ss;
  ss << "    </Grid>\n";
  ss << "  </Domain>\n";
  ss << "</Xdmf>\n";
  return ss.str();
}
}  // namespace local_

class XDMFH5Exporter
{
 public:
  /**
   //@{   *
   *
   * @param dofhandler DoFHandler
   * @param basis      specral basis
   * @param phys_map   physically relevant phys dofs
   * @param num_buf    write num_buf timesteps together to disk
   * @param n          output frequency
   */
  template <typename DOFHANDLER, typename SPECTRAL_BASIS>
  XDMFH5Exporter(const DOFHANDLER& dofhandler,
                 const SPECTRAL_BASIS& basis,
                 const dealii::IndexSet& phys_dofs,
                 unsigned int num_buf = 10,
                 unsigned int n = 1);

  template <typename VECTOR>
  void operator()(const VECTOR& solution_vector, double time);

  std::string get_xdmf_content(const unsigned int indent_level, double entry_time) const;

  const Epetra_MultiVector& get_vmass() const;
  const Epetra_MultiVector& get_venergy() const;
  const Epetra_MultiVector& get_vvelocity() const;

 private:
  void set_current_solution_file(int n = -1);

 private:
  MQEval mq_coeffs_;
  /// no. phys dofs
  unsigned int L_;
  // no. of locally owned phys dofs
  unsigned int Lloc_;
  /// no. velocity dofs
  unsigned int N_;
  /// output solution every n_ steps
  unsigned int n_;
  /// write to disk after num_buf_ entries
  unsigned int num_buf_;
  ///
  unsigned int num_cells_;
  unsigned int num_nodes_;

  //@{
  /// Epetra objects
  std::shared_ptr<Epetra_MultiVector> mass_;
  std::shared_ptr<Epetra_MultiVector> energy_;
  std::shared_ptr<Epetra_MultiVector> U_;
  std::shared_ptr<Epetra_MultiVector> q_;  // heat flow
  std::shared_ptr<Epetra_MultiVector> M_;  // momentum flow
  std::shared_ptr<Epetra_MultiVector> P_;  // pressure tensor
  std::shared_ptr<Epetra_MultiVector> r_;  // energy flow
  std::shared_ptr<Epetra_BlockMap> scalar_map_;
  std::shared_ptr<Epetra_BlockMap> vector_map_;
  std::shared_ptr<Epetra_BlockMap> tensor_map_;
  Epetra_MpiComm mpi_comm;
  //@}

  //@{
  /// internal counters / variables
  unsigned int buf_counter_;
  unsigned int timestep_ = 0;
  std::string h5_mesh_filename_ = "mesh.hdf5";
  std::string solution_file_tl_ = "solution_vector%06d.hdf5";
  std::string current_solution_file_ = "";
  std::string xdmf_file_ = "solution.xdmf";
  //@}

  //@{
  /// xdmf
  std::stringstream xdmf_fbuf_;  // buffer for xdmf file
  std::map<std::string, unsigned int> xdmf_entries_;
  //@}
};

template <typename DOFHANDLER, typename SPECTRAL_BASIS>
XDMFH5Exporter::XDMFH5Exporter(const DOFHANDLER& dofhandler,
                               const SPECTRAL_BASIS& basis,
                               const dealii::IndexSet& phys_dofs,
                               unsigned int num_buf,
                               unsigned int n)
    : L_(dofhandler.n_dofs())
    , Lloc_(phys_dofs.n_elements())
    , N_(basis.n_dofs())
    , n_(n)
    , num_buf_(num_buf)
    , mpi_comm(MPI_COMM_WORLD)
    , buf_counter_(0)
{
  std::vector<dealii::types::global_dof_index> indices;
  phys_dofs.fill_index_vector(indices);
  int index_base = 0;
  assert(num_buf > 0);

  this->set_current_solution_file(0);

  int L = dofhandler.n_dofs();
  scalar_map_ = std::make_shared<Epetra_BlockMap>(
      L, indices.size(), reinterpret_cast<int*>(indices.data()), 1, index_base, mpi_comm);

  vector_map_ = std::make_shared<Epetra_BlockMap>(
      L, indices.size(), reinterpret_cast<int*>(indices.data()), 3, index_base, mpi_comm);

  tensor_map_ = std::make_shared<Epetra_BlockMap>(
      L, indices.size(), reinterpret_cast<int*>(indices.data()), 9, index_base, mpi_comm);

  mass_ = std::make_shared<Epetra_MultiVector>(*scalar_map_, num_buf);
  energy_ = std::make_shared<Epetra_MultiVector>(*scalar_map_, num_buf);
  U_ = std::make_shared<Epetra_MultiVector>(*vector_map_, num_buf);
  q_ = std::make_shared<Epetra_MultiVector>(*vector_map_, num_buf);
  M_ = std::make_shared<Epetra_MultiVector>(*tensor_map_, num_buf);
  P_ = std::make_shared<Epetra_MultiVector>(*tensor_map_, num_buf);
  r_ = std::make_shared<Epetra_MultiVector>(*vector_map_, num_buf);
  mq_coeffs_.init(basis);

  num_cells_ = dofhandler.get_triangulation().n_active_quads();
  num_nodes_ = dofhandler.n_dofs();

  /* ---------- register output quantities ---------- */
  xdmf_entries_["m"] = 0;
  xdmf_entries_["e"] = 0;
  xdmf_entries_["u"] = 1;
  xdmf_entries_["q"] = 1;
  xdmf_entries_["r"] = 1;
  xdmf_entries_["P"] = 2;
  xdmf_entries_["M"] = 2;

  // write mesh to disk
  const unsigned int myrank = dealii::Utilities::MPI::this_mpi_process(mpi_comm.Comm());
  if (myrank == 0) export_mesh(dofhandler, h5_mesh_filename_.c_str());
}

void
XDMFH5Exporter::set_current_solution_file(int n)
{
  int next = (timestep_ / n_ / num_buf_) * n_ * num_buf_ + n_ * num_buf_;
  if (n == 0) next = n;
  char fname[256];
  std::sprintf(fname, solution_file_tl_.c_str(), next);
  current_solution_file_ = fname;
}

//
template <typename VECTOR>
void
XDMFH5Exporter::operator()(const VECTOR& solution_vector, double time)
{
  if (timestep_ % n_ == 0) {
    typedef Eigen::Map<const Eigen::VectorXd> vec_t;

    // compute quantities
    auto mq_evaluator = mq_coeffs_.evaluator();

    typedef Eigen::Map<MQEval::evaluator_t::vector_t> mvector_t;
    int size_vec = mvector_t::SizeAtCompileTime;
    typedef Eigen::Map<MQEval::evaluator_t::tensor_t> mtensor_t;
    int size_ten = mtensor_t::SizeAtCompileTime;

    const double* sol = solution_vector.begin();
    const unsigned int Lloc = solution_vector.local_size() / N_;
    assert(solution_vector.local_size() % N_ == 0);
    for (unsigned int l = 0; l < Lloc; ++l) {
      mq_evaluator(sol + l * N_, N_);
      // access observables & fill vectors
      assert(l < (unsigned int)(mass_->MyLength()));
      (*mass_)[buf_counter_][l] = mq_evaluator.m;
      (*energy_)[buf_counter_][l] = mq_evaluator.e;
      mvector_t((*U_)[buf_counter_] + l * size_vec) = mq_evaluator.v;  // velocity (vector)
      mvector_t((*q_)[buf_counter_] + l * size_vec) = mq_evaluator.q;  // heat flow (vector)
      mvector_t((*r_)[buf_counter_] + l * size_vec) = mq_evaluator.r;  // energy flow (vector)
      mtensor_t((*P_)[buf_counter_] + l * size_ten) = mq_evaluator.P;  // pressure (tensor)
      mtensor_t((*M_)[buf_counter_] + l * size_ten) = mq_evaluator.M;  // momentum flow (tensor)
    }
    xdmf_fbuf_ << this->get_xdmf_content(1, time);

    buf_counter_++;
    if (buf_counter_ == num_buf_) {
      // todo write to disk
      epetra_helpers::HDF5 exporter(current_solution_file_);
      exporter.Write("m", *mass_);
      exporter.Write("e", *energy_);
      exporter.Write("u", *U_);
      exporter.Write("q", *q_);
      exporter.Write("r", *r_);
      exporter.Write("P", *P_);
      exporter.Write("M", *M_);
      // write xdmf file to disk
      const unsigned int pid = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

      if (pid == 0) {
        std::ofstream fout(xdmf_file_);
        fout << xdmf_fbuf_.str();
        fout.close();
      }
      // reset internal variables
      // xdmf_fbuf_.str("");
      buf_counter_ = 0;
      set_current_solution_file();
    }
  }
  timestep_++;
}

std::string
XDMFH5Exporter::get_xdmf_content(const unsigned int indent_level, double entry_time) const
{
  std::stringstream ss;

  using namespace local_;

  const int dimension = 2;

  ss << indent(indent_level + 0) << "<Grid Name=\"mesh\" GridType=\"Uniform\">\n";
  ss << indent(indent_level + 1) << "<Time Value=\"" << entry_time << "\"/>\n";
  ss << indent(indent_level + 1) << "<Geometry GeometryType=\"" << (dimension == 2 ? "XY" : "XYZ")
     << "\">\n";
  ss << indent(indent_level + 2) << "<DataItem Dimensions=\"" << L_ << " " << dimension
     << "\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n";
  ss << indent(indent_level + 3) << h5_mesh_filename_ << ":/nodes\n";
  ss << indent(indent_level + 2) << "</DataItem>\n";
  ss << indent(indent_level + 1) << "</Geometry>\n";
  // If we have cells defined, use a quadrilateral (2D) or hexahedron (3D) topology
  if (num_cells_ > 0) {
    ss << indent(indent_level + 1) << "<Topology TopologyType=\""
       << (dimension == 2 ? "Quadrilateral" : "Hexahedron") << "\" NumberOfElements=\""
       << num_cells_ << "\">\n";
    ss << indent(indent_level + 2) << "<DataItem Dimensions=\"" << num_cells_ << " "
       << (2 << (dimension - 1)) << "\" NumberType=\"UInt\" Format=\"HDF\">\n";
    ss << indent(indent_level + 3) << h5_mesh_filename_ << ":/cells\n";
    ss << indent(indent_level + 2) << "</DataItem>\n";
    ss << indent(indent_level + 1) << "</Topology>\n";
  } else {
    // Otherwise, we assume the points are isolated in space and use a Polyvertex topology
    ss << indent(indent_level + 1) << "<Topology TopologyType=\"Polyvertex\" NumberOfElements=\""
       << num_nodes_ << "\">\n";
    ss << indent(indent_level + 1) << "</Topology>\n";
  }

  // std::string h5_sol_filename = "out.hdf5";

  std::map<int, std::string> dim_value;
  dim_value[0] = "Scalar";
  dim_value[1] = "Vector";
  dim_value[2] = "Tensor";
  std::map<int, int> data_dim;
  data_dim[0] = 1;
  data_dim[1] = 3;
  data_dim[2] = 9;

  for (auto it = xdmf_entries_.begin(); it != xdmf_entries_.end(); ++it) {
    ss << indent(indent_level + 1) << "<Attribute Name=\"" << it->first << "\" AttributeType=\""
       << (dim_value.at(it->second)) << "\" Center=\"Node\">\n";
    // write data entry
    if (num_buf_ == 1) {
      ss << indent(indent_level + 2) << "<DataItem Dimensions=\"" << num_nodes_ << " "
         << (data_dim.at(it->second))
         << "\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n";
      ss << indent(indent_level + 3) << current_solution_file_ << ":/" << it->first << "/Values"
         << "\n";
    } else {
      // TODO write hyperslab entry
      ss << indent(indent_level + 2) << "<DataItem ItemType=\"HyperSlab\" "
         << "Dimensions=\"" << num_buf_ << " " << data_dim.at(it->second) * L_ << "\" "
         << "Type=\"HyperSlab\">\n";
      ss << indent(indent_level + 3) << "<DataItem Dimensions=\"3 2\" Format=\"XML\"> \n";
      ss << indent(indent_level + 4) << buf_counter_ << " " << 0 << "\n"                   // origin
         << indent(indent_level + 4) << 1 << " " << 1 << "\n"                              // stride
         << indent(indent_level + 4) << 1 << " " << data_dim.at(it->second) * L_ << "\n";  // count
      ss << indent(indent_level + 3) << "</DataItem>\n";
      ss << indent(indent_level + 3) << "<DataItem Name=\"" << it->first << "\" Dimensions=\""
         << num_buf_ << " " << data_dim.at(it->second) * L_ << "\" "
         << "Format=\"HDF\" NumberType=\"Float\" Precision=\"8\">\n";
      ss << indent(indent_level + 4) << current_solution_file_ << ":/" << it->first << "/Values"
         << "\n";
      ss << indent(indent_level + 3) << "</DataItem>\n";
    }
    ss << indent(indent_level + 2) << "</DataItem>\n";
    ss << indent(indent_level + 1) << "</Attribute>\n";
  }

  ss << indent(indent_level + 0) << "</Grid>\n";

  return ss.str();
}

const Epetra_MultiVector&
XDMFH5Exporter::get_vmass() const
{
  return *mass_;
}

const Epetra_MultiVector&
XDMFH5Exporter::get_venergy() const
{
  return *energy_;
}

const Epetra_MultiVector&
XDMFH5Exporter::get_vvelocity() const
{
  return *U_;
}

}  // namespace boltzmann

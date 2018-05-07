#include <hdf5.h>
#include <omp.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <Eigen/Sparse>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "bte_config.h"
#include "aux/filtered_range.hpp"
#include "aux/message.hpp"
#include "aux/timer.hpp"
#include "base/numbers.hpp"
#include "collision_tensor/assembly/gain.hpp"
#include "collision_tensor/collision_tensor_factory.hpp"
#include "collision_tensor/collision_tensor_galerkin.hpp"
#include "collision_tensor/time_stepping/rk4.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"
#include "post_processing/energy.hpp"
#include "post_processing/mass.hpp"
#include "post_processing/momentum.hpp"
#include "spectral/basis/toolbox/spectral_basis.hpp"
#include "spectral/utility/utility.hpp"

using namespace std;
using namespace boltzmann;
namespace po = boost::program_options;

typedef SpectralBasisFactoryKS basis_factory_t;

int main(int argc, char* argv[])
{
  boltzmann::Timer<> timer;
  double beta, dt;
  int nsteps;
  string tensor_file;
  bool adaptiveL, verbose;

  po::options_description options("options");
  options.add_options()
      ("help", "produce help message")
      ("beta,b", po::value<double>(&beta)->default_value(2), "beta")
      ("dt,t", po::value<double>(&dt)->default_value(0.001), "delta t")
      ("nsteps,n", po::value<int>(&nsteps)->default_value(200), "#timesteps")
      ("adapt,a", po::value<bool>(&adaptiveL)->default_value(false), "use adaptive L-range")
      ("ct,T", po::value<string>(&tensor_file),
      "path to tensor file \n the files `spectral_basis.desc` and \n `spectral_basis_test.desc` "
      "must reside in the same directory")
      ("init,i", po::value<string>()->required(), "coefficients hdf5 in Polar Laguerre basis")
      ("ifdata", po::value<string>()->default_value("coeffs"), " path to data in HDF5 file `if`")
      ("verbose,v", po::value<bool>(&verbose)->default_value(false), "verbose output");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, options), vm);
  po::notify(vm);

  if (vm.count("help")) {
    cout << options << "\n";
    return 0;
  }

  std::string version_id = GIT_SHA1;
  cout << "VersionID: " << version_id << "@" << GIT_BNAME << std::endl;
  cout << "CMD:\n";
  for (int i = 0; i < argc; ++i) {
    cout << argv[i] << "\t";
  }
  cout << "\n\n\n";

  cout << "Command line parameters::\n"
       << right << setw(15) << "beta" << setw(12) << beta << endl
       << right << setw(15) << "dt" << setw(12) << dt << "\n\n\n";

  typedef typename basis_factory_t::basis_type basis_type;
  typedef boost::filesystem::path path_t;

  // load spectral basis
  basis_type trial_basis;
  path_t tensor_fpath(tensor_file.c_str());
  string trial_basis_fname = (tensor_fpath.parent_path() / path_t("spectral_basis.desc")).string();
  basis_factory_t::create(trial_basis, trial_basis_fname);

  // read tensor from file
  timer.start();
  CollisionTensorGalerkin ct(trial_basis);
  ct.read_hdf5(tensor_file.c_str());
  print_timer(timer.stop(), "read collision tensor from file");
  cout << "------------------------------\n";

  const int N = trial_basis.n_dofs();
  Eigen::VectorXd coeffs(N);
  string fname = vm["init"].as<string>();
  hid_t h5_init = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  eigen2hdf::load(h5_init, vm["ifdata"].as<string>(), coeffs);

  cout << "---------- MOMENTSs ----------\n";
  Mass mass;
  mass.init(trial_basis);
  double m0 = mass.compute(coeffs.data());
  cout << "\t mass = " << m0 << endl;

  Energy energy;
  energy.init(trial_basis);
  energy.compute(coeffs.data());

  Momentum momentum;
  momentum.init(trial_basis);
  momentum.compute(coeffs.data());
  cout << "\t energy = " << energy.compute(coeffs.data()) << endl;

  // ------------------------------------------------------------
  // TIME STEPPING
  typedef Eigen::VectorXd vec_t;
  vec_t out(N);
  hg::RK4<> rk4(N);
  const double tol = 1e-10;
  auto find_relevant_range = [&](const double* solution) {
    const int L = spectral::get_max_l(trial_basis);
    typedef typename basis_type::elem_t elem_t;
    typedef typename basis_factory_t::fa_type angular_elem_t;
    // start from l=Lmax until  coefficients are above threshold
    const int llower_bound = 2;
    int lupper_bound = 2;
    typename elem_t::Acc::template get<angular_elem_t> get_xir;
    for (int l = L; l > llower_bound; --l) {
      std::function<bool(const elem_t&)> pred = [&](const elem_t& e) {
        return (get_xir(e).get_id().l == l);
      };
      // get current l-range of spectral basis
      auto range = filtered_range(trial_basis.begin(), trial_basis.end(), pred);
      bool is_below_tre = true;
      for (auto it = std::get<0>(range); it != std::get<1>(range); ++it) {
        unsigned int idx = trial_basis.get_dof_index(it->get_id());
        if (std::abs(solution[idx]) > tol) {
          is_below_tre = false;
          break;
        }
      }
      if (!is_below_tre) {
        lupper_bound = l;
        break;
      }
    }
    // look for the upper-bound iterator in trial_basis
    std::function<bool(int, const elem_t& e)> comp = [&](int l, const elem_t& e) {
      return l < get_xir(e).get_id().l;
    };
    auto it_max = std::upper_bound(trial_basis.begin(), trial_basis.end(), lupper_bound, comp);
    if (verbose) cout << "Adpative Basis: L_max = " << lupper_bound << endl;
    // returns nmax => relevant range is (0, nmax)
    return (it_max - trial_basis.begin());
  };

  hid_t file, gdata;
  file = H5Fcreate("coefficients.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  gdata = H5Gcreate1(file, "data", 0);
  eigen2hdf::save(gdata, "0", coeffs);
  H5Fflush(file, H5F_SCOPE_GLOBAL);

  double t = 0;
  for (int i = 0; i < nsteps; ++i) {
    auto coeffsn = coeffs;
    timer.start();
    auto f = [&](double* dst, const double* src) {
      if (adaptiveL) {
        int nmax = find_relevant_range(coeffs.data());
        if (verbose) cout << "Adaptive nmax = " << nmax << endl;
        ct.apply_adaptive(dst, src, nmax);
      } else {
        ct.apply(dst, src);
      }
    };

    rk4.apply(coeffsn.data(), coeffs.data(), f, dt);
    ct.project(coeffsn.data(), coeffs.data());
    print_timer(timer.stop(), "RK4");
    t += dt;
    timer.start();
    coeffs = coeffsn;
    double m = mass.compute(coeffs.data());
    if(std::isnan(m)) {
      H5Gclose(gdata);
      H5Fclose(file);
      throw std::runtime_error("blow up...");
    }
    double e = energy.compute(coeffs.data());
    auto mom = momentum.compute(coeffs.data());
    print_timer(timer.stop(), "compute moments");

    cout << "::MOMENTS::\t" << setw(8) << i + 1 << setw(20) << setprecision(10) << t << setw(30)
         << setprecision(20) << scientific << m << setw(30) << setprecision(20) << scientific << e
         << setw(30) << setprecision(20) << scientific << mom[0] << setw(30) << setprecision(20)
         << scientific << mom[1] << endl;

    // save results to HDF5
    eigen2hdf::save(gdata, boost::lexical_cast<string>(i + 1), coeffs);
    H5Fflush(file, H5F_SCOPE_GLOBAL);
  }

  H5Gclose(gdata);
  H5Fclose(file);

  return 0;
}

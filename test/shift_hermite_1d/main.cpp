#include <hdf5.h>
#include <quadmath.h>
#include <iostream>

#include "aux/eigen2hdf.hpp"
#include "spectral/shift_hermite.hpp"

using namespace std;
using namespace boltzmann;

typedef long double numeric_t;

int main(int argc, char *argv[])
{
  if (argc < 3) {
    cout << "usage: " << argv[0] << " x  N\n";
    return 1;
  }

  double xs = atof(argv[1]);
  int N = atoi(argv[2]);

  hid_t file = H5Fcreate("matrices.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  HShiftMatrix<numeric_t> M(N);

  // initialize shift matrix
  M.setx(xs);
  // write to hdf
  Eigen::MatrixXd tmp = M.get().cast<double>();
  eigen2hdf::save(file, "plus", tmp);

  // initialize shift matrix
  M.setx(-xs);
  tmp = M.get().cast<double>();
  // write to hdf
  eigen2hdf::save(file, "minus", tmp);

  H5Fclose(file);

  return 0;
}

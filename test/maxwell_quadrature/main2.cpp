
// #include <iostream>
// #include <string>
// #include <boost/lexical_cast.hpp>
// #include <fstream>
// #include <iomanip>

// #include "quadrature/qmaxwell.hpp"
// #include "quadrature/qmidpoint.hpp"
// #include "quadrature/tensor_product_quadrature.hpp"
// #include "quadrature/quadrature_handler.hpp"

// #include "spectral/basis/spectral_basis_factory_ks.hpp"
// #include "matrix/assembly/velocity_radial_integrator.hpp"
// #include "matrix/assembly/velocity_var_form.hpp"
// #include "matrix/assembly/weight.hpp"

// #include "spectral/laguerren.hpp"
// #include "spectral/laguerrenw.hpp"

// using namespace std;
// using namespace boltzmann;

// // typedef boltzmann::QuadratureHandler<
// //   boltzmann::TensorProductQuadratureC<boltzmann::QMidpoint, QMaxwell> > quad_type;

// typedef double numeric_t;

// template<typename MAP>
// void print(const MAP& m, std::ofstream& fout, string title)
// {
//   fout << "----- " << title << endl;
//   for (auto it = m.begin(); it != m.end(); ++it) {
//     fout << it->first.first << "\t"
//          << it->first.second
//          << "\t"
//          << setprecision(16) << it->second
//          << endl;
//   }
// }

// template<typename CONT>
// void print_basis(const CONT& cont, std::ofstream& fout)
// {
//   for (int i = 0; i < cont.size(); ++i) {
//     fout  << cont[i].get_id() << endl;
//   }
// }

// // ----------------------------------------------------------------------
// int main(int argc, char *argv[])
// {
//   const double beta = 2;

//   if ( argc < 3) {
//     cerr << "info: " << argv[0] << " K q"
//          << endl
//          << "q: No. quad. points\n";
//     return 1;
//   }
//   int K = atoi(argv[1]);
//   int N = atoi(argv[2]);
//   const int digits = 256;

//   QMaxwell qmaxwell(1, N, digits);

//   std::ofstream fout("quadrule_order" + boost::lexical_cast<string>(N) + "_" +
//   boost::lexical_cast<string>(digits) + ".dat");
//   for (unsigned int i = 0; i < qmaxwell.size(); ++i) {
//     fout << setprecision(30) << qmaxwell.pts(i)
//          << "\t"
//          << setprecision(30) << qmaxwell.wts(i)
//          << endl;
//   }
//   fout.close();

//   typedef boltzmann::SpectralBasisFactoryKS basis_factory_t;
//   typedef typename basis_factory_t::basis_type basis_type;

//   L2Weight weight(beta);

//   basis_type basis;
//   basis_factory_t::create(basis, K, K, beta);

//   typedef typename std::tuple_element<1, typename basis_type::elem_t::container_t>::type rad_t;
//   typedef typename basis_type::DimAcc::template get_vec<rad_t> accessor_t;

//   const auto& radial_basis = accessor_t()(basis);

//   LaguerreN<numeric_t> L(K);
//   std::vector<numeric_t> r2(N);
//   std::transform(qmaxwell.pts().begin(), qmaxwell.pts().end(), r2.begin(), [](double r) {return
//   r*r;} );
//   L.compute(r2);

//   ofstream foutn("errors-normalized.dat");
//   for (auto elem = radial_basis.begin(); elem != radial_basis.end(); ++elem) {
//     unsigned int n = elem->get_degree();
//     unsigned int alpha = elem->get_order();
//     unsigned int k = elem->get_id().k;
//     unsigned int j = elem->get_id().j;
//     const numeric_t* values = L.get(n, alpha);
//     double I = 0;
//     for (unsigned int q = 0; q < qmaxwell.size(); ++q) {
//       const double r2j = std::pow(qmaxwell.pts(q), 4*j+ 2* (k%2) );
//       I += r2j*values[q] * values[q] * qmaxwell.wts(q);
//     }
//     foutn << elem->get_id() << "\t" << setw(30) << setprecision(20) << scientific <<
//     std::abs(I-0.5) << endl;
//   }
//   foutn.close();

//   ofstream foutnw("errors-normalized-weighted.dat");
//   LaguerreNW<numeric_t> LW(K);
//   LW.compute(r2);
//   for (auto elem = radial_basis.begin(); elem != radial_basis.end(); ++elem) {
//     unsigned int n = elem->get_degree();
//     unsigned int alpha = elem->get_order();
//     unsigned int k = elem->get_id().k;
//     unsigned int j = elem->get_id().j;
//     const numeric_t* values = LW.get(n, alpha);
//     double I = 0;
//     for (unsigned int q = 0; q < qmaxwell.size(); ++q) {
//       const double r = qmaxwell.pts(q);
//       const double r2j = std::pow(r, 4*j+ 2* (k%2) );

//       I += r2j*(values[q] * ::math::exp(r*r*0.5))* (values[q] * ::math::exp(r*r*0.5)) *
//       qmaxwell.wts(q);
//     }
//     foutnw << elem->get_id() << "\t" << setw(30) << setprecision(20) << scientific <<
//     std::abs(I-0.5) << endl;
//   }
//   foutnw.close();

//   return 0;
// }

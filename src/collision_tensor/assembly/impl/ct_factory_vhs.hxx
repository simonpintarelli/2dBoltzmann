namespace boltzmann {
namespace collision_tensor_assembly {

/**
 * @brief VHS kernel with \f$ b(\cos \theta) = \frac{1}{2 \pi} \f$
 *
 */
template<typename BASIS_TYPE,
         typename QANGLE,
         typename QRADIAL>
class CollisionOperator<BASIS_TYPE,
                        KERNEL_TYPE::VHS,
                        QANGLE,
                        QRADIAL>
  : public CollisionOperatorBase
{

private:
  typedef GainEvaluator<BASIS_TYPE> gain_t;
  typedef QuadratureHandler< TensorProductQuadratureC< QANGLE, QRADIAL> >
  quad_handler_t;

  typedef typename BASIS_TYPE::elem_t elem_t;
  typedef typename std::tuple_element<0, typename elem_t::container_t>::type angular_elem_t;
  typedef typename std::tuple_element<1, typename elem_t::container_t>::type radial_elem_t;

public:
  CollisionOperator(const BASIS_TYPE& test_basis_,
                    const BASIS_TYPE& trial_basis_,
                    const unsigned int nptsa_,
                    const unsigned int nptsr_,
                    const unsigned int nptsi_,
                    const double lambda_,
                    const double beta_ = 2.0)
    :
    test_basis(test_basis_),
    trial_basis(trial_basis_),
    nptsa(nptsa_),
    nptsr(nptsr_),
    nptsi(nptsi_),
    lambda(lambda_),
    beta(beta_)
  {
    const auto& quad = quad_handler.get_quad(1.0/beta_, nptsa_, nptsr_);
    CollisionOperatorBase::init(test_basis_, trial_basis_, quad, beta_);

    omp_init_lock(&omp_lock);
  }


  ~CollisionOperator()
  {
    omp_destroy_lock(&omp_lock);
  }


  template<typename EXPORTER>
  void compute( EXPORTER& exporter,
                const std::vector<unsigned int>& work);

private:
  template<typename QUAD>
  void innermx(boost::multi_array<double,2>& innerMX,
               const QUAD& quad,
               const unsigned int j);

  template<typename QUAD,
           typename EXPORTER>
  void apply_quad(EXPORTER& exporter,
                  const boost::multi_array<double,2>& innerMX,
                  const QUAD& quad,
                  const unsigned int j);

private:
  const BASIS_TYPE& test_basis;
  const BASIS_TYPE& trial_basis;

  const unsigned int nptsa;
  const unsigned int nptsr;
  const unsigned int nptsi;
  /// kernel has the form \f$ | v- v_*|^\lambda \f$
  const double lambda;
  /// exponetial weight of basis function 1/beta
  const double beta;
  quad_handler_t quad_handler;
  using CollisionOperatorBase::var_form;

  omp_lock_t omp_lock;
};



// ----------------------------------------------------------------------
template<typename BASIS_TYPE,
         typename QANGLE,
         typename QRADIAL>
template<typename EXPORTER>
void
CollisionOperator<BASIS_TYPE,
                  KERNEL_TYPE::VHS,
                  QANGLE,
                  QRADIAL>::
compute( EXPORTER& exporter,
         const std::vector<unsigned int>& work)
{
  Timer<> timer;

  const auto& quad = quad_handler.get_quad(1.0/beta, nptsa, nptsr);
  const unsigned int quad_size = quad.size();
  #pragma omp parallel
  {
    // allocate memory for innerMX
    boost::multi_array<double, 2> innerMX(boost::extents[quad_size][quad_size]);

    #pragma omp for
    for (auto it = work.begin(); it < work.end(); ++it) {
      const unsigned int j = *it;

      timer.start();
      this->innermx(innerMX, quad, j);
      print_timer(timer.stop(), "setting up inner MX");

      apply_quad(exporter, innerMX, quad, j);
    }
  }

  // write mass matrix to export object
  CollisionOperatorBase::export_mass_matrix(exporter, trial_basis.n_dofs());
}



// ----------------------------------------------------------------------
template<typename BASIS_TYPE,
         typename QANGLE,
         typename QRADIAL>
template<typename QUAD>
void
CollisionOperator<BASIS_TYPE,
                  KERNEL_TYPE::VHS,
                  QANGLE,
                  QRADIAL>::
innermx(boost::multi_array<double,2>& innerMX,
        const QUAD& quad,
        const unsigned int j)
{
  const double inner_product_weight = 1.0 - 2.0/beta;
  const double bcostheta = 1.0/(2*numbers::PI);
  const auto& test_elem = test_basis.get_elem(j);
  gain_t gain(test_elem, inner_product_weight, nptsi);

  const auto& id = test_elem.get_id();
  // inner factor
  // gcc 5.1 does not like const ref's inside ...
  int t = boost::fusion::at_key<angular_elem_t>(id).t;
  int l = boost::fusion::at_key<angular_elem_t>(id).l;

  auto prefactor = [&] (const double arg) {
    if (t==TRIG::COS)
      return std::cos(l*arg);
    else if (t==TRIG::SIN)
      return std::sin(l*arg);
    else {
      return std::nan("nan");
    }
  };

  for (unsigned int q1 = 0; q1 < quad.size(); ++q1) {
    // v
    const std::complex<double>& v = quad.ptsC(q1);

    // I^+(v, v_*) is NOT symmetric for VHS!
    for (unsigned int q2 = 0; q2 < quad.size(); ++q2) {
      // v*
      const std::complex<double>& vs = quad.ptsC(q2);
      const double c = 0.5*std::abs(v+vs);
      // |v-v*|/2
      const double d = 0.5*std::abs(v-vs);
      const double alpha = std::arg(v+vs);

      const double res = gain.compute(c, d);
      const double bj =  test_elem.evaluate_weighted(std::arg(v), std::abs(v));
      innerMX[q1][q2] = (bcostheta*prefactor(alpha)*res - bj) * std::pow(2*d, lambda);
    }
  }

  gain.print_info();
}



// ----------------------------------------------------------------------

template<typename BASIS_TYPE,
         typename QANGLE,
         typename QRADIAL>
template<typename QUAD,
         typename EXPORTER>
void
CollisionOperator<BASIS_TYPE,
                  KERNEL_TYPE::VHS,
                  QANGLE,
                  QRADIAL>::
apply_quad(EXPORTER& exporter,
           const boost::multi_array<double,2>& innerMX,
           const QUAD& quad,
           const unsigned int j)
{
  const auto& test_elem = test_basis.get_elem(j);
  const auto& id = test_elem.get_id();
  int l = boost::fusion::at_key<angular_elem_t>(id).l;
  Timer<> timer;

  timer.start();
  const unsigned int n = trial_basis.n_dofs();

  local_::SparseMatrixWrapper storage(n);

  auto check_sparsity = [&](int l1, int l2) {
    if ( std::abs(l1+l2) == l || std::abs(l1-l2) == l || std::abs(l2-l1) == l)
      return true;
    else
      return false;
  };

  const unsigned int Nq = quad.size();
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_t;
  Eigen::Map<const matrix_t> I(innerMX.origin(), Nq, Nq);
  for (unsigned int l1 = 0; l1 < Loffsets.size()-1; ++l1) {
    int k1size = Loffsets[l1+1] - Loffsets[l1];
    Eigen::Map<const matrix_t> B1(basis_functions[Loffsets[l1]].origin(),
                                  k1size, /* rows */
                                  Nq /* cols */);
    for (unsigned int l2 = 0; l2 < Loffsets.size()-1; ++l2) {
      int k2size = Loffsets[l2+1] - Loffsets[l2];
      Eigen::Map<const matrix_t> B2(basis_functions[Loffsets[l2]].origin(),
                                    k2size, /* rows */
                                    Nq /* cols */);
      if (check_sparsity(l1, l2)) {
        Eigen::MatrixXd res(k1size, k2size);
        res.noalias() = B1*I*B2.transpose();
        // insert entries
        for (unsigned int j1 = Loffsets[l1]; j1 < Loffsets[l1+1]; ++j1) {
          for (unsigned int j2 = Loffsets[l2]; j2 < Loffsets[l2+1]; ++j2) {
            int i1 = j1-Loffsets[l1];
            int i2 = j2-Loffsets[l2];
            storage.insert(j1, j2, res(i1, i2));
          }
        }
      }
    }
  }
  print_timer(timer.stop(), "assembly slice " + boost::lexical_cast<std::string>(j));

  timer.start();
  // export computed entries
  omp_set_lock(&omp_lock);
  exporter.write_slice(j, storage.get());
  omp_unset_lock(&omp_lock);
  print_timer(timer.stop(), "export slice");
}

}  // collision_tensor_assembly
}  // boltzmann

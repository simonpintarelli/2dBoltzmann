#pragma once

#include <deal.II/fe/fe_values.h>

namespace boltzmann {

template <int dim, template <int> class QUADRATURE>
class VarForm
{
 public:
  typedef QUADRATURE<dim> quadrature_t;
  typedef QUADRATURE<dim - 1> face_quadrature_t;
  typedef dealii::FEValues<dim> fe_values_t;
  typedef dealii::FEFaceValues<dim> fe_face_values_t;

 public:
  template <typename FE>
  VarForm(const FE& fe, int nquad_points)
      : quad(nquad_points)
      , face_quad(nquad_points)
      , fe_values(fe, quad, update_flags)
      , fe_face_values(fe, face_quad, face_update_flags)
  { /* empty */
  }

  template <typename CELL>
  void init_cell(const CELL& cell)
  {
    fe_values.reinit(cell);
  }

  template <typename CELL>
  void init_face(const CELL& cell, int face_idx)
  {
    fe_face_values.reinit(cell, face_idx);
  }

  fe_face_values_t& get_fe_face_values() { return fe_face_values; }
  fe_values_t& get_fe_values() { return fe_values; }
  const face_quadrature_t& get_face_quadrature() const { return face_quad; }
  const quadrature_t& get_quadrature() const { return quad; }

 protected:
  quadrature_t quad;
  face_quadrature_t face_quad;
  fe_values_t fe_values;
  fe_face_values_t fe_face_values;

 private:
  static const dealii::UpdateFlags update_flags;
  static const dealii::UpdateFlags face_update_flags;
};

template <int dim, template <int> class QUADRATURE>
const dealii::UpdateFlags VarForm<dim, QUADRATURE>::update_flags =
    dealii::update_values | dealii::update_gradients | dealii::update_JxW_values |
    dealii::update_quadrature_points;

template <int dim, template <int> class QUADRATURE>
const dealii::UpdateFlags VarForm<dim, QUADRATURE>::face_update_flags =
    dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points |
    dealii::update_normal_vectors;

}  // end namespace boltzmann

#pragma once

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/numerics/data_out.h>

namespace dealii {
template <int dim>
class FilteredDataOut : public DataOut<dim>
{
 public:
  FilteredDataOut(const unsigned int subdomain_id)
      : subdomain_id(subdomain_id)
  {
  }
  virtual typename DataOut<dim>::cell_iterator first_cell()
  {
    typename DataOut<dim>::active_cell_iterator cell = this->dofs->begin_active();
    while ((cell != this->dofs->end()) && (cell->subdomain_id() != subdomain_id)) ++cell;
    return cell;
  }
  virtual typename DataOut<dim>::cell_iterator next_cell(
      const typename DataOut<dim>::cell_iterator &old_cell)
  {
    if (old_cell != this->dofs->end()) {
      const IteratorFilters::SubdomainEqualTo predicate(subdomain_id);
      return ++(FilteredIterator<typename DataOut<dim>::active_cell_iterator>(predicate, old_cell));
    } else
      return old_cell;
  }

 private:
  const unsigned int subdomain_id;
};
}  // end namesapce dealii

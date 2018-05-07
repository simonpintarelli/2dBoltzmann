#pragma once

namespace boltzmann {

template <int DIM>
class GlobalIndexer
{
};

template <>
class GlobalIndexer<2>
{
 public:
  typedef unsigned int size_type;

  struct idx_type
  {
    idx_type(int i_, int j_)
        : i(i_)
        , j(j_)
    { /* empty */ }

    int i;  // physical domain node index
    int j;  // velocity domain spectral index
  };

 public:
  GlobalIndexer(unsigned int nphys_dofs, unsigned int nvelo_dofs) __attribute__((deprecated))
  : nphys_dofs_(nphys_dofs)
  , nvelo_dofs_(nvelo_dofs)
  , n_dofs_(nphys_dofs * nvelo_dofs){};

  inline size_type to_global(int ix, int j) const { return nvelo_dofs_ * ix + j; }

  // idx_type get_index(unsigned int ig) const;
  size_t n_dofs() const { return n_dofs_; }

  inline idx_type get_index(unsigned int ig) const
  {
    // physical index
    unsigned int ix = ig / nvelo_dofs_;
    // spectral index jx
    int jx = ig % nvelo_dofs_;

    return idx_type(ix, jx);
  }

  inline size_type N() const { return nvelo_dofs_; }
  inline size_type L() const { return nphys_dofs_; }

 private:
  const size_t nphys_dofs_;
  const size_t nvelo_dofs_;
  const size_t n_dofs_;
};


}  // end namespace boltzmann

#include <Eigen/Dense>
#include <boost/mpl/at.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include "bte_config.h"
#include "aux/rdtsc_timer.hpp"
#include "collision_tensor/collision_tensor_galerkin.hpp"
#include "collision_tensor/dense/collision_tensor_zlastAM_eigen.hpp"
#include "collision_tensor/dense/multi_slices_factory.hpp"
#include "collision_tensor/dense/storage/vbcrs_sparsity.hpp"
#include "collision_tensor/dense/cluster_vbcrs_sparsity.hpp"
#include "aux/filtered_range.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"

namespace po = boost::program_options;

#include <yaml-cpp/yaml.h>

using namespace std;
using namespace boltzmann;

typedef ct_dense::CollisionTensorZLastAMEigen ct_dense_t;
typedef SpectralBasisFactoryKS basis_factory_t;
typedef SpectralBasisFactoryKS::basis_type basis_type;


int
main(int argc, char* argv[])
{
  std::string version_id = GIT_SHA1;
  cout << "VersionID: " << version_id << "@" << GIT_BNAME << std::endl;

  int K;
  int min_blk_size;
  if (argc < 3) {
    cerr << "usage: " << argv[0] << " K blksize"
         << "\nK: polynomial degree"
         << "\nblksize: minimum admissible block size"
         << "\n";
    return 1;
  } else {
    K = atoi(argv[1]);
    min_blk_size = atoi(argv[2]);
    cerr << "K: " << K << "\n";
  }

  // create basis
  basis_type basis;
  SpectralBasisFactoryKS::create(basis, K);
  unsigned int N = basis.n_dofs();

  ct_dense::multi_slices_factory::container_t multi_slices;
  ct_dense::multi_slices_factory::create(multi_slices, basis);

  std::vector<ct_dense::VBCRSSparsity<>> vbcrs_sparsity_patterns(2 * K - 1);
  int i = 0;
  cout << setw(10) << "slice id" << setw(15) << "msize" << setw(15) << "dimz"
       << "\n";
  for (auto& mslice : multi_slices) {
    auto& vbcrs = vbcrs_sparsity_patterns[i++];
    vbcrs.init(mslice.second.data(), K);
    unsigned int msize = vbcrs.memsize();
    cout << setw(10) << i << setw(15) << msize << setw(15) << vbcrs.dimz() << "\n";
  }
  cout << "----------------------------------------------------------------------"
       << "\n";

  typedef SpectralBasisFactoryKS::elem_t elem_t;
  typedef typename boost::mpl::at_c<typename elem_t::types_t, 0>::type fa_type;
  typename elem_t::Acc::template get<fa_type> fa_accessor;
  // find elements to cluster (combine)
 struct block_t
  {
    int l;
    enum TRIG t;
    int index_first;
    int index_last;
  };
 auto cmp = [&fa_accessor](const elem_t& e, int l, enum TRIG t) {
   auto id = fa_accessor(e).get_id();
   return (id.l == l && TRIG(id.t) == t);
 };

  std::vector<block_t> blocks;
  int offset = 0;
  for (int l = 0; l < K; ++l) {
    for (auto t : {TRIG::COS, TRIG::SIN}) {
      auto range_z =
          filtered_range(basis.begin(), basis.end(), std::bind(cmp, std::placeholders::_1, l, t));
      std::vector<elem_t> elemsz(std::get<0>(range_z), std::get<1>(range_z));
      if (elemsz.size() == 0) continue;
      int size = elemsz.size();
      block_t block = {l, t, offset, offset + size};
      blocks.push_back(block);
      offset += size;
    }
  }


  struct super_block_t
  {
    int extent = 0;
    int index_first = std::numeric_limits<int>::max();
    int index_last = -1;
    std::vector<block_t> elems;
    void insert(const block_t& block)
    {
      elems.push_back(block);
      extent += block.index_last - block.index_first;
      index_first = std::min(block.index_first, index_first);
      index_last = std::max(block.index_last, index_last);
    }
  };

  std::vector<super_block_t> super_blocks;

  while (!blocks.empty()) {
    auto elem = blocks.back();
    int extent = elem.index_last - elem.index_first;
    auto& last_super_block = super_blocks.back();
    if (!super_blocks.empty() && last_super_block.extent < min_blk_size) {
      last_super_block.insert(elem);
    } else {
      super_block_t sblock;
      sblock.insert(elem);
      super_blocks.push_back(sblock);
    }
    // remove last elem
    blocks.pop_back();
  }

  std::sort(super_blocks.begin(),
            super_blocks.end(),
            [](const super_block_t& a, const super_block_t& b) {
              return a.index_first < b.index_first;
            });
  cout << "found " << super_blocks.size() << " super blocks"
       << "\n";

  for (auto sblock : super_blocks) {
    cout << "sblock.extent: " << sblock.extent << " count: " << sblock.elems.size() << ", "
         << sblock.index_first << " -> " << sblock.index_last << "\n";
  }

  ct_dense::MultiSlice::key_t k(4, TRIG::COS);
  ct_dense::VBCRSSparsity<> vbcrs_blocked;
  vbcrs_blocked.init(multi_slices[k].data(), super_blocks);
  cout << "memreq blocked: "
       << vbcrs_blocked.memsize() << "\n"
       << "nblocks: " << vbcrs_blocked.nblocks() << "\n"
       << "nrows: " << vbcrs_blocked.nblock_rows() << "\n";

  std::ofstream fout_blocked("vbcrs_blocked.dat");
  vbcrs_blocked.save(fout_blocked);
  fout_blocked.close();

  ct_dense::VBCRSSparsity<> vbcrs;
  vbcrs.init(multi_slices[k].data(), K);
  cout << "memreq: "
       << vbcrs.memsize() << "\n"
       << "nblocks: " << vbcrs.nblocks() << "\n"
       << "nrows: " << vbcrs.nblock_rows() << "\n";
  std::ofstream fout("vbcrs.dat");
  vbcrs.save(fout);
  fout.close();

  // ==================================================
  // use `cluster_vbcrs_sparsity`
  std::vector<ct_dense::VBCRSSparsity<>> vb_blocked;
  cluster_vbcrs_sparsity::cluster(vb_blocked, /* dst */
                                  vbcrs_sparsity_patterns,
                                  multi_slices,
                                  basis,
                                  min_blk_size);

  return 0;
}

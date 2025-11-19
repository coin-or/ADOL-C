/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     sparse/sparsedrivers.cpp
 Revision: $Id$
 Contents: "Easy To Use" C++ interfaces of SPARSE package


 Copyright (c) Andrea Walther, Benjamin Letschert, Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#include <adolc/adalloc.h>
#include <adolc/adolcerror.h>
#include <adolc/adtl_indo.h>
#include <adolc/dvlparms.h>
#include <adolc/interfaces.h>
#include <adolc/oplate.h>
#include <adolc/sparse/sparsedrivers.h>
#include <adolc/tape_interface.h>
#include <adolc/valuetape/sparseinfos.h>
#include <adolc/valuetape/valuetape.h>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <math.h>
#include <numeric>

#ifdef SPARSE
namespace ADOLC::Sparse {
namespace detail {

/// Specialization of input consistency check for Tight control-flow mode.
template <>
void checkBVPInput<ControlFlowMode::Tight>(const double *basepoint) {
  if (basepoint == nullptr)
    ADOLCError::fail(ADOLCError::ErrorType::SPARSE_JAC_NO_BP, CURRENT_LOCATION);
}

void extract(size_t wordIdx, size_t stripIdx, size_t currentStripWordIdx,
             BvpData<BitPatternPropagationDirection::Forward> &data,
             std::span<uint *> &compressedRowStorage) {

  size_t numNonzeros = std::accumulate(data.indepWordHasNonzero_.begin(),
                                       data.indepWordHasNonzero_.end(), 0UL);

  if (numNonzeros > 0) {
    uint *row = nullptr;
    if (stripIdx == 0) {
      row = new uint[numNonzeros + 1];
      row[0] = 0;
    } else {
      row = new uint[compressedRowStorage[wordIdx][0] + numNonzeros + 1];
      std::copy(compressedRowStorage[wordIdx],
                compressedRowStorage[wordIdx] +
                    compressedRowStorage[wordIdx][0] + 1,
                row);
    }

    uint idx = row[0] + 1;
    for (size_t i = 0; i < data.indepWordHasNonzero_.size(); i++) {
      if (data.indepWordHasNonzero_[i]) {
        row[idx++] = currentStripWordIdx + i;
        data.indepWordHasNonzero_[i] = 0;
      }
    }
    row[0] = idx - 1;
    delete[] compressedRowStorage[wordIdx];
    compressedRowStorage[wordIdx] = row;
  }
}

void extract(size_t wordIdx, size_t stripIdx,
             BvpData<BitPatternPropagationDirection::Reverse> &data,
             std::span<uint *> &compressedRowStorage) {

  size_t numNonzeros = std::accumulate(data.indepWordHasNonzero_.begin(),
                                       data.indepWordHasNonzero_.end(), 0UL);
  uint *row = new uint[numNonzeros + 1];

  row[0] = numNonzeros;

  uint idx = 1;
  for (size_t i = 0; i < data.indepWordHasNonzero_.size(); ++i) {
    if (data.indepWordHasNonzero_[i]) {
      row[idx++] = i;
      data.indepWordHasNonzero_[i] = 0;
    }
  }
  delete[] compressedRowStorage[wordIdx];
  compressedRowStorage[wordIdx] = row;
}

} // namespace detail

int absnormal_jac_pat(short tag, int depen, int indep, int numsw,
                      const double *basepoint,
                      std::span<uint *> &compressedRowStorage) {
  detail::resetInput(compressedRowStorage);
  return indopro_forward_absnormal(tag, depen, indep, numsw, basepoint,
                                   compressedRowStorage.data());
}

#include <adolc/adtl_indo.h>
SparseJacInfos sJInfos;
namespace adtl_indo {

int ADOLC_get_sparse_jacobian(func_ad<::adtl::adouble> *const fun,
                              func_ad<adtl_indo::adouble> *const fun_indo,
                              int n, int m, int repeat, double *basepoints,
                              int *nnz, unsigned int **rind,
                              unsigned int **cind, double **values) {
  int i;
  unsigned int j;
  int ret_val = -1;
  if (!repeat) {
    // setNumDir(n);
    // setMode(ADTL_INDO);
    {
      adtl_indo::adouble *x, *y;
      x = new adtl_indo::adouble[n];
      y = new adtl_indo::adouble[m];
      for (i = 0; i < n; i++) {
        x[i] = basepoints[i];
        // x[i].setADValue(i,1);
      }
      ret_val = adtl_indo::ADOLC_Init_sparse_pattern(x, n, 0);

      ret_val = (*fun_indo)(n, x, m, y);

      if (ret_val < 0) {
        printf(" ADOL-C error in tapeless sparse_jac() \n");
        return ret_val;
      }

      ret_val = adtl_indo::ADOLC_get_sparse_pattern(y, m, sJInfos.JP_);
      delete[] x;
      delete[] y;
    }
    sJInfos.depen_ = m;
    sJInfos.nnzIn_ = 0;
    for (i = 0; i < m; i++) {
      for (j = 1; j <= sJInfos.JP_[i][0]; j++)
        sJInfos.nnzIn_++;
    }
    *nnz = sJInfos.nnzIn_;
    /* sJInfos.Seed is memory managed by ColPack and will be deleted
     * along with g. We only keep it in sJInfos for the repeat != 0 case */

    sJInfos.g_ =
        std::make_unique<ColPack::BipartiteGraphPartialColoringInterface>(
            SRC_MEM_ADOLC, sJInfos.JP_.data(), m, n);
    sJInfos.jr1d_ = std::make_unique<ColPack::JacobianRecovery1D>();

    sJInfos.g_->GenerateSeedJacobian(&(sJInfos.Seed_), &(sJInfos.seedRows_),
                                     &(sJInfos.seedClms_), "SMALLEST_LAST",
                                     "COLUMN_PARTIAL_DISTANCE_TWO");
    sJInfos.seedRows_ = m;

    sJInfos.B_ = myalloc2(sJInfos.seedRows_, sJInfos.seedClms_);
    sJInfos.y_ = myalloc1(m);

    if (sJInfos.nnzIn_ != *nnz) {
      printf(" ADOL-C error in sparse_jac():"
             " Number of nonzeros not consistent,"
             " repeat call with repeat = 0 \n");
      return -3;
    }
  }
  //  ret_val = fov_forward(tag, depen, indep, sJInfos.seed_clms, basepoint,
  //  sJInfos.Seed, sJInfos.y, sJInfos.B);
  ::adtl::setNumDir(sJInfos.seedClms_);
  // setMode(ADTL_FOV);
  {
    ::adtl::adouble *x, *y;
    x = new ::adtl::adouble[n];
    y = new ::adtl::adouble[m];

    for (i = 0; i < n; i++) {
      x[i] = basepoints[i];
      for (int jj = 0; jj < sJInfos.seedClms_; jj++)
        x[i].setADValue(jj, sJInfos.Seed_[i][jj]);
    }

    ret_val = (*fun)(n, x, m, y);

    for (i = 0; i < m; i++)
      for (int jj = 0; jj < sJInfos.seedClms_; jj++)
        sJInfos.B_[i][jj] = y[i].getADValue(jj);

    delete[] x;
    delete[] y;
  }
  /* recover compressed Jacobian => ColPack library */

  if (*values != nullptr)
    free(*values);
  if (*rind != nullptr)
    free(*rind);
  if (*cind != nullptr)
    free(*cind);

  sJInfos.jr1d_->RecoverD2Cln_CoordinateFormat_unmanaged(
      sJInfos.g_.get(), sJInfos.B_, sJInfos.JP_.data(), rind, cind, values);

  // delete g;
  // delete jr1d;

  return ret_val;
}

/****************************************************************************/
/*                                                               THAT'S ALL
 */
} // namespace adtl_indo
} // namespace ADOLC::Sparse

#endif
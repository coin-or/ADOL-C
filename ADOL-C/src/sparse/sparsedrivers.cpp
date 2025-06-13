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
#include <adolc/dvlparms.h>
#include <adolc/interfaces.h>
#include <adolc/oplate.h>
#include <adolc/sparse/sparsedrivers.h>

#if defined(ADOLC_INTERNAL)
#if HAVE_CONFIG_H
#include "config.h"
#endif
#endif

#if HAVE_LIBCOLPACK
#include <ColPack/ColPackHeaders.h>
#endif

#include <cstring>
#include <math.h>

#if HAVE_LIBCOLPACK
using namespace ColPack;
#endif

/****************************************************************************/
/*******       sparse Jacobains, separate drivers             ***************/
/****************************************************************************/

/*--------------------------------------------------------------------------*/
/*                                                sparsity pattern Jacobian */
/*--------------------------------------------------------------------------*/
/*                                                                         */

int jac_pat(
    short tag,               /* tape identification                       */
    int depen,               /* number of dependent variables             */
    int indep,               /* number of independent variables           */
    const double *basepoint, /* independent variable values               */
    unsigned int **crs,
    /* returned compressed row block-index storage                         */
    int *options
    /* control options
                    options[0] : way of sparsity pattern computation
                               0 - propagation of index domains (default)
                               1 - propagation of bit pattern
                    options[1] : test the computational graph control flow
                               0 - safe mode (default)
                               1 - tight mode
                    options[2] : way of bit pattern propagation
                               0 - automatic detection (default)
                               1 - forward mode
                               2 - reverse mode                            */
) {
  int rc = -1;
  int i, ctrl_options[2];

  if (!crs)
    ADOLCError::fail(ADOLCError::ErrorType::SPARSE_CRS, CURRENT_LOCATION);

  else
    for (i = 0; i < depen; i++)
      crs[i] = NULL;

  if ((options[0] < 0) || (options[0] > 1))
    options[0] = 0; /* default */
  if ((options[1] < 0) || (options[1] > 1))
    options[1] = 0; /* default */
  if ((options[2] < -1) || (options[2] > 2))
    options[2] = 0; /* default */

  if (options[0] == 0) {
    if (options[1] == 1)
      rc = indopro_forward_tight(tag, depen, indep, basepoint, crs);
    else {
      rc = indopro_forward_safe(tag, depen, indep, basepoint, crs);
    }
  } else {
    ctrl_options[0] = options[1];
    ctrl_options[1] = options[2];
    rc =
        bit_vector_propagation(tag, depen, indep, basepoint, crs, ctrl_options);
  }

  return (rc);
}

int absnormal_jac_pat(short tag, /* tape identification                       */
                      int depen, /* number of dependent variables             */
                      int indep, /* number of independent variables           */
                      int numsw, /* number of switches                        */
                      const double *basepoint, /* independent variable values */
                      unsigned int **crs
                      /* returned compressed row block-index storage */
) {

  if (!crs)
    ADOLCError::fail(ADOLCError::ErrorType::SPARSE_CRS, CURRENT_LOCATION);
  else
    for (int i = 0; i < depen + numsw; i++)
      crs[i] = NULL;
  return indopro_forward_absnormal(tag, depen, indep, numsw, basepoint, crs);
}
/*--------------------------------------------------------------------------*/
/*                                                 seed matrix for Jacobian */
/*--------------------------------------------------------------------------*/

void generate_seed_jac(int m, int n, unsigned int **JP, double ***Seed, int *p,
                       int option
                       /* control options
                                       option : way of compression
                                                  0 - column compression
                          (default) 1 - row compression                */
)
#if HAVE_LIBCOLPACK
{
  int dummy;

  BipartiteGraphPartialColoringInterface *g =
      new BipartiteGraphPartialColoringInterface(SRC_MEM_ADOLC, JP, m, n);

  if (option == 1)
    g->GenerateSeedJacobian_unmanaged(Seed, p, &dummy, "SMALLEST_LAST",
                                      "ROW_PARTIAL_DISTANCE_TWO");
  else
    g->GenerateSeedJacobian_unmanaged(Seed, &dummy, p, "SMALLEST_LAST",
                                      "COLUMN_PARTIAL_DISTANCE_TWO");
  delete g;
}
#else
{
  ADOLCError::fail(ADOLCError::ErrorType::NO_COLPACK, CURRENT_LOCATION);
}
#endif

/****************************************************************************/
/*******        sparse Hessians, separate drivers             ***************/
/****************************************************************************/

/*---------------------------------------------------------------------------*/
/*                                                  sparsity pattern Hessian */
/*                                                                           */

int hess_pat(short tag, /* tape identification                        */
             int indep, /* number of independent variables            */
             const double *basepoint, /* independent variable values */
             unsigned int **crs,
             /* returned compressed row block-index storage */
             int option
             /* control option
                option : test the computational graph control flow
                                        0 - safe mode (default)
                                        1 - tight mode
                                        2 - old safe mode
                                        3 - old tight mode */

) {
  int rc = -1;
  int i;

  if (!crs)
    ADOLCError::fail(ADOLCError::ErrorType::SPARSE_CRS, CURRENT_LOCATION);
  else
    for (i = 0; i < indep; i++)
      crs[i] = nullptr;

  if ((option < 0) || (option > 3))
    option = 0; /* default */

  if (option == 3)
    rc = nonl_ind_old_forward_tight(tag, 1, indep, basepoint, crs);
  else if (option == 2)
    rc = nonl_ind_old_forward_safe(tag, 1, indep, basepoint, crs);
  else if (option == 1)
    rc = nonl_ind_forward_tight(tag, 1, indep, basepoint, crs);
  else
    rc = nonl_ind_forward_safe(tag, 1, indep, basepoint, crs);

  return (rc);
}

/*--------------------------------------------------------------------------*/
/*                                                  seed matrix for Hessian */
/*--------------------------------------------------------------------------*/

void generate_seed_hess(int n, unsigned int **HP, double ***Seed, int *p,
                        int option
                        /* control options
                                        option : way of compression
                                                   0 - indirect recovery
                           (default) 1 - direct recovery                */
)
#if HAVE_LIBCOLPACK
{
  int seed_rows;

  GraphColoringInterface *g = new GraphColoringInterface(SRC_MEM_ADOLC, HP, n);

  if (option == 0)
    g->GenerateSeedHessian_unmanaged(Seed, &seed_rows, p, "SMALLEST_LAST",
                                     "ACYCLIC_FOR_INDIRECT_RECOVERY");
  else
    g->GenerateSeedHessian_unmanaged(Seed, &seed_rows, p, "SMALLEST_LAST",
                                     "STAR");
  delete g;
}
#else
{
  ADOLCError::fail(ADOLCError::ErrorType::NO_COLPACK, CURRENT_LOCATION);
}
#endif

static void deepcopy_HP(unsigned int ***HPnew, unsigned int **HP, int indep) {
  int i, j, s;
  *HPnew = (unsigned int **)malloc(indep * sizeof(unsigned int *));
  for (i = 0; i < indep; i++) {
    s = HP[i][0];
    (*HPnew)[i] = (unsigned int *)malloc((s + 1) * (sizeof(unsigned int)));
    for (j = 0; j <= s; j++)
      (*HPnew)[i][j] = HP[i][j];
  }
}

/****************************************************************************/
/*******       sparse Jacobians, complete driver              ***************/
/****************************************************************************/

int sparse_jac(short tag,  /* tape identification                     */
               int depen,  /* number of dependent variables           */
               int indep,  /* number of independent variables         */
               int repeat, /* indicated repeated call with same seed  */
               const double *basepoint, /* independent variable values */
               int *nnz, /* number of nonzeros                      */
               unsigned int **rind, /* row index */
               unsigned int **cind, /* column index */
               double **values, /* non-zero values                         */
               int *options
               /* control options
                               options[0] : way of sparsity pattern computation
                                          0 - propagation of index domains
                  (default) 1 - propagation of bit pattern options[1] : test the
                  computational graph control flow 0 - safe mode (default) 1 -
                  tight mode options[2] : way of bit pattern propagation 0 -
                  automatic detection (default) 1 - forward mode 2 - reverse
                  mode options[3] : way of compression 0 - column compression
                  (default) 1 - row compression                         */
)
#if HAVE_LIBCOLPACK
{
  std::shared_ptr<ValueTape> tape = getTape(tag);
  int i;
  unsigned int j;
  SparseJacInfos sJinfos;
  int ret_val = 0;
  BipartiteGraphPartialColoringInterface *g;
  JacobianRecovery1D *jr1d;
  JacobianRecovery1D jr1d_loc;

  if (repeat == 0) {
    if ((options[0] < 0) || (options[0] > 1))
      options[0] = 0; /* default */
    if ((options[1] < 0) || (options[1] > 1))
      options[1] = 0; /* default */
    if ((options[2] < -1) || (options[2] > 2))
      options[2] = 0; /* default */
    if ((options[3] < 0) || (options[3] > 1))
      options[3] = 0; /* default */

    sJinfos.JP = (unsigned int **)malloc(depen * sizeof(unsigned int *));
    ret_val = jac_pat(tag, depen, indep, basepoint, sJinfos.JP, options);

    if (ret_val < 0) {
      printf(" ADOL-C error in sparse_jac() \n");
      return ret_val;
    }

    sJinfos.depen = depen;
    sJinfos.nnz_in = depen;
    sJinfos.nnz_in = 0;
    for (i = 0; i < depen; i++) {
      for (j = 1; j <= sJinfos.JP[i][0]; j++)
        sJinfos.nnz_in++;
    }

    *nnz = sJinfos.nnz_in;

    if (options[2] == -1) {
      (*rind) = (unsigned int *)calloc(*nnz, sizeof(unsigned int));
      (*cind) = (unsigned int *)calloc(*nnz, sizeof(unsigned int));
      unsigned int index = 0;
      for (i = 0; i < depen; i++)
        for (j = 1; j <= sJinfos.JP[i][0]; j++) {
          (*rind)[index] = i;
          (*cind)[index++] = sJinfos.JP[i][j];
        }
    }

    /* sJinfos.Seed is memory managed by ColPack and will be deleted
     * along with g. We only keep it in sJinfos for the repeat != 0 case */

    g = new BipartiteGraphPartialColoringInterface(SRC_MEM_ADOLC, sJinfos.JP,
                                                   depen, indep);
    jr1d = new JacobianRecovery1D;

    if (options[3] == 1) {
      g->GenerateSeedJacobian(&(sJinfos.Seed), &(sJinfos.seed_rows),
                              &(sJinfos.seed_clms), "SMALLEST_LAST",
                              "ROW_PARTIAL_DISTANCE_TWO");
      sJinfos.seed_clms = indep;
      ret_val = sJinfos.seed_rows;
    } else {
      g->GenerateSeedJacobian(&(sJinfos.Seed), &(sJinfos.seed_rows),
                              &(sJinfos.seed_clms), "SMALLEST_LAST",
                              "COLUMN_PARTIAL_DISTANCE_TWO");
      sJinfos.seed_rows = depen;
      ret_val = sJinfos.seed_clms;
    }

    sJinfos.B = myalloc2(sJinfos.seed_rows, sJinfos.seed_clms);
    sJinfos.y = myalloc1(depen);

    sJinfos.g = (void *)g;
    sJinfos.jr1d = (void *)jr1d;
    setTapeInfoJacSparse(tag, sJinfos);
  } else {
    sJinfos.depen = tape.sJinfos.depen;
    sJinfos.nnz_in = tape.sJinfos.nnz_in;
    sJinfos.JP = tape.sJinfos.JP;
    sJinfos.B = tape.sJinfos.B;
    sJinfos.y = tape.sJinfos.y;
    sJinfos.Seed = tape.sJinfos.Seed;
    sJinfos.seed_rows = tape.sJinfos.seed_rows;
    sJinfos.seed_clms = tape.sJinfos.seed_clms;
    g = (BipartiteGraphPartialColoringInterface *)tape.sJinfos.g;
    jr1d = (JacobianRecovery1D *)tape.sJinfos.jr1d;
  }

  if (sJinfos.nnz_in != *nnz) {
    printf(" ADOL-C error in sparse_jac():"
           " Number of nonzeros not consistent,"
           " repeat call with repeat = 0 \n");
    return -3;
  }

  if (options[2] == -1)
    return ret_val;

  /* compute jacobian times matrix product */

  if (options[3] == 1) {
    ret_val = zos_forward(tag, depen, indep, 1, basepoint, sJinfos.y);
    if (ret_val < 0)
      return ret_val;
    MINDEC(ret_val, fov_reverse(tag, depen, indep, sJinfos.seed_rows,
                                sJinfos.Seed, sJinfos.B));
  } else
    ret_val = fov_forward(tag, depen, indep, sJinfos.seed_clms, basepoint,
                          sJinfos.Seed, sJinfos.y, sJinfos.B);

  /* recover compressed Jacobian => ColPack library */

  if (*values != NULL && *rind != NULL && *cind != NULL) {
    // everything is preallocated, we assume correctly
    // call usermem versions
    if (options[3] == 1)
      jr1d->RecoverD2Row_CoordinateFormat_usermem(g, sJinfos.B, sJinfos.JP,
                                                  rind, cind, values);
    else
      jr1d->RecoverD2Cln_CoordinateFormat_usermem(g, sJinfos.B, sJinfos.JP,
                                                  rind, cind, values);
  } else {
    // at least one of rind cind values is not allocated, deallocate others
    // and call unmanaged versions
    if (*values != NULL)
      free(*values);
    if (*rind != NULL)
      free(*rind);
    if (*cind != NULL)
      free(*cind);
    if (options[3] == 1)
      jr1d->RecoverD2Row_CoordinateFormat_unmanaged(g, sJinfos.B, sJinfos.JP,
                                                    rind, cind, values);
    else
      jr1d->RecoverD2Cln_CoordinateFormat_unmanaged(g, sJinfos.B, sJinfos.JP,
                                                    rind, cind, values);
  }

  return ret_val;
}
#else
{
  ADOLCError::fail(ADOLCError::ErrorType::NO_COLPACK, CURRENT_LOCATION);
  return -1;
}
#endif

/****************************************************************************/
/*******        sparse Hessians, complete driver              ***************/
/****************************************************************************/

int sparse_hess(short tag,  /* tape identification                     */
                int indep,  /* number of independent variables         */
                int repeat, /* indicated repeated call with same seed  */
                const double *basepoint, /* independent variable values */
                int *nnz, /* number of nonzeros                      */
                unsigned int **rind, /* row index */
                unsigned int **cind, /* column index */
                double **values, /* non-zero values                         */
                int *options
                /* control options
                                options[0] :test the computational graph control
                   flow 0 - safe mode (default) 1 - tight mode 2 - old safe mode
                                           3 - old tight mode
                                options[1] : way of recovery
                                           0 - indirect recovery
                                           1 - direct recovery */
)
#if HAVE_LIBCOLPACK
{
  ValueTape &tape = findTape(tag);
  int i, l;
  unsigned int j;
  SparseHessInfos sHinfos;
  double **Seed;
  int dummy;
  double y;
  int ret_val = -1;
  GraphColoringInterface *g;
  HessianRecovery *hr;

  /* Generate sparsity pattern, determine nnz, allocate memory */
  if (repeat <= 0) {
    if ((options[0] < 0) || (options[0] > 3))
      options[0] = 0; /* default */
    if ((options[1] < 0) || (options[1] > 1))
      options[1] = 0; /* default */

    if (repeat == 0) {
      sHinfos.HP = (unsigned int **)malloc(indep * sizeof(unsigned int *));

      /* generate sparsity pattern */
      ret_val = hess_pat(tag, indep, basepoint, sHinfos.HP, options[0]);

      if (ret_val < 0) {
        printf(" ADOL-C error in sparse_hess() \n");
        return ret_val;
      }
    } else {
      if (indep != tape.sHinfos.indep)
        ADOLCError::fail(ADOLCError::ErrorType::SPARSE_HESS_IND,
                         CURRENT_LOCATION);
      deepcopy_HP(&sHinfos.HP, tape.sHinfos.HP, indep);
    }

    sHinfos.indep = indep;
    sHinfos.nnz_in = 0;

    for (i = 0; i < indep; i++) {
      for (j = 1; j <= sHinfos.HP[i][0]; j++)
        if ((int)sHinfos.HP[i][j] >= i)
          sHinfos.nnz_in++;
    }

    *nnz = sHinfos.nnz_in;

    /* compute seed matrix => ColPack library */

    Seed = NULL;

    g = new GraphColoringInterface(SRC_MEM_ADOLC, sHinfos.HP, indep);
    hr = new HessianRecovery;

    if (options[1] == 0)
      g->GenerateSeedHessian(&Seed, &dummy, &sHinfos.p, "SMALLEST_LAST",
                             "ACYCLIC_FOR_INDIRECT_RECOVERY");
    else
      g->GenerateSeedHessian(&Seed, &dummy, &sHinfos.p, "SMALLEST_LAST",
                             "STAR");

    sHinfos.Hcomp = myalloc2(indep, sHinfos.p);
    sHinfos.Xppp = myalloc3(indep, sHinfos.p, 1);

    for (i = 0; i < indep; i++)
      for (l = 0; l < sHinfos.p; l++)
        sHinfos.Xppp[i][l][0] = Seed[i][l];

    /* Seed will be freed by ColPack when g is freed */
    Seed = NULL;

    sHinfos.Yppp = myalloc3(1, sHinfos.p, 1);

    sHinfos.Zppp = myalloc3(sHinfos.p, indep, 2);

    sHinfos.Upp = myalloc2(1, 2);
    sHinfos.Upp[0][0] = 1;
    sHinfos.Upp[0][1] = 0;

    sHinfos.g = (void *)g;
    sHinfos.hr = (void *)hr;

    setTapeInfoHessSparse(tag, sHinfos);

  } else {
    sHinfos.nnz_in = tape.sHinfos.nnz_in;
    sHinfos.HP = tape.sHinfos.HP;
    sHinfos.Hcomp = tape.sHinfos.Hcomp;
    sHinfos.Xppp = tape.sHinfos.Xppp;
    sHinfos.Yppp = tape.sHinfos.Yppp;
    sHinfos.Zppp = tape.sHinfos.Zppp;
    sHinfos.Upp = tape.sHinfos.Upp;
    sHinfos.p = tape.sHinfos.p;
    g = (GraphColoringInterface *)tape.sHinfos.g;
    hr = (HessianRecovery *)tape.sHinfos.hr;
  }

  if (sHinfos.Upp == NULL) {
    printf(" ADOL-C error in sparse_hess():"
           " First call with repeat = 0 \n");
    return -3;
  }

  if (sHinfos.nnz_in != *nnz) {
    printf(" ADOL-C error in sparse_hess():"
           " Number of nonzeros not consistent,"
           " new call with repeat = 0 \n");
    return -3;
  }

  if (repeat == -1)
    return ret_val;

  //     this is the most efficient variant. However, there was somewhere a bug
  //     in hos_ov_reverse
  ret_val = hov_wk_forward(tag, 1, indep, 1, 2, sHinfos.p, basepoint,
                           sHinfos.Xppp, &y, sHinfos.Yppp);
  MINDEC(ret_val, hos_ov_reverse(tag, 1, indep, 1, sHinfos.p, sHinfos.Upp,
                                 sHinfos.Zppp));

  for (i = 0; i < sHinfos.p; ++i)
    for (l = 0; l < indep; ++l)
      sHinfos.Hcomp[l][i] = sHinfos.Zppp[i][l][1];

  if (*values != NULL && *rind != NULL && *cind != NULL) {
    // everything is preallocated, we assume correctly
    // call usermem versions
    if (options[1] == 0)
      hr->IndirectRecover_CoordinateFormat_usermem(g, sHinfos.Hcomp, sHinfos.HP,
                                                   rind, cind, values);
    else
      hr->DirectRecover_CoordinateFormat_usermem(g, sHinfos.Hcomp, sHinfos.HP,
                                                 rind, cind, values);
  } else {
    // at least one of rind cind values is not allocated, deallocate others
    // and call unmanaged versions
    if (*values != NULL)
      free(*values);
    if (*rind != NULL)
      free(*rind);
    if (*cind != NULL)
      free(*cind);
    if (options[1] == 0)
      hr->IndirectRecover_CoordinateFormat_unmanaged(
          g, sHinfos.Hcomp, sHinfos.HP, rind, cind, values);
    else
      hr->DirectRecover_CoordinateFormat_unmanaged(g, sHinfos.Hcomp, sHinfos.HP,
                                                   rind, cind, values);
  }
  return ret_val;
}
#else
{
  ADOLCError::fail(ADOLCError::ErrorType::NO_COLPACK, CURRENT_LOCATION);
  return -1;
}
#endif

/****************************************************************************/
/*******      sparse Hessians, set and get sparsity pattern   ***************/
/****************************************************************************/

void set_HP(short tag, /* tape identification                     */
            int indep, /* number of independent variables         */
            unsigned int **HP)
#ifdef SPARSE
{
  std::shared_ptr<ValueTape> tape = getTape(tag);
  SparseHessInfos sHinfos;

  sHinfos.nnz_in = 0;
  deepcopy_HP(&sHinfos.HP, HP, indep);
  sHinfos.Hcomp = NULL;
  sHinfos.Xppp = NULL;
  sHinfos.Yppp = NULL;
  sHinfos.Zppp = NULL;
  sHinfos.Upp = NULL;
  sHinfos.p = 0;
  sHinfos.g = NULL;
  sHinfos.hr = NULL;
  sHinfos.indep = indep;
  setTapeInfoHessSparse(tag, sHinfos);
}
#else
{
  ADOLCError::fail(ADOLCError::ErrorType::NO_COLPACK, CURRENT_LOCATION);
}
#endif

void get_HP(short tag, /* tape identification                     */
            int indep, /* number of independent variables         */
            unsigned int ***HP)
#ifdef SPARSE
{
  deepcopy_HP(HP, findTape(tag).sHinfos.HP, indep);
}
#else
{
  ADOLCError::fail(ADOLCError::ErrorType::NO_COLPACK, CURRENT_LOCATION);
}
#endif

/*****************************************************************************/
/*                                                    JACOBIAN BLOCK PATTERN */

/* ------------------------------------------------------------------------- */
int bit_vector_propagation(
    short tag,               /* tape identification                */
    int depen,               /* number of dependent variables      */
    int indep,               /* number of independent variables    */
    const double *basepoint, /* independent variable values         */
    unsigned int **crs,
    /* compressed block row storage                                  */
    int *options /* control options                            */
                 /* options[0] : way of bit pattern propagation
                                 0 - automatic detection (default)
                                 1 - forward mode
                                 2 - reverse mode
                    options[1] : test the computational graph control flow
                                 0 - safe variant (default)
                                 1 - tight variant  */
) {

  int rc = 3;
  char forward_mode, tight_mode;
  int i, ii, j, jj, k, k_old, bits_per_long, i_blocks_per_strip,
      d_blocks_per_strip;
  int this_strip_i_bl_idx, next_strip_i_bl_idx, this_strip_d_bl_idx,
      next_strip_d_bl_idx;
  int stripmined_calls, strip_idx;
  int p_stripmine, q_stripmine, p_ind_bl_bp, q_dep_bl_bp, i_bl_idx, d_bl_idx;
  size_t value1, v;
  size_t **seed = NULL, *s, **jac_bit_pat = NULL, *jac;
  unsigned char *indep_blocks_flags = NULL, *i_b_flags;
  double *valuepoint = NULL;

  if (options[1] == 0) {
    if (depen >= indep / 2)
      options[1] = 1; /* forward */
    else
      options[1] = 2; /* reverse */
  }

  if (options[1] == 1)
    forward_mode = 1;
  else
    forward_mode = 0;

  if (options[0] == 1)
    tight_mode = 1;
  else
    tight_mode = 0;

  if (!forward_mode)
    valuepoint = myalloc1(depen);

  /* bit pattern parameters */

  /* number of bits in an size_t variable */
  bits_per_long = 8 * sizeof(size_t);
  /* olvo 20000214 nl: inserted explicit cast to size_t */
  value1 = (size_t)1 << (bits_per_long - 1); /* 10000....0 */

  /* =================================================== forward propagation */
  if (forward_mode) {

    if ((tight_mode) && !basepoint)
      ADOLCError::fail(ADOLCError::ErrorType::SPARSE_JAC_NO_BP,
                       CURRENT_LOCATION);

    /* indep partial derivatives for the whole Jacobian */

    /* number of size_ts to store the whole seed / Jacobian matrice */
    p_ind_bl_bp = indep / bits_per_long + ((indep % bits_per_long) != 0);

    /* number of size_ts to store the seed / Jacobian strips */
    if (p_ind_bl_bp <= PQ_STRIPMINE_MAX) {
      p_stripmine = p_ind_bl_bp;
      stripmined_calls = 1;
    } else {
      p_stripmine = PQ_STRIPMINE_MAX;
      stripmined_calls = p_ind_bl_bp / PQ_STRIPMINE_MAX +
                         ((p_ind_bl_bp % PQ_STRIPMINE_MAX) != 0);
    }

    /* number of independent blocks per seed / Jacobian strip */
    i_blocks_per_strip = p_stripmine * bits_per_long;

    /* allocate memory --------------------------------------------------- */

    if (!(indep_blocks_flags =
              (unsigned char *)calloc(i_blocks_per_strip, sizeof(char))))
      ADOLCError::fail(
          ADOLCError::ErrorType::SPARSE_JAC_MALLOC,
          ADOLCError::FailInfo{.info2 = i_blocks_per_strip * sizeof(char)},
          CURRENT_LOCATION);

    seed = myalloc2_ulong(indep, p_stripmine);
    jac_bit_pat = myalloc2_ulong(depen, p_stripmine);

    /* strip-mining : repeated forward calls -----------------------------
     */

    for (strip_idx = 0; strip_idx < stripmined_calls; strip_idx++) {
      /* build a partition of the seed matrix (indep x indep_blocks) --- */
      /* (indep x i_blocks_per_strip) as a bit pattern                   */
      s = seed[0];
      for (i = 0; i < indep; i++)
        for (ii = 0; ii < p_stripmine; ii++) /* 2 loops if short -> int !!! */
          *s++ = 0;                          /* set old seed matrix to 0 */

      this_strip_i_bl_idx = strip_idx * i_blocks_per_strip;
      next_strip_i_bl_idx = (strip_idx + 1) * i_blocks_per_strip;
      if (next_strip_i_bl_idx > indep)
        next_strip_i_bl_idx = indep;
      v = value1; /* 10000....0 */

      for (i = 0; i < indep; i++)
        if ((this_strip_i_bl_idx <= i) && (i < next_strip_i_bl_idx)) {
          ii = (i - this_strip_i_bl_idx) / bits_per_long;
          seed[i][ii] = v >> ((i - this_strip_i_bl_idx) % bits_per_long);
        }

      /* bit pattern propagation by forward ---------------------------- */

      if (tight_mode) {
        rc = int_forward_tight(tag, depen, indep, p_stripmine, basepoint, seed,
                               valuepoint, jac_bit_pat);
      } else {
        rc =
            int_forward_safe(tag, depen, indep, p_stripmine, seed, jac_bit_pat);
      }

      /* extract  pattern from bit patterns --------------------- */

      for (j = 0; j < depen; j++) {
        ii = -1;
        v = 0;

        jac = jac_bit_pat[j];
        i_b_flags = indep_blocks_flags;
        for (i_bl_idx = 0; i_bl_idx < i_blocks_per_strip; i_bl_idx++) {
          if (!v) {
            v = value1; /* 10000....0 */
            ii++;
          }
          if (v & jac[ii])
            *i_b_flags = 1;
          i_b_flags++;

          v = v >> 1;
        }

        if (strip_idx == 0)
          k_old = 0;
        else
          k_old = crs[j][0];
        k = 0;
        i_b_flags = indep_blocks_flags;
        for (i = 0; i < i_blocks_per_strip; i++)
          k += *i_b_flags++;

        if ((k > 0) || (strip_idx == 0)) {
          if (!(crs[j] = (unsigned int *)realloc(
                    crs[j], (k_old + k + 1) * sizeof(unsigned int))))
            ADOLCError::fail(
                ADOLCError::ErrorType::SPARSE_JAC_MALLOC,
                ADOLCError::FailInfo{.info2 = (k_old + k + 1) *
                                              sizeof(unsigned int)},
                CURRENT_LOCATION);

          if (strip_idx == 0)
            crs[j][0] = 0;
          if (k > 0) {
            k = crs[j][0] + 1;
            i_b_flags = indep_blocks_flags;
            for (i = 0; i < i_blocks_per_strip; i++) {
              if (*i_b_flags) {
                crs[j][k++] = this_strip_i_bl_idx + i;
                *i_b_flags = 0;
              }
              i_b_flags++;
            }
            /* current/total number of non-zero blocks of indep. vars. */
            crs[j][0] = k - 1;
          }
        }
      }
    } /* strip_idx */

  } /* forward */

  /* =================================================== reverse propagation */
  else {
    /* depen weight vectors for the whole Jacobian */

    /* number of size_ts to store the whole seed / Jacobian matrice */
    q_dep_bl_bp = depen / bits_per_long + ((depen % bits_per_long) != 0);

    /* number of size_ts to store the seed / Jacobian strips */
    if (q_dep_bl_bp <= PQ_STRIPMINE_MAX) {
      q_stripmine = q_dep_bl_bp;
      stripmined_calls = 1;
    } else {
      q_stripmine = PQ_STRIPMINE_MAX;
      stripmined_calls = q_dep_bl_bp / PQ_STRIPMINE_MAX +
                         ((q_dep_bl_bp % PQ_STRIPMINE_MAX) != 0);
    }

    /* number of dependent blocks per seed / Jacobian strip */
    d_blocks_per_strip = q_stripmine * bits_per_long;

    /* allocate memory ---------------------------------------------------
     */
    if (!(indep_blocks_flags =
              (unsigned char *)calloc(indep, sizeof(unsigned char))))
      ADOLCError::fail(
          ADOLCError::ErrorType::SPARSE_JAC_MALLOC,
          ADOLCError::FailInfo{.info2 = indep * sizeof(unsigned char)},
          CURRENT_LOCATION);
    seed = myalloc2_ulong(q_stripmine, depen);
    jac_bit_pat = myalloc2_ulong(q_stripmine, indep);

    /* olvo 20000214: call to forward required in tight mode only,
       in safe mode no basepoint available! */
    if (tight_mode) {
      if (!basepoint)
        ADOLCError::fail(ADOLCError::ErrorType::SPARSE_JAC_NO_BP,
                         CURRENT_LOCATION);

      rc = zos_forward(tag, depen, indep, 1, basepoint, valuepoint);
    }

    /* strip-mining : repeated reverse calls -----------------------------
     */

    for (strip_idx = 0; strip_idx < stripmined_calls; strip_idx++) {
      /* build a partition of the seed matrix (depen_blocks x depen)     */
      /* (d_blocks_per_strip x depen) as a bit pattern                   */
      s = seed[0];
      for (jj = 0; jj < q_stripmine; jj++) /* 2 loops if short -> int !!! */
        for (j = 0; j < depen; j++)
          *s++ = 0; /* set old seed matrix to 0 */

      this_strip_d_bl_idx = strip_idx * d_blocks_per_strip;
      next_strip_d_bl_idx = (strip_idx + 1) * d_blocks_per_strip;
      if (next_strip_d_bl_idx > depen)
        next_strip_d_bl_idx = depen;
      v = value1; /* 10000....0 */

      for (j = 0; j < depen; j++)
        if ((this_strip_d_bl_idx <= j) && (j < next_strip_d_bl_idx)) {
          jj = (j - this_strip_d_bl_idx) / bits_per_long;
          seed[jj][j] = v >> ((j - this_strip_d_bl_idx) % bits_per_long);
        }

      /* bit pattern propagation by reverse ---------------------------- */

      if (tight_mode)
        rc = int_reverse_tight(tag, depen, indep, q_stripmine, seed,
                               jac_bit_pat);
      else
        rc =
            int_reverse_safe(tag, depen, indep, q_stripmine, seed, jac_bit_pat);

      /* extract pattern from bit patterns --------------------- */

      jj = -1;
      v = 0;
      for (d_bl_idx = this_strip_d_bl_idx; d_bl_idx < next_strip_d_bl_idx;
           d_bl_idx++) {
        if (!v) {
          v = value1; /* 10000....0 */
          jj++;
        }
        jac = jac_bit_pat[jj];
        for (i = 0; i < indep; i++) {
          if (v & *jac++) {
            indep_blocks_flags[i] = 1;
          }
        }

        v = v >> 1;

        k = 0;
        i_b_flags = indep_blocks_flags;
        for (i = 0; i < indep; i++)
          k += *i_b_flags++;

        if (!(crs[d_bl_idx] =
                  (unsigned int *)malloc((k + 1) * sizeof(unsigned int))))
          ADOLCError::fail(
              ADOLCError::ErrorType::SPARSE_JAC_MALLOC,
              ADOLCError::FailInfo{.info2 = (k + 1) * sizeof(unsigned int)},
              CURRENT_LOCATION)

              crs[d_bl_idx][0] = k; /* number of non-zero indep. blocks */
        k = 1;
        i_b_flags = indep_blocks_flags;
        for (i = 0; i < indep; i++) {
          if (*i_b_flags) {
            crs[d_bl_idx][k++] = i;
            *i_b_flags = 0;
          }
          i_b_flags++;
        }
      }

    } /* strip_idx */

  } /* reverse */

  if (!forward_mode) {
    free((char *)valuepoint);
    valuepoint = NULL;
  }
  free((char *)*seed);
  free((char *)seed);
  seed = NULL;
  free((char *)*jac_bit_pat);
  free((char *)jac_bit_pat);
  jac_bit_pat = NULL;
  free((char *)indep_blocks_flags);
  indep_blocks_flags = NULL;

  return (rc);
}

#include <adolc/adtl_indo.h>

// namespace adtl {

#ifdef SPARSE
SparseJacInfos sJinfos = {NULL, NULL, NULL, NULL, NULL, NULL, 0, 0, 0, 0};
#endif

int ADOLC_get_sparse_jacobian(func_ad<adtl::adouble> *const fun,
                              func_ad<adtl_indo::adouble> *const fun_indo,
                              int n, int m, int repeat, double *basepoints,
                              int *nnz, unsigned int **rind,
                              unsigned int **cind, double **values)
#if HAVE_LIBCOLPACK
{
  int i;
  unsigned int j;
  int ret_val = -1;
  if (!repeat) {
    freeSparseJacInfos(sJinfos.y, sJinfos.B, sJinfos.JP, sJinfos.g,
                       sJinfos.jr1d, sJinfos.seed_rows, sJinfos.seed_clms,
                       sJinfos.depen);
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

      ret_val = adtl_indo::ADOLC_get_sparse_pattern(y, m, sJinfos.JP);
      delete[] x;
      delete[] y;
    }
    sJinfos.depen = m;
    sJinfos.nnz_in = 0;
    for (i = 0; i < m; i++) {
      for (j = 1; j <= sJinfos.JP[i][0]; j++)
        sJinfos.nnz_in++;
    }
    *nnz = sJinfos.nnz_in;
    /* sJinfos.Seed is memory managed by ColPack and will be deleted
     * along with g. We only keep it in sJinfos for the repeat != 0 case */
    BipartiteGraphPartialColoringInterface *g;
    JacobianRecovery1D *jr1d;

    g = new BipartiteGraphPartialColoringInterface(SRC_MEM_ADOLC, sJinfos.JP, m,
                                                   n);
    jr1d = new JacobianRecovery1D;

    g->GenerateSeedJacobian(&(sJinfos.Seed), &(sJinfos.seed_rows),
                            &(sJinfos.seed_clms), "SMALLEST_LAST",
                            "COLUMN_PARTIAL_DISTANCE_TWO");
    sJinfos.seed_rows = m;

    sJinfos.B = myalloc2(sJinfos.seed_rows, sJinfos.seed_clms);
    sJinfos.y = myalloc1(m);

    sJinfos.g = (void *)g;
    sJinfos.jr1d = (void *)jr1d;

    if (sJinfos.nnz_in != *nnz) {
      printf(" ADOL-C error in sparse_jac():"
             " Number of nonzeros not consistent,"
             " repeat call with repeat = 0 \n");
      return -3;
    }
  }
  //  ret_val = fov_forward(tag, depen, indep, sJinfos.seed_clms, basepoint,
  //  sJinfos.Seed, sJinfos.y, sJinfos.B);
  adtl::setNumDir(sJinfos.seed_clms);
  // setMode(ADTL_FOV);
  {
    adtl::adouble *x, *y;
    x = new adtl::adouble[n];
    y = new adtl::adouble[m];

    for (i = 0; i < n; i++) {
      x[i] = basepoints[i];
      for (int jj = 0; jj < sJinfos.seed_clms; jj++)
        x[i].setADValue(jj, sJinfos.Seed[i][jj]);
    }

    ret_val = (*fun)(n, x, m, y);

    for (i = 0; i < m; i++)
      for (int jj = 0; jj < sJinfos.seed_clms; jj++)
        sJinfos.B[i][jj] = y[i].getADValue(jj);

    delete[] x;
    delete[] y;
  }
  /* recover compressed Jacobian => ColPack library */

  if (*values != NULL)
    free(*values);
  if (*rind != NULL)
    free(*rind);
  if (*cind != NULL)
    free(*cind);
  BipartiteGraphPartialColoringInterface *g;
  JacobianRecovery1D *jr1d;
  g = (BipartiteGraphPartialColoringInterface *)sJinfos.g;
  jr1d = (JacobianRecovery1D *)sJinfos.jr1d;
  jr1d->RecoverD2Cln_CoordinateFormat_unmanaged(g, sJinfos.B, sJinfos.JP, rind,
                                                cind, values);

  // delete g;
  // delete jr1d;

  return ret_val;
}
#else
{
  ADOLCError::fail(ADOLCError::ErrorType::NO_COLPACK, CURRENT_LOCATION);
  return -1;
}
#endif

//}

/****************************************************************************/
/*                                                               THAT'S ALL */

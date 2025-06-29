/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     ho_rev.cpp
 Revision: $Id$
 Contents: Contains the routines :
           hos_reverse (higher-order-scalar reverse mode):
              define _HOS_
           hos_ov_reverse (higher-order-scalar reverse mode on vectors):
              define _HOS_OV_
           hov_reverse (higher-order-vector reverse mode):
              define _HOV_

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz,
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel,
               Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

/*****************************************************************************

  There are four basic versions of the procedure `reverse', which
  are optimized for the cases of scalar or vector reverse sweeps
  with first or higher derivatives, respectively. In the calling
  sequence this distinction is apparent from the type of the
  parameters `lagrange' and `results'. The former may be left out
  and the integer parameters `depen', `indep', `degre', and `nrows'
  must be set or default according to the following matrix of
  calling cases.

           no lagrange         double* lagrange     double** lagrange

double*   gradient of scalar   weight vector times    infeasible
results   valued function      Jacobian product       combination

          ( depen = 1 ,         ( depen > 0 ,
            degre = 0 ,           degre = 0 ,              ------
            nrows = 1 )           nrows = 1 )

double**  Jacobian of vector   weight vector times     weight matrix
results   valued function      Taylor-Jacobians        times Jacobian

          ( 0 < depen           ( depen > 0 ,          ( depen > 0 ,
              = nrows ,           degre > 0 ,            degre = 0 ,
            degre = 0 )           nrows = 1 )            nrows > 0 )

double*** full family of         ------------          weight matrix x
results   Taylor-Jacobians       ------------          Taylor Jacobians

*****************************************************************************/

/****************************************************************************/
/*                                                                   MACROS */
#undef _ADOLC_VECTOR_

/*--------------------------------------------------------------------------*/
#ifdef _HOS_
#define GENERATED_FILENAME "hos_reverse"

#define RESULTS(l, indexi, k) results[indexi][k]
#define LAGRANGE(l, indexd, k) lagrange[indexd][k]

#define HOV_INC(T, degree)                                                     \
  {                                                                            \
  }
#define HOS_OV_INC(T, degree)                                                  \
  {                                                                            \
  }

#define GET_TAYL(loc, depth, p)                                                \
  {                                                                            \
    UPDATE_TAYLORREAD(depth)                                                   \
    tape.get_taylors(loc, depth);                                              \
  }

/*--------------------------------------------------------------------------*/
#elif _HOS_OV_
#define GENERATED_FILENAME "hos_ov_reverse"

#define RESULTS(l, indexi, k) results[l][indexi][k]
#define LAGRANGE(l, indexd, k) lagrange[indexd][k]

#define HOV_INC(T, degree) T += degree;
#define HOS_OV_INC(T, degree) T += degree;

#define GET_TAYL(loc, depth, p)                                                \
  {                                                                            \
    UPDATE_TAYLORREAD(depth *p)                                                \
    tape.get_taylors_p(loc, depth, p);                                         \
  }

/*--------------------------------------------------------------------------*/
#elif _HOV_
#define GENERATED_FILENAME "hov_reverse"

#define _ADOLC_VECTOR_

#define RESULTS(l, indexi, k) results[l][indexi][k]
#define LAGRANGE(l, indexd, k) lagrange[l][indexd][k]

#define IF_HOV_
#define ENDIF_HOV_

#define HOV_INC(T, degree) T += degree;
#define HOS_OV_INC(T, degree)

#define GET_TAYL(loc, depth, p)                                                \
  {                                                                            \
    UPDATE_TAYLORREAD(depth)                                                   \
    tape.get_taylors(loc, depth);                                              \
  }

#else
#error Error ! Define [_HOS_ | _HOS_OV_ | _HOV_]
#endif

/*--------------------------------------------------------------------------*/
/*                                                     access to variables  */

#if _HOS_OV_
#define ARES *Ares
#define AARG *Aarg
#define AARG1 *Aarg1
#define AARG2 *Aarg2

#define ARES_INC *Ares++
#define AARG_INC *Aarg++
#define AARG1_INC *Aarg1++
#define AARG2_INC *Aarg2++

#define ARES_INC_O Ares++
#define AARG_INC_O Aarg++
#define AARG1_INC_O Aarg1++
#define AARG2_INC_O Aarg2++

#define ASSIGN_A(a, b) a = b;
#define HOS_OV_ASSIGN_A(a, b) a = b;

#else /* _FOV_, _HOS_, _HOV_ */
#define ARES *Ares
#define AARG *Aarg
#define AARG1 *Aarg1
#define AARG2 *Aarg2

#define ARES_INC *Ares++
#define AARG_INC *Aarg++
#define AARG1_INC *Aarg1++
#define AARG2_INC *Aarg2++

#define ARES_INC_O Ares++
#define AARG_INC_O Aarg++
#define AARG1_INC_O Aarg1++
#define AARG2_INC_O Aarg2++

#define ASSIGN_A(a, b) a = b;
#endif

/*--------------------------------------------------------------------------*/
/*                                                              loop stuff  */

#ifdef _HOV_
#define FOR_0_LE_l_LT_pk1 for (int l = 0; l < pk1; l++)
#define FOR_0_LE_l_LT_pk for (int l = 0; l < k; l++)
#elif _FOV_
#define FOR_0_LE_l_LT_pk1 for (int l = 0; l < p; l++)
#define FOR_0_LE_l_LT_pk for (int l = 0; l < k; l++)
#elif _HOS_
#define FOR_0_LE_l_LT_pk1 for (int l = 0; l < k1; l++)
#define FOR_0_LE_l_LT_pk for (int l = 0; l < k; l++)
#elif _HOS_OV_
#define FOR_0_LE_l_LT_pk1 for (int l = 0; l < pk1; l++)
#define FOR_0_LE_l_LT_pk for (int l = 0; l < p * k; l++)
#else
#define FOR_0_LE_l_LT_pk1
#define FOR_0_LE_l_LT_pk
#endif

/*--------------------------------------------------------------------------*/
/*                                                         VEC_COMPUTED_* */
#ifdef _ADOLC_VECTOR
#define VEC_COMPUTED_INIT computed = 0;
#define VEC_COMPUTED_CHECK                                                     \
  if (computed == 0) {                                                         \
    computed = 1;
#define VEC_COMPUTED_END }
#else
#define VEC_COMPUTED_INIT
#define VEC_COMPUTED_CHECK
#define VEC_COMPUTED_END
#endif

/* END Macros */

/****************************************************************************/
/*                                                       NECESSARY INCLUDES */
#include <adolc/adalloc.h>
#include <adolc/adolcerror.h>
#include <adolc/convolut.h>
#include <adolc/dvlparms.h>
#include <adolc/externfcts.h>
#include <adolc/interfaces.h>
#include <adolc/oplate.h>
#include <adolc/tape_interface.h>
#include <adolc/valuetape/valuetape.h>
#include <math.h>

#if defined(ADOLC_DEBUG)
#include <string.h>
#endif /* ADOLC_DEBUG */

BEGIN_C_DECLS

/*--------------------------------------------------------------------------*/
/*                                                   Local Static Variables */

#ifdef _HOS_
/***************************************************************************/
/* Higher Order Scalar Reverse Pass.                                       */
/***************************************************************************/
int hos_reverse(short tnum,             /* tape id */
                int depen,              /* consistency chk on # of deps */
                int indep,              /* consistency chk on # of indeps */
                int degre,              /* highest derivative degree  */
                const double *lagrange, /* range weight vector       */
                double **results)       /* matrix of coefficient vectors */
{
  double **L = myalloc2(depen, degre + 1);
  for (int i = 0; i < depen; ++i) {
    L[i][0] = lagrange[i];
    for (int j = 1; j <= degre; ++j)
      L[i][j] = 0.0;
  }
  int rc = hos_ti_reverse(tnum, depen, indep, degre, L, results);
  myfree2(L);
  return rc;
}

int hos_ti_reverse(short tnum, /* tape id */
                   int depen,  /* consistency chk on # of deps */
                   int indep,  /* consistency chk on # of indeps */
                   int degre,  /* highest derivative degre  */
                   const double *const *lagrange, /* range weight vectors */
                   double **results) /* matrix of coefficient vectors */

#elif _HOS_OV_

/***************************************************************************/
/* Higher Order Scalar Reverse Pass, Vector Keep.                          */
/***************************************************************************/
int hos_ov_reverse(short tnum, /* tape id */
                   int depen,  /* consistency chk on # of deps */
                   int indep,  /* consistency chk on # of indeps */
                   int degre,  /* highest derivative degre  */
                   int nrows,  /* # of Jacobian rows calculated */
                   const double *const *lagrange, /* range weight vector */
                   double ***results) /* matrix of coefficient vectors */

#elif _HOV_
/***************************************************************************/
/* Higher Order Vector Reverse Pass.                                       */
/***************************************************************************/
int hov_reverse(short tnum, /* tape id */
                int depen,  /* consistency chk on # of deps */
                int indep,  /* consistency chk on # of indeps */
                int degre,  /* highest derivative degre */
                int nrows,  /* # of Jacobian rows calculated */
                const double *const *lagrange, /* domain weight vector */
                double ***results, /* matrix of coefficient vectors */
                short **nonzero)   /* structural sparsity  pattern  */
{
  double ***L = myalloc3(nrows, depen, degre + 1);
  for (int k = 0; k < nrows; ++k)
    for (int i = 0; i < depen; ++i) {
      L[k][i][0] = lagrange[k][i];
      for (int j = 1; j <= degre; ++j)
        L[k][i][j] = 0.0;
    }
  int rc =
      hov_ti_reverse(tnum, depen, indep, degre, nrows, L, results, nonzero);
  myfree3(L);
  return rc;
}

int hov_ti_reverse(
    short tnum,                           /* tape id */
    int depen,                            /* consistency chk on # of deps */
    int indep,                            /* consistency chk on # of indeps */
    int degre,                            /* highest derivative degre */
    int nrows,                            /* # of Jacobian rows calculated */
    const double *const *const *lagrange, /* domain weight vectors */
    double ***results,                    /* matrix of coefficient vectors */
    short **nonzero)                      /* structural sparsity  pattern  */

#endif

{
  ValueTape &tape = findTape(tnum);
  /************************************************************************/
  /*                                                       ALL VARIABLES  */
  unsigned char operation; /* operation code */
  int dc, ret_c = 3;

  locint size = 0;
  locint res = 0;
  locint arg = 0;
  locint arg1 = 0;
  locint arg2 = 0;

  double coval = 0;

  int indexi = 0, indexd = 0;

  /* other necessary variables */
  double *x = nullptr;
  size_t *jj = nullptr;
  int taycheck;
  int numdep, numind;

  /*----------------------------------------------------------------------*/
  /* Taylor stuff */
  double *Tres = nullptr;
  double *Targ = nullptr;
  double *Targ1 = nullptr;
  double *Targ2 = nullptr;
  double *rp_Ttemp = nullptr;
  double *rp_Ttemp2 = nullptr;
  double **rpp_T = nullptr;

  /*----------------------------------------------------------------------*/
  /* Adjoint stuff */
  double *Ares = nullptr;
  double *Aarg = nullptr;
  double *Aarg1 = nullptr;
  double *Aarg2 = nullptr;
  double *rp_Atemp = nullptr;
  double *rp_Atemp2 = nullptr;
  double **rpp_A = nullptr;
  double *AP1 = nullptr;
  double *AP2 = nullptr;

  /*----------------------------------------------------------------------*/
  const int k = degre + 1;
  const int k1 = k + 1;
  revreal comp;

#ifdef _HOV_
  const int p = nrows;
  const int pk1 = p * k1;
  const int q = 1;
#elif _HOS_OV_
  const int p = nrows;
  const int pk1 = p * k1;
  const int q = p;
#else
  const int q = 1;
  const int p = 1;
#endif

  /****************************************************************************/
  /*                                          extern diff. function variables */
#if defined(_HOS_)
#define ADOLC_EXT_FCT_U dpp_U
#define ADOLC_EXT_FCT_Z dpp_Z
#define ADOLC_EXT_FCT_POINTER hos_ti_reverse
#define ADOLC_EXT_FCT_COMPLETE                                                 \
  hos_ti_reverse(edfct->tapeId, m, dpp_U, n, degre, dpp_Z, dpp_x, dpp_y)
#else
#define ADOLC_EXT_FCT_U dppp_U
#define ADOLC_EXT_FCT_Z dppp_Z
#define ADOLC_EXT_FCT_POINTER hov_reverse
#define ADOLC_EXT_FCT_COMPLETE                                                 \
  hov_reverse(edfct->tapeId, m, p, edfct->dpp_U, n, degre, edfct->dppp_Z,      \
              edfct->spp_nz)
#endif

#if defined(ADOLC_DEBUG)
  /************************************************************************/
  /*                                                       DEBUG MESSAGES */
  fprintf(DIAG_OUT, "Call of %s(..) with tag: %d, n: %d, m %d,\n",
          GENERATED_FILENAME, tnum, indep, depen);

  fprintf(DIAG_OUT, "                    degree: %d\n", degre);
#ifdef _ADOLC_VECTOR_
  fprintf(DIAG_OUT, "                    p: %d\n\n", nrows);
#endif

#endif

  /************************************************************************/
  /*                                                                INITs */

  /*----------------------------------------------------------------------*/
  /* Set up stuff for the tape */

  /* Initialize the Reverse Sweep */
  tape.init_rev_sweep(tnum);

  if ((depen != tape.tapestats(TapeInfos::NUM_DEPENDENTS)) ||
      (indep != tape.tapestats(TapeInfos::NUM_INDEPENDENTS)))
    ADOLCError::fail(ADOLCError::ErrorType::REVERSE_COUNTS_MISMATCH,
                     CURRENT_LOCATION,
                     ADOLCError::FailInfo{
                         .info1 = tape.tapeId(),
                         .info3 = depen,
                         .info4 = indep,
                         .info5 = tape.tapestats(TapeInfos::NUM_DEPENDENTS),
                         .info6 = tape.tapestats(TapeInfos::NUM_INDEPENDENTS)});

  indexi = tape.tapestats(TapeInfos::NUM_INDEPENDENTS) - 1;
  indexd = tape.tapestats(TapeInfos::NUM_DEPENDENTS) - 1;

  /************************************************************************/
  /*                                              MEMORY ALLOCATION STUFF */

  /*----------------------------------------------------------------------*/
#ifdef _HOS_ /* HOS */
  rpp_A = myalloc2(tape.tapestats(TapeInfos::NUM_MAX_LIVES), k1);
  rpp_T = myalloc2(tape.tapestats(TapeInfos::NUM_MAX_LIVES), k);
  rp_Atemp = myalloc1(k1);
  rp_Atemp2 = myalloc1(k1);
  rp_Ttemp2 = myalloc1(k);
  tape.workMode(TapeInfos::HOS_REVERSE);

  locint n, m;
  ext_diff_fct *edfct = nullptr;
  int oldTraceFlag;
  /*----------------------------------------------------------------------*/
#elif _HOV_    /* HOV */
  rpp_A = myalloc2(tape.tapestats(TapeInfos::NUM_MAX_LIVES), pk1);
  rpp_T = myalloc2(tape.tapestats(TapeInfos::NUM_MAX_LIVES), k);
  rp_Atemp = myalloc1(pk1);
  rp_Atemp2 = myalloc1(pk1);
  rp_Ttemp2 = myalloc1(k);
  tape.workMode(TapeInfos::HOV_REVERSE);
  /*----------------------------------------------------------------------*/
#elif _HOS_OV_ /* HOS_OV */
  rpp_A = myalloc2(tape.tapestats(TapeInfos::NUM_MAX_LIVES), pk1);
  rpp_T = myalloc2(tape.tapestats(TapeInfos::NUM_MAX_LIVES), p * k);
  rp_Atemp = myalloc1(pk1);
  rp_Atemp2 = myalloc1(pk1);
  rp_Ttemp2 = myalloc1(p * k);
  tape.workMode(TapeInfos::HOV_REVERSE);
#endif
  rp_Ttemp = myalloc1(k);
  x = myalloc1(q);
  jj = myalloc1_ulong(q);

  /************************************************************************/
  /*                                                TAYLOR INITIALIZATION */
  tape.rpp_A(rpp_A);
  tape.rpp_T(rpp_T);
  tape.taylor_back(tnum, &numdep, &numind, &taycheck);

  if (taycheck != degre)
    ADOLCError::fail(ADOLCError::ErrorType::REVERSE_NO_FOWARD, CURRENT_LOCATION,
                     ADOLCError::FailInfo{.info3 = degre, .info4 = degre + 1});

  if ((numdep != depen) || (numind != indep))
    ADOLCError::fail(ADOLCError::ErrorType::REVERSE_TAYLOR_COUNTS_MISMATCH,
                     CURRENT_LOCATION, ADOLCError::FailInfo{.info1 = tnum});

  /************************************************************************/
  /*                                                        REVERSE SWEEP */

#if defined(ADOLC_DEBUG)
  int v = 0;
  unsigned int countPerOperation[256], taylorPerOperation[256];
  memset(countPerOperation, 0, 1024);
  memset(taylorPerOperation, 0, 1024);
#define UPDATE_TAYLORREAD(X) taylorPerOperation[operation] += X;
#else
#define UPDATE_TAYLORREAD(X)
#endif /* ADOLC_DEBUG */

  operation = tape.get_op_r();
#if defined(ADOLC_DEBUG)
  ++countPerOperation[operation];
#endif /* ADOLC_DEBUG */

  while (operation != start_of_tape) {
    /* Switch statement to execute the operations in Reverse */
    switch (operation) {

      /************************************************************/
      /*                                                  MARKERS */

      /*----------------------------------------------------------*/
    case end_of_op: /* end_of_op */
      tape.get_op_block_r();
      operation = tape.get_op_r();
      /* Skip next operation, it's another end_of_op */
      break;

      /*----------------------------------------------------------*/
    case end_of_int:          /* end_of_int */
      tape.get_loc_block_r(); /* Get the next int block */
      break;

      /*----------------------------------------------------------*/
    case end_of_val:          /* end_of_val */
      tape.get_val_block_r(); /* Get the next val block */
      break;

      /*----------------------------------------------------------*/
    case start_of_tape: /* start_of_tape */
      break;
    case end_of_tape: /* end_of_tape */
      tape.discard_params_r();
      break;

      /************************************************************/
      /*                                               COMPARISON */

      /*----------------------------------------------------------*/
    case eq_zero: /* eq_zero */
      arg = tape.get_locint_r();

      ret_c = 0;
      break;

      /*----------------------------------------------------------*/
    case neq_zero: /* neq_zero */
    case gt_zero:  /* gt_zero */
    case lt_zero:  /* lt_zero */
      arg = tape.get_locint_r();
      break;

      /*----------------------------------------------------------*/
    case ge_zero: /* ge_zero */
    case le_zero: /* le_zero */
      arg = tape.get_locint_r();

      if (*rpp_T[arg] == 0)
        ret_c = 0;
      break;

      /************************************************************/
      /*                                              ASSIGNMENTS */

      /*----------------------------------------------------------*/
    case assign_a: /* assign an adouble variable an    assign_a */
      /* adouble value. (=) */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();

      ASSIGN_A(Aarg, rpp_A[arg])
      ASSIGN_A(Ares, rpp_A[res])

      for (int l = 0; l < p; l++)
        if (0 == ARES) {
          HOV_INC(Aarg, k1)
          HOV_INC(Ares, k1)
        } else {
          MAXDEC(AARG, ARES);
          AARG_INC_O;
          ARES_INC = 0.0;
          for (int i = 0; i < k; i++) { /* ! no temporary */
            AARG_INC += ARES;
            ARES_INC = 0.0;
          }
        }

      GET_TAYL(res, k, p)
      break;

      /*----------------------------------------------------------*/
    case assign_d: /* assign an adouble variable a    assign_d */
      /* double value. (=) */
      res = tape.get_locint_r();
      coval = tape.get_val_r();

      ASSIGN_A(Ares, rpp_A[res])

      FOR_0_LE_l_LT_pk1 ARES_INC = 0.0;

      GET_TAYL(res, k, p)
      break;

    case assign_p: /* assign an adouble variable a    assign_d */
      /* double value. (=) */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();
      coval = tape.paramstore()[arg];

      ASSIGN_A(Ares, rpp_A[res])

      FOR_0_LE_l_LT_pk1 ARES_INC = 0.0;

      GET_TAYL(res, k, p)
      break;

      /*----------------------------------------------------------*/
    case assign_d_zero: /* assign an adouble a        assign_d_zero */
    case assign_d_one:  /* double value. (=)           assign_d_one */
      res = tape.get_locint_r();

      ASSIGN_A(Ares, rpp_A[res])

      FOR_0_LE_l_LT_pk1 ARES_INC = 0.0;

      GET_TAYL(res, k, p)
      break;

      /*--------------------------------------------------------------------------*/
    case assign_ind: /* assign an adouble variable an    assign_ind */
      /* independent double value (<<=) */
      res = tape.get_locint_r();

      ASSIGN_A(Ares, rpp_A[res])

      for (int l = 0; l < p; l++) {
#ifdef _HOV_
        if (nonzero) /* ??? question: why here? */
          nonzero[l][indexi] = (int)ARES;
#endif /* _HOV_ */
        ARES_INC_O;
        for (int i = 0; i < k; i++)
          RESULTS(l, indexi, i) = ARES_INC;
      }

      GET_TAYL(res, k, p)
      indexi--;
      break;

      /*--------------------------------------------------------------------------*/
    case assign_dep: /* assign a float variable a    assign_dep */
      /* dependent adouble value. (>>=) */
      res = tape.get_locint_r();

      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg, rpp_A[res]) /* just a helpful pointers */

      for (int l = 0; l < p; l++) {
        ARES_INC_O;
        dc = -1;
        for (int i = 0; i < k; i++) {
          ARES_INC = LAGRANGE(l, indexd, i);
          if (LAGRANGE(l, indexd, i))
            dc = i;
        }
        AARG = (dc < 0) ? 0.0 : (dc > 0) ? 2.0 : 1.0;
        HOV_INC(Aarg, k1)
      }
      indexd--;
      break;

      /****************************************************************************/
      /*                                                   OPERATION +
       * ASSIGNMENT */

      /*--------------------------------------------------------------------------*/
    case eq_plus_d: /* Add a floating point to an    eq_plus_d */
      /* adouble. (+=) */
      res = tape.get_locint_r();
      coval = tape.get_val_r();

      GET_TAYL(res, k, p)
      break;

      /*--------------------------------------------------------------------------*/
    case eq_plus_a: /* Add an adouble to another    eq_plus_a */
      /* adouble. (+=) */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();

      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg, rpp_A[arg]);

      for (int l = 0; l < p; l++)
        if (0 == ARES) {
          HOV_INC(Ares, k1)
          HOV_INC(Aarg, k1)
        } else {
          MAXDEC(AARG, ARES);
          AARG_INC_O;
          ARES_INC_O;
          for (int i = 0; i < k; i++)
            AARG_INC += ARES_INC;
        }

      GET_TAYL(res, k, p)
      break;

      /*--------------------------------------------------------------------------*/
    case eq_min_d: /* Subtract a floating point from an    eq_min_d */
      /* adouble. (-=) */
      res = tape.get_locint_r();
      coval = tape.get_val_r();

      GET_TAYL(res, k, p)
      break;

      /*--------------------------------------------------------------------------*/
    case eq_min_a: /* Subtract an adouble from another    eq_min_a */
      /* adouble. (-=) */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();

      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg, rpp_A[arg])

      for (int l = 0; l < p; l++)
        if (0 == ARES) {
          HOV_INC(Ares, k1)
          HOV_INC(Aarg, k1)
        } else {
          MAXDEC(AARG, ARES);
          AARG_INC_O;
          ARES_INC_O;
          for (int i = 0; i < k; i++)
            AARG_INC -= ARES_INC;
        }

      GET_TAYL(res, k, p)
      break;

      /*--------------------------------------------------------------------------*/
    case eq_mult_d: /* Multiply an adouble by a    eq_mult_d */
      /* floating point. (*=) */
      res = tape.get_locint_r();
      coval = tape.get_val_r();

      ASSIGN_A(Ares, rpp_A[res])

      for (int l = 0; l < p; l++)
        if (0 == ARES_INC)
          HOV_INC(Ares, k) else for (int i = 0; i < k; i++) ARES_INC *= coval;

      GET_TAYL(res, k, p)
      break;

      /*--------------------------------------------------------------------------*/
    case eq_mult_a: /* Multiply one adouble by another    eq_mult_a */
      /* (*=) */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();

      GET_TAYL(res, k, p)

      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg, rpp_A[arg])
      Tres = rpp_T[res];
      Targ = rpp_T[arg];

      for (int l = 0; l < p; l++) {
        if (0 == ARES) {
          HOV_INC(Aarg, k1)
          HOV_INC(Ares, k1)
        } else {
          MAXDEC(ARES, 2.0);
          MAXDEC(AARG, ARES);
          AARG_INC_O;
          ARES_INC_O;
          conv(k, Ares, Targ, rp_Atemp);
          if (arg != res) {
            inconv(k, Ares, Tres, Aarg);
            for (int i = 0; i < k; i++)
              ARES_INC = rp_Atemp[i];
          } else
            for (int i = 0; i < k; i++)
              ARES_INC = 2.0 * rp_Atemp[i];
          HOV_INC(Aarg, k)
          HOS_OV_INC(Tres, k)
          HOS_OV_INC(Targ, k)
        }
      }
      break;

      /*--------------------------------------------------------------------------*/
    case incr_a: /* Increment an adouble    incr_a */
    case decr_a: /* Increment an adouble    decr_a */
      res = tape.get_locint_r();

      GET_TAYL(res, k, p)
      break;

      /****************************************************************************/
      /*                                                        BINARY
       * OPERATIONS */

      /*--------------------------------------------------------------------------*/
    case plus_a_a: /* : Add two adoubles. (+)    plus a_a */
      res = tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();

      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg1, rpp_A[arg1])
      ASSIGN_A(Aarg2, rpp_A[arg2])

      for (int l = 0; l < p; l++)
        if (0 == ARES) {
          HOV_INC(Ares, k1)
          HOV_INC(Aarg1, k1)
          HOV_INC(Aarg2, k1)
        } else {
          double aTmp = ARES;
          ARES_INC = 0.0;
          MAXDEC(AARG1, aTmp);
          MAXDEC(AARG2, aTmp);
          AARG2_INC_O;
          AARG1_INC_O;
          for (int i = 0; i < k; i++) {
            aTmp = ARES;
            ARES_INC = 0.0;
            AARG1_INC += aTmp;
            AARG2_INC += aTmp;
          }
        }

      GET_TAYL(res, k, p)
      break;
      /*--------------------------------------------------------------------------*/
    case plus_d_a: /* Add an adouble and a double    plus_d_a */
      /* (+) */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();
      coval = tape.get_val_r();

      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg, rpp_A[arg])

      for (int l = 0; l < p; l++)
        if (0 == ARES) {
          HOV_INC(Ares, k1)
          HOV_INC(Aarg, k1)
        } else {
          double aTmp = ARES;
          ARES_INC = 0.0;
          MAXDEC(AARG, aTmp);
          AARG_INC_O;
          for (int i = 0; i < k; i++) {
            aTmp = ARES;
            ARES_INC = 0.0;
            AARG_INC += aTmp;
          }
        }

      GET_TAYL(res, k, p)
      break;

      /*--------------------------------------------------------------------------*/
    case min_a_a: /* Subtraction of two adoubles    min_a_a */
      /* (-) */
      res = tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();

      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg1, rpp_A[arg1])
      ASSIGN_A(Aarg2, rpp_A[arg2])

      for (int l = 0; l < p; l++)
        if (0 == ARES) {
          HOV_INC(Ares, k1)
          HOV_INC(Aarg1, k1)
          HOV_INC(Aarg2, k1)
        } else {
          double aTmp = ARES;
          ARES_INC = 0.0;
          MAXDEC(AARG1, aTmp);
          MAXDEC(AARG2, aTmp);
          AARG2_INC_O;
          AARG1_INC_O;
          for (int i = 0; i < k; i++) {
            aTmp = ARES;
            ARES_INC = 0.0;
            AARG1_INC += aTmp;
            AARG2_INC -= aTmp;
          }
        }

      GET_TAYL(res, k, p)
      break;

      /*--------------------------------------------------------------------------*/
    case min_d_a: /* Subtract an adouble from a    min_d_a */
      /* double (-) */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();
      coval = tape.get_val_r();

      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg, rpp_A[arg])

      for (int l = 0; l < p; l++)
        if (0 == ARES) {
          HOV_INC(Ares, k1)
          HOV_INC(Aarg, k1)
        } else {
          double aTmp = ARES;
          ARES_INC = 0.0;
          MAXDEC(AARG, aTmp);
          AARG_INC_O;
          for (int i = 0; i < k; i++) {
            aTmp = ARES;
            ARES_INC = 0.0;
            AARG_INC -= aTmp;
          }
        }

      GET_TAYL(res, k, p)
      break;

      /*--------------------------------------------------------------------------*/
    case mult_a_a: /* Multiply two adoubles (*)    mult_a_a */
      /* Obtain indices for result and argument variables. */
      res = tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();

      /* Read Taylor polynomial into rpp_T. */
      GET_TAYL(res, k, p)

      /* Set pointer to result and argument variables. */
      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg2, rpp_A[arg2])
      ASSIGN_A(Aarg1, rpp_A[arg1])

      /* Set pointer to Taylor polynomial for argument variables. */
      Targ1 = rpp_T[arg1];
      Targ2 = rpp_T[arg2];

      /* Loop over all input weight vectors (in vector mode).
         In scalar mode this loop is trivial. */
      for (int l = 0; l < p; l++)
        if (0 == ARES) {
          /* This branch is taken if the input of this operation is
           * independent of the independent variables.  For example if it is
           * some constant that happens to be stored as an adouble.  The
           * derivative of that is zero.
           */
          HOV_INC(Aarg1, k1)
          HOV_INC(Aarg2, k1)
          HOV_INC(Ares, k1)
        } else {
          /* The output includes the functional relation between
             input and output.  For multiplication this is at
             least polynomial unless the input already has a more
             generic relation on its own inputs (e.g., rational,
             trancendental or non-smooth).
             See the parameter `nz` of `hov_reverse` and the table of values
             in page 44 of the manual.
             */
          comp = (ARES > 2.0) ? ARES : 2.0;
          ARES_INC = 0.0;
          MAXDEC(AARG1, comp);
          MAXDEC(AARG2, comp);
          /* Skip first value of input: these again represent
             functional relation. */
          AARG1_INC_O;
          AARG2_INC_O;

          /* Copy to a temporary variables in case one of the
             arguments uses the same storage as the result. */
          copyAndZeroset(k, Ares, rp_Atemp);

          // Aarg2 += convolution of rp_Atemp with Targ1
          inconv(k, rp_Atemp, Targ1, Aarg2);

          // Aarg1 += convolution of rp_Atemp with Targ2
          inconv(k, rp_Atemp, Targ2, Aarg1);

          /* Vector mode: update pointers for next loop iteration
             (see loop above) */
          HOV_INC(Ares, k)
          HOV_INC(Aarg1, k)
          HOV_INC(Aarg2, k)
          HOS_OV_INC(Targ1, k)
          HOS_OV_INC(Targ2, k)
        }
      break;

      /*--------------------------------------------------------------------------*/
      /* olvo 991122: new op_code with recomputation */
    case eq_plus_prod: /* increment a product of           eq_plus_prod */
      /* two adoubles (*) */
      res = tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();

      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg2, rpp_A[arg2])
      ASSIGN_A(Aarg1, rpp_A[arg1])
      Targ1 = rpp_T[arg1];
      Targ2 = rpp_T[arg2];

      /* RECOMPUTATION */
      Tres = rpp_T[res];
#if !defined(_HOS_OV_)
      deconv1(k, Targ1, Targ2, Tres);
#endif

      for (int l = 0; l < p; l++) {
#if defined(_HOS_OV_)
        deconv1(k, Targ1, Targ2, Tres);
#endif
        if (0 == ARES) {
          HOV_INC(Aarg1, k1)
          HOV_INC(Aarg2, k1)
          HOV_INC(Ares, k1)
        } else {
          comp = (ARES > 2.0) ? ARES : 2.0;
          ARES_INC = comp;
          MAXDEC(AARG1, comp);
          MAXDEC(AARG2, comp);
          AARG1_INC_O;
          AARG2_INC_O;

          inconv(k, Ares, Targ1, Aarg2);
          inconv(k, Ares, Targ2, Aarg1);

          HOV_INC(Ares, k)
          HOV_INC(Aarg1, k)
          HOV_INC(Aarg2, k)
          HOS_OV_INC(Targ1, k)
          HOS_OV_INC(Targ2, k)
          HOS_OV_INC(Tres, k)
        }
      }
      break;

      /*--------------------------------------------------------------------------*/
      /* olvo 991122: new op_code with recomputation */
    case eq_min_prod: /* decrement a product of             eq_min_prod */
      /* two adoubles (*) */
      res = tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();

      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg2, rpp_A[arg2])
      ASSIGN_A(Aarg1, rpp_A[arg1])
      Targ1 = rpp_T[arg1];
      Targ2 = rpp_T[arg2];

      /* RECOMPUTATION */
      Tres = rpp_T[res];
#if !defined(_HOS_OV_)
      inconv1(k, Targ1, Targ2, Tres);
#endif

      for (int l = 0; l < p; l++) {
#if defined(_HOS_OV_)
        inconv1(k, Targ1, Targ2, Tres);
#endif
        if (0 == ARES) {
          HOV_INC(Aarg1, k1)
          HOV_INC(Aarg2, k1)
          HOV_INC(Ares, k1)
        } else {
          comp = (ARES > 2.0) ? ARES : 2.0;
          ARES_INC = comp;
          MAXDEC(AARG1, comp);
          MAXDEC(AARG2, comp);
          AARG1_INC_O;
          AARG2_INC_O;

          deconv1(k, Ares, Targ1, Aarg2);
          deconv1(k, Ares, Targ2, Aarg1);

          HOV_INC(Ares, k)
          HOV_INC(Aarg1, k)
          HOV_INC(Aarg2, k)
          HOS_OV_INC(Targ1, k)
          HOS_OV_INC(Targ2, k)
          HOS_OV_INC(Tres, k)
        }
      }
      break;

      /*--------------------------------------------------------------------------*/
    case mult_d_a: /* Multiply an adouble by a double    mult_d_a */
      /* (*) */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();
      coval = tape.get_val_r();

      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg, rpp_A[arg])

      for (int l = 0; l < p; l++)
        if (0 == ARES) {
          HOV_INC(Ares, k1)
          HOV_INC(Aarg, k1)
        } else {
          double aTmp = ARES;
          ARES_INC = 0.0;
          MAXDEC(AARG, aTmp);
          AARG_INC_O;
          for (int i = 0; i < k; i++) {
            aTmp = ARES;
            ARES_INC = 0.0;
            AARG_INC += coval * aTmp;
          }
        }

      GET_TAYL(res, k, p)
      break;

      /*--------------------------------------------------------------------------*/
    case div_a_a: /* Divide an adouble by an adouble    div_a_a */
      /* (/) */
      res = tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();

      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg2, rpp_A[arg2])
      ASSIGN_A(Aarg1, rpp_A[arg1])
      Tres = rpp_T[res];
      Targ2 = rpp_T[arg2];

      /* olvo 980922 allows reflexive operation */
      if (arg2 == res) {
        FOR_0_LE_l_LT_pk rp_Ttemp2[l] = Tres[l];
        Tres = rp_Ttemp2;
        GET_TAYL(res, k, p)
      }

      VEC_COMPUTED_INIT
      for (int l = 0; l < p; l++) {
        if (0 == ARES) {
          HOV_INC(Ares, k1)
          HOV_INC(Aarg1, k1)
          HOV_INC(Aarg2, k1)
        } else {
          double aTmp = ARES;
          ARES_INC = 0.0;
          MAXDEC(AARG1, 3.0);
          MAXDEC(AARG1, aTmp);
          MAXDEC(AARG2, 3.0);
          MAXDEC(AARG2, aTmp);
          AARG1_INC_O;
          AARG2_INC_O;

          VEC_COMPUTED_CHECK
          recipr(k, 1.0, Targ2, rp_Ttemp);
          conv0(k, rp_Ttemp, Tres, rp_Atemp2);
          VEC_COMPUTED_END
          copyAndZeroset(k, Ares, rp_Atemp);
          inconv(k, rp_Atemp, rp_Ttemp, Aarg1);
          deconv(k, rp_Atemp, rp_Atemp2, Aarg2);

          HOV_INC(Ares, k)
          HOV_INC(Aarg1, k)
          HOV_INC(Aarg2, k)
          HOS_OV_INC(Tres, k)
          HOS_OV_INC(Targ2, k)
        }
      }

      if (res != arg2)
        GET_TAYL(res, k, p)
      break;

      /*--------------------------------------------------------------------------*/
    case div_d_a: /* Division double - adouble (/)    div_d_a */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();
      coval = tape.get_val_r();

      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg, rpp_A[arg])
      Tres = rpp_T[res];
      Targ = rpp_T[arg];

      /* olvo 980922 allows reflexive operation */
      if (arg == res) {
        FOR_0_LE_l_LT_pk rp_Ttemp2[l] = Tres[l];
        Tres = rp_Ttemp2;
        GET_TAYL(arg, k, p)
      }

      VEC_COMPUTED_INIT
      for (int l = 0; l < p; l++) {
        if (0 == ARES) {
          HOV_INC(Ares, k1)
          HOV_INC(Aarg, k1)
        } else {
          double aTmp = ARES;
          ARES_INC = 0.0;
          MAXDEC(AARG, aTmp);
          MAXDEC(AARG, 3.0);
          AARG_INC_O;

          VEC_COMPUTED_CHECK
          recipr(k, 1.0, Targ, rp_Ttemp);
          conv0(k, rp_Ttemp, Tres, rp_Atemp);
          VEC_COMPUTED_END
          deconv0(k, Ares, rp_Atemp, Aarg);

          HOV_INC(Ares, k)
          HOV_INC(Aarg, k)
          HOS_OV_INC(Tres, k)
          HOS_OV_INC(Targ, k)
        }
      }

      if (arg != res)
        GET_TAYL(res, k, p)
      break;

      /****************************************************************************/
      /*                                                         SIGN
       * OPERATIONS
       */

      /*--------------------------------------------------------------------------*/
    case pos_sign_a: /* pos_sign_a */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();

      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg, rpp_A[arg])

      for (int l = 0; l < p; l++)
        if (0 == ARES) {
          HOV_INC(Ares, k1)
          HOV_INC(Aarg, k1)
        } else {
          double aTmp = ARES;
          ARES_INC = 0.0;
          MAXDEC(AARG, aTmp);
          AARG_INC_O;
          for (int i = 0; i < k; i++) {
            aTmp = ARES;
            ARES_INC = 0.0;
            AARG_INC += aTmp;
          }
        }

      GET_TAYL(res, k, p)
      break;

      /*--------------------------------------------------------------------------*/
    case neg_sign_a: /* neg_sign_a */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();

      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg, rpp_A[arg])

      for (int l = 0; l < p; l++)
        if (0 == ARES) {
          HOV_INC(Ares, k1)
          HOV_INC(Aarg, k1)
        } else {
          double aTmp = ARES;
          ARES_INC = 0.0;
          MAXDEC(AARG, aTmp);
          AARG_INC_O;
          for (int i = 0; i < k; i++) {
            aTmp = ARES;
            ARES_INC = 0.0;
            AARG_INC -= aTmp;
          }
        }

      GET_TAYL(res, k, p)
      break;

      /****************************************************************************/
      /*                                                         UNARY
       * OPERATIONS */

      /*--------------------------------------------------------------------------*/
    case exp_op: /* exponent operation    exp_op */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();

      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg, rpp_A[arg])
      Tres = rpp_T[res];
      Targ = rpp_T[arg];

      for (int l = 0; l < p; l++) {
        if (0 == ARES) {
          HOV_INC(Aarg, k1)
          HOV_INC(Ares, k1)
        } else {
          double aTmp = ARES;
          ARES_INC = 0.0;
          MAXDEC(AARG, aTmp);
          MAXDEC(AARG, 4.0);
          AARG_INC_O;

          inconv0(k, Ares, Tres, Aarg);

          HOV_INC(Ares, k)
          HOV_INC(Aarg, k)
          HOS_OV_INC(Tres, k)
        }
      }

      GET_TAYL(res, k, p)
      break;

      /*--------------------------------------------------------------------------*/
    case sin_op: /* sine operation    sin_op */
      res = tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();

      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg1, rpp_A[arg1])
      Targ2 = rpp_T[arg2];

      for (int l = 0; l < p; l++) {
        if (0 == ARES) {
          HOV_INC(Aarg1, k1)
          HOV_INC(Ares, k1)
        } else {
          double aTmp = ARES;
          ARES_INC = 0.0;
          MAXDEC(AARG1, aTmp);
          MAXDEC(AARG1, 4.0);
          AARG1_INC_O;

          inconv0(k, Ares, Targ2, Aarg1);

          HOV_INC(Ares, k)
          HOV_INC(Aarg1, k)
          HOS_OV_INC(Targ2, k)
        }
      }

      GET_TAYL(res, k, p)
      GET_TAYL(arg2, k, p) /* olvo 980710 covalue */
      /* NOTE: rpp_A[arg2] should be 0 already */
      break;

      /*--------------------------------------------------------------------------*/
    case cos_op: /* cosine operation    cos_op */
      res = tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();

      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg1, rpp_A[arg1])
      Targ2 = rpp_T[arg2];

      for (int l = 0; l < p; l++) {
        if (0 == ARES) {
          HOV_INC(Aarg1, k1)
          HOV_INC(Ares, k1)
        } else {
          double aTmp = ARES;
          ARES_INC = 0.0;
          MAXDEC(AARG1, aTmp);
          MAXDEC(AARG1, 4.0);
          AARG1_INC_O;

          deconv0(k, Ares, Targ2, Aarg1);

          HOV_INC(Ares, k)
          HOV_INC(Aarg1, k)
          HOS_OV_INC(Targ2, k)
        }
      }

      GET_TAYL(res, k, p)
      GET_TAYL(arg2, k, p) /* olvo 980710 covalue */
      /* NOTE: rpp_A[arg2] should be 0 already */
      break;
      /*xxx*/
      /*--------------------------------------------------------------------------*/
    case atan_op:  /* atan_op  */
    case asin_op:  /* asin_op  */
    case acos_op:  /* acos_op  */
    case asinh_op: /* asinh_op */
    case acosh_op: /* acosh_op */
    case atanh_op: /* atanh_op */
    case erf_op:   /* erf_op   */
    case erfc_op:  /* erfc_op  */
      res = tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();

      GET_TAYL(res, k, p)

      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg1, rpp_A[arg1])
      Targ2 = rpp_T[arg2];

      for (int l = 0; l < p; l++) {
        if (0 == ARES) {
          HOV_INC(Aarg1, k1)
          HOV_INC(Ares, k1)
        } else {
          double aTmp = ARES;
          ARES_INC = 0.0;
          MAXDEC(AARG1, aTmp);
          MAXDEC(AARG1, 4.0);
          AARG1_INC_O;

          inconv0(k, Ares, Targ2, Aarg1);

          HOV_INC(Aarg1, k)
          HOV_INC(Ares, k)
          HOS_OV_INC(Targ2, k)
        }
      }
      break;

      /*--------------------------------------------------------------------------*/
    case log_op: /* log_op */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();

      GET_TAYL(res, k, p)

      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg, rpp_A[arg])
      Targ = rpp_T[arg];

      VEC_COMPUTED_INIT
      for (int l = 0; l < p; l++) {
        if (0 == ARES) {
          HOV_INC(Aarg, k1)
          HOV_INC(Ares, k1)
        } else {
          double aTmp = ARES;
          ARES_INC = 0.0;
          MAXDEC(AARG, aTmp);
          MAXDEC(AARG, 4.0);
          AARG_INC_O;

          VEC_COMPUTED_CHECK
          recipr(k, 1.0, Targ, rp_Ttemp);
          VEC_COMPUTED_END
          inconv0(k, Ares, rp_Ttemp, Aarg);

          HOV_INC(Ares, k)
          HOV_INC(Aarg, k)
          HOS_OV_INC(Targ, k)
        }
      }
      break;

      /*--------------------------------------------------------------------------*/
    case pow_op: /* pow_op */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();
      coval = tape.get_val_r();

      Targ = rpp_T[arg];
      Tres = rpp_T[res];
      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg, rpp_A[arg])

      /* olvo 980921 allows reflexive operation */
      if (arg == res) {
        FOR_0_LE_l_LT_pk rp_Ttemp2[l] = Tres[l];
        Tres = rp_Ttemp2;
        GET_TAYL(arg, k, p)
      }

      VEC_COMPUTED_INIT
      for (int l = 0; l < p; l++)
        if (0 == ARES) {
          HOV_INC(Aarg, k1)
          HOV_INC(Ares, k1)
        } else {
          double aTmp = ARES;
          ARES_INC = 0.0;
          MAXDEC(AARG, aTmp);
          MAXDEC(AARG, 4.0);
          AARG_INC_O;

          VEC_COMPUTED_CHECK
          if (fabs(Targ[0]) > ADOLC_EPS) {
            divide(k, Tres, Targ, rp_Ttemp);
            for (int i = 0; i < k; i++) {
              rp_Ttemp[i] *= coval;
              /*                 printf(" EPS i %d %f\n",i,rp_Ttemp[i]); */
            }
            inconv0(k, Ares, rp_Ttemp, Aarg);
          } else {
            if (coval <= 0.0) {
              for (int i = 0; i < k; i++) {
                Aarg[i] = tape.make_nan();
                Ares[i] = 0;
              }
            } else {
              /* coval not a whole number */
              if (coval - floor(coval) != 0) {
                for (int i = 0; i < k; i++) {
                  if (coval - i > 1) {
                    Aarg[i] = 0;
                    Ares[i] = 0;
                  }
                  if ((coval - i < 1) && (coval - i > 0)) {
                    Aarg[i] = tape.make_inf();
                    Ares[i] = 0;
                  }
                  if (coval - i < 0) {
                    Aarg[i] = tape.make_nan();
                    Ares[i] = 0;
                  }
                }
              } else {
                if (coval == 1) {
                  for (int i = 0; i < k; i++) { /* ! no temporary */
                    Aarg[i] += Ares[i];
                    Ares[i] = 0.0;
                  }
                } else {
                  /* coval is an int > 1 */
                  /* the following is not efficient but at least it works */
                  /* it reformulates x^n into x* ... *x n times */

                  copyAndZeroset(k, Ares, rp_Atemp);
                  inconv(k, rp_Atemp, Targ, Aarg);
                  inconv(k, rp_Atemp, Targ, Aarg);
                  if (coval == 3) {
                    conv(k, Aarg, Targ, rp_Atemp);
                    for (int i = 0; i < k; i++)
                      Aarg[i] = 2.0 * rp_Atemp[i];
                  }
                }
              }
            }
          }
          VEC_COMPUTED_END

          HOV_INC(Ares, k)
          HOV_INC(Aarg, k)
          HOS_OV_INC(Tres, k)
          HOS_OV_INC(Targ, k)
        }

      if (arg != res)
        GET_TAYL(res, k, p)
      break;

      /*--------------------------------------------------------------------------*/
    case sqrt_op: /* sqrt_op */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();

      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg, rpp_A[arg])
      Tres = rpp_T[res];

      VEC_COMPUTED_INIT
      for (int l = 0; l < p; l++)
        if (0 == ARES) {
          HOV_INC(Aarg, k1)
          HOV_INC(Ares, k1)
        } else {
          double aTmp = ARES;
          ARES_INC = 0.0;
          MAXDEC(AARG, aTmp);
          MAXDEC(AARG, 4.0);
          AARG_INC_O;

          VEC_COMPUTED_CHECK
          recipr(k, 0.5, Tres, rp_Ttemp);
          VEC_COMPUTED_END
          inconv0(k, Ares, rp_Ttemp, Aarg);

          HOV_INC(Ares, k)
          HOV_INC(Aarg, k)
          HOS_OV_INC(Tres, k)
        }

      GET_TAYL(res, k, p)
      break;

      /*--------------------------------------------------------------------------*/
    case cbrt_op: /* cbrt_op */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();
      ADOLCError::fail(ADOLCError::ErrorType::HO_OP_NOT_IMPLEMENTED,
                       CURRENT_LOCATION,
                       ADOLCError::FailInfo{.info7 = operation});

      break;

      /*--------------------------------------------------------------------------*/
    case gen_quad: /* gen_quad */
      res = tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();
      coval = tape.get_val_r();
      coval = tape.get_val_r();

      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg1, rpp_A[arg1])
      Targ2 = rpp_T[arg2];

      for (int l = 0; l < p; l++)
        if (0 == ARES) {
          HOV_INC(Aarg1, k1)
          HOV_INC(Ares, k1)
        } else {
          double aTmp = ARES;
          ARES_INC = 0.0;
          MAXDEC(AARG1, aTmp);
          MAXDEC(AARG1, 4.0);
          AARG1_INC_O;

          inconv0(k, Ares, Targ2, Aarg1);

          HOV_INC(Aarg1, k)
          HOV_INC(Ares, k)
          HOS_OV_INC(Targ2, k)
        }

      GET_TAYL(res, k, p)
      break;

      /*--------------------------------------------------------------------------*/
    case min_op: /* min_op */

      res = tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();
      coval = tape.get_val_r();

      GET_TAYL(res, k, p)

      ASSIGN_A(Aarg1, rpp_A[arg1])
      ASSIGN_A(Aarg2, rpp_A[arg2])
      ASSIGN_A(Ares, rpp_A[res])
      Targ1 = rpp_T[arg1];
      Targ2 = rpp_T[arg2];
      ASSIGN_A(AP1, NULL)
      ASSIGN_A(AP2, Ares)

      if (Targ1[0] > Targ2[0]) {
        for (int l = 0; l < p; l++) {
          if ((coval) && (*AP2))
            MINDEC(ret_c, 2);

          // increment the adjoint of res
          HOV_INC(AP2, k1)
        }

        // select the adjoint of the minimum
        AP1 = Aarg2;

        // used to indicate that we can decide which value is smaller
        arg = 0;
      } else if (Targ1[0] < Targ2[0]) {
        for (int l = 0; l < p; l++) {
          if ((!coval) && (*AP2))
            MINDEC(ret_c, 2);

          // increment the adjoint of res
          HOV_INC(AP2, k1)
        }

        // select the adjoint of the minimum
        AP1 = Aarg1;

        // used to indicate that we can decide which value is smaller
        arg = 0;

      }
      // both input args are equal
      // for hos_ov we have to select the taylors for every direction
      else {
#ifdef _HOS_OV_
        printf("TIE POINT HOS_OV IS NOT SUPPORTED");
        ADOLCError::fail(ADOLCError::ErrorType::HO_OP_NOT_IMPLEMENTED,
                         CURRENT_LOCATION,
                         ADOLCError::FailInfo{.info7 = min_op});
#endif
        for (int i = 1; i < k; i++) {
          if (Targ1[i] > Targ2[i]) {
            for (int l = 0; l < p; l++) {
              if (*AP2)
                MINDEC(ret_c, 1);
              HOV_INC(AP2, k1)
            }
            AP1 = Aarg2;

            // used to indicate that we have a tie in input args but can decide
            // based on taylors
            arg = i + 1;
          } else if (Targ1[i] < Targ2[i]) {
            for (int l = 0; l < p; l++) {
              if (*AP2)
                MINDEC(ret_c, 1);
              HOV_INC(AP2, k1)
            }
            AP1 = Aarg1;

            // used to indicate that we have a tie in input args but can decide
            // based on taylors
            arg = i + 1;
          }
          if (AP1 != NULL)
            break;
        }
      }
      // we selected a minimum
      if (AP1 != NULL)
        for (int l = 0; l < p; l++) {
          if (0 == ARES) {
            HOV_INC(AP1, k1)
            HOV_INC(Ares, k1);
          } else {
            double aTmp = ARES;
            ARES_INC = 0.0;

            // we are at the tie in input args but can decide based on taylors
            if (arg)
              *AP1 = 5.0;
            else
              MAXDEC(*AP1, aTmp);
            AP1++;
            for (int i = 0; i < k; i++) {
              aTmp = ARES;
              ARES_INC = 0.0;
              *AP1++ += aTmp;
            }
          }
        }
      // both input arg and tangent are identical
      else {
        for (int l = 0; l < p; l++) {
          if (0 == ARES) {
            HOV_INC(Aarg1, k1)
            HOV_INC(Aarg2, k1)
            HOV_INC(Ares, k1)
          } else {
            double aTmp = ARES;
            ARES_INC = 0.0;
            MAXDEC(AARG1, aTmp); /*assume sthg like fmin(x,x) */
            MAXDEC(AARG2, aTmp);
            AARG1_INC_O;
            AARG2_INC_O;
            for (int i = 0; i < k; i++) {
              aTmp = ARES;
              ARES_INC = 0.0;
              AARG1_INC += aTmp / 2;
              AARG2_INC += aTmp / 2;
            }
          }
        }
        if (arg1 != arg2)
          MINDEC(ret_c, 1);
      }
      break;

      /*--------------------------------------------------------------------------*/
    case abs_val: /* abs_val */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();
      coval = tape.get_val_r();
      /* must be changed for hos_ov, but how? */
      /* seems to influence the return value  */
      GET_TAYL(res, k, p)

      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg, rpp_A[arg])
      Targ = rpp_T[arg];

      for (int l = 0; l < q; ++l) {
        x[l] = 0.0;
        jj[l] = 0;
        for (int i = 0; i < k; i++)
          if ((x[l] == 0.0) && (Targ[i] != 0.0)) {
            jj[l] = i;
            if (Targ[i] < 0.0)
              x[l] = -1.0;
            else
              x[l] = 1.0;
          }
        HOS_OV_INC(Targ, k)
      }
      Targ = rpp_T[arg];
      for (int l = 0; l < p; l++) {
        if (0 == ARES) {
          HOV_INC(Aarg, k1)
          HOV_INC(Ares, k1)
        } else {
          if (Targ[0] == 0.0) {
            ARES_INC = 0.0;
            AARG_INC = 5.0;
          } else {
            double aTmp = ARES;
            ARES_INC = 0.0;
            MAXDEC(AARG, aTmp);
            AARG_INC_O;
          }
          if (Targ[0] == 0.0)
            MINDEC(ret_c, 1);
          for (int i = 0; i < jj[l]; i++)
            ARES_INC = 0.0;
          Aarg += jj[l];
          for (int i = jj[l]; i < k; i++) {
            double aTmp = ARES;
            ARES_INC = 0.0;
            if ((coval) && (x[l] < 0) && (aTmp))
              MINDEC(ret_c, 2);
            if ((!coval) && (x[l] > 0) && (aTmp))
              MINDEC(ret_c, 2);
            AARG_INC += x[l] * aTmp;
          }
        }
        HOS_OV_INC(Targ, k)
      }
      break;

      /*--------------------------------------------------------------------------*/
    case ceil_op: /* ceil_op */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();
      coval = tape.get_val_r();

      GET_TAYL(res, k, p)

      coval = (coval != ceil(*rpp_T[arg]));

      ASSIGN_A(Ares, rpp_A[res])

      for (int l = 0; l < p; l++)
        if (0 == ARES) {
          HOV_INC(Aarg, k1)
          HOV_INC(Ares, k1)
        } else {
          ARES_INC = 0.0;
          AARG_INC = 5.0;
          for (int i = 0; i < k; i++) {
            if ((coval) && (ARES))
              MINDEC(ret_c, 2);
            ARES_INC = 0.0;
          }
          HOV_INC(Aarg, k)
        }
      break;

      /*--------------------------------------------------------------------------*/
    case floor_op: /* floor_op */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();
      coval = tape.get_val_r();

      GET_TAYL(res, k, p)

      coval = (coval != floor(*rpp_T[arg]));

      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg, rpp_A[arg])

      for (int l = 0; l < p; l++)
        if (0 == ARES) {
          HOV_INC(Aarg, k1)
          HOV_INC(Ares, k1)
        } else {
          ARES = 0.0;
          AARG_INC = 5.0;
          for (int i = 0; i < k; i++) {
            if ((coval) && (ARES))
              MINDEC(ret_c, 2);
            ARES_INC = 0.0;
          }
          HOV_INC(Aarg, k)
        }
      break;

      /****************************************************************************/
      /*                                                             CONDITIONALS
       */

      /*--------------------------------------------------------------------------*/
    case cond_assign: /* cond_assign */
      res = tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();
      arg = tape.get_locint_r();
      coval = tape.get_val_r();

      GET_TAYL(res, k, p)

      ASSIGN_A(Aarg1, rpp_A[arg1])
      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg2, rpp_A[arg2])
      Targ = rpp_T[arg];

      /* olvo 980925 changed code a little bit */
      if (*Targ > 0.0) {
        if (res != arg1)
          for (int l = 0; l < p; l++) {
            if (0 == ARES) {
              HOV_INC(Ares, k1)
              HOV_INC(Aarg1, k1)
            } else {
              if (coval <= 0.0)
                MINDEC(ret_c, 2);
              MAXDEC(AARG1, ARES);
              ARES_INC = 0.0;
              AARG1_INC_O;
              for (int i = 0; i < k; i++) {
                AARG1_INC += ARES;
                ARES_INC = 0;
              }
            }
          }
        else
          for (int l = 0; l < p; l++) {
            if ((coval <= 0.0) && (ARES))
              MINDEC(ret_c, 2);
            HOV_INC(Ares, k1)
          }
      } else /* TARG <= 0.0 */
      {
        if (res != arg2)
          for (int l = 0; l < p; l++) {
            if (0 == ARES) {
              HOV_INC(Ares, k1)
              HOV_INC(Aarg2, k1)
            } else {
              if (*Targ == 0.0) /* we are at the tie */
              {
                MINDEC(ret_c, 0);
                AARG1 = 5.0;
                AARG2_INC = 5.0;
              } else {
                if (coval <= 0.0)
                  MINDEC(ret_c, 2);
                MAXDEC(AARG2, ARES);
                AARG2_INC_O;
              }
              ARES_INC = 0.0;

              for (int i = 0; i < k; i++) {
                AARG2_INC += ARES;
                ARES_INC = 0;
              }
            }
            HOV_INC(Aarg1, k1)
          }
        else
          for (int l = 0; l < p; l++) {
            if (ARES) {
              if (*Targ == 0.0) /* we are at the tie */
              {
                MINDEC(ret_c, 0);
                AARG1 = 5.0;
                AARG2 = 5.0;
              } else if (coval <= 0.0)
                MINDEC(ret_c, 2);
            }
            HOV_INC(Ares, k1)
            HOV_INC(Aarg1, k1)
            HOV_INC(Aarg2, k1)
          }
      }
      break;

    case cond_eq_assign: /* cond_eq_assign */
      res = tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();
      arg = tape.get_locint_r();
      coval = tape.get_val_r();

      GET_TAYL(res, k, p)

      ASSIGN_A(Aarg1, rpp_A[arg1])
      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg2, rpp_A[arg2])
      Targ = rpp_T[arg];

      /* olvo 980925 changed code a little bit */
      if (*Targ >= 0.0) {
        if (res != arg1)
          for (int l = 0; l < p; l++) {
            if (0 == ARES) {
              HOV_INC(Ares, k1)
              HOV_INC(Aarg1, k1)
            } else {
              if (coval < 0.0)
                MINDEC(ret_c, 2);
              MAXDEC(AARG1, ARES);
              ARES_INC = 0.0;
              AARG1_INC_O;
              for (int i = 0; i < k; i++) {
                AARG1_INC += ARES;
                ARES_INC = 0;
              }
            }
          }
        else
          for (int l = 0; l < p; l++) {
            if ((coval < 0.0) && (ARES))
              MINDEC(ret_c, 2);
            HOV_INC(Ares, k1)
          }
      } else /* TARG < 0.0 */
      {
        if (res != arg2)
          for (int l = 0; l < p; l++) {
            if (0 == ARES) {
              HOV_INC(Ares, k1)
              HOV_INC(Aarg2, k1)
            } else {
              if (coval < 0.0)
                MINDEC(ret_c, 2);
              MAXDEC(AARG2, ARES);
              AARG2_INC_O;
              ARES_INC = 0.0;

              for (int i = 0; i < k; i++) {
                AARG2_INC += ARES;
                ARES_INC = 0;
              }
            }
            HOV_INC(Aarg1, k1)
          }
        else
          for (int l = 0; l < p; l++) {
            if (ARES) {
              if (coval < 0.0)
                MINDEC(ret_c, 2);
            }
            HOV_INC(Ares, k1)
            HOV_INC(Aarg1, k1)
            HOV_INC(Aarg2, k1)
          }
      }
      break;

      /*--------------------------------------------------------------------------*/
    case cond_assign_s: /* cond_assign_s */
      res = tape.get_locint_r();
      arg1 = tape.get_locint_r();
      arg = tape.get_locint_r();
      coval = tape.get_val_r();

      GET_TAYL(res, k, p)

      ASSIGN_A(Aarg1, rpp_A[arg1])
      ASSIGN_A(Ares, rpp_A[res])
      Targ = rpp_T[arg];

      /* olvo 980925 changed code a little bit */
      if (*Targ == 0.0) /* we are at the tie */
      {
        for (int l = 0; l < p; l++) {
          if (ARES)
            AARG1 = 5.0;
          HOV_INC(Aarg1, k1)
          HOV_INC(Ares, k1)
        }
        MINDEC(ret_c, 0);
      } else if (*Targ > 0.0) {
        if (res != arg1)
          for (int l = 0; l < p; l++) {
            if (0 == ARES) {
              HOV_INC(Ares, k1)
              HOV_INC(Aarg1, k1)
            } else {
              if (coval <= 0.0)
                MINDEC(ret_c, 2);
              MAXDEC(AARG1, ARES);
              ARES_INC = 0.0;
              AARG1_INC_O;
              for (int i = 0; i < k; i++) {
                (AARG1_INC) += ARES;
                ARES_INC = 0;
              }
            }
          }
        else
          for (int l = 0; l < p; l++) {
            if ((coval <= 0.0) && (ARES))
              MINDEC(ret_c, 2);
            HOV_INC(Ares, k1)
          }
      }
      break;
    case cond_eq_assign_s: /* cond_eq_assign_s */
      res = tape.get_locint_r();
      arg1 = tape.get_locint_r();
      arg = tape.get_locint_r();
      coval = tape.get_val_r();

      GET_TAYL(res, k, p)

      ASSIGN_A(Aarg1, rpp_A[arg1])
      ASSIGN_A(Ares, rpp_A[res])
      Targ = rpp_T[arg];

      /* olvo 980925 changed code a little bit */
      if (*Targ >= 0.0) {
        if (res != arg1)
          for (int l = 0; l < p; l++) {
            if (0 == ARES) {
              HOV_INC(Ares, k1)
              HOV_INC(Aarg1, k1)
            } else {
              if (coval < 0.0)
                MINDEC(ret_c, 2);
              MAXDEC(AARG1, ARES);
              ARES_INC = 0.0;
              AARG1_INC_O;
              for (int i = 0; i < k; i++) {
                (AARG1_INC) += ARES;
                ARES_INC = 0;
              }
            }
          }
        else
          for (int l = 0; l < p; l++) {
            if ((coval < 0.0) && (ARES))
              MINDEC(ret_c, 2);
            HOV_INC(Ares, k1)
          }
      }
      break;
      /*--------------------------------------------------------------------------*/
      /* NEW CONDITIONALS */
      /*--------------------------------------------------------------------------*/
#if defined(ADOLC_ADVANCED_BRANCHING)
    case neq_a_a:
    case eq_a_a:
    case le_a_a:
    case ge_a_a:
    case lt_a_a:
    case gt_a_a:
      res = tape.get_locint_r();
      arg1 = tape.get_locint_r();
      arg = tape.get_locint_r();
      coval = tape.get_val_r();
      ASSIGN_A(Ares, rpp_A[res])

      FOR_0_LE_l_LT_pk1 ARES_INC = 0.0;

      GET_TAYL(res, k, p)
      break;
#endif

      /*--------------------------------------------------------------------------*/
    case subscript:
      coval = tape.get_val_r();
      {
        size_t idx, numval = (size_t)trunc(fabs(coval));
        locint vectorloc;
        res = tape.get_locint_r();
        vectorloc = tape.get_locint_r();
        arg = tape.get_locint_r();
        Targ = rpp_T[arg];
        idx = (size_t)trunc(fabs(*Targ));
        if (idx >= numval)
          fprintf(DIAG_OUT,
                  "ADOL-C warning: index out of bounds while subscripting "
                  "n=%zu, idx=%zu\n",
                  numval, idx);
        arg1 = vectorloc + idx;
        ASSIGN_A(Aarg1, rpp_A[arg1])
        ASSIGN_A(Ares, rpp_A[res])

        for (int l = 0; l < p; l++)
          if (0 == ARES) {
            HOV_INC(Aarg1, k1)
            HOV_INC(Ares, k1)
          } else {
            MAXDEC(AARG1, ARES);
            AARG1_INC_O;
            ARES_INC = 0.0;
            for (int i = 0; i < k; i++) {
              AARG1_INC += ARES;
              ARES_INC = 0.0;
            }
          }
        GET_TAYL(res, k, p)
      }
      break;

    case subscript_ref:
      coval = tape.get_val_r();
      {
        size_t idx, numval = (size_t)trunc(fabs(coval));
        locint vectorloc;
        res = tape.get_locint_r();
        vectorloc = tape.get_locint_r();
        arg = tape.get_locint_r();
        Targ = rpp_T[arg];
        Tres = rpp_T[res];
        idx = (size_t)trunc(fabs(*Targ));
        if (idx >= numval)
          fprintf(DIAG_OUT,
                  "ADOL-C warning: index out of bounds while subscripting "
                  "(ref) n=%zu, idx=%zu\n",
                  numval, idx);
        arg1 = (size_t)trunc(fabs(*Tres));
        /*
         * This is actually NOP
         * basically all we need is that arg1 == vectorloc[idx]
         * so doing a check here is probably good
         */
        if (arg1 != vectorloc + idx)
          ADOLCError::fail(
              ADOLCError::ErrorType::ADUBREF_SAFE_MODE, CURRENT_LOCATION,
              ADOLCError::FailInfo{.info5 = vectorloc + idx, .info6 = arg1});

        GET_TAYL(res, k, p)
      }
      break;

    case ref_copyout:
      res = tape.get_locint_r();
      arg1 = tape.get_locint_r();

      Targ1 = rpp_T[arg1];
      arg = (size_t)trunc(fabs(*Targ1));

      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg, rpp_A[arg])

      for (int l = 0; l < p; l++)
        if (0 == ARES) {
          HOV_INC(Aarg, k1)
          HOV_INC(Ares, k1)
        } else {
          MAXDEC(AARG, ARES);
          AARG_INC_O;
          ARES_INC = 0.0;
          for (int i = 0; i < k; i++) {
            AARG_INC += ARES;
            ARES_INC = 0.0;
          }
        }
      GET_TAYL(res, k, p)
      break;

    case ref_incr_a: /* Increment an adouble    incr_a */
    case ref_decr_a: /* Increment an adouble    decr_a */
      arg1 = tape.get_locint_r();

      Targ1 = rpp_T[arg1];
      res = (size_t)trunc(fabs(*Targ1));

      GET_TAYL(res, k, p)
      break;

    case ref_assign_d: /* assign an adouble variable a    assign_d */
      /* double value. (=) */
      coval = tape.get_val_r();
      /* fallthrough */
    case ref_assign_d_zero: /* assign an adouble a        assign_d_zero */
    case ref_assign_d_one:  /* double value. (=)           assign_d_one */
      arg1 = tape.get_locint_r();

      Targ1 = rpp_T[arg1];
      res = (size_t)trunc(fabs(*Targ1));

      ASSIGN_A(Ares, rpp_A[res])

      FOR_0_LE_l_LT_pk1 ARES_INC = 0.0;

      GET_TAYL(res, k, p)
      break;

    case ref_assign_a: /* assign an adouble variable an    assign_a */
      /* adouble value. (=) */
      arg1 = tape.get_locint_r();
      arg = tape.get_locint_r();

      Targ1 = rpp_T[arg1];
      res = (size_t)trunc(fabs(*Targ1));

      ASSIGN_A(Aarg, rpp_A[arg])
      ASSIGN_A(Ares, rpp_A[res])

      for (int l = 0; l < p; l++)
        if (0 == ARES) {
          HOV_INC(Aarg, k1)
          HOV_INC(Ares, k1)
        } else {
          MAXDEC(AARG, ARES);
          AARG_INC_O;
          ARES_INC = 0.0;
          for (int i = 0; i < k; i++) { /* ! no temporary */
            AARG_INC += ARES;
            ARES_INC = 0.0;
          }
        }

      GET_TAYL(res, k, p)
      break;

    case ref_assign_ind: /* assign an adouble variable an    assign_ind */
      /* independent double value (<<=) */
      arg1 = tape.get_locint_r();
      Targ1 = rpp_T[arg1];
      res = (size_t)trunc(fabs(*Targ1));
      ASSIGN_A(Ares, rpp_A[res])

      for (int l = 0; l < p; l++) {
#ifdef _HOV_
        if (nonzero) /* ??? question: why here? */
          nonzero[l][indexi] = (int)ARES;
#endif /* _HOV_ */
        ARES_INC_O;
        for (int i = 0; i < k; i++)
          RESULTS(l, indexi, i) = ARES_INC;
      }

      GET_TAYL(res, k, p)
      indexi--;
      break;

    case ref_eq_plus_d: /* Add a floating point to an    eq_plus_d */
                        /* adouble. (+=) */
      arg1 = tape.get_locint_r();
      Targ1 = rpp_T[arg1];
      res = (size_t)trunc(fabs(*Targ1));
      coval = tape.get_val_r();

      GET_TAYL(res, k, p)
      break;

    case ref_eq_plus_a: /* Add an adouble to another    eq_plus_a */
      /* adouble. (+=) */
      arg1 = tape.get_locint_r();
      arg = tape.get_locint_r();

      Targ1 = rpp_T[arg1];
      res = (size_t)trunc(fabs(*Targ1));
      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg, rpp_A[arg])

      for (int l = 0; l < p; l++)
        if (0 == ARES) {
          HOV_INC(Ares, k1)
          HOV_INC(Aarg, k1)
        } else {
          MAXDEC(AARG, ARES);
          AARG_INC_O;
          ARES_INC_O;
          for (int i = 0; i < k; i++)
            AARG_INC += ARES_INC;
        }

      GET_TAYL(res, k, p)
      break;

    case ref_eq_min_d: /* Subtract a floating point from an    eq_min_d */
      /* adouble. (-=) */
      arg1 = tape.get_locint_r();
      Targ1 = rpp_T[arg1];
      res = (size_t)trunc(fabs(*Targ1));
      coval = tape.get_val_r();

      GET_TAYL(res, k, p)
      break;

    case ref_eq_min_a: /* Subtract an adouble from another    eq_min_a */
      /* adouble. (-=) */
      arg1 = tape.get_locint_r();
      arg = tape.get_locint_r();

      Targ1 = rpp_T[arg1];
      res = (size_t)trunc(fabs(*Targ1));
      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg, rpp_A[arg])

      for (int l = 0; l < p; l++)
        if (0 == ARES) {
          HOV_INC(Ares, k1)
          HOV_INC(Aarg, k1)
        } else {
          MAXDEC(AARG, ARES);
          AARG_INC_O;
          ARES_INC_O;
          for (int i = 0; i < k; i++)
            AARG_INC -= ARES_INC;
        }

      GET_TAYL(res, k, p)
      break;

    case ref_eq_mult_d: /* Multiply an adouble by a    eq_mult_d */
      /* floating point. (*=) */
      arg1 = tape.get_locint_r();
      Targ1 = rpp_T[arg1];
      res = (size_t)trunc(fabs(*Targ1));
      coval = tape.get_val_r();

      ASSIGN_A(Ares, rpp_A[res])

      for (int l = 0; l < p; l++)
        if (0 == ARES_INC)
          HOV_INC(Ares, k) else for (int i = 0; i < k; i++) ARES_INC *= coval;

      GET_TAYL(res, k, p)
      break;

    case ref_eq_mult_a: /* Multiply one adouble by another    eq_mult_a */
      /* (*=) */
      arg1 = tape.get_locint_r();
      arg = tape.get_locint_r();
      Targ1 = rpp_T[arg1];
      res = (size_t)trunc(fabs(*Targ1));

      GET_TAYL(res, k, p)

      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg, rpp_A[arg])
      Tres = rpp_T[res];
      Targ = rpp_T[arg];

      for (int l = 0; l < p; l++) {
        if (0 == ARES) {
          HOV_INC(Aarg, k1)
          HOV_INC(Ares, k1)
        } else {
          MAXDEC(ARES, 2.0);
          MAXDEC(AARG, ARES);
          AARG_INC_O;
          ARES_INC_O;
          conv(k, Ares, Targ, rp_Atemp);
          if (arg != res) {
            inconv(k, Ares, Tres, Aarg);
            for (int i = 0; i < k; i++)
              ARES_INC = rp_Atemp[i];
          } else
            for (int i = 0; i < k; i++)
              ARES_INC = 2.0 * rp_Atemp[i];
          HOV_INC(Aarg, k)
          HOS_OV_INC(Tres, k)
          HOS_OV_INC(Targ, k)
        }
      }
      break;

    case vec_copy:

      res = tape.get_locint_r();
      size = tape.get_locint_r();
      arg = tape.get_locint_r();

      for (locint qq = 0; qq < size; qq++) {

        ASSIGN_A(Aarg, rpp_A[arg + qq])
        ASSIGN_A(Ares, rpp_A[res + qq])

        for (int l = 0; l < p; l++)
          if (0 == ARES) {
            HOV_INC(Aarg, k1)
            HOV_INC(Ares, k1)
          } else {
            MAXDEC(AARG, ARES);
            AARG_INC_O;
            ARES_INC = 0.0;
            for (int i = 0; i < k; i++) { /* ! no temporary */
              AARG_INC += ARES;
              ARES_INC = 0.0;
            }
          }

        GET_TAYL(res + qq, k, p)
      }

      break;

    case vec_dot:
      res = tape.get_locint_r();
      size = tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();
      for (locint qq = 0; qq < size; qq++) {
        ASSIGN_A(Ares, rpp_A[res])
        ASSIGN_A(Aarg2, rpp_A[arg2 + qq])
        ASSIGN_A(Aarg1, rpp_A[arg1 + qq])
        Targ1 = rpp_T[arg1 + qq];
        Targ2 = rpp_T[arg2 + qq];
        for (int l = 0; l < p; l++) {
          if (0 == ARES) {
            HOV_INC(Aarg1, k1)
            HOV_INC(Aarg2, k1)
            HOV_INC(Ares, k1)
          } else {
            comp = (ARES > 2.0) ? ARES : 2.0;
            ARES_INC = comp;
            MAXDEC(AARG1, comp);
            MAXDEC(AARG2, comp);
            AARG1_INC_O;
            AARG2_INC_O;

            inconv(k, Ares, Targ1, Aarg2);
            inconv(k, Ares, Targ2, Aarg1);

            HOV_INC(Ares, k)
            HOV_INC(Aarg1, k)
            HOV_INC(Aarg2, k)
            HOS_OV_INC(Targ1, k)
            HOS_OV_INC(Targ2, k)
            HOS_OV_INC(Tres, k)
          }
        }
      }
      GET_TAYL(res, k, p)
      break;

    case vec_axpy:
      res = tape.get_locint_r();
      size = tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();
      arg = tape.get_locint_r();
      for (locint qq = 0; qq < size; qq++) {
        ASSIGN_A(Ares, rpp_A[res + qq])
        ASSIGN_A(Aarg, rpp_A[arg])
        ASSIGN_A(Aarg2, rpp_A[arg2 + qq])
        ASSIGN_A(Aarg1, rpp_A[arg1 + qq])
        Targ = rpp_T[arg];
        Targ1 = rpp_T[arg1 + qq];
        if (0 == ARES) {
          HOV_INC(Aarg, k1)
          HOV_INC(Aarg1, k1)
          HOV_INC(Aarg2, k1)
          HOV_INC(Ares, k1)
        } else {
          comp = (ARES > 2.0) ? ARES : 2.0;
          MAXDEC(AARG2, ARES);
          ARES_INC = 0.0;
          MAXDEC(AARG, comp);
          MAXDEC(AARG1, comp);
          AARG_INC_O;
          AARG1_INC_O;
          AARG2_INC_O;
          copyAndZeroset(k, Ares, rp_Atemp);
          inconv(k, rp_Atemp, Targ1, Aarg);
          inconv(k, rp_Atemp, Targ, Aarg1);
          for (int i = 0; i < k; i++)
            AARG2_INC += rp_Atemp[i];

          HOV_INC(Ares, k)
          HOV_INC(Aarg, k)
          HOV_INC(Aarg1, k)
          HOS_OV_INC(Targ, k)
          HOS_OV_INC(Targ1, k)
        }
        GET_TAYL(res + qq, k, p)
      }
      break;

    case ref_cond_assign: /* cond_assign */
    {
      revreal *Tref;
      locint ref = tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();
      arg = tape.get_locint_r();
      coval = tape.get_val_r();

      Tref = rpp_T[ref];

      res = (size_t)trunc(fabs(*Tref));
      GET_TAYL(res, k, p)

      ASSIGN_A(Aarg1, rpp_A[arg1])
      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg2, rpp_A[arg2])
      Targ = rpp_T[arg];

      /* olvo 980925 changed code a little bit */
      if (*Targ > 0.0) {
        if (res != arg1)
          for (int l = 0; l < p; l++) {
            if (0 == ARES) {
              HOV_INC(Ares, k1)
              HOV_INC(Aarg1, k1)
            } else {
              if (coval <= 0.0)
                MINDEC(ret_c, 2);
              MAXDEC(AARG1, ARES);
              ARES_INC = 0.0;
              AARG1_INC_O;
              for (int i = 0; i < k; i++) {
                AARG1_INC += ARES;
                ARES_INC = 0;
              }
            }
          }
        else
          for (int l = 0; l < p; l++) {
            if ((coval <= 0.0) && (ARES))
              MINDEC(ret_c, 2);
            HOV_INC(Ares, k1)
          }
      } else /* TARG <= 0.0 */
      {
        if (res != arg2)
          for (int l = 0; l < p; l++) {
            if (0 == ARES) {
              HOV_INC(Ares, k1)
              HOV_INC(Aarg2, k1)
            } else {
              if (*Targ == 0.0) /* we are at the tie */
              {
                MINDEC(ret_c, 0);
                AARG1 = 5.0;
                AARG2_INC = 5.0;
              } else {
                if (coval <= 0.0)
                  MINDEC(ret_c, 2);
                MAXDEC(AARG2, ARES);
                AARG2_INC_O;
              }
              ARES_INC = 0.0;

              for (int i = 0; i < k; i++) {
                AARG2_INC += ARES;
                ARES_INC = 0;
              }
            }
            HOV_INC(Aarg1, k1)
          }
        else
          for (int l = 0; l < p; l++) {
            if (ARES) {
              if (*Targ == 0.0) /* we are at the tie */
              {
                MINDEC(ret_c, 0);
                AARG1 = 5.0;
                AARG2 = 5.0;
              } else if (coval <= 0.0)
                MINDEC(ret_c, 2);
            }
            HOV_INC(Ares, k1)
            HOV_INC(Aarg1, k1)
            HOV_INC(Aarg2, k1)
          }
      }
    } break;
    case ref_cond_eq_assign: /* cond_eq_assign */
    {
      revreal *Tref;
      locint ref = tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();
      arg = tape.get_locint_r();
      coval = tape.get_val_r();

      Tref = rpp_T[ref];

      res = (size_t)trunc(fabs(*Tref));
      GET_TAYL(res, k, p)

      ASSIGN_A(Aarg1, rpp_A[arg1])
      ASSIGN_A(Ares, rpp_A[res])
      ASSIGN_A(Aarg2, rpp_A[arg2])
      Targ = rpp_T[arg];

      /* olvo 980925 changed code a little bit */
      if (*Targ >= 0.0) {
        if (res != arg1)
          for (int l = 0; l < p; l++) {
            if (0 == ARES) {
              HOV_INC(Ares, k1)
              HOV_INC(Aarg1, k1)
            } else {
              if (coval < 0.0)
                MINDEC(ret_c, 2);
              MAXDEC(AARG1, ARES);
              ARES_INC = 0.0;
              AARG1_INC_O;
              for (int i = 0; i < k; i++) {
                AARG1_INC += ARES;
                ARES_INC = 0;
              }
            }
          }
        else
          for (int l = 0; l < p; l++) {
            if ((coval < 0.0) && (ARES))
              MINDEC(ret_c, 2);
            HOV_INC(Ares, k1)
          }
      } else /* TARG < 0.0 */
      {
        if (res != arg2)
          for (int l = 0; l < p; l++) {
            if (0 == ARES) {
              HOV_INC(Ares, k1)
              HOV_INC(Aarg2, k1)
            } else {
              if (coval < 0.0)
                MINDEC(ret_c, 2);
              MAXDEC(AARG2, ARES);
              AARG2_INC_O;
              ARES_INC = 0.0;

              for (int i = 0; i < k; i++) {
                AARG2_INC += ARES;
                ARES_INC = 0;
              }
            }
            HOV_INC(Aarg1, k1)
          }
        else
          for (int l = 0; l < p; l++) {
            if (ARES) {
              if (coval < 0.0)
                MINDEC(ret_c, 2);
            }
            HOV_INC(Ares, k1)
            HOV_INC(Aarg1, k1)
            HOV_INC(Aarg2, k1)
          }
      }
    } break;

    case ref_cond_assign_s: /* cond_assign_s */
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();
      arg = tape.get_locint_r();
      coval = tape.get_val_r();

      Targ2 = rpp_T[arg2];
      res = (size_t)trunc(fabs(*Targ2));

      GET_TAYL(res, k, p)

      ASSIGN_A(Aarg1, rpp_A[arg1])
      ASSIGN_A(Ares, rpp_A[res])
      Targ = rpp_T[arg];

      /* olvo 980925 changed code a little bit */
      if (*Targ == 0.0) /* we are at the tie */
      {
        for (int l = 0; l < p; l++) {
          if (ARES)
            AARG1 = 5.0;
          HOV_INC(Aarg1, k1)
          HOV_INC(Ares, k1)
        }
        MINDEC(ret_c, 0);
      } else if (*Targ > 0.0) {
        if (res != arg1)
          for (int l = 0; l < p; l++) {
            if (0 == ARES) {
              HOV_INC(Ares, k1)
              HOV_INC(Aarg1, k1)
            } else {
              if (coval <= 0.0)
                MINDEC(ret_c, 2);
              MAXDEC(AARG1, ARES);
              ARES_INC = 0.0;
              AARG1_INC_O;
              for (int i = 0; i < k; i++) {
                (AARG1_INC) += ARES;
                ARES_INC = 0;
              }
            }
          }
        else
          for (int l = 0; l < p; l++) {
            if ((coval <= 0.0) && (ARES))
              MINDEC(ret_c, 2);
            HOV_INC(Ares, k1)
          }
      }
      break;

    case ref_cond_eq_assign_s: /* cond_eq_assign_s */
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();
      arg = tape.get_locint_r();
      coval = tape.get_val_r();

      Targ2 = rpp_T[arg2];
      res = (size_t)trunc(fabs(*Targ2));

      GET_TAYL(res, k, p)

      ASSIGN_A(Aarg1, rpp_A[arg1])
      ASSIGN_A(Ares, rpp_A[res])
      Targ = rpp_T[arg];

      /* olvo 980925 changed code a little bit */
      if (*Targ >= 0.0) {
        if (res != arg1)
          for (int l = 0; l < p; l++) {
            if (0 == ARES) {
              HOV_INC(Ares, k1)
              HOV_INC(Aarg1, k1)
            } else {
              if (coval < 0.0)
                MINDEC(ret_c, 2);
              MAXDEC(AARG1, ARES);
              ARES_INC = 0.0;
              AARG1_INC_O;
              for (int i = 0; i < k; i++) {
                (AARG1_INC) += ARES;
                ARES_INC = 0;
              }
            }
          }
        else
          for (int l = 0; l < p; l++) {
            if ((coval < 0.0) && (ARES))
              MINDEC(ret_c, 2);
            HOV_INC(Ares, k1)
          }
      }
      break;

      /****************************************************************************/
      /*                                                          REMAINING
       * STUFF */

      /*--------------------------------------------------------------------------*/
    case take_stock_op: /* take_stock_op */
      res = tape.get_locint_r();
      size = tape.get_locint_r();
      tape.get_val_v_r(size);

      res += size;
      for (int ls = size; ls > 0; ls--) {
        res--;

        ASSIGN_A(Ares, rpp_A[res])

        FOR_0_LE_l_LT_pk1 ARES_INC = 0.0;
      }
      break;

      /*--------------------------------------------------------------------------*/
    case death_not: /* death_not */
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();

      for (int j = arg1; j <= arg2; j++) {
        ASSIGN_A(Aarg1, rpp_A[j])

        for (int l = 0; l < p; l++)
          for (int i = 0; i < k1; i++)
            AARG1_INC = 0.0;
      }

      for (int j = arg1; j <= arg2; j++)
        GET_TAYL(j, k, p)

      break;
#ifdef _HOS_ /* HOS */

      /*--------------------------------------------------------------------------*/
    case ext_diff: /* extern differentiated function */
    {
      tape.cp_index(tape.get_locint_r());
      tape.lowestYLoc_rev(tape.get_locint_r());
      tape.lowestXLoc_rev(tape.get_locint_r());
      m = tape.get_locint_r();
      n = tape.get_locint_r();
      tape.ext_diff_fct_index(tape.get_locint_r());
      edfct = get_ext_diff_fct(tape.tapeId(), tape.ext_diff_fct_index());

      oldTraceFlag = tape.traceFlag();
      tape.traceFlag(0);

      /* degree is not known when registering external functions,
         so do memory allocation here (at least for now) */
      double **dpp_U = new double *[m];
      double **dpp_Z = new double *[n];

      if (edfct->ADOLC_EXT_FCT_POINTER == NULL)
        ADOLCError::fail(ADOLCError::ErrorType::EXT_DIFF_NULLPOINTER_FUNCTION,
                         CURRENT_LOCATION);
      if (m > 0) {
        if (ADOLC_EXT_FCT_U == NULL)
          ADOLCError::fail(ADOLCError::ErrorType::EXT_DIFF_NULLPOINTER_ARGUMENT,
                           CURRENT_LOCATION);
        if (edfct->dp_y == NULL)
          ADOLCError::fail(ADOLCError::ErrorType::EXT_DIFF_NULLPOINTER_ARGUMENT,
                           CURRENT_LOCATION);
      }
      if (n > 0) {
        if (ADOLC_EXT_FCT_Z == NULL)
          ADOLCError::fail(ADOLCError::ErrorType::EXT_DIFF_NULLPOINTER_ARGUMENT,
                           CURRENT_LOCATION);
        if (edfct->dp_x == NULL)
          ADOLCError::fail(ADOLCError::ErrorType::EXT_DIFF_NULLPOINTER_ARGUMENT,
                           CURRENT_LOCATION);
      }
      arg = tape.lowestYLoc_rev() + m - 1;
      for (int loop = 0; loop < m; ++loop) {
        // First entry of rpp_A[arg] is algorithmic dependency --> skip that!
        dpp_U[loop] = rpp_A[arg] + 1;
        ++arg;
      }

      arg = tape.lowestXLoc_rev();
      for (int loop = 0; loop < n; ++loop) {
        // This should copy data in case `revreal` is not double.
        // (Note: copy back below doesn't actually do anything until this is
        // changed to a copy.) (Note: first entry is alg. dependency which we
        // just skip here for now.)
        dpp_Z[loop] = rpp_A[arg] + 1;
        ++arg;
      }
      arg = tape.lowestXLoc_rev();
      double **dpp_x = rpp_T + arg; // TODO: change to copy, use loop below
      for (int loop = 0; loop < n; ++loop, ++arg) {
        // TODO: copy rpp_T[arg][0,...,keep] -> dpp_x[loop][0,...,keep]
        // edfct->dp_x[loop] = rpp_T[arg];
      }
      arg = tape.lowestYLoc_rev();
      double **dpp_y = rpp_T + arg; // TODO: change to copy, use loop below
      for (int loop = 0; loop < m; ++loop, ++arg) {
        // TODO: copy rpp_T[arg][0,...,keep] -> dpp_y[loop][0,...,keep]
        // edfct->dp_y[loop] = rpp_T[arg];
      }
      int ext_retc = edfct->ADOLC_EXT_FCT_COMPLETE;
      MINDEC(ret_c, ext_retc);

      res = tape.lowestYLoc_rev();
      // Ares = A[res];
      for (int loop = 0; loop < m; ++loop) {
        for (int l = 0; l < q; ++l) {
          // ADJOINT_BUFFER_RES_L = 0.; /* \bar{v}_i = 0 !!! */
          // rpp_T[res][l] = 0.0;
          rpp_A[res][l] = 0.0;
        }
        ++res;
      }
      res = tape.lowestXLoc_rev();
      for (int loop = 0; loop < n; ++loop) {
        // ADOLC_EXT_FCT_COPY_ADJOINTS_BACK(ADOLC_EXT_FCT_Z[loop],ADJOINT_BUFFER_RES);
        // Hmm, ist das nicht falsch? Wir sollten rpp_T vermutlich nicht
        // anfassen. Sonst ändert sich ja das Ergebnis wenn man das Band
        // nochmal abspielt? rpp_T[res] = dpp_Z[loop]; Assume non-smooth?
        rpp_A[res][0] = 5.0;
        for (int i = 0; i < k; i++) {
          // `i+1` to do something similar a `ARES_INC_O`
          // I still don't understand this ARES_INC_O,
          // but apparently the first place is used to store
          // some flag or so I don't understand (see me wondering
          // in mult_a_a...)
          rpp_A[res][i + 1] = dpp_Z[loop][i];
        }
        ++res;
      }
      if (edfct->dp_y_priorRequired) {
        arg = tape.lowestYLoc_rev() + m - 1;
        for (int loop = 0; loop < m; ++loop, --arg) {
          // ADOLC_GET_TAYLOR(arg);
          GET_TAYL(arg, k, p);
        }
      }
      if (edfct->dp_x_changes) {
        arg = tape.lowestXLoc_rev() + n - 1;
        for (int loop = 0; loop < n; ++loop, --arg) {
          // ADOLC_GET_TAYLOR(arg);
          GET_TAYL(arg, k, p);
        }
      }
      tape.traceFlag(oldTraceFlag);

      delete[] dpp_Z;
      delete[] dpp_U;

      break;
    }
#endif
      /*--------------------------------------------------------------------------*/
    default: /* default */
      /*             Die here, we screwed up     */
      ADOLCError::fail(ADOLCError::ErrorType::NO_SUCH_OP, CURRENT_LOCATION,
                       ADOLCError::FailInfo{.info7 = operation});
      break;
    } /* endswitch */

    /* Get the next operation */
    operation = tape.get_op_r();
#if defined(ADOLC_DEBUG)
    ++countPerOperation[operation];
#endif /* ADOLC_DEBUG */
  }

#if defined(ADOLC_DEBUG)
  printf("\nTape contains:\n");
  for (v = 0; v < 256; ++v)
    if (countPerOperation[v] > 0)
      printf("operation %3d: %6d time(s) - %6d taylors read (%10.2f per "
             "operation)\n",
             v, countPerOperation[v], taylorPerOperation[v],
             (double)taylorPerOperation[v] / (double)countPerOperation[v]);
  printf("\n");
#endif /* ADOLC_DEBUG */

  /* clean up */
  myfree2(rpp_T);
  tape.rpp_T(nullptr);

  myfree2(rpp_A);
  tape.rpp_A(nullptr);

  myfree1(rp_Ttemp);
  myfree1(rp_Ttemp2);
  myfree1(rp_Atemp);
  myfree1(rp_Atemp2);

  myfree1_ulong(jj);
  myfree1(x);

  tape.workMode(TapeInfos::ADOLC_NO_MODE);
  tape.end_sweep();

  return ret_c;
}

/****************************************************************************/
/*                                                               THAT'S ALL */

END_C_DECLS

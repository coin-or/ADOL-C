/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     fo_rev.cpp
 Revision: $Id$
 Contents: Contains the routines :
           fos_reverse (first-order-scalar reverse mode)  : define _FOS_
           fov_reverse (first-order-vector reverse mode)  : define _FOV_
           int_reverse_tight,
                ( first-order-vector reverse mode for bit patterns,
                  checks all dependences on taylors and real values,
                  more precize)
           int_reverse_safe,
                ( first-order-vector reverse mode for bit patterns,
                  return always 3,
                  no dependences on taylors and real values,
                  faster than tight)

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

double*** full family of         ------------          weigth matrix x
results   Taylor-Jacobians       ------------          Taylor Jacobians

*****************************************************************************/

/****************************************************************************/
/*                                                                   MACROS */
#undef _ADOLC_VECTOR_

/*--------------------------------------------------------------------------*/
#ifdef _FOS_
#ifdef _ABS_NORM_
#define GENERATED_FILENAME "fos_pl_reverse"
#else
#ifdef _ABS_NORM_SIG_
#define GENERATED_FILENAME "fos_pl_sig_reverse"
#else
#define GENERATED_FILENAME "fos_reverse"
#endif
#endif

#define RESULTS(l, indexi) results[indexi]
#define LAGRANGE(l, indexd) lagrange[indexd]
#define RESULTSTRANS(l, indexi) results[indexi]
#define LAGRANGETRANS(l, indexd) lagrange[indexd]

/*--------------------------------------------------------------------------*/
#elif _FOV_
#define GENERATED_FILENAME "fov_reverse"

#define _ADOLC_VECTOR_

#define RESULTS(l, indexi) results[l][indexi]
#define LAGRANGE(l, indexd) lagrange[l][indexd]
#define RESULTSTRANS(l, indexi) results[indexi][l]
#define LAGRANGETRANS(l, indexd) lagrange[indexd][l]

#else
#if defined(_INT_REV_)
#if defined(_TIGHT_)
#define GENERATED_FILENAME "int_reverse_t"
#endif
#if defined(_NTIGHT_)
#define GENERATED_FILENAME "int_reverse_s"
#endif
#define RESULTS(l, indexi) results[l][indexi]
#define LAGRANGE(l, indexd) lagrange[l][indexd]
#define RESULTSTRANS(l, indexi) results[indexi][l]
#define LAGRANGETRANS(l, indexd) lagrange[indexd][l]
#else
#error Error ! Define [_FOS_ | _FOV_ | _INT_REV_SAFE_ | _INT_REV_TIGHT_ ]
#endif
#endif
/*--------------------------------------------------------------------------*/
/*                                                     access to variables  */

#ifdef _FOS_
#define AARG *Aarg
#define AARG1 *Aarg1
#define AARG2 *Aarg2

#define ARES_INC *Ares
#define AARG_INC *Aarg
#define AARG1_INC *Aarg1
#define AARG2_INC *Aarg2

#define ARES_INC_O Ares
#define AARG_INC_O / adAarg
#define AARG1_INC_O Aarg1
#define AARG2_INC_O Aarg2

#define ASSIGN_A(a, b) a = &b;

#else /* _FOV_ */
#ifdef _FOV_
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
#else
#ifdef _INT_REV_
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
#endif
#endif

#define TRES rp_T[res]
#define TARG rp_T[arg]
#define TARG1 rp_T[arg1]
#define TARG2 rp_T[arg2]

/*--------------------------------------------------------------------------*/
/*                                                              loop stuff  */
#ifdef _ADOLC_VECTOR_
#define FOR_0_LE_l_LT_p for (int l = 0; l < p; l++)
#else
#ifdef _INT_REV_
#define FOR_0_LE_l_LT_p for (int l = 0; l < p; l++) // Apparently not used
#else
#define FOR_0_LE_l_LT_p
#endif
#endif

/* END Macros */

/****************************************************************************/
/*                                                       NECESSARY INCLUDES */
#include <adolc/adalloc.h>
#include <adolc/adolcerror.h>
#include <adolc/dvlparms.h>
#include <adolc/externfcts.h>
#include <adolc/interfaces.h>
#include <adolc/oplate.h>
#include <adolc/tape_interface.h>
#include <adolc/valuetape/valuetape.h>
#include <math.h>
#include <string.h>

#ifdef ADOLC_MEDIPACK_SUPPORT
#include <adolc/medipacksupport_p.h>
#endif
#ifdef ADOLC_AMPI_SUPPORT
#include "ampi/ampi.h"
#include "ampi/libCommon/modified.h"
#endif

BEGIN_C_DECLS

/****************************************************************************/
/*                                                             NOW THE CODE */

#ifdef _FOS_
/****************************************************************************/
/* First-Order Scalar Reverse Pass.                                         */
/****************************************************************************/
#ifdef _ABS_NORM_
/****************************************************************************/
/* Abs-Normal extended adjoint row computation.                             */
/****************************************************************************/
int fos_pl_reverse(short tnum,      /* tape id */
                   int depen,       /* consistency chk on # of deps */
                   int indep,       /* consistency chk on # of indeps */
                   int swchk,       /* consistency chk on # of switches */
                   int rownum,      /* required row no. of abs-normal form */
                   double *results) /*  coefficient vectors */
#elif defined(_ABS_NORM_SIG_)
/****************************************************************************/
/* Abs-Normal extended adjoint row computation.                             */
/****************************************************************************/
int fos_pl_sig_reverse(short tnum, /* tape id */
                       int depen,  /* consistency chk on # of deps */
                       int indep,  /* consistency chk on # of indeps */
                       int swchk,  /* consistency chk on # of switches */
                       const short *siggrad, const double *lagrange,
                       double *results) /*  coefficient vectors */
#else
int fos_reverse(short tnum, /* tape id */
                int depen,  /* consistency chk on # of deps */
                int indep,  /* consistency chk on # of indeps */
                const double *lagrange,
                double *results) /*  coefficient vectors */

#endif
#elif _FOV_
/****************************************************************************/
/* First-Order Vector Reverse Pass.                                         */
/****************************************************************************/

int fov_reverse(short tnum, /* tape id */
                int depen,  /* consistency chk on # of deps */
                int indep,  /* consistency chk on # of indeps */
                int nrows,  /* # of Jacobian rows being calculated */
                const double *const *lagrange, /* domain weight vector */
                double **results) /* matrix of coefficient vectors */

#elif defined(_INT_REV_)
#if defined(_TIGHT_)
/****************************************************************************/
/* First Order Vector version of the reverse mode for bit patterns, tight   */
/****************************************************************************/
int int_reverse_tight(
    short tnum,                    /* tape id                               */
    int depen,                     /* consistency chk on # of deps          */
    int indep,                     /* consistency chk on # of indeps        */
    int nrows,                     /* # of Jacobian rows being calculated   */
    const size_t *const *lagrange, /* domain weight vector[var][row](in)*/
    size_t **results)              /* matrix of coeff. vectors[var][row]*/

#elif defined(_NTIGHT_)
/****************************************************************************/
/* First Order Vector version of the reverse mode, bit pattern, safe        */
/****************************************************************************/
int int_reverse_safe(
    short tnum,                    /* tape id                               */
    int depen,                     /* consistency chk on # of deps          */
    int indep,                     /* consistency chk on # of indeps        */
    int nrows,                     /* # of Jacobian rows being calculated   */
    const size_t *const *lagrange, /* domain weight vector[var][row](in)*/
    size_t **results)              /* matrix of coeff. vectors[var][row]*/
#else
#error Neither _TIGHT_ nor _NTIGHT_ defined
#endif
#endif
{
  ValueTape &tape = findTape(tnum);
  /****************************************************************************/
  /*                                                           ALL VARIABLES  */
  unsigned char operation; /* operation code */
  int ret_c = 3;           /* return value */

  locint size = 0;
  locint res = 0;
  locint arg = 0;
  locint arg1 = 0;
  locint arg2 = 0;

#if !defined(_NTIGHT_)
  double coval = 0;
#endif

  int indexi = 0, indexd = 0;
#if defined(_ABS_NORM_) || defined(_ABS_NORM_SIG_)
  int switchnum;
#endif

  /* loop indices */
  int j, ls;

  /* other necessary variables */
#if !defined(_NTIGHT_)
  double r0, r_0;
  int taycheck;
  int numdep, numind;
#endif

  /*--------------------------------------------------------------------------*/
  /* Adjoint stuff */
#ifdef _FOS_
  double *rp_A = nullptr;
#endif
#ifdef _FOV_
  double **rpp_A = nullptr;
#endif
#if !defined(_NTIGHT_)
  double *rp_T = nullptr;
#endif /* !_NTIGHT_ */
#if !defined _INT_REV_
  double *Ares = nullptr;
  double *Aarg = nullptr;
  double *Aarg1 = nullptr;
  double *Aarg2 = nullptr;
#else
  size_t **upp_A = nullptr;
  size_t *Ares = nullptr;
  size_t *Aarg = nullptr;
  size_t *Aarg1 = nullptr;
  size_t *Aarg2 = nullptr;
#endif

  /*--------------------------------------------------------------------------*/

#ifdef _ADOLC_VECTOR_
  int p = nrows;
#endif
#ifdef _INT_REV_
  int p = nrows;
#endif

  /****************************************************************************/
  /*                                          extern diff. function variables */
#if defined(_FOS_)
#define ADOLC_EXT_FCT_U edfct->dp_U
#define ADOLC_EXT_FCT_Z edfct->dp_Z
#define ADOLC_EXT_FCT_POINTER fos_reverse
#define ADOLC_EXT_FCT_IARR_POINTER fos_reverse_iArr
#define ADOLC_EXT_FCT_COMPLETE                                                 \
  fos_reverse(edfct->tapeId, m, edfct->dp_U, n, edfct->dp_Z, edfct->dp_x,      \
              edfct->dp_y)
#define ADOLC_EXT_FCT_IARR_COMPLETE                                            \
  fos_reverse_iArr(edfct->tapeId, iArrLength, iArr, m, edfct->dp_U, n,         \
                   edfct->dp_Z, edfct->dp_x, edfct->dp_y)
#define ADOLC_EXT_FCT_SAVE_NUMDIRS
#define ADOLC_EXT_FCT_V2_U edfct2->up
#define ADOLC_EXT_FCT_V2_Z edfct2->zp
#define ADOLC_EXT_FCT_V2_COMPLETE                                              \
  fos_reverse(edfct->tapeId, iArrLength, iArr, nout, nin, (int *)outsz,        \
              edfct2->up, (int *)insz, edfct2->zp, edfct2->x, edfct2->y,       \
              edfct2->context)
#else
#define ADOLC_EXT_FCT_U edfct->dpp_U
#define ADOLC_EXT_FCT_Z edfct->dpp_Z
#define ADOLC_EXT_FCT_POINTER fov_reverse
#define ADOLC_EXT_FCT_IARR_POINTER fov_reverse_iArr
#define ADOLC_EXT_FCT_COMPLETE                                                 \
  fov_reverse(edfct->tapeId, m, p, edfct->dpp_U, n, edfct->dpp_Z, edfct->dp_x, \
              edfct->dp_y)
#define ADOLC_EXT_FCT_IARR_COMPLETE                                            \
  fov_reverse_iArr(edfct->tapeId, iArrLength, iArr, m, p, edfct->dpp_U, n,     \
                   edfct->dpp_Z, edfct->dp_x, edfct->dp_y)
#define ADOLC_EXT_FCT_SAVE_NUMDIRS tape.numDirs_rev(nrows)
#define ADOLC_EXT_FCT_V2_U edfct2->Up
#define ADOLC_EXT_FCT_V2_Z edfct2->Zp
#define ADOLC_EXT_FCT_V2_COMPLETE                                              \
  fov_reverse(edfct->tapeId, iArrLength, iArr, nout, nin, (int *)outsz, p,     \
              edfct2->Up, (int *)insz, edfct2->Zp, edfct2->x, edfct2->y,       \
              edfct2->context)
#endif
#if !defined(_INT_REV_)
  locint n, m;
  ext_diff_fct *edfct = nullptr;
  ext_diff_fct_v2 *edfct2 = nullptr;
  int iArrLength;
  int *iArr = nullptr;
  int loop, oloop;
  int ext_retc;
  int oldTraceFlag;
  locint *insz = nullptr;
  locint *outsz = nullptr;
  locint nin, nout;
#endif
#ifdef ADOLC_AMPI_SUPPORT
  MPI_Op op;
  void *buf = nullptr;
  void *rbuf = nullptr;
  int count, rcount;
  MPI_Datatype datatype, rtype;
  int src;
  int tag;
  enum AMPI_PairedWith_E pairedWith;
  MPI_Comm comm;
  MPI_Status *status = nullptr;
  struct AMPI_Request_S request;
#endif

#if defined(ADOLC_DEBUG)
  /****************************************************************************/
  /*                                                           DEBUG MESSAGES */
  fprintf(DIAG_OUT, "Call of %s(..) with tag: %d, n: %d, m %d,\n",
          GENERATED_FILENAME, tnum, indep, depen);
#ifdef _ADOLC_VECTOR_
  fprintf(DIAG_OUT, "                    p: %d\n\n", nrows);
#endif

#endif

  /****************************************************************************/
  /*                                                                    INITs */

  /*------------------------------------------------------------------------*/
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

#if defined(_ABS_NORM_) || defined(_ABS_NORM_SIG_)
  if (!tape.tapestats(TapeInfos::NO_MIN_MAX))
    ADOLCError::fail(ADOLCError::ErrorType::NO_MINMAX, CURRENT_LOCATION,
                     ADOLCError::FailInfo{.info1 = tnum});
  else if (swchk != tape.tapestats(TapeInfos::NUM_SWITCHES))
    ADOLCError::fail(
        ADOLCError::ErrorType::SWITCHES_MISMATCH, CURRENT_LOCATION,
        ADOLCError::FailInfo{.info1 = tnum,
                             .info3 = swchk,
                             .info6 = tape.tapestats(TapeInfos::NUM_SWITCHES)});
  else
    switchnum = swchk - 1;
#endif

  /****************************************************************************/
  /*                                                  MEMORY ALLOCATION STUFF
   */

  /*--------------------------------------------------------------------------*/
#ifdef _FOS_ /* FOS */
  rp_A = myalloc1(tape.tapestats(TapeInfos::NUM_MAX_LIVES));
  tape.rp_A(rp_A);
  rp_T = myalloc1(tape.tapestats(TapeInfos::NUM_MAX_LIVES));
  tape.workMode(TapeInfos::FOS_REVERSE);
#ifdef _ABS_NORM_
  memset(results, 0, sizeof(double) * (indep + swchk));
#endif
#define ADJOINT_BUFFER rp_A
#define ADJOINT_BUFFER_ARG_L rp_A[arg]
#define ADJOINT_BUFFER_RES_L rp_A[res]
#define ADJOINT_BUFFER_ARG rp_A[arg]
#define ADJOINT_BUFFER_RES rp_A[res]
#define ADOLC_EXT_FCT_U_L_LOOP edfct->dp_U[loop]
#define ADOLC_EXT_FCT_Z_L_LOOP edfct->dp_Z[loop]
#define ADOLC_EXT_FCT_V2_U_LOOP edfct2->up[oloop][loop]
#define ADOLC_EXT_FCT_V2_Z_LOOP edfct2->zp[oloop][loop]
#define ADOLC_EXT_FCT_COPY_ADJOINTS(dest, src) dest = src
#define ADOLC_EXT_FCT_COPY_ADJOINTS_BACK(dest, src) src = dest

  /*--------------------------------------------------------------------------*/
#else
#if defined _FOV_ /* FOV */
  rpp_A = myalloc2(tape.tapestats(TapeInfos::NUM_MAX_LIVES), p);
  tape.rpp_A(rpp_A);
  rp_T = myalloc1(tape.tapestats(TapeInfos::NUM_MAX_LIVES));
  tape.workMode(TapeInfos::FOV_REVERSE);
#define ADJOINT_BUFFER rpp_A
#define ADJOINT_BUFFER_ARG_L rpp_A[arg][l]
#define ADJOINT_BUFFER_RES_L rpp_A[res][l]
#define ADJOINT_BUFFER_ARG rpp_A[arg]
#define ADJOINT_BUFFER_RES rpp_A[res]
#define ADOLC_EXT_FCT_U_L_LOOP edfct->dpp_U[l][loop]
#define ADOLC_EXT_FCT_Z_L_LOOP edfct->dpp_Z[l][loop]
#define ADOLC_EXT_FCT_V2_U_LOOP edfct2->Up[oloop][loop]
#define ADOLC_EXT_FCT_V2_Z_LOOP edfct2->Zp[oloop][loop]
#define ADOLC_EXT_FCT_COPY_ADJOINTS(dest, src) dest = src
#define ADOLC_EXT_FCT_COPY_ADJOINTS_BACK(dest, src)
#else
#if defined _INT_REV_
  upp_A = myalloc2_ulong(tape.tapestats(TapeInfos::NUM_MAX_LIVES), p);
#if defined _TIGHT_
  tape.upp_A(upp_A);
  rp_T = myalloc1(tape.tapestats(TapeInfos::NUM_MAX_LIVES));
#endif
#define ADJOINT_BUFFER upp_A
#define ADJOINT_BUFFER_ARG_L upp_A[arg][l]
#define ADJOINT_BUFFER_RES_L upp_A[res][l]
#endif
#endif
#endif

  /****************************************************************************/
  /*                                                    TAYLOR INITIALIZATION */

#if !defined(_NTIGHT_)

  tape.rp_T(rp_T);

  tape.taylor_back(tnum, &numdep, &numind, &taycheck);

  if (taycheck < 0)
    ADOLCError::fail(ADOLCError::ErrorType::REVERSE_NO_FOWARD, CURRENT_LOCATION,
                     ADOLCError::FailInfo{.info3 = 0, .info4 = 1});

  if ((numdep != depen) || (numind != indep))
    ADOLCError::fail(ADOLCError::ErrorType::REVERSE_TAYLOR_COUNTS_MISMATCH,
                     CURRENT_LOCATION,
                     ADOLCError::FailInfo{.info1 = tape.tapeId()});

#endif /* !_NTIGHT_ */

  /****************************************************************************/
  /*                                                            REVERSE SWEEP */
  operation = tape.get_op_r();
  while (operation != start_of_tape) { /* Switch statement to execute the
                                          operations in Reverse */

    switch (operation) {

      /****************************************************************************/
      /*                                                                  MARKERS
       */

      /*--------------------------------------------------------------------------*/
    case end_of_op: /* end_of_op */
      tape.get_op_block_r();
      operation = tape.get_op_r();
      /* Skip next operation, it's another end_of_op */
      break;

      /*--------------------------------------------------------------------------*/
    case end_of_int:          /* end_of_int */
      tape.get_loc_block_r(); /* Get the next int block */
      break;

      /*--------------------------------------------------------------------------*/
    case end_of_val:          /* end_of_val */
      tape.get_val_block_r(); /* Get the next val block */
      break;

      /*--------------------------------------------------------------------------*/
    case start_of_tape: /* start_of_tape */
      break;
    case end_of_tape: /* end_of_tape */
      tape.discard_params_r();
      break;

      /****************************************************************************/
      /*                                                               COMPARISON
       */

      /*--------------------------------------------------------------------------*/
    case eq_zero: /* eq_zero */
      arg = tape.get_locint_r();

#if !defined(_NTIGHT_)
      ret_c = 0;
#endif /* !_NTIGHT_ */
      break;

      /*--------------------------------------------------------------------------*/
    case neq_zero: /* neq_zero */
    case gt_zero:  /* gt_zero */
    case lt_zero:  /* lt_zero */
      arg = tape.get_locint_r();
      break;

      /*--------------------------------------------------------------------------*/
    case ge_zero: /* ge_zero */
    case le_zero: /* le_zero */
      arg = tape.get_locint_r();

#if !defined(_NTIGHT_)
      if (TARG == 0)
        ret_c = 0;
#endif /* !_NTIGHT_ */
      break;

      /****************************************************************************/
      /*                                                              ASSIGNMENTS
       */

      /*--------------------------------------------------------------------------*/
    case assign_a: /* assign an adouble variable an    assign_a */
      /* adouble value. (=) */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();

      ASSIGN_A(Aarg, ADJOINT_BUFFER[arg])
      ASSIGN_A(Ares, ADJOINT_BUFFER[res])

      FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
        AARG_INC |= *Ares;
        ARES_INC = 0;
#else
        AARG_INC += *Ares;
        ARES_INC = 0.0;
#endif
      }

#if !defined(_NTIGHT_)
      tape.get_taylor(res);
#endif /* !_NTIGHT_ */
      break;

      /*--------------------------------------------------------------------------*/
    case assign_d: /* assign an adouble variable a    assign_d */
      /* double value. (=) */
      res = tape.get_locint_r();
#if !defined(_NTIGHT_)
      coval = tape.get_val_r();
#endif /* !_NTIGHT_ */

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])

      FOR_0_LE_l_LT_p
#if defined(_INT_REV_)
          ARES_INC = 0;
#else
          ARES_INC = 0.0;
#endif

#if !defined(_NTIGHT_)
      tape.get_taylor(res);
#endif /* !_NTIGHT_ */
      break;

    case assign_p: /* assign an adouble variable a    assign_d */
      /* double value. (=) */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();
#if !defined(_NTIGHT_)
      coval = tape.paramstore()[arg];
#endif /* !_NTIGHT_ */

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])

      FOR_0_LE_l_LT_p
#if defined(_INT_REV_)
          ARES_INC = 0;
#else
          ARES_INC = 0.0;
#endif

#if !defined(_NTIGHT_)
      tape.get_taylor(res);
#endif /* !_NTIGHT_ */
      break;
      /*--------------------------------------------------------------------------*/
    case assign_d_zero: /* assign an adouble variable a    assign_d_zero */
    case assign_d_one:  /* double value (0 or 1). (=)       assign_d_one */
      res = tape.get_locint_r();

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])

      FOR_0_LE_l_LT_p
#if defined(_INT_REV_)
          ARES_INC = 0;
#else
          ARES_INC = 0.0;
#endif

#if !defined(_NTIGHT_)
      tape.get_taylor(res);
#endif /* !_NTIGHT_ */
      break;

      /*--------------------------------------------------------------------------*/
    case assign_ind: /* assign an adouble variable an    assign_ind */
      /* independent double value (<<=) */
      res = tape.get_locint_r();

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])

      if (tape.in_nested_ctx()) {
        FOR_0_LE_l_LT_p RESULTSTRANS(l, indexi) += ARES_INC;
      } else {
        FOR_0_LE_l_LT_p RESULTS(l, indexi) = ARES_INC;
      }
#if !defined(_NTIGHT_)
      tape.get_taylor(res);
#endif /* !_NTIGHT_ */
      indexi--;
      break;

      /*--------------------------------------------------------------------------*/
    case assign_dep: /* assign a float variable a    assign_dep */
      /* dependent adouble value. (>>=) */
      res = tape.get_locint_r();

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])

#if defined(_ABS_NORM_)
      if (indexd + swchk == rownum)
        *Ares = 1.0;
      else
        *Ares = 0.0;
#else
      if (tape.in_nested_ctx()) {
        FOR_0_LE_l_LT_p { ARES_INC = LAGRANGETRANS(l, indexd); }
      } else {
        FOR_0_LE_l_LT_p ARES_INC = LAGRANGE(l, indexd);
      }
#endif
      indexd--;
      break;

      /****************************************************************************/
      /*                                                   OPERATION +
       * ASSIGNMENT */

      /*--------------------------------------------------------------------------*/
    case eq_plus_d: /* Add a floating point to an    eq_plus_d */
      /* adouble. (+=) */
      res = tape.get_locint_r();
#if !defined(_NTIGHT_)
      coval = tape.get_val_r();

      tape.get_taylor(res);
#endif /* !_NTIGHT_ */
      break;

      /*--------------------------------------------------------------------------*/
    case eq_plus_a: /* Add an adouble to another    eq_plus_a */
      /* adouble. (+=) */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg, ADJOINT_BUFFER[arg]);

      FOR_0_LE_l_LT_p
#if defined(_INT_REV_)
          AARG_INC |= ARES_INC;
#else
          AARG_INC += ARES_INC;
#endif

#if !defined(_NTIGHT_)
      tape.get_taylor(res);
#endif /* !_NTIGHT_ */
      break;

      /*--------------------------------------------------------------------------*/
    case eq_min_d: /* Subtract a floating point from an    eq_min_d */
      /* adouble. (-=) */
      res = tape.get_locint_r();
#if !defined(_NTIGHT_)
      coval = tape.get_val_r();

      tape.get_taylor(res);
#endif /* !_NTIGHT_ */
      break;

      /*--------------------------------------------------------------------------*/
    case eq_min_a: /* Subtract an adouble from another    eq_min_a */
      /* adouble. (-=) */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg, ADJOINT_BUFFER[arg])

      FOR_0_LE_l_LT_p
#if defined(_INT_REV_)
          AARG_INC |= ARES_INC;
#else
          AARG_INC -= ARES_INC;
#endif

#if !defined(_NTIGHT_)
      tape.get_taylor(res);
#endif /* !_NTIGHT_ */
      break;

      /*--------------------------------------------------------------------------*/
    case eq_mult_d: /* Multiply an adouble by a    eq_mult_d */
      /* flaoting point. (*=) */
      res = tape.get_locint_r();
#if !defined(_NTIGHT_)
      coval = tape.get_val_r();
#endif /* !_NTIGHT_ */

#if !defined(_INT_REV_)
      ASSIGN_A(Ares, ADJOINT_BUFFER[res])

      FOR_0_LE_l_LT_p ARES_INC *= coval;
#endif

#if !defined(_NTIGHT_)
      tape.get_taylor(res);
#endif /* !_NTIGHT_ */
      break;

      /*--------------------------------------------------------------------------*/
    case eq_mult_a: /* Multiply one adouble by another    eq_mult_a */
      /* (*=) */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();

#if !defined(_NTIGHT_)
      tape.get_taylor(res);
#endif /* !_NTIGHT_ */

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg, ADJOINT_BUFFER[arg])

      FOR_0_LE_l_LT_p
#if defined(_INT_REV_)
          AARG_INC |= ARES_INC;
#else
      {
        revreal aTmp = *Ares;
        ARES_INC = aTmp * TARG;
        AARG_INC += aTmp * TRES;
      }
#endif
      break;

      /*--------------------------------------------------------------------------*/
    case incr_a: /* Increment an adouble    incr_a */
    case decr_a: /* Increment an adouble    decr_a */
      res = tape.get_locint_r();

#if !defined(_NTIGHT_)
      tape.get_taylor(res);
#endif /* !_NTIGHT_ */
      break;

      /****************************************************************************/
      /*                                                        BINARY
       * OPERATIONS */

      /*--------------------------------------------------------------------------*/
    case plus_a_a: /* : Add two adoubles. (+)    plus a_a */
      res = tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg1, ADJOINT_BUFFER[arg1])
      ASSIGN_A(Aarg2, ADJOINT_BUFFER[arg2])

      FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
        size_t aTmp = *Ares;
        ARES_INC = 0;
        AARG1_INC |= aTmp;
        AARG2_INC |= aTmp;
#else
        revreal aTmp = *Ares;
        ARES_INC = 0.0;
        AARG1_INC += aTmp;
        AARG2_INC += aTmp;
#endif
      }

#if !defined(_NTIGHT_)
      tape.get_taylor(res);
#endif /* !_NTIGHT_ */
      break;

      /*--------------------------------------------------------------------------*/
    case plus_d_a: /* Add an adouble and a double    plus_d_a */
      /* (+) */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();
#if !defined(_NTIGHT_)
      coval = tape.get_val_r();
#endif /* !_NTIGHT_ */

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg, ADJOINT_BUFFER[arg])

      FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
        size_t aTmp = *Ares;
        ARES_INC = 0;
        AARG_INC |= aTmp;
#else
        revreal aTmp = *Ares;
        ARES_INC = 0.0;
        AARG_INC += aTmp;
#endif
      }

#if !defined(_NTIGHT_)
      tape.get_taylor(res);
#endif /* !_NTIGHT_ */
      break;

      /*--------------------------------------------------------------------------*/
    case min_a_a: /* Subtraction of two adoubles    min_a_a */
      /* (-) */
      res = tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg1, ADJOINT_BUFFER[arg1])
      ASSIGN_A(Aarg2, ADJOINT_BUFFER[arg2])

      FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
        size_t aTmp = *Ares;
        ARES_INC = 0;
        AARG1_INC |= aTmp;
        AARG2_INC |= aTmp;
#else
        revreal aTmp = *Ares;
        ARES_INC = 0.0;
        AARG1_INC += aTmp;
        AARG2_INC -= aTmp;
#endif
      }

#if !defined(_NTIGHT_)
      tape.get_taylor(res);
#endif /* !_NTIGHT_ */
      break;

      /*--------------------------------------------------------------------------*/
    case min_d_a: /* Subtract an adouble from a    min_d_a */
      /* double (-) */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();
#if !defined(_NTIGHT_)
      coval = tape.get_val_r();
#endif /* !_NTIGHT_ */

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg, ADJOINT_BUFFER[arg])

      FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
        size_t aTmp = *Ares;
        ARES_INC = 0;
        AARG_INC |= aTmp;
#else
        revreal aTmp = *Ares;
        ARES_INC = 0.0;
        AARG_INC -= aTmp;
#endif
      }

#if !defined(_NTIGHT_)
      tape.get_taylor(res);
#endif /* !_NTIGHT_ */
      break;

      /*--------------------------------------------------------------------------*/
    case mult_a_a: /* Multiply two adoubles (*)    mult_a_a */
      res = tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();

#if !defined(_NTIGHT_)
      tape.get_taylor(res);
#endif /* !_NTIGHT_ */

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg2, ADJOINT_BUFFER[arg2])
      ASSIGN_A(Aarg1, ADJOINT_BUFFER[arg1])

      FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
        size_t aTmp = *Ares;
        ARES_INC = 0;
        AARG2_INC |= aTmp;
        AARG1_INC |= aTmp;
#else
        revreal aTmp = *Ares;
        ARES_INC = 0.0;
        AARG2_INC += aTmp * TARG1;
        AARG1_INC += aTmp * TARG2;
#endif
      }
      break;

      /*--------------------------------------------------------------------------*/
      /* olvo 991122: new op_code with recomputation */
    case eq_plus_prod: /* increment a product of           eq_plus_prod */
      /* two adoubles (*) */
      res = tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg2, ADJOINT_BUFFER[arg2])
      ASSIGN_A(Aarg1, ADJOINT_BUFFER[arg1])

#if !defined(_NTIGHT_)
      /* RECOMPUTATION */
      TRES -= TARG1 * TARG2;
#endif /* !_NTIGHT_ */

      FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
        AARG2_INC |= *Ares;
        AARG1_INC |= ARES_INC;
#else
        AARG2_INC += (*Ares) * TARG1;
        AARG1_INC += ARES_INC * TARG2;
#endif
      }
      break;

      /*--------------------------------------------------------------------------*/
      /* olvo 991122: new op_code with recomputation */
    case eq_min_prod: /* decrement a product of            eq_min_prod */
      /* two adoubles (*) */
      res = tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg2, ADJOINT_BUFFER[arg2])
      ASSIGN_A(Aarg1, ADJOINT_BUFFER[arg1])

#if !defined(_NTIGHT_)
      /* RECOMPUTATION */
      TRES += TARG1 * TARG2;
#endif /* !_NTIGHT_ */

      FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
        AARG2_INC |= *Ares;
        AARG1_INC |= ARES_INC;
#else
        AARG2_INC -= (*Ares) * TARG1;
        AARG1_INC -= ARES_INC * TARG2;
#endif
      }
      break;

      /*--------------------------------------------------------------------------*/
    case mult_d_a: /* Multiply an adouble by a double    mult_d_a */
      /* (*) */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();
#if !defined(_NTIGHT_)
      coval = tape.get_val_r();
#endif /* !_NTIGHT_ */

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg, ADJOINT_BUFFER[arg])

      FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
        size_t aTmp = *Ares;
        ARES_INC = 0;
        AARG_INC |= aTmp;
#else
        revreal aTmp = *Ares;
        ARES_INC = 0.0;
        AARG_INC += coval * aTmp;
#endif
      }

#if !defined(_NTIGHT_)
      tape.get_taylor(res);
#endif /* !_NTIGHT_ */
      break;

      /*--------------------------------------------------------------------------*/
    case div_a_a: /* Divide an adouble by an adouble    div_a_a */
      /* (/) */
      res = tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg2, ADJOINT_BUFFER[arg2])
      ASSIGN_A(Aarg1, ADJOINT_BUFFER[arg1])

      /* olvo 980922 changed order to allow x=y/x */
#if !defined(_NTIGHT_)
      r_0 = -TRES;
      tape.get_taylor(res);
      r0 = 1.0 / TARG2;
      r_0 *= r0;
#endif /* !_NTIGHT_ */

      FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
        size_t aTmp = *Ares;
        ARES_INC = 0;
        AARG1_INC |= aTmp;
        AARG2_INC |= aTmp;
#else
        revreal aTmp = *Ares;
        ARES_INC = 0.0;
        AARG1_INC += aTmp * r0;
        AARG2_INC += aTmp * r_0;
#endif
      }

      break;

      /*--------------------------------------------------------------------------*/
    case div_d_a: /* Division double - adouble (/)    div_d_a */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();

#if !defined(_NTIGHT_)
      coval = tape.get_val_r();
#endif /* !_NTIGHT_ */

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg, ADJOINT_BUFFER[arg])

#if !defined(_NTIGHT_)
      /* olvo 980922 changed order to allow x=d/x */
      r0 = -TRES;
      if (arg == res)
        tape.get_taylor(arg);
      r0 /= TARG;
#endif /* !_NTIGHT_ */

      FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
        size_t aTmp = *Ares;
        ARES_INC = 0;
        AARG_INC |= aTmp;
#else
        revreal aTmp = *Ares;
        ARES_INC = 0.0;
        AARG_INC += aTmp * r0;
#endif
      }

#if !defined(_NTIGHT_)
      if (arg != res)
        tape.get_taylor(res);
#endif /* !_NTIGHT_ */
      break;

      /****************************************************************************/
      /*                                                         SIGN OPERATIONS
       */

      /*--------------------------------------------------------------------------*/
    case pos_sign_a: /* pos_sign_a */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg, ADJOINT_BUFFER[arg])

      FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
        size_t aTmp = *Ares;
        ARES_INC = 0;
        AARG_INC |= aTmp;
#else
        revreal aTmp = *Ares;
        ARES_INC = 0.0;
        AARG_INC += aTmp;
#endif
      }

#if !defined(_NTIGHT_)
      tape.get_taylor(res);
#endif /* !_NTIGHT_ */
      break;

      /*--------------------------------------------------------------------------*/
    case neg_sign_a: /* neg_sign_a */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg, ADJOINT_BUFFER[arg])

      FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
        size_t aTmp = *Ares;
        ARES_INC = 0;
        AARG_INC |= aTmp;
#else
        revreal aTmp = *Ares;
        ARES_INC = 0.0;
        AARG_INC -= aTmp;
#endif
      }

#if !defined(_NTIGHT_)
      tape.get_taylor(res);
#endif /* !_NTIGHT_ */
      break;

      /****************************************************************************/
      /*                                                         UNARY
       * OPERATIONS */

      /*--------------------------------------------------------------------------*/
    case exp_op: /* exponent operation    exp_op */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg, ADJOINT_BUFFER[arg])

      FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
        size_t aTmp = *Ares;
        ARES_INC = 0;
        AARG_INC |= aTmp;
#else
        revreal aTmp = *Ares;
        ARES_INC = 0.0;
        AARG_INC += aTmp * TRES;
#endif
      }

#if !defined(_NTIGHT_)
      tape.get_taylor(res);
#endif /* !_NTIGHT_ */
      break;

      /*--------------------------------------------------------------------------*/
    case sin_op: /* sine operation    sin_op */
      res = tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg1, ADJOINT_BUFFER[arg1])

      FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
        size_t aTmp = *Ares;
        ARES_INC = 0;
        AARG1_INC |= aTmp;
#else
        revreal aTmp = *Ares;
        ARES_INC = 0.0;
        AARG1_INC += aTmp * TARG2;
#endif
      }

#if !defined(_NTIGHT_)
      tape.get_taylor(res);
      tape.get_taylor(arg2); /* olvo 980710 covalue */
      /* NOTE: ADJOINT_BUFFER[arg2] should be 0 already */
#endif /* !_NTIGHT_ */
      break;

      /*--------------------------------------------------------------------------*/
    case cos_op: /* cosine operation    cos_op */
      res = tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg1, ADJOINT_BUFFER[arg1])

      FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
        size_t aTmp = *Ares;
        ARES_INC = 0;
        AARG1_INC |= aTmp;
#else
        revreal aTmp = *Ares;
        ARES_INC = 0.0;
        AARG1_INC -= aTmp * TARG2;
#endif
      }

#if !defined(_NTIGHT_)
      tape.get_taylor(res);
      tape.get_taylor(arg2); /* olvo 980710 covalue */
                             /* NOTE ADJOINT_BUFFER[arg2] should be 0 already */
#endif                       /* !_NTIGHT_ */
      break;

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

#if !defined(_NTIGHT_)
      tape.get_taylor(res);
#endif /* !_NTIGHT_ */

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg1, ADJOINT_BUFFER[arg1])

      FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
        size_t aTmp = *Ares;
        ARES_INC = 0;
        AARG1_INC |= aTmp;
#else
        revreal aTmp = *Ares;
        ARES_INC = 0.0;
        AARG1_INC += aTmp * TARG2;
#endif
      }
      break;

      /*--------------------------------------------------------------------------*/
    case log_op: /* log_op */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();

#if !defined(_NTIGHT_)
      tape.get_taylor(res);
#endif /* !_NTIGHT_ */

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg, ADJOINT_BUFFER[arg])

#if !defined(_INT_REV_)
      r0 = 1.0 / TARG;
#endif

      FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
        size_t aTmp = *Ares;
        ARES_INC = 0;
        AARG_INC |= aTmp;
#else
        revreal aTmp = *Ares;
        ARES_INC = 0.0;
        AARG_INC += aTmp * r0;
#endif
      }
      break;

      /*--------------------------------------------------------------------------*/
    case pow_op: /* pow_op */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();
#if !defined(_NTIGHT_)
      coval = tape.get_val_r();
#endif /* !_NTIGHT_ */

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg, ADJOINT_BUFFER[arg])

#if !defined(_NTIGHT_)
      /* olvo 980921 changed order to allow x=pow(x,n) */
      r0 = TRES; // r0 = pow(x, n)
      if (arg == res)
        tape.get_taylor(arg);
      if (TARG == 0.0)
        r0 = 0.0;
      else
        r0 *= coval / TARG; // r0 = r0 * n / x
#endif                      /* !_NTIGHT_ */

      FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
        size_t aTmp = *Ares;
        ARES_INC = 0;
        AARG_INC |= aTmp;
#else
        revreal aTmp = *Ares;
        ARES_INC = 0.0;
        AARG_INC += aTmp * r0;
#endif
      }

#if !defined(_NTIGHT_)
      if (res != arg)
        tape.get_taylor(res);
#endif /* !_NTIGHT_ */
      break;

      /*--------------------------------------------------------------------------*/
    case sqrt_op: /* sqrt_op */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg, ADJOINT_BUFFER[arg])

#if !defined(_NTIGHT_)
      if (TRES == 0.0)
        r0 = 0.0;
      else
        r0 = 0.5 / TRES;
#endif /* !_NTIGHT_ */

      FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
        size_t aTmp = *Ares;
        ARES_INC = 0;
        AARG_INC |= aTmp;
#else
        revreal aTmp = *Ares;
        ARES_INC = 0.0;
        AARG_INC += aTmp * r0;
#endif
      }

#if !defined(_NTIGHT_)
      tape.get_taylor(res);
#endif /* !_NTIGHT_ */
      break;

      /*--------------------------------------------------------------------------*/
    case cbrt_op: /* cbrt_op */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg, ADJOINT_BUFFER[arg])

#if !defined(_NTIGHT_)
      if (TRES == 0.0)
        r0 = 0.0;
      else
        r0 = 1.0 / (3.0 * TRES * TRES);
#endif /* !_NTIGHT_ */

      FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
        size_t aTmp = *Ares;
        ARES_INC = 0;
        AARG_INC |= aTmp;
#else
        revreal aTmp = *Ares;
        ARES_INC = 0.0;
        AARG_INC += aTmp * r0;
#endif
      }

#if !defined(_NTIGHT_)
      tape.get_taylor(res);
#endif /* !_NTIGHT_ */
      break;

      /*--------------------------------------------------------------------------*/
    case gen_quad: /* gen_quad */
      res = tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();
#if !defined(_NTIGHT_)
      coval = tape.get_val_r();
      coval = tape.get_val_r();
#endif /* !_NTIGHT_ */

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg1, ADJOINT_BUFFER[arg1])

      FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
        size_t aTmp = *Ares;
        ARES_INC = 0;
        AARG1_INC |= aTmp;
#else
        revreal aTmp = *Ares;
        ARES_INC = 0.0;
        AARG1_INC += aTmp * TARG2;
#endif
      }

#if !defined(_NTIGHT_)
      tape.get_taylor(res);
#endif /* !_NTIGHT_ */
      break;

      /*--------------------------------------------------------------------------*/
    case min_op: /* min_op */
      res = tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();
#if !defined(_NTIGHT_)
      coval = tape.get_val_r();

      tape.get_taylor(res);
#endif /* !_NTIGHT_ */

      ASSIGN_A(Aarg1, ADJOINT_BUFFER[arg1])
      ASSIGN_A(Aarg2, ADJOINT_BUFFER[arg2])
      ASSIGN_A(Ares, ADJOINT_BUFFER[res])

#if !defined(_NTIGHT_)
      if (TARG1 > TARG2)
        FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
          size_t aTmp = *Ares;
          ARES_INC = 0;
#else
          revreal aTmp = *Ares;
          ARES_INC = 0.0;
#endif
          if ((coval) && (aTmp))
            MINDEC(ret_c, 2);
#if defined(_INT_REV_)
          AARG2_INC |= aTmp;
#else
          AARG2_INC += aTmp;
#endif
        }
      else if (TARG1 < TARG2)
        FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
          size_t aTmp = *Ares;
          ARES_INC = 0;
#else
          revreal aTmp = *Ares;
          ARES_INC = 0.0;
#endif
          if ((!coval) && (aTmp))
            MINDEC(ret_c, 2);
#if defined(_INT_REV_)
          AARG1_INC |= aTmp;
#else
          AARG1_INC += aTmp;
#endif
        }
      else { /* both are equal */
        FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
          size_t aTmp = *Ares;
          ARES_INC = 0;
          AARG2_INC |= aTmp;
          AARG1_INC |= aTmp;
#else
          revreal aTmp = *Ares / 2.0;
          fprintf(DIAG_OUT, "ADOL-C warning: fmin/fmax used with equal "
                            "arguments, adjoints might be incorrect.\n");
          ARES_INC = 0.0;
          AARG2_INC += aTmp;
          AARG1_INC += aTmp;
#endif
        }
        if (arg1 != arg2)
          MINDEC(ret_c, 1);
      }
#else
      FOR_0_LE_l_LT_p {
        size_t aTmp = *Ares;
        ARES_INC = 0;
        AARG1_INC |= aTmp;
        AARG2_INC |= aTmp;
      }
#endif /* !_NTIGHT_ */
      break;

      /*--------------------------------------------------------------------------*/
    case abs_val: /* abs_val */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();
#if !defined(_NTIGHT_)
      coval = tape.get_val_r();

      tape.get_taylor(res);
#endif /* !_NTIGHT_ */

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg, ADJOINT_BUFFER[arg])

#if defined(_ABS_NORM_)
      if (rownum == switchnum) {
        AARG = 1.0;
      } else {
        results[indep + switchnum] = *Ares;
        *Ares = 0.0;
      }
      switchnum--;
#elif defined(_ABS_NORM_SIG_)
      ARES_INC = 0.0;
      AARG_INC += siggrad[switchnum] * (*Ares);
      switchnum--;
#else
#if !defined(_NTIGHT_)
      if (TARG < 0.0)
        FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
          size_t aTmp = *Ares;
          ARES_INC = 0;
#else
          revreal aTmp = *Ares;
          ARES_INC = 0.0;
#endif
          if ((coval) && (aTmp))
            MINDEC(ret_c, 2);
#if defined(_INT_REV_)
          AARG_INC |= aTmp;
#else
          AARG_INC -= aTmp;
#endif
        }
      else if (TARG > 0.0)
        FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
          size_t aTmp = *Ares;
          ARES_INC = 0;
#else
          revreal aTmp = *Ares;
          ARES_INC = 0.0;
#endif
          if ((!coval) && (aTmp))
            MINDEC(ret_c, 2);
#if defined(_INT_REV_)
          AARG_INC |= aTmp;
#else
          AARG_INC += aTmp;
#endif
        }
      else
        FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
          size_t aTmp = *Ares;
          ARES_INC = 0;
#else
          revreal aTmp = *Ares;
          ARES_INC = 0.0;
#endif
          if (aTmp)
            MINDEC(ret_c, 1);
        }
#else
      FOR_0_LE_l_LT_p {
        size_t aTmp = *Ares;
        ARES_INC = 0;
        AARG_INC |= aTmp;
      }
#endif /* !_NTIGHT_ */
#endif /* _ABS_NORM */
      break;

      /*--------------------------------------------------------------------------*/
    case ceil_op: /* ceil_op */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();
#if !defined(_NTIGHT_)
      coval = tape.get_val_r();

      tape.get_taylor(res);
#endif /* !_NTIGHT_ */

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])

#if !defined(_NTIGHT_)
      coval = (coval != ceil(TARG));
#endif /* !_NTIGHT_ */

      FOR_0_LE_l_LT_p {
#if !defined(_NTIGHT_)
        if ((coval) && (*Ares))
          MINDEC(ret_c, 2);
#endif /* !_NTIGHT_ */
#if defined(_INT_REV_)
        ARES_INC = 0;
#else
        ARES_INC = 0.0;
#endif
      }
      break;

      /*--------------------------------------------------------------------------*/
    case floor_op: /* floor_op */
      res = tape.get_locint_r();
      arg = tape.get_locint_r();
#if !defined(_NTIGHT_)
      coval = tape.get_val_r();

      tape.get_taylor(res);
#endif /* !_NTIGHT_ */

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg, ADJOINT_BUFFER[arg])

#if !defined(_NTIGHT_)
      coval = (coval != floor(TARG1));
#endif /* !_NTIGHT_ */

      FOR_0_LE_l_LT_p {
#if !defined(_NTIGHT_)
        if ((coval) && (*Ares))
          MINDEC(ret_c, 2);
#endif /* !_NTIGHT_ */
#if defined(_INT_REV_)
        ARES_INC = 0;
#else
        ARES_INC = 0.0;
#endif
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
#if !defined(_NTIGHT_)
      coval = tape.get_val_r();

      tape.get_taylor(res);
#endif /* !_NTIGHT_ */

      ASSIGN_A(Aarg1, ADJOINT_BUFFER[arg1])
      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg2, ADJOINT_BUFFER[arg2])

#if !defined(_NTIGHT_)
      /* olvo 980924 changed code a little bit */
      if (TARG > 0.0) {
        if (res != arg1)
          FOR_0_LE_l_LT_p {
            if ((coval <= 0.0) && (*Ares))
              MINDEC(ret_c, 2);
#if defined(_INT_REV_)
            AARG1_INC |= *Ares;
            ARES_INC = 0;
#else
            AARG1_INC += *Ares;
            ARES_INC = 0.0;
#endif
          }
        else
          FOR_0_LE_l_LT_p if ((coval <= 0.0) && (ARES_INC)) MINDEC(ret_c, 2);
      } else {
        if (res != arg2)
          FOR_0_LE_l_LT_p {
            if ((coval <= 0.0) && (*Ares))
              MINDEC(ret_c, 2);
#if defined(_INT_REV_)
            AARG2_INC |= *Ares;
            ARES_INC = 0;
#else
            AARG2_INC += *Ares;
            ARES_INC = 0.0;
#endif
          }
        else
          FOR_0_LE_l_LT_p if ((coval <= 0.0) && (ARES_INC)) MINDEC(ret_c, 2);
      }
#else
      if (res != arg1) {
        FOR_0_LE_l_LT_p AARG1_INC |= ARES_INC;
        ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      }
      if (res != arg2) {
        FOR_0_LE_l_LT_p AARG2_INC |= ARES_INC;
        ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      }
      if ((res != arg1) && (res != arg2))
        FOR_0_LE_l_LT_p ARES_INC = 0;
#endif /* !_NTIGHT_ */
      break;

    case cond_eq_assign: /* cond_assign */
      res = tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();
      arg = tape.get_locint_r();
#if !defined(_NTIGHT_)
      coval = tape.get_val_r();

      tape.get_taylor(res);
#endif /* !_NTIGHT_ */

      ASSIGN_A(Aarg1, ADJOINT_BUFFER[arg1])
      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg2, ADJOINT_BUFFER[arg2])

#if !defined(_NTIGHT_)
      /* olvo 980924 changed code a little bit */
      if (TARG >= 0.0) {
        if (res != arg1)
          FOR_0_LE_l_LT_p {
            if ((coval < 0.0) && (*Ares))
              MINDEC(ret_c, 2);
#if defined(_INT_REV_)
            AARG1_INC |= *Ares;
            ARES_INC = 0;
#else
            AARG1_INC += *Ares;
            ARES_INC = 0.0;
#endif
          }
        else
          FOR_0_LE_l_LT_p if ((coval < 0.0) && (ARES_INC)) MINDEC(ret_c, 2);
      } else {
        if (res != arg2)
          FOR_0_LE_l_LT_p {
            if ((coval < 0.0) && (*Ares))
              MINDEC(ret_c, 2);
#if defined(_INT_REV_)
            AARG2_INC |= *Ares;
            ARES_INC = 0;
#else
            AARG2_INC += *Ares;
            ARES_INC = 0.0;
#endif
          }
        else
          FOR_0_LE_l_LT_p if ((coval < 0.0) && (ARES_INC)) MINDEC(ret_c, 2);
      }
#else
      if (res != arg1) {
        FOR_0_LE_l_LT_p AARG1_INC |= ARES_INC;
        ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      }
      if (res != arg2) {
        FOR_0_LE_l_LT_p AARG2_INC |= ARES_INC;
        ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      }
      if ((res != arg1) && (res != arg2))
        FOR_0_LE_l_LT_p ARES_INC = 0;
#endif /* !_NTIGHT_ */
      break;

      /*--------------------------------------------------------------------------*/
    case cond_assign_s: /* cond_assign_s */
      res = tape.get_locint_r();
      arg1 = tape.get_locint_r();
      arg = tape.get_locint_r();
#if !defined(_NTIGHT_)
      coval = tape.get_val_r();

      tape.get_taylor(res);
#endif /* !_NTIGHT_ */

      ASSIGN_A(Aarg1, ADJOINT_BUFFER[arg1])
      ASSIGN_A(Ares, ADJOINT_BUFFER[res])

#if !defined(_NTIGHT_)
      /* olvo 980924 changed code a little bit */
      if (TARG > 0.0) {
        if (res != arg1)
          FOR_0_LE_l_LT_p {
            if ((coval <= 0.0) && (*Ares))
              MINDEC(ret_c, 2);
#if defined(_INT_REV_)
            AARG1_INC |= *Ares;
            ARES_INC = 0.0;
#else
            AARG1_INC += *Ares;
            ARES_INC = 0.0;
#endif
          }
        else
          FOR_0_LE_l_LT_p if ((coval <= 0.0) && (ARES_INC)) MINDEC(ret_c, 2);
      } else if (TARG == 0.0) /* we are at the tie */
        FOR_0_LE_l_LT_p if (ARES_INC) MINDEC(ret_c, 0);
#else
      if (res != arg1)
        FOR_0_LE_l_LT_p {
          AARG1 |= *Ares;
          ARES_INC = 0;
        }
#endif /* !_NTIGHT_ */
      break;

    case cond_eq_assign_s: /* cond_eq_assign_s */
      res = tape.get_locint_r();
      arg1 = tape.get_locint_r();
      arg = tape.get_locint_r();
#if !defined(_NTIGHT_)
      coval = tape.get_val_r();

      tape.get_taylor(res);
#endif /* !_NTIGHT_ */

      ASSIGN_A(Aarg1, ADJOINT_BUFFER[arg1])
      ASSIGN_A(Ares, ADJOINT_BUFFER[res])

#if !defined(_NTIGHT_)
      /* olvo 980924 changed code a little bit */
      if (TARG >= 0.0) {
        if (res != arg1)
          FOR_0_LE_l_LT_p {
            if ((coval < 0.0) && (*Ares))
              MINDEC(ret_c, 2);
#if defined(_INT_REV_)
            AARG1_INC |= *Ares;
            ARES_INC = 0.0;
#else
            AARG1_INC += *Ares;
            ARES_INC = 0.0;
#endif
          }
        else
          FOR_0_LE_l_LT_p if ((coval < 0.0) && (ARES_INC)) MINDEC(ret_c, 2);
      }
#else
      if (res != arg1)
        FOR_0_LE_l_LT_p {
          AARG1 |= *Ares;
          ARES_INC = 0;
        }
#endif /* !_NTIGHT_ */
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
#if !defined(_NTIGHT_)
      coval = tape.get_val_r();
#endif
      ASSIGN_A(Ares, ADJOINT_BUFFER[res])

      FOR_0_LE_l_LT_p
#if defined(_INT_REV_)
          ARES_INC = 0;
#else
          ARES_INC = 0.0;
#endif

#if !defined(_NTIGHT_)
      tape.get_taylor(res);
#endif /* !_NTIGHT_ */

      break;
#endif
      /*--------------------------------------------------------------------------*/
    case subscript: {
#if !defined(_NTIGHT_)
      double val =
#endif
          tape.get_val_r();
      res = tape.get_locint_r();
#if !defined(_NTIGHT_)
      size_t idx, numval = (size_t)trunc(fabs(val));
      locint vectorloc;
      vectorloc =
#endif
          tape.get_locint_r();
      arg = tape.get_locint_r();
#if !defined(_NTIGHT_)
      idx = (size_t)trunc(fabs(TARG));
      if (idx >= numval)
        fprintf(DIAG_OUT,
                "ADOL-C warning: index out of bounds while subscripting n=%zu, "
                "idx=%zu\n",
                numval, idx);
      arg1 = vectorloc + idx;
      ASSIGN_A(Aarg1, ADJOINT_BUFFER[arg1])
      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
        AARG1_INC |= *Ares;
        ARES_INC = 0;
#else
        AARG1_INC += *Ares;
        *Ares = 0.0;
#endif
      }
      tape.get_taylor(res);
#else
      ADOLCError::fail(ADOLCError::ErrorType::ACTIVE_SUBSCRIPTING,
                       CURRENT_LOCATION);
#endif /* !_NTIGHT_ */
    } break;

    case subscript_ref: {
#if !defined(_NTIGHT_)
      double val =
#endif
          tape.get_val_r();
      res = tape.get_locint_r();
#if !defined(_NTIGHT_)
      size_t idx, numval = (size_t)trunc(fabs(val));
      locint vectorloc;
      vectorloc =
#endif
          tape.get_locint_r();
      arg = tape.get_locint_r();
#if !defined(_NTIGHT_)
      idx = (size_t)trunc(fabs(TARG));
      if (idx >= numval)
        fprintf(DIAG_OUT,
                "ADOL-C warning: index out of bounds while subscripting (ref) "
                "n=%zu, idx=%zu\n",
                numval, idx);
      arg1 = (size_t)trunc(fabs(TRES));
      /*
       * This is actually NOP
       * basically all we need is that arg1 == vectorloc+idx
       * so doing a check here is probably good
       */
      if (arg1 != vectorloc + idx)
        ADOLCError::fail(
            ADOLCError::ErrorType::ADUBREF_SAFE_MODE, CURRENT_LOCATION,
            ADOLCError::FailInfo{.info5 = vectorloc + idx, .info6 = arg1});
      tape.get_taylor(res);
#else
      ADOLCError::fail(ADOLCError::ErrorType::ACTIVE_SUBSCRIPTING,
                       CURRENT_LOCATION);
#endif /* !_NTIGHT_ */
    } break;

    case ref_copyout:
      res = tape.get_locint_r();
      arg1 = tape.get_locint_r();
#if !defined(_NTIGHT_)
      arg = (size_t)trunc(fabs(TARG1));
      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg, ADJOINT_BUFFER[arg])

      FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
        AARG_INC |= *Ares;
        ARES_INC = 0;
#else
        AARG_INC += *Ares;
        ARES_INC = 0.0;
#endif
      }
      tape.get_taylor(res);
#else
      ADOLCError::fail(ADOLCError::ErrorType::ADUBREF_VE_REF, CURRENT_LOCATION);
#endif
      break;

    case ref_incr_a: /* Increment an adouble    incr_a */
    case ref_decr_a: /* Increment an adouble    decr_a */
      arg1 = tape.get_locint_r();

#if !defined(_NTIGHT_)
      res = (size_t)trunc(fabs(TARG1));
      tape.get_taylor(res);
#else
      ADOLCError::fail(ADOLCError::ErrorType::ADUBREF_VE_REF, CURRENT_LOCATION);
#endif /* !_NTIGHT_ */
      break;

    case ref_assign_d: /* assign an adouble variable a    assign_d */
      /* double value. (=) */
      arg1 = tape.get_locint_r();
#if !defined(_NTIGHT_)
      res = (size_t)trunc(fabs(TARG1));
      coval = tape.get_val_r();

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])

      FOR_0_LE_l_LT_p
#if defined(_INT_REV_)
          ARES_INC = 0;
#else
          ARES_INC = 0.0;
#endif

      tape.get_taylor(res);
#else
      ADOLCError::fail(ADOLCError::ErrorType::ADUBREF_VE_REF, CURRENT_LOCATION);
#endif /* !_NTIGHT_ */
      break;

    case ref_assign_d_zero: /* assign an adouble variable a    assign_d_zero */
    case ref_assign_d_one:  /* double value (0 or 1). (=)       assign_d_one */
      arg1 = tape.get_locint_r();

#if !defined(_NTIGHT_)
      res = (size_t)trunc(fabs(TARG1));

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])

      FOR_0_LE_l_LT_p
#if defined(_INT_REV_)
          ARES_INC = 0;
#else
          ARES_INC = 0.0;
#endif
      tape.get_taylor(res);
#else
      ADOLCError::fail(ADOLCError::ErrorType::ADUBREF_VE_REF, CURRENT_LOCATION);
#endif /* !_NTIGHT_ */
      break;

    case ref_assign_a: /* assign an adouble variable an    assign_a */
      /* adouble value. (=) */
      arg1 = tape.get_locint_r();
      arg = tape.get_locint_r();

#if !defined(_NTIGHT_)
      res = (size_t)trunc(fabs(TARG1));

      ASSIGN_A(Aarg, ADJOINT_BUFFER[arg])
      ASSIGN_A(Ares, ADJOINT_BUFFER[res])

      FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
        AARG_INC |= *Ares;
        ARES_INC = 0;
#else
        AARG_INC += *Ares;
        ARES_INC = 0.0;
#endif
      }

      tape.get_taylor(res);
#else
      ADOLCError::fail(ADOLCError::ErrorType::ADUBREF_VE_REF, CURRENT_LOCATION);
#endif /* !_NTIGHT_ */
      break;

    case ref_assign_ind: /* assign an adouble variable an    assign_ind */
      /* independent double value (<<=) */
      arg1 = tape.get_locint_r();

#if !defined(_NTIGHT_)
      res = (size_t)trunc(fabs(TARG1));

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])

      FOR_0_LE_l_LT_p RESULTS(l, indexi) = ARES_INC;

      tape.get_taylor(res);
#else
      ADOLCError::fail(ADOLCError::ErrorType::ADUBREF_VE_REF, CURRENT_LOCATION);
#endif /* !_NTIGHT_ */
      indexi--;
      break;

    case ref_eq_plus_d: /* Add a floating point to an    eq_plus_d */
      /* adouble. (+=) */
      arg1 = tape.get_locint_r();
#if !defined(_NTIGHT_)
      res = (size_t)trunc(fabs(TARG1));
      coval = tape.get_val_r();

      tape.get_taylor(res);
#else
      ADOLCError::fail(ADOLCError::ErrorType::ADUBREF_VE_REF, CURRENT_LOCATION);
#endif /* !_NTIGHT_ */
      break;

    case ref_eq_plus_a: /* Add an adouble to another    eq_plus_a */
      /* adouble. (+=) */
      arg1 = tape.get_locint_r();
      arg = tape.get_locint_r();

#if !defined(_NTIGHT_)
      res = (size_t)trunc(fabs(TARG1));

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg, ADJOINT_BUFFER[arg]);

      FOR_0_LE_l_LT_p
#if defined(_INT_REV_)
          AARG_INC |= ARES_INC;
#else
          AARG_INC += ARES_INC;
#endif

      tape.get_taylor(res);
#else
      ADOLCError::fail(ADOLCError::ErrorType::ADUBREF_VE_REF, CURRENT_LOCATION);
#endif /* !_NTIGHT_ */
      break;

    case ref_eq_min_d: /* Subtract a floating point from an    eq_min_d */
      /* adouble. (-=) */
      arg1 = tape.get_locint_r();
#if !defined(_NTIGHT_)
      res = (size_t)trunc(fabs(TARG1));
      coval = tape.get_val_r();

      tape.get_taylor(res);
#else
      ADOLCError::fail(ADOLCError::ErrorType::ADUBREF_VE_REF, CURRENT_LOCATION);
#endif /* !_NTIGHT_ */
      break;

    case ref_eq_min_a: /* Subtract an adouble from another    eq_min_a */
      /* adouble. (-=) */
      arg1 = tape.get_locint_r();
      arg = tape.get_locint_r();

#if !defined(_NTIGHT_)
      res = (size_t)trunc(fabs(TARG1));
      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg, ADJOINT_BUFFER[arg])

      FOR_0_LE_l_LT_p
#if defined(_INT_REV_)
          AARG_INC |= ARES_INC;
#else
          AARG_INC -= ARES_INC;
#endif

      tape.get_taylor(res);
#else
      ADOLCError::fail(ADOLCError::ErrorType::ADUBREF_VE_REF, CURRENT_LOCATION);
#endif /* !_NTIGHT_ */
      break;

    case ref_eq_mult_d: /* Multiply an adouble by a    eq_mult_d */
      /* flaoting point. (*=) */
      arg1 = tape.get_locint_r();
#if !defined(_NTIGHT_)
      res = (size_t)trunc(fabs(TARG1));
      coval = tape.get_val_r();

#if !defined(_INT_REV_)
      ASSIGN_A(Ares, ADJOINT_BUFFER[res])

      FOR_0_LE_l_LT_p ARES_INC *= coval;
#endif

      tape.get_taylor(res);
#else
      ADOLCError::fail(ADOLCError::ErrorType::ADUBREF_VE_REF, CURRENT_LOCATION);
#endif /* !_NTIGHT_ */
      break;

    case ref_eq_mult_a: /* Multiply one adouble by another    eq_mult_a */
      /* (*=) */
      arg1 = tape.get_locint_r();
      arg = tape.get_locint_r();

#if !defined(_NTIGHT_)
      res = (size_t)trunc(fabs(TARG1));
      tape.get_taylor(res);

      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg, ADJOINT_BUFFER[arg])

      FOR_0_LE_l_LT_p
#if defined(_INT_REV_)
          AARG_INC |= ARES_INC;
#else
      {
        revreal aTmp = *Ares;
        ARES_INC = aTmp * TARG;
        AARG_INC += aTmp * TRES;
      }
#endif
#else
      ADOLCError::fail(ADOLCError::ErrorType::ADUBREF_VE_REF, CURRENT_LOCATION);
#endif /* !_NTIGHT_ */
      break;

    case vec_copy:
      res = tape.get_locint_r();
      size = tape.get_locint_r();
      arg = tape.get_locint_r();
      for (locint qq = 0; qq < size; qq++) {

        ASSIGN_A(Aarg, ADJOINT_BUFFER[arg + qq])
        ASSIGN_A(Ares, ADJOINT_BUFFER[res + qq])

        FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
          AARG_INC |= *Ares;
          ARES_INC = 0;
#else
          AARG_INC += *Ares;
          ARES_INC = 0.0;
#endif
        }

#if !defined(_NTIGHT_)
        tape.get_taylor(res + qq);
#endif /* !_NTIGHT_ */
      }

      break;

    case vec_dot:
      res = tape.get_locint_r();
      size = tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();
      for (locint qq = 0; qq < size; qq++) {
        ASSIGN_A(Ares, ADJOINT_BUFFER[res])
        ASSIGN_A(Aarg2, ADJOINT_BUFFER[arg2])
        ASSIGN_A(Aarg1, ADJOINT_BUFFER[arg1])
        FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
          AARG2_INC |= *Ares;
          AARG1_INC |= ARES_INC;
#else
          AARG2_INC += (*Ares) * TARG1;
          AARG1_INC += ARES_INC * TARG2;
#endif
        }
        arg2++;
        arg1++;
      }
#if !defined(_NTIGHT_)
      tape.get_taylor(res);
#endif /* !_NTIGHT_ */
      break;

    case vec_axpy:
      res = tape.get_locint_r();
      size = tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();
      arg = tape.get_locint_r();
      for (locint qq = 0; qq < size; qq++) {
        ASSIGN_A(Ares, ADJOINT_BUFFER[res])
        ASSIGN_A(Aarg, ADJOINT_BUFFER[arg])
        ASSIGN_A(Aarg2, ADJOINT_BUFFER[arg2])
        ASSIGN_A(Aarg1, ADJOINT_BUFFER[arg1])
        FOR_0_LE_l_LT_p {
#if defined(_INT_REV_)
          AARG_INC |= *Ares;
          AARG2_INC |= *Ares;
          AARG1_INC |= ARES_INC;
#else
          AARG2_INC += *Ares;
          AARG1_INC += (*Ares) * TARG;
          AARG_INC += ARES_INC * TARG1;
#endif
        }
#if !defined(_NTIGHT_)
        tape.get_taylor(res);
#endif /* !_NTIGHT_ */
        arg2++;
        arg1++;
        res++;
      }
      break;

    case ref_cond_assign: /* cond_assign */
    {
#if !defined(_NTIGHT_)
      locint ref =
#endif
          tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();
      arg = tape.get_locint_r();
#if !defined(_NTIGHT_)
      coval = tape.get_val_r();
      res = (size_t)trunc(fabs(rp_T[ref]));

      tape.get_taylor(res);

      ASSIGN_A(Aarg1, ADJOINT_BUFFER[arg1])
      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg2, ADJOINT_BUFFER[arg2])

      /* olvo 980924 changed code a little bit */
      if (TARG > 0.0) {
        if (res != arg1)
          FOR_0_LE_l_LT_p {
            if ((coval <= 0.0) && (*Ares))
              MINDEC(ret_c, 2);
#if defined(_INT_REV_)
            AARG1_INC |= *Ares;
            ARES_INC = 0;
#else
            AARG1_INC += *Ares;
            ARES_INC = 0.0;
#endif
          }
        else
          FOR_0_LE_l_LT_p if ((coval <= 0.0) && (ARES_INC)) MINDEC(ret_c, 2);
      } else {
        if (res != arg2)
          FOR_0_LE_l_LT_p {
            if ((coval <= 0.0) && (*Ares))
              MINDEC(ret_c, 2);
#if defined(_INT_REV_)
            AARG2_INC |= *Ares;
            ARES_INC = 0;
#else
            AARG2_INC += *Ares;
            ARES_INC = 0.0;
#endif
          }
        else
          FOR_0_LE_l_LT_p if ((coval <= 0.0) && (ARES_INC)) MINDEC(ret_c, 2);
      }
#else
      ADOLCError::fail(ADOLCError::ErrorType::ADUBREF_VE_REF, CURRENT_LOCATION);
#endif /* !_NTIGHT_ */
    } break;

    case ref_cond_eq_assign: /* cond_eq_assign */
    {
#if !defined(_NTIGHT_)
      locint ref =
#endif
          tape.get_locint_r();
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();
      arg = tape.get_locint_r();
#if !defined(_NTIGHT_)
      coval = tape.get_val_r();
      res = (size_t)trunc(fabs(rp_T[ref]));

      tape.get_taylor(res);

      ASSIGN_A(Aarg1, ADJOINT_BUFFER[arg1])
      ASSIGN_A(Ares, ADJOINT_BUFFER[res])
      ASSIGN_A(Aarg2, ADJOINT_BUFFER[arg2])

      /* olvo 980924 changed code a little bit */
      if (TARG >= 0.0) {
        if (res != arg1)
          FOR_0_LE_l_LT_p {
            if ((coval < 0.0) && (*Ares))
              MINDEC(ret_c, 2);
#if defined(_INT_REV_)
            AARG1_INC |= *Ares;
            ARES_INC = 0;
#else
            AARG1_INC += *Ares;
            ARES_INC = 0.0;
#endif
          }
        else
          FOR_0_LE_l_LT_p if ((coval < 0.0) && (ARES_INC)) MINDEC(ret_c, 2);
      } else {
        if (res != arg2)
          FOR_0_LE_l_LT_p {
            if ((coval < 0.0) && (*Ares))
              MINDEC(ret_c, 2);
#if defined(_INT_REV_)
            AARG2_INC |= *Ares;
            ARES_INC = 0;
#else
            AARG2_INC += *Ares;
            ARES_INC = 0.0;
#endif
          }
        else
          FOR_0_LE_l_LT_p if ((coval < 0.0) && (ARES_INC)) MINDEC(ret_c, 2);
      }
#else
      ADOLCError::fail(ADOLCError::ErrorType::ADUBREF_VE_REF, CURRENT_LOCATION);
#endif /* !_NTIGHT_ */
    } break;

    case ref_cond_assign_s: /* cond_assign_s */
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();
      arg = tape.get_locint_r();
#if !defined(_NTIGHT_)
      coval = tape.get_val_r();
      res = (size_t)trunc(fabs(TARG2));
      tape.get_taylor(res);

      ASSIGN_A(Aarg1, ADJOINT_BUFFER[arg1])
      ASSIGN_A(Ares, ADJOINT_BUFFER[res])

      /* olvo 980924 changed code a little bit */
      if (TARG > 0.0) {
        if (res != arg1)
          FOR_0_LE_l_LT_p {
            if ((coval <= 0.0) && (*Ares))
              MINDEC(ret_c, 2);
#if defined(_INT_REV_)
            AARG1_INC |= *Ares;
            ARES_INC = 0.0;
#else
            AARG1_INC += *Ares;
            ARES_INC = 0.0;
#endif
          }
        else
          FOR_0_LE_l_LT_p if ((coval <= 0.0) && (ARES_INC)) MINDEC(ret_c, 2);
      } else if (TARG == 0.0) /* we are at the tie */
        FOR_0_LE_l_LT_p if (ARES_INC) MINDEC(ret_c, 0);
#else
      ADOLCError::fail(ADOLCError::ErrorType::ADUBREF_VE_REF, CURRENT_LOCATION);
#endif /* !_NTIGHT_ */
      break;

    case ref_cond_eq_assign_s: /* cond_eq_assign_s */
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();
      arg = tape.get_locint_r();
#if !defined(_NTIGHT_)
      coval = tape.get_val_r();
      res = (size_t)trunc(fabs(TARG2));
      tape.get_taylor(res);

      ASSIGN_A(Aarg1, ADJOINT_BUFFER[arg1])
      ASSIGN_A(Ares, ADJOINT_BUFFER[res])

      /* olvo 980924 changed code a little bit */
      if (TARG >= 0.0) {
        if (res != arg1)
          FOR_0_LE_l_LT_p {
            if ((coval < 0.0) && (*Ares))
              MINDEC(ret_c, 2);
#if defined(_INT_REV_)
            AARG1_INC |= *Ares;
            ARES_INC = 0.0;
#else
            AARG1_INC += *Ares;
            ARES_INC = 0.0;
#endif
          }
        else
          FOR_0_LE_l_LT_p if ((coval < 0.0) && (ARES_INC)) MINDEC(ret_c, 2);
      }
#else
      ADOLCError::fail(ADOLCError::ErrorType::ADUBREF_VE_REF, CURRENT_LOCATION);
#endif /* !_NTIGHT_ */
      break;

      /****************************************************************************/
      /*                                                          REMAINING
       * STUFF */

      /*--------------------------------------------------------------------------*/
    case take_stock_op: /* take_stock_op */
      res = tape.get_locint_r();
      size = tape.get_locint_r();
#if !defined(_NTIGHT_)
      tape.get_val_v_r(size);
#endif /* !_NTIGHT_ */

      res += size;
      for (ls = size; ls > 0; ls--) {
        res--;

        ASSIGN_A(Ares, ADJOINT_BUFFER[res])

        FOR_0_LE_l_LT_p ARES_INC = 0.0;
      }
      break;

      /*--------------------------------------------------------------------------*/
    case death_not: /* death_not */
      arg2 = tape.get_locint_r();
      arg1 = tape.get_locint_r();

      for (j = arg1; j <= arg2; j++) {
        ASSIGN_A(Aarg1, ADJOINT_BUFFER[j])

        FOR_0_LE_l_LT_p AARG1_INC = 0.0;
      }

#if !defined(_NTIGHT_)
      for (j = arg1; j <= arg2; j++)
        tape.get_taylor(j);
#endif /* !_NTIGHT_ */
      break;

#if !defined(_INT_REV_)
      /*--------------------------------------------------------------------------*/
    case ext_diff: /* extern differntiated function */
      tape.cp_index(tape.get_locint_r());
      tape.lowestYLoc_rev(tape.get_locint_r());
      tape.lowestXLoc_rev(tape.get_locint_r());
      m = tape.get_locint_r();
      n = tape.get_locint_r();
      tape.ext_diff_fct_index(tape.get_locint_r());
      ADOLC_EXT_FCT_SAVE_NUMDIRS;
      edfct = get_ext_diff_fct(tape.tapeId(), tape.ext_diff_fct_index());

      oldTraceFlag = tape.traceFlag();
      tape.traceFlag(0);

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
      arg = tape.lowestYLoc_rev();
      for (loop = 0; loop < m; ++loop) {
        ADOLC_EXT_FCT_COPY_ADJOINTS(ADOLC_EXT_FCT_U[loop], ADJOINT_BUFFER_ARG);
        ++arg;
      }

      arg = tape.lowestXLoc_rev();
      for (loop = 0; loop < n; ++loop) {
        ADOLC_EXT_FCT_COPY_ADJOINTS(ADOLC_EXT_FCT_Z[loop], ADJOINT_BUFFER_ARG);
        ++arg;
      }
      arg = tape.lowestXLoc_rev();
      for (loop = 0; loop < n; ++loop, ++arg) {
        edfct->dp_x[loop] = TARG;
      }
      arg = tape.lowestYLoc_rev();
      for (loop = 0; loop < m; ++loop, ++arg) {
        edfct->dp_y[loop] = TARG;
      }
      ext_retc = edfct->ADOLC_EXT_FCT_COMPLETE;
      MINDEC(ret_c, ext_retc);

      res = tape.lowestYLoc_rev();
      for (loop = 0; loop < m; ++loop) {
        FOR_0_LE_l_LT_p { ADJOINT_BUFFER_RES_L = 0.; /* \bar{v}_i = 0 !!! */ }
        ++res;
      }
      res = tape.lowestXLoc_rev();
      for (loop = 0; loop < n; ++loop) {
        ADOLC_EXT_FCT_COPY_ADJOINTS_BACK(ADOLC_EXT_FCT_Z[loop],
                                         ADJOINT_BUFFER_RES);
        ++res;
      }
      if (edfct->dp_y_priorRequired) {
        arg = tape.lowestYLoc_rev() + m - 1;
        for (loop = 0; loop < m; ++loop, --arg) {
          tape.get_taylor(arg);
        }
      }
      if (edfct->dp_x_changes) {
        arg = tape.lowestXLoc_rev() + n - 1;
        for (loop = 0; loop < n; ++loop, --arg) {
          tape.get_taylor(arg);
        }
      }
      tape.traceFlag(oldTraceFlag);

      break;
    case ext_diff_iArr: /* extern differntiated function */
      tape.cp_index(tape.get_locint_r());
      tape.lowestYLoc_rev(tape.get_locint_r());
      tape.lowestXLoc_rev(tape.get_locint_r());
      m = tape.get_locint_r();
      n = tape.get_locint_r();
      tape.ext_diff_fct_index(tape.get_locint_r());
      iArrLength = tape.get_locint_r();
      iArr = new int[iArrLength];
      for (loop = iArrLength - 1; loop >= 0; --loop)
        iArr[loop] = tape.get_locint_r();
      tape.get_locint_r(); /* get it again */
      ADOLC_EXT_FCT_SAVE_NUMDIRS;
      edfct = get_ext_diff_fct(tape.tapeId(), tape.ext_diff_fct_index());

      oldTraceFlag = tape.traceFlag();
      tape.traceFlag(0);

      if (edfct->ADOLC_EXT_FCT_IARR_POINTER == NULL)
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
      arg = tape.lowestYLoc_rev();
      for (loop = 0; loop < m; ++loop) {
        ADOLC_EXT_FCT_COPY_ADJOINTS(ADOLC_EXT_FCT_U[loop], ADJOINT_BUFFER_ARG);
        ++arg;
      }

      arg = tape.lowestXLoc_rev();
      for (loop = 0; loop < n; ++loop) {
        ADOLC_EXT_FCT_COPY_ADJOINTS(ADOLC_EXT_FCT_Z[loop], ADJOINT_BUFFER_ARG);
        ++arg;
      }
      arg = tape.lowestXLoc_rev();
      for (loop = 0; loop < n; ++loop, ++arg) {
        edfct->dp_x[loop] = TARG;
      }
      arg = tape.lowestYLoc_rev();
      for (loop = 0; loop < m; ++loop, ++arg) {
        edfct->dp_y[loop] = TARG;
      }
      ext_retc = edfct->ADOLC_EXT_FCT_IARR_COMPLETE;
      MINDEC(ret_c, ext_retc);

      res = tape.lowestYLoc_rev();
      for (loop = 0; loop < m; ++loop) {
        FOR_0_LE_l_LT_p { ADJOINT_BUFFER_RES_L = 0.; /* \bar{v}_i = 0 !!! */ }
        ++res;
      }
      res = tape.lowestXLoc_rev();
      for (loop = 0; loop < n; ++loop) {
        ADOLC_EXT_FCT_COPY_ADJOINTS_BACK(ADOLC_EXT_FCT_Z[loop],
                                         ADJOINT_BUFFER_RES);
        ++res;
      }
      if (edfct->dp_y_priorRequired) {
        arg = tape.lowestYLoc_rev() + m - 1;
        for (loop = 0; loop < m; ++loop, --arg) {
          tape.get_taylor(arg);
        }
      }
      if (edfct->dp_x_changes) {
        arg = tape.lowestXLoc_rev() + n - 1;
        for (loop = 0; loop < n; ++loop, --arg) {
          tape.get_taylor(arg);
        }
      }
      tape.traceFlag(oldTraceFlag);

      break;
    case ext_diff_v2:
      nout = tape.get_locint_r();
      nin = tape.get_locint_r();
      insz = new locint[2 * (nin + nout)];
      outsz = insz + nin;
      tape.lowestXLoc_ext_v2(outsz + nout);
      tape.lowestYLoc_ext_v2(outsz + nout + nin);
      for (loop = nout - 1; loop >= 0; --loop) {
        tape.lowestYLoc_ext_v2()[loop] = tape.get_locint_r();
        outsz[loop] = tape.get_locint_r();
      }
      for (loop = nin - 1; loop >= 0; --loop) {
        tape.lowestXLoc_ext_v2()[loop] = tape.get_locint_r();
        insz[loop] = tape.get_locint_r();
      }
      tape.get_locint_r(); /* nout again */
      tape.get_locint_r(); /* nin again */
      iArrLength = tape.get_locint_r();
      iArr = new int[iArrLength];
      for (loop = iArrLength - 1; loop >= 0; --loop)
        iArr[loop] = tape.get_locint_r();
      tape.get_locint_r(); /* iArrLength again */
      tape.ext_diff_fct_index(tape.get_locint_r());
      ADOLC_EXT_FCT_SAVE_NUMDIRS;
      edfct2 = get_ext_diff_fct_v2(tape.tapeId(), tape.ext_diff_fct_index());
      oldTraceFlag = tape.traceFlag();
      tape.traceFlag(0);

      if (edfct2->ADOLC_EXT_FCT_POINTER == NULL)
        ADOLCError::fail(ADOLCError::ErrorType::EXT_DIFF_NULLPOINTER_FUNCTION,
                         CURRENT_LOCATION);
      if (nout > 0) {
        if (ADOLC_EXT_FCT_V2_U == NULL)
          ADOLCError::fail(ADOLCError::ErrorType::EXT_DIFF_NULLPOINTER_ARGUMENT,
                           CURRENT_LOCATION);
        if (edfct2->y == NULL)
          ADOLCError::fail(ADOLCError::ErrorType::EXT_DIFF_NULLPOINTER_ARGUMENT,
                           CURRENT_LOCATION);
      }
      if (nin > 0) {
        if (ADOLC_EXT_FCT_V2_Z == NULL)
          ADOLCError::fail(ADOLCError::ErrorType::EXT_DIFF_NULLPOINTER_ARGUMENT,
                           CURRENT_LOCATION);
        if (edfct2->x == NULL)
          ADOLCError::fail(ADOLCError::ErrorType::EXT_DIFF_NULLPOINTER_ARGUMENT,
                           CURRENT_LOCATION);
      }
      for (oloop = 0; oloop < nout; ++oloop) {
        arg = tape.lowestYLoc_ext_v2()[oloop];
        for (loop = 0; loop < outsz[oloop]; ++loop) {
          ADOLC_EXT_FCT_COPY_ADJOINTS(ADOLC_EXT_FCT_V2_U_LOOP,
                                      ADJOINT_BUFFER_ARG);
          edfct2->y[oloop][loop] = TARG;
          ++arg;
        }
      }
      for (oloop = 0; oloop < nin; ++oloop) {
        arg = tape.lowestXLoc_ext_v2()[oloop];
        for (loop = 0; loop < insz[oloop]; ++loop) {
          ADOLC_EXT_FCT_COPY_ADJOINTS(ADOLC_EXT_FCT_V2_Z_LOOP,
                                      ADJOINT_BUFFER_ARG);
          edfct2->x[oloop][loop] = TARG;
          ++arg;
        }
      }
      ext_retc = edfct2->ADOLC_EXT_FCT_V2_COMPLETE;
      MINDEC(ret_c, ext_retc);
      for (oloop = 0; oloop < nout; ++oloop) {
        res = tape.lowestYLoc_ext_v2()[oloop];
        for (loop = 0; loop < outsz[oloop]; ++loop) {
          FOR_0_LE_l_LT_p {
            ADJOINT_BUFFER_RES_L = 0.0; /* \bar{v}_i = 0 !!! */
          }
          ++res;
        }
      }
      for (oloop = 0; oloop < nin; ++oloop) {
        res = tape.lowestXLoc_ext_v2()[oloop];
        for (loop = 0; loop < insz[oloop]; ++loop) {
          ADOLC_EXT_FCT_COPY_ADJOINTS_BACK(ADOLC_EXT_FCT_V2_Z_LOOP,
                                           ADJOINT_BUFFER_RES);
          ++res;
        }
      }
      if (edfct2->dp_y_priorRequired) {
        for (oloop = nout - 1; oloop >= 0; --oloop) {
          arg = tape.lowestYLoc_ext_v2()[oloop] + outsz[oloop] - 1;
          for (loop = outsz[oloop] - 1; loop >= 0; --loop) {
            tape.get_taylor(arg);
            --arg;
          }
        }
      }
      if (edfct2->dp_x_changes) {
        for (oloop = nin - 1; oloop >= 0; --oloop) {
          arg = tape.lowestXLoc_ext_v2()[oloop] + insz[oloop] - 1;
          for (loop = insz[oloop] - 1; loop >= 0; --loop) {
            tape.get_taylor(arg);
            --arg;
          }
        }
      }
      tape.traceFlag(oldTraceFlag);
      delete iArr;
      delete insz;
      insz = nullptr;
      iArr = nullptr;
      outsz = nullptr;
      tape.lowestXLoc_ext_v2(nullptr);
      tape.lowestYLoc_ext_v2(nullptr);
      break;
#ifdef ADOLC_MEDIPACK_SUPPORT
      /*--------------------------------------------------------------------------*/
    case medi_call: {
      locint mediIndex = tape.get_locint_r();
      short tapeId = tape.tapeId();

#if defined _FOS_
      mediCallHandleReverse(tapeId, mediIndex, rp_T, &ADJOINT_BUFFER, 1);
#elif defined _FOV_
      mediCallHandleReverse(tapeId, mediIndex, rp_T, ADJOINT_BUFFER, p);
#endif
      break;
    }
#endif
#ifdef ADOLC_AMPI_SUPPORT
      /*--------------------------------------------------------------------------*/
    case ampi_send: {
      BW_AMPI_Send(buf, count, datatype, src, tag, pairedWith, comm);
      break;
    }
    case ampi_recv: {
      BW_AMPI_Recv(buf, count, datatype, src, tag, pairedWith, comm, status);
      break;
    }
    case ampi_isend: {
      BW_AMPI_Isend(buf, count, datatype, src, tag, pairedWith, comm, &request);
      break;
    }
    case ampi_irecv: {
      BW_AMPI_Irecv(buf, count, datatype, src, tag, pairedWith, comm, &request);
      break;
    }
    case ampi_wait: {
      BW_AMPI_Wait(&request, status);
      break;
    }
    case ampi_barrier: {
      BW_AMPI_Barrier(comm);
      break;
    }
    case ampi_gather: {
      BW_AMPI_Gather(buf, count, datatype, rbuf, rcount, rtype, src, comm);
      break;
    }
    case ampi_scatter: {
      BW_AMPI_Scatter(rbuf, rcount, rtype, buf, count, datatype, src, comm);
      break;
    }
    case ampi_allgather: {
      BW_AMPI_Allgather(buf, count, datatype, rbuf, rcount, rtype, comm);
      break;
    }
    case ampi_gatherv: {
      BW_AMPI_Gatherv(buf, count, datatype, rbuf, NULL, NULL, rtype, src, comm);
      break;
    }
    case ampi_scatterv: {
      BW_AMPI_Scatterv(rbuf, NULL, NULL, rtype, buf, count, datatype, src,
                       comm);
      break;
    }
    case ampi_allgatherv: {
      BW_AMPI_Allgatherv(buf, count, datatype, rbuf, NULL, NULL, rtype, comm);
      break;
    }
    case ampi_bcast: {
      BW_AMPI_Bcast(buf, count, datatype, src, comm);
      break;
    }
    case ampi_reduce: {
      BWB_AMPI_Reduce(buf, rbuf, count, datatype, op, src, comm);
      break;
    }
    case ampi_allreduce: {
      BW_AMPI_Allreduce(buf, rbuf, count, datatype, op, comm);
      break;
    }
#endif
#endif /* !_INT_REV_ */
      /*--------------------------------------------------------------------------*/
    default: /* default */
      /*             Die here, we screwed up     */

      ADOLCError::fail(ADOLCError::ErrorType::NO_SUCH_OP, CURRENT_LOCATION,
                       ADOLCError::FailInfo{.info7 = operation});
      break;
    } /* endswitch */

    /* Get the next operation */
    operation = tape.get_op_r();
  } /* endwhile */

  /* clean up */
#ifdef _FOS_
  myfree1(rp_A);
  myfree1(rp_T);
  tape.rp_T(nullptr);
  tape.rp_A(nullptr);
#endif
#ifdef _FOV_
  myfree2(rpp_A);
  myfree1(rp_T);
  tape.rp_T(nullptr);
  tape.rpp_A(nullptr);
#endif
#ifdef _INT_REV_
  myfree2_ulong(upp_A);
#ifdef _TIGHT_
  tape.upp_A(nullptr);
  myfree1(rp_T);
  tape.rp_T(nullptr);
#endif
#endif

  tape.workMode(TapeInfos::ADOLC_NO_MODE);
  tape.end_sweep();

  return ret_c;
}

/****************************************************************************/
/*                                                               THAT'S ALL */

END_C_DECLS

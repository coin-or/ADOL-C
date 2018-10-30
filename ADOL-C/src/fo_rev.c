/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     fo_rev.c
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

#define RESULTS(l,indexi)  results[indexi]
#define LAGRANGE(l,indexd) lagrange[indexd]
#define RESULTSTRANS(l,indexi)  results[indexi]
#define LAGRANGETRANS(l,indexd) lagrange[indexd]

/*--------------------------------------------------------------------------*/
#elif _FOV_
#define GENERATED_FILENAME "fov_reverse"

#define _ADOLC_VECTOR_

#define RESULTS(l,indexi) results[l][indexi]
#define LAGRANGE(l,indexd) lagrange[l][indexd]
#define RESULTSTRANS(l,indexi) results[indexi][l]
#define LAGRANGETRANS(l,indexd) lagrange[indexd][l]

#else
#if defined(_INT_REV_)
#if defined(_TIGHT_)
#define GENERATED_FILENAME "int_reverse_t"
#endif
#if defined(_NTIGHT_)
#define GENERATED_FILENAME "int_reverse_s"
#endif
#define RESULTS(l,indexi)  results[l][indexi]
#define LAGRANGE(l,indexd) lagrange[l][indexd]
#define RESULTSTRANS(l,indexi) results[indexi][l]
#define LAGRANGETRANS(l,indexd) lagrange[indexd][l]
#else
#error Error ! Define [_FOS_ | _FOV_ | _INT_REV_SAFE_ | _INT_REV_TIGHT_ ]
#endif
#endif
/*--------------------------------------------------------------------------*/
/*                                                     access to variables  */

#ifdef _FOS_
#define ARES       *Ares
#define AARG       *Aarg
#define AARG1      *Aarg1
#define AARG2      *Aarg2

#define ARES_INC   *Ares
#define AARG_INC   *Aarg
#define AARG1_INC  *Aarg1
#define AARG2_INC  *Aarg2

#define ARES_INC_O  Ares
#define AARG_INC_O  /adAarg
#define AARG1_INC_O Aarg1
#define AARG2_INC_O Aarg2

#define ASSIGN_A(a,b)  a = &b;

#else  /* _FOV_ */
#ifdef _FOV_
#define ARES       *Ares
#define AARG       *Aarg
#define AARG1      *Aarg1
#define AARG2      *Aarg2

#define ARES_INC   *Ares++
#define AARG_INC   *Aarg++
#define AARG1_INC  *Aarg1++
#define AARG2_INC  *Aarg2++

#define ARES_INC_O  Ares++
#define AARG_INC_O  Aarg++
#define AARG1_INC_O Aarg1++
#define AARG2_INC_O Aarg2++

#define ASSIGN_A(a,b)  a = b;
#else
#ifdef _INT_REV_
#define ARES       *Ares
#define AARG       *Aarg
#define AARG1      *Aarg1
#define AARG2      *Aarg2

#define ARES_INC   *Ares++
#define AARG_INC   *Aarg++
#define AARG1_INC  *Aarg1++
#define AARG2_INC  *Aarg2++

#define ARES_INC_O  Ares++
#define AARG_INC_O  Aarg++
#define AARG1_INC_O Aarg1++
#define AARG2_INC_O Aarg2++

#define ASSIGN_A(a,b)  a = b;
#endif
#endif
#endif

#define TRES       rp_T[res]
#define TARG       rp_T[arg]
#define TARG1      rp_T[arg1]
#define TARG2      rp_T[arg2]

/*--------------------------------------------------------------------------*/
/*                                                              loop stuff  */
#ifdef _ADOLC_VECTOR_
#define FOR_0_LE_l_LT_p for (l=0; l<p; l++)
#define FOR_p_GT_l_GE_0 for (l=p-1; l>=0; l--)
#else
#ifdef _INT_REV_
#define FOR_0_LE_l_LT_p for (l=0; l<p; l++)
#define FOR_p_GT_l_GE_0 for (l=p-1; l>=0; l--)
#else
#define FOR_0_LE_l_LT_p
#define FOR_p_GT_l_GE_0
#endif
#endif

#ifdef _FOV_
#define FOR_0_LE_l_LT_pk1 for (l=0; l<p; l++)
#else
#ifdef _INT_REV_
#define FOR_0_LE_l_LT_pk1 for (l=0; l<p; l++)
#else
#define FOR_0_LE_l_LT_pk1
#endif
#endif

/* END Macros */


/****************************************************************************/
/*                                                       NECESSARY INCLUDES */
#include <adolc/interfaces.h>
#include <adolc/adalloc.h>
#include "oplate.h"
#include "taping_p.h"
#include <adolc/externfcts.h>
#include "externfcts_p.h"
#include "dvlparms.h"

#include <math.h>
#include <string.h>

#ifdef ADOLC_MEDIPACK_SUPPORT
#include "medipacksupport_p.h"
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
int fos_pl_reverse(short  tnum,     /* tape id */
		   int    depen,     /* consistency chk on # of deps */
		   int    indep,     /* consistency chk on # of indeps */
		   int    swchk,    /* consistency chk on # of switches */
		   int    rownum,   /* required row no. of abs-normal form */
		   double *results) /*  coefficient vectors */
#elif defined(_ABS_NORM_SIG_)
/****************************************************************************/
/* Abs-Normal extended adjoint row computation.                             */
/****************************************************************************/
int fos_pl_sig_reverse(short  tnum,     /* tape id */
		   int    depen,     /* consistency chk on # of deps */
		   int    indep,     /* consistency chk on # of indeps */
		   int    swchk,    /* consistency chk on # of switches */
   	           short   *siggrad,
                   double  *lagrange,
		   double *results) /*  coefficient vectors */
#else
int fos_reverse(short   tnum,       /* tape id */
                int     depen,      /* consistency chk on # of deps */
                int     indep,      /* consistency chk on # of indeps */
                double  *lagrange,
                double  *results)   /*  coefficient vectors */

#endif
#else
#if _FOV_
/****************************************************************************/
/* First-Order Vector Reverse Pass.                                         */
/****************************************************************************/

int fov_reverse(short   tnum,        /* tape id */
                int     depen,       /* consistency chk on # of deps */
                int     indep,       /* consistency chk on # of indeps */
                int     nrows,       /* # of Jacobian rows being calculated */
                double  **lagrange,  /* domain weight vector */
                double  **results)   /* matrix of coefficient vectors */

#else
#if defined(_INT_REV_)
#if defined(_TIGHT_)
/****************************************************************************/
/* First Order Vector version of the reverse mode for bit patterns, tight   */
/****************************************************************************/
int int_reverse_tight(
        short             tnum,  /* tape id                               */
        int               depen, /* consistency chk on # of deps          */
        int               indep, /* consistency chk on # of indeps        */
        int               nrows, /* # of Jacobian rows being calculated   */
        unsigned long int **lagrange,/* domain weight vector[var][row](in)*/
        unsigned long int **results) /* matrix of coeff. vectors[var][row]*/

#endif
#if defined(_NTIGHT_)
/****************************************************************************/
/* First Order Vector version of the reverse mode, bit pattern, safe        */
/****************************************************************************/
int int_reverse_safe(
        short             tnum,  /* tape id                               */
        int               depen, /* consistency chk on # of deps          */
        int               indep, /* consistency chk on # of indeps        */
        int               nrows, /* # of Jacobian rows being calculated   */
        unsigned long int **lagrange,/* domain weight vector[var][row](in)*/
        unsigned long int **results) /* matrix of coeff. vectors[var][row]*/
#endif
#endif
#endif
#endif
{
    /****************************************************************************/
    /*                                                           ALL VARIABLES  */
    unsigned char operation;   /* operation code */
    int ret_c = 3;             /* return value */

    locint size = 0;
    locint res  = 0;
    locint arg  = 0;
    locint arg1 = 0;
    locint arg2 = 0;

#if !defined (_NTIGHT_)
    double coval = 0;
#endif

    int indexi = 0,  indexd = 0;
#if defined(_ABS_NORM_) || defined(_ABS_NORM_SIG_)
    int switchnum;
#endif

    /* loop indices */
#if defined(_FOV_)
    int l;
#endif
#if defined(_INT_REV_)
    int l;
#endif
    int j, ls;

    /* other necessary variables */
#if !defined (_NTIGHT_)
    double r0, r_0;
    int taycheck;
    int numdep,numind;
#endif

    /*--------------------------------------------------------------------------*/
    /* Adjoint stuff */
#ifdef _FOS_
    revreal *rp_A;
    revreal aTmp;
#endif
#ifdef _FOV_
    revreal **rpp_A, *Aqo;
    revreal aTmp;
#endif
#if !defined(_NTIGHT_)
    revreal *rp_T;
#endif /* !_NTIGHT_ */
#if !defined _INT_REV_
    revreal  *Ares, *Aarg, *Aarg1, *Aarg2;
#else
    unsigned long int **upp_A;
    unsigned long int *Ares, *Aarg, *Aarg1, *Aarg2;
    unsigned long int aTmp;
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
# define ADOLC_EXT_FCT_U edfct->dp_U
# define ADOLC_EXT_FCT_Z edfct->dp_Z
# define ADOLC_EXT_FCT_POINTER fos_reverse
# define ADOLC_EXT_FCT_IARR_POINTER fos_reverse_iArr
# define ADOLC_EXT_FCT_COMPLETE \
  fos_reverse(m, edfct->dp_U, n, edfct->dp_Z, edfct->dp_x, edfct->dp_y)
# define ADOLC_EXT_FCT_IARR_COMPLETE \
  fos_reverse_iArr(iArrLength,iArr, m, edfct->dp_U, n, edfct->dp_Z, edfct->dp_x, edfct->dp_y)
# define ADOLC_EXT_FCT_SAVE_NUMDIRS
# define ADOLC_EXT_FCT_V2_U edfct2->up
# define ADOLC_EXT_FCT_V2_Z edfct2->zp
# define ADOLC_EXT_FCT_V2_COMPLETE \
    fos_reverse(iArrLength,iArr,nout,nin,outsz,edfct2->up,insz,edfct2->zp,edfct2->x,edfct2->y,edfct2->context)
#else
# define ADOLC_EXT_FCT_U edfct->dpp_U
# define ADOLC_EXT_FCT_Z edfct->dpp_Z
# define ADOLC_EXT_FCT_POINTER fov_reverse
# define ADOLC_EXT_FCT_IARR_POINTER fov_reverse_iArr
# define ADOLC_EXT_FCT_COMPLETE \
  fov_reverse(m, p, edfct->dpp_U, n, edfct->dpp_Z, edfct->dp_x, edfct->dp_y)
# define ADOLC_EXT_FCT_IARR_COMPLETE \
  fov_reverse_iArr(iArrLength, iArr, m, p, edfct->dpp_U, n, edfct->dpp_Z, edfct->dp_x, edfct->dp_y)
# define ADOLC_EXT_FCT_SAVE_NUMDIRS ADOLC_CURRENT_TAPE_INFOS.numDirs_rev = nrows
# define ADOLC_EXT_FCT_V2_U edfct2->Up
# define ADOLC_EXT_FCT_V2_Z edfct2->Zp
# define ADOLC_EXT_FCT_V2_COMPLETE \
  fov_reverse(iArrLength,iArr,nout,nin,outsz,p,edfct2->Up,insz,edfct2->Zp,edfct2->x,edfct2->y, edfct2->context)
#endif
#if !defined(_INT_REV_)
    locint n, m;
    ext_diff_fct *edfct;
    ext_diff_fct_v2 *edfct2;
    int iArrLength;
    int *iArr;
    int loop,oloop;
    int ext_retc;
    int oldTraceFlag;
    locint *insz, *outsz, nin, nout;
#endif
#ifdef ADOLC_AMPI_SUPPORT
    MPI_Op op;
    void *buf, *rbuf;
    int count, rcount;
    MPI_Datatype datatype, rtype;
    int src; 
    int tag;
    enum AMPI_PairedWith_E pairedWith;
    MPI_Comm comm;
    MPI_Status* status;
    struct AMPI_Request_S request;
#endif
	locint qq;

	ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

#if defined(ADOLC_DEBUG)
    /****************************************************************************/
    /*                                                           DEBUG MESSAGES */
    fprintf(DIAG_OUT,"Call of %s(..) with tag: %d, n: %d, m %d,\n",
            GENERATED_FILENAME, tnum, indep, depen);
#ifdef _ADOLC_VECTOR_
    fprintf(DIAG_OUT,"                    p: %d\n\n",nrows);
#endif

#endif

    /****************************************************************************/
    /*                                                                    INITs */

    /*------------------------------------------------------------------------*/
    /* Set up stuff for the tape */

    /* Initialize the Reverse Sweep */
    init_rev_sweep(tnum);

    failAdditionalInfo3 = depen;
    failAdditionalInfo4 = indep;
    if ( (depen != ADOLC_CURRENT_TAPE_INFOS.stats[NUM_DEPENDENTS]) ||
            (indep != ADOLC_CURRENT_TAPE_INFOS.stats[NUM_INDEPENDENTS]) )
        fail(ADOLC_REVERSE_COUNTS_MISMATCH);

    indexi = ADOLC_CURRENT_TAPE_INFOS.stats[NUM_INDEPENDENTS] - 1;
    indexd = ADOLC_CURRENT_TAPE_INFOS.stats[NUM_DEPENDENTS] - 1;

#if defined(_ABS_NORM_) || defined(_ABS_NORM_SIG_)
    if (! ADOLC_CURRENT_TAPE_INFOS.stats[NO_MIN_MAX] ) {
	fprintf(DIAG_OUT, "ADOL-C error: Tape %d was not created compatible "
		"with %s(..)\n              Please call enableMinMaxUsingAbs() "
		"before trace_on(%d)\n", tnum, GENERATED_FILENAME, tnum);
	adolc_exit(-1,"",__func__,__FILE__,__LINE__);
    }
    else if (swchk != ADOLC_CURRENT_TAPE_INFOS.stats[NUM_SWITCHES]) {
	fprintf(DIAG_OUT, "ADOL-C error: Number of switches passed %d does not "
		"match with the one recorded on tape %d (%zu)\n",swchk,tnum,
		ADOLC_CURRENT_TAPE_INFOS.stats[NUM_SWITCHES]);
	adolc_exit(-1,"",__func__,__FILE__,__LINE__);
    }
    else
	switchnum = swchk - 1;
#endif

    
    /****************************************************************************/
    /*                                                  MEMORY ALLOCATION STUFF */

    /*--------------------------------------------------------------------------*/
#ifdef _FOS_                                                         /* FOS */
    rp_A = (revreal*) calloc(ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES], sizeof(revreal));
    if (rp_A == NULL) fail(ADOLC_MALLOC_FAILED);
    ADOLC_CURRENT_TAPE_INFOS.rp_A = rp_A;
    rp_T = (revreal *)malloc(ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES] *
            sizeof(revreal));
    if (rp_T == NULL) fail(ADOLC_MALLOC_FAILED);
    ADOLC_CURRENT_TAPE_INFOS.workMode = ADOLC_FOS_REVERSE;
#ifdef _ABS_NORM_
    memset(results,0,sizeof(double)*(indep+swchk));
#endif
# define ADJOINT_BUFFER rp_A
# define ADJOINT_BUFFER_ARG_L rp_A[arg]
# define ADJOINT_BUFFER_RES_L rp_A[res]
# define ADJOINT_BUFFER_ARG rp_A[arg]
# define ADJOINT_BUFFER_RES rp_A[res]
# define ADOLC_EXT_FCT_U_L_LOOP edfct->dp_U[loop]
# define ADOLC_EXT_FCT_Z_L_LOOP edfct->dp_Z[loop]
# define ADOLC_EXT_FCT_V2_U_LOOP edfct2->up[oloop][loop]
# define ADOLC_EXT_FCT_V2_Z_LOOP edfct2->zp[oloop][loop]
# define ADOLC_EXT_FCT_COPY_ADJOINTS(dest,src) dest=src
# define ADOLC_EXT_FCT_COPY_ADJOINTS_BACK(dest,src) src=dest

    /*--------------------------------------------------------------------------*/
#else
#if defined _FOV_                                                          /* FOV */
    rpp_A = (revreal**)malloc(ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES] *
            sizeof(revreal*));
    if (rpp_A == NULL) fail(ADOLC_MALLOC_FAILED);
    Aqo = (revreal*)malloc(ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES] * p *
            sizeof(revreal));
    if (Aqo == NULL) fail(ADOLC_MALLOC_FAILED);
    for (j=0; j<ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES]; j++) {
        rpp_A[j] = Aqo + j*p;
    }
    ADOLC_CURRENT_TAPE_INFOS.rpp_A = rpp_A;
    rp_T = (revreal *)malloc(ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES] *
            sizeof(revreal));
    if (rp_T == NULL) fail(ADOLC_MALLOC_FAILED);
    ADOLC_CURRENT_TAPE_INFOS.workMode = ADOLC_FOV_REVERSE;
#if !defined(ADOLC_USE_CALLOC)
    c_Ptr = (char *) ADOLC_GLOBAL_TAPE_VARS.dpp_A;
    *c_Ptr = 0;
    memcpy(c_Ptr + 1, c_Ptr, sizeof(double) * p *
            ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES] - 1);
#endif
# define ADJOINT_BUFFER rpp_A
# define ADJOINT_BUFFER_ARG_L rpp_A[arg][l]
# define ADJOINT_BUFFER_RES_L rpp_A[res][l]
# define ADJOINT_BUFFER_ARG rpp_A[arg]
# define ADJOINT_BUFFER_RES rpp_A[res]
# define ADOLC_EXT_FCT_U_L_LOOP edfct->dpp_U[l][loop]
# define ADOLC_EXT_FCT_Z_L_LOOP edfct->dpp_Z[l][loop]
# define ADOLC_EXT_FCT_V2_U_LOOP edfct2->Up[oloop][loop]
# define ADOLC_EXT_FCT_V2_Z_LOOP edfct2->Zp[oloop][loop]
# define ADOLC_EXT_FCT_COPY_ADJOINTS(dest,src) dest=src
# define ADOLC_EXT_FCT_COPY_ADJOINTS_BACK(dest,src)
#else
#if defined _INT_REV_
    upp_A = myalloc2_ulong(ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES], p);
#if defined _TIGHT_
    ADOLC_CURRENT_TAPE_INFOS.upp_A = upp_A;
    rp_T = (revreal *)malloc(ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES] *
			     sizeof(revreal));
    if (rp_T == NULL) fail(ADOLC_MALLOC_FAILED);
#endif
# define ADJOINT_BUFFER upp_A
# define ADJOINT_BUFFER_ARG_L upp_A[arg][l]
# define ADJOINT_BUFFER_RES_L upp_A[res][l]
#endif
#endif
#endif

    /****************************************************************************/
    /*                                                    TAYLOR INITIALIZATION */

#if !defined(_NTIGHT_)

    ADOLC_CURRENT_TAPE_INFOS.rp_T = rp_T;

    taylor_back(tnum, &numdep, &numind, &taycheck);

    if (taycheck < 0) {
        fprintf(DIAG_OUT,"\n ADOL-C error: reverse fails because it was not"
                " preceded\nby a forward sweep with degree>0, keep=1!\n");
        adolc_exit(-2,"",__func__,__FILE__,__LINE__);
    };

    if((numdep != depen)||(numind != indep))
        fail(ADOLC_REVERSE_TAYLOR_COUNTS_MISMATCH);

#endif /* !_NTIGHT_ */


    /****************************************************************************/
    /*                                                            REVERSE SWEEP */
    operation=get_op_r();
    while (operation != start_of_tape) { /* Switch statement to execute the operations in Reverse */

        switch (operation) {


                /****************************************************************************/
                /*                                                                  MARKERS */

                /*--------------------------------------------------------------------------*/
            case end_of_op:                                          /* end_of_op */
                get_op_block_r();
                operation = get_op_r();
                /* Skip next operation, it's another end_of_op */
                break;

                /*--------------------------------------------------------------------------*/
            case end_of_int:                                        /* end_of_int */
                get_loc_block_r(); /* Get the next int block */
                break;

                /*--------------------------------------------------------------------------*/
            case end_of_val:                                        /* end_of_val */
                get_val_block_r(); /* Get the next val block */
                break;

                /*--------------------------------------------------------------------------*/
            case start_of_tape:                                  /* start_of_tape */
                break;
            case end_of_tape:                                      /* end_of_tape */
                discard_params_r();
                break;


                /****************************************************************************/
                /*                                                               COMPARISON */

                /*--------------------------------------------------------------------------*/
            case eq_zero  :                                            /* eq_zero */
                arg   = get_locint_r();

#if !defined(_NTIGHT_)
                ret_c = 0;
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case neq_zero :                                           /* neq_zero */
            case gt_zero  :                                            /* gt_zero */
            case lt_zero :                                             /* lt_zero */
                arg   = get_locint_r();
                break;

                /*--------------------------------------------------------------------------*/
            case ge_zero :                                             /* ge_zero */
            case le_zero :                                             /* le_zero */
                arg   = get_locint_r();

#if !defined(_NTIGHT_)
                if (TARG == 0)
                    ret_c = 0;
#endif /* !_NTIGHT_ */
                break;


                /****************************************************************************/
                /*                                                              ASSIGNMENTS */

                /*--------------------------------------------------------------------------*/
            case assign_a:           /* assign an adouble variable an    assign_a */
                /* adouble value. (=) */
                res = get_locint_r();
                arg = get_locint_r();

                ASSIGN_A( Aarg, ADJOINT_BUFFER[arg])
                ASSIGN_A( Ares, ADJOINT_BUFFER[res])

                FOR_0_LE_l_LT_p
                {
#if defined(_INT_REV_)                 
		  AARG_INC |= ARES;
                  ARES_INC = 0;
#else
                  AARG_INC += ARES;
                  ARES_INC = 0.0;
#endif
            }

#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case assign_d:            /* assign an adouble variable a    assign_d */
                /* double value. (=) */
                res   = get_locint_r();
#if !defined(_NTIGHT_)
                coval = get_val_r();
#endif /* !_NTIGHT_ */

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])

                FOR_0_LE_l_LT_p
#if defined(_INT_REV_)                 
                ARES_INC = 0;
#else
                ARES_INC = 0.0;
#endif

#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case neg_sign_p:
            case recipr_p:
            case assign_p:            /* assign an adouble variable a    assign_d */
                /* double value. (=) */
                res   = get_locint_r();
                arg   = get_locint_r();
#if !defined(_NTIGHT_)
                coval = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore[arg];
#endif /* !_NTIGHT_ */

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])

                FOR_0_LE_l_LT_p
#if defined(_INT_REV_)                 
                ARES_INC = 0;
#else
                ARES_INC = 0.0;
#endif

#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
                break;
                /*--------------------------------------------------------------------------*/
            case assign_d_zero:  /* assign an adouble variable a    assign_d_zero */
            case assign_d_one:   /* double value (0 or 1). (=)       assign_d_one */
                res   = get_locint_r();

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])

                FOR_0_LE_l_LT_p
#if defined(_INT_REV_)                 
                ARES_INC = 0;
#else
                ARES_INC = 0.0;
#endif

#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case assign_ind:       /* assign an adouble variable an    assign_ind */
                /* independent double value (<<=) */
                res = get_locint_r();

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])

                if (ADOLC_CURRENT_TAPE_INFOS.in_nested_ctx) {
                FOR_0_LE_l_LT_p
                    RESULTSTRANS(l,indexi) += ARES_INC;
                } else {
                FOR_0_LE_l_LT_p
                    RESULTS(l,indexi) = ARES_INC;
                }
#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
                indexi--;
                break;

                /*--------------------------------------------------------------------------*/
            case assign_dep:           /* assign a float variable a    assign_dep */
                /* dependent adouble value. (>>=) */
                res = get_locint_r();

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])

#if defined(_ABS_NORM_)
		if (indexd + swchk == rownum)
		    ARES = 1.0;
		else
		    ARES = 0.0;
#else
                if (ADOLC_CURRENT_TAPE_INFOS.in_nested_ctx) {
                    FOR_0_LE_l_LT_p {
                        ARES_INC = LAGRANGETRANS(l,indexd);
                        LAGRANGETRANS(l,indexd) = 0.0;
                    }
                } else {
                FOR_0_LE_l_LT_p
                    ARES_INC = LAGRANGE(l,indexd);
                }
#endif
                indexd--;
                break;


                /****************************************************************************/
                /*                                                   OPERATION + ASSIGNMENT */

                /*--------------------------------------------------------------------------*/
            case eq_plus_d:            /* Add a floating point to an    eq_plus_d */
                /* adouble. (+=) */
                res   = get_locint_r();
#if !defined(_NTIGHT_)
                coval = get_val_r();

                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case eq_plus_p:            /* Add a floating point to an    eq_plus_d */
                /* adouble. (+=) */
                res   = get_locint_r();
                arg   = get_locint_r();
#if !defined(_NTIGHT_)
                coval = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore[arg];

                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case eq_plus_a:             /* Add an adouble to another    eq_plus_a */
                /* adouble. (+=) */
                res = get_locint_r();
                arg = get_locint_r();

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg, ADJOINT_BUFFER[arg]);

                FOR_0_LE_l_LT_p
#if defined(_INT_REV_)
		AARG_INC |= ARES_INC;
#else
                AARG_INC += ARES_INC;
#endif

#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case eq_min_d:       /* Subtract a floating point from an    eq_min_d */
                /* adouble. (-=) */
                res   = get_locint_r();
#if !defined(_NTIGHT_)
                coval = get_val_r();

                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
               break;

                /*--------------------------------------------------------------------------*/
            case eq_min_p:       /* Subtract a floating point from an    eq_min_d */
                /* adouble. (-=) */
                res   = get_locint_r();
                arg   = get_locint_r();
#if !defined(_NTIGHT_)
                coval = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore[arg];

                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
               break;

                /*--------------------------------------------------------------------------*/
            case eq_min_a:        /* Subtract an adouble from another    eq_min_a */
                /* adouble. (-=) */
                res = get_locint_r();
                arg = get_locint_r();

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg, ADJOINT_BUFFER[arg])

                FOR_0_LE_l_LT_p
#if defined(_INT_REV_)
		AARG_INC |= ARES_INC;
#else
                AARG_INC -= ARES_INC;
#endif

#if !defined(_NTIGHT_)
               ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case eq_mult_d:              /* Multiply an adouble by a    eq_mult_d */
                /* flaoting point. (*=) */
                res   = get_locint_r();
#if !defined(_NTIGHT_)
                coval = get_val_r();
#endif /* !_NTIGHT_ */

#if!defined(_INT_REV_)
                ASSIGN_A( Ares, ADJOINT_BUFFER[res])

                FOR_0_LE_l_LT_p
                ARES_INC *= coval;
#endif

#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case eq_mult_p:              /* Multiply an adouble by a    eq_mult_p */
                /* flaoting point. (*=) */
                res   = get_locint_r();
                arg   = get_locint_r();
#if !defined(_NTIGHT_)
                coval = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore[arg];
#endif /* !_NTIGHT_ */

#if!defined(_INT_REV_)
                ASSIGN_A( Ares, ADJOINT_BUFFER[res])

                FOR_0_LE_l_LT_p
                ARES_INC *= coval;
#endif

#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case eq_mult_a:       /* Multiply one adouble by another    eq_mult_a */
                /* (*=) */
                res = get_locint_r();
                arg = get_locint_r();

#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg, ADJOINT_BUFFER[arg])

                FOR_0_LE_l_LT_p
#if defined(_INT_REV_)
                AARG_INC |= ARES_INC;
#else
                { aTmp = ARES;
                  /* olvo 980713 nn: ARES = 0.0; */
                  ARES_INC =  (aTmp==0)?0:(aTmp * TARG);
                  AARG_INC += (aTmp==0)?0:(aTmp * TRES);
                }
#endif      
		break;

                /*--------------------------------------------------------------------------*/
            case incr_a:                        /* Increment an adouble    incr_a */
            case decr_a:                        /* Increment an adouble    decr_a */
                res   = get_locint_r();

#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
                break;


                /****************************************************************************/
                /*                                                        BINARY OPERATIONS */

                /*--------------------------------------------------------------------------*/
            case plus_a_a:                 /* : Add two adoubles. (+)    plus a_a */
                res  = get_locint_r();
                arg2 = get_locint_r();
                arg1 = get_locint_r();

                ASSIGN_A( Ares,  ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg1, ADJOINT_BUFFER[arg1])
                ASSIGN_A( Aarg2, ADJOINT_BUFFER[arg2])

                FOR_0_LE_l_LT_p
                { aTmp = ARES;
#if defined(_INT_REV_)
                  ARES_INC = 0;
                  AARG1_INC |= aTmp;
                  AARG2_INC |= aTmp;
#else
                  ARES_INC = 0.0;
                  AARG1_INC += aTmp;
                  AARG2_INC += aTmp;
#endif
            }

#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case plus_d_a:             /* Add an adouble and a double    plus_d_a */
                /* (+) */
                res   = get_locint_r();
                arg   = get_locint_r();
#if !defined(_NTIGHT_)
                coval = get_val_r();
#endif /* !_NTIGHT_ */

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg, ADJOINT_BUFFER[arg])

                FOR_0_LE_l_LT_p
                { aTmp = ARES;
#if defined(_INT_REV_)
                  ARES_INC = 0;
                  AARG_INC |= aTmp;
#else
                  ARES_INC = 0.0;
                  AARG_INC += aTmp;
#endif
            }

#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case plus_a_p:             /* Add an adouble and a double    plus_a_p */
            case min_a_p:                /* Subtract an adouble from a    min_d_a */
                /* (+) */
                res   = get_locint_r();
                arg1  = get_locint_r();
                arg   = get_locint_r();
#if !defined(_NTIGHT_)
                coval = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore[arg1];
#endif /* !_NTIGHT_ */

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg, ADJOINT_BUFFER[arg])

                FOR_0_LE_l_LT_p
                { aTmp = ARES;
#if defined(_INT_REV_)
                  ARES_INC = 0;
                  AARG_INC |= aTmp;
#else
                  ARES_INC = 0.0;
                  AARG_INC += aTmp;
#endif
            }

#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case min_a_a:              /* Subtraction of two adoubles    min_a_a */
                /* (-) */
                res  = get_locint_r();
                arg2 = get_locint_r();
                arg1 = get_locint_r();

                ASSIGN_A( Ares,  ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg1, ADJOINT_BUFFER[arg1])
                ASSIGN_A( Aarg2, ADJOINT_BUFFER[arg2])

                FOR_0_LE_l_LT_p
                { aTmp = ARES;
#if defined(_INT_REV_)
                  ARES_INC = 0;
                  AARG1_INC |= aTmp;
                  AARG2_INC |= aTmp;
#else
                  ARES_INC = 0.0;
                  AARG1_INC += aTmp;
                  AARG2_INC -= aTmp;
#endif
            }

#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case min_d_a:                /* Subtract an adouble from a    min_d_a */
                /* double (-) */
                res   = get_locint_r();
                arg   = get_locint_r();
#if !defined(_NTIGHT_)
                coval = get_val_r();
#endif /* !_NTIGHT_ */

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg, ADJOINT_BUFFER[arg])

                FOR_0_LE_l_LT_p
                { aTmp = ARES;
#if defined(_INT_REV_)
                  ARES_INC = 0;
                  AARG_INC |= aTmp;
#else
                  ARES_INC = 0.0;
                  AARG_INC -= aTmp;
#endif
            }

#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case mult_a_a:               /* Multiply two adoubles (*)    mult_a_a */
                res  = get_locint_r();
                arg2 = get_locint_r();
                arg1 = get_locint_r();

#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */

                ASSIGN_A( Ares,  ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg2, ADJOINT_BUFFER[arg2])
                ASSIGN_A( Aarg1, ADJOINT_BUFFER[arg1])

                FOR_0_LE_l_LT_p
                { aTmp = ARES;
#if defined(_INT_REV_)
                  ARES_INC = 0;
                  AARG2_INC |= aTmp;
                  AARG1_INC |= aTmp;
#else
                  ARES_INC = 0.0;
                  AARG2_INC += (aTmp==0)?0:(aTmp * TARG1);
                  AARG1_INC += (aTmp==0)?0:(aTmp * TARG2);
#endif
            }
                break;

                /*--------------------------------------------------------------------------*/
                /* olvo 991122: new op_code with recomputation */
            case eq_plus_prod:   /* increment a product of           eq_plus_prod */
                /* two adoubles (*) */
                res  = get_locint_r();
                arg2 = get_locint_r();
                arg1 = get_locint_r();

                ASSIGN_A( Ares,  ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg2, ADJOINT_BUFFER[arg2])
                ASSIGN_A( Aarg1, ADJOINT_BUFFER[arg1])

#if !defined(_NTIGHT_)
                /* RECOMPUTATION */
                TRES -= TARG1*TARG2;
#endif /* !_NTIGHT_ */

                FOR_0_LE_l_LT_p
                { 
#if defined(_INT_REV_)
                  AARG2_INC |= ARES;
                  AARG1_INC |= ARES_INC;
#else
		  AARG2_INC += ARES    * TARG1;
                  AARG1_INC += ARES_INC * TARG2;
#endif
            }
                break;

                /*--------------------------------------------------------------------------*/
                /* olvo 991122: new op_code with recomputation */
            case eq_min_prod:    /* decrement a product of            eq_min_prod */
                /* two adoubles (*) */
                res  = get_locint_r();
                arg2 = get_locint_r();
                arg1 = get_locint_r();

                ASSIGN_A( Ares,  ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg2, ADJOINT_BUFFER[arg2])
                ASSIGN_A( Aarg1, ADJOINT_BUFFER[arg1])

#if !defined(_NTIGHT_)
                /* RECOMPUTATION */
                TRES += TARG1*TARG2;
#endif /* !_NTIGHT_ */

                FOR_0_LE_l_LT_p
                { 
#if defined(_INT_REV_)
                  AARG2_INC |= ARES;
                  AARG1_INC |= ARES_INC;
#else                  
                  AARG2_INC -= ARES    * TARG1;
                  AARG1_INC -= ARES_INC * TARG2;
#endif
            }
                break;

                /*--------------------------------------------------------------------------*/
            case mult_d_a:         /* Multiply an adouble by a double    mult_d_a */
                /* (*) */
                res   = get_locint_r();
                arg   = get_locint_r();
#if !defined(_NTIGHT_)
                coval = get_val_r();
#endif /* !_NTIGHT_ */

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg, ADJOINT_BUFFER[arg])

                FOR_0_LE_l_LT_p
                { aTmp = ARES;
#if defined(_INT_REV_)
                  ARES_INC = 0;
		  AARG_INC |= aTmp;
#else
                  ARES_INC = 0.0;
                  AARG_INC += (aTmp==0)?0:(coval * aTmp);
#endif
            }

#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case mult_a_p:         /* Multiply an adouble by a double    mult_a_p */
                /* (*) */
                res   = get_locint_r();
                arg1  = get_locint_r();
                arg   = get_locint_r();
#if !defined(_NTIGHT_)
                coval = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore[arg1];
#endif /* !_NTIGHT_ */

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg, ADJOINT_BUFFER[arg])

                FOR_0_LE_l_LT_p
                { aTmp = ARES;
#if defined(_INT_REV_)
                  ARES_INC = 0;
		  AARG_INC |= aTmp;
#else
                  ARES_INC = 0.0;
                  AARG_INC += (aTmp==0)?0:(coval * aTmp);
#endif
            }

#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case div_a_a:           /* Divide an adouble by an adouble    div_a_a */
                /* (/) */
                res  = get_locint_r();
                arg2 = get_locint_r();
                arg1 = get_locint_r();

                ASSIGN_A( Ares,  ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg2, ADJOINT_BUFFER[arg2])
                ASSIGN_A( Aarg1, ADJOINT_BUFFER[arg1])

                /* olvo 980922 changed order to allow x=y/x */
#if !defined(_NTIGHT_)
                r_0 = -TRES;
                ADOLC_GET_TAYLOR(res);
                r0  = 1.0 / TARG2;
                r_0 *= r0;
#endif /* !_NTIGHT_ */

                FOR_0_LE_l_LT_p
                { aTmp = ARES;
#if defined(_INT_REV_)
                  ARES_INC = 0;
                  AARG1_INC |= aTmp;
                  AARG2_INC |= aTmp;
#else
                  ARES_INC = 0.0;
                  AARG1_INC += (aTmp==0)?0:(aTmp * r0);
                  AARG2_INC += (aTmp==0)?0:(aTmp * r_0);
#endif
            }

                break;

                /*--------------------------------------------------------------------------*/
            case div_d_a:             /* Division double - adouble (/)    div_d_a */
                res   = get_locint_r();
                arg   = get_locint_r();

#if !defined(_NTIGHT_)
                coval = get_val_r();
#endif /* !_NTIGHT_ */

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg, ADJOINT_BUFFER[arg])

#if !defined(_NTIGHT_)
                /* olvo 980922 changed order to allow x=d/x */
                r0 = -TRES;
                if (arg == res)
                    ADOLC_GET_TAYLOR(arg);
                r0 /= TARG;
#endif /* !_NTIGHT_ */

                FOR_0_LE_l_LT_p
                { aTmp = ARES;
#if defined(_INT_REV_)
                  ARES_INC = 0;
                  AARG_INC |= aTmp;
#else
                  ARES_INC = 0.0;
                  AARG_INC += (aTmp==0)?0:(aTmp * r0);
#endif
                }

#if !defined(_NTIGHT_)
		if (arg != res)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
                break;


                /****************************************************************************/
            case div_p_a:             /* Division double - adouble (/)    div_p_a */
                res   = get_locint_r();
                arg1  = get_locint_r();
                arg   = get_locint_r();

#if !defined(_NTIGHT_)
                coval = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore[arg1];
#endif /* !_NTIGHT_ */

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg, ADJOINT_BUFFER[arg])

#if !defined(_NTIGHT_)
                /* olvo 980922 changed order to allow x=d/x */
                r0 = -TRES;
                if (arg == res)
                    ADOLC_GET_TAYLOR(arg);
                r0 /= TARG;
#endif /* !_NTIGHT_ */

                FOR_0_LE_l_LT_p
                { aTmp = ARES;
#if defined(_INT_REV_)
                  ARES_INC = 0;
                  AARG_INC |= aTmp;
#else
                  ARES_INC = 0.0;
                  AARG_INC += (aTmp==0)?0:(aTmp * r0);
#endif
                }

#if !defined(_NTIGHT_)
		if (arg != res)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
                break;


                /****************************************************************************/
                /*                                                         SIGN  OPERATIONS */

                /*--------------------------------------------------------------------------*/
            case pos_sign_a:                                        /* pos_sign_a */
                res   = get_locint_r();
                arg   = get_locint_r();

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg, ADJOINT_BUFFER[arg])

                FOR_0_LE_l_LT_p
                { aTmp = ARES;
#if defined(_INT_REV_)
                  ARES_INC = 0;
                  AARG_INC |= aTmp;
#else
                  ARES_INC = 0.0;
                  AARG_INC += aTmp;
#endif
            }

#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case neg_sign_a:                                        /* neg_sign_a */
                res   = get_locint_r();
                arg   = get_locint_r();

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg, ADJOINT_BUFFER[arg])

                FOR_0_LE_l_LT_p
                { aTmp = ARES;
#if defined(_INT_REV_)
                  ARES_INC = 0;
                  AARG_INC |= aTmp;
#else
                  ARES_INC = 0.0;
                  AARG_INC -= aTmp;
#endif
            }

#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
                break;


                /****************************************************************************/
                /*                                                         UNARY OPERATIONS */

                /*--------------------------------------------------------------------------*/
            case exp_op:                          /* exponent operation    exp_op */
                res = get_locint_r();
                arg = get_locint_r();

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg, ADJOINT_BUFFER[arg])

                FOR_0_LE_l_LT_p
                { aTmp = ARES;
#if defined(_INT_REV_)
                  ARES_INC = 0;
                  AARG_INC |= aTmp;
#else
                  ARES_INC = 0.0;
                  AARG_INC += (aTmp==0)?0:(aTmp*TRES);
#endif
            }

#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case sin_op:                              /* sine operation    sin_op */
                res  = get_locint_r();
                arg2 = get_locint_r();
                arg1 = get_locint_r();

                ASSIGN_A( Ares,  ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg1, ADJOINT_BUFFER[arg1])

                FOR_0_LE_l_LT_p
                { aTmp = ARES;
#if defined(_INT_REV_)
                  ARES_INC = 0;
                  AARG1_INC |= aTmp;
#else
                  ARES_INC = 0.0;
                  AARG1_INC += (aTmp==0)?0:(aTmp * TARG2);
#endif
            }

#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
                ADOLC_GET_TAYLOR(arg2); /* olvo 980710 covalue */
                /* NOTE: ADJOINT_BUFFER[arg2] should be 0 already */
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case cos_op:                            /* cosine operation    cos_op */
                res  = get_locint_r();
                arg2 = get_locint_r();
                arg1 = get_locint_r();

                ASSIGN_A( Ares,  ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg1, ADJOINT_BUFFER[arg1])

                FOR_0_LE_l_LT_p
                { aTmp = ARES;
#if defined(_INT_REV_)
                  ARES_INC = 0;
                  AARG1_INC |= aTmp;
#else
                  ARES_INC = 0.0;
                  AARG1_INC -= (aTmp==0)?0:(aTmp * TARG2);
#endif
            }

#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
                ADOLC_GET_TAYLOR(arg2); /* olvo 980710 covalue */
                /* NOTE ADJOINT_BUFFER[arg2] should be 0 already */
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case atan_op:                                             /* atan_op  */
            case asin_op:                                             /* asin_op  */
            case acos_op:                                             /* acos_op  */
            case asinh_op:                                            /* asinh_op */
            case acosh_op:                                            /* acosh_op */
            case atanh_op:                                            /* atanh_op */
            case erf_op:                                              /* erf_op   */
                res  = get_locint_r();
                arg2 = get_locint_r();
                arg1 = get_locint_r();

#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */

                ASSIGN_A( Ares,  ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg1, ADJOINT_BUFFER[arg1])

                FOR_0_LE_l_LT_p
                { aTmp = ARES;
#if defined(_INT_REV_)
                  ARES_INC = 0;
                  AARG1_INC |= aTmp;
#else
                  ARES_INC = 0.0;
                  AARG1_INC += (aTmp==0)?0:(aTmp * TARG2);
#endif
                }
                break;

                /*--------------------------------------------------------------------------*/
            case log_op:                                                /* log_op */
                res = get_locint_r();
                arg = get_locint_r();

#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg, ADJOINT_BUFFER[arg])

#if !defined(_INT_REV_)
                r0 = 1.0/TARG;
#endif

                FOR_0_LE_l_LT_p
                { aTmp = ARES;
#if defined(_INT_REV_)
                  ARES_INC = 0;
                  AARG_INC |= aTmp;
#else
                  ARES_INC = 0.0;
                  AARG_INC += (aTmp==0)?0:(aTmp * r0);
#endif
            }
                break;

                /*--------------------------------------------------------------------------*/
            case pow_op:                                                /* pow_op */
                res   = get_locint_r();
                arg   = get_locint_r();
#if !defined(_NTIGHT_)
                coval = get_val_r();
#endif /* !_NTIGHT_ */

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg, ADJOINT_BUFFER[arg])

#if !defined(_NTIGHT_)
                /* olvo 980921 changed order to allow x=pow(x,n) */
                r0 = TRES;
                if (arg == res)
                    ADOLC_GET_TAYLOR(arg);
                if (TARG == 0.0)
                    r0 = 0.0;
                else
                    r0 *= coval/TARG;
#endif /* !_NTIGHT_ */

                FOR_0_LE_l_LT_p {
                    aTmp = ARES;
#if defined(_INT_REV_)
                    ARES_INC = 0;
                    AARG_INC |= aTmp;
#else
                    ARES_INC = 0.0;
                    AARG_INC += (aTmp==0)?0:(aTmp * r0);
#endif
            }

#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case pow_op_p:                                                /* pow_op_p */
                res   = get_locint_r();
                arg1  = get_locint_r();
                arg   = get_locint_r();
#if !defined(_NTIGHT_)
                coval = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore[arg1];
#endif /* !_NTIGHT_ */

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg, ADJOINT_BUFFER[arg])

#if !defined(_NTIGHT_)
                /* olvo 980921 changed order to allow x=pow(x,n) */
                r0 = TRES;
                if (arg == res)
                    ADOLC_GET_TAYLOR(arg);
                if (TARG == 0.0)
                    r0 = 0.0;
                else
                    r0 *= coval/TARG;
#endif /* !_NTIGHT_ */

                FOR_0_LE_l_LT_p {
                    aTmp = ARES;
#if defined(_INT_REV_)
                    ARES_INC = 0;
                    AARG_INC |= aTmp;
#else
                    ARES_INC = 0.0;
                    AARG_INC += (aTmp==0)?0:(aTmp * r0);
#endif
            }

#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case sqrt_op:                                              /* sqrt_op */
                res = get_locint_r();
                arg = get_locint_r();

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg, ADJOINT_BUFFER[arg])

#if !defined(_NTIGHT_)
                if (TRES == 0.0)
                    r0 = 0.0;
                else
                    r0 = 0.5 / TRES;
#endif /* !_NTIGHT_ */

                FOR_0_LE_l_LT_p {
                    aTmp = ARES;
#if defined(_INT_REV_)
                    ARES_INC = 0;
                    AARG_INC |= aTmp;
#else
                    ARES_INC = 0.0;
                    AARG_INC += (aTmp==0)?0:(aTmp * r0);
#endif
                }

#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
              break;

                /*--------------------------------------------------------------------------*/
            case cbrt_op:                                              /* cbrt_op */
                res = get_locint_r();
                arg = get_locint_r();

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg, ADJOINT_BUFFER[arg])

#if !defined(_NTIGHT_)
                if (TRES == 0.0)
                    r0 = 0.0;
                else
                    r0 = 1.0 / (3.0 * TRES * TRES);
#endif /* !_NTIGHT_ */

                FOR_0_LE_l_LT_p {
                    aTmp = ARES;
#if defined(_INT_REV_)
                    ARES_INC = 0;
                    AARG_INC |= aTmp;
#else
                    ARES_INC = 0.0;
                    AARG_INC += (aTmp==0)?0:(aTmp * r0);
#endif
                }

#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
              break;

                /*--------------------------------------------------------------------------*/
            case gen_quad:                                            /* gen_quad */
                res   = get_locint_r();
                arg2  = get_locint_r();
                arg1  = get_locint_r();
#if !defined(_NTIGHT_)
                coval = get_val_r();
                coval = get_val_r();
#endif /* !_NTIGHT_ */

                ASSIGN_A( Ares,  ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg1, ADJOINT_BUFFER[arg1])

                FOR_0_LE_l_LT_p
                { aTmp = ARES;
#if defined(_INT_REV_)
                  ARES_INC = 0;
                  AARG1_INC |= aTmp;
#else
                  ARES_INC = 0.0;
                  AARG1_INC += (aTmp==0)?0:(aTmp * TARG2);
#endif
            }

#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case min_op:                                                /* min_op */
                res   = get_locint_r();
                arg2  = get_locint_r();
                arg1  = get_locint_r();
#if !defined(_NTIGHT_)
                coval = get_val_r();

                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */

                ASSIGN_A( Aarg1, ADJOINT_BUFFER[arg1])
                ASSIGN_A( Aarg2, ADJOINT_BUFFER[arg2])
                ASSIGN_A( Ares,  ADJOINT_BUFFER[res])

#if !defined(_NTIGHT_)
                if (TARG1 > TARG2)
                    FOR_0_LE_l_LT_p
                    { aTmp = ARES;
#if defined(_INT_REV_)
                      ARES_INC = 0;
#else
                      ARES_INC = 0.0;
#endif
                      if ((coval) && (aTmp))
                      MINDEC(ret_c,2);
#if defined(_INT_REV_)
                      AARG2_INC |= aTmp;
#else
                      AARG2_INC += aTmp;
#endif
                    } else
                        if (TARG1 < TARG2)
                                FOR_0_LE_l_LT_p
                                { aTmp = ARES;
#if defined(_INT_REV_)
                                  ARES_INC = 0;
#else
                                  ARES_INC = 0.0;
#endif
                                  if ((!coval) && (aTmp))
                                  MINDEC(ret_c,2);
#if defined(_INT_REV_)
                                  AARG1_INC |= aTmp;
#else
                                  AARG1_INC += aTmp;
#endif
                                } else { /* both are equal */
                                    FOR_0_LE_l_LT_p
                                    { 
#if defined(_INT_REV_)
                                      aTmp = ARES;
				      ARES_INC = 0;
				      AARG2_INC |= aTmp;
				      AARG1_INC |= aTmp;
#else
				      aTmp = ARES / 2.0;
                                      ARES_INC = 0.0;
                                      AARG2_INC += aTmp;
                                      AARG1_INC += aTmp;
#endif
                                    }
                                    if (arg1 != arg2)
                                            MINDEC(ret_c,1);
                                    }
#else
                    FOR_0_LE_l_LT_p
                    { aTmp = ARES;
                      ARES_INC = 0;
                      AARG1_INC |= aTmp;
                      AARG2_INC |= aTmp;
                    }
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case abs_val:                                                        /* abs_val */
                res   = get_locint_r();
                arg   = get_locint_r();
#if !defined(_NTIGHT_)
                coval = get_val_r();

                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg, ADJOINT_BUFFER[arg])

#if defined(_ABS_NORM_) 
		    if (rownum == switchnum) {
			AARG = 1.0;
		    } else {
			results[indep+switchnum] = ARES;
			AARG = 0.0;
			ARES = 0.0;
		    }
		    switchnum--;
#elif defined(_ABS_NORM_SIG_) 
		    aTmp = ARES;
		    ARES_INC = 0.0;
		    AARG_INC += siggrad[switchnum]*aTmp;
		    switchnum--;
#else
#if !defined(_NTIGHT_)
                if (TARG < 0.0)
                    FOR_0_LE_l_LT_p
                    { aTmp = ARES;
#if defined(_INT_REV_)
                      ARES_INC = 0;
#else
                      ARES_INC = 0.0;
#endif
                      if ((coval) && (aTmp))
                      MINDEC(ret_c,2);
#if defined(_INT_REV_)
                      AARG_INC |= aTmp;
#else
                      AARG_INC -= aTmp;
#endif
                    } else
                        if (TARG > 0.0)
                                FOR_0_LE_l_LT_p
                                { aTmp = ARES;
#if defined(_INT_REV_)
                                  ARES_INC = 0;
#else
                                  ARES_INC = 0.0;
#endif
                                  if ((!coval) && (aTmp))
                                  MINDEC(ret_c,2);
#if defined(_INT_REV_)
                                  AARG_INC |= aTmp;
#else
                                  AARG_INC += aTmp;
#endif
                                } else
                                    FOR_0_LE_l_LT_p {
                                        aTmp = ARES;
#if defined(_INT_REV_)
                                        ARES_INC = 0;
#else
                                        ARES_INC = 0.0;
#endif
                                        if (aTmp)
                                            MINDEC(ret_c,1);
                                        }
#else
                                            FOR_0_LE_l_LT_p
                                            { aTmp = ARES;
                                              ARES_INC = 0;
                                              AARG_INC |= aTmp;
                                            }
#endif /* !_NTIGHT_ */
#endif /* _ABS_NORM */
             break;

            /*--------------------------------------------------------------------------*/
        case ceil_op:                                              /* ceil_op */
               res   = get_locint_r();
                arg   = get_locint_r();
#if !defined(_NTIGHT_)
                coval = get_val_r();

                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])

#if !defined(_NTIGHT_)
                coval = (coval != ceil(TARG) );
#endif /* !_NTIGHT_ */

                FOR_0_LE_l_LT_p
                {
#if !defined(_NTIGHT_)
                  if ((coval) && (ARES))
                  MINDEC(ret_c,2);
#endif /* !_NTIGHT_ */
#if defined(_INT_REV_)
                  ARES_INC = 0;
#else
                  ARES_INC = 0.0;
#endif
                }
                break;

            /*--------------------------------------------------------------------------*/
        case floor_op:                                            /* floor_op */
                res   = get_locint_r();
                arg   = get_locint_r();
#if !defined(_NTIGHT_)
                coval = get_val_r();

                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg, ADJOINT_BUFFER[arg])

#if !defined(_NTIGHT_)
                coval = ( coval != floor(TARG1) );
#endif /* !_NTIGHT_ */

                FOR_0_LE_l_LT_p
                {
#if !defined(_NTIGHT_)
                  if ( (coval) && (ARES) )
                  MINDEC(ret_c,2);
#endif /* !_NTIGHT_ */
#if defined(_INT_REV_)
                  ARES_INC = 0;
#else
                  ARES_INC = 0.0;
#endif
                }
                break;


            /****************************************************************************/
            /*                                                             CONDITIONALS */

            /*--------------------------------------------------------------------------*/
        case cond_assign:                                      /* cond_assign */
            res    = get_locint_r();
                arg2   = get_locint_r();
                arg1   = get_locint_r();
                arg    = get_locint_r();
#if !defined(_NTIGHT_)
                coval  = get_val_r();

                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */

                ASSIGN_A( Aarg1, ADJOINT_BUFFER[arg1])
                ASSIGN_A( Ares,  ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg2, ADJOINT_BUFFER[arg2])

#if !defined(_NTIGHT_)
                /* olvo 980924 changed code a little bit */
                if (TARG > 0.0) {
                    if (res != arg1)
                        FOR_0_LE_l_LT_p
                        { if ((coval <= 0.0) && (ARES))
                          MINDEC(ret_c,2);
#if defined(_INT_REV_)
                              AARG1_INC |= ARES;
                              ARES_INC = 0;
#else
                          AARG1_INC += ARES;
                          ARES_INC = 0.0;
#endif
                        } else
                            FOR_0_LE_l_LT_p
                            if ((coval <= 0.0) && (ARES_INC))
                                    MINDEC(ret_c,2);
                } else {
                    if (res != arg2)
                        FOR_0_LE_l_LT_p
                        { if ((coval <= 0.0) && (ARES))
                          MINDEC(ret_c,2);
#if defined(_INT_REV_)
                          AARG2_INC |= ARES;
                          ARES_INC = 0;
#else
                          AARG2_INC += ARES;
                          ARES_INC = 0.0;
#endif
                        } else
                            FOR_0_LE_l_LT_p
                            if ((coval <= 0.0) && (ARES_INC))
                                    MINDEC(ret_c,2);
                }
#else
                    if (res != arg1) {
                        FOR_0_LE_l_LT_p
                        AARG1_INC |= ARES_INC;
                        ASSIGN_A( Ares,  ADJOINT_BUFFER[res])
                    }
                    if (res != arg2) {
                        FOR_0_LE_l_LT_p
                        AARG2_INC |= ARES_INC;
                        ASSIGN_A( Ares,  ADJOINT_BUFFER[res])
                    }
                    if ((res != arg1) && (res != arg2))
                        FOR_0_LE_l_LT_p
                        ARES_INC = 0;
#endif /* !_NTIGHT_ */
                break;

        case cond_eq_assign:                                      /* cond_assign */
            res    = get_locint_r();
                arg2   = get_locint_r();
                arg1   = get_locint_r();
                arg    = get_locint_r();
#if !defined(_NTIGHT_)
                coval  = get_val_r();

                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */

                ASSIGN_A( Aarg1, ADJOINT_BUFFER[arg1])
                ASSIGN_A( Ares,  ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg2, ADJOINT_BUFFER[arg2])

#if !defined(_NTIGHT_)
                /* olvo 980924 changed code a little bit */
                if (TARG >= 0.0) {
                    if (res != arg1)
                        FOR_0_LE_l_LT_p
                        { if ((coval < 0.0) && (ARES))
                          MINDEC(ret_c,2);
#if defined(_INT_REV_)
                              AARG1_INC |= ARES;
                              ARES_INC = 0;
#else
                          AARG1_INC += ARES;
                          ARES_INC = 0.0;
#endif
                        } else
                            FOR_0_LE_l_LT_p
                            if ((coval < 0.0) && (ARES_INC))
                                    MINDEC(ret_c,2);
                } else {
                    if (res != arg2)
                        FOR_0_LE_l_LT_p
                        { if ((coval < 0.0) && (ARES))
                          MINDEC(ret_c,2);
#if defined(_INT_REV_)
                          AARG2_INC |= ARES;
                          ARES_INC = 0;
#else
                          AARG2_INC += ARES;
                          ARES_INC = 0.0;
#endif
                        } else
                            FOR_0_LE_l_LT_p
                            if ((coval < 0.0) && (ARES_INC))
                                    MINDEC(ret_c,2);
                }
#else
                    if (res != arg1) {
                        FOR_0_LE_l_LT_p
                        AARG1_INC |= ARES_INC;
                        ASSIGN_A( Ares,  ADJOINT_BUFFER[res])
                    }
                    if (res != arg2) {
                        FOR_0_LE_l_LT_p
                        AARG2_INC |= ARES_INC;
                        ASSIGN_A( Ares,  ADJOINT_BUFFER[res])
                    }
                    if ((res != arg1) && (res != arg2))
                        FOR_0_LE_l_LT_p
                        ARES_INC = 0;
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case cond_assign_s:                                  /* cond_assign_s */
                res   = get_locint_r();
                arg1  = get_locint_r();
                arg   = get_locint_r();
#if !defined(_NTIGHT_)
                coval = get_val_r();

                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */

                ASSIGN_A( Aarg1, ADJOINT_BUFFER[arg1])
                ASSIGN_A( Ares,  ADJOINT_BUFFER[res])

#if !defined(_NTIGHT_)
                /* olvo 980924 changed code a little bit */
                if (TARG > 0.0) {
                    if (res != arg1)
                        FOR_0_LE_l_LT_p
                        { if ((coval <= 0.0) && (ARES))
                          MINDEC(ret_c,2);
#if defined(_INT_REV_)
                          AARG1_INC |= ARES;
                          ARES_INC = 0.0;
#else
                          AARG1_INC += ARES;
                          ARES_INC = 0.0;
#endif
                        } else
                            FOR_0_LE_l_LT_p
                            if ((coval <= 0.0) && (ARES_INC))
                                    MINDEC(ret_c,2);
                } else
                    if (TARG == 0.0) /* we are at the tie */
                        FOR_0_LE_l_LT_p
                        if (ARES_INC)
                            MINDEC(ret_c,0);
#else
                    if (res != arg1)
                        FOR_0_LE_l_LT_p
                        { AARG1 |= ARES;
                          ARES_INC = 0;
                        }
#endif /* !_NTIGHT_ */
                break;

            case cond_eq_assign_s:                                  /* cond_eq_assign_s */
                res   = get_locint_r();
                arg1  = get_locint_r();
                arg   = get_locint_r();
#if !defined(_NTIGHT_)
                coval = get_val_r();

                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */

                ASSIGN_A( Aarg1, ADJOINT_BUFFER[arg1])
                ASSIGN_A( Ares,  ADJOINT_BUFFER[res])

#if !defined(_NTIGHT_)
                /* olvo 980924 changed code a little bit */
                if (TARG >= 0.0) {
                    if (res != arg1)
                        FOR_0_LE_l_LT_p
                        { if ((coval < 0.0) && (ARES))
                          MINDEC(ret_c,2);
#if defined(_INT_REV_)
                          AARG1_INC |= ARES;
                          ARES_INC = 0.0;
#else
                          AARG1_INC += ARES;
                          ARES_INC = 0.0;
#endif
                        } else
                            FOR_0_LE_l_LT_p
                            if ((coval < 0.0) && (ARES_INC))
                                    MINDEC(ret_c,2);
                }
#else
                    if (res != arg1)
                        FOR_0_LE_l_LT_p
                        { AARG1 |= ARES;
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
            case neq_a_p:
            case eq_a_p:
            case le_a_p:
            case ge_a_p:
            case lt_a_p:
            case gt_a_p:
		res = get_locint_r();
		arg1 = get_locint_r();
		arg = get_locint_r();
#if !defined(_NTIGHT_)
		coval = get_val_r();
#endif
                ASSIGN_A( Ares, ADJOINT_BUFFER[res])

                FOR_0_LE_l_LT_p
#if defined(_INT_REV_)
                ARES_INC = 0;
#else
                ARES_INC = 0.0;
#endif

#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */

		break;
#endif
                /*--------------------------------------------------------------------------*/
            case subscript:
	        {
#if !defined(_NTIGHT_)
		    double val = 
#endif
		    get_val_r();
		    res = get_locint_r();
#if !defined(_NTIGHT_)
		    size_t idx, numval = (size_t)trunc(fabs(val));
		    locint vectorloc;
		    vectorloc = 
#endif
		    get_locint_r();
		    arg = get_locint_r();
#if !defined(_NTIGHT_)
		    idx = (size_t)trunc(fabs(TARG));
		    if (idx >= numval)
			fprintf(DIAG_OUT, "ADOL-C warning: index out of bounds while subscripting n=%zu, idx=%zu\n", numval, idx);
		    arg1 = vectorloc+idx;
		    ASSIGN_A( Aarg1, ADJOINT_BUFFER[arg1])
		    ASSIGN_A( Ares, ADJOINT_BUFFER[res])
		    FOR_0_LE_l_LT_p
		    {
#if defined(_INT_REV_)
			AARG1_INC |= ARES;
			ARES_INC = 0;
#else
			AARG1_INC += ARES;
			ARES = 0.0;
#endif
		    }
		    ADOLC_GET_TAYLOR(res);
#else
		    fprintf(DIAG_OUT, "ADOL-C error: active subscripting does not work in safe mode, please use tight mode\n");
		    adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* !_NTIGHT_ */
		}
		break;

            case subscript_ref:
	        {
#if !defined(_NTIGHT_)
		    double val = 
#endif
		    get_val_r();
		    res = get_locint_r();
#if !defined(_NTIGHT_)
		    size_t idx, numval = (size_t)trunc(fabs(val));
		    locint vectorloc;
		    vectorloc = 
#endif
		    get_locint_r();
		    arg = get_locint_r();
#if !defined(_NTIGHT_)
		    idx = (size_t)trunc(fabs(TARG));
		    if (idx >= numval)
			fprintf(DIAG_OUT, "ADOL-C warning: index out of bounds while subscripting (ref) n=%zu, idx=%zu\n", numval, idx);
		    arg1 = (size_t)trunc(fabs(TRES));
		    /*
		     * This is actually NOP
                     * basically all we need is that arg1 == vectorloc+idx
                     * so doing a check here is probably good
                     */
		    if (arg1 != vectorloc+idx) {
			fprintf(DIAG_OUT, "ADOL-C error: indexed active position does not match referenced position\nindexed = %zu, referenced = %d\n", vectorloc+idx, arg1);
			adolc_exit(-2,"",__func__,__FILE__,__LINE__);
		    }
		    ADOLC_GET_TAYLOR(res);
#else
		    fprintf(DIAG_OUT, "ADOL-C error: active subscripting does not work in safe mode, please use tight mode\n");
		    adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* !_NTIGHT_ */
		}
		break;

	    case ref_copyout:
		res = get_locint_r();
		arg1 = get_locint_r();
#if !defined(_NTIGHT_)
		arg = (size_t)trunc(fabs(TARG1));
		ASSIGN_A( Ares, ADJOINT_BUFFER[res])
		ASSIGN_A( Aarg, ADJOINT_BUFFER[arg])
		
	        FOR_0_LE_l_LT_p
		{
#if defined(_INT_REV_)                 
		  AARG_INC |= ARES;
                  ARES_INC = 0;
#else
                  AARG_INC += ARES;
                  ARES_INC = 0.0;
#endif
		}
		ADOLC_GET_TAYLOR(res);
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif 
		break;


            case ref_incr_a:                        /* Increment an adouble    incr_a */
            case ref_decr_a:                        /* Increment an adouble    decr_a */
                arg1   = get_locint_r();

#if !defined(_NTIGHT_)
		res = (size_t)trunc(fabs(TARG1));
                ADOLC_GET_TAYLOR(res);
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* !_NTIGHT_ */
                break;

            case ref_assign_d:            /* assign an adouble variable a    assign_d */
                /* double value. (=) */
                arg1   = get_locint_r();
#if !defined(_NTIGHT_)
		res = (size_t)trunc(fabs(TARG1));
                coval = get_val_r();

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])

                FOR_0_LE_l_LT_p
#if defined(_INT_REV_)                 
                ARES_INC = 0;
#else
                ARES_INC = 0.0;
#endif

                ADOLC_GET_TAYLOR(res);
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* !_NTIGHT_ */
                break;

            case ref_assign_p:            /* assign an adouble variable a    assign_p */
                arg    = get_locint_r();
                arg1   = get_locint_r();
#if !defined(_NTIGHT_)
		res = (size_t)trunc(fabs(TARG1));
                coval = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore[arg];

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])

                FOR_0_LE_l_LT_p
#if defined(_INT_REV_)                 
                ARES_INC = 0;
#else
                ARES_INC = 0.0;
#endif

                ADOLC_GET_TAYLOR(res);
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* !_NTIGHT_ */
                break;

            case ref_assign_d_zero:  /* assign an adouble variable a    assign_d_zero */
            case ref_assign_d_one:   /* double value (0 or 1). (=)       assign_d_one */
                arg1 = get_locint_r();

#if !defined(_NTIGHT_)
		res = (size_t)trunc(fabs(TARG1));

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])

                FOR_0_LE_l_LT_p
#if defined(_INT_REV_)                 
                ARES_INC = 0;
#else
                ARES_INC = 0.0;
#endif
                ADOLC_GET_TAYLOR(res);
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* !_NTIGHT_ */
                break;

            case ref_assign_a:           /* assign an adouble variable an    assign_a */
                /* adouble value. (=) */
                arg1 = get_locint_r();
                arg = get_locint_r();

#if !defined(_NTIGHT_)
		res = (size_t)trunc(fabs(TARG1));

                ASSIGN_A( Aarg, ADJOINT_BUFFER[arg])
                ASSIGN_A( Ares, ADJOINT_BUFFER[res])

                FOR_0_LE_l_LT_p
                {
#if defined(_INT_REV_)                 
		  AARG_INC |= ARES;
                  ARES_INC = 0;
#else
                  AARG_INC += ARES;
                  ARES_INC = 0.0;
#endif
		}

                ADOLC_GET_TAYLOR(res);
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* !_NTIGHT_ */
                break;

            case ref_assign_ind:       /* assign an adouble variable an    assign_ind */
                /* independent double value (<<=) */
                arg1 = get_locint_r();

#if !defined(_NTIGHT_)
		res = (size_t)trunc(fabs(TARG1));

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])

                FOR_0_LE_l_LT_p
                    RESULTS(l,indexi) = ARES_INC;

                ADOLC_GET_TAYLOR(res);
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* !_NTIGHT_ */
                indexi--;
                break;

            case ref_eq_plus_d:            /* Add a floating point to an    eq_plus_d */
                /* adouble. (+=) */
                arg1   = get_locint_r();
#if !defined(_NTIGHT_)
		res = (size_t)trunc(fabs(TARG1));
                coval = get_val_r();

                ADOLC_GET_TAYLOR(res);
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* !_NTIGHT_ */
                break;

            case ref_eq_plus_p:            /* Add a floating point to an    eq_plus_d */
                /* adouble. (+=) */
                arg1   = get_locint_r();
                arg    = get_locint_r();
#if !defined(_NTIGHT_)
		res = (size_t)trunc(fabs(TARG1));
                coval = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore[arg];

                ADOLC_GET_TAYLOR(res);
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* !_NTIGHT_ */
                break;

            case ref_eq_plus_a:             /* Add an adouble to another    eq_plus_a */
                /* adouble. (+=) */
                arg1 = get_locint_r();
                arg = get_locint_r();

#if !defined(_NTIGHT_)
		res = (size_t)trunc(fabs(TARG1));

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg, ADJOINT_BUFFER[arg]);

                FOR_0_LE_l_LT_p
#if defined(_INT_REV_)
		AARG_INC |= ARES_INC;
#else
                AARG_INC += ARES_INC;
#endif

                ADOLC_GET_TAYLOR(res);
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* !_NTIGHT_ */
                break;

            case ref_eq_min_d:       /* Subtract a floating point from an    eq_min_d */
                /* adouble. (-=) */
                arg1   = get_locint_r();
#if !defined(_NTIGHT_)
		res = (size_t)trunc(fabs(TARG1));
                coval = get_val_r();

                ADOLC_GET_TAYLOR(res);
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* !_NTIGHT_ */
               break;

            case ref_eq_min_p:       /* Subtract a floating point from an    eq_min_p */
                /* adouble. (-=) */
                arg1   = get_locint_r();
                arg    = get_locint_r();
#if !defined(_NTIGHT_)
		res = (size_t)trunc(fabs(TARG1));
                coval = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore[arg];

                ADOLC_GET_TAYLOR(res);
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* !_NTIGHT_ */
               break;

            case ref_eq_min_a:        /* Subtract an adouble from another    eq_min_a */
                /* adouble. (-=) */
                arg1 = get_locint_r();
                arg = get_locint_r();

#if !defined(_NTIGHT_)
		res = (size_t)trunc(fabs(TARG1));
                ASSIGN_A( Ares, ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg, ADJOINT_BUFFER[arg])

                FOR_0_LE_l_LT_p
#if defined(_INT_REV_)
		AARG_INC |= ARES_INC;
#else
                AARG_INC -= ARES_INC;
#endif

               ADOLC_GET_TAYLOR(res);
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* !_NTIGHT_ */
                break;

            case ref_eq_mult_d:              /* Multiply an adouble by a    eq_mult_d */
                /* flaoting point. (*=) */
                arg1   = get_locint_r();
#if !defined(_NTIGHT_)
		res = (size_t)trunc(fabs(TARG1));
                coval = get_val_r();

#if !defined(_INT_REV_)
                ASSIGN_A( Ares, ADJOINT_BUFFER[res])

                FOR_0_LE_l_LT_p
                ARES_INC *= coval;
#endif

                ADOLC_GET_TAYLOR(res);
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* !_NTIGHT_ */
                break;

            case ref_eq_mult_p:              /* Multiply an adouble by a    eq_mult_p */
                /* flaoting point. (*=) */
                arg1   = get_locint_r();
                arg    = get_locint_r();
#if !defined(_NTIGHT_)
		res = (size_t)trunc(fabs(TARG1));
                coval = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore[arg];

#if !defined(_INT_REV_)
                ASSIGN_A( Ares, ADJOINT_BUFFER[res])

                FOR_0_LE_l_LT_p
                ARES_INC *= coval;
#endif

                ADOLC_GET_TAYLOR(res);
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* !_NTIGHT_ */
                break;

            case ref_eq_mult_a:       /* Multiply one adouble by another    eq_mult_a */
                /* (*=) */
                arg1 = get_locint_r();
                arg = get_locint_r();

#if !defined(_NTIGHT_)
		res = (size_t)trunc(fabs(TARG1));
                ADOLC_GET_TAYLOR(res);

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg, ADJOINT_BUFFER[arg])

                FOR_0_LE_l_LT_p
#if defined(_INT_REV_)
                AARG_INC |= ARES_INC;
#else
                { aTmp = ARES;
                  /* olvo 980713 nn: ARES = 0.0; */
		    ARES_INC =  (aTmp==0)?0:(aTmp * TARG);
		    AARG_INC += (aTmp==0)?0:(aTmp * TRES);
                }
#endif      
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* !_NTIGHT_ */
		break;

        case vec_copy:
                res = get_locint_r();
                size = get_locint_r();
                arg = get_locint_r();
                for (qq=0;qq<size;qq++) {

                ASSIGN_A( Aarg, ADJOINT_BUFFER[arg+qq])
                ASSIGN_A( Ares, ADJOINT_BUFFER[res+qq])

                FOR_0_LE_l_LT_p
                {
#if defined(_INT_REV_)
		  AARG_INC |= ARES;
                  ARES_INC = 0;
#else
                  AARG_INC += ARES;
                  ARES_INC = 0.0;
#endif
                }

#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res+qq);
#endif /* !_NTIGHT_ */
                }

                break;

        case vec_dot:
                res = get_locint_r();
                size = get_locint_r();
                arg2 = get_locint_r();
                arg1 = get_locint_r();
                for (qq=0;qq<size;qq++) {
                    ASSIGN_A( Ares,  ADJOINT_BUFFER[res])
                    ASSIGN_A( Aarg2, ADJOINT_BUFFER[arg2])
                    ASSIGN_A( Aarg1, ADJOINT_BUFFER[arg1])
                    FOR_0_LE_l_LT_p
                    { 
#if defined(_INT_REV_)
                        AARG2_INC |= ARES;
                        AARG1_INC |= ARES_INC;
#else
                        AARG2_INC += ARES    * TARG1;
                        AARG1_INC += ARES_INC * TARG2;
#endif
                    }
                    arg2++;
                    arg1++;
                }
#if !defined(_NTIGHT_)
                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
                break;

        case vec_axpy:
                res = get_locint_r();
                size = get_locint_r();
                arg2 = get_locint_r();
                arg1 = get_locint_r();
                arg = get_locint_r();
                for (qq=0;qq<size;qq++) {
                    ASSIGN_A( Ares,  ADJOINT_BUFFER[res])
                    ASSIGN_A( Aarg,  ADJOINT_BUFFER[arg])
                    ASSIGN_A( Aarg2, ADJOINT_BUFFER[arg2])
                    ASSIGN_A( Aarg1, ADJOINT_BUFFER[arg1])
                    FOR_0_LE_l_LT_p
                    { 
#if defined(_INT_REV_)
                        AARG_INC |= ARES;
                        AARG2_INC |= ARES;
                        AARG1_INC |= ARES_INC;
#else
                        AARG2_INC += ARES;
                        AARG1_INC += ARES * TARG;
                        AARG_INC += ARES_INC * TARG1;
#endif
                    }
#if !defined(_NTIGHT_)
                    ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */
                    arg2++;
                    arg1++;
                    res++;
                }
                break;

        case ref_cond_assign:                                      /* cond_assign */
	   {
#if !defined(_NTIGHT_)
                locint ref    = 
#endif
		get_locint_r();
                arg2   = get_locint_r();
                arg1   = get_locint_r();
                arg    = get_locint_r();
#if !defined(_NTIGHT_)
                coval  = get_val_r();
		res = (size_t)trunc(fabs(rp_T[ref]));

                ADOLC_GET_TAYLOR(res);

                ASSIGN_A( Aarg1, ADJOINT_BUFFER[arg1])
                ASSIGN_A( Ares,  ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg2, ADJOINT_BUFFER[arg2])

                /* olvo 980924 changed code a little bit */
                if (TARG > 0.0) {
                    if (res != arg1)
                        FOR_0_LE_l_LT_p
                        { if ((coval <= 0.0) && (ARES))
                          MINDEC(ret_c,2);
#if defined(_INT_REV_)
                              AARG1_INC |= ARES;
                              ARES_INC = 0;
#else
                          AARG1_INC += ARES;
                          ARES_INC = 0.0;
#endif
                        } else
                            FOR_0_LE_l_LT_p
                            if ((coval <= 0.0) && (ARES_INC))
                                    MINDEC(ret_c,2);
                } else {
                    if (res != arg2)
                        FOR_0_LE_l_LT_p
                        { if ((coval <= 0.0) && (ARES))
                          MINDEC(ret_c,2);
#if defined(_INT_REV_)
                          AARG2_INC |= ARES;
                          ARES_INC = 0;
#else
                          AARG2_INC += ARES;
                          ARES_INC = 0.0;
#endif
                        } else
                            FOR_0_LE_l_LT_p
                            if ((coval <= 0.0) && (ARES_INC))
                                    MINDEC(ret_c,2);
                }
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* !_NTIGHT_ */
	        }
                break;

        case ref_cond_eq_assign:                                      /* cond_eq_assign */
	   {
#if !defined(_NTIGHT_)
                locint ref    = 
#endif
		get_locint_r();
                arg2   = get_locint_r();
                arg1   = get_locint_r();
                arg    = get_locint_r();
#if !defined(_NTIGHT_)
                coval  = get_val_r();
		res = (size_t)trunc(fabs(rp_T[ref]));

                ADOLC_GET_TAYLOR(res);

                ASSIGN_A( Aarg1, ADJOINT_BUFFER[arg1])
                ASSIGN_A( Ares,  ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg2, ADJOINT_BUFFER[arg2])

                /* olvo 980924 changed code a little bit */
                if (TARG >= 0.0) {
                    if (res != arg1)
                        FOR_0_LE_l_LT_p
                        { if ((coval < 0.0) && (ARES))
                          MINDEC(ret_c,2);
#if defined(_INT_REV_)
                              AARG1_INC |= ARES;
                              ARES_INC = 0;
#else
                          AARG1_INC += ARES;
                          ARES_INC = 0.0;
#endif
                        } else
                            FOR_0_LE_l_LT_p
                            if ((coval < 0.0) && (ARES_INC))
                                    MINDEC(ret_c,2);
                } else {
                    if (res != arg2)
                        FOR_0_LE_l_LT_p
                        { if ((coval < 0.0) && (ARES))
                          MINDEC(ret_c,2);
#if defined(_INT_REV_)
                          AARG2_INC |= ARES;
                          ARES_INC = 0;
#else
                          AARG2_INC += ARES;
                          ARES_INC = 0.0;
#endif
                        } else
                            FOR_0_LE_l_LT_p
                            if ((coval < 0.0) && (ARES_INC))
                                    MINDEC(ret_c,2);
                }
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* !_NTIGHT_ */
	        }
                break;

            case ref_cond_assign_s:                                  /* cond_assign_s */
                arg2   = get_locint_r();
                arg1  = get_locint_r();
                arg   = get_locint_r();
#if !defined(_NTIGHT_)
                coval = get_val_r();
		res = (size_t)trunc(fabs(TARG2));
                ADOLC_GET_TAYLOR(res);

                ASSIGN_A( Aarg1, ADJOINT_BUFFER[arg1])
                ASSIGN_A( Ares,  ADJOINT_BUFFER[res])

                /* olvo 980924 changed code a little bit */
                if (TARG > 0.0) {
                    if (res != arg1)
                        FOR_0_LE_l_LT_p
                        { if ((coval <= 0.0) && (ARES))
                          MINDEC(ret_c,2);
#if defined(_INT_REV_)
                          AARG1_INC |= ARES;
                          ARES_INC = 0.0;
#else
                          AARG1_INC += ARES;
                          ARES_INC = 0.0;
#endif
                        } else
                            FOR_0_LE_l_LT_p
                            if ((coval <= 0.0) && (ARES_INC))
                                    MINDEC(ret_c,2);
                } else
                    if (TARG == 0.0) /* we are at the tie */
                        FOR_0_LE_l_LT_p
                        if (ARES_INC)
                            MINDEC(ret_c,0);
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* !_NTIGHT_ */
                break;

            case ref_cond_eq_assign_s:                                  /* cond_eq_assign_s */
                arg2   = get_locint_r();
                arg1  = get_locint_r();
                arg   = get_locint_r();
#if !defined(_NTIGHT_)
                coval = get_val_r();
		res = (size_t)trunc(fabs(TARG2));
                ADOLC_GET_TAYLOR(res);

                ASSIGN_A( Aarg1, ADJOINT_BUFFER[arg1])
                ASSIGN_A( Ares,  ADJOINT_BUFFER[res])

                /* olvo 980924 changed code a little bit */
                if (TARG >= 0.0) {
                    if (res != arg1)
                        FOR_0_LE_l_LT_p
                        { if ((coval < 0.0) && (ARES))
                          MINDEC(ret_c,2);
#if defined(_INT_REV_)
                          AARG1_INC |= ARES;
                          ARES_INC = 0.0;
#else
                          AARG1_INC += ARES;
                          ARES_INC = 0.0;
#endif
                        } else
                            FOR_0_LE_l_LT_p
                            if ((coval < 0.0) && (ARES_INC))
                                    MINDEC(ret_c,2);
                }
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* !_NTIGHT_ */
                break;


                /****************************************************************************/
                /*                                                          REMAINING STUFF */

                /*--------------------------------------------------------------------------*/
            case take_stock_op:                                  /* take_stock_op */
                res  = get_locint_r();
                size = get_locint_r();
#if !defined(_NTIGHT_)
		get_val_v_r(size);
#endif /* !_NTIGHT_ */

                res += size;
                for (ls=size; ls>0; ls--) {
                    res--;

                    ASSIGN_A( Ares, ADJOINT_BUFFER[res])

                    FOR_0_LE_l_LT_p
                    ARES_INC = 0.0;
                }
                break;

                /*--------------------------------------------------------------------------*/
            case death_not:                                          /* death_not */
                arg2 = get_locint_r();
                arg1 = get_locint_r();

                for (j=arg1;j<=arg2;j++) {
                    ASSIGN_A(Aarg1, ADJOINT_BUFFER[j])

                    FOR_0_LE_l_LT_p
                    AARG1_INC = 0.0;
                }

#if !defined(_NTIGHT_)
                for (j=arg1;j<=arg2;j++)
                    ADOLC_GET_TAYLOR(j);
#endif /* !_NTIGHT_ */
                break;

#if !defined(_INT_REV_)
                /*--------------------------------------------------------------------------*/
            case ext_diff:                       /* extern differntiated function */
                ADOLC_CURRENT_TAPE_INFOS.cpIndex = get_locint_r();
                ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_rev = get_locint_r();
                ADOLC_CURRENT_TAPE_INFOS.lowestXLoc_rev = get_locint_r();
                m = get_locint_r();
                n = get_locint_r();
                ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index = get_locint_r();
                ADOLC_EXT_FCT_SAVE_NUMDIRS;
                edfct = get_ext_diff_fct(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);

                oldTraceFlag = ADOLC_CURRENT_TAPE_INFOS.traceFlag;
                ADOLC_CURRENT_TAPE_INFOS.traceFlag = 0;

                if (edfct->ADOLC_EXT_FCT_POINTER == NULL)
                    fail(ADOLC_EXT_DIFF_NULLPOINTER_FUNCTION);
                if (m>0) {
                    if (ADOLC_EXT_FCT_U == NULL) fail(ADOLC_EXT_DIFF_NULLPOINTER_ARGUMENT);
                    if (edfct->dp_y==NULL) fail(ADOLC_EXT_DIFF_NULLPOINTER_ARGUMENT);
                }
                if (n>0) {
                    if (ADOLC_EXT_FCT_Z == NULL) fail(ADOLC_EXT_DIFF_NULLPOINTER_ARGUMENT);
                    if (edfct->dp_x==NULL) fail(ADOLC_EXT_DIFF_NULLPOINTER_ARGUMENT);
                }
                arg = ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_rev;
                for (loop = 0; loop < m; ++loop) {
                    ADOLC_EXT_FCT_COPY_ADJOINTS(ADOLC_EXT_FCT_U[loop],ADJOINT_BUFFER_ARG);
                    ++arg;
                }

                arg = ADOLC_CURRENT_TAPE_INFOS.lowestXLoc_rev;
                for (loop = 0; loop < n; ++loop) {
                    ADOLC_EXT_FCT_COPY_ADJOINTS(ADOLC_EXT_FCT_Z[loop],ADJOINT_BUFFER_ARG);
                    ++arg;
                }
                arg = ADOLC_CURRENT_TAPE_INFOS.lowestXLoc_rev;
                for (loop = 0; loop < n; ++loop,++arg) {
                  edfct->dp_x[loop]=TARG;
                }
                arg = ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_rev;
                for (loop = 0; loop < m; ++loop,++arg) {
                  edfct->dp_y[loop]=TARG;
                }
                ext_retc = edfct->ADOLC_EXT_FCT_COMPLETE;
                MINDEC(ret_c, ext_retc);

                res = ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_rev;
                for (loop = 0; loop < m; ++loop) {
                    FOR_0_LE_l_LT_p {
                        ADJOINT_BUFFER_RES_L = 0.; /* \bar{v}_i = 0 !!! */
                    }
                    ++res;
                }
                res = ADOLC_CURRENT_TAPE_INFOS.lowestXLoc_rev;
                for (loop = 0; loop < n; ++loop) {
                    ADOLC_EXT_FCT_COPY_ADJOINTS_BACK(ADOLC_EXT_FCT_Z[loop],ADJOINT_BUFFER_RES);
                    ++res;
                }
                if (edfct->dp_y_priorRequired) {
                  arg = ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_rev+m-1;
                  for (loop = 0; loop < m; ++loop,--arg) {
                    ADOLC_GET_TAYLOR(arg);
                  }
                }
                if (edfct->dp_x_changes) {
                  arg = ADOLC_CURRENT_TAPE_INFOS.lowestXLoc_rev+n-1;
                  for (loop = 0; loop < n; ++loop,--arg) {
                    ADOLC_GET_TAYLOR(arg);
                  }
                }
                ADOLC_CURRENT_TAPE_INFOS.traceFlag = oldTraceFlag;

                break;
            case ext_diff_iArr:                       /* extern differntiated function */
                ADOLC_CURRENT_TAPE_INFOS.cpIndex = get_locint_r();
                ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_rev = get_locint_r();
                ADOLC_CURRENT_TAPE_INFOS.lowestXLoc_rev = get_locint_r();
                m = get_locint_r();
                n = get_locint_r();
                ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index = get_locint_r();
                iArrLength=get_locint_r();
                iArr=(int*)malloc(iArrLength*sizeof(int));
                for (loop=iArrLength-1;loop>=0;--loop) iArr[loop]=get_locint_r();
                get_locint_r(); /* get it again */
                ADOLC_EXT_FCT_SAVE_NUMDIRS;
                edfct = get_ext_diff_fct(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);

                oldTraceFlag = ADOLC_CURRENT_TAPE_INFOS.traceFlag;
                ADOLC_CURRENT_TAPE_INFOS.traceFlag = 0;

                if (edfct->ADOLC_EXT_FCT_IARR_POINTER == NULL)
                    fail(ADOLC_EXT_DIFF_NULLPOINTER_FUNCTION);
                if (m>0) {
                    if (ADOLC_EXT_FCT_U == NULL) fail(ADOLC_EXT_DIFF_NULLPOINTER_ARGUMENT);
                    if (edfct->dp_y==NULL) fail(ADOLC_EXT_DIFF_NULLPOINTER_ARGUMENT);
                }
                if (n>0) {
                    if (ADOLC_EXT_FCT_Z == NULL) fail(ADOLC_EXT_DIFF_NULLPOINTER_ARGUMENT);
                    if (edfct->dp_x==NULL) fail(ADOLC_EXT_DIFF_NULLPOINTER_ARGUMENT);
                }
                arg = ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_rev;
                for (loop = 0; loop < m; ++loop) {
                    ADOLC_EXT_FCT_COPY_ADJOINTS(ADOLC_EXT_FCT_U[loop],ADJOINT_BUFFER_ARG);
                    ++arg;
                }

                arg = ADOLC_CURRENT_TAPE_INFOS.lowestXLoc_rev;
                for (loop = 0; loop < n; ++loop) {
                    ADOLC_EXT_FCT_COPY_ADJOINTS(ADOLC_EXT_FCT_Z[loop],ADJOINT_BUFFER_ARG);
                    ++arg;
                }
                arg = ADOLC_CURRENT_TAPE_INFOS.lowestXLoc_rev;
                for (loop = 0; loop < n; ++loop,++arg) {
                  edfct->dp_x[loop]=TARG;
                }
                arg = ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_rev;
                for (loop = 0; loop < m; ++loop,++arg) {
                  edfct->dp_y[loop]=TARG;
                }
                ext_retc = edfct->ADOLC_EXT_FCT_IARR_COMPLETE;
                MINDEC(ret_c, ext_retc);

                res = ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_rev;
                for (loop = 0; loop < m; ++loop) {
                    FOR_0_LE_l_LT_p {
                        ADJOINT_BUFFER_RES_L = 0.; /* \bar{v}_i = 0 !!! */
                    }
                    ++res;
                }
                res = ADOLC_CURRENT_TAPE_INFOS.lowestXLoc_rev;
                for (loop = 0; loop < n; ++loop) {
                    ADOLC_EXT_FCT_COPY_ADJOINTS_BACK(ADOLC_EXT_FCT_Z[loop],ADJOINT_BUFFER_RES);
                    ++res;
                }
                if (edfct->dp_y_priorRequired) {
                  arg = ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_rev+m-1;
                  for (loop = 0; loop < m; ++loop,--arg) {
                    ADOLC_GET_TAYLOR(arg);
                  }
                }
                if (edfct->dp_x_changes) {
                  arg = ADOLC_CURRENT_TAPE_INFOS.lowestXLoc_rev+n-1;
                  for (loop = 0; loop < n; ++loop,--arg) {
                    ADOLC_GET_TAYLOR(arg);
                  }
                }
                ADOLC_CURRENT_TAPE_INFOS.traceFlag = oldTraceFlag;

                break;
            case ext_diff_v2:
                nout = get_locint_r();
                nin = get_locint_r();
                insz = malloc(2*(nin+nout)*sizeof(locint));
                outsz = insz + nin;
                ADOLC_CURRENT_TAPE_INFOS.lowestXLoc_ext_v2 = outsz + nout;
                ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_ext_v2 = outsz + nout + nin;
                for (loop=nout-1;loop>=0;--loop) {
                    ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_ext_v2[loop] = get_locint_r();
                    outsz[loop] = get_locint_r();
                }
                for (loop=nin-1;loop>=0;--loop) {
                    ADOLC_CURRENT_TAPE_INFOS.lowestXLoc_ext_v2[loop] = get_locint_r();
                    insz[loop] = get_locint_r();
                }
                get_locint_r(); /* nout again */
                get_locint_r(); /* nin again */
                iArrLength = get_locint_r();
                iArr = malloc(iArrLength*sizeof(int));
                for (loop=iArrLength-1;loop>=0;--loop) iArr[loop] = get_locint_r();
                get_locint_r(); /* iArrLength again */
                ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index=get_locint_r();
                ADOLC_EXT_FCT_SAVE_NUMDIRS;
                edfct2 = get_ext_diff_fct_v2(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
                oldTraceFlag = ADOLC_CURRENT_TAPE_INFOS.traceFlag;
                ADOLC_CURRENT_TAPE_INFOS.traceFlag = 0;
                
                if (edfct2->ADOLC_EXT_FCT_POINTER == NULL)
                    fail(ADOLC_EXT_DIFF_NULLPOINTER_FUNCTION);
                if (nout>0) {
                    if (ADOLC_EXT_FCT_V2_U == NULL) fail(ADOLC_EXT_DIFF_NULLPOINTER_ARGUMENT);
                    if (edfct2->y == NULL) fail(ADOLC_EXT_DIFF_NULLPOINTER_ARGUMENT);
                }
                if (nin>0) {
                    if (ADOLC_EXT_FCT_V2_Z == NULL) fail(ADOLC_EXT_DIFF_NULLPOINTER_ARGUMENT);
                    if (edfct2->x == NULL) fail(ADOLC_EXT_DIFF_NULLPOINTER_ARGUMENT);
                }
                for (oloop=0;oloop<nout;++oloop) {
                    arg = ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_ext_v2[oloop];
                    for (loop = 0; loop < outsz[oloop]; ++loop) {
                        ADOLC_EXT_FCT_COPY_ADJOINTS(ADOLC_EXT_FCT_V2_U_LOOP,ADJOINT_BUFFER_ARG);
                        edfct2->y[oloop][loop]=TARG;
                        ++arg;
                    }
                }
                for (oloop=0;oloop<nin;++oloop) {
                    arg = ADOLC_CURRENT_TAPE_INFOS.lowestXLoc_ext_v2[oloop];
                    for (loop =0; loop < insz[oloop]; ++loop) {
                        ADOLC_EXT_FCT_COPY_ADJOINTS(ADOLC_EXT_FCT_V2_Z_LOOP, ADJOINT_BUFFER_ARG);
                        edfct2->x[oloop][loop]=TARG;
                        ++arg;
                    }
                }
                ext_retc = edfct2->ADOLC_EXT_FCT_V2_COMPLETE;
                MINDEC(ret_c, ext_retc);
                for (oloop=0;oloop<nout;++oloop) {
                    res = ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_ext_v2[oloop];
                    for (loop =0; loop < outsz[oloop]; ++loop) {
                        FOR_0_LE_l_LT_p {
                            ADJOINT_BUFFER_RES_L = 0.0; /* \bar{v}_i = 0 !!! */
                        }
                        ++res;
                    }
                }
                for (oloop=0;oloop<nin;++oloop) {
                    res = ADOLC_CURRENT_TAPE_INFOS.lowestXLoc_ext_v2[oloop];
                    for(loop = 0; loop<insz[oloop]; ++loop) {
                        ADOLC_EXT_FCT_COPY_ADJOINTS_BACK(ADOLC_EXT_FCT_V2_Z_LOOP,ADJOINT_BUFFER_RES);
                        ++res;
                    }
                }
                if (edfct2->dp_y_priorRequired) {
                    for(oloop=nout-1;oloop>=0;--oloop) {
                        arg = ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_ext_v2[oloop]+outsz[oloop]-1;
                        for (loop=outsz[oloop]-1; loop>=0; --loop) {
                            ADOLC_GET_TAYLOR(arg);
                            --arg;
                        }
                    }
                }
                if (edfct2->dp_x_changes) {
                    for(oloop=nin-1;oloop>=0;--oloop) {
                        arg = ADOLC_CURRENT_TAPE_INFOS.lowestXLoc_ext_v2[oloop]+insz[oloop]-1;
                        for (loop=insz[oloop]-1; loop>=0; --loop) {
                            ADOLC_GET_TAYLOR(arg);
                            --arg;
                        }
                    }
                }
                ADOLC_CURRENT_TAPE_INFOS.traceFlag = oldTraceFlag;
                free(iArr);
                free(insz);
                insz = 0;
                iArr = 0;
                outsz = 0;
                ADOLC_CURRENT_TAPE_INFOS.lowestXLoc_ext_v2 = 0;
                ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_ext_v2 = 0;
                break;
#ifdef ADOLC_MEDIPACK_SUPPORT
                /*--------------------------------------------------------------------------*/
            case medi_call: {
                locint mediIndex = get_locint_r();
                short tapeId = ADOLC_CURRENT_TAPE_INFOS.tapeID;

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
              BW_AMPI_Send(buf,
                           count,
                           datatype,
                           src,
                           tag,
                           pairedWith,
                           comm);
              break;
            }
            case ampi_recv: {
	      BW_AMPI_Recv(buf,
			   count,
			   datatype,
			   src,
			   tag,
			   pairedWith,
			   comm,
			   status);
	      break;
	    }
	  case ampi_isend: { 
	    BW_AMPI_Isend(buf,
			  count,
			  datatype,
			  src,
			  tag,
			  pairedWith,
			  comm,
			  &request);
	    break;
	  }
          case ampi_irecv: {
            BW_AMPI_Irecv(buf,
                          count,
                          datatype,
                          src,
                          tag,
                          pairedWith,
                          comm,
                          &request);
            break;
          }
	  case ampi_wait: { 
	    BW_AMPI_Wait(&request,
			 status);
	    break;
	  }
	  case ampi_barrier: {
	    BW_AMPI_Barrier(comm);
	    break;
	  }
	  case ampi_gather: { 
	    BW_AMPI_Gather(buf,
			   count,
			   datatype,
			   rbuf,
			   rcount,
			   rtype,
			   src,
			   comm);
	    break;
	  }
	  case ampi_scatter: {
	    BW_AMPI_Scatter(rbuf,
			    rcount,
			    rtype,
			    buf,
			    count,
			    datatype,
			    src,
			    comm);
	    break;
	  }
	  case ampi_allgather: {
	    BW_AMPI_Allgather(buf,
	                      count,
	                      datatype,
	                      rbuf,
	                      rcount,
	                      rtype,
	                      comm);
	    break;
	  }
	  case ampi_gatherv: {
	    BW_AMPI_Gatherv(buf,
			    count,
			    datatype,
			    rbuf,
			    NULL,
			    NULL,
			    rtype,
			    src,
			    comm);
	    break;
	  }
	  case ampi_scatterv: { 
	    BW_AMPI_Scatterv(rbuf,
			     NULL,
			     NULL,
			     rtype,
			     buf,
			     count,
			     datatype,
			     src,
			     comm);
	    break;
	  }
	  case ampi_allgatherv: {
	    BW_AMPI_Allgatherv(buf,
	                       count,
	                       datatype,
	                       rbuf,
	                       NULL,
	                       NULL,
	                       rtype,
	                       comm);
	    break;
	  }
	  case ampi_bcast: {
	    BW_AMPI_Bcast(buf,
			  count,
			  datatype,
			  src,
			  comm);
	    break;
	  }
	  case ampi_reduce: {
	    BWB_AMPI_Reduce(buf,
			   rbuf,
			   count,
			   datatype,
			   op,
			   src,
			   comm);
	    break;
	  }
	  case ampi_allreduce: {
	    BW_AMPI_Allreduce(buf,
	                      rbuf,
	                      count,
	                      datatype,
	                      op,
	                      comm);
	    break;
	  }
#endif
#endif /* !_INT_REV_ */
                /*--------------------------------------------------------------------------*/
            default:                                                   /* default */
                /*             Die here, we screwed up     */

                fprintf(DIAG_OUT,"ADOL-C fatal error in " GENERATED_FILENAME " ("
                        __FILE__
                        ") : no such operation %d\n", operation);
                adolc_exit(-1,"",__func__,__FILE__,__LINE__);
                break;
        } /* endswitch */

        /* Get the next operation */
        operation=get_op_r();
    } /* endwhile */

    /* clean up */
#if !defined(_INT_REV_)
    free(rp_T);
#endif
#ifdef _FOS_
    free(rp_A);
#endif
#ifdef _FOV_
    free(Aqo);
    free(rpp_A);
#endif
#ifdef _INT_REV_
    free(upp_A);
#endif

    ADOLC_CURRENT_TAPE_INFOS.workMode = ADOLC_NO_MODE;
    end_sweep();

    return ret_c;
}


/****************************************************************************/
/*                                                               THAT'S ALL */

END_C_DECLS

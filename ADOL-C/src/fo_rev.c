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
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
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
#define GENERATED_FILENAME "fos_reverse"

#define RESULTS(l,indexi)  results[indexi]
#define LAGRANGE(l,indexd) lagrange[indexd]

/*--------------------------------------------------------------------------*/
#elif _FOV_
#define GENERATED_FILENAME "fov_reverse"

#define _ADOLC_VECTOR_

#define RESULTS(l,indexi)  results[l][indexi]
#define LAGRANGE(l,indexd) lagrange[l][indexd]

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
#include <interfaces.h>
#include <adalloc.h>
#include <oplate.h>
#include <taping_p.h>
#include <externfcts.h>
#include <externfcts_p.h>

#include <math.h>

BEGIN_C_DECLS

/****************************************************************************/
/*                                                             NOW THE CODE */

#ifdef _FOS_
/****************************************************************************/
/* First-Order Scalar Reverse Pass.                                         */
/****************************************************************************/
int fos_reverse(short   tnum,       /* tape id */
                int     depen,      /* consistency chk on # of deps */
                int     indep,      /* consistency chk on # of indeps */
                double  *lagrange,
                double  *results)   /*  coefficient vectors */

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
    double coval = 0, *d = 0;
#endif

    int indexi = 0,  indexd = 0;

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

#if !defined(ADOLC_USE_CALLOC)
    char * c_Ptr;
#endif

    /****************************************************************************/
    /*                                          extern diff. function variables */
#if defined(_FOS_)
# define ADOLC_EXT_FCT_U edfct->dp_U
# define ADOLC_EXT_FCT_Z edfct->dp_Z
# define ADOLC_EXT_FCT_POINTER fos_reverse
# define ADOLC_EXT_FCT_COMPLETE \
  fos_reverse(m, edfct->dp_U, n, edfct->dp_Z)
# define ADOLC_EXT_FCT_SAVE_NUMDIRS
#else
# define ADOLC_EXT_FCT_U edfct->dpp_U
# define ADOLC_EXT_FCT_Z edfct->dpp_Z
# define ADOLC_EXT_FCT_POINTER fov_reverse
# define ADOLC_EXT_FCT_COMPLETE \
  fov_reverse(m, edfct->dpp_U, n, edfct->dpp_Z)
# define ADOLC_EXT_FCT_SAVE_NUMDIRS ADOLC_CURRENT_TAPE_INFOS.numDirs_rev = nrows
#endif
#if !defined(_INT_REV_)
    locint n, m;
    ext_diff_fct *edfct;
    int loop;
    int ext_retc;
    int oldTraceFlag;
#endif


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


    /****************************************************************************/
    /*                                                  MEMORY ALLOCATION STUFF */

    /*--------------------------------------------------------------------------*/
#ifdef _FOS_                                                         /* FOS */
    rp_A = (revreal*) malloc(ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES] * sizeof(revreal));
    if (rp_A == NULL) fail(ADOLC_MALLOC_FAILED);
    ADOLC_CURRENT_TAPE_INFOS.rp_A = rp_A;
    rp_T = (revreal *)malloc(ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES] *
            sizeof(revreal));
    if (rp_T == NULL) fail(ADOLC_MALLOC_FAILED);
#if !defined(ADOLC_USE_CALLOC)
    c_Ptr = (char *) ADOLC_GLOBAL_TAPE_VARS.rp_A;
    *c_Ptr = 0;
    memcpy(c_Ptr + 1, c_Ptr, sizeof(double) *
            ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES] - 1);
#endif
# define ADJOINT_BUFFER rp_A
# define ADJOINT_BUFFER_ARG_L rp_A[arg]
# define ADJOINT_BUFFER_RES_L rp_A[res]
# define ADOLC_EXT_FCT_U_L_LOOP edfct->dp_U[loop]
# define ADOLC_EXT_FCT_Z_L_LOOP edfct->dp_Z[loop]

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
        rpp_A[j] = Aqo;
        Aqo += p;
    }
    ADOLC_CURRENT_TAPE_INFOS.rpp_A = rpp_A;
    rp_T = (revreal *)malloc(ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES] *
            sizeof(revreal));
    if (rp_T == NULL) fail(ADOLC_MALLOC_FAILED);
#if !defined(ADOLC_USE_CALLOC)
    c_Ptr = (char *) ADOLC_GLOBAL_TAPE_VARS.dpp_A;
    *c_Ptr = 0;
    memcpy(c_Ptr + 1, c_Ptr, sizeof(double) * p *
            ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES] - 1);
#endif
# define ADJOINT_BUFFER rpp_A
# define ADJOINT_BUFFER_ARG_L rpp_A[arg][l]
# define ADJOINT_BUFFER_RES_L rpp_A[res][l]
# define ADOLC_EXT_FCT_U_L_LOOP edfct->dpp_U[l][loop]
# define ADOLC_EXT_FCT_Z_L_LOOP edfct->dpp_Z[l][loop]
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
                " preceeded\nby a forward sweep with degree>0, keep=1!\n");
        exit(-2);
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
            case end_of_tape:                                      /* end_of_tape */
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

                FOR_0_LE_l_LT_p
                    RESULTS(l,indexi) = ARES_INC;

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

                FOR_0_LE_l_LT_p
                ARES_INC = LAGRANGE(l,indexd);

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
                  ARES_INC =  aTmp * TARG;
                  AARG_INC += aTmp * TRES;
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
                  AARG2_INC += aTmp * TARG1;
                  AARG1_INC += aTmp * TARG2;
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
                  AARG_INC += coval * aTmp;
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
                  AARG1_INC += aTmp * r0;
                  AARG2_INC += aTmp * r_0;
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
                  AARG_INC += aTmp * r0;
#endif
                }

#if !defined(_NTIGHT_)
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
                  AARG_INC += aTmp * TRES;
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
                  AARG1_INC += aTmp * TARG2;
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
                  AARG1_INC -= aTmp * TARG2;
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
                  AARG1_INC += aTmp * TARG2;
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
                  AARG_INC += aTmp * r0;
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
                    AARG_INC += aTmp * r0;
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
                    AARG_INC += aTmp * r0;
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
                  AARG1_INC += aTmp * TARG2;
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
            case abs_val:                                              /* abs_val */
                res   = get_locint_r();
                arg   = get_locint_r();
#if !defined(_NTIGHT_)
                coval = get_val_r();

                ADOLC_GET_TAYLOR(res);
#endif /* !_NTIGHT_ */

                ASSIGN_A( Ares, ADJOINT_BUFFER[res])
                ASSIGN_A( Aarg, ADJOINT_BUFFER[arg])

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

                /****************************************************************************/
                /*                                                          REMAINING STUFF */

                /*--------------------------------------------------------------------------*/
            case take_stock_op:                                  /* take_stock_op */
                res  = get_locint_r();
                size = get_locint_r();
#if !defined(_NTIGHT_)
                d    = get_val_v_r(size);
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
                if (m>0)
                    if (ADOLC_EXT_FCT_U == NULL) fail(ADOLC_EXT_DIFF_NULLPOINTER_ARGUMENT);
                if (n>0)
                    if (ADOLC_EXT_FCT_Z == NULL) fail(ADOLC_EXT_DIFF_NULLPOINTER_ARGUMENT);

                arg = ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_rev;
                for (loop = 0; loop < m; ++loop) {
                    FOR_0_LE_l_LT_p {
                        ADOLC_EXT_FCT_U_L_LOOP = ADJOINT_BUFFER_ARG_L;
                    }
                    ++arg;
                }
                for (loop = 0; loop < m; ++loop) {
                    --arg;
                    ADOLC_GET_TAYLOR(arg);
                }
                arg = ADOLC_CURRENT_TAPE_INFOS.lowestXLoc_rev;
                for (loop = 0; loop < n; ++loop) {
                    FOR_0_LE_l_LT_p {
                        ADOLC_EXT_FCT_Z_L_LOOP = ADJOINT_BUFFER_ARG_L;
                    }
                    ++arg;
                }
                for (loop = 0; loop < n; ++loop) {
                    --arg;
                    ADOLC_GET_TAYLOR(arg);
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
                    FOR_0_LE_l_LT_p {
                        ADJOINT_BUFFER_RES_L = ADOLC_EXT_FCT_Z_L_LOOP;
                    }
                    ++res;
                }

                ADOLC_CURRENT_TAPE_INFOS.traceFlag = oldTraceFlag;

                break;
#endif /* !_INT_REV_ */
                /*--------------------------------------------------------------------------*/
            default:                                                   /* default */
                /*             Die here, we screwed up     */

                fprintf(DIAG_OUT,"ADOL-C fatal error in " GENERATED_FILENAME " ("
                        __FILE__
                        ") : no such operation %d\n", operation);
                exit(-1);
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
    myfree2(rpp_A);
#endif
#ifdef _INT_REV_
    free(upp_A);
#endif

    end_sweep();

    return ret_c;
}


/****************************************************************************/
/*                                                               THAT'S ALL */

END_C_DECLS

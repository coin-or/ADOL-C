/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     ho_rev.c
 Revision: $Id$
 Contents: Contains the routines :
           hos_reverse (higher-order-scalar reverse mode): 
              define _HOS_
           hos_ov_reverse (higher-order-scalar reverse mode on vectors): 
              define _HOS_OV_
           hov_reverse (higher-order-vector reverse mode): 
              define _HOV_
 
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
#undef _HIGHER_ORDER_

/*--------------------------------------------------------------------------*/
#ifdef _HOS_
#define GENERATED_FILENAME "hos_reverse"

#define _HIGHER_ORDER_

#define RESULTS(l,indexi,k) results[indexi][k]
#define LAGRANGE(l,indexd,k)  lagrange[indexd][k]

#define HOV_INC(T,degree) {}
#define HOS_OV_INC(T,degree) {}

#define GET_TAYL(loc,depth,p) \
    { \
        UPDATE_TAYLORREAD(depth) \
        get_taylors(loc,depth); \
    }

/*--------------------------------------------------------------------------*/
#elif _HOS_OV_
#define GENERATED_FILENAME "hos_ov_reverse"

#define _HIGHER_ORDER_

#define RESULTS(l,indexi,k) results[l][indexi][k]
#define LAGRANGE(l,indexd,k)  lagrange[indexd][k]

#define HOV_INC(T,degree) T += degree;
#define HOS_OV_INC(T,degree) T += degree;

#define GET_TAYL(loc,depth,p) \
    { \
        UPDATE_TAYLORREAD(depth * p) \
        get_taylors_p(loc,depth,p); \
    }

/*--------------------------------------------------------------------------*/
#elif _HOV_
#define GENERATED_FILENAME "hov_reverse"

#define _ADOLC_VECTOR_
#define _HIGHER_ORDER_

#define RESULTS(l,indexi,k) results[l][indexi][k]
#define LAGRANGE(l,indexd,k)  lagrange[l][indexd][k]

#define IF_HOV_
#define ENDIF_HOV_

#define HOV_INC(T,degree) T += degree;
#define HOS_OV_INC(T,degree)

#define GET_TAYL(loc,depth,p) \
    { \
        UPDATE_TAYLORREAD(depth) \
        get_taylors(loc,depth); \
    }

#else
#error Error ! Define [_HOS_ | _HOS_OV_ | _HOV_]
#endif

/*--------------------------------------------------------------------------*/
/*                                                     access to variables  */

#ifdef _FOS_                                     /* why?, not in fo_rev.c ? */
#define ARES       *Ares
#define AARG       *Aarg
#define AARG1      *Aarg1
#define AARG2      *Aarg2
#define AQO        *Aqo

#define ARES_INC   *Ares
#define AARG_INC   *Aarg
#define AARG1_INC  *Aarg1
#define AARG2_INC  *Aarg2
#define AQO_INC    *Aqo

#define ARES_INC_O  Ares
#define AARG_INC_O  Aarg
#define AARG1_INC_O Aarg1
#define AARG2_INC_O Aarg2
#define AQO_INC_O   Aqo

#define ASSIGN_A(a,b)  a = &b;
#define HOS_OV_ASSIGN_A(Aqo,  dp_Atemp)
#define FOR_0_LE_l_LT_q l = 0;

#elif _HOS_OV_
#define ARES       *Ares
#define AARG       *Aarg
#define AARG1      *Aarg1
#define AARG2      *Aarg2
#define AQO        *Aqo

#define ARES_INC   *Ares++
#define AARG_INC   *Aarg++
#define AARG1_INC  *Aarg1++
#define AARG2_INC  *Aarg2++
#define AQO_INC    *Aqo++

#define ARES_INC_O  Ares++
#define AARG_INC_O  Aarg++
#define AARG1_INC_O Aarg1++
#define AARG2_INC_O Aarg2++
#define AQO_INC_O   Aqo++

#define ASSIGN_A(a,b)  a = b;
#define HOS_OV_ASSIGN_A(a, b) a = b;
#define FOR_0_LE_l_LT_q for(l=0;l<q;l++)

#else  /* _FOV_, _HOS_, _HOV_ */
#define ARES       *Ares
#define AARG       *Aarg
#define AARG1      *Aarg1
#define AARG2      *Aarg2
#define AQO        *Aqo

#define ARES_INC   *Ares++
#define AARG_INC   *Aarg++
#define AARG1_INC  *Aarg1++
#define AARG2_INC  *Aarg2++
#define AQO_INC    *Aqo++

#define ARES_INC_O  Ares++
#define AARG_INC_O  Aarg++
#define AARG1_INC_O Aarg1++
#define AARG2_INC_O Aarg2++
#define AQO_INC_O   Aqo++

#define ASSIGN_A(a,b)  a = b;
#define HOS_OV_ASSIGN_A(Aqo, Atemp)
#define FOR_0_LE_l_LT_q l = 0;
#endif

#ifdef _HIGHER_ORDER_

#define TRES      *Tres                  /* why ? not used here */
#define TARG      *Targ
#define TARG1     *Targ1
#define TARG2     *Targ2

#define ASSIGN_T(a,b)  a = b;
#else

#define TRES       rpp_T[res]
#define TARG       rpp_T[arg]
#define TARG1      rpp_T[arg1]
#define TARG2      rpp_T[arg2]

#define ASSIGN_T(a,b)
#endif

/*--------------------------------------------------------------------------*/
/*                                                              loop stuff  */
#ifdef _ADOLC_VECTOR_
#define FOR_0_LE_l_LT_p for (l=0; l<p; l++)
#define FOR_p_GT_l_GE_0 for (l=p-1; l>=0; l--)  /* why ? not used here */
#elif _HOS_OV_
#define FOR_0_LE_l_LT_p for (l=0; l<p; l++)
#define FOR_p_GT_l_GE_0                         /* why ? not used here */
#else
#define FOR_0_LE_l_LT_p
#define FOR_p_GT_l_GE_0                         /* why ? not used here */
#endif

#ifdef _HIGHER_ORDER_
#define FOR_0_LE_i_LT_k for (i=0; i<k; i++)
#define FOR_k_GT_i_GE_0 for (i=k-1; i>=0; i--)
#else
#define FOR_0_LE_i_LT_k
#define FOR_k_GT_i_GE_0
#endif

#ifdef _HOV_
#define FOR_0_LE_l_LT_pk1 for (l=0; l<pk1; l++)
#define FOR_0_LE_l_LT_pk for (l=0; l<k; l++)
#elif _FOV_
#define FOR_0_LE_l_LT_pk1 for (l=0; l<p; l++)
#define FOR_0_LE_l_LT_pk for (l=0; l<k; l++)
#elif _HOS_
#define FOR_0_LE_l_LT_pk1 for (l=0; l<k1; l++)
#define FOR_0_LE_l_LT_pk for (l=0; l<k; l++)
#elif _HOS_OV_
#define FOR_0_LE_l_LT_pk1 for (l=0; l<pk1; l++)
#define FOR_0_LE_l_LT_pk for (l=0; l<p*k; l++)
#else
#define FOR_0_LE_l_LT_pk1
#define FOR_0_LE_l_LT_pk
#endif

/*--------------------------------------------------------------------------*/
/*                                                         VEC_COMPUTED_* */
#ifdef _ADOLC_VECTOR
#define VEC_COMPUTED_INIT   computed = 0;
#define VEC_COMPUTED_CHECK  if (computed == 0) { computed = 1;
#define VEC_COMPUTED_END    }
#else
#define VEC_COMPUTED_INIT
#define VEC_COMPUTED_CHECK
#define VEC_COMPUTED_END
#endif

/* END Macros */


/****************************************************************************/
/*                                                       NECESSARY INCLUDES */
#include <interfaces.h>
#include <adalloc.h>
#include <oplate.h>
#include <taping_p.h>
#include <convolut.h>

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
int hos_reverse(short   tnum,        /* tape id */
                int     depen,       /* consistency chk on # of deps */
                int     indep,       /* consistency chk on # of indeps */
                int     degre,       /* highest derivative degre  */
                double  *lagrange,   /* range weight vector       */
                double  **results)   /* matrix of coefficient vectors */
{ int i, j, rc;
    double** L = myalloc2(depen,degre+1);
    for ( i = 0; i < depen; ++i ) {
        L[i][0] = lagrange[i];
        for ( j = 1; j <= degre; ++j )
            L[i][j] = 0.0;
    }
    rc = hos_ti_reverse(tnum,depen,indep,degre,L,results);
    myfree2(L);
    return rc;
}

int hos_ti_reverse(
    short   tnum,        /* tape id */
    int     depen,       /* consistency chk on # of deps */
    int     indep,       /* consistency chk on # of indeps */
    int     degre,       /* highest derivative degre  */
    double  **lagrange,  /* range weight vectors       */
    double  **results)   /* matrix of coefficient vectors */

#elif _HOS_OV_

/***************************************************************************/
/* Higher Order Scalar Reverse Pass, Vector Keep.                          */
/***************************************************************************/
int hos_ov_reverse(short   tnum,       /* tape id */
                   int     depen,       /* consistency chk on # of deps */
                   int     indep,       /* consistency chk on # of indeps */
                   int     degre,       /* highest derivative degre  */
                   int     nrows,       /* # of Jacobian rows calculated */
                   double  **lagrange,  /* range weight vector       */
                   double  ***results)  /* matrix of coefficient vectors */

#elif _HOV_
/***************************************************************************/
/* Higher Order Vector Reverse Pass.                                       */
/***************************************************************************/
int hov_reverse(short   tnum,        /* tape id */
                int     depen,       /* consistency chk on # of deps */
                int     indep,       /* consistency chk on # of indeps */
                int     degre,       /* highest derivative degre */
                int     nrows,       /* # of Jacobian rows calculated */
                double  **lagrange,  /* domain weight vector */
                double  ***results,  /* matrix of coefficient vectors */
                short   **nonzero )  /* structural sparsity  pattern  */
{ int i, j, k, rc;
    double*** L = myalloc3(nrows,depen,degre+1);
    for ( k = 0; k < nrows; ++k )
        for ( i = 0; i < depen; ++i ) {
            L[k][i][0] = lagrange[k][i];
            for ( j = 1; j <= degre; ++j )
                L[k][i][j] = 0.0;
        }
    rc = hov_ti_reverse(tnum,depen,indep,degre,nrows,L,results,nonzero);
    myfree3(L);
    return rc;
}

int hov_ti_reverse(
    short   tnum,        /* tape id */
    int     depen,       /* consistency chk on # of deps */
    int     indep,       /* consistency chk on # of indeps */
    int     degre,       /* highest derivative degre */
    int     nrows,       /* # of Jacobian rows calculated */
    double  ***lagrange, /* domain weight vectors */
    double  ***results,  /* matrix of coefficient vectors */
    short   **nonzero )  /* structural sparsity  pattern  */

#endif

{
    /************************************************************************/
    /*                                                       ALL VARIABLES  */
    ADOLC_OPENMP_THREAD_NUMBER;
    unsigned char operation;   /* operation code */
    int dc, ret_c=3;

    locint size = 0;
    locint res  = 0;
    locint arg  = 0;
    locint arg1 = 0;
    locint arg2 = 0;

    double coval = 0, *d = 0;

    int indexi = 0,  indexd = 0;

    /* loop indices */
    int i, j, l, ls;

    /* other necessary variables */
    double *x;
    int *jj;
    int taycheck;
    int numdep,numind;
    double aTmp;

    /*----------------------------------------------------------------------*/
    /* Taylor stuff */
    revreal *Tres, *Targ, *Targ1, *Targ2, *Tqo, *rp_Ttemp, *rp_Ttemp2;
    revreal **rpp_T;

    /*----------------------------------------------------------------------*/
    /* Adjoint stuff */
#ifdef _FOS_
    double Atemp;
# define A_TEMP Atemp
#endif
    double *Ares, *Aarg=NULL, *Aarg1, *Aarg2, *Aqo, *dp_Atemp, *dp_Atemp2;
    double **dpp_A, *AP1, *AP2;

    /*----------------------------------------------------------------------*/
    int k = degre + 1;
    int k1 = k + 1;
    revreal comp;

#ifdef _ADOLC_VECTOR_
    int p = nrows;
#endif

#ifdef _HOV_
    int pk1 = p*k1;
    int q = 1;
#elif _HOS_OV_
    int p = nrows;
    int pk1 = p*k1;
    int q = p;
#else
    int q = 1;
#endif

    ADOLC_OPENMP_GET_THREAD_NUMBER;

#if defined(ADOLC_DEBUG)
    /************************************************************************/
    /*                                                       DEBUG MESSAGES */
    fprintf(DIAG_OUT,"Call of %s(..) with tag: %d, n: %d, m %d,\n",
            GENERATED_FILENAME, tnum, indep, depen);

#ifdef _HIGHER_ORDER_
    fprintf(DIAG_OUT,"                    degree: %d\n",degre);
#endif
#ifdef _ADOLC_VECTOR_
    fprintf(DIAG_OUT,"                    p: %d\n\n",nrows);
#endif

#endif


    /************************************************************************/
    /*                                                                INITs */

    /*----------------------------------------------------------------------*/
    /* Set up stuff for the tape */

    /* Initialize the Reverse Sweep */
    init_rev_sweep(tnum);

    if ( (depen != ADOLC_CURRENT_TAPE_INFOS.stats[NUM_DEPENDENTS]) ||
            (indep != ADOLC_CURRENT_TAPE_INFOS.stats[NUM_INDEPENDENTS]) )
        fail(ADOLC_REVERSE_COUNTS_MISMATCH);

    indexi = ADOLC_CURRENT_TAPE_INFOS.stats[NUM_INDEPENDENTS] - 1;
    indexd = ADOLC_CURRENT_TAPE_INFOS.stats[NUM_DEPENDENTS] - 1;


    /************************************************************************/
    /*                                              MEMORY ALLOCATION STUFF */

    /*----------------------------------------------------------------------*/
#ifdef _HOS_                                                         /* HOS */
    dpp_A = myalloc2(ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES], k1);
    rpp_T = (revreal**)malloc(ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES] *
            sizeof(revreal*));
    if (rpp_T == NULL) fail(ADOLC_MALLOC_FAILED);
    Tqo = (revreal *)malloc(ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES] *
            k * sizeof(revreal));
    if (Tqo ==NULL) fail(ADOLC_MALLOC_FAILED);
    for (i=0; i<ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES]; i++) {
        rpp_T[i] = Tqo;
        Tqo += k;
    }
    dp_Atemp  = myalloc1(k1);
    dp_Atemp2 = myalloc1(k1);
    rp_Ttemp2 = (revreal *)malloc(k * sizeof(revreal));
    /*----------------------------------------------------------------------*/
#elif _HOV_                                                          /* HOV */
    dpp_A = myalloc2(ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES], pk1);
    rpp_T = (revreal**)malloc(ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES] *
            sizeof(revreal*));
    if (rpp_T == NULL) fail(ADOLC_MALLOC_FAILED);
    Tqo = (revreal*)malloc(ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES] *
            k * sizeof(revreal));
    if (Tqo == NULL) fail(ADOLC_MALLOC_FAILED);
    for (i=0; i<ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES]; i++) {
        rpp_T[i] = Tqo;
        Tqo += k;
    }
    dp_Atemp  = myalloc1(pk1);
    dp_Atemp2 = myalloc1(pk1);
    rp_Ttemp2 = (revreal*) malloc(k * sizeof(revreal));
    /*----------------------------------------------------------------------*/
#elif _HOS_OV_                                                    /* HOS_OV */
    dpp_A = myalloc2(ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES], pk1);
    rpp_T = (revreal**)malloc(ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES] *
            sizeof(revreal*));
    if (rpp_T == NULL) fail(ADOLC_MALLOC_FAILED);
    Tqo = (revreal*)malloc(ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES] *
            p * k * sizeof(revreal));
    if (Tqo == NULL) fail(ADOLC_MALLOC_FAILED);
    for (i=0; i<ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES]; i++) {
        rpp_T[i] = Tqo;
        Tqo += p*k;
    }
    dp_Atemp  = myalloc1(pk1);
    dp_Atemp2 = myalloc1(pk1);
    rp_Ttemp2 = (revreal*) malloc(p*k*sizeof(revreal));
#endif
    rp_Ttemp  = (revreal*) malloc(k*sizeof(revreal));
    if (rp_Ttemp == NULL) fail(ADOLC_MALLOC_FAILED);
    if (rp_Ttemp2 == NULL) fail(ADOLC_MALLOC_FAILED);
    x = myalloc1(q);
    jj = (int*)malloc(q*sizeof(int));
    if (jj == NULL) fail(ADOLC_MALLOC_FAILED);

    /************************************************************************/
    /*                                                TAYLOR INITIALIZATION */
    ADOLC_CURRENT_TAPE_INFOS.dpp_A = dpp_A;
    ADOLC_CURRENT_TAPE_INFOS.rpp_T = rpp_T;
    taylor_back(tnum,&numdep,&numind,&taycheck);

    if(taycheck != degre) {
        fprintf(DIAG_OUT,"\n ADOL-C error: reverse fails because it was not"
                " preceeded\nby a forward sweep with degree>%i,"
                " keep=%i!\n",degre,degre+1);
        exit(-2);
    };

    if((numdep != depen)||(numind != indep)) {
        fprintf(DIAG_OUT,"\n ADOL-C error: reverse fails on tape %d because "
                "the number of\nindependent and/or dependent variables"
                " given to reverse are\ninconsistent with that of the"
                "  internal taylor array.\n",tnum);
        exit(-2);
    }


    /************************************************************************/
    /*                                                        REVERSE SWEEP */

#if defined(ADOLC_DEBUG)
    int v = 0;
    unsigned int countPerOperation[256], taylorPerOperation[256];
    memset(countPerOperation, 0, 1024);
    memset(taylorPerOperation, 0, 1024);
#   define UPDATE_TAYLORREAD(X) taylorPerOperation[operation] += X;
#else
#   define UPDATE_TAYLORREAD(X)
#endif /* ADOLC_DEBUG */

    operation=get_op_r();
#if defined(ADOLC_DEBUG)
    ++countPerOperation[operation];
#endif /* ADOLC_DEBUG */
    while (operation != start_of_tape) { 
        /* Switch statement to execute the operations in Reverse */
        switch (operation) {


                /************************************************************/
                /*                                                  MARKERS */

                /*----------------------------------------------------------*/
            case end_of_op:                                    /* end_of_op */
                get_op_block_r();
                operation = get_op_r();
                /* Skip next operation, it's another end_of_op */
                break;

                /*----------------------------------------------------------*/
            case end_of_int:                                  /* end_of_int */
                get_loc_block_r(); /* Get the next int block */
                break;

                /*----------------------------------------------------------*/
            case end_of_val:                                  /* end_of_val */
                get_val_block_r(); /* Get the next val block */
                break;

                /*----------------------------------------------------------*/
            case start_of_tape:                            /* start_of_tape */
            case end_of_tape:                                /* end_of_tape */
                break;


                /************************************************************/
                /*                                               COMPARISON */

                /*----------------------------------------------------------*/
            case eq_zero  :                                      /* eq_zero */
                arg   = get_locint_r();

                ret_c = 0;
                break;

                /*----------------------------------------------------------*/
            case neq_zero :                                     /* neq_zero */
            case gt_zero  :                                      /* gt_zero */
            case lt_zero :                                       /* lt_zero */
                arg   = get_locint_r();
                break;

                /*----------------------------------------------------------*/
            case ge_zero :                                       /* ge_zero */
            case le_zero :                                       /* le_zero */
                arg   = get_locint_r();

                if (*rpp_T[arg] == 0)
                    ret_c = 0;
                break;


                /************************************************************/
                /*                                              ASSIGNMENTS */

                /*----------------------------------------------------------*/
            case assign_a:     /* assign an adouble variable an    assign_a */
                /* adouble value. (=) */
                res = get_locint_r();
                arg = get_locint_r();

                ASSIGN_A(Aarg, dpp_A[arg])
                ASSIGN_A(Ares, dpp_A[res])

                FOR_0_LE_l_LT_p
                if  (0 == ARES) {
                    HOV_INC(Aarg, k1)
                    HOV_INC(Ares, k1)
                } else {
                    MAXDEC(AARG,ARES);
                    AARG_INC_O;
                    ARES_INC = 0.0;
                    FOR_0_LE_i_LT_k
                    { /* ! no tempory */
                        AARG_INC += ARES;
                        ARES_INC = 0.0;
                    }
                }

                GET_TAYL(res,k,p)
                break;

                /*----------------------------------------------------------*/
            case assign_d:      /* assign an adouble variable a    assign_d */
                /* double value. (=) */
                res   = get_locint_r();
                coval = get_val_r();

                ASSIGN_A(Ares, dpp_A[res])

                FOR_0_LE_l_LT_pk1
                ARES_INC = 0.0;

                GET_TAYL(res,k,p)
                break;

                /*----------------------------------------------------------*/
            case assign_d_zero: /* assign an adouble a        assign_d_zero */
            case assign_d_one:  /* double value. (=)           assign_d_one */
                res   = get_locint_r();

                ASSIGN_A(Ares, dpp_A[res])

                FOR_0_LE_l_LT_pk1
                ARES_INC = 0.0;

                GET_TAYL(res,k,p)
                break;

                /*--------------------------------------------------------------------------*/
            case assign_ind:       /* assign an adouble variable an    assign_ind */
                /* independent double value (<<=) */
                res = get_locint_r();

                ASSIGN_A(Ares, dpp_A[res])

                FOR_0_LE_l_LT_p
                {
#ifdef _HOV_
                    if (nonzero) /* ??? question: why here? */
                    nonzero[l][indexi] = (int)ARES;
#endif /* _HOV_ */
                    ARES_INC_O;
                    FOR_0_LE_i_LT_k
                        RESULTS(l,indexi,i) = ARES_INC;
                }

                GET_TAYL(res,k,p)
                    indexi--;
                break;

                /*--------------------------------------------------------------------------*/
            case assign_dep:           /* assign a float variable a    assign_dep */
                /* dependent adouble value. (>>=) */
                res = get_locint_r();

                ASSIGN_A(Ares, dpp_A[res])
                ASSIGN_A(Aarg, dpp_A[res])   /* just a helpful pointers */

                FOR_0_LE_l_LT_p
                { ARES_INC_O;
                  dc = -1;
                  FOR_0_LE_i_LT_k
                  { ARES_INC = LAGRANGE(l,indexd,i);
                    if (LAGRANGE(l,indexd,i)) dc = i;
                  }
                  AARG = (dc < 0)? 0.0 : (dc > 0)? 2.0 : 1.0;
                  HOV_INC(Aarg, k1)
                }
                indexd--;
            break;


            /****************************************************************************/
            /*                                                   OPERATION + ASSIGNMENT */

            /*--------------------------------------------------------------------------*/
        case eq_plus_d:            /* Add a floating point to an    eq_plus_d */
            /* adouble. (+=) */
            res   = get_locint_r();
                coval = get_val_r();

                GET_TAYL(res,k,p)
                break;

                /*--------------------------------------------------------------------------*/
            case eq_plus_a:             /* Add an adouble to another    eq_plus_a */
                /* adouble. (+=) */
                res = get_locint_r();
                arg = get_locint_r();

                ASSIGN_A(Ares, dpp_A[res])
                ASSIGN_A(Aarg, dpp_A[arg]);

                FOR_0_LE_l_LT_p
                if  (0 == ARES) {
                    HOV_INC(Ares, k1)
                    HOV_INC(Aarg, k1)
                } else {
                    MAXDEC(AARG,ARES);
                    AARG_INC_O;
                    ARES_INC_O;
                    FOR_0_LE_i_LT_k
                    AARG_INC += ARES_INC;
                }

                GET_TAYL(res,k,p)
                break;

                /*--------------------------------------------------------------------------*/
            case eq_min_d:       /* Subtract a floating point from an    eq_min_d */
                /* adouble. (-=) */
                res   = get_locint_r();
                coval = get_val_r();

                GET_TAYL(res,k,p)
                break;

                /*--------------------------------------------------------------------------*/
            case eq_min_a:        /* Subtract an adouble from another    eq_min_a */
                /* adouble. (-=) */
                res = get_locint_r();
                arg = get_locint_r();

                ASSIGN_A(Ares, dpp_A[res])
                ASSIGN_A(Aarg, dpp_A[arg])

                FOR_0_LE_l_LT_p
                if  (0==ARES) {
                    HOV_INC(Ares, k1)
                    HOV_INC(Aarg, k1)
                } else {
                    MAXDEC(AARG,ARES);
                    AARG_INC_O;
                    ARES_INC_O;
                    FOR_0_LE_i_LT_k
                    AARG_INC -= ARES_INC;
                }

                GET_TAYL(res,k,p)
                break;

                /*--------------------------------------------------------------------------*/
            case eq_mult_d:              /* Multiply an adouble by a    eq_mult_d */
                /* flaoting point. (*=) */
                res   = get_locint_r();
                coval = get_val_r();

                ASSIGN_A(Ares, dpp_A[res])

                FOR_0_LE_l_LT_p
                if ( 0 == ARES_INC )
                    HOV_INC(Ares, k)
                    else
                        FOR_0_LE_i_LT_k
                        ARES_INC *= coval;

                GET_TAYL(res,k,p)
                break;

                /*--------------------------------------------------------------------------*/
            case eq_mult_a:       /* Multiply one adouble by another    eq_mult_a */
                /* (*=) */
                res = get_locint_r();
                arg = get_locint_r();

                GET_TAYL(res,k,p)

                ASSIGN_A(Ares, dpp_A[res])
                ASSIGN_A(Aarg, dpp_A[arg])
                ASSIGN_A(Aqo,  dp_Atemp)
                ASSIGN_T(Tres, rpp_T[res])
                ASSIGN_T(Targ, rpp_T[arg])

                FOR_0_LE_l_LT_p {
                    if (0 == ARES) {
                    HOV_INC(Aarg, k1)
                        HOV_INC(Ares, k1)
                    } else {
                        MAXDEC(ARES,2.0);
                        MAXDEC(AARG,ARES);
                        AARG_INC_O;
                        ARES_INC_O;
                        conv(k,Ares,Targ,dp_Atemp);
                        if(arg != res) {
                            inconv(k,Ares,Tres,Aarg);
                            FOR_0_LE_i_LT_k
                            ARES_INC = AQO_INC;
                        } else
                            FOR_0_LE_i_LT_k
                            ARES_INC = 2.0 * AQO_INC;
                        HOV_INC(Aarg,k)
                        HOS_OV_INC(Tres,k)
                        HOS_OV_INC(Targ,k)
                        HOS_OV_ASSIGN_A(Aqo,  dp_Atemp)
                    }
            }
                break;

                /*--------------------------------------------------------------------------*/
            case incr_a:                        /* Increment an adouble    incr_a */
            case decr_a:                        /* Increment an adouble    decr_a */
                res   = get_locint_r();

                GET_TAYL(res,k,p)
                break;


                /****************************************************************************/
                /*                                                        BINARY OPERATIONS */

                /*--------------------------------------------------------------------------*/
            case plus_a_a:                 /* : Add two adoubles. (+)    plus a_a */
                res  = get_locint_r();
                arg2 = get_locint_r();
                arg1 = get_locint_r();

                ASSIGN_A(Ares,  dpp_A[res])
                ASSIGN_A(Aarg1, dpp_A[arg1])
                ASSIGN_A(Aarg2, dpp_A[arg2])

                FOR_0_LE_l_LT_p
                if  (0 == ARES) {
                    HOV_INC(Ares,  k1)
                    HOV_INC(Aarg1, k1)
                    HOV_INC(Aarg2, k1)
                } else {
                    aTmp = ARES;
                    ARES_INC = 0.0;
                    MAXDEC(AARG1,aTmp);
                    MAXDEC(AARG2,aTmp);
                    AARG2_INC_O;
                    AARG1_INC_O;
                    FOR_0_LE_i_LT_k
                    { aTmp = ARES;
                      ARES_INC = 0.0;
                      AARG1_INC += aTmp;
                      AARG2_INC += aTmp;
                    }
                }

                GET_TAYL(res,k,p)
                break;

                /*--------------------------------------------------------------------------*/
            case plus_d_a:             /* Add an adouble and a double    plus_d_a */
                /* (+) */
                res   = get_locint_r();
                arg   = get_locint_r();
                coval = get_val_r();

                ASSIGN_A(Ares, dpp_A[res])
                ASSIGN_A(Aarg, dpp_A[arg])

                FOR_0_LE_l_LT_p
                if  (0 == ARES) {
                    HOV_INC(Ares, k1)
                    HOV_INC(Aarg, k1)
                } else {
                    aTmp = ARES;
                    ARES_INC = 0.0;
                    MAXDEC(AARG,aTmp);
                    AARG_INC_O;
                    FOR_0_LE_i_LT_k
                    { aTmp = ARES;
                      ARES_INC = 0.0;
                      AARG_INC += aTmp;
                    }
                }

                GET_TAYL(res,k,p)
                break;

                /*--------------------------------------------------------------------------*/
            case min_a_a:              /* Subtraction of two adoubles    min_a_a */
                /* (-) */
                res  = get_locint_r();
                arg2 = get_locint_r();
                arg1 = get_locint_r();

                ASSIGN_A(Ares,  dpp_A[res])
                ASSIGN_A(Aarg1, dpp_A[arg1])
                ASSIGN_A(Aarg2, dpp_A[arg2])

                FOR_0_LE_l_LT_p
                if  (0 == ARES) {
                    HOV_INC(Ares,  k1)
                    HOV_INC(Aarg1, k1)
                    HOV_INC(Aarg2, k1)
                } else {
                    aTmp = ARES;
                    ARES_INC = 0.0;
                    MAXDEC(AARG1,aTmp);
                    MAXDEC(AARG2,aTmp);
                    AARG2_INC_O;
                    AARG1_INC_O;
                    FOR_0_LE_i_LT_k
                    { aTmp = ARES;
                      ARES_INC = 0.0;
                      AARG1_INC += aTmp;
                      AARG2_INC -= aTmp;
                    }
                }

                GET_TAYL(res,k,p)
                break;

                /*--------------------------------------------------------------------------*/
            case min_d_a:                /* Subtract an adouble from a    min_d_a */
                /* double (-) */
                res   = get_locint_r();
                arg   = get_locint_r();
                coval = get_val_r();

                ASSIGN_A(Ares, dpp_A[res])
                ASSIGN_A(Aarg, dpp_A[arg])

                FOR_0_LE_l_LT_p
                if (0 == ARES) {
                    HOV_INC(Ares, k1)
                    HOV_INC(Aarg, k1)
                } else {
                    aTmp = ARES;
                    ARES_INC = 0.0;
                    MAXDEC(AARG,aTmp);
                    AARG_INC_O;
                    FOR_0_LE_i_LT_k
                    { aTmp = ARES;
                      ARES_INC = 0.0;
                      AARG_INC -= aTmp;
                    }
                }

                GET_TAYL(res,k,p)
                break;

                /*--------------------------------------------------------------------------*/
            case mult_a_a:               /* Multiply two adoubles (*)    mult_a_a */
                res  = get_locint_r();
                arg2 = get_locint_r();
                arg1 = get_locint_r();

                GET_TAYL(res,k,p)

                ASSIGN_A(Ares,  dpp_A[res])
                ASSIGN_A(Aarg2, dpp_A[arg2])
                ASSIGN_A(Aarg1, dpp_A[arg1])
                ASSIGN_T(Targ1, rpp_T[arg1])
                ASSIGN_T(Targ2, rpp_T[arg2])

                FOR_0_LE_l_LT_p
                if (0 == ARES) {
                    HOV_INC(Aarg1, k1)
                    HOV_INC(Aarg2, k1)
                    HOV_INC(Ares,  k1)
                } else {
                    comp = (ARES > 2.0) ? ARES : 2.0 ;
                    ARES_INC = 0.0;
                    MAXDEC(AARG1,comp);
                    MAXDEC(AARG2,comp);
                    AARG1_INC_O;
                    AARG2_INC_O;

                    copyAndZeroset(k,Ares,dp_Atemp);
                    inconv(k,dp_Atemp,Targ1,Aarg2);
                    inconv(k,dp_Atemp,Targ2,Aarg1);

                    HOV_INC(Ares,  k)
                    HOV_INC(Aarg1, k)
                    HOV_INC(Aarg2, k)
                    HOS_OV_INC(Targ1, k)
                    HOS_OV_INC(Targ2, k)
                }
                break;

                /*--------------------------------------------------------------------------*/
                /* olvo 991122: new op_code with recomputation */
            case eq_plus_prod:   /* increment a product of           eq_plus_prod */
                /* two adoubles (*) */
                res  = get_locint_r();
                arg2 = get_locint_r();
                arg1 = get_locint_r();


                ASSIGN_A(Ares,  dpp_A[res])
                ASSIGN_A(Aarg2, dpp_A[arg2])
                ASSIGN_A(Aarg1, dpp_A[arg1])
                ASSIGN_T(Targ1, rpp_T[arg1])
                ASSIGN_T(Targ2, rpp_T[arg2])

                /* RECOMPUTATION */
                ASSIGN_T( Tres,  rpp_T[res])
                deconv1(k,Targ1,Targ2,Tres);

                FOR_0_LE_l_LT_p
                if (0 == ARES) {
                    HOV_INC(Aarg1, k1)
                    HOV_INC(Aarg2, k1)
                    HOV_INC(Ares,  k1)
                } else {
                    comp = (ARES > 2.0) ? ARES : 2.0 ;
                    ARES_INC = comp;
                    MAXDEC(AARG1,comp);
                    MAXDEC(AARG2,comp);
                    AARG1_INC_O;
                    AARG2_INC_O;

                    inconv(k,Ares,Targ1,Aarg2);
                    inconv(k,Ares,Targ2,Aarg1);

                    HOV_INC(Ares,  k)
                    HOV_INC(Aarg1, k)
                    HOV_INC(Aarg2, k)
                    HOS_OV_INC(Targ1, k)
                    HOS_OV_INC(Targ2, k)
                }
                break;

                /*--------------------------------------------------------------------------*/
                /* olvo 991122: new op_code with recomputation */
            case eq_min_prod:   /* decrement a product of             eq_min_prod */
                /* two adoubles (*) */
                res  = get_locint_r();
                arg2 = get_locint_r();
                arg1 = get_locint_r();


                ASSIGN_A(Ares,  dpp_A[res])
                ASSIGN_A(Aarg2, dpp_A[arg2])
                ASSIGN_A(Aarg1, dpp_A[arg1])
                ASSIGN_T(Targ1, rpp_T[arg1])
                ASSIGN_T(Targ2, rpp_T[arg2])

                /* RECOMPUTATION */
                ASSIGN_T( Tres,  rpp_T[res])
                inconv1(k,Targ1,Targ2,Tres);

                FOR_0_LE_l_LT_p
                if (0 == ARES) {
                    HOV_INC(Aarg1, k1)
                    HOV_INC(Aarg2, k1)
                    HOV_INC(Ares,  k1)
                } else {
                    comp = (ARES > 2.0) ? ARES : 2.0 ;
                    ARES_INC = comp;
                    MAXDEC(AARG1,comp);
                    MAXDEC(AARG2,comp);
                    AARG1_INC_O;
                    AARG2_INC_O;

                    deconv0(k,Ares,Targ1,Aarg2);
                    deconv0(k,Ares,Targ2,Aarg1);

                    HOV_INC(Ares,  k)
                    HOV_INC(Aarg1, k)
                    HOV_INC(Aarg2, k)
                    HOS_OV_INC(Targ1, k)
                    HOS_OV_INC(Targ2, k)
                }
                break;

                /*--------------------------------------------------------------------------*/
            case mult_d_a:         /* Multiply an adouble by a double    mult_d_a */
                /* (*) */
                res   = get_locint_r();
                arg   = get_locint_r();
                coval = get_val_r();

                ASSIGN_A(Ares, dpp_A[res])
                ASSIGN_A(Aarg, dpp_A[arg])

                FOR_0_LE_l_LT_p
                if (0 == ARES) {
                    HOV_INC(Ares, k1)
                    HOV_INC(Aarg, k1)
                } else {
                    aTmp = ARES;
                    ARES_INC = 0.0;
                    MAXDEC(AARG,aTmp);
                    AARG_INC_O;
                    FOR_0_LE_i_LT_k
                    { aTmp = ARES;
                      ARES_INC = 0.0;
                      AARG_INC += coval * aTmp;
                    }
                }

                GET_TAYL(res,k,p)
                break;

                /*--------------------------------------------------------------------------*/
            case div_a_a:           /* Divide an adouble by an adouble    div_a_a */
                /* (/) */
                res  = get_locint_r();
                arg2 = get_locint_r();
                arg1 = get_locint_r();

                ASSIGN_A(Ares,  dpp_A[res])
                ASSIGN_A(Aarg2, dpp_A[arg2])
                ASSIGN_A(Aarg1, dpp_A[arg1])
                ASSIGN_T(Tres,  rpp_T[res])
                ASSIGN_T(Targ2, rpp_T[arg2])

                /* olvo 980922 allows reflexive operation */
                if (arg2 == res) {
                    FOR_0_LE_l_LT_pk
                    rp_Ttemp2[l] = Tres[l];
                    Tres = rp_Ttemp2;
                    GET_TAYL(res,k,p)
                }

                VEC_COMPUTED_INIT
                FOR_0_LE_l_LT_p
                { if (0 == ARES) {
                  HOV_INC(Ares,  k1)
                      HOV_INC(Aarg1, k1)
                      HOV_INC(Aarg2, k1)
                  } else {
                      aTmp = ARES;
                      ARES_INC = 0.0;
                      MAXDEC(AARG1,3.0);
                      MAXDEC(AARG1,aTmp);
                      MAXDEC(AARG2,3.0);
                      MAXDEC(AARG2,aTmp);
                      AARG1_INC_O;
                      AARG2_INC_O;

                      VEC_COMPUTED_CHECK
                      recipr(k,1.0,Targ2,rp_Ttemp);
                      conv0(k ,rp_Ttemp,
                           Tres, dp_Atemp2);
                      VEC_COMPUTED_END
                      copyAndZeroset(k,Ares,dp_Atemp);
                      inconv(k, dp_Atemp,
                             rp_Ttemp, Aarg1);
                      deconv(k, dp_Atemp,
                             dp_Atemp2, Aarg2);

                      HOV_INC(Ares,  k)
                      HOV_INC(Aarg1, k)
                      HOV_INC(Aarg2, k)
                      HOS_OV_INC(Tres, k)
                      HOS_OV_INC(Targ2, k)
                  }
            }

                if (res != arg2)
                    GET_TAYL(res,k,p)
                    break;

                /*--------------------------------------------------------------------------*/
            case div_d_a:             /* Division double - adouble (/)    div_d_a */
                res   = get_locint_r();
                arg   = get_locint_r();
                coval = get_val_r();

                ASSIGN_A(Ares, dpp_A[res])
                ASSIGN_A(Aarg, dpp_A[arg])
                ASSIGN_T(Tres, rpp_T[res])
                ASSIGN_T(Targ, rpp_T[arg])

                /* olvo 980922 allows reflexive operation */
                if (arg == res) {
                    FOR_0_LE_l_LT_pk
                    rp_Ttemp2[l] = Tres[l];
                    Tres = rp_Ttemp2;
                    GET_TAYL(arg,k,p)
                }

                VEC_COMPUTED_INIT
                FOR_0_LE_l_LT_p
                { if (0 == ARES) {
                  HOV_INC(Ares, k1)
                      HOV_INC(Aarg, k1)
                  } else {
                      aTmp = ARES;
                      ARES_INC = 0.0;
                      MAXDEC(AARG,aTmp);
                      MAXDEC(AARG,3.0);
                      AARG_INC_O;

                      VEC_COMPUTED_CHECK
                      recipr(k,1.0,Targ,rp_Ttemp);
                      conv0(k, rp_Ttemp,
                           Tres, dp_Atemp);
                      VEC_COMPUTED_END
                      deconv(k,Ares,dp_Atemp,Aarg);

                      HOV_INC(Ares, k)
                      HOV_INC(Aarg, k)
                      HOS_OV_INC(Tres, k)
                      HOS_OV_INC(Targ, k)
                  }
            }

                GET_TAYL(res,k,p)
                break;


                /****************************************************************************/
                /*                                                         SIGN  OPERATIONS */

                /*--------------------------------------------------------------------------*/
            case pos_sign_a:                                        /* pos_sign_a */
                res   = get_locint_r();
                arg   = get_locint_r();

                ASSIGN_A(Ares, dpp_A[res])
                ASSIGN_A(Aarg, dpp_A[arg])

                FOR_0_LE_l_LT_p
                if  (0 == ARES) {
                    HOV_INC(Ares, k1)
                    HOV_INC(Aarg, k1)
                } else {
                    aTmp = ARES;
                    ARES_INC = 0.0;
                    MAXDEC(AARG,aTmp);
                    AARG_INC_O;
                    FOR_0_LE_i_LT_k
                    { aTmp = ARES;
                      ARES_INC = 0.0;
                      AARG_INC += aTmp;
                    }
                }

                GET_TAYL(res,k,p)
                break;

                /*--------------------------------------------------------------------------*/
            case neg_sign_a:                                        /* neg_sign_a */
                res   = get_locint_r();
                arg   = get_locint_r();

                ASSIGN_A(Ares, dpp_A[res])
                ASSIGN_A(Aarg, dpp_A[arg])

                FOR_0_LE_l_LT_p
                if  (0 == ARES) {
                    HOV_INC(Ares, k1)
                    HOV_INC(Aarg, k1)
                } else {
                    aTmp = ARES;
                    ARES_INC = 0.0;
                    MAXDEC(AARG,aTmp);
                    AARG_INC_O;
                    FOR_0_LE_i_LT_k
                    { aTmp = ARES;
                      ARES_INC = 0.0;
                      AARG_INC -= aTmp;
                    }
                }

                GET_TAYL(res,k,p)
                break;


                /****************************************************************************/
                /*                                                         UNARY OPERATIONS */

                /*--------------------------------------------------------------------------*/
            case exp_op:                          /* exponent operation    exp_op */
                res = get_locint_r();
                arg = get_locint_r();

                ASSIGN_A(Ares, dpp_A[res])
                ASSIGN_A(Aarg, dpp_A[arg])
                ASSIGN_T(Tres, rpp_T[res])
                ASSIGN_T(Targ, rpp_T[arg])

                FOR_0_LE_l_LT_p
                { if (0 == ARES) {
                  HOV_INC(Aarg, k1)
                      HOV_INC(Ares, k1)
                  } else {
                      aTmp = ARES;
                      ARES_INC = 0.0;
                      MAXDEC(AARG,aTmp);
                      MAXDEC(AARG,4.0);
                      AARG_INC_O;

                      inconv0(k,Ares,Tres,Aarg);

                      HOV_INC(Ares, k)
                      HOV_INC(Aarg, k)
                      HOS_OV_INC(Tres, k)
                  }
            }

                GET_TAYL(res,k,p)
                break;

                /*--------------------------------------------------------------------------*/
            case sin_op:                              /* sine operation    sin_op */
                res  = get_locint_r();
                arg2 = get_locint_r();
                arg1 = get_locint_r();

                ASSIGN_A(Ares,  dpp_A[res])
                ASSIGN_A(Aarg1, dpp_A[arg1])
                ASSIGN_T(Targ2, rpp_T[arg2])

                FOR_0_LE_l_LT_p
                { if (0 == ARES) {
                  HOV_INC(Aarg1, k1)
                      HOV_INC(Ares,  k1)
                  } else {
                      aTmp = ARES;
                      ARES_INC = 0.0;
                      MAXDEC(AARG1,aTmp);
                      MAXDEC(AARG1,4.0);
                      AARG1_INC_O;

                      inconv0(k,Ares,Targ2,Aarg1);

                      HOV_INC(Ares,  k)
                      HOV_INC(Aarg1, k)
                      HOS_OV_INC(Targ2, k)
                  }
            }

                GET_TAYL(res,k,p)
                GET_TAYL(arg2,k,p) /* olvo 980710 covalue */
                /* NOTE: dpp_A[arg2] should be 0 already */
                break;

                /*--------------------------------------------------------------------------*/
            case cos_op:                            /* cosine operation    cos_op */
                res  = get_locint_r();
                arg2 = get_locint_r();
                arg1 = get_locint_r();

                ASSIGN_A(Ares,  dpp_A[res])
                ASSIGN_A(Aarg1, dpp_A[arg1])
                ASSIGN_T(Targ2, rpp_T[arg2])

                FOR_0_LE_l_LT_p
                { if (0 == ARES) {
                  HOV_INC(Aarg1, k1)
                      HOV_INC(Ares,  k1)
                  } else {
                      aTmp = ARES;
                      ARES_INC = 0.0;
                      MAXDEC(AARG1,aTmp);
                      MAXDEC(AARG1,4.0);
                      AARG1_INC_O;

                      deconv0(k,Ares,Targ2,Aarg1);

                      HOV_INC(Ares,  k)
                      HOV_INC(Aarg1, k)
                      HOS_OV_INC(Targ2, k)
                  }
            }

                GET_TAYL(res,k,p)
                GET_TAYL(arg2,k,p) /* olvo 980710 covalue */
                /* NOTE: dpp_A[arg2] should be 0 already */
                break;
                /*xxx*/
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

                GET_TAYL(res,k,p)

                ASSIGN_A(Ares,  dpp_A[res])
                ASSIGN_A(Aarg1, dpp_A[arg1])
                ASSIGN_T(Targ2, rpp_T[arg2])

                FOR_0_LE_l_LT_p
                { if (0 == ARES) {
                  HOV_INC(Aarg1, k1)
                      HOV_INC(Ares,  k1)
                  } else {
                      aTmp = ARES;
                      ARES_INC = 0.0;
                      MAXDEC(AARG1,aTmp);
                      MAXDEC(AARG1,4.0);
                      AARG1_INC_O;

                      inconv0(k,Ares,Targ2,Aarg1);

                      HOV_INC(Aarg1, k)
                      HOV_INC(Ares,  k)
                      HOS_OV_INC(Targ2, k)
                  }
            }
                break;

                /*--------------------------------------------------------------------------*/
            case log_op:                                                /* log_op */
                res = get_locint_r();
                arg = get_locint_r();

                GET_TAYL(res,k,p)

                ASSIGN_A(Ares, dpp_A[res])
                ASSIGN_A(Aarg, dpp_A[arg])
                ASSIGN_T(Targ, rpp_T[arg])

                VEC_COMPUTED_INIT
                FOR_0_LE_l_LT_p
                { if (0 == ARES) {
                  HOV_INC(Aarg, k1)
                      HOV_INC(Ares, k1)
                  } else {
                      aTmp = ARES;
                      ARES_INC = 0.0;
                      MAXDEC(AARG,aTmp);
                      MAXDEC(AARG,4.0);
                      AARG_INC_O;

                      VEC_COMPUTED_CHECK
                      recipr(k,1.0,Targ,rp_Ttemp);
                      VEC_COMPUTED_END
                      inconv0(k,Ares,rp_Ttemp,Aarg);

                      HOV_INC(Ares, k)
                      HOV_INC(Aarg, k)
                      HOS_OV_INC(Targ2, k)
                  }
            }
                break;

                /*--------------------------------------------------------------------------*/
            case pow_op:                                                /* pow_op */
                res   = get_locint_r();
                arg   = get_locint_r();
                coval = get_val_r();

                ASSIGN_T(Targ, rpp_T[arg])
                ASSIGN_T(Tres, rpp_T[res])
                ASSIGN_A(Ares, dpp_A[res])
                ASSIGN_A(Aarg, dpp_A[arg])

                /* olvo 980921 allows reflexive operation */
                if (arg == res) {
                    FOR_0_LE_l_LT_pk
                    rp_Ttemp2[l] = Tres[l];
                    Tres = rp_Ttemp2;
                    GET_TAYL(arg,k,p)
                }

                VEC_COMPUTED_INIT
                FOR_0_LE_l_LT_p
                if (0 == ARES) {
                    HOV_INC(Aarg, k1)
                    HOV_INC(Ares, k1)
                } else {
                    aTmp = ARES;
                    ARES_INC = 0.0;
                    MAXDEC(AARG,aTmp);
                    MAXDEC(AARG,4.0);
                    AARG_INC_O;

                    VEC_COMPUTED_CHECK
                    if (fabs(Targ[0]) > ADOLC_EPS) {
                        divide(k,Tres,Targ,rp_Ttemp);
                        for (i=0;i<k;i++) {
                            rp_Ttemp[i] *= coval;
                            /*                 printf(" EPS i %d %f\n",i,rp_Ttemp[i]); */
                        }
                        inconv0(k,Ares,rp_Ttemp,Aarg);
                    } else {
                        if (coval <= 0.0) {
                            FOR_0_LE_i_LT_k
                            {
                                Aarg[i] = make_nan();
                                Ares[i] = 0;
                            }
                        } else {
                            /* coval not a whole number */
                            if (coval - floor(coval) != 0) {
                                i = 0;
                                FOR_0_LE_i_LT_k
                                {
                                    if (coval - i > 1) {
                                    Aarg[i] = 0;
                                        Ares[i] = 0;
                                    }
                                    if ((coval - i < 1) && (coval - i > 0)) {
                                    Aarg[i] = make_inf();
                                        Ares[i] = 0;
                                    }
                                    if (coval - i < 0) {
                                    Aarg[i] = make_nan();
                                        Ares[i] = 0;
                                    }
                                }
                            } else {
                                if (coval == 1) {
                                    FOR_0_LE_i_LT_k
                                    { /* ! no tempory */
                                        AARG_INC += ARES;
                                        ARES_INC = 0.0;
                                    }
                                } else {
                                    /* coval is an int > 1 */
                                    /* the following is not efficient but at least it works */
                                    /* it reformulates x^n into x* ... *x n times */

                                    copyAndZeroset(k,Ares,dp_Atemp);
                                    inconv(k,dp_Atemp,Targ,Aarg);
                                    inconv(k,dp_Atemp,Targ,Aarg);
                                    if (coval == 3) {
                                        conv(k,Aarg,Targ,dp_Atemp);
                                        FOR_0_LE_i_LT_k
                                        Aarg[i] = 2.0 * dp_Atemp[i];
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

                GET_TAYL(res,k,p)
                break;

                /*--------------------------------------------------------------------------*/
            case sqrt_op:                                              /* sqrt_op */
                res = get_locint_r();
                arg = get_locint_r();

                ASSIGN_A(Ares, dpp_A[res])
                ASSIGN_A(Aarg, dpp_A[arg])
                ASSIGN_T(Tres, rpp_T[res])

                VEC_COMPUTED_INIT
                FOR_0_LE_l_LT_p
                if (0 == ARES) {
                    HOV_INC(Aarg, k1)
                    HOV_INC(Ares, k1)
                } else {
                    aTmp = ARES;
                    ARES_INC = 0.0;
                    MAXDEC(AARG,aTmp);
                    MAXDEC(AARG,4.0);
                    AARG_INC_O;

                    VEC_COMPUTED_CHECK
                    recipr(k,0.5,Tres,rp_Ttemp);
                    VEC_COMPUTED_END
                    inconv0(k,Ares,rp_Ttemp,Aarg);

                    HOV_INC(Ares, k)
                    HOV_INC(Aarg, k)
                    HOS_OV_INC(Tres,k)
                }

                GET_TAYL(res,k,p)
                break;

                /*--------------------------------------------------------------------------*/
            case gen_quad:                                            /* gen_quad */
                res   = get_locint_r();
                arg2  = get_locint_r();
                arg1  = get_locint_r();
                coval = get_val_r();
                coval = get_val_r();

                ASSIGN_A(Ares,  dpp_A[res])
                ASSIGN_A(Aarg1, dpp_A[arg1])
                ASSIGN_T(Targ2, rpp_T[arg2])

                FOR_0_LE_l_LT_p
                if (0 == ARES) {
                    HOV_INC(Aarg1, k1)
                    HOV_INC(Ares,  k1)
                } else {
                    aTmp = ARES;
                    ARES_INC = 0.0;
                    MAXDEC(AARG1,aTmp);
                    MAXDEC(AARG1,4.0);
                    AARG1_INC_O;

                    inconv0(k,Ares,Targ2,Aarg1);

                    HOV_INC(Aarg1, k)
                    HOV_INC(Ares,  k)
                    HOS_OV_INC(Targ2,  k)
                }

                GET_TAYL(res,k,p)
                break;

                /*--------------------------------------------------------------------------*/
            case min_op:                                                /* min_op */

#ifdef _HOS_OV_

                fprintf(DIAG_OUT," operation min_op not implemented for hos_ov");
                break;
#endif
                res   = get_locint_r();
                arg2  = get_locint_r();
                arg1  = get_locint_r();
                coval = get_val_r();

                GET_TAYL(res,k,p)

                ASSIGN_A(Aarg1, dpp_A[arg1])
                ASSIGN_A(Aarg2, dpp_A[arg2])
                ASSIGN_A(Ares,  dpp_A[res])
                ASSIGN_T(Targ1, rpp_T[arg1])
                ASSIGN_T(Targ2, rpp_T[arg2])
                ASSIGN_A(AP1,   NULL)
                ASSIGN_A(AP2,   Ares)

                if (Targ1[0] > Targ2[0]) {
                    FOR_0_LE_l_LT_p
                    { if ((coval) && (*AP2))
                      MINDEC(ret_c,2);
                      HOV_INC(AP2,k1)
                    }
                    AP1 = Aarg2;
                arg = 0;
            } else
                if (Targ1[0] < Targ2[0]) {
                        FOR_0_LE_l_LT_p
                        { if ((!coval) && (*AP2))
                          MINDEC(ret_c,2);
                          HOV_INC(AP2,k1)
                        }
                        AP1 = Aarg1;
                    arg = 0;
                } else /* both are equal */ /* must be changed for hos_ov, but how? */
                    /* seems to influence the return value */
                    for (i=1;i<k;i++) {
                            if (Targ1[i] > Targ2[i]) {
                                FOR_0_LE_l_LT_p
                                { if (*AP2)
                                  MINDEC(ret_c,1);
                                  HOV_INC(AP2,k1)
                                }
                                AP1 = Aarg2;
                            arg = i+1;
                        } else
                            if (Targ1[i] < Targ2[i]) {
                                    FOR_0_LE_l_LT_p
                                    { if (*AP2)
                                      MINDEC(ret_c,1);
                                      HOV_INC(AP2,k1)
                                    }
                                    AP1 = Aarg1;
                                arg = i+1;
                            }
                        if (AP1 != NULL)
                                break;
                        }

                if (AP1 != NULL)
                    FOR_0_LE_l_LT_p
                    { if (0 == ARES) {
                      HOV_INC(AP1, k1)
                          HOV_INC(Ares,k1);
                      } else {
                          aTmp = ARES;
                          ARES_INC = 0.0;
                          if (arg)  /* we are at the tie */
                              *AP1 = 5.0;
                          else
                              MAXDEC(*AP1,aTmp);
                          AP1++;
                          for (i=0;i<k;i++) {
                              aTmp = ARES;
                              ARES_INC = 0.0;
                              *AP1++ += aTmp;
                          }
                      }
            }
                else /* both are identical */
                {
                    FOR_0_LE_l_LT_p
                    { if (0 == ARES) {
                      HOV_INC(Aarg1,k1)
                          HOV_INC(Aarg2,k1)
                          HOV_INC(Ares, k1)
                      } else {
                          aTmp = ARES;
                          ARES_INC = 0.0;
                          MAXDEC(AARG1,aTmp);  /*assume sthg like fmin(x,x) */
                          MAXDEC(AARG2,aTmp);
                          AARG1_INC_O;
                          AARG2_INC_O;
                          for (i=0;i<k;i++) {
                              aTmp = ARES;
                              ARES_INC = 0.0;
                              AARG1_INC += aTmp/2;
                              AARG2_INC += aTmp/2;
                          }
                      }
                    }
                    if (arg1 != arg2)
                        MINDEC(ret_c,1);
                }
                break;


                /*--------------------------------------------------------------------------*/
            case abs_val:                                              /* abs_val */
                res   = get_locint_r();
                arg   = get_locint_r();
                coval = get_val_r();
                /* must be changed for hos_ov, but how? */
                /* seems to influence the return value  */
                GET_TAYL(res,k,p)

                ASSIGN_A(Ares, dpp_A[res])
                ASSIGN_A(Aarg, dpp_A[arg])
                ASSIGN_T(Targ, rpp_T[arg])

                FOR_0_LE_l_LT_q
                {
                    x[l] = 0.0;
                    jj[l] = 0;
                    for (i=0;i<k;i++)
                    if ( (x[l] == 0.0) && (Targ[i] != 0.0) ) {
                        jj[l] = i;
                            if (Targ[i] < 0.0)
                                x[l] = -1.0;
                            else
                                x[l] = 1.0;
                        }
                    HOS_OV_INC(Targ,k)
            }
                ASSIGN_T(Targ, rpp_T[arg])
                FOR_0_LE_l_LT_p
                { if (0 == ARES) {
                  HOV_INC(Aarg, k1)
                      HOV_INC(Ares, k1)
                  } else {
                      if (Targ[0] == 0.0) {
                          ARES_INC = 0.0;
                          AARG_INC = 5.0;
                      } else {
                          aTmp = ARES;
                          ARES_INC = 0.0;
                          MAXDEC(AARG,aTmp);
                          AARG_INC_O;
                      }
                      if(Targ[0] == 0.0)
                          MINDEC(ret_c,1);
                      for (i=0;i<jj[l];i++)
                          ARES_INC = 0.0;
                      Aarg += jj[l];
                      for (i=jj[l];i<k;i++) {
                          aTmp = ARES;
                          ARES_INC = 0.0;
                          if ( (coval) && (x[l]<0) && (aTmp) )
                              MINDEC(ret_c,2);
                          if ( (!coval) && (x[l]>0) && (aTmp))
                              MINDEC(ret_c,2);
                          AARG_INC += x[l] * aTmp;
                      }
                  }
                  HOS_OV_INC(Targ,k)
            }
                break;

                /*--------------------------------------------------------------------------*/
            case ceil_op:                                              /* ceil_op */
                res   = get_locint_r();
                arg   = get_locint_r();
                coval = get_val_r();

                GET_TAYL(res,k,p)

                coval = (coval != ceil(*rpp_T[arg]) );

                ASSIGN_A(Ares, dpp_A[res])

                FOR_0_LE_l_LT_p
                if (0 == ARES) {
                    HOV_INC(Aarg,  k1)
                    HOV_INC(Ares,  k1)
                } else {
                    ARES_INC = 0.0;
                    AARG_INC = 5.0;
                    FOR_0_LE_i_LT_k
                    { if ((coval) && (ARES))
                      MINDEC(ret_c,2);
                      ARES_INC = 0.0;
                    }
                    HOV_INC(Aarg, k)
                    }
                break;

                /*--------------------------------------------------------------------------*/
            case floor_op:                                            /* floor_op */
                res   = get_locint_r();
                arg   = get_locint_r();
                coval = get_val_r();

                GET_TAYL(res,k,p)

                coval = ( coval != floor(*rpp_T[arg]) );

                ASSIGN_A(Ares, dpp_A[res])
                ASSIGN_A(Aarg, dpp_A[arg])

                FOR_0_LE_l_LT_p
                if (0 == ARES) {
                    HOV_INC(Aarg, k1)
                    HOV_INC(Ares, k1)
                } else {
                    ARES = 0.0;
                    AARG_INC = 5.0;
                    FOR_0_LE_i_LT_k
                    { if ( (coval) && (ARES) )
                      MINDEC(ret_c,2);
                      ARES_INC = 0.0;
                    }
                    HOV_INC(Aarg, k)
                    }
                break;


                /****************************************************************************/
                /*                                                             CONDITIONALS */

                /*--------------------------------------------------------------------------*/
            case cond_assign:                                      /* cond_assign */
                res   = get_locint_r();
                arg2  = get_locint_r();
                arg1  = get_locint_r();
                arg   = get_locint_r();
                coval = get_val_r();

                GET_TAYL(res,k,p)

                ASSIGN_A(Aarg1, dpp_A[arg1])
                ASSIGN_A(Ares,  dpp_A[res])
                ASSIGN_A(Aarg2, dpp_A[arg2])
                ASSIGN_T(Targ,  rpp_T[arg])

                /* olvo 980925 changed code a little bit */
                if (TARG > 0.0) {
                    if (res != arg1)
                        FOR_0_LE_l_LT_p
                        { if (0 == ARES) {
                          HOV_INC(Ares,  k1)
                              HOV_INC(Aarg1, k1)
                          } else {
                              if (coval <= 0.0)
                                  MINDEC(ret_c,2);
                              MAXDEC(AARG1,ARES);
                              ARES_INC = 0.0;
                              AARG1_INC_O;
                              FOR_0_LE_i_LT_k
                              { AARG1_INC += ARES;
                                ARES_INC = 0;
                              }
                          }
                    }
                    else
                        FOR_0_LE_l_LT_p {
                            if ((coval <= 0.0) && (ARES))
                            MINDEC(ret_c,2);
                            HOV_INC(Ares,  k1)
                        }
                    } else /* TARG <= 0.0 */
            {
                if (res != arg2)
                        FOR_0_LE_l_LT_p
                        { if (0 == ARES) {
                          HOV_INC(Ares,  k1)
                              HOV_INC(Aarg2, k1)
                          } else {
                              if (TARG == 0.0) /* we are at the tie */
                              { MINDEC(ret_c,0);
                                  AARG1 = 5.0;
                                  AARG2_INC = 5.0;
                              } else {
                                  if (coval <= 0.0)
                                      MINDEC(ret_c,2);
                                  MAXDEC(AARG2,ARES);
                                  AARG2_INC_O;
                              }
                              ARES_INC = 0.0;

                              FOR_0_LE_i_LT_k
                              { AARG2_INC += ARES;
                                ARES_INC = 0;
                              }
                          }
                      HOV_INC(Aarg1, k1)
                    } else
                        FOR_0_LE_l_LT_p {
                            if (ARES) {
                            if (TARG == 0.0) /* we are at the tie */
                                { MINDEC(ret_c,0);
                                    AARG1 = 5.0;
                                    AARG2 = 5.0;
                                } else
                                    if (coval <= 0.0)
                                        MINDEC(ret_c,2);
                            }
                        HOV_INC(Ares,  k1)
                        HOV_INC(Aarg1, k1)
                        HOV_INC(Aarg2, k1)
                    }
                }
                break;

                /*--------------------------------------------------------------------------*/
            case cond_assign_s:                                  /* cond_assign_s */
                res   = get_locint_r();
                arg1  = get_locint_r();
                arg   = get_locint_r();
                coval = get_val_r();

                GET_TAYL(res,k,p)

                ASSIGN_A(Aarg1, dpp_A[arg1])
                ASSIGN_A(Ares,  dpp_A[res])
                ASSIGN_T(Targ,  rpp_T[arg])

                /* olvo 980925 changed code a little bit */
                if (TARG == 0.0) /* we are at the tie */
                { FOR_0_LE_l_LT_p
                    { if  (ARES)
                      AARG1 = 5.0;
                      HOV_INC(Aarg1, k1)
                      HOV_INC(Ares,  k1)
                    }
                    MINDEC(ret_c,0);
                } else
                    if (TARG > 0.0) {
                        if (res != arg1)
                            FOR_0_LE_l_LT_p
                            { if  (0 == ARES) {
                              HOV_INC(Ares,  k1)
                                  HOV_INC(Aarg1, k1)
                              } else {
                                  if (coval <= 0.0)
                                      MINDEC(ret_c,2);
                                  MAXDEC(AARG1,ARES);
                                  ARES_INC = 0.0;
                                  AARG1_INC_O;
                                  FOR_0_LE_i_LT_k
                                  { (AARG1_INC) += ARES;
                                    ARES_INC = 0;
                                  }
                              }
                        }
                        else
                            FOR_0_LE_l_LT_p {
                                if ((coval <= 0.0) && (ARES))
                                MINDEC(ret_c,2);
                                HOV_INC(Ares,  k1)
                            }
                        }
            break;

                /****************************************************************************/
                /*                                                          REMAINING STUFF */

                /*--------------------------------------------------------------------------*/
            case take_stock_op:                                  /* take_stock_op */
                res = get_locint_r();
                size = get_locint_r();
                d = get_val_v_r(size);

                res += size;
                for (ls=size;ls>0;ls--) {
                    res--;

                    ASSIGN_A( Ares, dpp_A[res])

                    FOR_0_LE_l_LT_pk1
                    ARES_INC = 0.0;
                }
                break;

                /*--------------------------------------------------------------------------*/
            case death_not:                                          /* death_not */
                arg2 = get_locint_r();
                arg1 = get_locint_r();

                for (j=arg1;j<=arg2;j++) {
                    ASSIGN_A(Aarg1, dpp_A[j])

                    FOR_0_LE_l_LT_p
                    for (i=0; i<k1; i++)
                        AARG1_INC = 0.0;
                }
                
                for (j=arg1;j<=arg2;j++)
                    GET_TAYL(j,k,p)

                break;

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
#if defined(ADOLC_DEBUG)
        ++countPerOperation[operation];
#endif /* ADOLC_DEBUG */
    }

#if defined(ADOLC_DEBUG)
    printf("\nTape contains:\n");
    for (v = 0; v < 256; ++v)
        if (countPerOperation[v] > 0)
            printf("operation %3d: %6d time(s) - %6d taylors read (%10.2f per operation)\n", v, countPerOperation[v], taylorPerOperation[v], (double)taylorPerOperation[v] / (double)countPerOperation[v]);
    printf("\n");
#endif /* ADOLC_DEBUG */

    /* clean up */
    free((char*)*rpp_T);
    free((char*) rpp_T);
    myfree2(dpp_A);
    free(rp_Ttemp);
    free(rp_Ttemp2);
    free(dp_Atemp);
    free(dp_Atemp2);

    free((char*) jj);
    free((char*) x);

    end_sweep();

    return ret_c;
}

/****************************************************************************/
/*                                                               THAT'S ALL */

END_C_DECLS

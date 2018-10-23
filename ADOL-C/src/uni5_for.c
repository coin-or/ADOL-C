
/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     uni5_for.c


 Revision: $Id$

 Contents: Contains the routines :
           zos_forward (zero-order-scalar forward mode):      define _ZOS_
           fos_forward (first-order-scalar forward mode):     define _FOS_
           hos_forward (higher-order-scalar forward mode):    define _HOS_
           fov_forward (first-order-vector forward mode):     define _FOV_
           hov_forward (higher-order-vector forward mode):    define _HOV_
           hov_wk_forward (higher-order-vector forward mode): define _HOV_WK_
           int_forward_safe:                                  define _INT_FOR_ and _NTIGHT__

           Uses the preprocessor to compile the 7 different object files
           with/without "keep" parameter:                     define _KEEP_

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz,
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel,
               Benjamin Letschert, Kshitij Kulshreshtha
               

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#include <adolc/interfaces.h>
#include <adolc/adalloc.h>
#include <adolc/taping.h>
#include "taping_p.h"
#include "oplate.h"
#include <adolc/externfcts.h>
#include "externfcts_p.h"
#include "dvlparms.h"

#include <math.h>
#include <string.h>

#if defined(ADOLC_DEBUG) || defined(_ZOS_)
#include <string.h>
#endif /* ADOLC_DEBUG */

#ifdef ADOLC_MEDIPACK_SUPPORT
#include "medipacksupport_p.h"
#endif
#ifdef ADOLC_AMPI_SUPPORT
#include "ampisupportAdolc.h"
#endif

/****************************************************************************/
/*                                                                   MACROS */
#undef _ADOLC_VECTOR_
#undef _HIGHER_ORDER_



/*--------------------------------------------------------------------------*/
#if defined(_ZOS_)
#if defined(_ABS_NORM_)
#  define GENERATED_FILENAME "zos_pl_forward"
#elif defined(_ABS_NORM_SIG_)
#  define GENERATED_FILENAME "zos_pl_sig_forward"
#else
#  define GENERATED_FILENAME "zos_forward"
#endif
/*--------------------------------------------------------------------------*/
#else
#if defined(_FOS_)
#if defined(_ABS_NORM_)
#define GENERATED_FILENAME "fos_pl_forward"
#elif defined(_ABS_NORM_SIG_)
#define GENERATED_FILENAME "fos_pl_sig_forward"
#else
#define GENERATED_FILENAME "fos_forward"
#endif

#define ARGUMENT(indexi,l,i) argument[indexi]
#define TAYLORS(indexd,l,i)   taylors[indexd]

/*--------------------------------------------------------------------------*/
#else
#if defined(_FOV_)
#if defined(_ABS_NORM_)
#define GENERATED_FILENAME "fov_pl_forward"
#elif defined(_ABS_NORM_SIG_)
#define GENERATED_FILENAME "fov_pl_sig_forward"
#if defined(_MSC_VER) && _MSC_VER < 180
#define fmin __min
#define fmax __max
#endif
#else
#define GENERATED_FILENAME "fov_forward"
#endif

#define _ADOLC_VECTOR_

#if defined(_CHUNKED_)
#define ARGUMENT(indexi,l,i) argument[indexi][l+offset]
#define TAYLORS(indexd,l,i)   taylors[indexd][l+offset]
#else
#define ARGUMENT(indexi,l,i) argument[indexi][l]
#define TAYLORS(indexd,l,i)   taylors[indexd][l]
#endif

/*--------------------------------------------------------------------------*/
#else
#if defined(_HOS_)
#define GENERATED_FILENAME "hos_forward"

#define _HIGHER_ORDER_

#define ARGUMENT(indexi,l,i) argument[indexi][i]
#define TAYLORS(indexd,l,i)   taylors[indexd][i]

/*--------------------------------------------------------------------------*/
#else
#if defined(_HOV_)
#define GENERATED_FILENAME "hov_forward"

#define _ADOLC_VECTOR_
#define _HIGHER_ORDER_

#define ARGUMENT(indexi,l,i) argument[indexi][l][i]
#define TAYLORS(indexd,l,i)   taylors[indexd][l][i]

/*--------------------------------------------------------------------------*/
#else
#if defined(_HOV_WK_)
#define GENERATED_FILENAME "hov_wk_forward"

#define _ADOLC_VECTOR_
#define _HIGHER_ORDER_

#define ARGUMENT(indexi,l,i) argument[indexi][l][i]
#define TAYLORS(indexd,l,i)   taylors[indexd][l][i]

/*--------------------------------------------------------------------------*/
#else
#if defined(_INT_FOR_)
#if defined(_TIGHT_)
#define GENERATED_FILENAME "int_forward_t"
#endif
#if defined(_NTIGHT_)
#define GENERATED_FILENAME "int_forward_s"
#endif
#define ARGUMENT(indexi,l,i) argument[indexi][l]
#define TAYLORS(indexd,l,i)   taylors[indexd][l]
/*--------------------------------------------------------------------------*/
#else
#if defined(_INDO_)
void copy_index_domain(int res, int arg, locint **ind_dom);
void merge_2_index_domains(int res, int arg, locint **ind_dom);
void combine_2_index_domains(int res, int arg1, int arg2, locint **ind_dom);
void merge_3_index_domains(int res, int arg1, int arg2, locint **ind_dom);

#define NUMNNZ 20
#define FMIN_ADOLC(x,y)  ((y<x)?y:x)

#if defined(_INDOPRO_) && !defined(_NONLIND_OLD_)
#if defined(_TIGHT_)
#define GENERATED_FILENAME "indopro_forward_t"
#endif
#if defined(_NTIGHT_)
#if defined(_ABS_NORM_)
#define GENERATED_FILENAME "indopro_forward_pl"
#else
#define GENERATED_FILENAME "indopro_forward_s"
#endif
#endif
#endif
#if defined(_NONLIND_)

typedef struct IndexElement {
    locint  entry;
    struct IndexElement* left;
    struct IndexElement* right;
}
IndexElement;

typedef struct IndexElement_sod {
    locint  entry;
    struct IndexElement_sod* left;
}
IndexElement_sod;

void free_tree(IndexElement* nlf, int num);
void traverse_crs(IndexElement* fod2, IndexElement_sod* sod, int num);
void traverse_unary(IndexElement* fod, IndexElement* nonl_dom,  IndexElement* fodi, int num, int maxopind);

#if defined(_TIGHT_)
#define GENERATED_FILENAME "nonl_ind_forward_t"
#elif defined(_NTIGHT_)
#define GENERATED_FILENAME "nonl_ind_forward_s"
#endif
#endif
#if defined(_NONLIND_OLD_)

/*
 * This is the type used for the list elements. The entry is either a counter
 * (first element of the NID list) or the index of an independent variable.
 */

void extend_nonlinearity_domain_binary_step
(int arg1, int arg2, locint **ind_dom, locint **nonl_dom);
void extend_nonlinearity_domain_unary
(int arg, locint **ind_dom, locint **nonl_dom);
void extend_nonlinearity_domain_binary
(int arg1, int arg2, locint **ind_dom, locint **nonl_dom);


#if defined(_TIGHT_)
#define GENERATED_FILENAME "nonl_ind_old_forward_t"
#endif
#if defined(_NTIGHT_)
#define GENERATED_FILENAME "nonl_ind_old_forward_s"
#endif
#endif


/*--------------------------------------------------------------------------*/
#else
#error Error ! Define [_ZOS_ | _FOS_ |\
   _HOS_ | _FOV_ | _HOV_ | _HOV_WK_  | _INT_FOR_SAFE_ | _INT_FOR_TIGHT_ | _INDOPRO_ | _NONLIND_ ] [{_KEEP_}]
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif

/*--------------------------------------------------------------------------*/
/*                                                               KEEP stuff */
#if defined(_KEEP_)

#if defined(_HOV_WK_) /* keep in this vector mode */
#define IF_KEEP_TAYLOR_CLOSE \
if (keep){\
  fprintf(DIAG_OUT,"Succeeding reverse sweep will fail!\n");\
  taylor_close(0);\
}
#define IF_KEEP_WRITE_TAYLOR(res,keep,k,p) \
    { \
        UPDATE_TAYLORWRITTEN(keep * k * p) \
        if (keep) \
        { \
            ADOLC_WRITE_SCAYLOR(dp_T0[res]); \
            if (keep > 1) \
            write_taylors(res,(keep-1),k,p); \
        } \
    }
#else
#if defined(_ADOLC_VECTOR_) /* otherwise no keep */
#define IF_KEEP_TAYLOR_CLOSE
#define IF_KEEP_WRITE_TAYLOR(res,keep,k,p)
#else /* _ZOS_, _FOS_, _HOS_ */
#define IF_KEEP_TAYLOR_CLOSE \
if (keep){\
  fprintf(DIAG_OUT,"Otherwise succeeding reverse sweep will fail!\n");\
  taylor_close(0);\
}
#if defined(_ZOS_)
#define IF_KEEP_WRITE_TAYLOR(res,keep,k,p) \
    { \
        UPDATE_TAYLORWRITTEN(keep) \
        if (keep) \
            ADOLC_WRITE_SCAYLOR(dp_T0[res]); \
    }
#else
#if defined(_FOS_)
#define IF_KEEP_WRITE_TAYLOR(res,keep,k,p) \
    { \
        UPDATE_TAYLORWRITTEN(keep) \
        if (keep) \
        { \
            ADOLC_WRITE_SCAYLOR(dp_T0[res]); \
            if (keep > 1) \
                ADOLC_WRITE_SCAYLOR(dp_T[res]); \
        } \
    }
#else
#if defined(_HOS_)
#define IF_KEEP_WRITE_TAYLOR(res,keep,k,p) \
    { \
        UPDATE_TAYLORWRITTEN(keep) \
        if (keep) \
        { \
            ADOLC_WRITE_SCAYLOR(dp_T0[res]); \
            if (keep > 1) \
                write_taylor(res,keep-1); \
        } \
    }
#endif
#endif
#endif
#endif
#endif

#else  /* no _KEEP_ */
#define IF_KEEP_TAYLOR_CLOSE
#define IF_KEEP_WRITE_TAYLOR(res,keep,k,p)
#endif

/*--------------------------------------------------------------------------*/
/*                                                      access to variables */
#if !defined(_ZOS_)
#if defined(_FOS_)
#define TRES         *Tres
#define TARG         *Targ
#define TARG1        *Targ1
#define TARG2        *Targ2
#define TQO          *Tqo

#define TRES_INC     *Tres
#define TARG_INC     *Targ
#define TARG1_INC    *Targ1
#define TARG2_INC    *Targ2
#define TQO_INC      *Tqo

#define TRES_DEC     *Tres
#define TARG_DEC     *Targ
#define TARG1_DEC    *Targ1
#define TARG2_DEC    *Targ2
#define TQO_DEC      *Tqo

#define TRES_FOINC   *Tres
#define TARG_FOINC   *Targ
#define TARG1_FOINC  *Targ1
#define TARG2_FOINC  *Targ2
#define TQO_FOINC    *Tqo

#define TRES_FODEC   *Tres
#define DEC_TRES_FO
#define TARG_FODEC   *Targ
#define TARG1_FODEC  *Targ1
#define TARG2_FODEC  *Targ2
#define TQO_FODEC    *Tqo

#define ASSIGN_T(a,b)  a = &b;

#else
#if defined(_INT_FOR_)
#define TRES         *Tres
#define TARG         *Targ
#define TARG1        *Targ1
#define TARG2        *Targ2
#define TQO          *Tqo

#define TRES_INC     *Tres++
#define TARG_INC     *Targ++
#define TARG1_INC    *Targ1++
#define TARG2_INC    *Targ2++
#define TQO_INC      *Tqo++

#define TRES_DEC     *Tres--
#define TARG_DEC     *Targ--
#define TARG1_DEC    *Targ1--
#define TARG2_DEC    *Targ2--
#define TQO_DEC      *Tqo--

#define TRES_FOINC   *Tres++
#define TARG_FOINC   *Targ++
#define TARG1_FOINC  *Targ1++
#define TARG2_FOINC  *Targ2++
#define TQO_FOINC    *Tqo++

#define TRES_FODEC   *Tres--
#define TARG_FODEC   *Targ--
#define TARG1_FODEC  *Targ1--
#define TARG2_FODEC  *Targ2--
#define TQO_FODEC    *Tqo--


#define ASSIGN_T(a,b)  a = b;

#else  /* _HOS_, _FOV_, _HOV_, _HOV_WK */
#define TRES         *Tres
#define TARG         *Targ
#define TARG1        *Targ1
#define TARG2        *Targ2
#define TQO          *Tqo

#define TRES_INC     *Tres++
#define TARG_INC     *Targ++
#define TARG1_INC    *Targ1++
#define TARG2_INC    *Targ2++
#define TQO_INC      *Tqo++

#define TRES_DEC     *Tres--
#define TARG_DEC     *Targ--
#define TARG1_DEC    *Targ1--
#define TARG2_DEC    *Targ2--
#define TQO_DEC      *Tqo--

#if defined(_FOV_)
#define TRES_FOINC   *Tres++
#define TARG_FOINC   *Targ++
#define TARG1_FOINC  *Targ1++
#define TARG2_FOINC  *Targ2++
#define TQO_FOINC    *Tqo++

#define TRES_FODEC   *Tres
#define DEC_TRES_FO  Tres--;
#define TARG_FODEC   *Targ--
#define TARG1_FODEC  *Targ1--
#define TARG2_FODEC  *Targ2--
#define TQO_FODEC    *Tqo--
#else /* _HOS_, _HOV_, _HOV_WK */
#define TRES_FOINC   *Tres
#define TARG_FOINC   *Targ
#define TARG1_FOINC  *Targ1
#define TARG2_FOINC  *Targ2
#define TQO_FOINC    *Tqo

#define TRES_FODEC   *Tres
#define DEC_TRES_FO
#define TARG_FODEC   *Targ
#define TARG1_FODEC  *Targ1
#define TARG2_FODEC  *Targ2
#define TQO_FODEC    *Tqo
#endif
#endif

#define ASSIGN_T(a,b)  a = b;
#endif
#endif


/*--------------------------------------------------------------------------*/
/*                                                               loop stuff */
#if defined(_ADOLC_VECTOR_)
#define FOR_0_LE_l_LT_p for (l=0; l<p; l++)
#define FOR_p_GT_l_GE_0 for (l=p-1; l>=0; l--)
#if defined(_ABS_NORM_) || defined(_ABS_NORM_SIG_)
#define FIRSTSIGN_P(x,y) firstsign(p,&(x),y)
#define COPYTAYL_P(x,y)  FOR_0_LE_l_LT_p x[l] = y[l]
#define EXT_FIRSTSIGN_P(sigx,sigd,x,y) ext_firstsign(sigx,sigd,p,&(x),y)
#define EXT_FIRSTSIGN2_P(sigx,x,y) ext_firstsign2(sigx,p,&(x),y)
#endif
#else
#if defined(_INT_FOR_)
#define FOR_0_LE_l_LT_p for (l=0; l<p; l++)
#define FOR_p_GT_l_GE_0 for (l=p-1; l>=0; l--)
#else
#define FOR_0_LE_l_LT_p
#define FOR_p_GT_l_GE_0
#if defined(_ABS_NORM_) || defined(_ABS_NORM_SIG_)
#define FIRSTSIGN_P(x,y) firstsign(1,&(x),y)
#define COPYTAYL_P(x,y)  x = *y
#define EXT_FIRSTSIGN_P(sigx,sigd,x,y) ext_firstsign(sigx,sigd,1,&(x),y)
#define EXT_FIRSTSIGN2_P(sigx,x,y) ext_firstsign2(sigx,1,&(x),y)
#endif
#endif
#endif

#if defined(_HIGHER_ORDER_)
#define FOR_0_LE_i_LT_k for (i=0; i<k; i++)
#define FOR_k_GT_i_GE_0 for (i=k-1; i>=0; i--)
#else
#define FOR_0_LE_i_LT_k
#define FOR_k_GT_i_GE_0
#endif

#if defined(_HOV_)
#define FOR_0_LE_l_LT_pk for (l=0; l<pk; l++)
#define INC_pk_1(T)      T += pk-1;
#define VEC_INC(T,inc)   T += inc;
#define HOV_INC(T,inc)   T += inc;
#else
#if defined(_HOV_WK_)
#define FOR_0_LE_l_LT_pk for (l=0; l<pk; l++)
#define INC_pk_1(T)      T += pk-1;
#define VEC_INC(T,inc)   T += inc;
#define HOV_INC(T,inc)   T += inc;
#else
#if defined(_FOV_)
#define FOR_0_LE_l_LT_pk for (l=0; l<p; l++)
#define INC_pk_1(T)      T += p-1;
#define VEC_INC(T,inc)   T++;
#define HOV_INC(T,inc)
#else
#if defined(_HOS_)
#define FOR_0_LE_l_LT_pk for (l=0; l<k; l++)
#define INC_pk_1(T)      T += k-1;
#define VEC_INC(T,inc)
#define HOV_INC(T,inc)
#else
#if defined(_INT_FOR_)
#define FOR_0_LE_l_LT_pk for (l=0; l<p; l++)
#define INC_pk_1(T)      T += p-1;
#define VEC_INC(T,inc)   T++;
#else
#define FOR_0_LE_l_LT_pk
#define INC_pk_1(T)
#define VEC_INC(T,inc)
#define HOV_INC(T,inc)
#endif
#endif
#endif
#endif
#endif

/*--------------------------------------------------------------------------*/
/*                                                        higher order case */
#if defined(_HIGHER_ORDER_)
#define BREAK_FOR_I break;
#else
#define BREAK_FOR_I ;
#endif

/* END Macros */

BEGIN_C_DECLS

#if defined(_ZOS_)
/****************************************************************************/
/* Zero Order Scalar version of the forward mode.                           */
/****************************************************************************/
#if defined(_ABS_NORM_)
int zos_pl_forward(short tnum,
                  int depcheck,
		  int indcheck,
		  int keep,
		  const double *basepoint,
		  double *valuepoint,
		  double *swargs)
#else
#if defined(_KEEP_)
int  zos_forward(
#else
int  zos_forward_nk(
#endif
    short  tnum,              /* tape id */
    int    depcheck,          /* consistency chk on # of deps */
    int    indcheck,          /* consistency chk on # of indeps */
#if defined(_KEEP_)
    int    keep,              /* flag for reverse sweep */
#endif
    const double *basepoint,  /* independant variable values */
    double       *valuepoint) /* dependent variable values */

#endif

#else
#if defined(_FOS_)
/****************************************************************************/
/* First Order Scalar version of the forward mode.                          */
/****************************************************************************/
#if defined(_ABS_NORM_)
int  fos_pl_forward(short tnum,
		    int depcheck,
		    int indcheck,
		    const double* basepoint,
		    double *argument,
		    double *valuepoint,
		    double *taylors,
		    double *swargs,
		    double *swtaylors)
#elif defined(_ABS_NORM_SIG_)
int  fos_pl_sig_forward(short tnum,
                        int depcheck,
                        int indcheck,
                        const double* basepoint,
                        double *argument,
                        int swcheck,
                        short *sigbase,
                        short *sigdir,
                        double *valuepoint,
                        double *taylors,
                        double *swargs,
                        double *swtaylors,
                        short *sigsw)
#else
#if defined(_KEEP_)
int  fos_forward(
#else
int  fos_forward_nk(
#endif
    short  tnum,        /* tape id */
    int    depcheck,    /* consistency chk on # of deps */
    int    indcheck,    /* consistency chk on # of indeps */
#if defined(_KEEP_)
    int    keep,        /* flag for reverse sweep */
#endif
    const double *basepoint,  /* independent variable values */
    double *argument,   /* Taylor coefficients (input) */
    double *valuepoint, /* Taylor coefficients (output) */
    double *taylors)    /* matrix of coefficient vectors */
/* the order of the indices in argument and taylors is [var][taylor] */
#endif

#else
#if defined(_INT_FOR_)
#if defined(_TIGHT_)
/****************************************************************************/
/* First Order Vector version of the forward mode for bit patterns, tight   */
/****************************************************************************/
int int_forward_tight(
    short               tnum,     /* tape id                              */
    int                 depcheck, /* consistency chk on # of dependents   */
    int                 indcheck, /* consistency chk on # of independents */
    int                 p,        /* # of taylor series, bit pattern      */
    const double       *basepoint,  /* independent variable values   (in)*/
    unsigned long int **argument,  /* Taylor coeff.                 (in)*/
    double             *valuepoint, /* dependent variable values    (out)*/
    unsigned long int **taylors)   /* matrix of coefficient vectors(out)*/

/* int_forward_tight( tag, m, n, p, x[n], X[n][p], y[m], Y[m][p]),

     nBV = number of Boolean Vectors to be packed
                      (see Chapter Dependence Analysis, ADOL-C Documentation)
     bits_per_long = 8*sizeof(unsigned long int)
     p = nBV / bits_per_long + ( (nBV % bits_per_long) != 0 )

     The order of the indices in argument and taylors is [var][taylor]

     For the full Jacobian matrix set
     p = indep / bits_per_long + ((indep % bits_per_long) != 0)
     and pass a bit pattern version of the identity matrix as an argument   */


#endif
#if defined (_NTIGHT_)
/****************************************************************************/
/* First Order Vector version of the forward mode, bit pattern, safe        */
/****************************************************************************/
int int_forward_safe(
    short             tnum,     /* tape id                              */
    int               depcheck, /* consistency chk on # of dependents   */
    int               indcheck, /* consistency chk on # of independents */
    int               p,        /* # of taylor series, bit pattern      */
    unsigned long int **argument, /* Taylor coeff.                  (in)*/
    unsigned long int **taylors)  /* matrix of coefficient vectors (out)*/

/* int_forward_safe( tag, m, n, p, X[n][p], Y[m][p]),

nBV = number of Boolean Vectors to be packed
(see Chapter Dependence Analysis, ADOL-C Documentation)
bits_per_long = 8*sizeof(unsigned long int)
p = nBV / bits_per_long + ( (nBV % bits_per_long) != 0 )

The order of the indices in argument and taylors is [var][taylor]

For the full Jacobian matrix set
p = indep / bits_per_long + ((indep % bits_per_long) != 0)
and pass a bit pattern version of the identity matrix as an argument    */
#endif
#else
#if defined(_INDOPRO_) && !defined(_NONLIND_OLD_)
#if defined(_TIGHT_)
/****************************************************************************/
/* First Order Vector version of the forward mode for bit patterns, tight   */
/****************************************************************************/
int indopro_forward_tight(
    short             tnum,        /* tape id                              */
    int               depcheck,    /* consistency chk on # of dependents   */
    int               indcheck,    /* consistency chk on # of independents */
    const double     *basepoint,  /* independent variable values   (in)   */
    unsigned int    **crs)        /* returned row index storage (out)     */

/* indopro_forward_tight( tag, m, n, x[n], *crs[m]),

  */
#endif
#if defined (_NTIGHT_)
#if defined(_ABS_NORM_)
int indopro_forward_absnormal(
    short             tnum,        /* tape id                              */
    int               depcheck,    /* consistency chk on # of dependents   */
    int               indcheck,    /* consistency chk on # of independents */
    int               swcheck,     /* consistency chk on # of switches    */
    const double     *basepoint,  /* independent variable values   (in)   */
    unsigned int    **crs)        /* returned row index storage (out)     */

/* indopro_forward_absnormal( tag, m, n, s, x[n], *crs[s+m]),
  */
#else
/****************************************************************************/
/* First Order Vector version of the forward mode, bit pattern, safe        */
/****************************************************************************/
int indopro_forward_safe(
    short             tnum,        /* tape id                              */
    int               depcheck,    /* consistency chk on # of dependents   */
    int               indcheck,    /* consistency chk on # of independents */
    const double     *basepoint,   /* independent variable values   (in)   */
    unsigned int    **crs)         /* returned row index storage (out)     */

/* indopro_forward_safe( tag, m, n, x[n], *crs[m]),

  */
#endif
#endif
#else
#if defined(_NONLIND_)
#if defined(_TIGHT_)
/****************************************************************************/
/* First Order Vector version of the forward mode for bit patterns, tight   */
/****************************************************************************/
int nonl_ind_forward_tight(
    short             tnum,        /* tape id                              */
    int               depcheck,    /* consistency chk on # of dependents   */
    int               indcheck,    /* consistency chk on # of independents */
    const double     *basepoint,  /* independent variable values   (in)   */
    unsigned int     **crs)        /* returned row index storage (out)     */

#endif
#if defined (_NTIGHT_)
/****************************************************************************/
/* First Order Vector version of the forward mode, bit pattern, safe        */
/****************************************************************************/
int nonl_ind_forward_safe(
    short             tnum,        /* tape id                              */
    int               depcheck,    /* consistency chk on # of dependents   */
    int               indcheck,    /* consistency chk on # of independents */
    const double      *basepoint,  /* independent variable values   (in)   */
    unsigned int    **crs)        /* returned row index storage (out)     */

/* indopro_forward_safe( tag, m, n, x[n], *crs[m]),

  */
#endif
#else
#if defined(_NONLIND_OLD_)
#if defined(_TIGHT_)
/****************************************************************************/
/* First Order Vector version of the forward mode for bit patterns, tight   */
/****************************************************************************/
int nonl_ind_old_forward_tight(
    short             tnum,        /* tape id                              */
    int               depcheck,    /* consistency chk on # of dependents   */
    int               indcheck,    /* consistency chk on # of independents */
    const double     *basepoint,  /* independent variable values   (in)   */
    unsigned int     **crs)        /* returned row index storage (out)     */

#endif
#if defined (_NTIGHT_)
/****************************************************************************/
/* First Order Vector version of the forward mode, bit pattern, safe        */
/****************************************************************************/
int nonl_ind_old_forward_safe(
    short             tnum,        /* tape id                              */
    int               depcheck,    /* consistency chk on # of dependents   */
    int               indcheck,    /* consistency chk on # of independents */
    const double      *basepoint,  /* independent variable values   (in)   */
    unsigned int    **crs)        /* returned row index storage (out)     */

/* indopro_forward_safe( tag, m, n, x[n], *crs[m]),

  */
#endif
#else
#if defined(_FOV_)
#if defined(_CHUNKED_)
/****************************************************************************/
/* First Order Vector version of the forward mode with p-offset in          */
/* **argument and **taylors                                                 */
/****************************************************************************/
int  fov_offset_forward(
    short  tnum,        /* tape id */
    int    depcheck,    /* consistency chk on # of deps */
    int    indcheck,    /* consistency chk on # of indeps */
    int    p,           /* # of taylor series */
    int    offset,      /* offset for assignments */
    const double *basepoint,  /* independent variable values */
    double **argument,  /* Taylor coefficients (input) */
    double *valuepoint, /* Taylor coefficients (output) */
    double **taylors)   /* matrix of coifficient vectors */
/* the order of the indices in argument and taylors is [var][taylor] */
#else
/****************************************************************************/
/* First Order Vector version of the forward mode.                          */
/****************************************************************************/
#if defined(_ABS_NORM_)
int  fov_pl_forward(
    short         tnum,        /* tape id */
    int           depcheck,    /* consistency chk on # of deps */
    int           indcheck,    /* consistency chk on # of indeps */
    int           p,           /* # of taylor series */
    const double *basepoint,   /* independent variable values */
    double      **argument,    /* Taylor coefficients (input) */
    double       *valuepoint,  /* Taylor coefficients (output) */
    double      **taylors,     /* matrix of coifficient vectors */
    double       *swargs,
    double      **swtaylors,
    short        *sigsw)
/* the order of the indices in argument and taylors is [var][taylor] */
#elif defined(_ABS_NORM_SIG_)
int  fov_pl_sig_forward(
    short         tnum,        /* tape id */
    int           depcheck,    /* consistency chk on # of deps */
    int           indcheck,    /* consistency chk on # of indeps */
    int           p,           /* # of taylor series */
    const double *basepoint,   /* independent variable values */
    double      **argument,    /* Taylor coefficients (input) */
    int           swcheck,     /* consistency chk in # of switches */
    short        *sigbase,     /* Signature of basepoint */
    short        *sigdir,      /* Signature of direction */
    double       *valuepoint,  /* Taylor coefficients (output) */
    double      **taylors,     /* matrix of coifficient vectors */
    double       *swargs,
    double      **swtaylors,
    short        *sigsw)
#else
int  fov_forward(
    short         tnum,        /* tape id */
    int           depcheck,    /* consistency chk on # of deps */
    int           indcheck,    /* consistency chk on # of indeps */
    int           p,           /* # of taylor series */
    const double *basepoint,   /* independent variable values */
    double      **argument,    /* Taylor coefficients (input) */
    double       *valuepoint,  /* Taylor coefficients (output) */
    double      **taylors)     /* matrix of coifficient vectors */
/* the order of the indices in argument and taylors is [var][taylor] */
#endif
#endif

#else
#if defined(_HOS_)
/****************************************************************************/
/* Higher Order Scalar version of the forward mode.                         */
/****************************************************************************/
#if defined(_KEEP_)
int  hos_forward(
#else
int  hos_forward_nk(
#endif
    short  tnum,        /* tape id */
    int    depcheck,    /* consistency chk on # of dependents */
    int    indcheck,    /* consistency chk on # of independents */
    int    gdegree,     /* highest derivative degree */
#if defined(_KEEP_)
    int    keep,        /* flag for reverse sweep */
#endif
    const double *basepoint,  /* independent variable values */
    double **argument,  /* independant variable values */
    double *valuepoint, /* Taylor coefficients (output) */
    double **taylors)   /* matrix of coifficient vectors */


#else
/****************************************************************************/
/* Higher Order Vector version of the forward mode.                         */
/****************************************************************************/
#if defined(_KEEP_)
int  hov_wk_forward(
#else
int  hov_forward(
#endif
    short  tnum,        /* tape id */
    int    depcheck,    /* consistency chk on # of deps */
    int    indcheck,    /* consistency chk on # of indeps */
    int    gdegree,     /* highest derivative degree */
#if defined(_KEEP_)
    int    keep,        /* flag for reverse sweep */
#endif
    int    p,           /* # of taylor series */
    const double *basepoint,  /* independent variable values */
    double ***argument, /* Taylor coefficients (input) */
    double *valuepoint, /* Taylor coefficients (output) */
    double ***taylors)  /* matrix of coifficient vectors */
/* the order of the indices in argument and taylors is [var][taylor][deriv] */

#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
{
    /****************************************************************************/
    /*                                                            ALL VARIABLES */

    unsigned char operation;   /* operation code */
    int ret_c =3;              /* return value */

    locint size = 0;
    locint res  = 0;
    locint arg  = 0;
    locint arg1 = 0;
    locint arg2 = 0;

#if !defined(_NTIGHT_)
    double coval = 0;
    double *d=NULL;
#endif

    int indexi = 0,  indexd = 0;

    /* loop indices */
#if !defined (_ZOS_)
#if !defined (_INT_FOR_)
    int i;
#if !defined (_INDO_)
    int ii;
#endif
#endif
#endif
#if defined (_HIGHER_ORDER_)
    int j, l=0;
#endif
    int ls;
#if defined(_ADOLC_VECTOR_)
#if !defined (_HIGHER_ORDER_)
    int l=0;
#endif
#endif
#if defined (_INT_FOR_)
    int l=0;
#endif
#if defined (_INDO_)
#if defined(_INDOPRO_)
    int l=0;
    int max_ind_dom;
    /* index domains */
    locint** ind_dom;
#endif 
#if defined(_NONLIND_)
    /* nonlinear interaction domains */
    locint* arg_index;
    IndexElement* fod;
    IndexElement* nonl_dom;
    IndexElement_sod* sod;
    IndexElement_sod* temp;
    IndexElement_sod* temp1;
    int maxopind;
    int opind;
    int ii;
#endif
#if defined(_NONLIND_OLD_)
    /* nonlinear interaction domains */
    locint** nonl_dom;
#endif
#endif

    /* other necessary variables */
#if !defined (_ZOS_)
#if !defined (_INDO_)
#if !defined (_INT_FOR_)
    double r0=0.0, x, y, divs;
#if defined(_HIGHER_ORDER_)
    int even;
#endif
#endif
#endif
#endif

#if defined(_INT_FOR_)
#ifdef _TIGHT_
    double  *dp_T0;
    double y;
#endif /* _TIGHT_ */

    /* Taylor stuff */
    unsigned long int  **up_T;

    unsigned long int         *Tres, *Targ, *Targ1, *Targ2;
#ifdef _TIGHT_
    unsigned long int         *Tqo;
    unsigned long int         *Targ1OP, *Targ2OP;
#endif

#define T0res  T0temp
#else
#if defined(_INDO_)
#ifdef _TIGHT_
    double  *dp_T0;
#endif /* _TIGHT_ */
#define T0res  T0temp
#define T0arg  T0temp

#else
    double *dp_T0;
#if !defined(_ZOS_)
#if  defined(_FOS_)
    double  *dp_T;
# define T_TEMP Ttemp;
# else
    double *dp_Ttemp, **dpp_T;
#endif
    double         *Tres, *Targ, *Targ1, *Targ2, *Tqo;

#if defined (_HIGHER_ORDER_)
    double         *TresOP, *TresOP2, *zOP;
    double *dp_z;
#endif
   double         *TargOP, *Targ1OP, *Targ2OP;
   double         T0temp;
#endif
#define T0res  T0temp
#define T0arg  T0temp
#endif
#endif

#if defined(_HIGHER_ORDER_)
    int k = gdegree;
#endif

#if defined(_KEEP_)
    int taylbuf=0;
#endif

#if defined(_HOV_)
    int pk = k*p;
#else
#if defined(_HOV_WK_)
    int pk = k*p;
#endif
#endif

    locint switchnum = 0;
#if !defined(_NTIGHT_)
    double* signature = NULL;
#endif

    /* extern diff. function variables */
#if defined(_EXTERN_)
#  undef (_EXTERN_)
#endif
    /* ZOS_FORWARD */
#if defined(_ZOS_)
#   define _EXTERN_ 1
#   define ADOLC_EXT_FCT_POINTER zos_forward
#   define ADOLC_EXT_FCT_IARR_POINTER zos_forward_iArr
#   define ADOLC_EXT_FCT_COMPLETE \
    zos_forward(n, edfct->dp_x, m, edfct->dp_y)
#   define ADOLC_EXT_FCT_IARR_COMPLETE \
    zos_forward_iArr(iArrLength, iArr, n, edfct->dp_x, m, edfct->dp_y)
#   define ADOLC_EXT_FCT_V2_COMPLETE \
    zos_forward(iArrLength, iArr, nin, nout, insz, edfct2->x, outsz, edfct2->y, edfct2->context)
#   define ADOLC_EXT_LOOP
#   define ADOLC_EXT_SUBSCRIPT
#   define ADOLC_EXT_SUBSCRIPT_START
#   define ADOLC_EXT_COPY_TAYLORS(dest,src)
#endif
    /* FOS_FORWARD */
#if defined(_FOS_)
#   define _EXTERN_ 1
#   define ADOLC_EXT_FCT_POINTER fos_forward
#   define ADOLC_EXT_FCT_IARR_POINTER fos_forward_iArr
#   define ADOLC_EXT_FCT_COMPLETE \
    fos_forward(n, edfct->dp_x, edfct->dp_X, m, edfct->dp_y, edfct->dp_Y)
#   define ADOLC_EXT_FCT_IARR_COMPLETE \
    fos_forward_iArr(iArrLength, iArr, n, edfct->dp_x, edfct->dp_X, m, edfct->dp_y, edfct->dp_Y)
#   define ADOLC_EXT_POINTER_X edfct->dp_X
#   define ADOLC_EXT_POINTER_Y edfct->dp_Y
#   define ADOLC_EXT_FCT_V2_COMPLETE \
    fos_forward(iArrLength, iArr, nin, nout, insz, edfct2->x, edfct2->xp, outsz, edfct2->y, edfct2->yp, edfct2->context)
#   define ADOLC_EXT_V2_POINTER_X edfct2->xp
#   define ADOLC_EXT_V2_POINTER_Y edfct2->yp
#   define ADOLC_EXT_LOOP
#   define ADOLC_EXT_SUBSCRIPT
#   define ADOLC_EXT_COPY_TAYLORS(dest,src) dest=src
#   define ADOLC_EXT_COPY_TAYLORS_BACK(dest,src) src=dest
#endif
    /* FOV_FORWARD */
#if defined(_FOV_)
#   define _EXTERN_ 1
#   define ADOLC_EXT_FCT_POINTER fov_forward
#   define ADOLC_EXT_FCT_IARR_POINTER fov_forward_iArr
#   define ADOLC_EXT_FCT_COMPLETE \
    fov_forward(n, edfct->dp_x,p, edfct->dpp_X, m, edfct->dp_y, edfct->dpp_Y)
#   define ADOLC_EXT_FCT_IARR_COMPLETE \
    fov_forward_iArr(iArrLength, iArr, n, edfct->dp_x,p, edfct->dpp_X, m, edfct->dp_y, edfct->dpp_Y)
#   define ADOLC_EXT_POINTER_X edfct->dpp_X
#   define ADOLC_EXT_POINTER_Y edfct->dpp_Y
#   define ADOLC_EXT_FCT_V2_COMPLETE \
    fov_forward(iArrLength, iArr, nin, nout, insz, edfct2->x, p, edfct2->Xp, outsz, edfct2->y, edfct2->Yp, edfct2->context)
#   define ADOLC_EXT_V2_POINTER_X edfct2->Xp
#   define ADOLC_EXT_V2_POINTER_Y edfct2->Yp
#   define ADOLC_EXT_LOOP for (loop2 = 0; loop2 < p; ++loop2)
#   define ADOLC_EXT_SUBSCRIPT [loop2]
#   define ADOLC_EXT_COPY_TAYLORS(dest,src) dest=src
#   define ADOLC_EXT_COPY_TAYLORS_BACK(dest,src) 
#endif

#if defined(_EXTERN_)
    locint n, m;
    ext_diff_fct *edfct;
    ext_diff_fct_v2 *edfct2;
    int loop,oloop;
    int iArrLength;
    int *iArr;
#   if defined(_FOV_)
        int loop2;
#   endif
    int ext_retc;
    int nin, nout;
    locint *insz, *outsz;
#endif

#if defined(_ABS_NORM_)
    short sig;
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
    MPI_Status status;
    struct AMPI_Request_S request;
#endif
    locint qq;

    ADOLC_OPENMP_THREAD_NUMBER;

#if defined(ADOLC_DEBUG)
    /****************************************************************************/
    /*                                                           DEBUG MESSAGES */
    fprintf(DIAG_OUT,"Call of %s(..) with tag: %d, n: %d, m %d,\n",
            GENERATED_FILENAME, tnum, indcheck, depcheck);
#if defined(_KEEP_)
    fprintf(DIAG_OUT,"                    keep: %d\n", keep);
#endif
#if defined(_HIGHER_ORDER_)
    fprintf(DIAG_OUT,"                    degree: %d\n",gdegree);
#endif
#if defined(_ADOLC_VECTOR_)
    fprintf(DIAG_OUT,"                    p: %d\n\n",p);
#endif

#endif

    /****************************************************************************/
    /*                                                                    INITs */

   /* Set up stuff for the tape */
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    /* Initialize the Forward Sweep */

    init_for_sweep(tnum);

      if ( (depcheck != ADOLC_CURRENT_TAPE_INFOS.stats[NUM_DEPENDENTS]) ||
            (indcheck != ADOLC_CURRENT_TAPE_INFOS.stats[NUM_INDEPENDENTS]) ) {
        fprintf(DIAG_OUT,"ADOL-C error: forward sweep on tape %d  aborted!\n"
                "Number of dependent(%u) and/or independent(%u) variables passed"
                " to forward is\ninconsistent with number "
                "recorded on tape (%zu, %zu) \n", tnum,
                depcheck, indcheck,
                ADOLC_CURRENT_TAPE_INFOS.stats[NUM_DEPENDENTS],
                ADOLC_CURRENT_TAPE_INFOS.stats[NUM_INDEPENDENTS]);
        adolc_exit(-1,"",__func__,__FILE__,__LINE__);
    }
#if defined(_ABS_NORM_) || defined(_ABS_NORM_SIG_)
      if (! ADOLC_CURRENT_TAPE_INFOS.stats[NO_MIN_MAX] ) {
	  fprintf(DIAG_OUT,"ADOL-C error: tape %d was not created compatible "
		  "with %s\n              Please call enableMinMaxUsingAbs() "
		  "before trace_on(%d)\n", tnum, __FUNCTION__, tnum);
	  adolc_exit(-1,"",__func__,__FILE__,__LINE__);
      }
#if defined(_ABS_NORM_SIG_) || defined(_INDOPRO_)
      if (swcheck != ADOLC_CURRENT_TAPE_INFOS.stats[NUM_SWITCHES]) {
        fprintf(DIAG_OUT,"ADOL-C error: forward sweep on tape %d  aborted!\n"
                "Number of switches(%u) passed"
                " to forward is\ninconsistent with number "
                "recorded on tape (%zu) \n", tnum,
                swcheck,
                ADOLC_CURRENT_TAPE_INFOS.stats[NUM_SWITCHES]);
        adolc_exit(-1,"",__func__,__FILE__,__LINE__);
      }
#endif
#endif
    /****************************************************************************/
    /*                                                        MEMORY ALLOCATION */
    /* olvo 980626 has to be revised for common blocks */

    /*--------------------------------------------------------------------------*/
#if !defined(_NTIGHT_)
    dp_T0 = myalloc1(ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES]);
    ADOLC_CURRENT_TAPE_INFOS.dp_T0 = dp_T0;

    if(ADOLC_CURRENT_TAPE_INFOS.stats[NO_MIN_MAX]) {
	if (ADOLC_CURRENT_TAPE_INFOS.signature == NULL) {
	    signature = myalloc1(ADOLC_CURRENT_TAPE_INFOS.stats[NUM_SWITCHES]);
	    ADOLC_CURRENT_TAPE_INFOS.signature = signature;
	} else
	    signature = ADOLC_CURRENT_TAPE_INFOS.signature;
    }

    ADOLC_CURRENT_TAPE_INFOS.dpp_T = &dp_T0;
    ADOLC_CURRENT_TAPE_INFOS.numTay = 0;
    ADOLC_CURRENT_TAPE_INFOS.gDegree = 0;
    ADOLC_CURRENT_TAPE_INFOS.workMode = ADOLC_ZOS_FORWARD;
#endif /* !_NTIGHT_ */
#if defined(_ZOS_)                                                   /* ZOS */

#if defined(_KEEP_)
    if (keep>1) {
        fprintf(DIAG_OUT,"\n ADOL-C error: zero order scalar forward cannot save"
                " more\nthan zero order taylor coefficients!\n");
        adolc_exit(-1,"",__func__,__FILE__,__LINE__);
    }
#endif
#if defined(_KEEP_)
    if (keep) {
      taylbuf = ADOLC_CURRENT_TAPE_INFOS.stats[TAY_BUFFER_SIZE];

        taylor_begin(taylbuf,keep-1);
    }
#endif

    /*--------------------------------------------------------------------------*/
#else                                                                /* FOS */
#if defined(_FOS_)
#if defined(_KEEP_)
    if (keep>2) {
        fprintf(DIAG_OUT,"\n ADOL-C error: first order scalar forward cannot save"
                " more  \nthan first order taylor coefficients!\n");
        adolc_exit(-1,"",__func__,__FILE__,__LINE__);
    }
#endif
    dp_T = myalloc1(ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES]);
    ADOLC_CURRENT_TAPE_INFOS.dpp_T = &dp_T;
    ADOLC_CURRENT_TAPE_INFOS.numTay = 1;
    ADOLC_CURRENT_TAPE_INFOS.gDegree = 1;
    ADOLC_CURRENT_TAPE_INFOS.workMode = ADOLC_FOS_FORWARD;
# define TAYLOR_BUFFER dp_T
#if defined(_KEEP_)
    if (keep) {
        taylbuf = ADOLC_CURRENT_TAPE_INFOS.stats[TAY_BUFFER_SIZE];
        taylor_begin(taylbuf,keep-1);
    }
#endif

    /*--------------------------------------------------------------------------*/
#else                                                                /* INF_FOR */
#if defined(_INT_FOR_)
        up_T     = myalloc2_ulong(ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES],p);
#define TAYLOR_BUFFER up_T

    /*--------------------------------------------------------------------------*/
#else                                                                /* INDOPRO */
#if defined(_INDO_)
#if defined(_INDOPRO_)
    /* index domains */
    ind_dom = (locint **)  malloc(sizeof(locint*) * ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES]);
    max_ind_dom = ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES];

    for(i=0;i<max_ind_dom;i++)
    {
        ind_dom[i] = (locint *)  malloc(sizeof(locint) * (NUMNNZ+2));
        ind_dom[i][0] = 0;
        ind_dom[i][1] = NUMNNZ;
    }
#if defined(_ABS_NORM_)
    indexd = swcheck;
#endif
#endif
#if defined(_NONLIND_)
	maxopind=ADOLC_CURRENT_TAPE_INFOS.stats[NUM_OPERATIONS]+ADOLC_CURRENT_TAPE_INFOS.stats[NUM_EQ_PROD];
	fod = (struct IndexElement*) malloc(sizeof(struct IndexElement) * maxopind);
	nonl_dom = (struct IndexElement*) malloc(sizeof(struct IndexElement) * indcheck);
	arg_index = (locint *)  malloc(sizeof(locint) * (ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES]));
        opind = 0;
	sod = (struct IndexElement_sod*) malloc(sizeof(struct IndexElement_sod) * indcheck);

	for(i=0;i<indcheck;i++)
	  {
	    sod[i].entry = 0;
	    sod[i].left = NULL;
	    nonl_dom[i].entry = indcheck+2;
	    nonl_dom[i].left = NULL;
	    nonl_dom[i].right = NULL;
	  }
#endif
#if defined(_NONLIND_OLD_)

    nonl_dom = (locint**) malloc(sizeof(locint*) * indcheck);
    for(i=0;i<indcheck;i++){
          nonl_dom[i] = (locint*) malloc(sizeof(locint)*(NUMNNZ+2));
          nonl_dom[i][0]=0;
          nonl_dom[i][1]=NUMNNZ;
       }
#endif

    /*--------------------------------------------------------------------------*/
#else                                                                /* FOV */
#if defined(_FOV_)
    dpp_T = myalloc2(ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES],p);
    ADOLC_CURRENT_TAPE_INFOS.dpp_T = dpp_T;
    ADOLC_CURRENT_TAPE_INFOS.numTay = p;
    ADOLC_CURRENT_TAPE_INFOS.gDegree = 1;
    ADOLC_CURRENT_TAPE_INFOS.workMode = ADOLC_FOV_FORWARD;
# define TAYLOR_BUFFER dpp_T
    dp_Ttemp = myalloc1(p);
# define T_TEMP dp_Ttemp;

    /*--------------------------------------------------------------------------*/
#else                                                                /* HOS */
#if defined(_HOS_)
    dpp_T = myalloc2(ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES],k);
    ADOLC_CURRENT_TAPE_INFOS.dpp_T = dpp_T;
    ADOLC_CURRENT_TAPE_INFOS.numTay = 1;
    ADOLC_CURRENT_TAPE_INFOS.gDegree = k;
    ADOLC_CURRENT_TAPE_INFOS.workMode = ADOLC_HOS_FORWARD;
# define TAYLOR_BUFFER dpp_T
    dp_z  = myalloc1(k);
    dp_Ttemp = myalloc1(k);
# define T_TEMP dp_Ttemp;
#if defined(_KEEP_)
    if (keep) {
        taylbuf = ADOLC_CURRENT_TAPE_INFOS.stats[TAY_BUFFER_SIZE];
        taylor_begin(taylbuf,keep-1);
    }
#endif

    /*--------------------------------------------------------------------------*/
#else                                                     /* HOV and HOV_WK */
    dpp_T = myalloc2(ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES],p*k);
    ADOLC_CURRENT_TAPE_INFOS.dpp_T = dpp_T;
    ADOLC_CURRENT_TAPE_INFOS.numTay = p;
    ADOLC_CURRENT_TAPE_INFOS.gDegree = k;
    ADOLC_CURRENT_TAPE_INFOS.workMode = ADOLC_HOV_FORWARD;
# define TAYLOR_BUFFER dpp_T
    dp_z  = myalloc1(k);
    dp_Ttemp = myalloc1(p*k);
# define T_TEMP dp_Ttemp;
#if defined(_KEEP_)
    if (keep) {
        taylbuf = ADOLC_CURRENT_TAPE_INFOS.stats[TAY_BUFFER_SIZE];
        taylor_begin(taylbuf,keep-1);
    }
#endif
#endif
#endif
#endif
#endif
#endif
#endif
    /****************************************************************************/
    /*                                                            FORWARD SWEEP */

#if defined(ADOLC_DEBUG)
/* #include <string.h> */
    int v = 0;
    unsigned int countPerOperation[256], taylorPerOperation[256];
    memset(countPerOperation, 0, 1024);
    memset(taylorPerOperation, 0, 1024);
#   define UPDATE_TAYLORWRITTEN(X) taylorPerOperation[operation] += X;
#else
#   define UPDATE_TAYLORWRITTEN(X)
#endif /* ADOLC_DEBUG */

    operation=get_op_f();
#if defined(ADOLC_DEBUG)
    ++countPerOperation[operation];
#endif /* ADOLC_DEBUG */

    while (operation !=end_of_tape) {
      
      switch (operation) {


                /****************************************************************************/
                /*                                                                  MARKERS */

                /*--------------------------------------------------------------------------*/
            case end_of_op:                                          /* end_of_op */
                get_op_block_f();
                operation=get_op_f();
                /* Skip next operation, it's another end_of_op */
                break;

                /*--------------------------------------------------------------------------*/
            case end_of_int:                                        /* end_of_int */
                get_loc_block_f();
                break;

                /*--------------------------------------------------------------------------*/
            case end_of_val:                                        /* end_of_val */
               get_val_block_f();
                break;
                /*--------------------------------------------------------------------------*/
            case start_of_tape:                                  /* start_of_tape */
            case end_of_tape:                                      /* end_of_tape */
                break;


                /****************************************************************************/
                /*                                                               COMPARISON */

                /*--------------------------------------------------------------------------*/
            case eq_zero:                                              /* eq_zero */
                arg = get_locint_f();

#if !defined(_NTIGHT_)
                if (dp_T0[arg] != 0) {
                    if (ADOLC_GLOBAL_TAPE_VARS.branchSwitchWarning)
                        fprintf(DIAG_OUT,
                                "ADOL-C Warning: Branch switch detected in comparison "
                                "(operator eq_zero).\n"
                                "Forward sweep aborted! Retaping recommended!\n");
                    ret_c = -1;
                    operation = end_of_tape;
                    continue;
                }
                ret_c = 0;
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case neq_zero:                                            /* neq_zero */
                arg = get_locint_f();

#if !defined(_NTIGHT_)
                if (dp_T0[arg] == 0) {
                    if (ADOLC_GLOBAL_TAPE_VARS.branchSwitchWarning)
                        fprintf(DIAG_OUT,
                                "ADOL-C Warning: Branch switch detected in comparison "
                                "(operator neq_zero).\n"
                                "Forward sweep aborted! Retaping recommended!\n");
                    ret_c = -1;
                    operation = end_of_tape;
                    continue;
                }
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case le_zero:                                              /* le_zero */
                arg = get_locint_f();

#if !defined(_NTIGHT_)
                if (dp_T0[arg] > 0) {
                    if (ADOLC_GLOBAL_TAPE_VARS.branchSwitchWarning)
                        fprintf(DIAG_OUT,
                                "ADOL-C Warning: Branch switch detected in comparison "
                                "(operator le_zero).\n"
                                "Forward sweep aborted! Retaping recommended!\n");
                    ret_c = -1;
                    operation = end_of_tape;
                    continue;
                }
                if (dp_T0[arg] == 0)
                    ret_c = 0;
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case gt_zero:                                              /* gt_zero */
                arg = get_locint_f();

#if !defined(_NTIGHT_)
                if (dp_T0[arg] <= 0) {
                    if (ADOLC_GLOBAL_TAPE_VARS.branchSwitchWarning)
                        fprintf(DIAG_OUT,
                                "ADOL-C Warning: Branch switch detected in comparison "
                                "(operator gt_zero).\n"
                                "Forward sweep aborted! Retaping recommended!\n");
                    ret_c = -1;
                    operation = end_of_tape;
                    continue;
                }
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case ge_zero:                                              /* ge_zero */
                arg = get_locint_f();

#if !defined(_NTIGHT_)
                if (dp_T0[arg] < 0) {
                    if (ADOLC_GLOBAL_TAPE_VARS.branchSwitchWarning)
                        fprintf(DIAG_OUT,
                                "ADOL-C Warning: Branch switch detected in comparison "
                                "(operator ge_zero).\n"
                                "Forward sweep aborted! Retaping recommended!\n");
                    ret_c = -1;
                    operation = end_of_tape;
                    continue;
                }
                if (dp_T0[arg] == 0)
                    ret_c = 0;
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case lt_zero:                                              /* lt_zero */
                arg = get_locint_f();

#if !defined(_NTIGHT_)
                if (dp_T0[arg] >= 0) {
                    if (ADOLC_GLOBAL_TAPE_VARS.branchSwitchWarning)
                        fprintf(DIAG_OUT,
                                "ADOL-C Warning: Branch switch detected in comparison "
                                "(operator lt_zero).\n"
                                "Forward sweep aborted! Retaping recommended!\n");
                    ret_c = -1;
                    operation = end_of_tape;
                    continue;
                }
#endif /* !_NTIGHT_ */
                break;


                /****************************************************************************/
                /*                                                              ASSIGNMENTS */

                /*--------------------------------------------------------------------------*/
            case assign_a:           /* assign an adouble variable an    assign_a */
                /* adouble value. (=) */
                arg = get_locint_f();
                res = get_locint_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)


#if !defined(_NTIGHT_)
                dp_T0[res] = dp_T0[arg];
#endif /* !_NTIGHT_ */

#if defined(_INDO_)
#if defined(_INDOPRO_)
                copy_index_domain(res, arg, ind_dom);
#endif
#if defined(_NONLIND_)
                arg_index[res] = arg_index[arg];		
#endif           
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Targ,TAYLOR_BUFFER[arg])
                ASSIGN_T(Tres,TAYLOR_BUFFER[res])

                FOR_0_LE_l_LT_pk
                TRES_INC = TARG_INC;
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;

                /*--------------------------------------------------------------------------*/
            case assign_d:            /* assign an adouble variable a    assign_d */
                /* double value. (=) */
                res   = get_locint_f();
#if !defined(_NTIGHT_)
                coval = 
#endif
		get_val_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res] = coval;
#endif /* !_NTIGHT_ */

#if defined(_INDO_)
#if defined(_INDOPRO_)
		ind_dom[res][0]=0;
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = NULL;
		fod[opind].right = NULL;
                arg_index[res] = opind++;		
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])

                FOR_0_LE_l_LT_pk
                TRES_INC = 0;
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;

                /*--------------------------------------------------------------------------*/
            case neg_sign_p:
            case recipr_p:
            case assign_p:            /* assign an adouble variable a    assign_d */
                /* double value. (=) */
                arg   = get_locint_f();
                res   = get_locint_f();
#if !defined(_NTIGHT_)
                coval = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore[arg];
#endif

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                if (operation == recipr_p) 
                    coval = 1.0/coval;
                else if (operation == neg_sign_p)
                    coval = - coval;
                dp_T0[res] = coval;
#endif /* !_NTIGHT_ */

#if defined(_INDO_)
#if defined(_INDOPRO_)
		ind_dom[res][0]=0;
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = NULL;
		fod[opind].right = NULL;
                arg_index[res] = opind++;		
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])

                FOR_0_LE_l_LT_pk
                TRES_INC = 0;
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;

                /*--------------------------------------------------------------------------*/
            case assign_d_zero:  /* assign an adouble variable a    assign_d_zero */
                /* double value. (0) (=) */
                res   = get_locint_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res] = 0.0;
#endif /* !_NTIGHT_ */

#if defined(_INDO_)
#if defined(_INDOPRO_)
		ind_dom[res][0]=0;
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = NULL;
		fod[opind].right = NULL;
                arg_index[res] = opind++;		
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])

                FOR_0_LE_l_LT_pk
                TRES_INC = 0;
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;

                /*--------------------------------------------------------------------------*/
            case assign_d_one:    /* assign an adouble variable a    assign_d_one */
                /* double value. (1) (=) */
                res   = get_locint_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res] = 1.0;
#endif /* !_NTIGHT_ */

#if defined(_INDO_)
#if defined(_INDOPRO_)
		ind_dom[res][0]=0;
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = NULL;
		fod[opind].right = NULL;
                arg_index[res] = opind++;		
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])

                FOR_0_LE_l_LT_pk
                TRES_INC = 0;

#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;

                /*--------------------------------------------------------------------------*/
            case assign_ind:       /* assign an adouble variable an    assign_ind */
                /* independent double value (<<=) */
                res = get_locint_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res] = basepoint[indexi];
#endif /* !_NTIGHT_ */

#if defined(_INDO_)
#if defined(_INDOPRO_)
		ind_dom[res][0] = 1;
		ind_dom[res][2] = indexi;
#endif		
#if defined(_NONLIND_)
		fod[opind].entry = indexi;
		fod[opind].left = NULL;
		fod[opind].right = NULL;
                arg_index[res] = opind++;		
#endif		
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])

#ifdef _INT_FOR_
                FOR_0_LE_l_LT_p
                TRES_INC = ARGUMENT(indexi,l,i);
#else
                FOR_0_LE_l_LT_p
                FOR_0_LE_i_LT_k
                TRES_INC = ARGUMENT(indexi,l,i);
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
                ++indexi;
                break;

                /*--------------------------------------------------------------------------*/
            case assign_dep:           /* assign a float variable a    assign_dep */
                /* dependent adouble value. (>>=) */
                res = get_locint_f();

#if !defined(_INDO_)
#if !defined(_NTIGHT_)
                if ( valuepoint != NULL )
                  valuepoint[indexd] = dp_T0[res];
#endif /* !_NTIGHT_ */
#endif

#if defined(_INDO_)
#if defined(_INDOPRO_) && !defined(_NONLIND_OLD_)
          if (ind_dom[res][0] != 0) {
            crs[indexd] = (unsigned int*) malloc(sizeof(unsigned int) * (ind_dom[res][0]+1));
            crs[indexd][0] = ind_dom[res][0];
            for(l=1;l<=crs[indexd][0];l++) {
              crs[indexd][l] = ind_dom[res][l+1];
            }
          }
          else {
            crs[indexd] = (unsigned int*) malloc(sizeof(unsigned int));
            crs[indexd][0] =0;
          }
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])

#ifdef _INT_FOR_
                if (taylors != 0 )  /* ??? question: why here? */
                    FOR_0_LE_l_LT_p
                    TAYLORS(indexd,l,i) = TRES_INC;
#else
                if (taylors != 0 )  /* ??? question: why here? */
                    FOR_0_LE_l_LT_p
                    FOR_0_LE_i_LT_k
                    TAYLORS(indexd,l,i) = TRES_INC;
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
                indexd++;
                break;


                /****************************************************************************/
                /*                                                   OPERATION + ASSIGNMENT */

                /*--------------------------------------------------------------------------*/
            case eq_plus_d:            /* Add a floating point to an    eq_plus_d */
                /* adouble. (+=) */
                res   = get_locint_f();
#if !defined(_NTIGHT_)
                coval = 
#endif
		get_val_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res] += coval;
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case eq_plus_p:            /* Add a floating point to an    eq_plus_p */
                /* adouble. (+=) */
                arg   = get_locint_f();
                res   = get_locint_f();
#if !defined(_NTIGHT_)
                coval = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore[arg];
#endif
                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res] += coval;
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case eq_plus_a:             /* Add an adouble to another    eq_plus_a */
                /* adouble. (+=) */
                arg = get_locint_f();
                res = get_locint_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res] += dp_T0[arg];
#endif /* !_NTIGHT_ */

#if defined(_INDO_)
#if defined(_INDOPRO_)
                merge_2_index_domains(res, arg, ind_dom);
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[res]];
		fod[opind].right = &fod[arg_index[arg]];
                arg_index[res] = opind++;		
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])
                ASSIGN_T(Targ, TAYLOR_BUFFER[arg])

#ifdef _INT_FOR_
                FOR_0_LE_l_LT_pk
                TRES_INC |= TARG_INC;
#else
                FOR_0_LE_l_LT_pk
                TRES_INC += TARG_INC;
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;

                /*--------------------------------------------------------------------------*/
            case eq_min_d:       /* Subtract a floating point from an    eq_min_d */
                /* adouble. (-=) */
                res = get_locint_f();
#if !defined(_NTIGHT_)
                coval = 
#endif
		get_val_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res] -= coval;
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case eq_min_p:       /* Subtract a floating point from an    eq_min_p */
                /* adouble. (-=) */
                arg = get_locint_f();
                res = get_locint_f();
#if !defined(_NTIGHT_)
                coval = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore[arg];
#endif

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res] -= coval;
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case eq_min_a:        /* Subtract an adouble from another    eq_min_a */
                /* adouble. (-=) */
                arg = get_locint_f();
                res = get_locint_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res] -= dp_T0[arg];
#endif /* !_NTIGHT_ */

#if defined(_INDO_)
#if defined(_INDOPRO_)
                merge_2_index_domains(res, arg, ind_dom);
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[res]];
		fod[opind].right = &fod[arg_index[arg]];
                arg_index[res] = opind++;		
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])
                ASSIGN_T(Targ, TAYLOR_BUFFER[arg])

#ifdef _INT_FOR_
                FOR_0_LE_l_LT_pk
                TRES_INC |= TARG_INC;
#else
                FOR_0_LE_l_LT_pk
                TRES_INC -= TARG_INC;
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;

                /*--------------------------------------------------------------------------*/
            case eq_mult_d:              /* Multiply an adouble by a    eq_mult_d */
                /* flaoting point. (*=) */
                res   = get_locint_f();
#if !defined(_NTIGHT_)
                coval = 
#endif
		get_val_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res] *= coval;
#endif /* !_NTIGHT_ */

#if !defined(_INDO_)
#if !defined(_ZOS_) /* BREAK_ZOS */
#if !defined( _INT_FOR_)

                FOR_0_LE_l_LT_pk
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])

                FOR_0_LE_l_LT_pk
                TRES_INC *= coval;
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;

                /*--------------------------------------------------------------------------*/
            case eq_mult_p:              /* Multiply an adouble by a    eq_mult_p */
                /* flaoting point. (*=) */
                arg   = get_locint_f();
                res   = get_locint_f();
#if !defined(_NTIGHT_)
                coval = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore[arg];
#endif

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res] *= coval;
#endif /* !_NTIGHT_ */

#if !defined(_INDO_)
#if !defined(_ZOS_) /* BREAK_ZOS */
#if !defined( _INT_FOR_)

                FOR_0_LE_l_LT_pk
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])

                FOR_0_LE_l_LT_pk
                TRES_INC *= coval;
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;

                /*--------------------------------------------------------------------------*/
            case eq_mult_a:       /* Multiply one adouble by another    eq_mult_a */
                /* (*=) */
                arg = get_locint_f();
                res = get_locint_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if defined(_INDO_)
#if defined(_INDOPRO_)
                merge_2_index_domains(res, arg, ind_dom);
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[res]];
		fod[opind].right = &fod[arg_index[arg]];
		traverse_unary(&fod[arg_index[res]], nonl_dom, &fod[arg_index[arg]], indcheck+1,maxopind+2);
		traverse_unary(&fod[arg_index[arg]], nonl_dom, &fod[arg_index[res]], indcheck+1,maxopind+2);
                arg_index[res] = opind++;		
#endif
#if defined(_NONLIND_OLD_)
                extend_nonlinearity_domain_binary(res, arg, ind_dom, nonl_dom);
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])
                ASSIGN_T(Targ, TAYLOR_BUFFER[arg])

                INC_pk_1(Tres)
                INC_pk_1(Targ)

#ifdef _INT_FOR_
                FOR_p_GT_l_GE_0
                TRES_FODEC |= TARG_DEC;
#else
                FOR_p_GT_l_GE_0
                FOR_k_GT_i_GE_0
                { TRES_FODEC = dp_T0[res]*TARG_DEC +
                               TRES*dp_T0[arg];
                  DEC_TRES_FO
#ifdef _HIGHER_ORDER_
                  TresOP = Tres-i;
                  TargOP = Targ;

                  for (j=0;j<i;j++)
                  *Tres += (*TresOP++) * (*TargOP--);
                  Tres--;
#endif /* _HIGHER_ORDER_ */
                }
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
#if !defined(_NTIGHT_)
               dp_T0[res] *= dp_T0[arg];
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case incr_a:                        /* Increment an adouble    incr_a */
                res   = get_locint_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res]++;
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case decr_a:                        /* Increment an adouble    decr_a */
                res   = get_locint_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res]--;
#endif /* !_NTIGHT_ */
                break;


                /****************************************************************************/
                /*                                                        BINARY OPERATIONS */

                /*--------------------------------------------------------------------------*/
            case plus_a_a:                 /* : Add two adoubles. (+)    plus a_a */
                arg1 = get_locint_f();
                arg2 = get_locint_f();
                res  = get_locint_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

		 
#if !defined(_NTIGHT_)
                dp_T0[res] = dp_T0[arg1] + dp_T0[arg2];
#endif /* !_NTIGHT_ */

#if defined(_INDO_)
#if defined(_INDOPRO_)
                combine_2_index_domains(res, arg1, arg2, ind_dom);
#endif      
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[arg1]];
		fod[opind].right = &fod[arg_index[arg2]];
                arg_index[res] = opind++;		
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres,  TAYLOR_BUFFER[res])
                ASSIGN_T(Targ1, TAYLOR_BUFFER[arg1])
                ASSIGN_T(Targ2, TAYLOR_BUFFER[arg2])

#ifdef _INT_FOR_
                FOR_0_LE_l_LT_pk
                TRES_INC = TARG1_INC | TARG2_INC;
#else
                FOR_0_LE_l_LT_pk
                TRES_INC = TARG1_INC + TARG2_INC;
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;

                /*--------------------------------------------------------------------------*/
            case plus_d_a:             /* Add an adouble and a double    plus_d_a */
                /* (+) */
                arg   = get_locint_f();
                res   = get_locint_f();
#if !defined(_NTIGHT_)
                coval = 
#endif
		get_val_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res] = dp_T0[arg] + coval;
#endif /* !_NTIGHT_ */

#if defined(_INDO_)
#if defined(_INDOPRO_)
                copy_index_domain(res, arg, ind_dom);
#endif
#if defined(_NONLIND_)
                arg_index[res] = arg_index[arg];
#endif               
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])
                ASSIGN_T(Targ, TAYLOR_BUFFER[arg])

                FOR_0_LE_l_LT_pk
                TRES_INC = TARG_INC;
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;

                /*--------------------------------------------------------------------------*/
            case plus_a_p:             /* Add an adouble and a double    plus_a_p */
            case min_a_p:                /* Subtract an adouble from a    min_a_p */
                /* (+) */
                arg   = get_locint_f();
                arg1  = get_locint_f();
                res   = get_locint_f();
#if !defined(_NTIGHT_)
                coval = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore[arg1];
#endif

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                if (operation == plus_a_p)
                    dp_T0[res] = dp_T0[arg] + coval;
                else if (operation == min_a_p)
                    dp_T0[res] = dp_T0[arg] - coval;
#endif /* !_NTIGHT_ */

#if defined(_INDO_)
#if defined(_INDOPRO_)
                copy_index_domain(res, arg, ind_dom);
#endif
#if defined(_NONLIND_)
                arg_index[res] = arg_index[arg];
#endif               
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])
                ASSIGN_T(Targ, TAYLOR_BUFFER[arg])

                FOR_0_LE_l_LT_pk
                TRES_INC = TARG_INC;
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;

                /*--------------------------------------------------------------------------*/
            case min_a_a:              /* Subtraction of two adoubles     min_a_a */
                /* (-) */
                arg1 = get_locint_f();
                arg2 = get_locint_f();
                res  = get_locint_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res] = dp_T0[arg1] -
                                               dp_T0[arg2];
#endif /* !_NTIGHT_ */


#if defined(_INDO_)    
#if defined(_INDOPRO_)
                combine_2_index_domains(res, arg1, arg2, ind_dom);
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[arg1]];
		fod[opind].right = &fod[arg_index[arg2]];
                arg_index[res] = opind++;		
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres,  TAYLOR_BUFFER[res])
                ASSIGN_T(Targ1, TAYLOR_BUFFER[arg1])
                ASSIGN_T(Targ2, TAYLOR_BUFFER[arg2])

#ifdef _INT_FOR_
                FOR_0_LE_l_LT_pk
                TRES_INC = TARG1_INC | TARG2_INC;
#else
                 FOR_0_LE_l_LT_pk
                TRES_INC = TARG1_INC - TARG2_INC;
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;

                /*--------------------------------------------------------------------------*/
            case min_d_a:                /* Subtract an adouble from a    min_d_a */
                /* double (-) */
                arg =get_locint_f();
                res = get_locint_f();
#if !defined(_NTIGHT_)
                coval = 
#endif
		get_val_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res] = coval - dp_T0[arg];
#endif /* !_NTIGHT_ */

#if defined(_INDO_)
#if defined(_INDOPRO_)
                copy_index_domain(res, arg, ind_dom);
#endif
#if defined(_NONLIND_)
                arg_index[res] = arg_index[arg];		
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])
                ASSIGN_T(Targ, TAYLOR_BUFFER[arg])

#ifdef _INT_FOR_
                FOR_0_LE_l_LT_pk
                TRES_INC = TARG_INC;
#else
                FOR_0_LE_l_LT_pk
                TRES_INC = -TARG_INC;
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;

                /*--------------------------------------------------------------------------*/
            case mult_a_a:               /* Multiply two adoubles (*)    mult_a_a */
                arg1 = get_locint_f();
                arg2 = get_locint_f();
                res  = get_locint_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if defined(_INDO_)
#if defined(_INDOPRO_)
                combine_2_index_domains(res, arg1, arg2, ind_dom);
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[arg1]];
		fod[opind].right = &fod[arg_index[arg2]];
		traverse_unary(&fod[arg_index[arg1]], nonl_dom, &fod[arg_index[arg2]], indcheck+1,maxopind+2);
		traverse_unary(&fod[arg_index[arg2]], nonl_dom, &fod[arg_index[arg1]], indcheck+1,maxopind+2);
		arg_index[res] = opind++;		
#endif
#if defined(_NONLIND_OLD_)
		extend_nonlinearity_domain_binary(arg1, arg2, ind_dom, nonl_dom);
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres,  TAYLOR_BUFFER[res])
                ASSIGN_T(Targ1, TAYLOR_BUFFER[arg1])
                ASSIGN_T(Targ2, TAYLOR_BUFFER[arg2])

#ifdef _INT_FOR_
                FOR_0_LE_l_LT_p
                TRES_FOINC = TARG2_INC | TARG1_INC;
#else
                /* olvo 980915 now in reverse order to allow x = x*x etc. */
                INC_pk_1(Tres)
                INC_pk_1(Targ1)
                INC_pk_1(Targ2)

                FOR_p_GT_l_GE_0
                FOR_k_GT_i_GE_0
                { TRES_FODEC = dp_T0[arg1]*TARG2_DEC +
                               TARG1_DEC*dp_T0[arg2];
                  DEC_TRES_FO
#if defined(_HIGHER_ORDER_)
                  Targ1OP = Targ1-i+1;
                  Targ2OP = Targ2;

                  for (j=0;j<i;j++) {
                  *Tres += (*Targ1OP++) * (*Targ2OP--);
                  }
                  Tres--;
#endif /* _HIGHER_ORDER_ */
            }
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
#if !defined(_NTIGHT_)
                dp_T0[res] = dp_T0[arg1] *
                                               dp_T0[arg2];
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
                /* olvo 991122: new op_code with recomputation */
            case eq_plus_prod:   /* increment a product of           eq_plus_prod */
                /* two adoubles (*) */
                arg1 = get_locint_f();
                arg2 = get_locint_f();
                res  = get_locint_f();

#if defined(_INDO_)
#if defined(_INDOPRO_)
                merge_3_index_domains(res, arg1, arg2, ind_dom);
#endif
#if defined(_NONLIND_)
		// operation: v = v+u*w
		// first step: z = u*w, index domains
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[arg1]];
		fod[opind].right = &fod[arg_index[arg2]];
		// first step: z = u*w, 
		traverse_unary(&fod[arg_index[arg1]], nonl_dom, &fod[arg_index[arg2]], indcheck+1,maxopind+2);
		traverse_unary(&fod[arg_index[arg2]], nonl_dom, &fod[arg_index[arg1]], indcheck+1,maxopind+2);
		opind++;
		// second step: v = v+z, index domains
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[res]];
		fod[opind].right = &fod[opind-1];
		// second step: v = v+z,
                arg_index[res] = opind++;		
#endif
#if defined(_NONLIND_OLD_)
                extend_nonlinearity_domain_binary(arg1, arg2, ind_dom, nonl_dom);
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres,  TAYLOR_BUFFER[res])
                ASSIGN_T(Targ1, TAYLOR_BUFFER[arg1])
                ASSIGN_T(Targ2, TAYLOR_BUFFER[arg2])

#ifdef _INT_FOR_
                FOR_0_LE_l_LT_p
                TRES_FOINC |= TARG2_INC | TARG1_INC;
#else
                /* olvo 980915 now in reverse order to allow x = x*x etc. */
                INC_pk_1(Tres)
                INC_pk_1(Targ1)
                INC_pk_1(Targ2)

                FOR_p_GT_l_GE_0
                FOR_k_GT_i_GE_0
                { TRES_FODEC += dp_T0[arg1]*TARG2_DEC +
                                TARG1_DEC*dp_T0[arg2];
                  DEC_TRES_FO
#if defined(_HIGHER_ORDER_)
                  Targ1OP = Targ1-i+1;
                  Targ2OP = Targ2;

                  for (j=0;j<i;j++)
                  *Tres += (*Targ1OP++) * (*Targ2OP--);
                  Tres--;
#endif /* _HIGHER_ORDER_ */
                }
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
#if !defined(_NTIGHT_)
                dp_T0[res] += dp_T0[arg1] *
                                                    dp_T0[arg2];
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
                /* olvo 991122: new op_code with recomputation */
            case eq_min_prod:    /* decrement a product of            eq_min_prod */
                /* two adoubles (*) */
                arg1 = get_locint_f();
                arg2 = get_locint_f();
                res  = get_locint_f();

#if defined(_INDO_)
#if defined(_INDOPRO_)
                merge_3_index_domains(res, arg1, arg2, ind_dom);
#endif
#if defined(_NONLIND_)
		// operation: v = v-u*w
		// first step: z = u*w, index domains
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[arg1]];
		fod[opind].right = &fod[arg_index[arg2]];
		// first step: z = u*w,
		traverse_unary(&fod[arg_index[arg1]], nonl_dom, &fod[opind], indcheck+1,maxopind+2);
		traverse_unary(&fod[arg_index[arg2]], nonl_dom, &fod[opind], indcheck+1,maxopind+2);
		opind++;
		// second step: v = v-z, index domains
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[res]];
		fod[opind].right = &fod[opind-1];
		// second step: v = v-z, 
                arg_index[res] = opind++;	
#endif
#if defined(_NONLIND_OLD_)
                extend_nonlinearity_domain_binary(arg1, arg2, ind_dom, nonl_dom);
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres,  TAYLOR_BUFFER[res])
                ASSIGN_T(Targ1, TAYLOR_BUFFER[arg1])
                ASSIGN_T(Targ2, TAYLOR_BUFFER[arg2])

#ifdef _INT_FOR_
                FOR_0_LE_l_LT_p
                TRES_FOINC |= TARG2_INC | TARG1_INC;
#else
                /* olvo 980915 now in reverse order to allow x = x*x etc. */
                INC_pk_1(Tres)
                INC_pk_1(Targ1)
                INC_pk_1(Targ2)

                FOR_p_GT_l_GE_0
                FOR_k_GT_i_GE_0
                { TRES_FODEC -= dp_T0[arg1]*TARG2_DEC +
                                TARG1_DEC*dp_T0[arg2];
                  DEC_TRES_FO
#if defined(_HIGHER_ORDER_)
                  Targ1OP = Targ1-i+1;
                  Targ2OP = Targ2;

                  for (j=0;j<i;j++)
                  *Tres -= (*Targ1OP++) * (*Targ2OP--);
                  Tres--;
#endif /* _HIGHER_ORDER_ */
                }
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */

#if !defined(_NTIGHT_)
                dp_T0[res] -= dp_T0[arg1] *
                                                    dp_T0[arg2];
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case mult_d_a:         /* Multiply an adouble by a double    mult_d_a */
                /* (*) */
                arg   = get_locint_f();
                res   = get_locint_f();
#if !defined(_NTIGHT_)
                coval = 
#endif
		get_val_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res] = dp_T0[arg] * coval;
#endif /* !_NTIGHT_ */

#if defined(_INDO_)
#if defined(_INDOPRO_)
                copy_index_domain(res, arg, ind_dom);
#endif
#if defined(_NONLIND_)
                arg_index[res] = arg_index[arg];		
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])
                ASSIGN_T(Targ, TAYLOR_BUFFER[arg])

#ifdef _INT_FOR_
                FOR_0_LE_l_LT_pk
                TRES_INC = TARG_INC;
#else
                FOR_0_LE_l_LT_pk
                TRES_INC = TARG_INC * coval;
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;

                /*--------------------------------------------------------------------------*/
            case mult_a_p:         /* Multiply an adouble by a double    mult_a_p */
                /* (*) */
                arg   = get_locint_f();
                arg1  = get_locint_f();
                res   = get_locint_f();
#if !defined(_NTIGHT_)
                coval = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore[arg1];
#endif

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res] = dp_T0[arg] * coval;
#endif /* !_NTIGHT_ */

#if defined(_INDO_)
#if defined(_INDOPRO_)
                copy_index_domain(res, arg, ind_dom);
#endif
#if defined(_NONLIND_)
                arg_index[res] = arg_index[arg];		
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])
                ASSIGN_T(Targ, TAYLOR_BUFFER[arg])

#ifdef _INT_FOR_
                FOR_0_LE_l_LT_pk
                TRES_INC = TARG_INC;
#else
                FOR_0_LE_l_LT_pk
                TRES_INC = TARG_INC * coval;
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;

                /*--------------------------------------------------------------------------*/
            case div_a_a:           /* Divide an adouble by an adouble    div_a_a */
                /* (/) */
                arg1 = get_locint_f();
                arg2 = get_locint_f();
                res  = get_locint_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
#if !defined(_ZOS_) && !defined(_INT_FOR_) && !defined(_INDO_)
                divs = 1.0 / dp_T0[arg2];
#endif /* ALL_TOGETHER_AGAIN */

                dp_T0[res] = dp_T0[arg1] /
                                               dp_T0[arg2];
#endif /* !_NTIGHT_ */

#if defined(_INDO_)
#if defined(_INDOPRO_)
                combine_2_index_domains(res, arg1, arg2, ind_dom);
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[arg1]];
		fod[opind].right = &fod[arg_index[arg2]];
		traverse_unary(&fod[arg_index[arg1]], nonl_dom, &fod[arg_index[arg2]], indcheck+1,maxopind+2);
		traverse_unary(&fod[arg_index[arg2]], nonl_dom, &fod[opind], indcheck+1,maxopind+2);
                arg_index[res] = opind++;		
#endif 
#if defined(_NONLIND_OLD_)
                extend_nonlinearity_domain_binary(arg1, arg2, ind_dom, nonl_dom);
                extend_nonlinearity_domain_unary(arg2, ind_dom, nonl_dom);
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres,  TAYLOR_BUFFER[res])
                ASSIGN_T(Targ1, TAYLOR_BUFFER[arg1])
                ASSIGN_T(Targ2, TAYLOR_BUFFER[arg2])

#ifdef _INT_FOR_
                FOR_0_LE_l_LT_p
                TRES_FOINC = TARG1_INC | TARG2_FOINC;
#else
                FOR_0_LE_l_LT_p
                FOR_0_LE_i_LT_k
                { /* olvo 980922 changed order to allow x = y/x */
#if defined(_HIGHER_ORDER_)
                    zOP      = dp_z+i;
                    (*zOP--) = -(*Targ2) * divs;
#endif /* _HIGHER_ORDER_ */

                    TRES_FOINC = TARG1_INC * divs + dp_T0[res] *
                                 (-TARG2_INC * divs);

#if defined(_HIGHER_ORDER_)
                    TresOP = Tres-i;

                    for (j=0;j<i;j++)
                    *Tres += (*TresOP++) * (*zOP--);
                    Tres++;
#endif /* _HIGHER_ORDER_ */
                }
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;

            /*--------------------------------------------------------------------------*/
        case div_d_a:             /* Division double - adouble (/)    div_d_a */
            arg   = get_locint_f();
                res   = get_locint_f();
#if !defined(_NTIGHT_)
                coval = 
#endif
		get_val_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

                /* olvo 980922 necessary for reverse 
                   kk 20140508 I don't think so
                if (arg == res) {
                    IF_KEEP_WRITE_TAYLOR(arg,keep,k,p)
                } */

#if !defined(_NTIGHT_)
#if !defined(_ZOS_) && !defined(_INT_FOR_) && !defined(_INDO_)
                divs = 1.0 / dp_T0[arg];
#endif /* ALL_TOGETHER_AGAIN */

                dp_T0[res] = coval / dp_T0[arg];
#endif /* !_NTIGHT_ */

#if defined(_INDO_)
#if defined(_INDOPRO_)
                copy_index_domain(res, arg, ind_dom);
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[arg]];
		fod[opind].right = NULL;
		traverse_unary(&fod[opind], nonl_dom, &fod[opind], indcheck+1,maxopind+2);
                arg_index[res] = opind++;		
#endif 
#if defined(_NONLIND_OLD_)
                extend_nonlinearity_domain_unary(arg, ind_dom, nonl_dom);
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])
                ASSIGN_T(Targ, TAYLOR_BUFFER[arg])

#ifdef _INT_FOR_
                FOR_0_LE_l_LT_p
                TRES_FOINC = TARG_FOINC;
#else
                FOR_0_LE_l_LT_p
                FOR_0_LE_i_LT_k
                { /* olvo 980922 changed order to allow x = d/x */
#if defined(_HIGHER_ORDER_)
                    zOP      = dp_z+i;
                    (*zOP--) = -(*Targ) * divs;
#endif /* _HIGHER_ORDER_ */

                    TRES_FOINC = dp_T0[res] * (-TARG_INC * divs);

#if defined(_HIGHER_ORDER_)
                    TresOP = Tres-i;

                    for (j=0;j<i;j++)
                    *Tres += (*TresOP++) * (*zOP--);
                    Tres++;
#endif /* _HIGHER_ORDER_ */
                }
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;


            /****************************************************************************/
            /*--------------------------------------------------------------------------*/
        case div_p_a:             /* Division double - adouble (/)    div_p_a */
                arg   = get_locint_f();
                arg1  = get_locint_f();
                res   = get_locint_f();
#if !defined(_NTIGHT_)
                coval = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore[arg1];
#endif

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

                /* olvo 980922 necessary for reverse 
                   kk 20140508 I don't think so
                if (arg == res) {
                    IF_KEEP_WRITE_TAYLOR(arg,keep,k,p)
                } */

#if !defined(_NTIGHT_)
#if !defined(_ZOS_) && !defined(_INT_FOR_) && !defined(_INDO_)
                divs = 1.0 / dp_T0[arg];
#endif /* ALL_TOGETHER_AGAIN */

                dp_T0[res] = coval / dp_T0[arg];
#endif /* !_NTIGHT_ */

#if defined(_INDO_)
#if defined(_INDOPRO_)
                copy_index_domain(res, arg, ind_dom);
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[arg]];
		fod[opind].right = NULL;
		traverse_unary(&fod[opind], nonl_dom, &fod[opind], indcheck+1,maxopind+2);
                arg_index[res] = opind++;		
#endif 
#if defined(_NONLIND_OLD_)
                extend_nonlinearity_domain_unary(arg, ind_dom, nonl_dom);
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])
                ASSIGN_T(Targ, TAYLOR_BUFFER[arg])

#ifdef _INT_FOR_
                FOR_0_LE_l_LT_p
                TRES_FOINC = TARG_FOINC;
#else
                FOR_0_LE_l_LT_p
                FOR_0_LE_i_LT_k
                { /* olvo 980922 changed order to allow x = d/x */
#if defined(_HIGHER_ORDER_)
                    zOP      = dp_z+i;
                    (*zOP--) = -(*Targ) * divs;
#endif /* _HIGHER_ORDER_ */

                    TRES_FOINC = dp_T0[res] * (-TARG_INC * divs);

#if defined(_HIGHER_ORDER_)
                    TresOP = Tres-i;

                    for (j=0;j<i;j++)
                    *Tres += (*TresOP++) * (*zOP--);
                    Tres++;
#endif /* _HIGHER_ORDER_ */
                }
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;


            /****************************************************************************/
            /*                                                         SIGN  OPERATIONS */

            /*--------------------------------------------------------------------------*/
        case pos_sign_a:                                        /* pos_sign_a */
            arg   = get_locint_f();
                res   = get_locint_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res] = dp_T0[arg];
#endif /* !_NTIGHT_ */

#if defined(_INDO_)
#if defined(_INDOPRO_)
                copy_index_domain(res, arg, ind_dom);
#endif
#if defined(_NONLIND_)
                arg_index[res] = arg_index[arg];			    
#endif  
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])
                ASSIGN_T(Targ, TAYLOR_BUFFER[arg])

                FOR_0_LE_l_LT_pk
                TRES_INC = TARG_INC;
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;

                /*--------------------------------------------------------------------------*/
            case neg_sign_a:                                        /* neg_sign_a */
                arg   = get_locint_f();
                res   = get_locint_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res] = -dp_T0[arg];
#endif /* !_NTIGHT_ */

#if defined(_INDO_)
#if defined(_INDOPRO_)
                copy_index_domain(res, arg, ind_dom);
#endif
#if defined(_NONLIND_)
                arg_index[res] = arg_index[arg];			    
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])
                ASSIGN_T(Targ, TAYLOR_BUFFER[arg])

#ifdef _INT_FOR_
                FOR_0_LE_l_LT_pk
                TRES_INC = TARG_INC;
#else
                FOR_0_LE_l_LT_pk
                TRES_INC = -TARG_INC;
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;


                /****************************************************************************/
                /*                                                         UNARY OPERATIONS */

                /*--------------------------------------------------------------------------*/
            case exp_op:                          /* exponent operation    exp_op */
                arg = get_locint_f();
                res = get_locint_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res] = exp(dp_T0[arg]);
#endif /* !_NTIGHT_ */

                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;

#if defined(_INDO_)
#if defined(_INDOPRO_)
                copy_index_domain(res, arg, ind_dom);
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[arg]];
		fod[opind].right = NULL;
		traverse_unary(&fod[opind], nonl_dom, &fod[opind], indcheck+1,maxopind+2);
                arg_index[res] = opind++;		
#endif
#if defined(_NONLIND_OLD_)
                extend_nonlinearity_domain_unary(arg, ind_dom, nonl_dom);
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])
                ASSIGN_T(Targ, TAYLOR_BUFFER[arg])

#ifdef _INT_FOR_
                FOR_0_LE_l_LT_p
                TRES_FOINC = TARG_FOINC;
#else
                FOR_0_LE_l_LT_p
                FOR_0_LE_i_LT_k
                { /* olvo 980915 changed order to allow x = exp(x) */
#if defined(_HIGHER_ORDER_)
                    zOP      = dp_z+i;
                    (*zOP--) = (i+1) * (*Targ);
#endif /* _HIGHER_ORDER_ */

                    TRES_FOINC = dp_T0[res] * TARG_INC;

#if defined(_HIGHER_ORDER_)
                    TresOP = Tres-i;

                    *Tres *= (i+1);
                    for (j=0;j<i;j++)
                    *Tres += (*TresOP++) * (*zOP--);
                    *Tres++ /= (i+1); /* important only for i>0 */
#endif /* _HIGHER_ORDER_ */
                }

#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;

            /*--------------------------------------------------------------------------*/
        case sin_op:                              /* sine operation    sin_op */
                arg1 = get_locint_f();
                arg2 = get_locint_f();
                res  = get_locint_f();

                IF_KEEP_WRITE_TAYLOR(arg2,keep,k,p) /* olvo 980710 covalue */
                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                /* Note: always arg2 != arg1 */
                dp_T0[arg2] = cos(dp_T0[arg1]);
                dp_T0[res]  = sin(dp_T0[arg1]);
#endif /* !_NTIGHT_ */

                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;

#if defined(_INDO_)
#if defined(_INDOPRO_)
                copy_index_domain(res, arg1, ind_dom);
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[arg1]];
		fod[opind].right = NULL;
		traverse_unary(&fod[opind], nonl_dom, &fod[opind], indcheck+1,maxopind+2);
                arg_index[res] = opind++;		
#endif
#if defined(_NONLIND_OLD_)
                extend_nonlinearity_domain_unary(arg1, ind_dom, nonl_dom);
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres,  TAYLOR_BUFFER[res])
                ASSIGN_T(Targ1, TAYLOR_BUFFER[arg1])
                ASSIGN_T(Targ2, TAYLOR_BUFFER[arg2])

#ifdef _INT_FOR_
                FOR_0_LE_l_LT_p
                { /* olvo 980923 changed order to allow x = sin(x) */
                    TARG2_FOINC =  TARG1;
                    TRES_FOINC  =  TARG1_FOINC;
            }
#else
                FOR_0_LE_l_LT_p
                FOR_0_LE_i_LT_k
                { /* olvo 980921 changed order to allow x = sin(x) */
#if defined(_HIGHER_ORDER_)
                    zOP      = dp_z+i;
                    (*zOP--) = (i+1) * (*Targ1);
#endif /* _HIGHER_ORDER_ */

                    /* Note: always arg2 != arg1 */
                    TARG2_FOINC = -dp_T0[res]  * TARG1;
                    TRES_FOINC  =  dp_T0[arg2] * TARG1_INC;

#if defined(_HIGHER_ORDER_)
                    TresOP  = Tres-i;
                    Targ2OP = Targ2-i;

                    *Tres  *= (i+1);
                    *Targ2 *= (i+1);
                    for (j=0;j<i;j++) {
                    *Tres  += (*Targ2OP++) * (*zOP);
                        *Targ2 -= (*TresOP++)  * (*zOP--);
                    }
                    *Targ2++ /= (i+1);
                    *Tres++  /= (i+1);
#endif /* _HIGHER_ORDER_ */
            }
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;

                /*--------------------------------------------------------------------------*/
            case cos_op:                            /* cosine operation    cos_op */
                arg1 = get_locint_f();
                arg2 = get_locint_f();
                res  = get_locint_f();

                IF_KEEP_WRITE_TAYLOR(arg2,keep,k,p) /* olvo 980710 covalue */
                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                /* Note: always arg2 != arg1 */
                dp_T0[arg2] = sin(dp_T0[arg1]);
                dp_T0[res]  = cos(dp_T0[arg1]);
#endif /* !_NTIGHT_ */

                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;

#if defined(_INDO_)
#if defined(_INDOPRO_)
                copy_index_domain(res, arg1, ind_dom);
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[arg1]];
		fod[opind].right = NULL;
		traverse_unary(&fod[opind], nonl_dom, &fod[opind], indcheck+1,maxopind+2);
                arg_index[res] = opind++;		
#endif
#if defined(_NONLIND_OLD_)
                extend_nonlinearity_domain_unary(arg1, ind_dom, nonl_dom);
#endif

#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres,  TAYLOR_BUFFER[res])
                ASSIGN_T(Targ1, TAYLOR_BUFFER[arg1])
                ASSIGN_T(Targ2, TAYLOR_BUFFER[arg2])

#ifdef _INT_FOR_
                FOR_0_LE_l_LT_p
                { /* olvo 980923 changed order to allow x = cos(x) */
                    TARG2_FOINC = TARG1;
                    TRES_FOINC  = TARG1_FOINC;
            }
#else
                FOR_0_LE_l_LT_p
                FOR_0_LE_i_LT_k
                { /* olvo 980921 changed order to allow x = cos(x) */
#if defined(_HIGHER_ORDER_)
                    zOP      = dp_z+i;
                    (*zOP--) = (i+1) * (*Targ1);
#endif /* _HIGHER_ORDER_ */

                    /* Note: always arg2 != arg1 */
                    TARG2_FOINC =  dp_T0[res]  * TARG1;
                    TRES_FOINC  = -dp_T0[arg2] * TARG1_INC;

#if defined(_HIGHER_ORDER_)
                    TresOP  = Tres-i;
                    Targ2OP = Targ2-i;

                    *Tres  *= (i+1);
                    *Targ2 *= (i+1);
                    for (j=0;j<i;j++) {
                    *Tres  -= (*Targ2OP++) * (*zOP);
                        *Targ2 += (*TresOP++)  * (*zOP--);
                    }
                    *Targ2++ /= (i+1);
                    *Tres++  /= (i+1);
#endif /* _HIGHER_ORDER_ */
            }
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;

                /*--------------------------------------------------------------------------*/
            case atan_op:                                              /* atan_op */
                arg1 = get_locint_f();
                arg2 = get_locint_f();
                res  = get_locint_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res]=atan(dp_T0[arg1]);
#endif /* !_NTIGHT_ */

                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;

#if defined(_INDO_)
#if defined(_INDOPRO_)
                copy_index_domain(res, arg1, ind_dom);
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[arg1]];
		fod[opind].right = NULL;
		traverse_unary(&fod[opind], nonl_dom, &fod[opind], indcheck+1,maxopind+2);
                arg_index[res] = opind++;		
#endif
#if defined(_NONLIND_OLD_)
                extend_nonlinearity_domain_unary(arg1, ind_dom, nonl_dom);
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres,  TAYLOR_BUFFER[res])
                ASSIGN_T(Targ1, TAYLOR_BUFFER[arg1])
#ifdef _INT_FOR_
                FOR_0_LE_l_LT_p
                TRES_FOINC = TARG1_FOINC;
#else
                ASSIGN_T(Targ2, TAYLOR_BUFFER[arg2])

                FOR_0_LE_l_LT_p
                { FOR_0_LE_i_LT_k
                  { /* olvo 980921 changed order to allow x = atan(x) */
#if defined(_HIGHER_ORDER_)
                      zOP      = dp_z+i;
                      (*zOP--) = (i+1) * (*Targ1);
#endif /* _HIGHER_ORDER_ */

                      TRES_FOINC = dp_T0[arg2] * TARG1_INC;

#if defined(_HIGHER_ORDER_)
                      Targ2OP = Targ2;

                      *Tres *= (i+1);
                      for (j=0;j<i;j++)
                      *Tres  += (*Targ2OP++) * (*zOP--);
                      *Tres++ /= (i+1);
#endif /* _HIGHER_ORDER_ */
                  }
                  HOV_INC(Targ2, k)
                }
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;

            /*--------------------------------------------------------------------------*/
        case asin_op:                                              /* asin_op */
            arg1 = get_locint_f();
                arg2 = get_locint_f();
                res  = get_locint_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res] = asin(dp_T0[arg1]);
#endif /* !_NTIGHT_ */

                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;

#if defined(_INDO_)
#if defined(_INDOPRO_)
                copy_index_domain(res, arg1, ind_dom);
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[arg1]];
		fod[opind].right = NULL;
		traverse_unary(&fod[opind], nonl_dom, &fod[opind], indcheck+1,maxopind+2);
                arg_index[res] = opind++;		
#endif
#if defined(_NONLIND_OLD_)
                extend_nonlinearity_domain_unary(arg1, ind_dom, nonl_dom);
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres,  TAYLOR_BUFFER[res])
                ASSIGN_T(Targ1, TAYLOR_BUFFER[arg1])
#ifdef _INT_FOR_
                FOR_0_LE_l_LT_p
                TRES_FOINC = TARG1_FOINC;
#else
                ASSIGN_T(Targ2, TAYLOR_BUFFER[arg2])

                if (dp_T0[arg1] == 1.0)
                    FOR_0_LE_l_LT_p
                    { FOR_0_LE_i_LT_k
                      if (TARG1 > 0.0) {
                      r0 = make_nan();
                          VEC_INC(Targ1, k-i)
                          BREAK_FOR_I
                      } else
                          if (TARG1 < 0.0) {
                          r0 = make_inf();
                              VEC_INC(Targ1, k-i)
                              BREAK_FOR_I
                          } else {
                              r0 = 0.0;
                              Targ1++;
                          }
                  TRES = r0;
                  VEC_INC(Tres, k)
            } else
                    if (dp_T0[arg1] == -1.0)
                        FOR_0_LE_l_LT_p
                        { FOR_0_LE_i_LT_k
                          if (TARG1 > 0.0) {
                          r0 = make_inf();
                              VEC_INC(Targ1, k-i)
                              BREAK_FOR_I
                          } else
                              if (TARG1 < 0.0) {
                              r0 = make_nan();
                                  VEC_INC(Targ1, k-i)
                                  BREAK_FOR_I
                              } else {
                                  r0 = 0.0;
                                  Targ1++;
                              }
                  TRES = r0;
                  VEC_INC(Tres, k)
                } else
                        FOR_0_LE_l_LT_p {
                            FOR_0_LE_i_LT_k
                            { /* olvo 980921 changed order to allow x = asin(x) */
#if defined(_HIGHER_ORDER_)
                                zOP      = dp_z+i;
                                (*zOP--) = (i+1) * (*Targ1);
#endif /* _HIGHER_ORDER_ */

                                TRES_FOINC = dp_T0[arg2] * TARG1_INC;

#if defined(_HIGHER_ORDER_)
                                Targ2OP = Targ2;

                                *Tres *= (i+1);
                                for (j=0;j<i;j++)
                                *Tres += (*Targ2OP++) * (*zOP--);
                                *Tres++ /= (i+1);
#endif /* _HIGHER_ORDER_ */
                            }
                            HOV_INC(Targ2, k)
                        }
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
                        break;

            /*--------------------------------------------------------------------------*/
        case acos_op:                                              /* acos_op */
                arg1 = get_locint_f();
                arg2 = get_locint_f();
                res  = get_locint_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res] = acos(dp_T0[arg1]);
#endif /* !_NTIGHT_ */

                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;

#if defined(_INDO_)
#if defined(_INDOPRO_)
                copy_index_domain(res, arg1, ind_dom);
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[arg1]];
		fod[opind].right = NULL;
		traverse_unary(&fod[opind], nonl_dom, &fod[opind], indcheck+1,maxopind+2);
                arg_index[res] = opind++;		
#endif
#if defined(_NONLIND_OLD_)
                extend_nonlinearity_domain_unary(arg1, ind_dom, nonl_dom);
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres,  TAYLOR_BUFFER[res])
                ASSIGN_T(Targ1, TAYLOR_BUFFER[arg1])
#ifdef _INT_FOR_
                FOR_0_LE_l_LT_p
                TRES_FOINC = TARG1_FOINC;
#else
                ASSIGN_T(Targ2, TAYLOR_BUFFER[arg2])

                if (dp_T0[arg1] == 1.0)
                    FOR_0_LE_l_LT_p
                    { FOR_0_LE_i_LT_k
                      if (TARG1 > 0.0) {
                      r0 = make_nan();
                          VEC_INC(Targ1, k-i)
                          BREAK_FOR_I
                      } else
                          if (TARG1 < 0.0) {
                          r0 = -make_inf();
                              VEC_INC(Targ1, k-i)
                              BREAK_FOR_I
                          } else {
                              r0 = 0.0;
                              Targ1++;
                          }
                  TRES = r0;
                  VEC_INC(Tres, k)
            } else
                    if (dp_T0[arg1] == -1.0)
                        FOR_0_LE_l_LT_p
                        { FOR_0_LE_i_LT_k
                          if (TARG1 > 0.0) {
                          r0 = -make_inf();
                              VEC_INC(Targ1, k-i)
                              BREAK_FOR_I
                          } else
                              if (TARG1 < 0.0) {
                              r0 = make_nan();
                                  VEC_INC(Targ1, k-i)
                                  BREAK_FOR_I
                              } else {
                                  r0 = 0.0;
                                  Targ1++;
                              }
                  TRES = r0;
                  VEC_INC(Tres, k)
                } else
                        FOR_0_LE_l_LT_p {
                            FOR_0_LE_i_LT_k
                            { /* olvo 980921 changed order to allow x = acos(x) */
#if defined(_HIGHER_ORDER_)
                                zOP      = dp_z+i;
                                (*zOP--) = (i+1) * (*Targ1);
#endif /* _HIGHER_ORDER_ */

                                TRES_FOINC = dp_T0[arg2] * TARG1_INC;

#if defined(_HIGHER_ORDER_)
                                Targ2OP = Targ2;

                                *Tres *= (i+1);
                                for (j=0;j<i;j++)
                                *Tres += (*Targ2OP++) * (*zOP--);
                                *Tres++ /= (i+1);
#endif /* _HIGHER_ORDER_ */
                            }
                            HOV_INC(Targ2, k)
                        }
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
                        break;

#ifdef ATRIG_ERF

            /*--------------------------------------------------------------------------*/
        case asinh_op:                                            /* asinh_op */
                arg1 = get_locint_f();
                arg2 = get_locint_f();
                res  = get_locint_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res] = asinh(dp_T0[arg1]);
#endif /* !_NTIGHT_ */

                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;

#if defined(_INDO_)
#if defined(_INDOPRO_)
                copy_index_domain(res, arg1, ind_dom);
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[arg1]];
		fod[opind].right = NULL;
		traverse_unary(&fod[opind], nonl_dom, &fod[opind], indcheck+1,maxopind+2);
                arg_index[res] = opind++;		
#endif
#if defined(_NONLIND_OLD_)
                extend_nonlinearity_domain_unary(arg1, ind_dom, nonl_dom);
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres,  TAYLOR_BUFFER[res])
                ASSIGN_T(Targ1, TAYLOR_BUFFER[arg1])
#ifdef _INT_FOR_
                FOR_0_LE_l_LT_p
                TRES_FOINC = TARG1_FOINC;
#else
                ASSIGN_T(Targ2, TAYLOR_BUFFER[arg2])

                FOR_0_LE_l_LT_p
                { FOR_0_LE_i_LT_k
                  { /* olvo 980921 changed order to allow x = asinh(x) */
#if defined(_HIGHER_ORDER_)
                      zOP      = dp_z+i;
                      (*zOP--) = (i+1) * (*Targ1);
#endif /* _HIGHER_ORDER_ */

                      TRES_FOINC = dp_T0[arg2] * TARG1_INC;

#if defined(_HIGHER_ORDER_)
                      Targ2OP = Targ2;

                      *Tres *= (i+1);
                      for (j=0;j<i;j++)
                      *Tres += (*Targ2OP++) * (*zOP--);
                      *Tres++ /= (i+1);
#endif /* _HIGHER_ORDER_ */
                  }
                  HOV_INC(Targ2, k)
                }
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;

            /*--------------------------------------------------------------------------*/
        case acosh_op:                                           /* acosh_op */
                arg1 = get_locint_f();
                arg2 = get_locint_f();
                res  = get_locint_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res] = acosh(dp_T0[arg1]);
#endif /* !_NTIGHT_ */

                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;

#if defined(_INDO_)
#if defined(_INDOPRO_)
                copy_index_domain(res, arg1, ind_dom);
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[arg1]];
		fod[opind].right = NULL;
		traverse_unary(&fod[opind], nonl_dom, &fod[opind], indcheck+1,maxopind+2);
                arg_index[res] = opind++;		
#endif
#if defined(_NONLIND_OLD_)
                extend_nonlinearity_domain_unary(arg1, ind_dom, nonl_dom);
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres,  TAYLOR_BUFFER[res])
                ASSIGN_T(Targ1, TAYLOR_BUFFER[arg1])
#ifdef _INT_FOR_
                FOR_0_LE_l_LT_p
                TRES_FOINC = TARG1_FOINC;
#else
                ASSIGN_T(Targ2, TAYLOR_BUFFER[arg2])

                if (dp_T0[arg1] == 1.0)
                    FOR_0_LE_l_LT_p
                    { FOR_0_LE_i_LT_k
                      if (TARG1 > 0.0) {
                      r0 = make_inf();
                          VEC_INC(Targ1, k-i)
                          BREAK_FOR_I
                      } else
                          if (TARG1 < 0.0) {
                          r0 = make_nan();
                              VEC_INC(Targ1, k-i)
                              BREAK_FOR_I
                          } else {
                              r0 = 0.0;
                              Targ1++;
                          }
                  TRES_INC = r0;
#if defined(_HIGHER_ORDER_)
                  for (i=1;i<k;i++)
                  *Tres++ = make_nan();
#endif /* _HIGHER_ORDER_ */
                } else
                    FOR_0_LE_l_LT_p {
                        FOR_0_LE_i_LT_k
                        { /* olvo 980921 changed order to allow x = acosh(x) */
#if defined(_HIGHER_ORDER_)
                            zOP      = dp_z+i;
                            (*zOP--) = (i+1) * (*Targ1);
#endif /* _HIGHER_ORDER_ */

                            TRES_FOINC = dp_T0[arg2] * TARG1_INC;

#if defined(_HIGHER_ORDER_)
                            Targ2OP = Targ2;

                            *Tres *= (i+1);
                            for (j=0;j<i;j++)
                                *Tres += (*Targ2OP++) * (*zOP--);
                                *Tres++ /= (i+1);
#endif /* _HIGHER_ORDER_ */
                            }
                            HOV_INC(Targ2, k)
                        }
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
                        break;

            /*--------------------------------------------------------------------------*/
        case atanh_op:                                            /* atanh_op */
                arg1 = get_locint_f();
                arg2 = get_locint_f();
                res  = get_locint_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res] = atanh(dp_T0[arg1]);
#endif /* !_NTIGHT_ */

                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;

#if defined(_INDO_)
#if defined(_INDOPRO_)
                copy_index_domain(res, arg1, ind_dom);
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[arg1]];
		fod[opind].right = NULL;
		traverse_unary(&fod[opind], nonl_dom, &fod[opind], indcheck+1,maxopind+2);
                arg_index[res] = opind++;		
#endif
#if defined(_NONLIND_OLD_)
                extend_nonlinearity_domain_unary(arg1, ind_dom, nonl_dom);
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres,  TAYLOR_BUFFER[res])
                ASSIGN_T(Targ1, TAYLOR_BUFFER[arg1])
#ifdef _INT_FOR_
                FOR_0_LE_l_LT_p
                TRES_FOINC = TARG1_FOINC;
#else
                ASSIGN_T(Targ2, TAYLOR_BUFFER[arg2])

                if (dp_T0[arg1] == 1.0)
                    FOR_0_LE_l_LT_p
                    { FOR_0_LE_i_LT_k
                      if (TARG1 > 0.0) {
                      r0 = make_nan();
                          VEC_INC(Targ1, k-i)
                          BREAK_FOR_I
                      } else
                          if (TARG1 < 0.0) {
                          r0 = make_inf();
                              VEC_INC(Targ1, k-i)
                              BREAK_FOR_I
                          } else {
                              r0 = 0.0;
                              Targ1++;
                          }
                  TRES_INC = r0;
#if defined(_HIGHER_ORDER_)
                  for (i=1;i<k;i++)
                  *Tres++ = make_nan();
#endif /* _HIGHER_ORDER_ */
                } else
                    if (dp_T0[arg1] == -1.0)
                            FOR_0_LE_l_LT_p
                            { FOR_0_LE_i_LT_k
                              if (TARG1 > 0.0) {
                              r0 = make_inf();
                                  VEC_INC(Targ1, k-i)
                                  BREAK_FOR_I
                              } else
                                  if (TARG1 < 0.0) {
                                  r0 = make_nan();
                                      VEC_INC(Targ1, k-i)
                                      BREAK_FOR_I
                                  } else {
                                      r0 = 0.0;
                                      Targ1++;
                                  }
                  TRES_INC = r0;
#if defined(_HIGHER_ORDER_)
                  for (i=1;i<k;i++)
                  *Tres++ = make_nan();
#endif /* _HIGHER_ORDER_ */
                        } else
                            FOR_0_LE_l_LT_p {
                                FOR_0_LE_i_LT_k
                                { /* olvo 980921 changed order to allow x = atanh(x) */
#if defined(_HIGHER_ORDER_)
                                    zOP      = dp_z+i;
                                    (*zOP--) = (i+1) * (*Targ1);
#endif /* _HIGHER_ORDER_ */

                                    TRES_FOINC = dp_T0[arg2] * TARG1_INC;

#if defined(_HIGHER_ORDER_)
                                    Targ2OP = Targ2;

                                    *Tres *= (i+1);
                                    for (j=0;j<i;j++)
                                        *Tres += (*Targ2OP++) * (*zOP--);
                                        *Tres++ /= (i+1);
#endif /* _HIGHER_ORDER_ */
                                    }
                                    HOV_INC(Targ2, k)
                                }
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
                                break;

            /*--------------------------------------------------------------------------*/
        case erf_op:                                                /* erf_op */
                arg1 = get_locint_f();
                arg2 = get_locint_f();
                res  = get_locint_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res] = erf(dp_T0[arg1]);
#endif /* !_NTIGHT_ */

                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
#if defined(_INDO_)
#if defined(_INDOPRO_)
                copy_index_domain(res, arg1, ind_dom);
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[arg1]];
		fod[opind].right = NULL;
		traverse_unary(&fod[opind], nonl_dom, &fod[opind], indcheck+1,maxopind+2);
                arg_index[res] = opind++;		
#endif        
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])
                ASSIGN_T(Targ1,TAYLOR_BUFFER[arg1])
#ifdef _INT_FOR_
                FOR_0_LE_l_LT_p
                TRES_FOINC = TARG1_FOINC;
#else
                ASSIGN_T(Targ2,TAYLOR_BUFFER[arg2])

                FOR_0_LE_l_LT_p
                { FOR_0_LE_i_LT_k
                  { /* olvo 980921 changed order to allow x = erf(x) */
#if defined(_HIGHER_ORDER_)
                      zOP      = dp_z+i;
                      (*zOP--) = (i+1) * (*Targ1);
#endif /* _HIGHER_ORDER_ */

                      TRES_FOINC = dp_T0[arg2] * TARG1_INC;

#if defined(_HIGHER_ORDER_)
                      Targ2OP = Targ2;

                      *Tres *= (i+1);
                      for (j=0;j<i;j++)
                      *Tres += (*Targ2OP++) * (*zOP--);
                      *Tres++ /= (i+1);
#endif /* _HIGHER_ORDER_ */
                  }
                  HOV_INC(Targ2, k)
                }
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;

#endif

            /*--------------------------------------------------------------------------*/
        case log_op:                                                /* log_op */
                arg = get_locint_f();
                res = get_locint_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if defined(_INDO_)
#if defined(_INDOPRO_)
                copy_index_domain(res, arg, ind_dom);
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[arg]];
		fod[opind].right = NULL;
		traverse_unary(&fod[opind], nonl_dom, &fod[opind], indcheck+1,maxopind+2);
                arg_index[res] = opind++;		
#endif
#if defined(_NONLIND_OLD_)
                extend_nonlinearity_domain_unary(arg, ind_dom, nonl_dom);
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])
                ASSIGN_T(Targ, TAYLOR_BUFFER[arg])

#ifdef _INT_FOR_
                FOR_0_LE_l_LT_p
                TRES_FOINC = TARG_INC;
#else
                divs = 1.0 / dp_T0[arg];
                FOR_0_LE_l_LT_p
                { if (dp_T0[arg] == 0.0) {
                  TargOP = Targ;
                  FOR_0_LE_i_LT_k
                  { if (*TargOP++ < 0.0) {
                        divs = make_nan();
                            BREAK_FOR_I
                        }
                      }
                  }

                  /* olvo 980921 changed order to allow x = log(x) */
                  FOR_0_LE_i_LT_k
                  { TRES_FOINC = TARG_INC * divs;
#if defined(_HIGHER_ORDER_)
                    TresOP = Tres - i;
                    zOP    = dp_z+i;

                    (*zOP--) = *Tres;
                    (*Tres) *= i+1;
                    for (j=0;j<i;j++)
                    (*Tres) -= (*zOP--) * (*TresOP++) * (j+1);
                    *Tres++ /= i+1;
#endif /* _HIGHER_ORDER_ */
                  }
                }
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
#if !defined(_NTIGHT_)
                dp_T0[res] = log(dp_T0[arg]);
#endif /* !_NTIGHT_ */

                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
                break;

                /*--------------------------------------------------------------------------*/
            case pow_op:                                                /* pow_op */
                arg   = get_locint_f();
                res   = get_locint_f();

#if !defined(_NTIGHT_)
                coval = 
#endif
                get_val_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

                /* olvo 980921 necessary for reverse */
                if (arg == res) {
                    IF_KEEP_WRITE_TAYLOR(arg,keep,k,p)
                }

#if !defined(_NTIGHT_)

#ifndef _ZOS_ /* BREAK_ZOS */
#if !defined(_INT_FOR_) && !defined(_INDO_)
                T0arg   = dp_T0[arg];
#endif
#endif /* ALL_TOGETHER_AGAIN */

                dp_T0[res] =
                    pow(dp_T0[arg], coval);
#endif /* !_NTIGHT_ */

                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
#if defined(_INDO_)
#if defined(_INDOPRO_)
                copy_index_domain(res, arg, ind_dom);
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[arg]];
		fod[opind].right = NULL;
		traverse_unary(&fod[opind], nonl_dom, &fod[opind], indcheck+1,maxopind+2);
                arg_index[res] = opind++;		
#endif
#if defined(_NONLIND_OLD_)
                extend_nonlinearity_domain_unary(arg, ind_dom, nonl_dom);
#endif
#else
#ifndef _ZOS_ /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])
                ASSIGN_T(Targ, TAYLOR_BUFFER[arg])

#ifdef _INT_FOR_
                FOR_0_LE_l_LT_p
                TRES_FOINC = TARG_INC;
#else
                if (T0arg == 0.0) {
                    if (coval <= 0.0)
                        FOR_0_LE_l_LT_pk
                        TRES_INC = make_nan();
                    else {
                        /* coval not a whole number */
                        if (coval - floor(coval) != 0) {
                            FOR_0_LE_l_LT_p
                            {
                                i = 0;
                                FOR_0_LE_i_LT_k
                                {
                                    if (coval - i > 1)
                                    TRES_INC = 0;
                                    if ((coval - i < 1) && (coval - i > 0))
                                        TRES_INC = make_inf();
                                        if (coval - i < 0)
                                            TRES_INC = make_nan();
                                        }
                                    }
                                } else {
                        if (coval == 1) {
                                FOR_0_LE_l_LT_pk
                                TRES_INC = TARG_INC;
                            } else
                                /* coval is an int > 1 */
                                /* the following is not efficient but at least it works */
                                /* it reformulates x^n into x* ... *x n times */
                            {
                                INC_pk_1(Targ)
                                INC_pk_1(Tres)

                                FOR_p_GT_l_GE_0
                                {
                                    FOR_k_GT_i_GE_0
                                    {
                                        *Tres = 0;
                                        DEC_TRES_FO
#if defined(_HIGHER_ORDER_)
                                        if (i == k-1) {
                                        zOP = dp_z+k-1;
                                        for(j=k-1;j>=0;j--) {
                                                (*zOP--) = (*Targ--);
                                            }
                                        }
                                        for (j=0;j<i;j++) {
                                        *Tres += dp_z[j] *
                                                     dp_z[i-j-1];
                                        }
                                        Tres--;
#endif /* _HIGHER_ORDER_ */
                                    }
                                }
                                for(ii=3;ii<=coval;ii++) {
                                    ASSIGN_T(Tres, TAYLOR_BUFFER[res])
                                    ASSIGN_T(Targ, TAYLOR_BUFFER[arg])
                                    INC_pk_1(Targ)
                                    INC_pk_1(Tres)

                                    FOR_p_GT_l_GE_0
                                    {
                                        FOR_k_GT_i_GE_0
                                        {
                                            *Tres = 0;
                                            DEC_TRES_FO
#if defined(_HIGHER_ORDER_)
                                            TresOP = Tres-i;
                                            for (j=0;j<i;j++)
                                            *Tres += TresOP[j] * dp_z[i-j-1];
                                            Tres--;
#endif /* _HIGHER_ORDER_ */
                                        }
                                    }
                                }
                        }
                    }
                }
            } else {
                r0 = 1.0 / T0arg;
                FOR_0_LE_l_LT_p
                FOR_0_LE_i_LT_k
                { /* olvo 980921 changed order to allow x = pow(x,n) */
#ifdef _HIGHER_ORDER_
                    zOP      = dp_z+i;
                    (*zOP--) = (*Targ) * r0;
#endif /* _HIGHER_ORDER_ */

                    TRES_FOINC = dp_T0[res] *
                                 TARG_INC * coval * r0;

#ifdef _HIGHER_ORDER_
                    TresOP = Tres-i;

                    (*Tres) *= i+1;
                    y = coval*i -1;
                    for (j=0;j<i;j++) {
                        *Tres += (*TresOP++) * (*zOP--) * y;
                            y -= coval + 1;
                        }
                        *Tres++ /= (i+1);
#endif /* _HIGHER_ORDER_ */
                    }
                }
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;

                /*--------------------------------------------------------------------------*/
            case pow_op_p:                                                /* pow_op_p */
                arg   = get_locint_f();
                arg1  = get_locint_f();
                res   = get_locint_f();
#if !defined(_NTIGHT_)
                coval = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore[arg1];
#endif

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

                /* olvo 980921 necessary for reverse */
                if (arg == res) {
                    IF_KEEP_WRITE_TAYLOR(arg,keep,k,p)
                }

#if !defined(_NTIGHT_)

#ifndef _ZOS_ /* BREAK_ZOS */
#if !defined(_INT_FOR_) && !defined(_INDO_)
                T0arg   = dp_T0[arg];
#endif
#endif /* ALL_TOGETHER_AGAIN */

                dp_T0[res] =
                    pow(dp_T0[arg], coval);
#endif /* !_NTIGHT_ */

                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
#if defined(_INDO_)
#if defined(_INDOPRO_)
                copy_index_domain(res, arg, ind_dom);
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[arg]];
		fod[opind].right = NULL;
		traverse_unary(&fod[opind], nonl_dom, &fod[opind], indcheck+1,maxopind+2);
                arg_index[res] = opind++;		
#endif
#if defined(_NONLIND_OLD_)
                extend_nonlinearity_domain_unary(arg, ind_dom, nonl_dom);
#endif
#else
#ifndef _ZOS_ /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])
                ASSIGN_T(Targ, TAYLOR_BUFFER[arg])

#ifdef _INT_FOR_
                FOR_0_LE_l_LT_p
                TRES_FOINC = TARG_INC;
#else
                if (T0arg == 0.0) {
                    if (coval <= 0.0)
                        FOR_0_LE_l_LT_pk
                        TRES_INC = make_nan();
                    else {
                        /* coval not a whole number */
                        if (coval - floor(coval) != 0) {
                            FOR_0_LE_l_LT_p
                            {
                                i = 0;
                                FOR_0_LE_i_LT_k
                                {
                                    if (coval - i > 1)
                                    TRES_INC = 0;
                                    if ((coval - i < 1) && (coval - i > 0))
                                        TRES_INC = make_inf();
                                        if (coval - i < 0)
                                            TRES_INC = make_nan();
                                        }
                                    }
                                } else {
                        if (coval == 1) {
                                FOR_0_LE_l_LT_pk
                                TRES_INC = TARG_INC;
                            } else
                                /* coval is an int > 1 */
                                /* the following is not efficient but at least it works */
                                /* it reformulates x^n into x* ... *x n times */
                            {
                                INC_pk_1(Targ)
                                INC_pk_1(Tres)

                                FOR_p_GT_l_GE_0
                                {
                                    FOR_k_GT_i_GE_0
                                    {
                                        *Tres = 0;
                                        DEC_TRES_FO
#if defined(_HIGHER_ORDER_)
                                        if (i == k-1) {
                                        zOP = dp_z+k-1;
                                        for(j=k-1;j>=0;j--) {
                                                (*zOP--) = (*Targ--);
                                            }
                                        }
                                        for (j=0;j<i;j++) {
                                        *Tres += dp_z[j] *
                                                     dp_z[i-j-1];
                                        }
                                        Tres--;
#endif /* _HIGHER_ORDER_ */
                                    }
                                }
                                for(ii=3;ii<=coval;ii++) {
                                    ASSIGN_T(Tres, TAYLOR_BUFFER[res])
                                    ASSIGN_T(Targ, TAYLOR_BUFFER[arg])
                                    INC_pk_1(Targ)
                                    INC_pk_1(Tres)

                                    FOR_p_GT_l_GE_0
                                    {
                                        FOR_k_GT_i_GE_0
                                        {
                                            *Tres = 0;
                                            DEC_TRES_FO
#if defined(_HIGHER_ORDER_)
                                            TresOP = Tres-i;
                                            for (j=0;j<i;j++)
                                            *Tres += TresOP[j] * dp_z[i-j-1];
                                            Tres--;
#endif /* _HIGHER_ORDER_ */
                                        }
                                    }
                                }
                        }
                    }
                }
            } else {
                r0 = 1.0 / T0arg;
                FOR_0_LE_l_LT_p
                FOR_0_LE_i_LT_k
                { /* olvo 980921 changed order to allow x = pow(x,n) */
#ifdef _HIGHER_ORDER_
                    zOP      = dp_z+i;
                    (*zOP--) = (*Targ) * r0;
#endif /* _HIGHER_ORDER_ */

                    TRES_FOINC = dp_T0[res] *
                                 TARG_INC * coval * r0;

#ifdef _HIGHER_ORDER_
                    TresOP = Tres-i;

                    (*Tres) *= i+1;
                    y = coval*i -1;
                    for (j=0;j<i;j++) {
                        *Tres += (*TresOP++) * (*zOP--) * y;
                            y -= coval + 1;
                        }
                        *Tres++ /= (i+1);
#endif /* _HIGHER_ORDER_ */
                    }
                }
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;

                /*--------------------------------------------------------------------------*/
            case sqrt_op:                                              /* sqrt_op */
                arg = get_locint_f();
                res = get_locint_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res] = sqrt(dp_T0[arg]);
#endif /* !_NTIGHT_ */

                ADOLC_OPENMP_RESTORE_THREAD_NUMBER;

#if defined(_INDO_)
#if defined(_INDOPRO_)
                copy_index_domain(res, arg, ind_dom);
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[arg]];
		fod[opind].right = NULL;
		traverse_unary(&fod[opind], nonl_dom, &fod[opind], indcheck+1,maxopind+2);
                arg_index[res] = opind++;		
#endif
#if defined(_NONLIND_OLD_)
                extend_nonlinearity_domain_unary(arg, ind_dom, nonl_dom);
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Targ, TAYLOR_BUFFER[arg])
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])

#ifdef _INT_FOR_
                FOR_0_LE_l_LT_p
                TRES_FOINC = TARG_INC;
#else
                FOR_0_LE_l_LT_p
                { TargOP = Targ;
                  if (dp_T0[arg] == 0.0)
                  /* Note: <=> dp_T0[res] == 0.0 */
              { r0 = 0.0;
                  FOR_0_LE_i_LT_k
                  { if (TARG>0.0) {
                        r0 = make_inf();
                            VEC_INC(Targ, k-i)
                            BREAK_FOR_I
                        } else
                            if (TARG<0.0) {
                            r0 = make_nan();
                                VEC_INC(Targ, k-i)
                                BREAK_FOR_I
                            } else
                                Targ++;
                              }
                          }
                  else {
                      r0 = 0.5/dp_T0[res];
                  }
                  Targ = TargOP;

#if defined(_HIGHER_ORDER_)
                  even = 1;
#endif
                  FOR_0_LE_i_LT_k
                  { TRES_FOINC = r0 * TARG_INC;
#if defined(_HIGHER_ORDER_)
                    TresOP  = Tres-i;
                    TresOP2 = Tres-1;

                    x = 0;
                    for (j=1;2*j-1<i;j++)
                    x += (*TresOP++) * (*TresOP2--);
                    x *= 2;
                    if (!even)
                        x += (*TresOP) * (*TresOP2); /* !!! */
                        even = !even;
                        *Tres++ -= r0*x;
#endif /* _HIGHER_ORDER_ */
                      }
                    }
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
                    break;

            /*--------------------------------------------------------------------------*/
        case gen_quad:                                            /* gen_quad */
            arg1 = get_locint_f();
                arg2 = get_locint_f();
                res  = get_locint_f();

#if !defined(_NTIGHT_)
                if (get_val_f()!=dp_T0[arg1]) {
                    fprintf(DIAG_OUT,
                            "ADOL-C Warning: forward sweep aborted; tape invalid!\n");
                    IF_KEEP_TAYLOR_CLOSE
                    end_sweep();
                    return -2;
                }
#endif /* !_NTIGHT_ */

#if !defined(_NTIGHT_)
                coval =
#endif 
		get_val_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res] = coval;
#endif /* !_NTIGHT_ */

#if defined(_INDO_)
               fprintf(DIAG_OUT,
                    "ADOL-C Warning: forward sweep aborted; sparse mode not available for gen_quad!\n");
               end_sweep();
               return -2;
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres,  TAYLOR_BUFFER[res])
                ASSIGN_T(Targ1, TAYLOR_BUFFER[arg1])
#ifdef _INT_FOR_
                FOR_0_LE_l_LT_p
                TRES_FOINC = TARG1_FOINC;
#else
                ASSIGN_T(Targ2, TAYLOR_BUFFER[arg2])

                FOR_0_LE_l_LT_p
                { FOR_0_LE_i_LT_k
                  { /* olvo 980922 changed order to allow x = gen_quad(x) */
#if defined(_HIGHER_ORDER_)
                      zOP      = dp_z+i;
                      (*zOP--) = (i+1) * (*Targ1);
#endif /* _HIGHER_ORDER_ */

                      TRES_FOINC = dp_T0[arg2] * TARG1_INC;

#if defined(_HIGHER_ORDER_)
                      Targ2OP = Targ2;

                      *Tres *= (i+1);
                      for (j=0;j<i;j++)
                      *Tres += (*Targ2OP++) * (*zOP--);
                      *Tres++ /= (i+1);
#endif /* _HIGHER_ORDER_ */
                  }
                  HOV_INC(Targ2, k)
                }
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;

            /*--------------------------------------------------------------------------*/
        case min_op:                                                /* min_op */
            arg1  = get_locint_f();
                arg2  = get_locint_f();
                res   = get_locint_f();
#if !defined(_NTIGHT_)
                coval = 
#endif
		get_val_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                /* olvo 980923 changed order to allow x = min(x,y) etc. */

                /* olvo/mitev 980721 return value (taken from below) */
                if (dp_T0[arg1] > dp_T0[arg2]) {
                    if (coval)
                        MINDEC(ret_c,2);
                } else
                    if (dp_T0[arg1] < dp_T0[arg2]) {
                        if (!coval)
                            MINDEC(ret_c,2);
                    } else
                        if (arg1 != arg2)
                            MINDEC(ret_c,1);
#endif /* !_NTIGHT_ */

#if defined (_INDO_)
#if defined (_INDOPRO_)
#if defined (_TIGHT_)
                    if (dp_T0[arg1] < dp_T0[arg2])
                        copy_index_domain(res, arg1, ind_dom);
                    else {
                        if (dp_T0[arg1] > dp_T0[arg2])
                            copy_index_domain(res, arg2, ind_dom);
                        else
                            combine_2_index_domains(res, arg1, arg2, ind_dom);
                    }
#else
                    combine_2_index_domains(res, arg1, arg2, ind_dom);
#endif
#endif
#if defined(_NONLIND_)
#ifdef _TIGHT_
                    if (dp_T0[arg1] < dp_T0[arg2])
		      {
			fod[opind].entry = maxopind+2;
			fod[opind].left = &fod[arg_index[arg1]];
			fod[opind].right = NULL;
			arg_index[res] = opind++;		
		      }		   
                    else {
                        if (dp_T0[arg1] > dp_T0[arg2])
			  {
			    fod[opind].entry = maxopind+2;
			    fod[opind].left = &fod[arg_index[arg2]];
			    fod[opind].right = NULL;
			    arg_index[res] = opind++;		

			  }		   
                        else
			  {
			    fod[opind].entry = maxopind+2;
			    fod[opind].left = &fod[arg_index[arg1]];
			    fod[opind].right = &fod[arg_index[arg2]];
			    arg_index[res] = opind++;		
			  }
                    }
#else
		    fod[opind].entry = maxopind+2;
		    fod[opind].left = &fod[arg_index[arg1]];
		    fod[opind].right = &fod[arg_index[arg2]];
		    arg_index[res] = opind++;		
		    arg_index[res] = opind++;		
#endif
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Targ1, TAYLOR_BUFFER[arg1])
                ASSIGN_T(Targ2, TAYLOR_BUFFER[arg2])
                ASSIGN_T(Tres,  TAYLOR_BUFFER[res])

#ifdef _INT_FOR_
#ifdef _TIGHT_
                Tqo = NULL;
                if (dp_T0[arg1] > dp_T0[arg2])
                    Tqo = Targ2;
                else
                    if (dp_T0[arg1] < dp_T0[arg2])
                        Tqo = Targ1;

                FOR_0_LE_l_LT_p
                { Targ = Tqo;
                  if (Targ == NULL) /* e.g. T0[arg1] == T0[arg2] */
                    { Targ1OP = Targ1;
                      Targ2OP = Targ2;
                      if (TARG1 > TARG2)
                          Targ = Targ2OP;
                      else
                          if (TARG1 < TARG2)
                              Targ = Targ1OP;
                      Targ1++;
                      Targ2++;
                      if (Targ == NULL) /* e.g. both are equal */
                          Targ = Targ1OP;
                  }

                  TRES_INC = TARG_INC;

                  if (Tqo)
                    Tqo++;
                }

                dp_T0[res] = MIN_ADOLC(dp_T0[arg1], dp_T0[arg2]);
#endif /* _TIGHT_ */
#ifdef _NTIGHT_
                TRES_INC = TARG1_INC | TARG2_INC;
#endif /* _NTIGHT_ */
#else
                Tqo = NULL;
                if (dp_T0[arg1] > dp_T0[arg2])
                    Tqo = Targ2;
                else
                    if (dp_T0[arg1] < dp_T0[arg2])
                        Tqo = Targ1;

                FOR_0_LE_l_LT_p
                { Targ = Tqo;
                  if (Targ == NULL) /* e.g. dp_T0[arg1] ==
                                                                                 dp_T0[arg2] */
              { Targ1OP = Targ1;
                  Targ2OP = Targ2;
                  FOR_0_LE_i_LT_k
                  { if (TARG1 > TARG2) {
                        Targ = Targ2OP;
                        VEC_INC(Targ1, k-i)
                            VEC_INC(Targ2, k-i)
                            BREAK_FOR_I
                        } else
                            if (TARG1 < TARG2) {
                            Targ = Targ1OP;
                            VEC_INC(Targ1, k-i)
                                VEC_INC(Targ2, k-i)
                                BREAK_FOR_I
                            }
                        Targ1++;
                        Targ2++;
                      }
                      if (Targ == NULL) /* e.g. both are equal */
                          Targ = Targ1OP;
                  }

                  FOR_0_LE_i_LT_k
                  TRES_INC = TARG_INC;

                  if (Tqo) {
                  VEC_INC(Tqo, k)
                  }
            }
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
#if !defined(_NTIGHT_)
                dp_T0[res] =
                    MIN_ADOLC( dp_T0[arg1],
                               dp_T0[arg2] );
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case abs_val:                                              /* abs_val */
                arg   = get_locint_f();
                res   = get_locint_f();
#if !defined(_NTIGHT_)
                coval = 
#endif
		get_val_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                /* olvo 980923 changed order to allow x = min(x,y) etc. */

                /* olvo/mitev 980721 ec n3l (taken from below) */
                if (dp_T0[arg] < 0.0) {
                    if (coval)
                        MINDEC(ret_c,2);
                } else
                    if (dp_T0[arg] > 0.0) {
                        if (!coval)
                            MINDEC(ret_c,2);
                    }
		if (ADOLC_CURRENT_TAPE_INFOS.stats[NO_MIN_MAX]) {
		    signature[switchnum] = dp_T0[arg];
#if defined(_ABS_NORM_) || defined(_ABS_NORM_SIG_)
		    swargs[switchnum] = dp_T0[arg];
#endif
		}
#endif /* !_NTIGHT_ */

#if defined(_INDO_)
#if defined(_INDOPRO_)
#if defined(_ABS_NORM_)
                if (ind_dom[arg][0] != 0) {
                    crs[switchnum] = (unsigned int*) malloc(sizeof(unsigned int)* (ind_dom[arg][0]+1));
                    crs[switchnum][0] = ind_dom[arg][0];
                    for(l=1;l<crs[switchnum][0];i++) {
                        crs[switchnum][l] = ind_dom[arg][l+1];
                    }
                } else {
                    crs[switchnum] = (unsigned int*) malloc(sizeof(unsigned int));
                    crs[switchnum][0] = 0;
                }
                ind_dom[res][0] = 1;
                ind_dom[res][2] = indcheck+switchnum;
#else
		copy_index_domain(res, arg, ind_dom);
#endif
#endif
#if defined(_NONLIND_)
                arg_index[res] = arg_index[arg];		
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])
                ASSIGN_T(Targ, TAYLOR_BUFFER[arg])

#ifdef _INT_FOR_
#ifdef _TIGHT_
                y = 0.0;
                if (dp_T0[arg] != 0.0) {
                    if (dp_T0[arg] < 0.0)
                        y = -1.0;
                    else
                        y = 1.0;
                }
                FOR_0_LE_l_LT_p
                { if ((y == 0.0) && (TARG != 0.0))
                  MINDEC(ret_c,1);

                  TRES_INC = TARG_INC;
                }

                dp_T0[res] = fabs(dp_T0[arg]);
#endif /* _TIGHT_ */
#ifdef _NTIGHT_
                FOR_0_LE_l_LT_p
                TRES_INC = TARG_INC;
#endif /* _NTIGHT_ */
#else
#ifdef _ABS_NORM_
		sig = FIRSTSIGN_P(dp_T0[arg],Targ);
#if defined(_FOV_)
		sigsw[switchnum] = sig;
#endif
		COPYTAYL_P(swtaylors[switchnum],Targ);
		FOR_0_LE_l_LT_p
		    TRES_INC = fabs(dp_T0[arg]+TARG_INC)-fabs(dp_T0[arg]);
#elif defined(_ABS_NORM_SIG_)
                if (sigdir == NULL)
                    y = EXT_FIRSTSIGN2_P(sigbase[switchnum],dp_T0[arg],Targ);
                else
                    y = EXT_FIRSTSIGN_P(sigbase[switchnum],sigdir[switchnum],dp_T0[arg],Targ);
		COPYTAYL_P(swtaylors[switchnum],Targ);
                sigsw[switchnum] = y;
		FOR_0_LE_l_LT_p
		    TRES_INC = y * TARG_INC;
#else
                y = 0.0;
                if (dp_T0[arg] != 0.0) {
                    if (dp_T0[arg] < 0.0)
                        y = -1.0;
                    else
                        y = 1.0;
                }

                FOR_0_LE_l_LT_p
                { x = y;
                  FOR_0_LE_i_LT_k
                  { if ((x == 0.0) && (TARG != 0.0)) {
                    MINDEC(ret_c,1);
                        if (TARG < 0.0)
                            x = -1.0;
                        else
                            x = 1.0;
                    }
                    TRES_INC = x * TARG_INC;
		  }
		}
#endif
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
#if !defined(_NTIGHT_)
                dp_T0[res] = fabs(dp_T0[arg]);
#endif /* !_NTIGHT_ */
		if (ADOLC_CURRENT_TAPE_INFOS.stats[NO_MIN_MAX])
		    switchnum++;
                break;

                /*--------------------------------------------------------------------------*/
            case ceil_op:                                              /* ceil_op */
                arg   = get_locint_f();
                res   = get_locint_f();
#if !defined(_NTIGHT_)
                coval = 
#endif
		get_val_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res]=ceil(dp_T0[arg]);
                /* olvo/mitev 980721 ec n2l (taken from below) */
                if (coval != dp_T0[res])
                    MINDEC(ret_c,2);
#endif /* !_NTIGHT_ */

#if defined(_INDO_)
#if defined(_INDOPRO_)
#ifdef _TIGHT_
                ind_dom[res][0] = 0;
#else
		copy_index_domain(res, arg, ind_dom);
#endif /* _TIGHT_ */
#endif
#if defined(_NONLIND_)
                arg_index[res] = arg_index[arg];		
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])

                FOR_0_LE_l_LT_pk
                TRES_INC = 0.0;
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;

                /*--------------------------------------------------------------------------*/
            case floor_op:                 /* Compute ceil of adouble    floor_op */
                arg   = get_locint_f();
                res   = get_locint_f();
#if !defined(_NTIGHT_)
                coval = 
#endif
		get_val_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

#if !defined(_NTIGHT_)
                dp_T0[res] = floor(dp_T0[arg]);
                /* olvo/mitev 980721 ec n2l (taken from below) */
                if (coval != dp_T0[res])
                    MINDEC(ret_c,2);
#endif /* !_NTIGHT_ */

#if defined(_INDO_)
#if defined(_INDOPRO_)
#ifdef _TIGHT_
                ind_dom[res][0] = 0;
#else
		copy_index_domain(res, arg, ind_dom);
#endif /* _TIGHT_ */
#endif
#if defined(_NONLIND_)
                arg_index[res] = arg_index[arg];		
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])

                FOR_0_LE_l_LT_pk
                TRES_INC = 0.0;
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;


                /****************************************************************************/
                /*                                                             CONDITIONALS */

                /*--------------------------------------------------------------------------*/
            case cond_assign:                                      /* cond_assign */
                arg   = get_locint_f();
                arg1  = get_locint_f();
                arg2  = get_locint_f();
                res   = get_locint_f();
#if !defined(_NTIGHT_)
                coval = 
#endif
		get_val_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

                /* olvo 980924 changed order to allow reflexive ops */
#if defined (_INDO_)
#if defined (_INDOPRO_)
#if defined (_TIGHT_)
                if (dp_T0[arg] > 0) {
                    if (coval <= 0.0)
                        MINDEC(ret_c,2);
                    dp_T0[res] = dp_T0[arg1];

		    copy_index_domain(res, arg1, ind_dom);

                } else {
                    if (coval > 0.0)
                        MINDEC(ret_c,2);
                    if (dp_T0[arg] == 0)
                        MINDEC(ret_c,0);
                    dp_T0[res] = dp_T0[arg2];
		    copy_index_domain(res, arg2, ind_dom);
                }
#else
		    combine_2_index_domains(res, arg1, arg2, ind_dom);
#endif
#endif
#if defined (_NONLIND_)
#ifdef _TIGHT_
                if (dp_T0[arg] > 0) {
                    if (coval <= 0.0)
                        MINDEC(ret_c,2);
                    dp_T0[res] = dp_T0[arg1];

		    arg_index[res] = arg_index[arg1];		
                } else {
                    if (coval > 0.0)
                        MINDEC(ret_c,2);
                    if (dp_T0[arg] == 0)
                        MINDEC(ret_c,0);
                    dp_T0[res] = dp_T0[arg2];

		    arg_index[res] = arg_index[arg2];		
                }

#else
               arg_index[res] = opind++;		
#endif
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres,  TAYLOR_BUFFER[res])
                ASSIGN_T(Targ1, TAYLOR_BUFFER[arg1])
                ASSIGN_T(Targ2, TAYLOR_BUFFER[arg2])
#endif /* ALL_TOGETHER_AGAIN */

#ifdef _INT_FOR_
#ifdef _TIGHT_
                coval = get_val_f();

                if (dp_T0[arg] > 0)
                    FOR_0_LE_l_LT_pk
                    TRES_INC = TARG1_INC;
                else
                    FOR_0_LE_l_LT_pk
                    TRES_INC = TARG2_INC;

                if (dp_T0[arg] > 0) {
                    if (coval <= 0.0)
                        MINDEC(ret_c,2);
                    dp_T0[res] = dp_T0[arg1];
                } else {
                    if (coval > 0.0)
                        MINDEC(ret_c,2);
                    if (dp_T0[arg] == 0)
                        MINDEC(ret_c,0);
                    dp_T0[res] = dp_T0[arg2];
                }
#endif /* _TIGHT_ */
#ifdef _NTIGHT_
                FOR_0_LE_l_LT_pk
                TRES_INC = TARG1_INC | TARG2_INC;
#endif /* _NTIGHT_ */
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                if (dp_T0[arg] > 0)
                    FOR_0_LE_l_LT_pk
                    TRES_INC = TARG1_INC;
                else
                    FOR_0_LE_l_LT_pk
                    TRES_INC = TARG2_INC;
#endif

                if (dp_T0[arg] > 0) {
                    if (coval <= 0.0)
                        MINDEC(ret_c,2);
                    dp_T0[res] = dp_T0[arg1];
                } else {
                    if (coval > 0.0)
                        MINDEC(ret_c,2);
                    if (dp_T0[arg] == 0)
                        MINDEC(ret_c,0);
                    dp_T0[res] = dp_T0[arg2];
                }
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;

            case cond_eq_assign:                                      /* cond_eq_assign */
                arg   = get_locint_f();
                arg1  = get_locint_f();
                arg2  = get_locint_f();
                res   = get_locint_f();
#if !defined(_NTIGHT_)
                coval = 
#endif
		get_val_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

                /* olvo 980924 changed order to allow reflexive ops */
#if defined (_INDO_)
#if defined (_INDOPRO_)
#if defined (_TIGHT_)
                if (dp_T0[arg] >= 0) {
                    if (coval < 0.0)
                        MINDEC(ret_c,2);
                    dp_T0[res] = dp_T0[arg1];

		    copy_index_domain(res, arg1, ind_dom);

                } else {
                    if (coval >= 0.0)
                        MINDEC(ret_c,2);
                    dp_T0[res] = dp_T0[arg2];
		    copy_index_domain(res, arg2, ind_dom);
                }
#else
		    combine_2_index_domains(res, arg1, arg2, ind_dom);
#endif
#endif
#if defined (_NONLIND_)
#ifdef _TIGHT_
                if (dp_T0[arg] >= 0) {
                    if (coval < 0.0)
                        MINDEC(ret_c,2);
                    dp_T0[res] = dp_T0[arg1];

		    arg_index[res] = arg_index[arg1];		
                } else {
                    if (coval >= 0.0)
                        MINDEC(ret_c,2);
                    dp_T0[res] = dp_T0[arg2];

		    arg_index[res] = arg_index[arg2];		
                }

#else
               arg_index[res] = opind++;		
#endif
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres,  TAYLOR_BUFFER[res])
                ASSIGN_T(Targ1, TAYLOR_BUFFER[arg1])
                ASSIGN_T(Targ2, TAYLOR_BUFFER[arg2])
#endif /* ALL_TOGETHER_AGAIN */

#ifdef _INT_FOR_
#ifdef _TIGHT_
                coval = get_val_f();

                if (dp_T0[arg] >= 0)
                    FOR_0_LE_l_LT_pk
                    TRES_INC = TARG1_INC;
                else
                    FOR_0_LE_l_LT_pk
                    TRES_INC = TARG2_INC;

                if (dp_T0[arg] >= 0) {
                    if (coval < 0.0)
                        MINDEC(ret_c,2);
                    dp_T0[res] = dp_T0[arg1];
                } else {
                    if (coval >= 0.0)
                        MINDEC(ret_c,2);
                    dp_T0[res] = dp_T0[arg2];
                }
#endif /* _TIGHT_ */
#ifdef _NTIGHT_
                FOR_0_LE_l_LT_pk
                TRES_INC = TARG1_INC | TARG2_INC;
#endif /* _NTIGHT_ */
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                if (dp_T0[arg] >= 0)
                    FOR_0_LE_l_LT_pk
                    TRES_INC = TARG1_INC;
                else
                    FOR_0_LE_l_LT_pk
                    TRES_INC = TARG2_INC;
#endif

                if (dp_T0[arg] >= 0) {
                    if (coval < 0.0)
                        MINDEC(ret_c,2);
                    dp_T0[res] = dp_T0[arg1];
                } else {
                    if (coval >= 0.0)
                        MINDEC(ret_c,2);
                    dp_T0[res] = dp_T0[arg2];
                }
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;

                /*--------------------------------------------------------------------------*/
            case cond_assign_s:                                  /* cond_assign_s */
                arg   = get_locint_f();
                arg1  = get_locint_f();
                res   = get_locint_f();
#if !defined(_NTIGHT_)
                coval = 
#endif
		get_val_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

                /* olvo 980924 changed order to allow reflexive ops */
#if defined(_INDO_)
#if defined(_INDOPRO_)
#ifdef _TIGHT_
		if (dp_T0[arg] > 0)
		    copy_index_domain(res, arg1, ind_dom);
#else
		merge_2_index_domains(res, arg1, ind_dom);
#endif
#endif
#if defined(_NONLIND_)
                arg_index[res] = arg_index[arg1];		
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres,  TAYLOR_BUFFER[res])
                ASSIGN_T(Targ1, TAYLOR_BUFFER[arg1])
#endif /* ALL_TOGETHER_AGAIN */

#ifdef _INT_FOR_
#ifdef _TIGHT_
                coval = get_val_f();

                if (dp_T0[arg] > 0)
#endif /* _TIGHT_ */
                    FOR_0_LE_l_LT_pk
                    TRES_INC = TARG1_INC;

#ifdef _TIGHT_
                if (dp_T0[arg] > 0) {
                    if (coval <= 0.0)
                        MINDEC(ret_c,2);
                    dp_T0[res] = dp_T0[arg1];
                } else
                    if (dp_T0[arg] == 0)
                        MINDEC(ret_c,0);
#endif /* _TIGHT_ */
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                if (dp_T0[arg] > 0)
                    FOR_0_LE_l_LT_pk
                    TRES_INC = TARG1_INC;
#endif
                if (dp_T0[arg] > 0) {
                    if (coval <= 0.0)
                        MINDEC(ret_c,2);
                    dp_T0[res] = dp_T0[arg1];
                } else
                    if (dp_T0[arg] == 0)
                        MINDEC(ret_c,0);
#endif
#endif /* ALL_TOGETHER_AGAIN */
                break;


            case cond_eq_assign_s:                                  /* cond_eq_assign_s */
                arg   = get_locint_f();
                arg1  = get_locint_f();
                res   = get_locint_f();
#if !defined(_NTIGHT_)
                coval = 
#endif
		get_val_f();

                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

                /* olvo 980924 changed order to allow reflexive ops */
#if defined(_INDO_)
#if defined(_INDOPRO_)
#ifdef _TIGHT_
		if (dp_T0[arg] >= 0)
		    copy_index_domain(res, arg1, ind_dom);
#else
		merge_2_index_domains(res, arg1, ind_dom);
#endif
#endif
#if defined(_NONLIND_)
                arg_index[res] = arg_index[arg1];		
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres,  TAYLOR_BUFFER[res])
                ASSIGN_T(Targ1, TAYLOR_BUFFER[arg1])
#endif /* ALL_TOGETHER_AGAIN */

#ifdef _INT_FOR_
#ifdef _TIGHT_
                coval = get_val_f();

                if (dp_T0[arg] >= 0)
#endif /* _TIGHT_ */
                    FOR_0_LE_l_LT_pk
                    TRES_INC = TARG1_INC;

#ifdef _TIGHT_
                if (dp_T0[arg] >= 0) {
                    if (coval < 0.0)
                        MINDEC(ret_c,2);
                    dp_T0[res] = dp_T0[arg1];
                }
#endif /* _TIGHT_ */
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                if (dp_T0[arg] >= 0)
                    FOR_0_LE_l_LT_pk
                    TRES_INC = TARG1_INC;
#endif
                if (dp_T0[arg] >= 0) {
                    if (coval < 0.0)
                        MINDEC(ret_c,2);
                    dp_T0[res] = dp_T0[arg1];
                }
#endif
#endif /* ALL_TOGETHER_AGAIN */
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
#if !defined(_NTIGHT_)
		coval = 
#endif
		get_val_f();
		arg = get_locint_f();
		arg1 = get_locint_f();
		res = get_locint_f();
#if !defined(_NTIGHT_)
		{
		    revreal retval = -1;
		    const char* opname = "";
		    switch (operation) {
		    case neq_a_a:
			retval = (revreal)(dp_T0[arg] != dp_T0[arg1]);
			opname = "neq_a_a";
			break;
		    case eq_a_a:
			retval = (revreal)(dp_T0[arg] == dp_T0[arg1]);
			opname = "eq_a_a";
			break;
		    case ge_a_a:
			retval = (revreal)(dp_T0[arg] >= dp_T0[arg1]);
			opname = "ge_a_a";
			break;
		    case le_a_a:
			retval = (revreal)(dp_T0[arg] <= dp_T0[arg1]);
			opname = "le_a_a";
			break;
		    case gt_a_a:
			retval = (revreal)(dp_T0[arg] > dp_T0[arg1]);
			opname = "gt_a_a";
			break;
		    case lt_a_a:
			retval = (revreal)(dp_T0[arg] < dp_T0[arg1]);
			opname = "lt_a_a";
			break;
		    case neq_a_p:
			retval = (revreal)(dp_T0[arg] != ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore[arg1]);
			opname = "neq_a_p";
			break;
		    case eq_a_p:
			retval = (revreal)(dp_T0[arg] == ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore[arg1]);
			opname = "eq_a_p";
			break;
		    case ge_a_p:
			retval = (revreal)(dp_T0[arg] >= ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore[arg1]);
			opname = "ge_a_p";
			break;
		    case le_a_p:
			retval = (revreal)(dp_T0[arg] <= ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore[arg1]);
			opname = "le_a_p";
			break;
		    case gt_a_p:
			retval = (revreal)(dp_T0[arg] > ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore[arg1]);
			opname = "gt_a_p";
			break;
		    case lt_a_p:
			retval = (revreal)(dp_T0[arg] < ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore[arg1]);
			opname = "lt_a_p";
			break;
		    }
		    if (retval != coval && ADOLC_GLOBAL_TAPE_VARS.branchSwitchWarning)
			fprintf(DIAG_OUT,
				"ADOL-C Warning: Branch switch detected in comparison "
				"(operator %s).\n"
				"Results may be unpredictable! Retaping recommended!\n",opname);
		    IF_KEEP_WRITE_TAYLOR(res,keep,k,p);
		    dp_T0[res] = retval;
		}
#endif
#if defined(_INDO_)
#if defined(_INDOPRO_)
		ind_dom[res][0]=0;
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = NULL;
		fod[opind].right = NULL;
                arg_index[res] = opind++;
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])

                FOR_0_LE_l_LT_pk
                TRES_INC = 0;
#endif
#endif /* ALL_TOGETHER_AGAIN */

		break;
#endif /* ADVANCED_BRANCHING */

                /*--------------------------------------------------------------------------*/
            case subscript:
#if !defined(_NTIGHT_)
		coval = 
#endif
		get_val_f();
		arg = get_locint_f();
		{
#if !defined(_NTIGHT_)
		    size_t idx, numvar = (size_t)trunc(fabs(coval));
		    locint vectorloc;
		    vectorloc = 
#endif
		    get_locint_f();
                    res = get_locint_f();
#if !defined(_NTIGHT_)
		    idx = (size_t)trunc(fabs(dp_T0[arg]));
		    if (idx >= numvar)
			fprintf(DIAG_OUT, "ADOL-C warning: index out of bounds while subscripting n=%zu, idx=%zu\n", numvar, idx);
		    arg1 = vectorloc+idx;
		    IF_KEEP_WRITE_TAYLOR(res,keep,k,p);
		    dp_T0[res] = dp_T0[arg1];
#if defined(_INDO_)
#if defined(_INDOPRO_)
		    copy_index_domain(res, arg1, ind_dom);
#endif
#if defined(_NONLIND_)
		    arg_index[res] = arg_index[arg1];
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
		    ASSIGN_T(Targ1,TAYLOR_BUFFER[arg1])
		    ASSIGN_T(Tres,TAYLOR_BUFFER[res])

		    FOR_0_LE_l_LT_pk
		    TRES_INC = TARG1_INC;
#endif
#endif
#else
		    fprintf(DIAG_OUT, "ADOL-C error: active subscripting does not work in safe mode, please use tight mode\n");
		    adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* ALL_TOGETHER_AGAIN */
		}
		break;

            case subscript_ref:
#if !defined(_NTIGHT_)
		coval = 
#endif
		get_val_f();
		arg = get_locint_f();
		{
#if !defined(_NTIGHT_)
		    size_t idx, numvar = (size_t)trunc(fabs(coval));
		    locint vectorloc;
		    vectorloc = 
#endif
		    get_locint_f();
                    res = get_locint_f();
#if !defined(_NTIGHT_)
		    idx = (size_t)trunc(fabs(dp_T0[arg]));
		    if (idx >= numvar)
			fprintf(DIAG_OUT, "ADOL-C warning: index out of bounds while subscripting (ref) n=%zu, idx=%zu\n", numvar, idx);
		    arg1 = vectorloc+idx;
		    IF_KEEP_WRITE_TAYLOR(res,keep,k,p);
		    dp_T0[res] = arg1;
#else
		    fprintf(DIAG_OUT, "ADOL-C error: active subscripting does not work in safe mode, please use tight mode\n");
		    adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif
		}
		break;

            case ref_copyout:
		arg = get_locint_f();
		res = get_locint_f();
#if !defined(_NTIGHT_)
		arg1 = (size_t)trunc(fabs(dp_T0[arg]));
		IF_KEEP_WRITE_TAYLOR(res,keep,k,p);
		dp_T0[res] = dp_T0[arg1];
#if defined(_INDO_)
#if defined(_INDOPRO_)
		copy_index_domain(res, arg1, ind_dom);
#endif
#if defined(_NONLIND_)
		arg_index[res] = arg_index[arg1];
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
		ASSIGN_T(Targ1,TAYLOR_BUFFER[arg1])
		ASSIGN_T(Tres,TAYLOR_BUFFER[res])

		FOR_0_LE_l_LT_pk
		TRES_INC = TARG1_INC;
#endif
#endif
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* ALL_TOGETHER_AGAIN */
		break;

            case ref_incr_a:
		arg = get_locint_f();
#if !defined(_NTIGHT_)
		arg1 = (size_t)trunc(fabs(dp_T0[arg]));
		IF_KEEP_WRITE_TAYLOR(arg1,keep,k,p);
		dp_T0[arg1]++;
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif
		break;

            case ref_decr_a:
		arg = get_locint_f();
#if !defined(_NTIGHT_)
		arg1 = (size_t)trunc(fabs(dp_T0[arg]));
		IF_KEEP_WRITE_TAYLOR(arg1,keep,k,p);
		dp_T0[arg1]--;
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif
		break;

            case ref_assign_d:
		arg = get_locint_f();
#if !defined(_NTIGHT_)
		coval = 
#endif
		get_val_f();
		
#if !defined(_NTIGHT_)
		arg1 = (size_t)trunc(fabs(dp_T0[arg]));
		IF_KEEP_WRITE_TAYLOR(arg1,keep,k,p)
		dp_T0[arg1] = coval;
#if defined(_INDO_)
#if defined(_INDOPRO_)
		ind_dom[arg1][0] = 0;
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = NULL;
		fod[opind].right = NULL;
                arg_index[arg1] = opind++;
#endif
#else
#if !defined(_ZOS_)
		ASSIGN_T(Targ1, TAYLOR_BUFFER[arg1])

		FOR_0_LE_l_LT_pk
		TARG1_INC = 0;
#endif
#endif
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif
		break;

            case ref_assign_p:
		arg = get_locint_f();
                res = get_locint_f();
#if !defined(_NTIGHT_)
                coval = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore[res];
#endif
		
#if !defined(_NTIGHT_)
		arg1 = (size_t)trunc(fabs(dp_T0[arg]));
		IF_KEEP_WRITE_TAYLOR(arg1,keep,k,p)
		dp_T0[arg1] = coval;
#if defined(_INDO_)
#if defined(_INDOPRO_)
		ind_dom[arg1][0] = 0;
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = NULL;
		fod[opind].right = NULL;
                arg_index[arg1] = opind++;
#endif
#else
#if !defined(_ZOS_)
		ASSIGN_T(Targ1, TAYLOR_BUFFER[arg1])

		FOR_0_LE_l_LT_pk
		TARG1_INC = 0;
#endif
#endif
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif
		break;

            case ref_assign_d_zero:
		arg = get_locint_f();

#if !defined(_NTIGHT_)
		arg1 = (size_t)trunc(fabs(dp_T0[arg]));
		IF_KEEP_WRITE_TAYLOR(arg1,keep,k,p)
		dp_T0[arg1] = 0.0;
#if defined(_INDO_)
#if defined(_INDOPRO_)
		ind_dom[arg1][0] = 0;
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = NULL;
		fod[opind].right = NULL;
                arg_index[arg1] = opind++;
#endif
#else
#if !defined(_ZOS_)
		ASSIGN_T(Targ1, TAYLOR_BUFFER[arg1])

		FOR_0_LE_l_LT_pk
		TARG1_INC = 0;
#endif
#endif
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif
		break;

            case ref_assign_d_one:
		arg = get_locint_f();

#if !defined(_NTIGHT_)
		arg1 = (size_t)trunc(fabs(dp_T0[arg]));
		IF_KEEP_WRITE_TAYLOR(arg1,keep,k,p)
		dp_T0[arg1] = 1.0;
#if defined(_INDO_)
#if defined(_INDOPRO_)
		ind_dom[arg1][0] = 0;
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = NULL;
		fod[opind].right = NULL;
                arg_index[arg1] = opind++;
#endif
#else
#if !defined(_ZOS_)
		ASSIGN_T(Targ1, TAYLOR_BUFFER[arg1])

		FOR_0_LE_l_LT_pk
		TARG1_INC = 0;
#endif
#endif
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif
		break;

            case ref_assign_a:           /* assign an adouble variable an    assign_a */
                /* adouble value. (=) */
                arg = get_locint_f();
                res = get_locint_f();

#if !defined(_NTIGHT_)
		arg1 = (size_t)trunc(fabs(dp_T0[res]));
                IF_KEEP_WRITE_TAYLOR(arg1,keep,k,p)
                dp_T0[arg1] = dp_T0[arg];
#if defined(_INDO_)
#if defined(_INDOPRO_)
                copy_index_domain(arg1, arg, ind_dom);
#endif
#if defined(_NONLIND_)
                arg_index[arg1] = arg_index[arg];
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Targ,TAYLOR_BUFFER[arg])
                ASSIGN_T(Targ1,TAYLOR_BUFFER[arg1])

                FOR_0_LE_l_LT_pk
                TARG1_INC = TARG_INC;
#endif
#endif 
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* ALL_TOGETHER_AGAIN */
                break;

            case ref_assign_ind:       /* assign an adouble variable an    assign_ind */
                /* independent double value (<<=) */
                arg = get_locint_f();


#if !defined(_NTIGHT_)
		res = (size_t)trunc(fabs(dp_T0[arg]));
                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)
                dp_T0[res] = basepoint[indexi];
#if defined(_INDO_)
#if defined(_INDOPRO_)
		ind_dom[res][0] = 1;
		ind_dom[res][2] = indexi;
#endif
#if defined(_NONLIND_)
		fod[opind].entry = indexi;
		fod[opind].left = NULL;
		fod[opind].right = NULL;
                arg_index[res] = opind++;
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])

#ifdef _INT_FOR_
                FOR_0_LE_l_LT_p
                TRES_INC = ARGUMENT(indexi,l,i);
#else
                FOR_0_LE_l_LT_p
                FOR_0_LE_i_LT_k
                TRES_INC = ARGUMENT(indexi,l,i);
#endif
#endif
#endif
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* ALL_TOGETHER_AGAIN */
                ++indexi;
                break;

            case ref_eq_plus_d:            /* Add a floating point to an    eq_plus_d */
                /* adouble. (+=) */
                arg  = get_locint_f();
#if !defined(_NTIGHT_)
                coval = 
#endif
		get_val_f();


#if !defined(_NTIGHT_)
		res = (size_t)trunc(fabs(dp_T0[arg]));
                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)
                dp_T0[res] += coval;
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case ref_eq_plus_p:            /* Add a floating point to an    eq_plus_p */
                /* adouble. (+=) */
                arg1  = get_locint_f();
                arg  = get_locint_f();
#if !defined(_NTIGHT_)
                coval = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore[arg1];
#endif

#if !defined(_NTIGHT_)
		res = (size_t)trunc(fabs(dp_T0[arg]));
                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)
                dp_T0[res] += coval;
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case ref_eq_plus_a:             /* Add an adouble to another    eq_plus_a */
                /* adouble. (+=) */
                arg = get_locint_f();
                arg1 = get_locint_f();

#if !defined(_NTIGHT_)
		res = (size_t)trunc(fabs(dp_T0[arg1]));
                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)
                dp_T0[res] += dp_T0[arg];
#if defined(_INDO_)
#if defined(_INDOPRO_)
                merge_2_index_domains(res, arg, ind_dom);
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[res]];
		fod[opind].right = &fod[arg_index[arg]];
                arg_index[res] = opind++;
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])
                ASSIGN_T(Targ, TAYLOR_BUFFER[arg])

#ifdef _INT_FOR_
                FOR_0_LE_l_LT_pk
                TRES_INC |= TARG_INC;
#else
                FOR_0_LE_l_LT_pk
                TRES_INC += TARG_INC;
#endif
#endif
#endif
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* ALL_TOGETHER_AGAIN */
                break;

            case ref_eq_min_d:       /* Subtract a floating point from an    eq_min_d */
                /* adouble. (-=) */
                arg = get_locint_f();
#if !defined(_NTIGHT_)
                coval = 
#endif
		get_val_f();

#if !defined(_NTIGHT_)
		res = (size_t)trunc(fabs(dp_T0[arg]));
                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)
                dp_T0[res] -= coval;
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* !_NTIGHT_ */
                break;

            case ref_eq_min_p:       /* Subtract a floating point from an    eq_min_p */
                /* adouble. (-=) */
                arg1 = get_locint_f();
                arg = get_locint_f();
#if !defined(_NTIGHT_)
                coval = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore[arg1];
#endif

#if !defined(_NTIGHT_)
		res = (size_t)trunc(fabs(dp_T0[arg]));
                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)
                dp_T0[res] -= coval;
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* !_NTIGHT_ */
                break;

                /*--------------------------------------------------------------------------*/
            case ref_eq_min_a:        /* Subtract an adouble from another    eq_min_a */
                /* adouble. (-=) */
                arg = get_locint_f();
                arg1 = get_locint_f();

#if !defined(_NTIGHT_)
		res = (size_t)trunc(fabs(dp_T0[arg1]));
                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)
                dp_T0[res] -= dp_T0[arg];
#if defined(_INDO_)
#if defined(_INDOPRO_)
                merge_2_index_domains(res, arg, ind_dom);
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[res]];
		fod[opind].right = &fod[arg_index[arg]];
                arg_index[res] = opind++;
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])
                ASSIGN_T(Targ, TAYLOR_BUFFER[arg])

#ifdef _INT_FOR_
                FOR_0_LE_l_LT_pk
                TRES_INC |= TARG_INC;
#else
                FOR_0_LE_l_LT_pk
                TRES_INC -= TARG_INC;
#endif
#endif
#endif
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* ALL_TOGETHER_AGAIN */
                break;

            case ref_eq_mult_d:              /* Multiply an adouble by a    eq_mult_d */
                /* flaoting point. (*=) */
                arg = get_locint_f();
#if !defined(_NTIGHT_)
                coval = 
#endif
		get_val_f();

#if !defined(_NTIGHT_)
		res = (size_t)trunc(fabs(dp_T0[arg]));
                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)
                dp_T0[res] *= coval;
#if !defined(_INDO_)
#if !defined(_ZOS_) /* BREAK_ZOS */
#if !defined( _INT_FOR_)

                FOR_0_LE_l_LT_pk
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])

                FOR_0_LE_l_LT_pk
                TRES_INC *= coval;
#endif
#endif
#endif
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* ALL_TOGETHER_AGAIN */
                break;

            case ref_eq_mult_p:              /* Multiply an adouble by a    eq_mult_p */
                /* flaoting point. (*=) */
                arg1 = get_locint_f();
                arg = get_locint_f();
#if !defined(_NTIGHT_)
                coval = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore[arg1];
#endif

#if !defined(_NTIGHT_)
		res = (size_t)trunc(fabs(dp_T0[arg]));
                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)
                dp_T0[res] *= coval;
#if !defined(_INDO_)
#if !defined(_ZOS_) /* BREAK_ZOS */
#if !defined( _INT_FOR_)

                FOR_0_LE_l_LT_pk
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])

                FOR_0_LE_l_LT_pk
                TRES_INC *= coval;
#endif
#endif
#endif
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* ALL_TOGETHER_AGAIN */
                break;

            case ref_eq_mult_a:       /* Multiply one adouble by another    eq_mult_a */
                /* (*=) */
                arg = get_locint_f();
                arg1 = get_locint_f();

#if !defined(_NTIGHT_)
		res = (size_t)trunc(fabs(dp_T0[arg1]));
                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)
#if defined(_INDO_)
#if defined(_INDOPRO_)
                merge_2_index_domains(res, arg, ind_dom);
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[res]];
		fod[opind].right = &fod[arg_index[arg]];
		traverse_unary(&fod[arg_index[res]], nonl_dom, &fod[arg_index[arg]], indcheck+1,maxopind+2);
		traverse_unary(&fod[arg_index[arg]], nonl_dom, &fod[arg_index[res]], indcheck+1,maxopind+2);
                arg_index[res] = opind++;
#endif
#if defined(_NONLIND_OLD_)
                extend_nonlinearity_domain_binary(res, arg, ind_dom, nonl_dom);
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])
                ASSIGN_T(Targ, TAYLOR_BUFFER[arg])

                INC_pk_1(Tres)
                INC_pk_1(Targ)

#ifdef _INT_FOR_
                FOR_p_GT_l_GE_0
                TRES_FODEC |= TARG_DEC;
#else
                FOR_p_GT_l_GE_0
                FOR_k_GT_i_GE_0
                { TRES_FODEC = dp_T0[res]*TARG_DEC +
                               TRES*dp_T0[arg];
                  DEC_TRES_FO
#ifdef _HIGHER_ORDER_
                  TresOP = Tres-i;
                  TargOP = Targ;

                  for (j=0;j<i;j++)
                  *Tres += (*TresOP++) * (*TargOP--);
                  Tres--;
#endif /* _HIGHER_ORDER_ */
                }
#endif
#endif
#endif
		dp_T0[res] *= dp_T0[arg];
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* !_NTIGHT_ */
                break;

            case vec_copy:
                arg = get_locint_f();
                size = get_locint_f();
                res = get_locint_f();

                for(qq=0;qq<size;qq++) {
                    IF_KEEP_WRITE_TAYLOR(res+qq,keep,k,p);
#if !defined(_NTIGHT_)
                    dp_T0[res+qq] = dp_T0[arg+qq];
#endif /* !_NTIGHT_ */

#if defined(_INDO_)
#if defined(_INDOPRO_)
                    copy_index_domain(res+qq, arg+qq, ind_dom);
#endif
#if defined(_NONLIND_)
                    arg_index[res+qq] = arg_index[arg+qq];
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                    ASSIGN_T(Targ,TAYLOR_BUFFER[arg+qq])
                    ASSIGN_T(Tres,TAYLOR_BUFFER[res+qq])

                    FOR_0_LE_l_LT_pk
                    TRES_INC = TARG_INC;
#endif
#endif /* ALL_TOGETHER_AGAIN */
                }

                break;

            case vec_dot:
                arg1 = get_locint_f();
                arg2 = get_locint_f();
                size = get_locint_f();
                res = get_locint_f();
                IF_KEEP_WRITE_TAYLOR(res,keep,k,p);
#if !defined(_NTIGHT_)
                dp_T0[res] = 0;
#endif
#if defined(_INDO_)
#if defined(_INDOPRO_)
		ind_dom[res][0]=0;
#endif
#if defined(_NONLIND_)
		fod[opind].entry = maxopind+2;
		fod[opind].left = NULL;
		fod[opind].right = NULL;
                arg_index[res] = opind++;		
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres, TAYLOR_BUFFER[res])

                FOR_0_LE_l_LT_pk
                TRES_INC = 0;
#endif
#endif /* ALL_TOGETHER_AGAIN */
                for(qq=0;qq<size;qq++) {
#if defined(_INDO_)
#if defined(_INDOPRO_)
                merge_3_index_domains(res, arg1+qq, arg2+qq, ind_dom);
#endif
#if defined(_NONLIND_)
		// operation: v = v+u*w
		// first step: z = u*w, index domains
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[arg1+qq]];
		fod[opind].right = &fod[arg_index[arg2+qq]];
		// first step: z = u*w,
		traverse_unary(&fod[arg_index[arg1+qq]], nonl_dom, &fod[opind], indcheck+1,maxopind+2);
		traverse_unary(&fod[arg_index[arg2+qq]], nonl_dom, &fod[opind], indcheck+1,maxopind+2);
		opind++;
		// second step: v = v+z, index domains
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[res]];
		fod[opind].right = &fod[opind-1];
		// second step: v = v+z,
                arg_index[res] = opind++;
#endif
#if defined(_NONLIND_OLD_)
                extend_nonlinearity_domain_binary(arg1+qq, arg2+qq, ind_dom, nonl_dom);
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres,  TAYLOR_BUFFER[res])
                ASSIGN_T(Targ1, TAYLOR_BUFFER[arg1+qq])
                ASSIGN_T(Targ2, TAYLOR_BUFFER[arg2+qq])

#ifdef _INT_FOR_
                FOR_0_LE_l_LT_p
                TRES_FOINC |= TARG2_INC | TARG1_INC;
#else
                /* olvo 980915 now in reverse order to allow x = x*x etc. */
                INC_pk_1(Tres)
                INC_pk_1(Targ1)
                INC_pk_1(Targ2)

                FOR_p_GT_l_GE_0
                FOR_k_GT_i_GE_0
                { TRES_FODEC += dp_T0[arg1+qq]*TARG2_DEC +
                                TARG1_DEC*dp_T0[arg2+qq];
                  DEC_TRES_FO
#if defined(_HIGHER_ORDER_)
                  Targ1OP = Targ1-i+1;
                  Targ2OP = Targ2;

                  for (j=0;j<i;j++)
                  *Tres += (*Targ1OP++) * (*Targ2OP--);
                  Tres--;
#endif /* _HIGHER_ORDER_ */
                }
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
#if !defined(_NTIGHT_)
                dp_T0[res] += dp_T0[arg1+qq] *  dp_T0[arg2+qq];
#endif /* !_NTIGHT_ */
                }
                break;

            case vec_axpy:
                arg = get_locint_f();
                arg1 = get_locint_f();
                arg2 = get_locint_f();
                size = get_locint_f();
                res = get_locint_f();
                
                for(qq=0;qq<size;qq++) {
                    IF_KEEP_WRITE_TAYLOR(res+qq,keep,k,p);
#if defined(_INDO_)
#if defined(_INDOPRO_)
                combine_2_index_domains(res+qq, arg, arg1+qq, ind_dom);
                merge_2_index_domains(res+qq, arg2+qq, ind_dom);
#endif
#if defined(_NONLIND_)
		// operation: v = u*w + y
		// first step: z = u*w, index domains
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[arg_index[arg]];
		fod[opind].right = &fod[arg_index[arg1+qq]];
		// first step: z = u*w,
		traverse_unary(&fod[arg_index[arg]], nonl_dom, &fod[opind], indcheck+1,maxopind+2);
		traverse_unary(&fod[arg_index[arg1+qq]], nonl_dom, &fod[opind], indcheck+1,maxopind+2);
                opind++;
		// second step: v = z+y, index domains
		fod[opind].entry = maxopind+2;
		fod[opind].left = &fod[opind-1];
		fod[opind].right = &fod[arg_index[arg2+qq]];
		// second step: v = v+z,
                arg_index[res+qq] = opind++;
#endif
#if defined(_NONLIND_OLD_)
		extend_nonlinearity_domain_binary(arg, arg1+qq, ind_dom, nonl_dom);
#endif
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Targ, TAYLOR_BUFFER[arg])
                ASSIGN_T(Tres,  TAYLOR_BUFFER[res+qq])
                ASSIGN_T(Targ1, TAYLOR_BUFFER[arg1+qq])
                ASSIGN_T(Targ2, TAYLOR_BUFFER[arg2+qq])
#ifdef _INT_FOR_
                FOR_0_LE_l_LT_p
                TRES_FOINC = TARG2_INC | TARG1_INC | TARG_INC;
#else
                INC_pk_1(Tres)
                INC_pk_1(Targ)
                INC_pk_1(Targ1)
                INC_pk_1(Targ2)
                FOR_p_GT_l_GE_0
                FOR_k_GT_i_GE_0
                {  
                    TRES_FODEC = dp_T0[arg] * TARG1_DEC + TARG_DEC * dp_T0[arg1+qq] + TARG2_DEC;
                    DEC_TRES_FO
#if defined(_HIGHER_ORDER_)
                    Targ1OP = Targ - i + 1;
                    Targ2OP = Targ1;

                    for (j=0;j<i;j++) {
                        *Tres += (*Targ1OP++) * (*Targ2OP--);
                    }
                    Tres--;
#endif /* _HIGHER_ORDER_ */
                }
#endif
#endif
#endif /* ALL_TOGETHER_AGAIN */
#if !defined(_NTIGHT_)
                dp_T0[res+qq] = dp_T0[arg] * dp_T0[arg1+qq] + dp_T0[arg2+qq];
#endif /* !_NTIGHT_ */

                }
                break;

            case ref_cond_assign:                                      /* cond_assign */
                arg   = get_locint_f();
                arg1  = get_locint_f();
                arg2  = get_locint_f();
		{ 
#if !defined(_NTIGHT_)
		    locint ref = 
#endif
		    get_locint_f();
#if !defined(_NTIGHT_)
		    coval = 
#endif
		    get_val_f();
#if !defined(_NTIGHT_)
		    res   = (size_t)trunc(fabs(dp_T0[ref]));

		    IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

                /* olvo 980924 changed order to allow reflexive ops */
#if defined(_INDO_)
                if (dp_T0[arg] > 0) {
                    if (coval <= 0.0)
                        MINDEC(ret_c,2);
                    dp_T0[res] = dp_T0[arg1];

#if defined(_INDOPRO_)
		    copy_index_domain(res, arg1, ind_dom);
#endif
#if defined(_NONLIND_)
		    arg_index[res] = arg_index[arg1];
#endif
                } else {
                    if (coval > 0.0)
                        MINDEC(ret_c,2);
                    if (dp_T0[arg] == 0)
                        MINDEC(ret_c,0);
                    dp_T0[res] = dp_T0[arg2];

#if defined(_INDOPRO_)
		    copy_index_domain(res, arg2, ind_dom);
#endif
#if defined(_NONLIND_)
		    arg_index[res] = arg_index[arg2];
#endif
                }
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres,  TAYLOR_BUFFER[res])
                ASSIGN_T(Targ1, TAYLOR_BUFFER[arg1])
                ASSIGN_T(Targ2, TAYLOR_BUFFER[arg2])
#endif /* ALL_TOGETHER_AGAIN */

#ifdef _INT_FOR_
                coval = get_val_f();

                if (dp_T0[arg] > 0)
                    FOR_0_LE_l_LT_pk
                    TRES_INC = TARG1_INC;
                else
                    FOR_0_LE_l_LT_pk
                    TRES_INC = TARG2_INC;

                if (dp_T0[arg] > 0) {
                    if (coval <= 0.0)
                        MINDEC(ret_c,2);
                    dp_T0[res] = dp_T0[arg1];
                } else {
                    if (coval > 0.0)
                        MINDEC(ret_c,2);
                    if (dp_T0[arg] == 0)
                        MINDEC(ret_c,0);
                    dp_T0[res] = dp_T0[arg2];
                }
                FOR_0_LE_l_LT_pk
                TRES_INC = TARG1_INC | TARG2_INC;
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                if (dp_T0[arg] > 0)
                    FOR_0_LE_l_LT_pk
                    TRES_INC = TARG1_INC;
                else
                    FOR_0_LE_l_LT_pk
                    TRES_INC = TARG2_INC;
#endif

                if (dp_T0[arg] > 0) {
                    if (coval <= 0.0)
                        MINDEC(ret_c,2);
                    dp_T0[res] = dp_T0[arg1];
                } else {
                    if (coval > 0.0)
                        MINDEC(ret_c,2);
                    if (dp_T0[arg] == 0)
                        MINDEC(ret_c,0);
                    dp_T0[res] = dp_T0[arg2];
                }
#endif
#endif
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* ALL_TOGETHER_AGAIN */
		}
                break;

            case ref_cond_eq_assign:                                      /* cond_eq_assign */
                arg   = get_locint_f();
                arg1  = get_locint_f();
                arg2  = get_locint_f();
		{ 
#if !defined(_NTIGHT_)
		    locint ref = 
#endif
		    get_locint_f();
#if !defined(_NTIGHT_)
		    coval = 
#endif
		    get_val_f();
#if !defined(_NTIGHT_)
		    res   = (size_t)trunc(fabs(dp_T0[ref]));

		    IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

                /* olvo 980924 changed order to allow reflexive ops */
#if defined(_INDO_)
                if (dp_T0[arg] >= 0) {
                    if (coval < 0.0)
                        MINDEC(ret_c,2);
                    dp_T0[res] = dp_T0[arg1];

#if defined(_INDOPRO_)
		    copy_index_domain(res, arg1, ind_dom);
#endif
#if defined(_NONLIND_)
		    arg_index[res] = arg_index[arg1];
#endif
                } else {
                    if (coval >= 0.0)
                        MINDEC(ret_c,2);
                    dp_T0[res] = dp_T0[arg2];

#if defined(_INDOPRO_)
		    copy_index_domain(res, arg2, ind_dom);
#endif
#if defined(_NONLIND_)
		    arg_index[res] = arg_index[arg2];
#endif
                }
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres,  TAYLOR_BUFFER[res])
                ASSIGN_T(Targ1, TAYLOR_BUFFER[arg1])
                ASSIGN_T(Targ2, TAYLOR_BUFFER[arg2])
#endif /* ALL_TOGETHER_AGAIN */

#ifdef _INT_FOR_
                coval = get_val_f();

                if (dp_T0[arg] >= 0)
                    FOR_0_LE_l_LT_pk
                    TRES_INC = TARG1_INC;
                else
                    FOR_0_LE_l_LT_pk
                    TRES_INC = TARG2_INC;

                if (dp_T0[arg] >= 0) {
                    if (coval < 0.0)
                        MINDEC(ret_c,2);
                    dp_T0[res] = dp_T0[arg1];
                } else {
                    if (coval >= 0.0)
                        MINDEC(ret_c,2);
                    dp_T0[res] = dp_T0[arg2];
                }
                FOR_0_LE_l_LT_pk
                TRES_INC = TARG1_INC | TARG2_INC;
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                if (dp_T0[arg] >= 0)
                    FOR_0_LE_l_LT_pk
                    TRES_INC = TARG1_INC;
                else
                    FOR_0_LE_l_LT_pk
                    TRES_INC = TARG2_INC;
#endif

                if (dp_T0[arg] >= 0) {
                    if (coval < 0.0)
                        MINDEC(ret_c,2);
                    dp_T0[res] = dp_T0[arg1];
                } else {
                    if (coval >= 0.0)
                        MINDEC(ret_c,2);
                    dp_T0[res] = dp_T0[arg2];
                }
#endif
#endif
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* ALL_TOGETHER_AGAIN */
		}
                break;

            case ref_cond_assign_s:                                  /* cond_assign_s */
                arg   = get_locint_f();
                arg1  = get_locint_f();
                arg2   = get_locint_f();
#if !defined(_NTIGHT_)
                coval = 
#endif
		get_val_f();

#if !defined(_NTIGHT_)
		res = (size_t)trunc(fabs(dp_T0[arg2]));
                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

                /* olvo 980924 changed order to allow reflexive ops */
#if defined(_INDO_)
		if (dp_T0[arg] > 0) {
#if defined(_INDOPRO_)
                    copy_index_domain(res, arg1, ind_dom);
#endif
#if defined(_NONLIND_)
		    arg_index[res] = arg_index[arg1];
#endif
		}
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres,  TAYLOR_BUFFER[res])
                ASSIGN_T(Targ1, TAYLOR_BUFFER[arg1])
#endif /* ALL_TOGETHER_AGAIN */

#ifdef _INT_FOR_
                coval = get_val_f();

                if (dp_T0[arg] > 0)
                    FOR_0_LE_l_LT_pk
                    TRES_INC = TARG1_INC;

                if (dp_T0[arg] > 0) {
                    if (coval <= 0.0)
                        MINDEC(ret_c,2);
                    dp_T0[res] = dp_T0[arg1];
                } else
                    if (dp_T0[arg] == 0)
                        MINDEC(ret_c,0);
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                if (dp_T0[arg] > 0)
                    FOR_0_LE_l_LT_pk
                    TRES_INC = TARG1_INC;
#endif
                if (dp_T0[arg] > 0) {
                    if (coval <= 0.0)
                        MINDEC(ret_c,2);
                    dp_T0[res] = dp_T0[arg1];
                } else
                    if (dp_T0[arg] == 0)
                        MINDEC(ret_c,0);
#endif
#endif
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* ALL_TOGETHER_AGAIN */
                break;

            case ref_cond_eq_assign_s:                                  /* cond_eq_assign_s */
                arg   = get_locint_f();
                arg1  = get_locint_f();
                arg2   = get_locint_f();
#if !defined(_NTIGHT_)
                coval = 
#endif
		get_val_f();

#if !defined(_NTIGHT_)
		res = (size_t)trunc(fabs(dp_T0[arg2]));
                IF_KEEP_WRITE_TAYLOR(res,keep,k,p)

                /* olvo 980924 changed order to allow reflexive ops */
#if defined(_INDO_)
		if (dp_T0[arg] >= 0) {
#if defined(_INDOPRO_)
                    copy_index_domain(res, arg1, ind_dom);
#endif
#if defined(_NONLIND_)
		    arg_index[res] = arg_index[arg1];
#endif
		}
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                ASSIGN_T(Tres,  TAYLOR_BUFFER[res])
                ASSIGN_T(Targ1, TAYLOR_BUFFER[arg1])
#endif /* ALL_TOGETHER_AGAIN */

#ifdef _INT_FOR_
                coval = get_val_f();

                if (dp_T0[arg] >= 0)
                    FOR_0_LE_l_LT_pk
                    TRES_INC = TARG1_INC;

                if (dp_T0[arg] >= 0) {
                    if (coval < 0.0)
                        MINDEC(ret_c,2);
                    dp_T0[res] = dp_T0[arg1];
                }
#else
#if !defined(_ZOS_) /* BREAK_ZOS */
                if (dp_T0[arg] >= 0)
                    FOR_0_LE_l_LT_pk
                    TRES_INC = TARG1_INC;
#endif
                if (dp_T0[arg] >= 0) {
                    if (coval < 0.0)
                        MINDEC(ret_c,2);
                    dp_T0[res] = dp_T0[arg1];
                }
#endif
#endif
#else
		fprintf(DIAG_OUT, "ADOL-C error: active vector element referencing does not work in safe mode, please use tight mode\n");
		adolc_exit(-2,"",__func__,__FILE__,__LINE__);
#endif /* ALL_TOGETHER_AGAIN */
                break;

                /****************************************************************************/
                /*                                                          REMAINING STUFF */

                /*--------------------------------------------------------------------------*/
            case take_stock_op:                                  /* take_stock_op */
                size = get_locint_f();
                res  = get_locint_f();
#if !defined(_NTIGHT_)
		d =
#endif
                get_val_v_f(size);

                for (ls=0;ls<size;ls++) {
#if !defined(_NTIGHT_)
                    dp_T0[res]=*d;
#endif /* !_NTIGHT_ */
#if !defined(_INDO_)
#if !defined(_ZOS_) /* BREAK_ZOS */
                    ASSIGN_T(Tres,TAYLOR_BUFFER[res])

                    FOR_0_LE_l_LT_pk
                    TRES_INC = 0;

#endif /* ALL_TOGETHER_AGAIN */
                    res++;
#if !defined(_NTIGHT_)
                    d++;
#endif /* !_NTIGHT_ */
#endif
                }
                break;

                /*--------------------------------------------------------------------------*/
            case death_not:                                          /* death_not */
                arg1=get_locint_f();
                arg2=get_locint_f();

#ifdef _KEEP_
                if (keep) {
                    do {
                        IF_KEEP_WRITE_TAYLOR(arg2,keep,k,p)
                    } while(arg1 < arg2-- );
                }
#endif
                break;

                /*--------------------------------------------------------------------------*/
#if defined(_EXTERN_) /* ZOS,  FOS, FOV up to now */
            case ext_diff:                       /* extern differntiated function */
                ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index=get_locint_f();
                n=get_locint_f();
                m=get_locint_f();
                ADOLC_CURRENT_TAPE_INFOS.lowestXLoc_for = get_locint_f();
                ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_for = get_locint_f();
                ADOLC_CURRENT_TAPE_INFOS.cpIndex = get_locint_f();
                edfct=get_ext_diff_fct(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);

                if (edfct->ADOLC_EXT_FCT_POINTER==NULL)
                    fail(ADOLC_EXT_DIFF_NULLPOINTER_DIFFFUNC);
                if (n>0) {
                    if (edfct->dp_x==NULL) fail(ADOLC_EXT_DIFF_NULLPOINTER_ARGUMENT);
#if !defined(_ZOS_)
                    if (ADOLC_EXT_POINTER_X==NULL) fail(ADOLC_EXT_DIFF_NULLPOINTER_ARGUMENT);
#endif
                }
                if (m>0) {
                    if (edfct->dp_y==NULL) fail(ADOLC_EXT_DIFF_NULLPOINTER_ARGUMENT);
#if !defined(_ZOS_)
                    if (ADOLC_EXT_POINTER_Y==NULL) fail(ADOLC_EXT_DIFF_NULLPOINTER_ARGUMENT);
#endif
                }

                arg = ADOLC_CURRENT_TAPE_INFOS.lowestXLoc_for;
                for (loop=0; loop<n; ++loop) {
                    if (edfct->dp_x_changes) {
                      IF_KEEP_WRITE_TAYLOR(arg, keep, k, p);
                    }
                    edfct->dp_x[loop]=dp_T0[arg];
#if !defined(_ZOS_)
                    ADOLC_EXT_COPY_TAYLORS(ADOLC_EXT_POINTER_X[loop],TAYLOR_BUFFER[arg]);
#endif
                    ++arg;
                }
                arg = ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_for;
                for (loop=0; loop<m; ++loop) {
                    if (edfct->dp_y_priorRequired) {
                      IF_KEEP_WRITE_TAYLOR(arg, keep, k, p);
                    }
                    edfct->dp_y[loop]=dp_T0[arg];
#if !defined(_ZOS_)
                    ADOLC_EXT_COPY_TAYLORS(ADOLC_EXT_POINTER_Y[loop],TAYLOR_BUFFER[arg]);
#endif
                    ++arg;
                }

                ext_retc = edfct->ADOLC_EXT_FCT_COMPLETE;
                MINDEC(ret_c, ext_retc);

                res = ADOLC_CURRENT_TAPE_INFOS.lowestXLoc_for;
                for (loop=0; loop<n; ++loop) {
                    dp_T0[res]=edfct->dp_x[loop];
#if !defined(_ZOS_)
                    ADOLC_EXT_COPY_TAYLORS_BACK(ADOLC_EXT_POINTER_X[loop],TAYLOR_BUFFER[res]);
#endif
                    ++res;
                }
                res = ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_for;
                for (loop=0; loop<m; ++loop) {
                    dp_T0[res]=edfct->dp_y[loop];
#if !defined(_ZOS_)
                    ADOLC_EXT_COPY_TAYLORS_BACK(ADOLC_EXT_POINTER_Y[loop],TAYLOR_BUFFER[res]);
#endif
                    ++res;
                }

                break;

            case ext_diff_iArr:                 /* extern differntiated function */
                iArrLength=get_locint_f();
                iArr=malloc(iArrLength*sizeof(int));
                for (loop=0;loop<iArrLength;++loop) iArr[loop]=get_locint_f();
                get_locint_f(); /* iArrLength again */
                ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index=get_locint_f();
                n=get_locint_f();
                m=get_locint_f();
                ADOLC_CURRENT_TAPE_INFOS.lowestXLoc_for = get_locint_f();
                ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_for = get_locint_f();
                ADOLC_CURRENT_TAPE_INFOS.cpIndex = get_locint_f();
                edfct=get_ext_diff_fct(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);

                if (edfct->ADOLC_EXT_FCT_IARR_POINTER==NULL)
                    fail(ADOLC_EXT_DIFF_NULLPOINTER_DIFFFUNC);
                if (n>0) {
                    if (edfct->dp_x==NULL) fail(ADOLC_EXT_DIFF_NULLPOINTER_ARGUMENT);
#if !defined(_ZOS_)
                    if (ADOLC_EXT_POINTER_X==NULL) fail(ADOLC_EXT_DIFF_NULLPOINTER_ARGUMENT);
#endif
                }
                if (m>0) {
                    if (edfct->dp_y==NULL) fail(ADOLC_EXT_DIFF_NULLPOINTER_ARGUMENT);
#if !defined(_ZOS_)
                    if (ADOLC_EXT_POINTER_Y==NULL) fail(ADOLC_EXT_DIFF_NULLPOINTER_ARGUMENT);
#endif
                }

                arg = ADOLC_CURRENT_TAPE_INFOS.lowestXLoc_for;
                for (loop=0; loop<n; ++loop) {
                    if (edfct->dp_x_changes) {
                      IF_KEEP_WRITE_TAYLOR(arg, keep, k, p);
                    }
                    edfct->dp_x[loop]=dp_T0[arg];
#if !defined(_ZOS_)
                    ADOLC_EXT_COPY_TAYLORS(ADOLC_EXT_POINTER_X[loop],TAYLOR_BUFFER[arg]);
#endif
                    ++arg;
                }
                arg = ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_for;
                for (loop=0; loop<m; ++loop) {
                    if (edfct->dp_y_priorRequired) {
                      IF_KEEP_WRITE_TAYLOR(arg, keep, k, p);
                    }
                    edfct->dp_y[loop]=dp_T0[arg];
#if !defined(_ZOS_)
                    ADOLC_EXT_COPY_TAYLORS(ADOLC_EXT_POINTER_Y[loop],TAYLOR_BUFFER[arg]);
#endif
                    ++arg;
                }

                ext_retc = edfct->ADOLC_EXT_FCT_IARR_COMPLETE;
                MINDEC(ret_c, ext_retc);

                res = ADOLC_CURRENT_TAPE_INFOS.lowestXLoc_for;
                for (loop=0; loop<n; ++loop) {
                    dp_T0[res]=edfct->dp_x[loop];
#if !defined(_ZOS_)
                    ADOLC_EXT_COPY_TAYLORS_BACK(ADOLC_EXT_POINTER_X[loop],TAYLOR_BUFFER[res]);
#endif
                    ++res;
                }
                res = ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_for;
                for (loop=0; loop<m; ++loop) {
                    dp_T0[res]=edfct->dp_y[loop];
#if !defined(_ZOS_)
                    ADOLC_EXT_COPY_TAYLORS_BACK(ADOLC_EXT_POINTER_Y[loop],TAYLOR_BUFFER[res]);
#endif
                    ++res;
                }
                free((void*)iArr); iArr=0;
                break;
            case ext_diff_v2:
                ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index=get_locint_f();
                iArrLength = get_locint_f();
                iArr = malloc(iArrLength*sizeof(int));
                for (loop=0;loop<iArrLength;++loop) iArr[loop] = get_locint_f();
                get_locint_f(); /* iArrLength again */
                nin = get_locint_f();
                nout = get_locint_f();
                insz = malloc(2*(nin+nout)*sizeof(locint));
                outsz = insz + nin;
                ADOLC_CURRENT_TAPE_INFOS.lowestXLoc_ext_v2 = outsz + nout;
                ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_ext_v2 = outsz + nout + nin;
                for (loop=0;loop<nin;++loop) {
                    insz[loop] = get_locint_f();
                    ADOLC_CURRENT_TAPE_INFOS.lowestXLoc_ext_v2[loop] = get_locint_f();
                }
                for (loop=0;loop<nout;++loop) {
                    outsz[loop] = get_locint_f();
                    ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_ext_v2[loop] = get_locint_f();
                }
                get_locint_f(); /* nin again */
                get_locint_f(); /* nout again */
                edfct2 = get_ext_diff_fct_v2(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
                if (edfct2->ADOLC_EXT_FCT_POINTER==NULL)
                    fail(ADOLC_EXT_DIFF_NULLPOINTER_DIFFFUNC);
                if (nin>0) {
                    if (edfct2->x == NULL) fail(ADOLC_EXT_DIFF_NULLPOINTER_ARGUMENT);
#if !defined(_ZOS_)
                    if (ADOLC_EXT_V2_POINTER_X==NULL) fail(ADOLC_EXT_DIFF_NULLPOINTER_ARGUMENT);
#endif
                }
                if (nout>0) {
                    if (edfct2->y == NULL) fail(ADOLC_EXT_DIFF_NULLPOINTER_ARGUMENT);
#if !defined(_ZOS_)
                    if (ADOLC_EXT_V2_POINTER_Y == NULL) fail(ADOLC_EXT_DIFF_NULLPOINTER_ARGUMENT);
#endif
                }

                for(oloop=0;oloop<nin;++oloop) {
                    arg = ADOLC_CURRENT_TAPE_INFOS.lowestXLoc_ext_v2[oloop];
                    memcpy(&edfct2->x[oloop][0],&dp_T0[arg],insz[oloop]*sizeof(double));
                    for(loop=0;loop<insz[oloop];++loop) {
                        if(edfct2->dp_x_changes) {
                            IF_KEEP_WRITE_TAYLOR(arg,keep,k,p);
                        }
#if !defined(_ZOS_)
                        ADOLC_EXT_COPY_TAYLORS(ADOLC_EXT_V2_POINTER_X[oloop][loop],TAYLOR_BUFFER[arg]);
#endif
                        ++arg;
                    }
                }
                for(oloop=0;oloop<nout;++oloop) {
                    arg=ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_ext_v2[oloop];
                    memcpy(&edfct2->y[oloop][0],&dp_T0[arg],outsz[oloop]*sizeof(double));
                    for(loop=0;loop<outsz[oloop];++loop) {
                        if (edfct2->dp_y_priorRequired) {
                            IF_KEEP_WRITE_TAYLOR(arg,keep,k,p);
                        }
#if !defined(_ZOS_)
                       ADOLC_EXT_COPY_TAYLORS(ADOLC_EXT_V2_POINTER_Y[oloop][loop],TAYLOR_BUFFER[arg]);
#endif

                        ++arg;
                    }
                }
                ext_retc = edfct2->ADOLC_EXT_FCT_V2_COMPLETE;
                MINDEC(ret_c,ext_retc);

                for(oloop=0;oloop<nin;++oloop) {
                    res = ADOLC_CURRENT_TAPE_INFOS.lowestXLoc_ext_v2[oloop];
                    memcpy(&dp_T0[res],&edfct2->x[oloop][0],insz[oloop]*sizeof(double));
#if !defined(_ZOS_)
                    for(loop=0;loop<insz[oloop];++loop) {
                        ADOLC_EXT_COPY_TAYLORS_BACK(ADOLC_EXT_V2_POINTER_X[oloop][loop],TAYLOR_BUFFER[res]);
                        ++res;
                    }
#endif
                }

                for(oloop=0;oloop<nout;++oloop) {
                    res = ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_ext_v2[oloop];
                    memcpy(&dp_T0[res],&edfct2->y[oloop][0],outsz[oloop]*sizeof(double));
#if !defined(_ZOS_)
                    for(loop=0;loop<outsz[oloop];++loop) {
                        ADOLC_EXT_COPY_TAYLORS_BACK(ADOLC_EXT_V2_POINTER_Y[oloop][loop],TAYLOR_BUFFER[res]);
                        ++res;
                    }
#endif
                }

                free(insz);
                free(iArr);
                insz = 0;
                iArr = 0;
                outsz = 0;
                ADOLC_CURRENT_TAPE_INFOS.lowestXLoc_ext_v2 = 0;
                ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_ext_v2 = 0;
                break;
#endif
#ifdef ADOLC_MEDIPACK_SUPPORT
                /*--------------------------------------------------------------------------*/
            case medi_call: {
                locint mediIndex = get_locint_f();
                short tapeId = ADOLC_CURRENT_TAPE_INFOS.tapeID;

#if defined(_ZOS_)
                mediCallHandlePrimal(tapeId, mediIndex, dp_T0);
#elif defined(_FOS_)
                mediCallHandleForward(tapeId, mediIndex, dp_T0, &TAYLOR_BUFFER, 1);
#elif defined(_FOV_)
                mediCallHandleForward(tapeId, mediIndex, dp_T0, TAYLOR_BUFFER, p);
#endif
                break;
            }
#endif
#ifdef ADOLC_AMPI_SUPPORT
                /*--------------------------------------------------------------------------*/
            case ampi_send: {
              ADOLC_TLM_AMPI_Send(buf,
                  count,
                  datatype,
                  src,
                  tag,
                  pairedWith,
                  comm);
              break;
            }
            case ampi_recv: {
              ADOLC_TLM_AMPI_Recv(buf,
                            count,
                            datatype,
                            src,
                            tag,
                            pairedWith,
                            comm,
                            &status);
              break;
            }
          case ampi_isend: {
            ADOLC_TLM_AMPI_Isend(buf,
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
            ADOLC_TLM_AMPI_Irecv(buf,
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
            ADOLC_TLM_AMPI_Wait(&request,
                          &status);
            break;
          }
          case ampi_barrier: {
            ADOLC_TLM_AMPI_Barrier(comm);
            break;
          }
          case ampi_gather: {
            ADOLC_TLM_AMPI_Gather(buf,
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
            ADOLC_TLM_AMPI_Scatter(rbuf,
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
            ADOLC_TLM_AMPI_Allgather(buf,
                               count,
                               datatype,
                               rbuf,
                               rcount,
                               rtype,
                               comm);
            break;
          }
          case ampi_gatherv: {
            ADOLC_TLM_AMPI_Gatherv(buf,
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
            ADOLC_TLM_AMPI_Scatterv(rbuf,
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
            ADOLC_TLM_AMPI_Allgatherv(buf,
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
            ADOLC_TLM_AMPI_Bcast(buf,
                           count,
                           datatype,
                           src,
                           comm);
            break;
          }
          case ampi_reduce: {
            ADOLC_TLM_AMPI_Reduce(buf,
                            rbuf,
                            count,
                            datatype,
                            op,
                            src,
                            comm);
            break;
          }
          case ampi_allreduce: {
            ADOLC_TLM_AMPI_Allreduce(buf,
                               rbuf,
                               count,
                               datatype,
                               op,
                               comm);
            break;
          }
#endif

                /*--------------------------------------------------------------------------*/

            default:                                                   /* default */
                /* Die here, we screwed up */

                fprintf(DIAG_OUT,"ADOL-C fatal error in " GENERATED_FILENAME " ("
                        __FILE__
                        ") : no such operation %d\n", operation);
                adolc_exit(-1,"",__func__,__FILE__,__LINE__);
                break;

        } /* endswitch */

        /* Read the next operation */
        operation=get_op_f();
#if defined(ADOLC_DEBUG)
        ++countPerOperation[operation];
#endif /* ADOLC_DEBUG */
    }  /* endwhile */


#if defined(ADOLC_DEBUG)
    printf("\nTape contains:\n");
    for (v = 0; v < 256; ++v)
        if (countPerOperation[v] > 0)
            printf("operation %3d: %6d time(s) - %6d taylors written (%10.2f per operation)\n", v, countPerOperation[v], taylorPerOperation[v], (double)taylorPerOperation[v] / (double)countPerOperation[v]);
    printf("\n");
#endif /* ADOLC_DEBUG */

#if defined(_KEEP_)
    if (keep) taylor_close(taylbuf);
#endif

    /* clean up */
#if !defined (_NTIGHT_)
    free(dp_T0);
#endif /* !_NTIGHT_ */
#if !defined(_INDO_)
#if !defined(_ZOS_)
#   if defined(_FOS_)
    free(dp_T);
#   else
#if !defined (_INT_FOR_)
    myfree2(dpp_T);
    free(dp_Ttemp);
#endif /* !_NTIGHT_ */
#endif
#endif
#endif
#if defined(_HIGHER_ORDER_)
    free(dp_z);
#endif

    ADOLC_CURRENT_TAPE_INFOS.workMode = ADOLC_NO_MODE;
    end_sweep();


#if defined(_INDO_)
#if defined(_INDOPRO_)
    for(i=0;i<max_ind_dom;i++)
      {
	free(ind_dom[i]);
      }
    free(ind_dom);
#endif
#if defined(_NONLIND_)
    for( i=0; i < indcheck; i++) {
      traverse_crs(&nonl_dom[i],&sod[i],indcheck+1);
      free_tree(&nonl_dom[i],indcheck+1);
      crs[i] = (unsigned int*) malloc(sizeof(unsigned int) * (sod[i].entry+1));
      crs[i][0] = sod[i].entry;
      temp = sod[i].left;
      for( ii=1; ii <=sod[i].entry; ii++)
	{
	  crs[i][ii] = temp->entry;
	  temp1 = temp->left;
	  free(temp);
	  temp = temp1;
	}
    }

    free(sod);
    free(nonl_dom);
    free(fod);
    free(arg_index);

#endif
#if defined(_NONLIND_OLD_)

    for( i=0; i < indcheck; i++) {
       crs[i] = (unsigned int*) malloc(sizeof(unsigned int) * (nonl_dom[i][0]+1));
       crs[i][0] = nonl_dom[i][0];
       for(l=1; l < crs[i][0]+1; l++)
          crs[i][l] = nonl_dom[i][l+1];
       free(nonl_dom[i]);
    }
    free(nonl_dom);

#endif
#endif
    return ret_c;
}


/****************************************************************************/
#if defined(_ZOS_) && defined(_ABS_NORM_)
int get_num_switches(short tapeID) {
    int nswitch;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    init_for_sweep(tapeID);
    if (!ADOLC_CURRENT_TAPE_INFOS.stats[NO_MIN_MAX]) {
	fprintf(DIAG_OUT,"ADOL-C error: tape %d was not created compatible "
		"with %s\n              Please call enableMinMaxUsingAbs() "
		"before trace_on(%d)\n", tapeID, __FUNCTION__, tapeID);
	adolc_exit(-1,"",__func__,__FILE__,__LINE__);
    }
    nswitch = ADOLC_CURRENT_TAPE_INFOS.stats[NUM_SWITCHES];
    end_sweep();
    return nswitch;
}
#endif
#if defined(_ABS_NORM_) && defined(_FOV_)
short firstsign(int p, double *u, double* du) {
    int i=0;
    short tmp;
    tmp=((*u)>1e-12)?1.0:(((*u)<-1e-12)?-1.0:0.0);
    while(i<p && tmp==0.0) {
	tmp=(du[i]>0.0)?1.0:((du[i]<0.0)?-1.0:0.0);
	i++;
    }
    return tmp;
}
#elif defined(_ABS_NORM_SIG_) && defined(_FOV_)
short ext_firstsign(double sigbase, double sigdir, int p, double *u, double* du) {
    if (sigbase == 0 && sigdir > 0) { 
        du[0] = fmax(0,du[0]);
        *u = 0;
    }
    else if (sigbase == 0 && sigdir < 0) {
        du[0] = fmin(0,du[0]);
        *u =0;
    }
    else if (sigbase == 0 && sigdir == 0) {
        du[0] = 0;
        *u = 0;
    }
    return firstsign(p,u,du);
}

short ext_firstsign2(double sigbase, int p, double *u, double* du) {
    if (sigbase == 0)
        *u=0;
    return firstsign(p,u,du);
}
#endif
/****************************************************************************/

#if defined(_INDOPRO_) && !defined(_NONLIND_OLD_)

/****************************************************************************/
/* set operations for propagation of index domains                          */

/*--------------------------------------------------------------------------*/
/* operations on index domains                                              */

#if defined(_TIGHT_)
void copy_index_domain(int res, int arg, locint **ind_dom) {

   int i;

   if (ind_dom[arg][0] > ind_dom[res][1])
     {
       free(ind_dom[res]);
       ind_dom[res] = (locint *)  malloc(sizeof(locint) * 2*(ind_dom[arg][0]+1));
       ind_dom[res][1] = 2*ind_dom[arg][0];
     }


    for(i=2;i<ind_dom[arg][0]+2;i++)
       ind_dom[res][i] = ind_dom[arg][i];
    ind_dom[res][0] = ind_dom[arg][0];
}


void merge_2_index_domains(int res, int arg, locint **ind_dom)
{

  int num,num1,num2, i,j,k,l;
  locint *temp_array, *arg_ind_dom, *res_ind_dom;

  if (ind_dom[res][0] == 0)
    copy_index_domain(res,arg,ind_dom);
  else
    {
      if (res != arg)
     {
       arg_ind_dom = ind_dom[arg];
       res_ind_dom = ind_dom[res];

       num  = ind_dom[res][0];
       num1 = arg_ind_dom[0];
       num2 = ind_dom[res][1];

       if (num2 < num1+num)
         num2 = num1+num;

       temp_array = (locint *)  malloc(sizeof(locint)* (num2+2));
       temp_array[1] = num2;

       i = 2;
       j = 2;
       k = 2;
       num += 2;
       num1 += 2;
       while ((i< num) && (j < num1))
         {
           if (res_ind_dom[i] < arg_ind_dom[j])
          {
            temp_array[k] = res_ind_dom[i];
            i++; k++;
          }
           else
          {
            if (res_ind_dom[i] == arg_ind_dom[j])
              {
                temp_array[k] = arg_ind_dom[j];
                i++;j++;k++;
              }
            else
              {
                temp_array[k] = arg_ind_dom[j];
                j++;k++;
              }
          }
         }
       for(l = i;l<num;l++)
         {
           temp_array[k] = res_ind_dom[l];
           k++;
         }
       for(l = j;l<num1;l++)
         {
           temp_array[k] = arg_ind_dom[l];
           k++;
         }
       temp_array[0] = k-2;
       free(ind_dom[res]);
       ind_dom[res]=temp_array;
     }
    }


}

void combine_2_index_domains(int res, int arg1, int arg2, locint **ind_dom) {

    if (res != arg1)
       copy_index_domain(res, arg1, ind_dom);

    merge_2_index_domains(res, arg2, ind_dom);
}

void merge_3_index_domains(int res, int arg1, int arg2, locint **ind_dom) {
    merge_2_index_domains(res, arg1, ind_dom);
    merge_2_index_domains(res, arg2, ind_dom);
}



#endif
#endif


#if defined(_NONLIND_)
#if defined(_TIGHT_)

void free_tree(IndexElement* tree, int num)
{

  if (tree->left != NULL)
    {
      free_tree(tree->left,num);
    }
  if (tree->right != NULL)
    {
      free_tree(tree->right,num);
     }
    {
      if (tree->entry == num)
	free(tree);

    }
  
}
void traverse_crs(IndexElement* tree,  IndexElement_sod* sod, int num)
{

  IndexElement_sod *temp, *temp1;

  if (tree->left != NULL)
    {
      traverse_crs(tree->left, sod, num);
    }
  if (tree->right != NULL)
    {
      traverse_crs(tree->right, sod, num);
    }
  if (tree->entry < num)
    {
      temp = sod->left;
      if (temp == NULL)
	{
	  temp = (struct IndexElement_sod*) malloc(sizeof(struct IndexElement_sod));
	  temp->left = NULL;
	  temp->entry = tree->entry;
	  sod->entry++;
	  sod->left=temp;
	}
      else
	{
	  while ((temp->entry < tree->entry) && (temp->left != NULL))
	    {
	      temp1 = temp;
	      temp = temp->left;
	    }
	  if (temp->left == NULL)
	    {
	      if(temp->entry < tree->entry)
		{
		  temp->left = (struct IndexElement_sod*) malloc(sizeof(struct IndexElement_sod));
		  temp = temp->left;
		  temp->left = NULL;
		  temp->entry = tree->entry;
		  sod->entry++;
		}
	      if(temp->entry > tree->entry)
		{
		  temp->left = (struct IndexElement_sod*) malloc(sizeof(struct IndexElement_sod));
		  temp->left->entry = temp->entry;
		  temp->left->left = NULL;
		  temp->entry = tree->entry;
		  sod->entry++;
		}
	    }
	  else
	    {
	      if (temp->entry > tree->entry)
		{
		  temp1 = (struct IndexElement_sod*) malloc(sizeof(struct IndexElement_sod));
		  temp1->left = temp->left;
		  temp1->entry = temp->entry;
		  temp->entry = tree->entry;
		  temp->left=temp1;
		  sod->entry++;
		}
	      
	    }
	}
    }
}

void traverse_unary(IndexElement* tree,  IndexElement* nonl_dom,  IndexElement* fodi, int num, int maxopind)
{
  IndexElement *temp;

  if (tree->left != NULL)
    {
      traverse_unary(tree->left, nonl_dom, fodi, num, maxopind);
      if (tree->right != NULL)
	{
	  traverse_unary(tree->right, nonl_dom, fodi, num, maxopind);
	}
     }
  else
    {
      if(tree->entry<maxopind)
	{
	  temp = (struct IndexElement*) malloc(sizeof(struct IndexElement));
	  temp->right = fodi;
	  temp->left = nonl_dom[tree->entry].left;
	  temp->entry= num;
	  nonl_dom[tree->entry].left = temp;
	}
    }
}

#endif
#endif

#if defined(_NONLIND_OLD_)
#if defined(_TIGHT_)

void extend_nonlinearity_domain_binary_step
(int arg1, int arg2, locint **ind_dom, locint **nonl_dom) 
{
  int index,num,num1, num2, i,j,k,l,m;
  locint *temp_nonl, *index_nonl_dom;

  num = ind_dom[arg2][0];

  for(m=2;m<ind_dom[arg1][0]+2;m++) 
    {
      index = ind_dom[arg1][m];
      index_nonl_dom = nonl_dom[index];

      if (index_nonl_dom[0] == 0)  /* empty list */
	{
	  if ( index_nonl_dom[1] < num)
	    {
	      free(index_nonl_dom);
	      index_nonl_dom = (locint*) malloc(sizeof(locint)*2*(num+1) );
	      index_nonl_dom[1] = 2*num;
	    }
	  for(i=2;i<num+2;i++)      /* append index domain list of "arg" */
	    index_nonl_dom[i] = ind_dom[arg2][i];
	  index_nonl_dom[0] = num;
	} 
      else 
	{ /* merge lists */
	  num1 = index_nonl_dom[0];
	  num2 = index_nonl_dom[1];
	  
	  if (num1+num > num2)
	    num2 = num1+num;
	  
	  temp_nonl = (locint*) malloc(sizeof(locint)*(num2+2));
	  temp_nonl[1] = num2;
	  
	  i = 2;
	  k = 2;
	  j = 2;
	  num1 +=2;
	  num2 = num+2;
	  while ((i<num1) && (j < num2)){
	    if (ind_dom[arg2][j] < index_nonl_dom[i]) /* < */ {
	      temp_nonl[k] = ind_dom[arg2][j];
	      j++; k++;
	    } else {
	      if (ind_dom[arg2][j] == index_nonl_dom[i])  /* == */ {
		temp_nonl[k] = ind_dom[arg2][j];
		j++; k++; i++;
	      } else {
		temp_nonl[k] = index_nonl_dom[i];
		i++; k++;
	      }
	    }
	  }
	  for(l = j;l<num2;l++) {
	    temp_nonl[k] = ind_dom[arg2][l];
	    k++;
	  }
	  for(l = i;l<num1;l++) {
	    temp_nonl[k] = index_nonl_dom[l];
	    k++;
	  }
	  temp_nonl[0] = k-2; 
	  free((char*) nonl_dom[index]);
	  nonl_dom[index] = temp_nonl;
	}
    }
}

void extend_nonlinearity_domain_unary
(int arg, locint **ind_dom, locint **nonl_dom) {
    extend_nonlinearity_domain_binary_step(arg, arg, ind_dom, nonl_dom);
}

void extend_nonlinearity_domain_binary
(int arg1, int arg2, locint **ind_dom, locint **nonl_dom) {
    extend_nonlinearity_domain_binary_step(arg1, arg2, ind_dom, nonl_dom);
    extend_nonlinearity_domain_binary_step(arg2, arg1, ind_dom, nonl_dom);
}


#endif
#endif
END_C_DECLS

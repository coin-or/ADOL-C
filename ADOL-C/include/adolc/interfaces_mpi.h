/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     interfaces_mpi.h
 Revision: $Id$
 Contents: C allocation of arrays of doubles in several dimensions

 Copyright (c) Andrea Walther, Benjamin Letschert

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#if !defined(ADOLC_INTERFACES_MPI_H)
#define ADOLC_INTERFACES_MPI_H 1

#include <adolc/common.h>
#include <adolc/interfaces.h>

#if defined(HAVE_MPI)
extern int mpi_initialized;
extern int all_root;

/* High level driver functions */
/* at first parameter this process-ID */

#ifdef __cplusplus

/* zos_forward(process id,procsize, tag, m, n, keep, x[n], y[m])*/
ADOLC_DLL_EXPORT int zos_forward(
    int,int,short,int,int,int,const double*,double*);

/* fos_forward(process id,procsize, tag, m, n, keep, x[n], X[n], y[m], Y[m])*/
ADOLC_DLL_EXPORT int fos_forward(
    int,int,short,int,int,int,const double*, double*,double*,double*);
/* fos_reverse(process id, procsize, tag, m, n, u[m], z[n])     */
ADOLC_DLL_EXPORT int fos_reverse(
    int, int, short,int,int, double*,double*);

/* hos_forward(process id,procsize, tag, m, n, d, keep, x[n], X[n][d], y[m], Y[m][d]) */
ADOLC_DLL_EXPORT int hos_forward(
    int, int, short, int, int, int, int, double*, double**, double*, double**);

/*  hos_reverse(process id,procsize, tag, m, n, d, u[m], Z[n][d+1])  */
ADOLC_DLL_EXPORT int hos_reverse(
    int, int, short, int, int, int, double*, double** );

/* fov_forward(process id, procsize, tag, m, n, p, x[n], X[n][p], y[m], Y[m][p]) */
ADOLC_DLL_EXPORT int fov_forward(
    int,int,short,int,int,int,const double*, double**,double*,double**);
/* fov_reverse(process id, procsize, tag, m, n, d, p, U[p][m], Z[p][n])  */
ADOLC_DLL_EXPORT int fov_reverse(
    int, int, short,int,int,int,double**,double**);

ADOLC_DLL_EXPORT int hov_forward(
    int, int, short,int,int,int,int, double*,double***,double*,double***);

ADOLC_DLL_EXPORT int hov_reverse(
    int, int, short,int,int,int,int, double**,double***,short**);

/* int_forward_tight(rank,size,tag, m, n, p, x[n], X[n][p], y[m], Y[m][p])            */
ADOLC_DLL_EXPORT int int_forward_tight(
    int,int,short,int,int,int,const double*,unsigned long int**,double*,unsigned long int**);

/* int_forward_safe(rank,size, tag, m, n, p, X[n][p], Y[m][p])                        */
ADOLC_DLL_EXPORT int int_forward_safe(
    int,int,short,int,int,int,unsigned long int**,unsigned long int**);

/* int_reverse_tight(tag, m, n, q, U[q][m], Z[q][n])                        */
ADOLC_DLL_EXPORT int int_reverse_tight
(int,int,short, int, int, int, unsigned long int**, unsigned long int**);
/* int_reverse_safe(tag, m, n, q, U[q][m], Z[q][n])                         */
ADOLC_DLL_EXPORT int int_reverse_safe
(int,int,short, int, int, int, unsigned long int**, unsigned long int**);

/* indopro_forward_tight(rank,size, tag, m, n, x[n], *crs[m])                         */
ADOLC_DLL_EXPORT int indopro_forward_tight(
    int, int, short, int, int,const double*, unsigned int** );

/* indopro_forward_safe( tag, m, n, x[n], *crs[m])                                   */
ADOLC_DLL_EXPORT int indopro_forward_safe(
    int, int, short, int, int,const double*,unsigned int**);

/* indopro_forward_tight( tag, m, n, x[n], *crs[m])   */
ADOLC_DLL_EXPORT int nonl_ind_forward_tight(
    int,int, short, int, int,const double*, unsigned int**);

/* indopro_forward_safe( tag, m, n, x[n], *crs[m])   */
ADOLC_DLL_EXPORT int nonl_ind_forward_safe(
    int, int, short, int, int,const double*, unsigned int**);

#endif /* __cplusplus */


BEGIN_C_DECLS
/* zos_forward(process id,procsize, tag, m, n, keep, x[n], y[m])*/
ADOLC_DLL_EXPORT int zos_forward_mpi(
    int,int,short,int,int,int,const double*,double*);

/* fos_forward(process id,procsize, tag, m, n, keep, x[n], X[n], y[m], Y[m])*/
ADOLC_DLL_EXPORT int fos_forward_mpi(
    int,int,short,int,int,int,const double*,double*,double*,double*);
/* fos_reverse(process id, procsize, tag, m, n, u[m], z[n])     */
ADOLC_DLL_EXPORT int fos_reverse_mpi(
    int, int, short,int,int,double*,double*);

/* hos_forward(process id,procsize, tag, m, n, d, keep, x[n], X[n][d], y[m], Y[m][d]) */
ADOLC_DLL_EXPORT int hos_forward_mpi(
    int, int, short, int, int, int, int, double*, double**, double*, double**);
/*  hos_reverse(process id,procsize, tag, m, n, d, u[m], Z[n][d+1])  */
ADOLC_DLL_EXPORT int hos_reverse_mpi(
    int, int, short, int, int, int, double*, double** );

ADOLC_DLL_EXPORT int hov_forward_mpi(
    int, int, short,int,int,int,int, double*,double***,double*,double***);

ADOLC_DLL_EXPORT int hov_reverse_mpi(
    int, int, short,int,int,int,int, double**,double***,short**);

/* fov_forward(process id, procsize, tag, m, n, p, x[n], X[n][p], y[m], Y[m][p]) */
ADOLC_DLL_EXPORT int fov_forward_mpi(
    int,int,short,int,int,int,const double*,double**,double*,double**);
/* fov_reverse(process id, procsize, tag, m, n, d, p, U[p][m], Z[p][n])  */
ADOLC_DLL_EXPORT int fov_reverse_mpi(
    int, int, short,int,int,int,double**,double**);

/* int_forward_tight(rank,size,tag, m, n, p, x[n], X[n][p], y[m], Y[m][p])            */
ADOLC_DLL_EXPORT int int_forward_tight_mpi(
    int,int,short,int,int,int,const double*,unsigned long int**,double*,unsigned long int**);

/* int_forward_safe(rank,size, tag, m, n, p, X[n][p], Y[m][p])                        */
ADOLC_DLL_EXPORT int int_forward_safe_mpi(
    int,int,short,int,int,int,unsigned long int**,unsigned long int**);

/* int_reverse_tight(rank, size,tag, m, n, q, U[q][m], Z[q][n])                        */
ADOLC_DLL_EXPORT int int_reverse_tight_mpi
(int,int,short, int, int, int, unsigned long int**, unsigned long int**);
/* int_reverse_safe(rank, size,tag, m, n, q, U[q][m], Z[q][n])                         */
ADOLC_DLL_EXPORT int int_reverse_safe_mpi
(int,int,short, int, int, int, unsigned long int**, unsigned long int**);
/* indopro_forward_tight(rank,size, tag, m, n, x[n], *crs[m])                         */
ADOLC_DLL_EXPORT int indopro_forward_tight_mpi(
    int, int, short, int, int,const double*, unsigned int** );

/* indopro_forward_safe( tag, m, n, x[n], *crs[m])                                   */
ADOLC_DLL_EXPORT int indopro_forward_safe_mpi(
    int, int, short, int, int,const double*,unsigned int**);

/* indopro_forward_tight( tag, m, n, x[n], *crs[m])   */
ADOLC_DLL_EXPORT int nonl_ind_forward_tight_mpi(
    int,int, short, int, int,const double*, unsigned int**);

/* indopro_forward_safe( tag, m, n, x[n], *crs[m])   */
ADOLC_DLL_EXPORT int nonl_ind_forward_safe_mpi(
    int, int, short, int, int,const double*, unsigned int**);

ADOLC_DLL_EXPORT int hos_ti_reverse_mpi( int mpi_id, int mpi_size,
    short   tnum,        /* tape id */
    int     depen,       /* consistency chk on # of deps */
    int     indep,       /* consistency chk on # of indeps */
    int     degre,       /* highest derivative degre  */
    double  **lagrange,  /* range weight vectors       */
    double  **results);   /* matrix of coefficient vectors */

ADOLC_DLL_EXPORT int hov_ti_reverse_mpi( int mpi_id, int mpi_size,
    short   tnum,        /* tape id */
    int     depen,       /* consistency chk on # of deps */
    int     indep,       /* consistency chk on # of indeps */
    int     degre,       /* highest derivative degre */
    int     nrows,       /* # of Jacobian rows calculated */
    double  ***lagrange, /* domain weight vectors */
    double  ***results,  /* matrix of coefficient vectors */
    short   **nonzero );  /* structural sparsity  pattern  */
END_C_DECLS

#endif /*HAVE_MPI*/

#endif
/* That's all*/

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

#if defined(HAVE_MPI_MPI_H)
#include <mpi/mpi.h>
#elif defined(HAVE_MPI_H)
#include <mpi.h>
#endif
#if defined(HAVE_MPI)
#include <adolc/common.h>
#include <adolc/adouble.h>
#include <adolc/adolc_mpi.h>
#include <adolc/interfaces.h>

/* High level driver functions */
/* at first parameter this process-ID */

#ifdef __cplusplus

/* zos_forward(process id,procsize, tag, m, n, keep, x[n], y[m])*/
int zos_forward(
    int,int,short,int,int,int,const double*,double*);

/* fos_forward(process id,procsize, tag, m, n, keep, x[n], X[n], y[m], Y[m])*/
int fos_forward(
    int,int,short,int,int,int,const double*,double*,double*,double*);
/* fos_reverse(process id, procsize, tag, m, n, u[m], z[n])     */
int fos_reverse(
    int, int, short,int,int,double*,double*);

/* hos_forward(process id,procsize, tag, m, n, d, keep, x[n], X[n][d], y[m], Y[m][d]) */
int hos_forward(
    int, int, short, int, int, int, int, double*, double**, double*, double**);
/*  hos_reverse(process id,procsize, tag, m, n, d, u[m], Z[n][d+1])  */
int hos_reverse(
    int, int, short, int, int, int, double*, double** );

/* fov_forward(process id, procsize, tag, m, n, p, x[n], X[n][p], y[m], Y[m][p]) */
int fov_forward(
    int,int,short,int,int,int,const double*,double**,double*,double**);
/* fov_reverse(process id, procsize, tag, m, n, d, p, U[p][m], Z[p][n])  */
int fov_reverse(
    int, int, short,int,int,int,double**,double**);

/* int_forward_tight(rank,size,tag, m, n, p, x[n], X[n][p], y[m], Y[m][p])            */
int int_forward_tight(
    int,int,short,int,int,int,double*,unsigned long int**,double*,unsigned long int**);

/* int_forward_safe(rank,size, tag, m, n, p, X[n][p], Y[m][p])                        */
int int_forward_safe(
    int,int,short,int,int,int,unsigned long int**,unsigned long int**);

/* indopro_forward_tight(rank,size, tag, m, n, x[n], *crs[m])                         */
int indopro_forward_tight(
    int, int, short, int, int, double*, unsigned int** );

/* indopro_forward_safe( tag, m, n, x[n], *crs[m])                                   */
int indopro_forward_safe(
    int, int, short, int, int, double*,unsigned int**);

/* indopro_forward_tight( tag, m, n, x[n], *crs[m])   */
int nonl_ind_forward_tight(
    int,int, short, int, int, double*, unsigned int**);

/* indopro_forward_safe( tag, m, n, x[n], *crs[m])   */
int nonl_ind_forward_safe(
    int, int, short, int, int, double*, unsigned int**);

#endif /* __cplusplus */

#ifdef __cpluplus
extern "C" {
#endif

int zos_forward_p(
    int,int,short,int,int,int,const double*,double*);

/* fos_forward(process id,procsize, tag, m, n, keep, x[n], X[n], y[m], Y[m])*/
int fos_forward_p(
    int,int,short,int,int,int,const double*,double*,double*,double*);
/* fos_reverse(process id, procsize, tag, m, n, u[m], z[n])     */
int fos_reverse_p(
    int, int, short,int,int,double*,double*);

/* hos_forward(process id,procsize, tag, m, n, d, keep, x[n], X[n][d], y[m], Y[m][d]) */
int hos_forward_p(
    int, int, short, int, int, int, int, double*, double**, double*, double**);
/*  hos_reverse(process id,procsize, tag, m, n, d, u[m], Z[n][d+1])  */
int hos_reverse_p(
    int, int, short, int, int, int, double*, double** );

/* fov_forward(process id, procsize, tag, m, n, p, x[n], X[n][p], y[m], Y[m][p]) */
int fov_forward_p(
    int,int,short,int,int,int,const double*,double**,double*,double**);
/* fov_reverse(process id, procsize, tag, m, n, d, p, U[p][m], Z[p][n])  */
int fov_reverse_p(
    int, int, short,int,int,int,double**,double**);

/* int_forward_tight(rank,size,tag, m, n, p, x[n], X[n][p], y[m], Y[m][p])            */
int int_forward_tight_p(
    int,int,short,int,int,int,double*,unsigned long int**,double*,unsigned long int**);

/* int_forward_safe(rank,size, tag, m, n, p, X[n][p], Y[m][p])                        */
int int_forward_safe_p(
    int,int,short,int,int,int,unsigned long int**,unsigned long int**);

/* indopro_forward_tight(rank,size, tag, m, n, x[n], *crs[m])                         */
int indopro_forward_tight_p(
    int, int, short, int, int, double*, unsigned int** );

/* indopro_forward_safe( tag, m, n, x[n], *crs[m])                                   */
int indopro_forward_safe_p(
    int, int, short, int, int, double*,unsigned int**);

/* indopro_forward_tight( tag, m, n, x[n], *crs[m])   */
int nonl_ind_forward_tight_p(
    int,int, short, int, int, double*, unsigned int**);

/* indopro_forward_safe( tag, m, n, x[n], *crs[m])   */
int nonl_ind_forward_safe_p(
    int, int, short, int, int, double*, unsigned int**);

#ifdef __cpluplus
}
#endif


#endif /*HAVE_MPI*/

#endif
/* That's all*/

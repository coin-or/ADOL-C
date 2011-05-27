/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     adolc_mpi.h
 Revision: $Id$
 Contents: C allocation of arrays of doubles in several dimensions

 Copyright (c) Andrea Walther, Benjamin Letschert

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#if !defined(ADOLC_ADOLC_MPI_H)
#define ADOLC_ADOLC_MPI_H 1

#if defined(HAVE_MPI_MPI_H)
#include <mpi/mpi.h>
#elif defined(HAVE_MPI_H)
#include <mpi.h>
#endif
#if defined(HAVE_MPI)
#include <adolc/common.h>
#include <adolc/adouble.h>

#define ADOLC_MPI_Datatype MPI_Datatype
#define MPI_ADOUBLE MPI_DOUBLE
#define ADOLC_MPI_COMM_WORLD MPI_COMM_WORLD
#define ADOLC_MPI_Comm MPI_Comm

#ifdef __cplusplus
extern "C" {
#endif

int ADOLC_MPI_Init(int* a, char*** b);
int ADOLC_MPI_Comm_size(ADOLC_MPI_Comm comm, int* size);
int ADOLC_MPI_Comm_rank(ADOLC_MPI_Comm vomm , int* rank);
int ADOLC_MPI_Get_processor_name(char* a, int* b) ;
int ADOLC_MPI_Barrier(ADOLC_MPI_Comm comm);
int ADOLC_MPI_Finalize() ;
#ifdef __cplusplus
}
#endif
extern int mpi_initialized;

#ifdef __cplusplus
int ADOLC_MPI_Send(
    adouble *buf, int count, ADOLC_MPI_Datatype datatype, int dest,
    int tag, ADOLC_MPI_Comm comm );

int ADOLC_MPI_Recv(
    adouble *buf, int count, ADOLC_MPI_Datatype datatype, int dest,
    int tag, ADOLC_MPI_Comm comm );

int trace_on(int, int, short);

/* High level driver functions */
/* at first parameter this process-ID */

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

/*********************************************************************/
/* Algorithmic Differentation Programs                               */

/* gradient(rank,size,tag, n, x[n], g[n])          */
int gradient(
    int,int,short,int,double*,double*);

/* hessian(rank,size,tag, n, x[n], H[n][n])         */
int hessian(
    int,int,short,int,double*,double**);

/* jacobian(rank,size,tag, m, n, x[n], J[m][n])                 */
int jacobian(
    int,int,short,int,int,const double*,double**);

/* generating tapes by process id, processes count, used tag, m,n, x[n], y[m] */
void tape_doc(
    int,int,short, int,int, double*, double*);
#endif /*__cplusplus*/

#endif /*HAVE_MPI*/

#endif
/* That's all*/

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
#include <adolc/interfaces_mpi.h>

#define ADOLC_MPI_Datatype MPI_Datatype
#define MPI_ADOUBLE MPI_DOUBLE
#define ADOLC_MPI_COMM_WORLD MPI_COMM_WORLD
#define ADOLC_MPI_Comm MPI_Comm
#define ADOLC_MPI_Op MPI_Op
#define ADOLC_MPI_LAND MPI_LAND
#define ADOLC_MPI_BAND MPI_BAND
#define ADOLC_MPI_LOR MPI_LOR
#define ADOLC_MPI_BOR MPI_BOR
#define ADOLC_MPI_LXOR MPI_LXOR
#define ADOLC_MPI_BXOR MPI_BXOR
#define ADOLC_MPI_MAX MPI_MAX
#define ADOLC_MPI_MIN MPI_MIN
#define ADOLC_MPI_SUM MPI_SUM
#define ADOLC_MPI_PROD MPI_PROD
#define ADOLC_MPI_MINLOC MPI_MINLOC
#define ADOLC_MPI_MAXLOC MPI_MAXLOC

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
extern int process_count;

#ifdef __cplusplus
int ADOLC_MPI_Send(
    adouble *buf, int count, ADOLC_MPI_Datatype datatype, int dest,
    int tag, ADOLC_MPI_Comm comm );

int ADOLC_MPI_Recv(
    adouble *buf, int count, ADOLC_MPI_Datatype datatype, int dest,
    int tag, ADOLC_MPI_Comm comm );

int ADOLC_MPI_Bcast(
    adouble *buf, int count, ADOLC_MPI_Datatype datatype, int root,
    ADOLC_MPI_Comm comm);

int ADOLC_MPI_Reduce(
    adouble *sendbuf, adouble* rec_buf, int count, ADOLC_MPI_Datatype type,
    ADOLC_MPI_Op op, int root, ADOLC_MPI_Comm comm);

int trace_on(int, int, short);

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

/* vec_jac(rank,size,tag, m, n, repeat, x[n], u[m], v[n])                             */
int vec_jac(
    int,int,short,int,int,int,double*,double*,double*);

/* jac_vec(rank,size,tag, m, n, x[n], v[n], u[m]);                                    */
int jac_vec(
    int,int,short,int,int,double*,double*,double*);

/* hess_vec(rank,size,tag, n, x[n], v[n], w[n])                                       */
int hess_vec(
    int,int,short,int,double*,double*,double*);

/* hess_mat(rank,size,tag, n, q, x[n], V[n][q], W[n][q])                              */
int hess_mat(
    int,int,short,int,int,double*,double**,double**);


/* lagra_hess_vec(rank,size, tag, m, n, x[n], v[n], u[m], w[n])                        */
int lagra_hess_vec(
    int,int,short,int,int,double*,double*,double*,double*);

/* generating tapes by process id, processes count, used tag, m,n, x[n], y[m] */
void tape_doc(
    int,int,short, int,int, double*, double*);
#endif /*__cplusplus*/

#ifdef __cplusplus
extern "C" {
#endif

/* C - functions                                   */
int trace_on_p(
    int, int, short);

int gradient_p(
    int,int,short,int,double*,double*);
int hessian_p(
    int,int,short,int,double*,double**);

int jacobian_p(
    int,int,short,int,int,const double*,double**);

int vec_jac_p(
    int,int,short,int,int,int,double*,double*,double*);

int jac_vec_p(
    int,int,short,int,int,double*,double*,double*);

int hess_vec_p(
    int,int,short,int,double*,double*,double*);

int hess_mat_p(
    int,int,short,int,int,double*,double**,double**);

int lagra_hess_vec_p(
    int,int,short,int,int,double*,double*,double*,double*);

void tape_doc_p(
    int,int,short, int,int, double*, double*);

#ifdef __cplusplus
}
#endif

#endif /*HAVE_MPI*/

#endif
/* That's all*/

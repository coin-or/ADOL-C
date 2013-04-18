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

#include <adolc/common.h>
#include <adolc/adouble.h>

#if defined(HAVE_MPI)
#include <adolc/interfaces_mpi.h>

BEGIN_C_DECLS

ADOLC_DLL_EXPORT int ADOLC_MPI_Init(int* a, char*** b);
ADOLC_DLL_EXPORT int ADOLC_MPI_Comm_size(ADOLC_MPI_Comm comm, int* size);
ADOLC_DLL_EXPORT int ADOLC_MPI_Comm_rank(ADOLC_MPI_Comm vomm , int* rank);
ADOLC_DLL_EXPORT int ADOLC_MPI_Get_processor_name(char* a, int* b) ;
ADOLC_DLL_EXPORT int ADOLC_MPI_Barrier(ADOLC_MPI_Comm comm);
ADOLC_DLL_EXPORT int ADOLC_MPI_Finalize() ;

END_C_DECLS

#ifdef __cplusplus
ADOLC_DLL_EXPORT int ADOLC_MPI_Send(
    adouble *buf, int count, ADOLC_MPI_Datatype datatype, int dest,
    int tag, ADOLC_MPI_Comm comm );

ADOLC_DLL_EXPORT int ADOLC_MPI_Recv(
    adouble *buf, int count, ADOLC_MPI_Datatype datatype, int dest,
    int tag, ADOLC_MPI_Comm comm );

ADOLC_DLL_EXPORT int ADOLC_MPI_Bcast(
    adouble *buf, int count, ADOLC_MPI_Datatype datatype, int root,
    ADOLC_MPI_Comm comm);

ADOLC_DLL_EXPORT int ADOLC_MPI_Reduce(
    adouble *sendbuf, adouble* rec_buf, int count, ADOLC_MPI_Datatype type,
    ADOLC_MPI_Op op, int root, ADOLC_MPI_Comm comm);

ADOLC_DLL_EXPORT int ADOLC_MPI_Gather(
    adouble *sendbuf, adouble *recvbuf, int count, ADOLC_MPI_Datatype type,
    int root, MPI_Comm comm);

ADOLC_DLL_EXPORT int ADOLC_MPI_Scatter(
    adouble *sendbuf, int sendcount, adouble *recvbuf,
    int recvcount, ADOLC_MPI_Datatype type, int root, MPI_Comm comm);

ADOLC_DLL_EXPORT int trace_on(int, int, short, int keep = 0);

/*********************************************************************/
/* Algorithmic Differentation Programs                               */
ADOLC_DLL_EXPORT int function(int, int, short, int , int, double*,double*);


/* gradient(rank,size,tag, n, x[n], g[n])          */
ADOLC_DLL_EXPORT int gradient(
    int,int,short,int,double*,double*);

/* hessian(rank,size,tag, n, x[n], H[n][n])         */
ADOLC_DLL_EXPORT int hessian(
    int,int,short,int,double*,double**);

/* jacobian(rank,size,tag, m, n, x[n], J[m][n])                 */
ADOLC_DLL_EXPORT int jacobian(
    int,int,short,int,int,const double*,double**);

/* vec_jac(rank,size,tag, m, n, repeat, x[n], u[m], v[n])                             */
ADOLC_DLL_EXPORT int vec_jac(
    int,int,short,int,int,int,double*,double*,double*);

/* jac_vec(rank,size,tag, m, n, x[n], v[n], u[m]);                                    */
ADOLC_DLL_EXPORT int jac_vec(
    int,int,short,int,int,double*,double*,double*);

/* hess_vec(rank,size,tag, n, x[n], v[n], w[n])                                       */
ADOLC_DLL_EXPORT int hess_vec(
    int,int,short,int,double*,double*,double*);

/* lagra_hess_vec(rank,size, tag, m, n, x[n], v[n], u[m], w[n])                        */
ADOLC_DLL_EXPORT int lagra_hess_vec(
    int,int,short,int,int,double*,double*,double*,double*);

/* generating tapes by process id, processes count, used tag, m,n, x[n], y[m] */
ADOLC_DLL_EXPORT void tape_doc(
    int,int,short, int,int, double*, double*);

/*********************************************************************/
/* Algorithmic Differentation Programs - user defined distribution   */
ADOLC_DLL_EXPORT int function_distrib(
     int, int, short, int , int, double*,double*);

/* gradient(rank,size,tag, n, x[n], g[n])          */
ADOLC_DLL_EXPORT int gradient_distrib(
    int,int,short,int,double*,double*);

/* generating tapes by process id, processes count, used tag, m,n, x[n], y[m] */
ADOLC_DLL_EXPORT void tape_doc_distrib(
    int,int,short, int,int, double*, double*);
/*  routines for tapestats */
ADOLC_DLL_EXPORT void tapestats(int id, int size, short tag, size_t *tape_stats);
ADOLC_DLL_EXPORT int  removeTape(short tapeID, short type, int root);
ADOLC_DLL_EXPORT void printTapeStats(FILE *stream, short tag, int root);

#endif /*__cplusplus*/

BEGIN_C_DECLS
/* C - functions                                   */

ADOLC_DLL_EXPORT int function_mpi(int, int, short, int , int, double*,double*);

ADOLC_DLL_EXPORT int gradient_mpi(
    int,int,short,int,double*,double*);

ADOLC_DLL_EXPORT int hessian_mpi(
    int,int,short,int,double*,double**);

ADOLC_DLL_EXPORT int jacobian_mpi(
    int,int,short,int,int,const double*,double**);

ADOLC_DLL_EXPORT int vec_jac_mpi(
    int,int,short,int,int,int,double*,double*,double*);

ADOLC_DLL_EXPORT int jac_vec_mpi(
    int,int,short,int,int,double*,double*,double*);

ADOLC_DLL_EXPORT int hess_vec_mpi(
    int,int,short,int,double*,double*,double*);

ADOLC_DLL_EXPORT int lagra_hess_vec_mpi(
    int,int,short,int,int,double*,double*,double*,double*);

ADOLC_DLL_EXPORT void tape_doc_mpi(
    int,int,short, int,int, double*, double*);

/* routines for user defined dependends and independends    */
ADOLC_DLL_EXPORT int function_mpi_distrib(
   int, int, short, int , int, double*,double*);

ADOLC_DLL_EXPORT int gradient_mpi_distrib(
    int,int,short,int,double*,double*);
ADOLC_DLL_EXPORT void tape_doc_mpi_distrib(
    int,int,short, int,int, double*, double*);
/*  routines for tapestats */
ADOLC_DLL_EXPORT void tapestats_mpi(int id, int size, short tag, size_t *tape_stats);
ADOLC_DLL_EXPORT int  removeTape_mpi(short tapeID, short type, int root);
ADOLC_DLL_EXPORT void printTapeStats_mpi(FILE *stream, short tag, int root);

END_C_DECLS

#endif /*HAVE_MPI*/

#endif
/* That's all*/

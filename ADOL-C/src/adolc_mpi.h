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

#if defined(HAVE_MPI_MPI_H)
#include <mpi/mpi.h>
#elif defined(HAVE_MPI_H)
#include <mpi.h>
#endif
#if defined(HAVE_MPI)
#include <adolc/interfaces_mpi.h>

#define ADOLC_MPI_Datatype MPI_Datatype
#define MPI_ADOUBLE MPI_DOUBLE
#define ADOLC_MPI_COMM_WORLD MPI_COMM_WORLD
#define ADOLC_MPI_Comm MPI_Comm

typedef enum ADOLC_MPI_Op_t {
    ADOLC_MPI_MAX=100,
    ADOLC_MPI_MIN,
    ADOLC_MPI_SUM,
    ADOLC_MPI_PROD,
    ADOLC_MPI_LAND,
    ADOLC_MPI_BAND,
    ADOLC_MPI_LOR,
    ADOLC_MPI_BOR,
    ADOLC_MPI_LXOR,
    ADOLC_MPI_BXOR,
    ADOLC_MPI_MINLOC,
    ADOLC_MPI_MAXLOC
} ADOLC_MPI_Op;

BEGIN_C_DECLS

ADOLC_DLL_EXPORT int ADOLC_MPI_Init(int* a, char*** b);
ADOLC_DLL_EXPORT int ADOLC_MPI_Comm_size(ADOLC_MPI_Comm comm, int* size);
ADOLC_DLL_EXPORT int ADOLC_MPI_Comm_rank(ADOLC_MPI_Comm vomm , int* rank);
ADOLC_DLL_EXPORT int ADOLC_MPI_Get_processor_name(char* a, int* b) ;
ADOLC_DLL_EXPORT int ADOLC_MPI_Barrier(ADOLC_MPI_Comm comm);
ADOLC_DLL_EXPORT int ADOLC_MPI_Finalize() ;

MPI_Op adolc_to_mpi_op(ADOLC_MPI_Op);

END_C_DECLS

extern int mpi_initialized;
extern int process_count;

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
END_C_DECLS

#endif /*HAVE_MPI*/

#endif
/* That's all*/

/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     sparse/sparsedrivers_mpi.h
 Revision: $Id$
 Contents: This file containts some "Easy To Use" parallel interfaces of sparse package.

 Copyright (c) Benjamin Letschert

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#if !defined (ADOLC_SPARSE_SPARSE_MPI_H)
#define ADOLC_SPARSE_SPARSE_MPI_H 1

#include <adolc/common.h>

#if defined(HAVE_MPI)

#if defined(__cplusplus)
ADOLC_DLL_EXPORT int jac_pat(
    int id, int size, short tag, int depen, int indep, const double *basepoint, unsigned int **crs, int *options);

ADOLC_DLL_EXPORT int hess_pat
( int id, int size, short tag, int indep, const double *basepoint, unsigned int **crs, int option);

ADOLC_DLL_EXPORT int sparse_jac(
    int id,int size ,short tag, int depen, int indep, int repeat, const double *basepoint, int *nnz, unsigned int **rind,
    unsigned int **cind, double **values,int *options );

ADOLC_DLL_EXPORT int sparse_hess
( int id, int size ,short tag ,int indep,int repeat, const double *basepoint, int *nnz ,unsigned int **rind, unsigned int **cind, double **values, int *options);
#endif

BEGIN_C_DECLS
ADOLC_DLL_EXPORT int jac_pat_mpi(
    int id, int size, short tag, int depen, int indep, const double *basepoint, unsigned int **crs, int *options);

ADOLC_DLL_EXPORT int hess_pat_mpi
( int id, int size, short tag, int indep, const double *basepoint, unsigned int **crs, int option);

ADOLC_DLL_EXPORT int sparse_jac_mpi(
    int id,int size ,short tag, int depen, int indep, int repeat, const double *basepoint, int *nnz, unsigned int **rind,
    unsigned int **cind, double **values,int *options );

ADOLC_DLL_EXPORT int sparse_hess_mpi
( int id, int size ,short tag ,int indep,int repeat, const double *basepoint, int *nnz ,unsigned int **rind, unsigned int **cind, double **values, int *options);
END_C_DECLS

#endif

#endif

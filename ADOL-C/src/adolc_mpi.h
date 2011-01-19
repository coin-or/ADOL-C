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

#define MPI_ADOUBLE MPI_DOUBLE
#define ADOLC_MPI_COMM_WORLD MPI_COMM_WORLD

#include <adolc/common.h>
#include <adolc/adouble.h>

#ifdef __cplusplus__
extern "C" {
#endif

int ADOLC_MPI_Init(int* a, char*** b);
int ADOLC_MPI_Comm_size(MPI_Comm comm, int* size);
int ADOLC_MPI_Comm_rank(MPI_Comm comm , int* rank);
int ADOLC_MPI_Get_processor_name(char* a, int* b) ;
int ADOLC_MPI_Barrier(MPI_Comm comm);
int ADOLC_MPI_Finalize() ;
#ifdef __cplusplus__
}
#endif

int ADOLC_MPI_Send(adouble *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm );
int ADOLC_MPI_Recv(adouble *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm );

int trace_on(int, int, short);

/* High level driver functions */
/* at first parameter this process-ID */

/* gradient(rank,size,tag, n, x[n], g[n])          */
int gradient(int,int,short,int,double*,double*); 

/* hessian(rank,size,tag, n, x[n], H[n][n])         */
int hessian(int,int,short,int,double*,double**); 

/* generating tapes by process id, processes count, used tag */
void tape_doc( int,int,short, int,int, double*, double*);

#endif /*HAVE_MPI_MPI_H*/

#endif
/* That's all*/
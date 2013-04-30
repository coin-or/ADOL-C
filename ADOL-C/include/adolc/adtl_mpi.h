/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     adtl_mpi.h
 Revision: $Id$
 Contents: C allocation of arrays of doubles in several dimensions

 Copyright (c) Benjamin Letschert

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#ifndef ADOLC_ADTL_MPI_H
#define ADOLC_ADTL_MPI_H 1

#include <adolc/common.h>

#if defined(HAVE_MPI)
#include <mpi.h>
#define ADTL_MPI_Datatype MPI_Datatype
#define MPI_ADTL MPI_DOUBLE
#define ADTL_MPI_COMM_WORLD MPI_COMM_WORLD
#define ADTL_MPI_Comm MPI_Comm

namespace adtl {

typedef enum ADTL_MPI_Op_t {
    ADTL_MPI_MAX=100,
    ADTL_MPI_MIN,
    ADTL_MPI_SUM,
    ADTL_MPI_PROD,
    ADTL_MPI_LAND,
    ADTL_MPI_BAND,
    ADTL_MPI_LOR,
    ADTL_MPI_BOR,
    ADTL_MPI_LXOR,
    ADTL_MPI_BXOR,
    ADTL_MPI_MINLOC,
    ADTL_MPI_MAXLOC
} ADTL_MPI_Op;

}

#include <adolc/adtl.h>

namespace adtl {

void ADTL_MPI_set_trade(adouble *buf, int count, size_t nd, double *trade);
void ADTL_MPI_get_trade(adouble *buf, int count, size_t nd, double *trade);
void ADTL_MPI_set_trade_uint(adouble *buf, int count, size_t nd, unsigned int *trade);
void ADTL_MPI_get_trade_uint(adouble *buf, int count, size_t nd, unsigned int *trade);

int ADTL_MPI_Init(int* a, char*** b);
int ADTL_MPI_Comm_size(ADTL_MPI_Comm comm, int* size);
int ADTL_MPI_Comm_rank(ADTL_MPI_Comm vomm , int* rank);
int ADTL_MPI_Get_processor_name(char* a, int* b) ;
int ADTL_MPI_Barrier(ADTL_MPI_Comm comm);
int ADTL_MPI_Finalize() ;

MPI_Op adolc_to_mpi_op(ADTL_MPI_Op);
bool same_elem (unsigned int first, unsigned int second);

int ADTL_MPI_Send(
    adouble *buf, int count, ADTL_MPI_Datatype datatype, int dest,
    int tag, ADTL_MPI_Comm comm );

int ADTL_MPI_Recv(
    adouble *buf, int count, ADTL_MPI_Datatype datatype, int dest,
    int tag, ADTL_MPI_Comm comm );

int ADTL_MPI_Bcast(
    adouble *buf, int count, ADTL_MPI_Datatype datatype, int root,
    ADTL_MPI_Comm comm);

int ADTL_MPI_Reduce(
    adouble *sendbuf, adouble* rec_buf, int count, ADTL_MPI_Datatype type,
    ADTL_MPI_Op op, int root, ADTL_MPI_Comm comm);

int ADTL_MPI_Gather(
    adouble *sendbuf, adouble *recvbuf, int count, ADTL_MPI_Datatype type,
    int root, MPI_Comm comm);

int ADTL_MPI_Scatter(
    adouble *sendbuf, int sendcount, adouble *recvbuf,
    int recvcount, ADTL_MPI_Datatype type, int root, MPI_Comm comm);

int ADTL_MPI_Allgather(
    adouble *sendbuf, int sendcount,ADTL_MPI_Datatype stype,
    adouble *recvbuf, int recvcount, ADTL_MPI_Datatype rtype, MPI_Comm comm);

int ADTL_MPI_Allreduce(
    adouble *send_buf, adouble *rec_buf, int count, ADTL_MPI_Datatype type,
    ADTL_MPI_Op op, MPI_Comm comm);

int ADTL_MPI_get_sparse_pattern( adouble *a, unsigned int count, ADTL_MPI_Datatype type,
    int root, ADTL_MPI_Comm comm , unsigned int **&pat);

}

#endif
#endif
/* That's all*/

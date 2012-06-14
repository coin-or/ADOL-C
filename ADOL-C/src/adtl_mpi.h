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
#define ADOLC_ADTL_MPI_H

#ifdef HAVE_MPI
#include <mpi.h>
#include <adolc/adtl.h>

#define ADOLC_MPI_Datatype MPI_Datatype
#define MPI_ADOUBLE MPI_DOUBLE
#define ADOLC_MPI_COMM_WORLD MPI_COMM_WORLD
#define ADOLC_MPI_Comm MPI_Comm

namespace adtl {

int ADOLC_MPI_Init(int* a, char*** b);
int ADOLC_MPI_Comm_size(ADOLC_MPI_Comm comm, int* size);
int ADOLC_MPI_Comm_rank(ADOLC_MPI_Comm vomm , int* rank);
int ADOLC_MPI_Get_processor_name(char* a, int* b) ;
int ADOLC_MPI_Barrier(ADOLC_MPI_Comm comm);
int ADOLC_MPI_Finalize() ;

MPI_Op adolc_to_mpi_op(ADOLC_MPI_Op);

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

int ADOLC_MPI_Gather(
    adouble *sendbuf, adouble *recvbuf, int count, ADOLC_MPI_Datatype type,
    int root, MPI_Comm comm);

int ADOLC_MPI_Scatter(
    adouble *sendbuf, int sendcount, adouble *recvbuf,
    int recvcount, ADOLC_MPI_Datatype type, int root, MPI_Comm comm);

int ADOLC_MPI_Allgather(
    adouble *sendbuf, int sendcount,ADOLC_MPI_Datatype stype,
    adouble *recvbuf, int recvcount, ADOLC_MPI_Datatype rtype, MPI_Comm comm);

int ADOLC_MPI_Allreduce(
    adouble *send_buf, adouble *rec_buf, int count, ADOLC_MPI_Datatype type,
    ADOLC_MPI_Op op, MPI_Comm comm);

}

#endif
#endif
/* That's all*/

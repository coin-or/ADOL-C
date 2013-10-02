#if !defined(ADOLC_AMPISUPPORTADOLC_H)
#define ADOLC_AMPISUPPORTADOLC_H 1

#ifdef ADOLC_AMPI_SUPPORT
#include "ampi/ampi.h"
#include "ampi/libCommon/modified.h"

#if defined(__cplusplus)
extern "C" {
#endif

int ADOLC_TLM_AMPI_Send(void* buf,
                        int count,
                        MPI_Datatype datatype,
                        int src,
                        int tag,
                        AMPI_PairedWith pairedWith,
                        MPI_Comm comm);

int ADOLC_TLM_AMPI_Recv(void* buf,
                        int count,
                        MPI_Datatype datatype,
                        int src,
                        int tag,
                        AMPI_PairedWith pairedWith,
                        MPI_Comm comm,
                        MPI_Status* status);

int ADOLC_TLM_AMPI_Isend (void* buf,
                          int count,
                          MPI_Datatype datatype,
                          int dest,
                          int tag,
                          AMPI_PairedWith pairedWith,
                          MPI_Comm comm,
                          AMPI_Request* request);

int ADOLC_TLM_AMPI_Irecv (void* buf,
                          int count,
                          MPI_Datatype datatype,
                          int src,
                          int tag,
                          AMPI_PairedWith pairedWith,
                          MPI_Comm comm,
                          AMPI_Request* request);

int ADOLC_TLM_AMPI_Wait(AMPI_Request *request,
                        MPI_Status *status);

int ADOLC_TLM_AMPI_Barrier(MPI_Comm comm);

int ADOLC_TLM_AMPI_Gather(void *sendbuf,
                          int sendcnt,
                          MPI_Datatype sendtype,
                          void *recvbuf,
                          int recvcnt,
                          MPI_Datatype recvtype,
                          int root,
                          MPI_Comm comm);

int ADOLC_TLM_AMPI_Scatter(void *sendbuf,
                           int sendcnt,
                           MPI_Datatype sendtype,
                           void *recvbuf,
                           int recvcnt,
                           MPI_Datatype recvtype,
                           int root, MPI_Comm comm);

int ADOLC_TLM_AMPI_Allgather(void *sendbuf,
                             int sendcnt,
                             MPI_Datatype sendtype,
                             void *recvbuf,
                             int recvcnt,
                             MPI_Datatype recvtype,
                             MPI_Comm comm);

int ADOLC_TLM_AMPI_Gatherv(void *sendbuf,
                           int sendcnt,
                           MPI_Datatype sendtype,
                           void *recvbuf,
                           int *recvcnts,
                           int *displs,
                           MPI_Datatype recvtype,
                           int root,
                           MPI_Comm comm);

int ADOLC_TLM_AMPI_Scatterv(void *sendbuf,
                            int *sendcnts,
                            int *displs,
                            MPI_Datatype sendtype,
                            void *recvbuf,
                            int recvcnt,
                            MPI_Datatype recvtype,
                            int root, MPI_Comm comm);

int ADOLC_TLM_AMPI_Allgatherv(void *sendbuf,
                              int sendcnt,
                              MPI_Datatype sendtype,
                              void *recvbuf,
                              int *recvcnts,
                              int *displs,
                              MPI_Datatype recvtype,
                              MPI_Comm comm);

int ADOLC_TLM_AMPI_Bcast(void* buf,
                         int count,
                         MPI_Datatype datatype,
                         int root,
                         MPI_Comm comm);

int ADOLC_TLM_AMPI_Reduce(void* sbuf,
                          void* rbuf,
                          int count,
                          MPI_Datatype datatype,
                          MPI_Op op,
                          int root,
                          MPI_Comm comm);

int ADOLC_TLM_AMPI_Allreduce(void* sbuf,
                             void* rbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op,
                             MPI_Comm comm);

#if defined(__cplusplus)
}
#endif
#endif
#endif

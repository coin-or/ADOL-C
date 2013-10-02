#include "taping_p.h"
#include "oplate.h"
#include "adolc/adouble.h"

#ifdef ADOLC_AMPI_SUPPORT
#include "ampisupportAdolc.h"


int ADOLC_TLM_AMPI_Send(void* buf,
                        int count,
                        MPI_Datatype datatype,
                        int src,
                        int tag,
                        AMPI_PairedWith pairedWith,
                        MPI_Comm comm) {
  // pop stuff here and then call
  return TLM_AMPI_Send(buf,
                      count,
                      datatype,
                      src,
                      tag,
                      pairedWith,
                      comm);
}

int ADOLC_TLM_AMPI_Recv(void* buf,
              int count,
              MPI_Datatype datatype,
              int src,
              int tag,
              AMPI_PairedWith pairedWith,
              MPI_Comm comm,
              MPI_Status* status) {
  return TLM_AMPI_Recv(buf,
                      count,
                      datatype,
                      src,
                      tag,
                      pairedWith,
                      comm,
                      status);
}

int ADOLC_TLM_AMPI_Isend (void* buf,
                int count,
                MPI_Datatype datatype,
                int dest,
                int tag,
                AMPI_PairedWith pairedWith,
                MPI_Comm comm,
                AMPI_Request* request) {
  return TLM_AMPI_Isend(buf,
                       count,
                       datatype,
                       dest,
                       tag,
                       pairedWith,
                       comm,
                       request);
}

int ADOLC_TLM_AMPI_Irecv (void* buf,
                int count,
                MPI_Datatype datatype,
                int src,
                int tag,
                AMPI_PairedWith pairedWith,
                MPI_Comm comm,
                AMPI_Request* request) {
  return TLM_AMPI_Irecv(buf,
                       count,
                       datatype,
                       src,
                       tag,
                       pairedWith,
                       comm,
                       request);
}

int ADOLC_TLM_AMPI_Wait(AMPI_Request *request,
              MPI_Status *status) {
  return TLM_AMPI_Wait(request,
                      status);
}

int ADOLC_TLM_AMPI_Barrier(MPI_Comm comm) {
  return TLM_AMPI_Barrier(comm);
}

int ADOLC_TLM_AMPI_Gather(void *sendbuf,
                int sendcnt,
                MPI_Datatype sendtype,
                void *recvbuf,
                int recvcnt,
                MPI_Datatype recvtype,
                int root,
                MPI_Comm comm) {
  return TLM_AMPI_Gather(sendbuf,
                        sendcnt,
                        sendtype,
                        recvbuf,
                        recvcnt,
                        recvtype,
                        root,
                        comm);
}

int ADOLC_TLM_AMPI_Scatter(void *sendbuf,
                 int sendcnt,
                 MPI_Datatype sendtype,
                 void *recvbuf,
                 int recvcnt,
                 MPI_Datatype recvtype,
                 int root, MPI_Comm comm) {
  return TLM_AMPI_Scatter(sendbuf,
                         sendcnt,
                         sendtype,
                         recvbuf,
                         recvcnt,
                         recvtype,
                         root,
                         comm);
}

int ADOLC_TLM_AMPI_Allgather(void *sendbuf,
                   int sendcnt,
                   MPI_Datatype sendtype,
                   void *recvbuf,
                   int recvcnt,
                   MPI_Datatype recvtype,
                   MPI_Comm comm) {
  return TLM_AMPI_Allgather(sendbuf,
                           sendcnt,
                           sendtype,
                           recvbuf,
                           recvcnt,
                           recvtype,
                           comm);
}

int ADOLC_TLM_AMPI_Gatherv(void *sendbuf,
                 int sendcnt,
                 MPI_Datatype sendtype,
                 void *recvbuf,
                 int *recvcnts,
                 int *displs,
                 MPI_Datatype recvtype,
                 int root,
                 MPI_Comm comm) {
  return TLM_AMPI_Gatherv(sendbuf,
			 sendcnt,
			 sendtype,
			 recvbuf,
			 recvcnts,
			 displs,
			 recvtype,
			 root,
			 comm);
}

int ADOLC_TLM_AMPI_Scatterv(void *sendbuf,
                  int *sendcnts,
                  int *displs,
                  MPI_Datatype sendtype,
                  void *recvbuf,
                  int recvcnt,
                  MPI_Datatype recvtype,
                  int root, MPI_Comm comm) {
  return TLM_AMPI_Scatterv(sendbuf,
			  sendcnts,
			  displs,
			  sendtype,
			  recvbuf,
			  recvcnt,
			  recvtype,
			  root,
			  comm);
}

int ADOLC_TLM_AMPI_Allgatherv(void *sendbuf,
                    int sendcnt,
                    MPI_Datatype sendtype,
                    void *recvbuf,
                    int *recvcnts,
                    int *displs,
                    MPI_Datatype recvtype,
                    MPI_Comm comm) {
  return TLM_AMPI_Allgatherv(sendbuf,
                           sendcnt,
                           sendtype,
                           recvbuf,
                           recvcnts,
                           displs,
                           recvtype,
                           comm);
}

int ADOLC_TLM_AMPI_Bcast(void* buf,
	       int count,
	       MPI_Datatype datatype,
	       int root,
	       MPI_Comm comm) {
  return TLM_AMPI_Bcast(buf,
		       count,
		       datatype,
		       root,
		       comm);
}

int ADOLC_TLM_AMPI_Reduce(void* sbuf,
		void* rbuf,
		int count,
		MPI_Datatype datatype,
		MPI_Op op,
		int root,
		MPI_Comm comm) {
  return TLM_AMPI_Reduce(sbuf,
			rbuf,
			count,
			datatype,
			op,
			root,
			comm);
}

int ADOLC_TLM_AMPI_Allreduce(void* sbuf,
                   void* rbuf,
                   int count,
                   MPI_Datatype datatype,
                   MPI_Op op,
                   MPI_Comm comm) {
  return TLM_AMPI_Allreduce(sbuf,
                           rbuf,
                           count,
                           datatype,
                           op,
                           comm);
}

#endif

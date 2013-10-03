#include <cassert>
#include <cstring>
#include "taping_p.h"
#include "oplate.h"
#include "adolc/adouble.h"

#ifdef ADOLC_AMPI_SUPPORT
#include "ampisupportAdolc.h"
#include "ampi/adTool/support.h"
#include "ampi/tape/support.h"


int ADOLC_TLM_AMPI_Send(void* buf,
                        int count,
                        MPI_Datatype datatype,
                        int src,
                        int tag,
                        AMPI_PairedWith pairedWith,
                        MPI_Comm comm) {
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

void allocatePack(void** buf,
                  const locint startLoc,
                  const int count,
                  int &packedCount,
                  const MPI_Datatype& datatype,
                  MPI_Datatype& packedDatatype) {
  int tayCount=ADOLC_CURRENT_TAPE_INFOS.gDegree*ADOLC_CURRENT_TAPE_INFOS.numTay;
  double* doubleBuf=(double*)malloc((tayCount+1)*count*sizeof(double));
  assert(doubleBuf);
  for (int i=0; i<count; ++i) {
    doubleBuf[i*(tayCount+1)]=ADOLC_CURRENT_TAPE_INFOS.dp_T0[startLoc+i];
    memcpy((void*)(doubleBuf+i*(tayCount+1)+1),(void*)(ADOLC_CURRENT_TAPE_INFOS.dpp_T[startLoc+i]),tayCount*sizeof(double));
  }
  *buf=(void*)doubleBuf;
  packedCount=(tayCount+1)*count;
  packedDatatype=ADTOOL_AMPI_FW_rawType(datatype);
}

void unpackDeallocate(void** buf,
                      const locint startLoc,
                      const int count,
                      const int packedCount,
                      const MPI_Datatype& datatype,
                      const MPI_Datatype& packedDatatype) {
  assert(buf);
  int tayCount=ADOLC_CURRENT_TAPE_INFOS.gDegree*ADOLC_CURRENT_TAPE_INFOS.numTay;
  double* doubleBuf=(double*)buf;
  for (int i=0; i<count; ++i) {
    ADOLC_CURRENT_TAPE_INFOS.dp_T0[startLoc+i]=doubleBuf[i*(tayCount+1)];
    memcpy((void*)(ADOLC_CURRENT_TAPE_INFOS.dpp_T[startLoc+i]),(void*)(doubleBuf+i*(tayCount+1)+1),tayCount*sizeof(double));
  }
  free(*buf);
  *buf=NULL;
}

int ADOLC_TLM_AMPI_Bcast(void* buf,
	       int count,
	       MPI_Datatype datatype,
	       int root,
	       MPI_Comm comm) {
  locint startLoc=get_locint_f();
  TAPE_AMPI_read_int(&count);
  TAPE_AMPI_read_MPI_Datatype(&datatype);
  TAPE_AMPI_read_int(&root);
  TAPE_AMPI_read_MPI_Comm(&comm);
  int packedCount=0;
  MPI_Datatype packedDatatype;
  allocatePack(&buf,
               startLoc,
               count,
               packedCount,
               datatype,
               packedDatatype);
  int rc=TLM_AMPI_Bcast(buf,
                        packedCount,
                        packedDatatype,
		        root,
		        comm);
  unpackDeallocate(&buf,
                   startLoc,
                   count,
                   packedCount,
                   datatype,
                   packedDatatype);
  return rc;
}

int ADOLC_TLM_AMPI_Reduce(void* sbuf,
                          void* rbuf,
                          int count,
                          MPI_Datatype datatype,
                          MPI_Op op,
                          int root,
                          MPI_Comm comm) {
  locint rbufStart = get_locint_f();
  locint sbufStart = get_locint_f();
  TAPE_AMPI_read_int(&count);
  int pushedResultsCount; TAPE_AMPI_read_int(&pushedResultsCount);
  double pushedDoubles;
  for (int i=0;i<count;++i) TAPE_AMPI_read_double(&pushedDoubles);
  if (pushedResultsCount>0) for (int i=0;i<count;++i) TAPE_AMPI_read_double(&pushedDoubles);
  TAPE_AMPI_read_int(&pushedResultsCount);
  TAPE_AMPI_read_MPI_Op(&op);
  TAPE_AMPI_read_int(&root); /* root */
  TAPE_AMPI_read_MPI_Comm(&comm);
  TAPE_AMPI_read_MPI_Datatype(&datatype);
  TAPE_AMPI_read_int(&count); /* count again */
  int packedCount=0;
  MPI_Datatype packedDatatype;
  allocatePack(&sbuf,
               sbufStart,
               count,
               packedCount,
               datatype,
               packedDatatype);
  allocatePack(&rbuf,
               rbufStart,
               count,
               packedCount,
               datatype,
               packedDatatype);
  int rc=TLM_AMPI_Reduce(sbuf,
                         rbuf,
                         packedCount,
                         packedDatatype,
                         op,
                         root,
                         comm);
  unpackDeallocate(&sbuf,
                   sbufStart,
                   count,
                   packedCount,
                   datatype,
                   packedDatatype);
  unpackDeallocate(&rbuf,
                   rbufStart,
                   count,
                   packedCount,
                   datatype,
                   packedDatatype);
  return rc;
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

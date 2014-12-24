/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     ampisupportAdolc.cpp
 Revision: $Id$

 Copyright (c) Jean Utke
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
----------------------------------------------------------------------------*/

#include <cassert>
#include <cstring>
#include "taping_p.h"
#include "oplate.h"
#include "adolc/adouble.h"

#ifdef ADOLC_AMPI_SUPPORT
#include "ampisupportAdolc.h"
#include "ampi/adTool/support.h"
#include "ampi/tape/support.h"

extern "C" void ADOLC_TLM_AMPI_PROD(void *invec, void *inoutvec, int *len, MPI_Datatype *dtype);

void ADOLC_TLM_AMPI_PROD(void *invec, void *inoutvec, int *len, MPI_Datatype *dtype){
  int order=ADOLC_CURRENT_TAPE_INFOS.gDegree;
  int dir=ADOLC_CURRENT_TAPE_INFOS.numTay;
  double *in=(double*)invec;
  double *inout=(double*)inoutvec;
  int count=(*len)/((order*dir)+1);
  assert((*len)%((order*dir)+1)==0); // has to evenly divide or something is wrong
  for (int i=0;i<count;++i) {
    for (int d=0;d<dir;++d) {
      // compute the Taylor coefficients highest to lowest per direction
      for (int o=order;o>0;--o) {
        double z=0;
        // do the convolution except for the 0-th coefficients
        for (int conv=1;conv<o;++conv) {
          z+=in[d*order+conv]*inout[d*order+o-conv];
        }
        // do the 0-th coeffients
        z+=in[d*order+o]*inout[0]+in[0]*inout[d*order+o];
        // set the coefficient
        inout[d*order+o]=z;
      }
    }
    // compute the value
    inout[0] *= in[0];
    // advance to the next block
    in+=(order*dir)+1;
    inout+=(order*dir)+1;
  }
}

static MPI_Op ourProdOp;

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
    if (ADOLC_CURRENT_TAPE_INFOS.workMode != ADOLC_ZOS_FORWARD) {
      if (ADOLC_CURRENT_TAPE_INFOS.workMode != ADOLC_FOS_FORWARD) {
        memcpy((void*)(doubleBuf+i*(tayCount+1)+1),(void*)(ADOLC_CURRENT_TAPE_INFOS.dpp_T[startLoc+i]),tayCount*sizeof(double));
      }
      else {  // dpp_T is set as &dp_T !
        doubleBuf[i*2+1]=ADOLC_CURRENT_TAPE_INFOS.dpp_T[0][startLoc+i];
      }
    }
  }
  *buf=(void*)doubleBuf;
  packedCount=(tayCount+1)*count;
  packedDatatype=ADTOOL_AMPI_FW_rawType(datatype);
}

void deallocate(void** buf) {
  free(*buf);
  *buf=NULL;
}

void unpackDeallocate(void** buf,
                      const locint startLoc,
                      const int count,
                      const int packedCount,
                      const MPI_Datatype& datatype,
                      const MPI_Datatype& packedDatatype) {
  assert(buf);
  int tayCount=ADOLC_CURRENT_TAPE_INFOS.gDegree*ADOLC_CURRENT_TAPE_INFOS.numTay;
  double* doubleBuf=(double*)(*buf);
  for (int i=0; i<count; ++i) {
    ADOLC_CURRENT_TAPE_INFOS.dp_T0[startLoc+i]=doubleBuf[i*(tayCount+1)];
    if (ADOLC_CURRENT_TAPE_INFOS.workMode != ADOLC_ZOS_FORWARD) {
      if (ADOLC_CURRENT_TAPE_INFOS.workMode != ADOLC_FOS_FORWARD) {
        memcpy((void*)(ADOLC_CURRENT_TAPE_INFOS.dpp_T[startLoc+i]),(void*)(doubleBuf+i*(tayCount+1)+1),tayCount*sizeof(double));
      }
      else {
        ADOLC_CURRENT_TAPE_INFOS.dpp_T[0][startLoc+i]=doubleBuf[i*2+1];
      }
    }
  }
  deallocate(buf);
}

MPI_Op opForPackedData(const MPI_Op& op) {
  MPI_Op rOp=op;
  if (op==MPI_PROD) rOp=ourProdOp;
  return rOp;
}

void ADOLC_TLM_init() {
  MPI_Op_create(ADOLC_TLM_AMPI_PROD,1,&ourProdOp);
}

int ADOLC_TLM_AMPI_Send(void* buf,
                        int count,
                        MPI_Datatype datatype,
                        int dest,
                        int tag,
                        AMPI_PairedWith pairedWith,
                        MPI_Comm comm) {
  int rc;
  locint startLoc=get_locint_f();
  TAPE_AMPI_read_int(&count);
  TAPE_AMPI_read_MPI_Datatype(&datatype);
  TAPE_AMPI_read_int(&dest);
  TAPE_AMPI_read_int(&tag);
  TAPE_AMPI_read_int((int*)&pairedWith);
  TAPE_AMPI_read_MPI_Comm(&comm);
  int packedCount=0;
  MPI_Datatype packedDatatype;
  allocatePack(&buf,
               startLoc,
               count,
               packedCount,
               datatype,
               packedDatatype);
  rc =   TLM_AMPI_Send(buf,
                       count,
                       datatype,
                       dest,
                       tag,
                       pairedWith,
                       comm);
  unpackDeallocate(&buf,
                   startLoc,
                   count,
                   packedCount,
                   datatype,
                   packedDatatype);
  return rc;
}

int ADOLC_TLM_AMPI_Recv(void* buf,
                        int count,
                        MPI_Datatype datatype,
                        int src,
                        int tag,
                        AMPI_PairedWith pairedWith,
                        MPI_Comm comm,
                        MPI_Status* status) {
  int rc;
  locint startLoc=get_locint_f();
  TAPE_AMPI_read_int(&count);
  TAPE_AMPI_read_MPI_Datatype(&datatype);
  TAPE_AMPI_read_int(&src);
  TAPE_AMPI_read_int(&tag);
  TAPE_AMPI_read_int((int*)&pairedWith);
  TAPE_AMPI_read_MPI_Comm(&comm);
  int packedCount=0;
  MPI_Datatype packedDatatype;
  allocatePack(&buf,
               startLoc,
               count,
               packedCount,
               datatype,
               packedDatatype);
  rc   = TLM_AMPI_Recv(buf,
                       count,
                       datatype,
                       src,
                       tag,
                       pairedWith,
                       comm,
                       status);
  unpackDeallocate(&buf,
                   startLoc,
                   count,
                   packedCount,
                   datatype,
                   packedDatatype);
  return rc;
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
  TAPE_AMPI_read_MPI_Comm(&comm);  
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
  int rc;
  int commSizeForRootOrNull;
  locint startRLoc = 0, startSLoc = 0;
  MPI_Datatype packedRDatatype;
  int packedRCount;
  MPI_Datatype packedSDatatype;
      int packedSCount;
  TAPE_AMPI_read_int(&commSizeForRootOrNull);
  if (commSizeForRootOrNull>0) {
      TAPE_AMPI_read_int(&recvcnt);
      startRLoc = get_locint_f();
      TAPE_AMPI_read_MPI_Datatype(&recvtype);
      allocatePack(&recvbuf,
                   startRLoc,
                   recvcnt,
                   packedRCount,
                   recvtype,
                   packedRDatatype);
  }
  startSLoc = get_locint_f();
  TAPE_AMPI_read_int(&sendcnt);
  TAPE_AMPI_read_MPI_Datatype(&sendtype);
  TAPE_AMPI_read_int(&root);
  TAPE_AMPI_read_MPI_Comm(&comm);
  if (sendcnt > 0) {
      allocatePack(&sendbuf,
                   startSLoc,
                   sendcnt,
                   packedSCount,
                   sendtype,
                   packedSDatatype);
  }
  rc =   TLM_AMPI_Gather(sendbuf,
                         sendcnt,
                         sendtype,
                         recvbuf,
                         recvcnt,
                         recvtype,
                         root,
                         comm);
  if (commSizeForRootOrNull>0) {
      unpackDeallocate(&recvbuf,
                       startRLoc,
                       recvcnt,
                       packedRCount,
                       recvtype,
                       packedRDatatype);
  }
  if (sendcnt > 0) {
      unpackDeallocate(&sendbuf,
                       startSLoc,
                       sendcnt,
                       packedSCount,
                       sendtype,
                       packedSDatatype);
  }
  TAPE_AMPI_read_int(&commSizeForRootOrNull);
  return rc;
}

int ADOLC_TLM_AMPI_Scatter(void *sendbuf,
                           int sendcnt,
                           MPI_Datatype sendtype,
                           void *recvbuf,
                           int recvcnt,
                           MPI_Datatype recvtype,
                           int root, MPI_Comm comm) {
  int rc;
  int commSizeForRootOrNull;
  locint startRLoc = 0, startSLoc = 0;
  MPI_Datatype packedRDatatype;
  int packedRCount;
  MPI_Datatype packedSDatatype;
  int packedSCount;
  TAPE_AMPI_read_int(&commSizeForRootOrNull);
  if (commSizeForRootOrNull>0) {
      TAPE_AMPI_read_int(&recvcnt);
      startRLoc = get_locint_f();
      TAPE_AMPI_read_MPI_Datatype(&recvtype);
      allocatePack(&recvbuf,
                   startRLoc,
                   recvcnt,
                   packedRCount,
                   recvtype,
                   packedRDatatype);
  }
  startSLoc = get_locint_f();
  TAPE_AMPI_read_int(&sendcnt);
  TAPE_AMPI_read_MPI_Datatype(&sendtype);
  TAPE_AMPI_read_int(&root);
  TAPE_AMPI_read_MPI_Comm(&comm);
  if (sendcnt > 0) {
      allocatePack(&sendbuf,
                   startSLoc,
                   sendcnt,
                   packedSCount,
                   sendtype,
                   packedSDatatype);
  }
  rc =   TLM_AMPI_Scatter(sendbuf,
                          sendcnt,
                          sendtype,
                          recvbuf,
                          recvcnt,
                          recvtype,
                          root,
                          comm);
  if (commSizeForRootOrNull>0) {
      unpackDeallocate(&recvbuf,
                       startRLoc,
                       recvcnt,
                       packedRCount,
                       recvtype,
                       packedRDatatype);
  }
  if (sendcnt > 0) {
      unpackDeallocate(&sendbuf,
                       startSLoc,
                       sendcnt,
                       packedSCount,
                       sendtype,
                       packedSDatatype);
  }
  TAPE_AMPI_read_int(&commSizeForRootOrNull);
  return rc;
}

int ADOLC_TLM_AMPI_Allgather(void *sendbuf,
                             int sendcnt,
                             MPI_Datatype sendtype,
                             void *recvbuf,
                             int recvcnt,
                             MPI_Datatype recvtype,
                             MPI_Comm comm) {
  int rc;
  int rootPlaceholder;
  int commSizeForRootOrNull;
  locint startRLoc = 0, startSLoc = 0;
  MPI_Datatype packedRDatatype;
  int packedRCount;
  MPI_Datatype packedSDatatype;
  int packedSCount;
  TAPE_AMPI_read_int(&commSizeForRootOrNull);
  if (commSizeForRootOrNull>0) {
      TAPE_AMPI_read_int(&recvcnt);
      startRLoc = get_locint_f();
      TAPE_AMPI_read_MPI_Datatype(&recvtype);
      allocatePack(&recvbuf,
                   startRLoc,
                   recvcnt,
                   packedRCount,
                   recvtype,
                   packedRDatatype);
  }
  startSLoc = get_locint_f();
  TAPE_AMPI_read_int(&sendcnt);
  TAPE_AMPI_read_MPI_Datatype(&sendtype);
  TAPE_AMPI_read_int(&rootPlaceholder);
  TAPE_AMPI_read_MPI_Comm(&comm);
  if (sendcnt > 0) {
      allocatePack(&sendbuf,
                   startSLoc,
                   sendcnt,
                   packedSCount,
                   sendtype,
                   packedSDatatype);
  }
  rc =   TLM_AMPI_Allgather(sendbuf,
                            sendcnt,
                            sendtype,
                            recvbuf,
                            recvcnt,
                            recvtype,
                            comm);
  if (commSizeForRootOrNull>0) {
      unpackDeallocate(&recvbuf,
                       startRLoc,
                       recvcnt,
                       packedRCount,
                       recvtype,
                       packedRDatatype);
  }
  if (sendcnt > 0) {
      unpackDeallocate(&sendbuf,
                       startSLoc,
                       sendcnt,
                       packedSCount,
                       sendtype,
                       packedSDatatype);
  }
  TAPE_AMPI_read_int(&commSizeForRootOrNull);
  return rc;
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
  int rc,i;
  int commSizeForRootOrNull;
  locint startRLoc, startSLoc;
  MPI_Datatype packedRDatatype, packedSDatatype;
  int packedRCount, packedSCount;
  int totalrecvcnt = 0;
  TAPE_AMPI_read_int(&commSizeForRootOrNull);
  for(i=0;i<commSizeForRootOrNull;++i) {
      TAPE_AMPI_read_int(&recvcnts[i]);
      TAPE_AMPI_read_int(&displs[i]);
      if ((recvcnts[i]>0) &&
          (totalrecvcnt<displs[i]+recvcnts[i]))
          totalrecvcnt=displs[i]+recvcnts[i];
  }
  if (commSizeForRootOrNull>0) {
      startRLoc = get_locint_f();
      TAPE_AMPI_read_MPI_Datatype(&recvtype);
      allocatePack(&recvbuf,
                   startRLoc,
                   totalrecvcnt,
                   packedRCount,
                   recvtype,
                   packedRDatatype);
  }
  startSLoc = get_locint_f();
  TAPE_AMPI_read_int(&sendcnt);
  TAPE_AMPI_read_MPI_Datatype(&sendtype);
  TAPE_AMPI_read_int(&root);
  TAPE_AMPI_read_MPI_Comm(&comm);
  if (sendcnt > 0) {
      allocatePack(&sendbuf,
                   startSLoc,
                   sendcnt,
                   packedSCount,
                   sendtype,
                   packedSDatatype);
  }
  rc =   TLM_AMPI_Gatherv(sendbuf,
                          sendcnt,
                          sendtype,
                          recvbuf,
                          recvcnts,
                          displs,
                          recvtype,
                          root,
                          comm);
  if (sendcnt > 0) {
     unpackDeallocate(&sendbuf,
                       startSLoc,
                       sendcnt,
                       packedSCount,
                       sendtype,
                       packedSDatatype);
  }
  if (commSizeForRootOrNull>0) {
      unpackDeallocate(&recvbuf,
                       startRLoc,
                       totalrecvcnt,
                       packedRCount,
                       recvtype,
                       packedRDatatype);
  }
  TAPE_AMPI_read_int(&commSizeForRootOrNull);
  return rc;
}

int ADOLC_TLM_AMPI_Scatterv(void *sendbuf,
                            int *sendcnts,
                            int *displs,
                            MPI_Datatype sendtype,
                            void *recvbuf,
                            int recvcnt,
                            MPI_Datatype recvtype,
                            int root, MPI_Comm comm) {
  int rc,i;
  int commSizeForRootOrNull;
  locint startRLoc, startSLoc;
  MPI_Datatype packedRDatatype, packedSDatatype;
  int packedRCount, packedSCount;
  int totalsendcnt = 0;
  TAPE_AMPI_read_int(&commSizeForRootOrNull);
  for(i=0;i<commSizeForRootOrNull;++i) {
      TAPE_AMPI_read_int(&sendcnts[i]);
      TAPE_AMPI_read_int(&displs[i]);
      if ((sendcnts[i]>0) &&
          (totalsendcnt<displs[i]+sendcnts[i]))
          totalsendcnt=displs[i]+sendcnts[i];
  }
  if (commSizeForRootOrNull>0) {
      startSLoc = get_locint_f();
      TAPE_AMPI_read_MPI_Datatype(&sendtype);
      allocatePack(&sendbuf,
                   startSLoc,
                   totalsendcnt,
                   packedSCount,
                   sendtype,
                   packedSDatatype);
  }
  startRLoc = get_locint_f();
  TAPE_AMPI_read_int(&recvcnt);
  TAPE_AMPI_read_MPI_Datatype(&recvtype);
  TAPE_AMPI_read_int(&root);
  TAPE_AMPI_read_MPI_Comm(&comm);
  if (recvcnt > 0) {
      allocatePack(&recvbuf,
                   startRLoc,
                   recvcnt,
                   packedRCount,
                   recvtype,
                   packedRDatatype);
  }
  rc =   TLM_AMPI_Scatterv(sendbuf,
                           sendcnts,
                           displs,
                           sendtype,
                           recvbuf,
                           recvcnt,
                           recvtype,
                           root,
                           comm);
  if (commSizeForRootOrNull>0) {
     unpackDeallocate(&sendbuf,
                       startSLoc,
                       totalsendcnt,
                       packedSCount,
                       sendtype,
                       packedSDatatype);
  }
  if (recvcnt > 0) {
      unpackDeallocate(&recvbuf,
                       startRLoc,
                       recvcnt,
                       packedRCount,
                       recvtype,
                       packedRDatatype);
  }
  TAPE_AMPI_read_int(&commSizeForRootOrNull);
  return rc;
}

int ADOLC_TLM_AMPI_Allgatherv(void *sendbuf,
                              int sendcnt,
                              MPI_Datatype sendtype,
                              void *recvbuf,
                              int *recvcnts,
                              int *displs,
                              MPI_Datatype recvtype,
                              MPI_Comm comm) {
  int rc,i;
  int commSizeForRootOrNull, rootPlaceholder;
  locint startRLoc, startSLoc;
  MPI_Datatype packedRDatatype, packedSDatatype;
  int packedRCount, packedSCount;
  int totalrecvcnt = 0;
  TAPE_AMPI_read_int(&commSizeForRootOrNull);
  for(i=0;i<commSizeForRootOrNull;++i) {
      TAPE_AMPI_read_int(&recvcnts[i]);
      TAPE_AMPI_read_int(&displs[i]);
      if ((recvcnts[i]>0) &&
          (totalrecvcnt<displs[i]+recvcnts[i]))
          totalrecvcnt=displs[i]+recvcnts[i];
  }
  if (commSizeForRootOrNull>0) {
      startRLoc = get_locint_f();
      TAPE_AMPI_read_MPI_Datatype(&recvtype);
      allocatePack(&recvbuf,
                   startRLoc,
                   totalrecvcnt,
                   packedRCount,
                   recvtype,
                   packedRDatatype);
  }
  startSLoc = get_locint_f();
  TAPE_AMPI_read_int(&sendcnt);
  TAPE_AMPI_read_MPI_Datatype(&sendtype);
  TAPE_AMPI_read_int(&rootPlaceholder);
  TAPE_AMPI_read_MPI_Comm(&comm);
  if (sendcnt > 0) {
      allocatePack(&sendbuf,
                   startSLoc,
                   sendcnt,
                   packedSCount,
                   sendtype,
                   packedSDatatype);
  }
  rc =   TLM_AMPI_Allgatherv(sendbuf,
                             sendcnt,
                             sendtype,
                             recvbuf,
                             recvcnts,
                             displs,
                             recvtype,
                             comm);
  if (sendcnt > 0) {
     unpackDeallocate(&sendbuf,
                       startSLoc,
                       sendcnt,
                       packedSCount,
                       sendtype,
                       packedSDatatype);
  }
  if (commSizeForRootOrNull>0) {
      unpackDeallocate(&recvbuf,
                       startRLoc,
                       totalrecvcnt,
                       packedRCount,
                       recvtype,
                       packedRDatatype);
  }
  TAPE_AMPI_read_int(&commSizeForRootOrNull);
  return rc;
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
  int myRank; MPI_Comm_rank(comm,&myRank);
  allocatePack(&sbuf,
               sbufStart,
               count,
               packedCount,
               datatype,
               packedDatatype);
  if (myRank==root) {
    allocatePack(&rbuf,
                 rbufStart,
                 count,
                 packedCount,
                 datatype,
                 packedDatatype);
  }
  MPI_Op packedOp=opForPackedData(op);
  int rc=TLB_AMPI_Reduce(sbuf,
                         rbuf,
                         packedCount,
                         packedDatatype,
                         packedOp,
                         root,
                         comm);
  deallocate(&sbuf);
  if (myRank==root) {
    unpackDeallocate(&rbuf,
                     rbufStart,
                     count,
                     packedCount,
                     datatype,
                     packedDatatype);
  }
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

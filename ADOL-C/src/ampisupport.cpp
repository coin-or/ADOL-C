#include <cassert>

#include "ampi/ampi.h"
#include "ampi/adTool/support.h"
#include "ampi/tape/support.h"
#include "ampi/libCommon/modified.h"

#include "taping_p.h"
#include "oplate.h"
#include "adolc/adouble.h"

void ADTOOL_AMPI_pushSRinfo(void* buf, 
			    int count,
			    MPI_Datatype datatype, 
			    int endPoint, 
			    int tag,
			    enum AMPI_PairedWith_E pairedWith,
			    MPI_Comm comm) { 
  if (count>0) { 
    assert(buf);
    locint start=((adouble*)(buf))->loc();
    locint end=(((adouble*)(buf))+(count-1))->loc();
    assert(start+count-1==end); // buf must have consecutive ascending locations 
    ADOLC_PUT_LOCINT(start); 
  }
  else {
    ADOLC_PUT_LOCINT(0); // have to put something 
  }    
  TAPE_AMPI_push_int(count);
  TAPE_AMPI_push_MPI_Datatype(datatype);
  TAPE_AMPI_push_int(endPoint);
  TAPE_AMPI_push_int(tag);
  TAPE_AMPI_push_int(pairedWith);
  TAPE_AMPI_push_MPI_Comm(comm);
}

void ADTOOL_AMPI_popSRinfo(void** buf, 
			   int* count,
			   MPI_Datatype* datatype, 
			   int* endPoint, 
			   int* tag,
			   AMPI_PairedWith_E* pairedWith,
			   MPI_Comm* comm) {
  TAPE_AMPI_pop_MPI_Comm(comm);
  TAPE_AMPI_pop_int((int*)pairedWith);
  TAPE_AMPI_pop_int(tag);
  TAPE_AMPI_pop_int(endPoint);
  TAPE_AMPI_pop_MPI_Datatype(datatype);
  TAPE_AMPI_pop_int(count);
  *buf=(void*)(&(ADOLC_CURRENT_TAPE_INFOS.rp_A[get_locint_r()]));
}

void ADTOOL_AMPI_push_CallCode(enum AMPI_PairedWith_E thisCall) { 
  
  switch(thisCall) { 
  case AMPI_WAIT:
    put_op(ampi_wait);
    break;
  case AMPI_SEND:
    put_op(ampi_send);
    break;
  case AMPI_RECV:
    put_op(ampi_recv);
    break;
  case AMPI_ISEND:
    put_op(ampi_isend);
    break;
  case AMPI_IRECV:
    put_op(ampi_irecv);
    break;
  default:
    assert(0);
    break;
  } 
}

void ADTOOL_AMPI_pop_CallCode(enum AMPI_PairedWith_E *thisCall) { 
  assert(0);
}

void ADTOOL_AMPI_push_AMPI_Request(struct AMPI_Request_S  *ampiRequest) { 
  ADTOOL_AMPI_pushSRinfo(ampiRequest->buf, 
		         ampiRequest->count,
			 ampiRequest->datatype,
			 ampiRequest->endPoint,
			 ampiRequest->tag,
			 ampiRequest->pairedWith,
			 ampiRequest->comm);
  TAPE_AMPI_push_MPI_Request(ampiRequest->tracedRequest);
  TAPE_AMPI_push_int(ampiRequest->origin);
}

void ADTOOL_AMPI_pop_AMPI_Request(struct AMPI_Request_S  *ampiRequest) { 
  TAPE_AMPI_pop_int((int*)&(ampiRequest->origin));
  TAPE_AMPI_pop_MPI_Request(&(ampiRequest->tracedRequest));
  ADTOOL_AMPI_popSRinfo(&(ampiRequest->adjointBuf), 
			&(ampiRequest->count),
			&(ampiRequest->datatype),
			&(ampiRequest->endPoint),
			&(ampiRequest->tag),
			&(ampiRequest->pairedWith),
			&(ampiRequest->comm));
}

void ADTOOL_AMPI_push_request(MPI_Request request) { 
  TAPE_AMPI_push_MPI_Request(request);
} 

MPI_Request ADTOOL_AMPI_pop_request() {
  MPI_Request r;
  TAPE_AMPI_pop_MPI_Request(&r);
  return r;
}

void * ADTOOL_AMPI_rawData(void* activeData) { 
  adouble* adouble_p=(adouble*)activeData; 
  return (void*)(&(ADOLC_GLOBAL_TAPE_VARS.store[adouble_p->loc()]));
}

void * ADTOOL_AMPI_rawAdjointData(void* activeData) {
  return activeData;
}

void ADTOOL_AMPI_mapBufForAdjoint(struct AMPI_Request_S  *ampiRequest,
				  void* buf) { 
  ampiRequest->buf=buf;
}

void ADTOOL_AMPI_setBufForAdjoint(struct AMPI_Request_S  *ampiRequest,
				  void* buf) { 
  /* do nothing */
}

void ADTOOL_AMPI_getAdjointCount(int *count,
				 MPI_Datatype datatype) { 
}

void ADTOOL_AMPI_setAdjointCount(struct AMPI_Request_S  *ampiRequest) { 
  /* for now we keep the count as is but for example in vector mode one would have to multiply by vector length */
  ampiRequest->adjointCount=ampiRequest->count;
}

void ADTOOL_AMPI_setAdjointCountAndTempBuf(struct AMPI_Request_S *ampiRequest) { 
  ADTOOL_AMPI_setAdjointCount(ampiRequest);
  ampiRequest->adjointTempBuf=
      ADTOOL_AMPI_allocateTempBuf(ampiRequest->adjointCount,
          ampiRequest->datatype,
          ampiRequest->comm);
}

void* ADTOOL_AMPI_allocateTempBuf(int adjointCount,
                                  MPI_Datatype dataType,
                                  MPI_Comm comm) {
  size_t s=0;
  void* buf;
  if(dataType==MPI_DOUBLE) s=sizeof(double);
  else if(dataType==MPI_FLOAT) s=sizeof(float);
  else MPI_Abort(comm, MPI_ERR_TYPE);
  buf=malloc(adjointCount*s);
  assert(buf);
  return buf;
}

void ADTOOL_AMPI_releaseAdjointTempBuf(void *tempBuf) {
  free(tempBuf);
}

void ADTOOL_AMPI_adjointIncrement(int adjointCount,
                                  MPI_Datatype datatype,
                                  MPI_Comm comm,
                                  void* target,
                                  void* adjointTarget,
                                  void* checkAdjointTarget,
                                  void *source) {
  for (unsigned int i=0; i<adjointCount; ++i) ((revreal*)(adjointTarget))[i]+=((revreal*)(source))[i];
}

void ADTOOL_AMPI_adjointNullify(int adjointCount,
                                MPI_Datatype datatype,
                                MPI_Comm comm,
                                void* target,
                                void* adjointTarget,
                                void* checkAdjointTarget) {
  for (unsigned int i=0; i<adjointCount; ++i) ((revreal*)(adjointTarget))[i]=0.0;
}


// tracing 

int AMPI_Send(void* buf,
              int count,
              MPI_Datatype datatype,
              AMPI_Activity isActive,
              int src,
              int tag,
              AMPI_PairedWith pairedWith,
              MPI_Comm comm) {
  return FW_AMPI_Send(buf,
                      count,
                      datatype,
                      isActive,
                      src,
                      tag,
                      pairedWith,
                      comm);
}

int AMPI_Recv(void* buf,
              int count,
              MPI_Datatype datatype,
              AMPI_Activity isActive,
              int src,
              int tag,
              AMPI_PairedWith pairedWith,
              MPI_Comm comm,
              MPI_Status* status) {
  return FW_AMPI_Recv(buf,
                      count,
                      datatype,
                      isActive,
                      src,
                      tag,
                      pairedWith,
                      comm,
                      status);
}

int AMPI_Isend (void* buf,
                int count,
                MPI_Datatype datatype,
                AMPI_Activity isActive,
                int dest,
                int tag,
                AMPI_PairedWith pairedWith,
                MPI_Comm comm,
                AMPI_Request* request) {
  return FW_AMPI_Isend(buf,
                       count,
                       datatype,
                       isActive,
                       dest,
                       tag,
                       pairedWith,
                       comm,
                       request);
}

int AMPI_Irecv (void* buf,
                int count,
                MPI_Datatype datatype,
                AMPI_Activity isActive,
                int src,
                int tag,
                AMPI_PairedWith pairedWith,
                MPI_Comm comm,
                AMPI_Request* request) {
  return FW_AMPI_Irecv(buf,
                       count,
                       datatype,
                       isActive,
                       src,
                       tag,
                       pairedWith,
                       comm,
                       request);
}

int AMPI_Wait(AMPI_Request *request,
              MPI_Status *status) {
  return FW_AMPI_Wait(request,
                      status);
}


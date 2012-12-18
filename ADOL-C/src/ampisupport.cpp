#include <cassert>

#include "ampi/ampi.h"
#include "ampi/adTool/support.h"
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
  ADOLC_PUT_LOCINT(count);
  ADOLC_PUT_LOCINT(datatype);
  ADOLC_PUT_LOCINT(endPoint);
  ADOLC_PUT_LOCINT(tag);
  ADOLC_PUT_LOCINT(pairedWith);
  ADOLC_PUT_LOCINT(comm);
}

void ADTOOL_AMPI_popSRinfo(void** buf, 
			   int* count,
			   MPI_Datatype* datatype, 
			   int* endPoint, 
			   int* tag,
			   AMPI_PairedWith_E* pairedWith,
			   MPI_Comm* comm) {
  *comm=get_locint_r();
  *pairedWith=(AMPI_PairedWith_E)get_locint_r();
  *tag=(int)get_locint_r();
  *endPoint=(int)get_locint_r();
  *datatype=(int)get_locint_r();
  *count=(int)get_locint_r();
  *buf=(void*)(&(ADOLC_CURRENT_TAPE_INFOS.rp_T[get_locint_r()]));
}

void ADTOOL_AMPI_push_CallCode(enum AMPI_PairedWith_E thisCall) { 
  
  switch(thisCall) { 
  case AMPI_WAIT:
    put_op(ampi_wait);
    break;
  case AMPI_RECV:
    put_op(ampi_recv);
    break;
  case AMPI_ISEND:
    put_op(ampi_isend);
    break;
  default:
    assert(0);
    break;
  } 
}

void ADTOOL_AMPI_pop_CallCode(enum AMPI_PairedWith_E *thisCall) { 
}

void ADTOOL_AMPI_push_AMPI_Request(struct AMPI_Request_S  *ampiRequest) { 
  ADTOOL_AMPI_pushSRinfo(ampiRequest->buf, 
		         ampiRequest->count,
			 ampiRequest->datatype,
			 ampiRequest->endPoint,
			 ampiRequest->tag,
			 ampiRequest->pairedWith,
			 ampiRequest->comm);
  ADOLC_PUT_LOCINT(ampiRequest->origin);
}

void ADTOOL_AMPI_pop_AMPI_Request(struct AMPI_Request_S  *ampiRequest) { 
}

void * ADTOOL_AMPI_rawData(void* activeData) { 
  adouble* adouble_p=(adouble*)activeData; 
  return (void*)(&(ADOLC_GLOBAL_TAPE_VARS.store[adouble_p->loc()]));
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
}

void ADTOOL_AMPI_setAdjointCountAndTempBuf(struct AMPI_Request_S *ampiRequest) { 
  ADTOOL_AMPI_setAdjointCount(ampiRequest);
  size_t s=0;
  switch(ampiRequest->datatype) { 
  case MPI_DOUBLE: 
    s=sizeof(double);
    break;
  case MPI_FLOAT: 
    s=sizeof(float);
    break;
  default:
    MPI_Abort(ampiRequest->comm, MPI_ERR_TYPE);
    break;
  }
  ampiRequest->adjointTempBuf=(void*)malloc(ampiRequest->adjointCount*s);
  assert(ampiRequest->adjointTempBuf);
}

void ADTOOL_AMPI_releaseAdjointTempBuf(struct AMPI_Request_S *ampiRequest) { 
}

void ADTOOL_AMPI_adjointIncrement(int adjointCount, void* target, void *source) { 
}

void ADTOOL_AMPI_adjointNullify(int adjointCount, void* buf) { 
}


// tracing 

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

int AMPI_Wait(AMPI_Request *request,
              MPI_Status *status) {
  return FW_AMPI_Wait(request,
                      status);
}


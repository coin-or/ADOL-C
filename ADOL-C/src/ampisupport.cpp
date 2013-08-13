#include <cassert>
#include <cstring>
#include <climits>

#include "taping_p.h"
#include "oplate.h"
#include "adolc/adouble.h"

#ifdef ADOLC_AMPI_SUPPORT
#include "ampi/ampi.h"
#include "ampi/adTool/support.h"
#include "ampi/tape/support.h"
#include "ampi/libCommon/modified.h"

void ADTOOL_AMPI_pushBcastInfo(void* buf,
			       int count,
			       MPI_Datatype datatype,
			       int root,
			       MPI_Comm comm) {
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    if (count>0) {
      assert(buf);
      locint start=((adouble*)(buf))->loc();
      locint end=(((adouble*)(buf))+(count-1))->loc();
      assert(start+count-1==end);
      ADOLC_PUT_LOCINT(start);
    }
    else {
      ADOLC_PUT_LOCINT(0);
    }
    TAPE_AMPI_push_int(count);
    TAPE_AMPI_push_MPI_Datatype(datatype);
    TAPE_AMPI_push_int(root);
    TAPE_AMPI_push_MPI_Comm(comm);
  }
}

void ADTOOL_AMPI_popBcastInfo(void** buf,
			      int* count,
			      MPI_Datatype* datatype,
			      int* root,
			      MPI_Comm* comm,
			      void **idx) {
  TAPE_AMPI_pop_MPI_Comm(comm);
  TAPE_AMPI_pop_int(root);
  TAPE_AMPI_pop_MPI_Datatype(datatype);
  TAPE_AMPI_pop_int(count);
  *buf=(void*)(&(ADOLC_CURRENT_TAPE_INFOS.rp_A[get_locint_r()]));
}

void ADTOOL_AMPI_pushDoubleArray(void* buf,
				 int count) {
  int i;
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    for (i=0;i<count;i++) {
      TAPE_AMPI_push_double(((adouble*)(buf))[i].value());
    }
  }
}

void ADTOOL_AMPI_popDoubleArray(void** buf,
				int* count) {
  int i;
  for (i=*count-1;i>=0;i--) {
    TAPE_AMPI_pop_double(((double**)buf)[i]);
  }
}

void ADTOOL_AMPI_pushReduceInfo(void* sbuf,
				void* rbuf,
				void* resultData,
				int pushResultData, /* push resultData if true */
				int count,
				MPI_Datatype datatype,
				MPI_Op op,
				int root,
				MPI_Comm comm) {
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    if (count>0) {
      assert(rbuf);
      locint rstart=((adouble*)(rbuf))->loc();
      locint rend=(((adouble*)(rbuf))+(count-1))->loc();
      assert(rstart+count-1==rend);
      ADOLC_PUT_LOCINT(rstart);
      assert(sbuf);
      locint sstart=((adouble*)(sbuf))->loc();
      locint send=(((adouble*)(sbuf))+(count-1))->loc();
      assert(sstart+count-1==send);
      ADOLC_PUT_LOCINT(sstart);
    }
    else {
      ADOLC_PUT_LOCINT(0);
      ADOLC_PUT_LOCINT(0);
    }
    TAPE_AMPI_push_int(count);
    TAPE_AMPI_push_int(pushResultData);
    ADTOOL_AMPI_pushDoubleArray(sbuf,count);
    if (pushResultData) ADTOOL_AMPI_pushDoubleArray(resultData,count);
    TAPE_AMPI_push_int(pushResultData);
    TAPE_AMPI_push_MPI_Op(op);
    TAPE_AMPI_push_int(root);
    TAPE_AMPI_push_MPI_Comm(comm);
    TAPE_AMPI_push_MPI_Datatype(datatype);
    TAPE_AMPI_push_int(count);
  }
}

void ADTOOL_AMPI_popReduceCountAndType(int* count,
				       MPI_Datatype* datatype) {
  TAPE_AMPI_pop_int(count);
  TAPE_AMPI_pop_MPI_Datatype(datatype);
}

void ADTOOL_AMPI_popReduceInfo(void** sbuf,
			       void** rbuf,
			       void** prevData,
			       void** resultData,
			       int* count,
			       MPI_Op* op,
			       int* root,
			       MPI_Comm* comm,
			       void **idx) {
  int popResultData;
  TAPE_AMPI_pop_MPI_Comm(comm);
  TAPE_AMPI_pop_int(root);
  TAPE_AMPI_pop_MPI_Op(op);
  TAPE_AMPI_pop_int(&popResultData);
  if (popResultData) ADTOOL_AMPI_popDoubleArray(resultData,count);
  ADTOOL_AMPI_popDoubleArray(prevData,count);
  TAPE_AMPI_pop_int(&popResultData);
  TAPE_AMPI_pop_int(count);
  *sbuf=(void*)(&(ADOLC_CURRENT_TAPE_INFOS.rp_A[get_locint_r()]));
  *rbuf=(void*)(&(ADOLC_CURRENT_TAPE_INFOS.rp_A[get_locint_r()]));
}

void ADTOOL_AMPI_pushSRinfo(void* buf, 
			    int count,
			    MPI_Datatype datatype, 
			    int endPoint, 
			    int tag,
			    enum AMPI_PairedWith_E pairedWith,
			    MPI_Comm comm) { 
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    int i, dt_idx = derivedTypeIdx(datatype);
    int total_actives, to_first_active, to_last_active;
    if (isDerivedType(dt_idx)) {
      derivedTypeData* dtdata = getDTypeData();
      int fst_active_idx = dtdata->first_active_indices[dt_idx];
      int lst_active_idx = dtdata->last_active_indices[dt_idx];
      total_actives = dtdata->num_actives[dt_idx]*count;
      to_first_active = dtdata->arrays_of_displacements[dt_idx][fst_active_idx];
      to_last_active = (count-1)*dtdata->mapsizes[dt_idx]
                                                  + dtdata->arrays_of_displacements[dt_idx][lst_active_idx]
                                                                                            + sizeof(adouble)*(dtdata->arrays_of_blocklengths[dt_idx][lst_active_idx]-1);
    }
    else { total_actives = count; to_first_active = 0; to_last_active = count-1; }
    if (count>0) {
      assert(buf);
      locint start=((adouble*)((char*)buf+to_first_active))->loc();
      locint end=((adouble*)((char*)buf+to_last_active))->loc();
      assert(start+total_actives-1==end); // buf must have consecutive ascending locations
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
}

void ADTOOL_AMPI_popSRinfo(void** buf,
			   int* count,
			   MPI_Datatype* datatype,
			   int* endPoint,
			   int* tag,
			   AMPI_PairedWith_E* pairedWith,
			   MPI_Comm* comm,
			   void **idx) {
  TAPE_AMPI_pop_MPI_Comm(comm);
  TAPE_AMPI_pop_int((int*)pairedWith);
  TAPE_AMPI_pop_int(tag);
  TAPE_AMPI_pop_int(endPoint);
  TAPE_AMPI_pop_MPI_Datatype(datatype);
  TAPE_AMPI_pop_int(count);
  locint l  = get_locint_r();
  *buf=(void*)(&(ADOLC_CURRENT_TAPE_INFOS.rp_A[l]));
}

void ADTOOL_AMPI_pushGSinfo(int commSizeForRootOrNull,
                            void *rbuf,
                            int rcnt,
                            MPI_Datatype rtype,
                            void *buf,
                            int  count,
                            MPI_Datatype type,
                            int  root,
                            MPI_Comm comm) {
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    int i;
    int minDispls=INT_MAX;
    TAPE_AMPI_push_int(commSizeForRootOrNull);  // counter at the beginning
    if(commSizeForRootOrNull>0) {
      TAPE_AMPI_push_int(rcnt);
      assert(rbuf);
      locint start=((adouble*)(rbuf))->loc();
      ADOLC_PUT_LOCINT(start);
      TAPE_AMPI_push_MPI_Datatype(rtype);
    }
    if (count>0) {
      assert(buf);
      locint start=((adouble*)(buf))->loc();
      ADOLC_PUT_LOCINT(start);
    }
    else {
      ADOLC_PUT_LOCINT(0); // have to put something
    }
    TAPE_AMPI_push_int(count);
    TAPE_AMPI_push_MPI_Datatype(type);
    TAPE_AMPI_push_int(root);
    TAPE_AMPI_push_MPI_Comm(comm);
    TAPE_AMPI_push_int(commSizeForRootOrNull); // counter at the end
  }
}

void ADTOOL_AMPI_popGScommSizeForRootOrNull(int *commSizeForRootOrNull) {
  TAPE_AMPI_pop_int(commSizeForRootOrNull);
}

void ADTOOL_AMPI_popGSinfo(int commSizeForRootOrNull,
                           void **rbuf,
                           int *rcnt,
                           MPI_Datatype *rtype,
                           void **buf,
                           int *count,
                           MPI_Datatype *type,
                           int *root,
                           MPI_Comm *comm) {
  int i;
  TAPE_AMPI_pop_MPI_Comm(comm);
  TAPE_AMPI_pop_int(root);
  TAPE_AMPI_pop_MPI_Datatype(type);
  TAPE_AMPI_pop_int(count);
  *buf=(void*)(&(ADOLC_CURRENT_TAPE_INFOS.rp_A[get_locint_r()]));
  if (commSizeForRootOrNull>0) {
    TAPE_AMPI_pop_MPI_Datatype(rtype);
    *rbuf=(void*)(&(ADOLC_CURRENT_TAPE_INFOS.rp_A[get_locint_r()]));
    TAPE_AMPI_pop_int(rcnt);
  }
  TAPE_AMPI_pop_int(&commSizeForRootOrNull);
}

void ADTOOL_AMPI_pushGSVinfo(int commSizeForRootOrNull,
                             void *rbuf,
                             int *rcnts,
                             int *displs,
                             MPI_Datatype rtype,
                             void *buf,
                             int  count,
                             MPI_Datatype type,
                             int  root,
                             MPI_Comm comm) { 
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    int i;
    int minDispls=INT_MAX;
    TAPE_AMPI_push_int(commSizeForRootOrNull);  // counter at the beginning
    for (i=0;i<commSizeForRootOrNull;++i) {
      TAPE_AMPI_push_int(rcnts[i]);
      TAPE_AMPI_push_int(displs[i]);
      if (rcnts[i]>0) {
        if (minDispls>displs[i])  minDispls=displs[i];
      }
    }
    if (commSizeForRootOrNull>0) {
      assert(minDispls==0); // don't want to make assumptions about memory layout for nonzero displacements
      assert(rbuf);
      locint start=((adouble*)(rbuf))->loc();
      ADOLC_PUT_LOCINT(start);
      TAPE_AMPI_push_MPI_Datatype(rtype);
    }
    if (count>0) {
      assert(buf);
      locint start=((adouble*)(buf))->loc();
      ADOLC_PUT_LOCINT(start);
    }
    else {
      ADOLC_PUT_LOCINT(0); // have to put something
    }
    TAPE_AMPI_push_int(count);
    TAPE_AMPI_push_MPI_Datatype(type);
    TAPE_AMPI_push_int(root);
    TAPE_AMPI_push_MPI_Comm(comm);
    TAPE_AMPI_push_int(commSizeForRootOrNull); // counter at the end
  }
}

void ADTOOL_AMPI_popGSVinfo(int commSizeForRootOrNull,
			    void **rbuf,
			    int *rcnts,
			    int *displs,
			    MPI_Datatype *rtype,
			    void **buf,
			    int *count,
			    MPI_Datatype *type,
			    int *root,
			    MPI_Comm *comm) { 
  int i;
  TAPE_AMPI_pop_MPI_Comm(comm);
  TAPE_AMPI_pop_int(root);
  TAPE_AMPI_pop_MPI_Datatype(type);
  TAPE_AMPI_pop_int(count);
  *buf=(void*)(&(ADOLC_CURRENT_TAPE_INFOS.rp_A[get_locint_r()]));
  if (commSizeForRootOrNull>0) { 
    TAPE_AMPI_pop_MPI_Datatype(rtype);
    *rbuf=(void*)(&(ADOLC_CURRENT_TAPE_INFOS.rp_A[get_locint_r()]));
  }
  for (i=commSizeForRootOrNull-1;i>=0;--i) { 
    TAPE_AMPI_pop_int(&(displs[i]));
    TAPE_AMPI_pop_int(&(rcnts[i]));
  }
  TAPE_AMPI_pop_int(&commSizeForRootOrNull);
}

void ADTOOL_AMPI_push_CallCode(enum AMPI_PairedWith_E thisCall) { 
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
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
      case AMPI_BCAST:
        put_op(ampi_bcast);
        break;
      case AMPI_REDUCE:
        put_op(ampi_reduce);
        break;
      case AMPI_GATHER:
        put_op(ampi_gather);
        break;
      case AMPI_SCATTER:
        put_op(ampi_scatter);
        break;
      case AMPI_GATHERV:
        put_op(ampi_gatherv);
        break;
      case AMPI_SCATTERV:
        put_op(ampi_scatterv);
        break;
      default:
        assert(0);
        break;
    }
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
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    TAPE_AMPI_push_MPI_Request(ampiRequest->tracedRequest);
    TAPE_AMPI_push_int(ampiRequest->origin);
  }
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
			&(ampiRequest->comm),
			&(ampiRequest->idx));
}

void ADTOOL_AMPI_push_request(MPI_Request request) { 
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) TAPE_AMPI_push_MPI_Request(request);
} 

MPI_Request ADTOOL_AMPI_pop_request() {
  MPI_Request r;
  TAPE_AMPI_pop_MPI_Request(&r);
  return r;
}

void * ADTOOL_AMPI_rawData(void* activeData, int *size) { 
  adouble* adouble_p=(adouble*)activeData; 
  return (void*)(&(ADOLC_GLOBAL_TAPE_VARS.store[adouble_p->loc()]));
}

void * ADTOOL_AMPI_rawDataV(void* activeData, int *counts, int *displs) { 
  adouble* adouble_p=(adouble*)activeData; 
  return (void*)(&(ADOLC_GLOBAL_TAPE_VARS.store[adouble_p->loc()]));
}

void * ADTOOL_AMPI_rawData_DType(void* indata, void* outdata, int* count, int idx) {
  if (idx==-1) return indata; /* not derived type, or only passive elements */
  int i, j, s, p_mapsize, in_offset, out_offset;
  MPI_Datatype datatype;
  derivedTypeData* dtdata = getDTypeData();
  p_mapsize = dtdata->p_mapsizes[idx];
  char *out_addr, *in_addr;
  for (j=0;j<*count;j++) {
    in_offset = j*dtdata->mapsizes[idx];
    out_offset = j*p_mapsize;
    for (i=0;i<dtdata->counts[idx];i++) {
      datatype = dtdata->arrays_of_types[idx][i];
      if (datatype==MPI_UB) break;
      out_addr = (char*)outdata + out_offset + (int)dtdata->arrays_of_p_displacements[idx][i];
      in_addr = (char*)indata + in_offset + (int)dtdata->arrays_of_displacements[idx][i];
      if (ADTOOL_AMPI_isActiveType(datatype)==AMPI_ACTIVE) {
	memcpy(out_addr,
	       ADTOOL_AMPI_rawData((void*)in_addr,&dtdata->arrays_of_blocklengths[idx][i]),
	       sizeof(revreal)*dtdata->arrays_of_blocklengths[idx][i]);
      }
      else {
	if (datatype==MPI_DOUBLE) s = (int)sizeof(double);
	else if (datatype==MPI_INT) s = (int)sizeof(int);
	else if (datatype==MPI_FLOAT) s = (int)sizeof(float);
	else if (datatype==MPI_CHAR) s = (int)sizeof(char);
	memcpy(out_addr,
	       in_addr,
	       s*dtdata->arrays_of_blocklengths[idx][i]);
      }
    }
  }
  return outdata;
}

void * ADTOOL_AMPI_unpackDType(void* indata, void* outdata, int* count, int idx) {
  if (idx==-1) return indata; /* not derived type, or only passive elements */
  int i, j, s, in_offset, out_offset;
  MPI_Datatype datatype;
  derivedTypeData* dtdata = getDTypeData();
  char *out_addr, *in_addr;
  for (j=0;j<*count;j++) {
    in_offset = j*dtdata->p_mapsizes[idx];
    out_offset = j*dtdata->mapsizes[idx];
    for (i=0;i<dtdata->counts[idx];i++) {
      datatype = dtdata->arrays_of_types[idx][i];
      if (datatype==MPI_UB) break;
      out_addr = (char*)outdata + out_offset + (int)dtdata->arrays_of_displacements[idx][i];
      in_addr = (char*)indata + in_offset + (int)dtdata->arrays_of_p_displacements[idx][i];
      if (ADTOOL_AMPI_isActiveType(datatype)==AMPI_ACTIVE) {
	memcpy(ADTOOL_AMPI_rawData((void*)out_addr,&dtdata->arrays_of_blocklengths[idx][i]),
	       in_addr,
	       sizeof(revreal)*dtdata->arrays_of_blocklengths[idx][i]);
      }
      else {
	if (datatype==MPI_DOUBLE) s = (int)sizeof(double);
	else if (datatype==MPI_INT) s = (int)sizeof(int);
	else if (datatype==MPI_FLOAT) s = (int)sizeof(float);
	else if (datatype==MPI_CHAR) s = (int)sizeof(char);
	memcpy(out_addr,
	       in_addr,
	       s*dtdata->arrays_of_blocklengths[idx][i]);
      }
    }
  }
  return outdata;
}

void ADTOOL_AMPI_writeData(void* activeData,int *size) {}

void ADTOOL_AMPI_writeDataV(void* activeData, int *counts, int* displs) {}

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
  int dt_idx = derivedTypeIdx(dataType);
  if (dataType==AMPI_ADOUBLE) s=sizeof(revreal);
  else if(dataType==MPI_DOUBLE) s=sizeof(double);
  else if(dataType==MPI_FLOAT) s=sizeof(float);
  else if(dt_idx!=-1) s=getDTypeData()->p_mapsizes[dt_idx];
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
                                  void *source,
				  void *idx) {
  for (unsigned int i=0; i<adjointCount; ++i) ((revreal*)(adjointTarget))[i]+=((revreal*)(source))[i];
}

void ADTOOL_AMPI_adjointMultiply(int adjointCount,
				 MPI_Datatype datatype,
				 MPI_Comm comm,
				 void* target,
				 void* adjointTarget,
				 void* checkAdjointTarget,
				 void *source,
				 void *idx) {
  for (unsigned int i=0; i<adjointCount; ++i) ((revreal*)(adjointTarget))[i]*=((revreal*)(source))[i];
}

void ADTOOL_AMPI_adjointDivide(int adjointCount,
			       MPI_Datatype datatype,
			       MPI_Comm comm,
			       void* target,
			       void* adjointTarget,
			       void* checkAdjointTarget,
			       void *source,
			       void *idx) {
  for (unsigned int i=0; i<adjointCount; ++i) ((revreal*)(adjointTarget))[i]/=((revreal*)(source))[i];
}

void ADTOOL_AMPI_adjointEquals(int adjointCount,
			       MPI_Datatype datatype,
			       MPI_Comm comm,
			       void* target,
			       void* adjointTarget,
			       void* checkAdjointTarget,
			       void *source1,
			       void *source2,
			       void *idx) {
  for (unsigned int i=0; i<adjointCount; ++i) ((revreal*)(adjointTarget))[i]=((revreal*)(source1))[i]==((revreal*)(source2))[i];
}

void ADTOOL_AMPI_adjointNullify(int adjointCount,
                                MPI_Datatype datatype,
                                MPI_Comm comm,
                                void* target,
                                void* adjointTarget,
                                void* checkAdjointTarget) {
  for (unsigned int i=0; i<adjointCount; ++i) ((revreal*)(adjointTarget))[i]=0.0;
}

AMPI_Activity ADTOOL_AMPI_isActiveType(MPI_Datatype datatype) {
  if (datatype==AMPI_ADOUBLE || datatype==AMPI_AFLOAT) return AMPI_ACTIVE;
  return AMPI_PASSIVE;
};

void ADTOOL_AMPI_setupTypes() {
  MPI_Type_contiguous(1,MPI_DOUBLE,&AMPI_ADOUBLE);
  MPI_Type_commit(&AMPI_ADOUBLE);
  MPI_Type_contiguous(1,MPI_FLOAT,&AMPI_AFLOAT);
  MPI_Type_commit(&AMPI_AFLOAT);

};

// tracing 

int AMPI_Send(void* buf,
              int count,
              MPI_Datatype datatype,
              int src,
              int tag,
              AMPI_PairedWith pairedWith,
              MPI_Comm comm) {
  return FW_AMPI_Send(buf,
                      count,
                      datatype,
                      src,
                      tag,
                      pairedWith,
                      comm);
}

int AMPI_Recv(void* buf,
              int count,
              MPI_Datatype datatype,
              int src,
              int tag,
              AMPI_PairedWith pairedWith,
              MPI_Comm comm,
              MPI_Status* status) {
  return FW_AMPI_Recv(buf,
                      count,
                      datatype,
                      src,
                      tag,
                      pairedWith,
                      comm,
                      status);
}

int AMPI_Isend (void* buf,
                int count,
                MPI_Datatype datatype,
                int dest,
                int tag,
                AMPI_PairedWith pairedWith,
                MPI_Comm comm,
                AMPI_Request* request) {
  return FW_AMPI_Isend(buf,
                       count,
                       datatype,
                       dest,
                       tag,
                       pairedWith,
                       comm,
                       request);
}

int AMPI_Irecv (void* buf,
                int count,
                MPI_Datatype datatype,
                int src,
                int tag,
                AMPI_PairedWith pairedWith,
                MPI_Comm comm,
                AMPI_Request* request) {
  return FW_AMPI_Irecv(buf,
                       count,
                       datatype,
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

int AMPI_Gather(void *sendbuf,
                int sendcnt,
                MPI_Datatype sendtype,
                void *recvbuf,
                int recvcnt,
                MPI_Datatype recvtype,
                int root,
                MPI_Comm comm) {
  return FW_AMPI_Gather(sendbuf,
                        sendcnt,
                        sendtype,
                        recvbuf,
                        recvcnt,
                        recvtype,
                        root,
                        comm);
}

int AMPI_Scatter(void *sendbuf,
                 int sendcnt,
                 MPI_Datatype sendtype,
                 void *recvbuf,
                 int recvcnt,
                 MPI_Datatype recvtype,
                 int root, MPI_Comm comm) {
  return FW_AMPI_Scatter(sendbuf,
                         sendcnt,
                         sendtype,
                         recvbuf,
                         recvcnt,
                         recvtype,
                         root,
                         comm);
}

int AMPI_Gatherv(void *sendbuf,
                 int sendcnt,
                 MPI_Datatype sendtype,
                 void *recvbuf,
                 int *recvcnts,
                 int *displs,
                 MPI_Datatype recvtype,
                 int root,
                 MPI_Comm comm) {
  return FW_AMPI_Gatherv(sendbuf,
			 sendcnt,
			 sendtype,
			 recvbuf,
			 recvcnts,
			 displs,
			 recvtype,
			 root,
			 comm);
}

int AMPI_Scatterv(void *sendbuf,
                  int *sendcnts,
                  int *displs,
                  MPI_Datatype sendtype,
                  void *recvbuf,
                  int recvcnt,
                  MPI_Datatype recvtype,
                  int root, MPI_Comm comm) {
  return FW_AMPI_Scatterv(sendbuf,
			  sendcnts,
			  displs,
			  sendtype,
			  recvbuf,
			  recvcnt,
			  recvtype,
			  root,
			  comm);
}

int AMPI_Bcast(void* buf,
	       int count,
	       MPI_Datatype datatype,
	       int root,
	       MPI_Comm comm) {
  return FW_AMPI_Bcast(buf,
		       count,
		       datatype,
		       root,
		       comm);
}

int AMPI_Reduce(void* sbuf,
		void* rbuf,
		int count,
		MPI_Datatype datatype,
		MPI_Op op,
		int root,
		MPI_Comm comm) {
  return FW_AMPI_Reduce(sbuf,
			rbuf,
			count,
			datatype,
			op,
			root,
			comm);
}
#endif

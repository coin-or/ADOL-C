/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     ampisupport.cpp
 Revision: $Id$

 Copyright (c) Jean Utke
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
----------------------------------------------------------------------------*/

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
#include "ampisupportAdolc.h"

MPI_Comm ADTOOL_AMPI_COMM_WORLD_SHADOW;

int AMPI_Init_NT(int* argc,
		 char*** argv) {
  int rc;
  rc=MPI_Init(argc,
              argv);
  ADTOOL_AMPI_setupTypes();
  ADOLC_TLM_init();
  ourADTOOL_AMPI_FPCollection.pushBcastInfo_fp=&ADTOOL_AMPI_pushBcastInfo;
  ourADTOOL_AMPI_FPCollection.popBcastInfo_fp=&ADTOOL_AMPI_popBcastInfo;
  ourADTOOL_AMPI_FPCollection.pushDoubleArray_fp=&ADTOOL_AMPI_pushDoubleArray;
  ourADTOOL_AMPI_FPCollection.popDoubleArray_fp=&ADTOOL_AMPI_popDoubleArray;
  ourADTOOL_AMPI_FPCollection.pushReduceInfo_fp=&ADTOOL_AMPI_pushReduceInfo;
  ourADTOOL_AMPI_FPCollection.popReduceCountAndType_fp=&ADTOOL_AMPI_popReduceCountAndType;
  ourADTOOL_AMPI_FPCollection.popReduceInfo_fp=&ADTOOL_AMPI_popReduceInfo; 
  ourADTOOL_AMPI_FPCollection.pushSRinfo_fp=&ADTOOL_AMPI_pushSRinfo;
  ourADTOOL_AMPI_FPCollection.popSRinfo_fp=&ADTOOL_AMPI_popSRinfo;
  ourADTOOL_AMPI_FPCollection.pushGSinfo_fp=&ADTOOL_AMPI_pushGSinfo;
  ourADTOOL_AMPI_FPCollection.popGScommSizeForRootOrNull_fp=&ADTOOL_AMPI_popGScommSizeForRootOrNull;
  ourADTOOL_AMPI_FPCollection.popGSinfo_fp=&ADTOOL_AMPI_popGSinfo;
  ourADTOOL_AMPI_FPCollection.pushGSVinfo_fp=&ADTOOL_AMPI_pushGSVinfo;
  ourADTOOL_AMPI_FPCollection.popGSVinfo_fp=&ADTOOL_AMPI_popGSVinfo;
  ourADTOOL_AMPI_FPCollection.push_CallCode_fp=&ADTOOL_AMPI_push_CallCode;
  ourADTOOL_AMPI_FPCollection.push_CallCodeReserve_fp=&ADTOOL_AMPI_push_CallCodeReserve;
  ourADTOOL_AMPI_FPCollection.pop_CallCode_fp=&ADTOOL_AMPI_pop_CallCode;
  ourADTOOL_AMPI_FPCollection.push_AMPI_Request_fp=&ADTOOL_AMPI_push_AMPI_Request;
  ourADTOOL_AMPI_FPCollection.pop_AMPI_Request_fp=&ADTOOL_AMPI_pop_AMPI_Request;
  ourADTOOL_AMPI_FPCollection.push_request_fp=&ADTOOL_AMPI_push_request;
  ourADTOOL_AMPI_FPCollection.pop_request_fp=&ADTOOL_AMPI_pop_request;
  ourADTOOL_AMPI_FPCollection.push_comm_fp=&ADTOOL_AMPI_push_comm;
  ourADTOOL_AMPI_FPCollection.pop_comm_fp=&ADTOOL_AMPI_pop_comm;
  ourADTOOL_AMPI_FPCollection.rawData_fp=&ADTOOL_AMPI_rawData;
  ourADTOOL_AMPI_FPCollection.rawDataV_fp=&ADTOOL_AMPI_rawDataV;
  ourADTOOL_AMPI_FPCollection.packDType_fp=&ADTOOL_AMPI_packDType;
  ourADTOOL_AMPI_FPCollection.unpackDType_fp=&ADTOOL_AMPI_unpackDType;
  ourADTOOL_AMPI_FPCollection.writeData_fp=&ADTOOL_AMPI_writeData;
  ourADTOOL_AMPI_FPCollection.writeDataV_fp=&ADTOOL_AMPI_writeDataV;
  ourADTOOL_AMPI_FPCollection.rawAdjointData_fp=&ADTOOL_AMPI_rawAdjointData;
  ourADTOOL_AMPI_FPCollection.Turn_fp=&ADTOOL_AMPI_Turn;
  ourADTOOL_AMPI_FPCollection.mapBufForAdjoint_fp=&ADTOOL_AMPI_mapBufForAdjoint;
  ourADTOOL_AMPI_FPCollection.setBufForAdjoint_fp=&ADTOOL_AMPI_setBufForAdjoint;
  ourADTOOL_AMPI_FPCollection.getAdjointCount_fp=&ADTOOL_AMPI_getAdjointCount;
  ourADTOOL_AMPI_FPCollection.setAdjointCount_fp=&ADTOOL_AMPI_setAdjointCount;
  ourADTOOL_AMPI_FPCollection.setAdjointCountAndTempBuf_fp=&ADTOOL_AMPI_setAdjointCountAndTempBuf;
  ourADTOOL_AMPI_FPCollection.allocateTempBuf_fp=&ADTOOL_AMPI_allocateTempBuf;
  ourADTOOL_AMPI_FPCollection.releaseAdjointTempBuf_fp=&ADTOOL_AMPI_releaseAdjointTempBuf;
  ourADTOOL_AMPI_FPCollection.incrementAdjoint_fp=&ADTOOL_AMPI_incrementAdjoint;
  ourADTOOL_AMPI_FPCollection.multiplyAdjoint_fp=&ADTOOL_AMPI_multiplyAdjoint;
  ourADTOOL_AMPI_FPCollection.divideAdjoint_fp=&ADTOOL_AMPI_divideAdjoint;
  ourADTOOL_AMPI_FPCollection.equalAdjoints_fp=&ADTOOL_AMPI_equalAdjoints;
  ourADTOOL_AMPI_FPCollection.nullifyAdjoint_fp=&ADTOOL_AMPI_nullifyAdjoint;
  ourADTOOL_AMPI_FPCollection.setupTypes_fp=&ADTOOL_AMPI_setupTypes;
  ourADTOOL_AMPI_FPCollection.cleanupTypes_fp=&ADTOOL_AMPI_cleanupTypes;
  ourADTOOL_AMPI_FPCollection.FW_rawType_fp=&ADTOOL_AMPI_FW_rawType;
  ourADTOOL_AMPI_FPCollection.BW_rawType_fp=&ADTOOL_AMPI_BW_rawType;
  ourADTOOL_AMPI_FPCollection.isActiveType_fp=&ADTOOL_AMPI_isActiveType;
  ourADTOOL_AMPI_FPCollection.allocateTempActiveBuf_fp=&ADTOOL_AMPI_allocateTempActiveBuf;
  ourADTOOL_AMPI_FPCollection.releaseTempActiveBuf_fp=&ADTOOL_AMPI_releaseTempActiveBuf;
  ourADTOOL_AMPI_FPCollection.copyActiveBuf_fp=&ADTOOL_AMPI_copyActiveBuf;
  return rc;
}

locint startLocAssertContiguous(adouble* adoubleBuffer, int count) { 
  locint start=0;
  if (count>0) { 
    start=adoubleBuffer->loc();
    assert(start+count-1==(adoubleBuffer+count-1)->loc()); // buf must have consecutive ascending locations
  }
  return start;
} 

void ADTOOL_AMPI_pushBcastInfo(void* buf,
			       int count,
			       MPI_Datatype datatype,
			       int root,
			       MPI_Comm comm) {
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    int i, dt_idx = derivedTypeIdx(datatype);
    int activeVarCount, bitCountToFirstActive, bitCountToLastActive;
    if (isDerivedType(dt_idx)) {
      derivedTypeData* dtdata = getDTypeData();
      activeVarCount = dtdata->num_actives[dt_idx]*count;
      bitCountToFirstActive = dtdata->first_active_blocks[dt_idx];;
      bitCountToLastActive = (count-1)*dtdata->extents[dt_idx]
	+ dtdata->last_active_blocks[dt_idx]
	+ sizeof(adouble)*(dtdata->last_active_block_lengths[dt_idx]-1);
    }
    else { 
      activeVarCount = count; 
      bitCountToFirstActive = 0; 
      bitCountToLastActive = (count-1)*sizeof(adouble); 
    }
    if (count>0) {
      assert(buf);
      locint start=((adouble*)((char*)buf+bitCountToFirstActive))->loc();
      locint end=((adouble*)((char*)buf+bitCountToLastActive))->loc();
      assert(start+activeVarCount-1==end); // buf must have consecutive ascending locations
      ADOLC_PUT_LOCINT(start);
    }
    else {
      ADOLC_PUT_LOCINT(0); // have to put something
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

void ADTOOL_AMPI_popDoubleArray(double* buf,
				int* count) {
  int i;
  for (i=*count-1;i>=0;i--) {
    TAPE_AMPI_pop_double(&(buf[i]));
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
      ADOLC_PUT_LOCINT(startLocAssertContiguous((adouble*)rbuf,count));
      ADOLC_PUT_LOCINT(startLocAssertContiguous((adouble*)sbuf,count));
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
  if (popResultData) ADTOOL_AMPI_popDoubleArray((double*)(*resultData),count);
  ADTOOL_AMPI_popDoubleArray((double*)(*prevData),count);
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
    int activeVarCount, bitCountToFirstActive, bitCountToLastActive;
    if (isDerivedType(dt_idx)) {
      derivedTypeData* dtdata = getDTypeData();
      activeVarCount = dtdata->num_actives[dt_idx]*count;
      bitCountToFirstActive = dtdata->first_active_blocks[dt_idx];
      bitCountToLastActive = (count-1)*dtdata->extents[dt_idx]
	+ dtdata->last_active_blocks[dt_idx]
	+ sizeof(adouble)*(dtdata->last_active_block_lengths[dt_idx]-1);
    }
    else { activeVarCount = count; bitCountToFirstActive = 0; bitCountToLastActive = (count-1)*sizeof(adouble); }
    if (count>0) {
      assert(buf);
      locint start=((adouble*)((char*)buf+bitCountToFirstActive))->loc();
      locint end=((adouble*)((char*)buf+bitCountToLastActive))->loc();
      assert(start+activeVarCount-1==end); // buf must have consecutive ascending locations
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
  *buf=(void*)(&(ADOLC_CURRENT_TAPE_INFOS.rp_A[get_locint_r()]));
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
    TAPE_AMPI_push_int(commSizeForRootOrNull);  // counter at the beginning
    if(commSizeForRootOrNull>0) {
      TAPE_AMPI_push_int(rcnt);
      assert(rbuf);
      ADOLC_PUT_LOCINT(startLocAssertContiguous((adouble*)rbuf,rcnt));
      TAPE_AMPI_push_MPI_Datatype(rtype);
    }
    locint start=0; // have to put something regardless
    if (buf!=MPI_IN_PLACE && count>0) {
      assert(buf);
      start=startLocAssertContiguous((adouble*)buf,count);
    }
    else {
      count=0;
    }
    ADOLC_PUT_LOCINT(start);
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
  locint bufLoc=get_locint_r();
  if (*count==0) { 
    if (commSizeForRootOrNull) *buf=MPI_IN_PLACE;
    else *buf=0;
  }
  else *buf=(void*)(&(ADOLC_CURRENT_TAPE_INFOS.rp_A[bufLoc]));
  if (commSizeForRootOrNull>0) {
    TAPE_AMPI_pop_MPI_Datatype(rtype);
    *rbuf=(void*)(&(ADOLC_CURRENT_TAPE_INFOS.rp_A[get_locint_r()]));
    TAPE_AMPI_pop_int(rcnt);
  }
  else { 
    // at least initialize to something nonrandom
    // because we know we always have valid addresses passed in here 
    // NOTE JU: may not be true for source transformation...
    *rbuf=0;
    *rcnt=0;
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
    int minDispls=INT_MAX,endOffsetMax=0;
    TAPE_AMPI_push_int(commSizeForRootOrNull);  // counter at the beginning
    for (i=0;i<commSizeForRootOrNull;++i) {
      TAPE_AMPI_push_int(rcnts[i]);
      TAPE_AMPI_push_int(displs[i]);
      if (rcnts[i]>0) {
        if (minDispls>displs[i])  minDispls=displs[i];
	if (endOffsetMax<displs[i]+rcnts[i]) endOffsetMax=displs[i]+rcnts[i]; 
      }
      if (endOffsetMax==0) minDispls=0;
    }
    if (commSizeForRootOrNull>0) {
      assert(minDispls==0); // don't want to make assumptions about memory layout for nonzero displacements
      assert(rbuf);
      ADOLC_PUT_LOCINT(startLocAssertContiguous((adouble*)rbuf,endOffsetMax));
      TAPE_AMPI_push_MPI_Datatype(rtype);
    }
    locint start=0; // have to put something regardless
    if (count>0 && buf!=MPI_IN_PLACE) {
      assert(buf);
      start=startLocAssertContiguous((adouble*)buf,count);
    }
    else {
      count=0;
    }
    ADOLC_PUT_LOCINT(start);
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
  locint bufLoc=get_locint_r();
  if (*count==0) { 
    if (commSizeForRootOrNull) *buf=MPI_IN_PLACE;
    else *buf=0;
  }
  else *buf=(void*)(&(ADOLC_CURRENT_TAPE_INFOS.rp_A[bufLoc]));
  if (commSizeForRootOrNull>0) { 
    TAPE_AMPI_pop_MPI_Datatype(rtype);
    *rbuf=(void*)(&(ADOLC_CURRENT_TAPE_INFOS.rp_A[get_locint_r()]));
  }
  else { 
    // at least initialize to something nonrandom
    // because we know we always have valid addresses passed in here 
    // NOTE JU: may not be true for source transformation...
    *rbuf=0;
  }
  for (i=commSizeForRootOrNull-1;i>=0;--i) { 
    TAPE_AMPI_pop_int(&(displs[i]));
    TAPE_AMPI_pop_int(&(rcnts[i]));
  }
  TAPE_AMPI_pop_int(&commSizeForRootOrNull);
}

void ADTOOL_AMPI_push_CallCode(enum AMPI_CallCode_E thisCall) {
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    switch(thisCall) {
      case AMPI_WAIT:
        put_op(ampi_wait);
        break;
      case AMPI_BARRIER:
         put_op(ampi_barrier);
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
      case AMPI_ALLREDUCE:
        put_op(ampi_allreduce);
        break;
      case AMPI_GATHER:
        put_op(ampi_gather);
        break;
      case AMPI_SCATTER:
        put_op(ampi_scatter);
        break;
      case AMPI_ALLGATHER:
        put_op(ampi_allgather);
        break;
      default:
        assert(0);
        break;
    }
  }
}

void ADTOOL_AMPI_push_CallCodeReserve(enum AMPI_CallCode_E thisCall, unsigned int numlocations) {
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    switch(thisCall) {
      case AMPI_GATHERV:
        put_op_reserve(ampi_gatherv, numlocations);
        break;
      case AMPI_SCATTERV:
        put_op_reserve(ampi_scatterv, numlocations);
        break;
      case AMPI_ALLGATHERV:
         put_op_reserve(ampi_allgatherv, numlocations);
         break;
      default:
        assert(0);
        break;
    }
  }
}

void ADTOOL_AMPI_pop_CallCode(enum AMPI_CallCode_E *thisCall) {
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

void ADTOOL_AMPI_push_comm(MPI_Comm comm) {
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) TAPE_AMPI_push_MPI_Comm(comm);
}

MPI_Comm ADTOOL_AMPI_pop_comm() {
  MPI_Comm c;
  TAPE_AMPI_pop_MPI_Comm(&c);
  return c;
}

void * ADTOOL_AMPI_rawData(void* activeData, int *size) { 
  void *ret=0;
  if (*size>0) {
    adouble* adouble_p=(adouble*)activeData;
    ret=(void*)(&(ADOLC_GLOBAL_TAPE_VARS.store[adouble_p->loc()]));
  }
  return ret;
}

void * ADTOOL_AMPI_rawDataV(void* activeData, int commSize, int *counts, int *displs) { 
  void *ret=NULL;
  int nonNullCount=0;
  int minDispls=INT_MAX;
  for (int i=0; i< commSize; ++i)  { 
    if (counts[i]>nonNullCount) nonNullCount=counts[i];
    if (minDispls>displs[i]) minDispls=displs[i];
  }
  if (nonNullCount>0) { 
    assert(minDispls==0);
    adouble* adouble_p=(adouble*)activeData; 
    ret=(void*)(&(ADOLC_GLOBAL_TAPE_VARS.store[adouble_p->loc()]));
  }
  return ret;
}

void * ADTOOL_AMPI_packDType(void* indata, void* outdata, int count, int idx) {
  if (!isDerivedType(idx)) return indata; /* not derived type, or only passive elements */
  int i, j, s, in_offset, out_offset, dt_idx;
  MPI_Aint p_extent, extent;
  MPI_Datatype datatype;
  derivedTypeData* dtdata = getDTypeData();
  char *out_addr, *in_addr;
  p_extent = dtdata->p_extents[idx];
  extent = dtdata->extents[idx];
  for (j=0;j<count;j++) {
    in_offset = j*extent;
    out_offset = j*p_extent;
    for (i=0;i<dtdata->counts[idx];i++) {
      datatype = dtdata->arrays_of_types[idx][i];
      if (datatype==MPI_UB || datatype==MPI_LB) assert(0);
      dt_idx = derivedTypeIdx(datatype);
      out_addr = (char*)outdata + out_offset + (int)dtdata->arrays_of_p_displacements[idx][i];
      in_addr = (char*)indata + in_offset + (int)dtdata->arrays_of_displacements[idx][i];
      if (ADTOOL_AMPI_isActiveType(datatype)==AMPI_ACTIVE) {
	memcpy(out_addr,
	       ADTOOL_AMPI_rawData((void*)in_addr,&dtdata->arrays_of_blocklengths[idx][i]),
	       sizeof(revreal)*dtdata->arrays_of_blocklengths[idx][i]);
      }
      else if (isDerivedType(dt_idx)) {
	ADTOOL_AMPI_packDType(in_addr,
			      out_addr,
			      dtdata->arrays_of_blocklengths[idx][i],
			      dt_idx);
      }
      else {
	if (datatype==MPI_DOUBLE) s = (int)sizeof(double);
	else if (datatype==MPI_INT) s = (int)sizeof(int);
	else if (datatype==MPI_FLOAT) s = (int)sizeof(float);
	else if (datatype==MPI_CHAR) s = (int)sizeof(char);
	else assert(0);
	memcpy(out_addr,
	       in_addr,
	       s*dtdata->arrays_of_blocklengths[idx][i]);
      }
    }
  }
  return outdata;
}

void * ADTOOL_AMPI_unpackDType(void* indata, void* outdata, int count, int idx) {
  if (!isDerivedType(idx)) return indata; /* not derived type, or only passive elements */
  int i, j, s, in_offset, out_offset, dt_idx;
  MPI_Aint p_extent, extent;
  MPI_Datatype datatype;
  derivedTypeData* dtdata = getDTypeData();
  char *out_addr, *in_addr;
  p_extent = dtdata->p_extents[idx];
  extent = dtdata->extents[idx];
  for (j=0;j<count;j++) {
    in_offset = j*p_extent;
    out_offset = j*extent;
    for (i=0;i<dtdata->counts[idx];i++) {
      datatype = dtdata->arrays_of_types[idx][i];
      if (datatype==MPI_UB || datatype==MPI_LB) assert(0);
      dt_idx = derivedTypeIdx(datatype);
      out_addr = (char*)outdata + out_offset + (int)dtdata->arrays_of_displacements[idx][i];
      in_addr = (char*)indata + in_offset + (int)dtdata->arrays_of_p_displacements[idx][i];
      if (ADTOOL_AMPI_isActiveType(datatype)==AMPI_ACTIVE) {
	memcpy(ADTOOL_AMPI_rawData((void*)out_addr,&dtdata->arrays_of_blocklengths[idx][i]),
	       in_addr,
	       sizeof(revreal)*dtdata->arrays_of_blocklengths[idx][i]);
      }
      else if (isDerivedType(dt_idx)) {
	ADTOOL_AMPI_unpackDType(in_addr,
				out_addr,
				dtdata->arrays_of_blocklengths[idx][i],
				dt_idx);
      }
      else {
	if (datatype==MPI_DOUBLE) s = (int)sizeof(double);
	else if (datatype==MPI_INT) s = (int)sizeof(int);
	else if (datatype==MPI_FLOAT) s = (int)sizeof(float);
	else if (datatype==MPI_CHAR) s = (int)sizeof(char);
	else assert(0);
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

void ADTOOL_AMPI_Turn(void* buf, void* adjointBuf) {}

void ADTOOL_AMPI_setBufForAdjoint(struct AMPI_Request_S  *ampiRequest,
				  void* buf) { 
  /* do nothing */
}

void ADTOOL_AMPI_getAdjointCount(int *count,
				 MPI_Datatype datatype) { 
  int dt_idx = derivedTypeIdx(datatype);
  if (isDerivedType(dt_idx)) *count *= getDTypeData()->num_actives[dt_idx];
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
  else if(isDerivedType(dt_idx)) s=getDTypeData()->p_extents[dt_idx];
  else MPI_Abort(comm, MPI_ERR_TYPE);
  buf=malloc(adjointCount*s);
  assert(buf);
  return buf;
}

void ADTOOL_AMPI_releaseAdjointTempBuf(void *tempBuf) {
  free(tempBuf);
}

void* ADTOOL_AMPI_allocateTempActiveBuf(int count,
					MPI_Datatype datatype,
					MPI_Comm comm) {
  int dt_idx = derivedTypeIdx(datatype);
  if (isDerivedType(dt_idx)) {
    int i, j, extent, struct_offset, block_offset;
    MPI_Datatype blocktype;
    derivedTypeData* dtdata = getDTypeData();
    void* buf;
    extent = dtdata->extents[dt_idx];
    buf = malloc(count*extent);
    assert(buf);
    for (j=0;j<count;j++) {
      struct_offset = j*extent;
      for (i=0;i<dtdata->counts[dt_idx];i++) {
	blocktype = dtdata->arrays_of_types[dt_idx][i];
	if (blocktype==MPI_UB || blocktype==MPI_LB) assert(0);
	block_offset = struct_offset + dtdata->arrays_of_displacements[dt_idx][i];
	if (blocktype==AMPI_ADOUBLE) {
	  new ((void*)((char*)buf + block_offset)) adouble[dtdata->arrays_of_blocklengths[dt_idx][i]];
	}
      }
    }
    return buf;
  }
  else if (datatype==AMPI_ADOUBLE) {
    adouble* buf = new adouble[count];
    assert(buf);
    return buf;
  }
  else assert(0);
}

void ADTOOL_AMPI_releaseTempActiveBuf(void *buf,
				      int count,
				      MPI_Datatype datatype) {
  int dt_idx = derivedTypeIdx(datatype);
  if (isDerivedType(dt_idx)) {
    int i, j, k, extent, struct_offset, block_offset;
    MPI_Datatype blocktype;
    derivedTypeData* dtdata = getDTypeData();
    extent = dtdata->extents[dt_idx];
    for (j=0;j<count;j++) {
      struct_offset = j*extent;
      for (i=0;i<dtdata->counts[dt_idx];i++) {
	blocktype = dtdata->arrays_of_types[dt_idx][i];
	block_offset = struct_offset + dtdata->arrays_of_displacements[dt_idx][i];
	if (blocktype==AMPI_ADOUBLE) {
	  for (k=0;k<dtdata->arrays_of_blocklengths[dt_idx][i];k++) {
	    ((adouble*)((char*)buf + block_offset + k*sizeof(adouble)))->~adouble();
	  }
	}
      }
    }
    free(buf);
  }
  else if (datatype==AMPI_ADOUBLE) delete[] (adouble*)buf;
  else assert(0);
}

void * ADTOOL_AMPI_copyActiveBuf(void* source,
				 void* target,
				 int count,
				 MPI_Datatype datatype,
				 MPI_Comm comm) {
  int s, k, dt_idx = derivedTypeIdx(datatype);
  if (ADTOOL_AMPI_isActiveType(datatype)==AMPI_ACTIVE) {
    for (k=0;k<count;k++) ((adouble*)target)[k] = ((adouble*)source)[k];
  }
  else if (isDerivedType(dt_idx)) {
    int i, j, extent, struct_offset, block_offset;
    MPI_Datatype blocktype;
    derivedTypeData* dtdata = getDTypeData();
    extent = dtdata->extents[dt_idx];
    for (j=0;j<count;j++) {
      struct_offset = j*extent;
      for (i=0;i<dtdata->counts[dt_idx];i++) {
	blocktype = dtdata->arrays_of_types[dt_idx][i];
	if (blocktype==MPI_UB || blocktype==MPI_LB) assert(0);
	block_offset = struct_offset + (int)dtdata->arrays_of_displacements[dt_idx][i];
	if (ADTOOL_AMPI_isActiveType(blocktype)==AMPI_ACTIVE) {
	  for (k=0;k<dtdata->arrays_of_blocklengths[dt_idx][i];k++) {
	    ((adouble*)((char*)target + block_offset))[k] = ((adouble*)((char*)source + block_offset))[k];
	  }
	}
	else {
	  if (blocktype==MPI_DOUBLE) s = sizeof(double);
	  else if (blocktype==MPI_INT) s = sizeof(int);
	  else if (blocktype==MPI_FLOAT) s = sizeof(float);
	  else if (blocktype==MPI_CHAR) s = sizeof(char);
	  memcpy((char*)target + block_offset,
		 (char*)source + block_offset,
		 s*dtdata->arrays_of_blocklengths[dt_idx][i]);
	}
      }
    }
  }
  else assert(0);
  return target;
}

void ADTOOL_AMPI_incrementAdjoint(int adjointCount,
                                  MPI_Datatype datatype,
                                  MPI_Comm comm,
                                  void* target,
                                  void *source,
				  void *idx) {
  for (unsigned int i=0; i<adjointCount; ++i) ((revreal*)(target))[i]+=((revreal*)(source))[i];
}

void ADTOOL_AMPI_multiplyAdjoint(int adjointCount,
				 MPI_Datatype datatype,
				 MPI_Comm comm,
				 void* target,
				 void *source,
				 void *idx) {
  for (unsigned int i=0; i<adjointCount; ++i) ((revreal*)(target))[i]*=((revreal*)(source))[i];
}

void ADTOOL_AMPI_divideAdjoint(int adjointCount,
			       MPI_Datatype datatype,
			       MPI_Comm comm,
			       void* target,
			       void *source,
			       void *idx) {
  for (unsigned int i=0; i<adjointCount; ++i) ((revreal*)(target))[i]/=((revreal*)(source))[i];
}

void ADTOOL_AMPI_equalAdjoints(int adjointCount,
			       MPI_Datatype datatype,
			       MPI_Comm comm,
			       void* target,
			       void *source1,
			       void *source2,
			       void *idx) {
  for (unsigned int i=0; i<adjointCount; ++i) ((revreal*)(target))[i]=((revreal*)(source1))[i]==((revreal*)(source2))[i];
}

void ADTOOL_AMPI_nullifyAdjoint(int adjointCount,
                                MPI_Datatype datatype,
                                MPI_Comm comm,
                                void* target) {
  for (unsigned int i=0; i<adjointCount; ++i) ((revreal*)(target))[i]=0.0;
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

void ADTOOL_AMPI_cleanupTypes() {
  if (AMPI_ADOUBLE!=MPI_DATATYPE_NULL) MPI_Type_free(&AMPI_ADOUBLE);
  if (AMPI_AFLOAT !=MPI_DATATYPE_NULL) MPI_Type_free(&AMPI_AFLOAT);
}

MPI_Datatype ADTOOL_AMPI_FW_rawType(MPI_Datatype datatype) {
  int dt_idx = derivedTypeIdx(datatype);
  if (datatype==AMPI_ADOUBLE) return MPI_DOUBLE;
  else if (datatype==AMPI_AFLOAT) return MPI_FLOAT;
  else if (isDerivedType(dt_idx)) return getDTypeData()->packed_types[dt_idx];
  else return datatype;
}

MPI_Datatype ADTOOL_AMPI_BW_rawType(MPI_Datatype datatype) {
  int dt_idx = derivedTypeIdx(datatype);
  if (datatype==AMPI_ADOUBLE) return MPI_DOUBLE;
  else if (datatype==AMPI_AFLOAT) return MPI_FLOAT;
  else if (isDerivedType(dt_idx)) return MPI_DOUBLE;
  else return datatype;
}

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

int AMPI_Barrier(MPI_Comm comm) {
  return FW_AMPI_Barrier(comm);
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

int AMPI_Allgather(void *sendbuf,
                   int sendcnt,
                   MPI_Datatype sendtype,
                   void *recvbuf,
                   int recvcnt,
                   MPI_Datatype recvtype,
                   MPI_Comm comm) {
  return FW_AMPI_Allgather(sendbuf,
                           sendcnt,
                           sendtype,
                           recvbuf,
                           recvcnt,
                           recvtype,
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

int AMPI_Allgatherv(void *sendbuf,
                    int sendcnt,
                    MPI_Datatype sendtype,
                    void *recvbuf,
                    int *recvcnts,
                    int *displs,
                    MPI_Datatype recvtype,
                    MPI_Comm comm) {
  return FW_AMPI_Allgatherv(sendbuf,
                           sendcnt,
                           sendtype,
                           recvbuf,
                           recvcnts,
                           displs,
                           recvtype,
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
  return FWB_AMPI_Reduce(sbuf,
			rbuf,
			count,
			datatype,
			op,
			root,
			comm);
}

int AMPI_Allreduce(void* sbuf,
                   void* rbuf,
                   int count,
                   MPI_Datatype datatype,
                   MPI_Op op,
                   MPI_Comm comm) {
  return FW_AMPI_Allreduce(sbuf,
                           rbuf,
                           count,
                           datatype,
                           op,
                           comm);
}

#endif

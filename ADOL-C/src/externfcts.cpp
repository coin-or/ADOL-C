/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     externfcts.cpp
 Revision: $Id$
 Contents: functions and data types for extern (differentiated) functions.
 
 Copyright (c) Andreas Kowarz, Jean Utke
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
         
----------------------------------------------------------------------------*/

#include <adolc/externfcts.h>
#include "externfcts_p.h"
#include "taping_p.h"
#include <adolc/adouble.h>
#include "oplate.h"
#include "buffer_temp.h"

#include <cstring>

/****************************************************************************/
/*                                    extern differentiated functions stuff */

#define ADOLC_BUFFER_TYPE \
   Buffer< ext_diff_fct, EDFCTS_BLOCK_SIZE >
static ADOLC_BUFFER_TYPE buffer(edf_zero);

void edf_zero(ext_diff_fct *edf) {
  // sanity settings
  edf->function=0;
  edf->function_iArr=0;

  edf->zos_forward=0;
  edf->zos_forward_iArr=0;

  edf->fos_forward=0;
  edf->fos_forward_iArr=0;
  edf->hos_forward=0;
  edf->hos_forward_iArr=0;
  edf->fov_forward=0;
  edf->fov_forward_iArr=0;
  edf->hov_forward=0;
  edf->hov_forward_iArr=0;

  edf->fos_reverse=0;
  edf->fos_reverse_iArr=0;
  edf->hos_reverse=0;
  edf->hos_reverse_iArr=0;
  edf->fov_reverse=0;
  edf->fov_reverse_iArr=0;
  edf->hov_reverse=0;
  edf->hov_reverse_iArr=0;

  edf->dp_x=0; 
  edf->dp_X=0; 
  edf->dpp_X=0;
  edf->dppp_X=0;
  edf->dp_y=0;  
  edf->dp_Y=0;  
  edf->dpp_Y=0; 
  edf->dppp_Y=0;

  edf->dp_U=0;  
  edf->dpp_U=0; 
  edf->dp_Z=0;  
  edf->dpp_Z=0; 
  edf->dppp_Z=0;

  edf->spp_nz=0;

  edf->max_n=0;
  edf->max_m=0;

  edf->nestedAdolc=true;
  edf->dp_x_changes=true;
  edf->dp_y_priorRequired=true;
}

ext_diff_fct *reg_ext_fct(ADOLC_ext_fct ext_fct) {
  // this call sets edf->index:
  ext_diff_fct * edf=buffer.append();
  edf->function=ext_fct;
  return edf;
}

ext_diff_fct *reg_ext_fct(ADOLC_ext_fct_iArr ext_fct) {
  // this call sets  edf->index:
  ext_diff_fct * edf=buffer.append();
  edf->function_iArr=ext_fct;
  return edf;
}

void call_ext_fct_commonPrior(ext_diff_fct *edfct,
                              int n, adouble *xa,
                              int m, adouble *ya,
                              int &numVals,
                              double *&vals,
                              int &oldTraceFlag) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (xa[n-1].loc()-xa[0].loc()!=(unsigned)n-1 || ya[m-1].loc()-ya[0].loc()!=(unsigned)m-1) fail(ADOLC_EXT_DIFF_LOCATIONGAP);
  if (edfct==NULL) fail(ADOLC_EXT_DIFF_NULLPOINTER_STRUCT);
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    ADOLC_PUT_LOCINT(edfct->index);
    ADOLC_PUT_LOCINT(n);
    ADOLC_PUT_LOCINT(m);
    ADOLC_PUT_LOCINT(xa[0].loc());
    ADOLC_PUT_LOCINT(ya[0].loc());
    ADOLC_PUT_LOCINT(0);               /* keep space for checkpointing index */
    oldTraceFlag=ADOLC_CURRENT_TAPE_INFOS.traceFlag;
    ADOLC_CURRENT_TAPE_INFOS.traceFlag=0;
  }
  else oldTraceFlag=0;
  if (edfct->nestedAdolc) {
    numVals = ADOLC_GLOBAL_TAPE_VARS.storeSize;
    vals = new double[numVals];
    memcpy(vals, ADOLC_GLOBAL_TAPE_VARS.store,
           numVals * sizeof(double));
  }
  edfct->max_n=(edfct->max_n<n)?n:edfct->max_n;
  edfct->max_m=(edfct->max_m<m)?m:edfct->max_m;

  /* update taylor buffer if keep != 0 ; possible double counting as in
   * adouble.cpp => correction in taping.c */

  if (oldTraceFlag != 0) {
    if (edfct->dp_x_changes) ADOLC_CURRENT_TAPE_INFOS.numTays_Tape += n;
    if (edfct->dp_y_priorRequired) ADOLC_CURRENT_TAPE_INFOS.numTays_Tape += m;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors) {
      if (edfct->dp_x_changes) for (int i=0; i<n; ++i) ADOLC_WRITE_SCAYLOR(xa[i].getValue());
      if (edfct->dp_y_priorRequired) for (int i=0; i<m; ++i) ADOLC_WRITE_SCAYLOR(ya[i].getValue());
    }
  }

  for (int i=0; i<n; ++i) edfct->dp_x[i]=xa[i].getValue();
  if (edfct->dp_y_priorRequired) for (int i=0; i<m; ++i) edfct->dp_y[i]=ya[i].getValue();
}

void call_ext_fct_commonPost(ext_diff_fct *edfct,
                              int n, adouble *xa,
                              int m, adouble *ya,
                              int &numVals,
                              double *&vals,
                              int &oldTraceFlag) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  if (edfct->nestedAdolc) {
    memcpy(ADOLC_GLOBAL_TAPE_VARS.store, vals,
        numVals * sizeof(double));
    delete[] vals;
    vals=0;
  }
  /* write back */
  if (edfct->dp_x_changes) for (int i=0; i<n; ++i) xa[i].setValue(edfct->dp_x[i]);
  for (int i=0; i<m; ++i) ya[i].setValue(edfct->dp_y[i]);
  ADOLC_CURRENT_TAPE_INFOS.traceFlag=oldTraceFlag;
}

int call_ext_fct(ext_diff_fct *edfct,
                 int n, adouble *xa,
                 int m, adouble *ya) {
  int ret;
  int numVals = 0;
  double *vals = NULL;
  int oldTraceFlag;
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) put_op(ext_diff);
  call_ext_fct_commonPrior (edfct,n,xa,m,ya,numVals,vals,oldTraceFlag);
  ret=edfct->function(n, edfct->dp_x, m, edfct->dp_y);
  call_ext_fct_commonPost (edfct,n,xa,m,ya,numVals,vals,oldTraceFlag);
  return ret;
}

int call_ext_fct(ext_diff_fct *edfct,
                 int iArrLength, int *iArr,
                 int n, adouble *xa,
                 int m, adouble *ya) {
  int ret;
  int numVals = 0;
  double *vals = NULL;
  int oldTraceFlag;
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op_reserve(ext_diff_iArr,iArrLength+2);
    ADOLC_PUT_LOCINT(iArrLength);
    for (int i=0; i<iArrLength; ++i) ADOLC_PUT_LOCINT(iArr[i]);
    ADOLC_PUT_LOCINT(iArrLength); // do it again so we can read in either direction
  }
  call_ext_fct_commonPrior (edfct,n,xa,m,ya,numVals,vals,oldTraceFlag);
  ret=edfct->function_iArr(iArrLength, iArr, n, edfct->dp_x, m, edfct->dp_y);
  call_ext_fct_commonPost (edfct,n,xa,m,ya,numVals,vals,oldTraceFlag);
  return ret;
}

ext_diff_fct *get_ext_diff_fct( int index ) {
    return buffer.getElement(index);
}

/****************************************************************************/
/*                                                               THAT'S ALL */


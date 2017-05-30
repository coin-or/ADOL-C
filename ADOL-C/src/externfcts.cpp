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

#include "taping_p.h"
#include <adolc/externfcts.h>
#include "externfcts_p.h"
#include <adolc/adouble.h>
#include <adolc/adalloc.h>
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
  if (edf->allmem != NULL)
      free(edf->allmem);
  edf->allmem=NULL;
  edf->user_allocated_mem=0;
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

/*
 * The externfcts.h had a comment previously that said the following:
 ****
 * The user has to preallocate the variables and set the pointers for any of the call back functions 
 * that will be called during trace interpretation.
 * The dimensions given below correspond to the formal arguments in the call back funtions signatures above. 
 * If the dimensions n and m change between multiple calls to the same external function, then the variables 
 * have to be preallocation with the maximum of the respective dimension values. 
 * The dp_x and dp_y pointers have to be valid during both, the tracing phase and the trace interpretation; 
 * all the other pointers are required to be valid only for the trace interpretation.
 ****
 * Doing this now internally saves the user from doing it, as well as updating
 * when using multiple problem sizes.
 */
static void update_ext_fct_memory(ext_diff_fct *edfct, int n, int m) {
  if (edfct->max_n<n || edfct->max_m<m) {
      /* We need memory stored in the edfct dp_x[n], dp_X[n], dp_Z[n], 
       * dp_y[m], dp_Y[m], dp_U[m], dpp_X[n][n], dpp_Y[m][n], 
       * dpp_U[m][m], dpp_Z[m][n]. We have no implementation for higher order
       * so leave it out.
       */
      size_t totalmem = (3*n + 3*m /*+ n*n + 2*n*m + m*m*/)*sizeof(double)
                         + (3*m+n)*sizeof(double*);
      char *tmp;
      if (edfct->allmem != NULL) free(edfct->allmem);
      edfct->allmem = (char*)malloc(totalmem);
      memset(edfct->allmem,0,totalmem);
      edfct->dp_x = (double*)edfct->allmem;
      edfct->dp_y = edfct->dp_x+n;
      edfct->dp_X = edfct->dp_y+m;
      edfct->dp_Y = edfct->dp_X+n;
      edfct->dp_U = edfct->dp_Y+m;
      edfct->dp_Z = edfct->dp_U+m;
      tmp = (char*)(edfct->dp_Z+n);
      edfct->dpp_X = (double**)tmp;
      edfct->dpp_Y = edfct->dpp_X + n;
      edfct->dpp_U = edfct->dpp_Y + m;
      edfct->dpp_Z = edfct->dpp_U + m;
      /*
      tmp = populate_dpp(&edfct->dpp_X, tmp, n,n);
      tmp = populate_dpp(&edfct->dpp_Y, tmp, m,n);
      tmp = populate_dpp(&edfct->dpp_U, tmp, m,m);
      tmp = populate_dpp(&edfct->dpp_Z, tmp, m,n);
      */
  }

  edfct->max_n=(edfct->max_n<n)?n:edfct->max_n;
  edfct->max_m=(edfct->max_m<m)?m:edfct->max_m;
  
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

  if (!edfct->user_allocated_mem)
      update_ext_fct_memory(edfct,n,m);

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
  ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index = edfct->index;
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

static int edfoo_wrapper_function(int n, double *x, int m, double* y) {
    ext_diff_fct* edf;
    EDFobject* ebase;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    // figure out which edf
    edf = get_ext_diff_fct(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
    ebase = reinterpret_cast<EDFobject*>(edf->obj);
    return ebase->function(n,x,m,y);
}
static int edfoo_wrapper_zos_forward(int n, double *x, int m, double* y) {
    ext_diff_fct* edf;
    EDFobject* ebase;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    // figure out which edf
    edf = get_ext_diff_fct(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
    ebase = reinterpret_cast<EDFobject*>(edf->obj);
    return ebase->zos_forward(n,x,m,y);
}

static int edfoo_wrapper_fos_forward(int n, double *dp_x, double *dp_X, int m, double *dp_y, double *dp_Y) {
    ext_diff_fct* edf;
    EDFobject* ebase;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    // figure out which edf
    edf = get_ext_diff_fct(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
    ebase = reinterpret_cast<EDFobject*>(edf->obj);
    return ebase->fos_forward(n,dp_x,dp_X,m,dp_y,dp_Y);
}

static int edfoo_wrapper_fov_forward(int n, double *dp_x, int p, double **dpp_X, int m, double *dp_y, double **dpp_Y) {
    ext_diff_fct* edf;
    EDFobject* ebase;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    // figure out which edf
    edf = get_ext_diff_fct(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
    ebase = reinterpret_cast<EDFobject*>(edf->obj);
    return ebase->fov_forward(n,dp_x,p,dpp_X,m,dp_y,dpp_Y);    
}

static int edfoo_wrapper_fos_reverse(int m, double *dp_U, int n, double *dp_Z, double *dp_x, double *dp_y) {
    ext_diff_fct* edf;
    EDFobject* ebase;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    // figure out which edf
    edf = get_ext_diff_fct(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
    ebase = reinterpret_cast<EDFobject*>(edf->obj);
    return ebase->fos_reverse(m,dp_U,n,dp_Z,dp_x,dp_y);    
}
static int edfoo_wrapper_fov_reverse(int m, int p, double **dpp_U, int n, double **dpp_Z, double *dp_x, double *dp_y) {
    ext_diff_fct* edf;
    EDFobject* ebase;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    // figure out which edf
    edf = get_ext_diff_fct(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
    ebase = reinterpret_cast<EDFobject*>(edf->obj);
    return ebase->fov_reverse(m,p,dpp_U,n,dpp_Z,dp_x,dp_y);    
}

void EDFobject::init_edf(EDFobject* ebase) {
    edf = buffer.append();
    edf->obj = reinterpret_cast<void*>(ebase);
    edf->function = edfoo_wrapper_function;
    edf->zos_forward = edfoo_wrapper_zos_forward;
    edf->fos_forward = edfoo_wrapper_fos_forward;
    edf->fov_forward = edfoo_wrapper_fov_forward;
    edf->fos_reverse = edfoo_wrapper_fos_reverse;
    edf->fov_reverse = edfoo_wrapper_fov_reverse;    
}

static int edfoo_iarr_wrapper_function(int iArrLength, int *iArr, int n, double *x, int m, double* y) {
    ext_diff_fct* edf;
    EDFobject_iArr* ebase;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    // figure out which edf
    edf = get_ext_diff_fct(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
    ebase = reinterpret_cast<EDFobject_iArr*>(edf->obj);
    return ebase->function(iArrLength,iArr,n,x,m,y);
}
static int edfoo_iarr_wrapper_zos_forward(int iArrLength, int *iArr, int n, double *x, int m, double* y) {
    ext_diff_fct* edf;
    EDFobject_iArr* ebase;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    // figure out which edf
    edf = get_ext_diff_fct(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
    ebase = reinterpret_cast<EDFobject_iArr*>(edf->obj);
    return ebase->zos_forward(iArrLength,iArr,n,x,m,y);
}

static int edfoo_iarr_wrapper_fos_forward(int iArrLength, int *iArr, int n, double *dp_x, double *dp_X, int m, double *dp_y, double *dp_Y) {
    ext_diff_fct* edf;
    EDFobject_iArr* ebase;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    // figure out which edf
    edf = get_ext_diff_fct(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
    ebase = reinterpret_cast<EDFobject_iArr*>(edf->obj);
    return ebase->fos_forward(iArrLength,iArr,n,dp_x,dp_X,m,dp_y,dp_Y);
}

static int edfoo_iarr_wrapper_fov_forward(int iArrLength, int *iArr, int n, double *dp_x, int p, double **dpp_X, int m, double *dp_y, double **dpp_Y) {
    ext_diff_fct* edf;
    EDFobject_iArr* ebase;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    // figure out which edf
    edf = get_ext_diff_fct(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
    ebase = reinterpret_cast<EDFobject_iArr*>(edf->obj);
    return ebase->fov_forward(iArrLength,iArr,n,dp_x,p,dpp_X,m,dp_y,dpp_Y);    
}

static int edfoo_iarr_wrapper_fos_reverse(int iArrLength, int *iArr, int m, double *dp_U, int n, double *dp_Z, double *dp_x, double *dp_y) {
    ext_diff_fct* edf;
    EDFobject_iArr* ebase;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    // figure out which edf
    edf = get_ext_diff_fct(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
    ebase = reinterpret_cast<EDFobject_iArr*>(edf->obj);
    return ebase->fos_reverse(iArrLength,iArr,m,dp_U,n,dp_Z,dp_x,dp_y);    
}
static int edfoo_iarr_wrapper_fov_reverse(int iArrLength, int *iArr, int m, int p, double **dpp_U, int n, double **dpp_Z, double *dp_x, double *dp_y) {
    ext_diff_fct* edf;
    EDFobject_iArr* ebase;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    // figure out which edf
    edf = get_ext_diff_fct(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
    ebase = reinterpret_cast<EDFobject_iArr*>(edf->obj);
    return ebase->fov_reverse(iArrLength,iArr,m,p,dpp_U,n,dpp_Z,dp_x,dp_y);    
}

void EDFobject_iArr::init_edf(EDFobject_iArr* ebase) {
    edf = buffer.append();
    edf->obj = reinterpret_cast<void*>(ebase);
    edf->function = edfoo_wrapper_function;
    edf->zos_forward_iArr = edfoo_iarr_wrapper_zos_forward;
    edf->fos_forward_iArr = edfoo_iarr_wrapper_fos_forward;
    edf->fov_forward_iArr = edfoo_iarr_wrapper_fov_forward;
    edf->fos_reverse_iArr = edfoo_iarr_wrapper_fos_reverse;
    edf->fov_reverse_iArr = edfoo_iarr_wrapper_fov_reverse;    
}

/****************************************************************************/
/*                                                               THAT'S ALL */


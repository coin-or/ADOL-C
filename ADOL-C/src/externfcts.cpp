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

#include <adolc/ad_types.h>
#include <adolc/adalloc.h>
#include <adolc/buffer_temp.h>
#include <adolc/externfcts.h>
#include <adolc/externfcts_p.h>
#include <adolc/oplate.h>
#include <adolc/taping_p.h>

#include <cstring>

/****************************************************************************/
/*                                    extern differentiated functions stuff */

#define ADOLC_BUFFER_TYPE Buffer<ext_diff_fct, EDFCTS_BLOCK_SIZE>
static ADOLC_BUFFER_TYPE buffer(edf_zero);

void edf_zero(ext_diff_fct *edf) {
  // sanity settings
  edf->function = 0;
  edf->function_iArr = 0;

  edf->zos_forward = 0;
  edf->zos_forward_iArr = 0;

  edf->fos_forward = 0;
  edf->fos_forward_iArr = 0;
  edf->hos_forward = 0;
  edf->hos_forward_iArr = 0;
  edf->fov_forward = 0;
  edf->fov_forward_iArr = 0;
  edf->hov_forward = 0;
  edf->hov_forward_iArr = 0;

  edf->fos_reverse = 0;
  edf->fos_reverse_iArr = 0;
  edf->hos_reverse = 0;
  edf->hos_reverse_iArr = 0;
  edf->fov_reverse = 0;
  edf->fov_reverse_iArr = 0;
  edf->hov_reverse = 0;
  edf->hov_reverse_iArr = 0;

  edf->dp_x = 0;
  edf->dp_X = 0;
  edf->dpp_X = 0;
  edf->dppp_X = 0;
  edf->dp_y = 0;
  edf->dp_Y = 0;
  edf->dpp_Y = 0;
  edf->dppp_Y = 0;

  edf->dp_U = 0;
  edf->dpp_U = 0;
  edf->dp_Z = 0;
  edf->dpp_Z = 0;
  edf->dppp_Z = 0;

  edf->spp_nz = 0;

  edf->max_n = 0;
  edf->max_m = 0;

  edf->nestedAdolc = true;
  edf->dp_x_changes = true;
  edf->dp_y_priorRequired = true;
  if (edf->allmem != nullptr)
    free(edf->allmem);
  edf->allmem = nullptr;
  edf->user_allocated_mem = 0;
}

ext_diff_fct *reg_ext_fct(ADOLC_ext_fct ext_fct) {
  // this call sets edf->index:
  ext_diff_fct *edf = buffer.append();
  edf->function = ext_fct;
  return edf;
}

ext_diff_fct *reg_ext_fct(ADOLC_ext_fct_iArr ext_fct) {
  // this call sets  edf->index:
  ext_diff_fct *edf = buffer.append();
  edf->function_iArr = ext_fct;
  return edf;
}

/*
 * The externfcts.h had a comment previously that said the following:
 ****
 * The user has to preallocate the variables and set the pointers for any of the
 *call back functions that will be called during trace interpretation. The
 *dimensions given below correspond to the formal arguments in the call back
 *funtions signatures above. If the dimensions n and m change between multiple
 *calls to the same external function, then the variables have to be
 *preallocation with the maximum of the respective dimension values. The dp_x
 *and dp_y pointers have to be valid during both, the tracing phase and the
 *trace interpretation; all the other pointers are required to be valid only for
 *the trace interpretation.
 ****
 * Doing this now internally saves the user from doing it, as well as updating
 * when using multiple problem sizes.
 */
static void update_ext_fct_memory(ext_diff_fct *edfct, size_t dim_x,
                                  size_t dim_y) {
  if (edfct->max_n < dim_x || edfct->max_m < dim_y) {
    /* We need memory stored in the edfct dp_x[n], dp_X[n], dp_Z[n],
     * dp_y[m], dp_Y[m], dp_U[m], dpp_X[n][n], dpp_Y[m][n],
     * dpp_U[m][m], dpp_Z[m][n]. We have no implementation for higher order
     * so leave it out.
     */
    size_t totalmem =
        (3 * dim_x + 3 * dim_y /*+ n*n + 2*n*m + m*m*/) * sizeof(double) +
        (3 * dim_y + dim_x) * sizeof(double *);
    char *tmp;
    if (edfct->allmem != NULL)
      free(edfct->allmem);
    edfct->allmem = (char *)malloc(totalmem);
    memset(edfct->allmem, 0, totalmem);
    edfct->dp_x = (double *)edfct->allmem;
    edfct->dp_y = edfct->dp_x + dim_x;
    edfct->dp_X = edfct->dp_y + dim_y;
    edfct->dp_Y = edfct->dp_X + dim_x;
    edfct->dp_U = edfct->dp_Y + dim_y;
    edfct->dp_Z = edfct->dp_U + dim_y;
    tmp = (char *)(edfct->dp_Z + dim_x);
    edfct->dpp_X = (double **)tmp;
    edfct->dpp_Y = edfct->dpp_X + dim_x;
    edfct->dpp_U = edfct->dpp_Y + dim_y;
    edfct->dpp_Z = edfct->dpp_U + dim_y;
    /*
    tmp = populate_dpp(&edfct->dpp_X, tmp, n,n);
    tmp = populate_dpp(&edfct->dpp_Y, tmp, m,n);
    tmp = populate_dpp(&edfct->dpp_U, tmp, m,m);
    tmp = populate_dpp(&edfct->dpp_Z, tmp, m,n);
    */
  }

  edfct->max_n = (edfct->max_n < dim_x) ? dim_x : edfct->max_n;
  edfct->max_m = (edfct->max_m < dim_y) ? dim_y : edfct->max_m;
}

void call_ext_fct_commonPrior(ext_diff_fct *edfct, size_t dim_x, adouble *xa,
                              size_t dim_y, adouble *ya, size_t &numVals,
                              double *&vals, size_t &oldTraceFlag) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (xa[dim_x - 1].loc() - xa[0].loc() != dim_x - 1 ||
      ya[dim_y - 1].loc() - ya[0].loc() != dim_y - 1)
    fail(ADOLC_EXT_DIFF_LOCATIONGAP);

  if (!edfct)
    fail(ADOLC_EXT_DIFF_NULLPOINTER_STRUCT);

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {

    ADOLC_PUT_LOCINT(edfct->index);
    ADOLC_PUT_LOCINT(dim_x);
    ADOLC_PUT_LOCINT(dim_y);
    ADOLC_PUT_LOCINT(xa[0].loc());

    oldTraceFlag = ADOLC_CURRENT_TAPE_INFOS.traceFlag;
    ADOLC_CURRENT_TAPE_INFOS.traceFlag = 0;

  } else
    oldTraceFlag = 0;

  if (edfct->nestedAdolc) {

    numVals = ADOLC_GLOBAL_TAPE_VARS.storeSize;
    vals = new double[numVals];
    memcpy(vals, ADOLC_GLOBAL_TAPE_VARS.store, numVals * sizeof(double));
  }

  if (!edfct->user_allocated_mem)
    update_ext_fct_memory(edfct, dim_x, dim_y);

  /* update taylor buffer if keep != 0 ; possible double counting as in
   * adouble.cpp => correction in taping.cpp */

  if (oldTraceFlag != 0) {
    if (edfct->dp_x_changes)
      ADOLC_CURRENT_TAPE_INFOS.numTays_Tape += dim_x;

    if (edfct->dp_y_priorRequired)
      ADOLC_CURRENT_TAPE_INFOS.numTays_Tape += dim_y;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors) {
      if (edfct->dp_x_changes)
        for (size_t i = 0; i < dim_x; ++i)
          ADOLC_WRITE_SCAYLOR(xa[i].value());

      if (edfct->dp_y_priorRequired)
        for (size_t i = 0; i < dim_y; ++i)
          ADOLC_WRITE_SCAYLOR(ya[i].value());
    }
  }

  for (size_t i = 0; i < dim_x; ++i)
    edfct->dp_x[i] = xa[i].value();

  if (edfct->dp_y_priorRequired)
    for (size_t i = 0; i < dim_y; ++i)
      edfct->dp_y[i] = ya[i].value();

  ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index = edfct->index;
}

void call_ext_fct_commonPost(ext_diff_fct *edfct, size_t dim_x, adouble *xa,
                             size_t dim_y, adouble *ya, size_t &numVals,
                             double *&vals, size_t &oldTraceFlag) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (edfct->nestedAdolc) {
    memcpy(ADOLC_GLOBAL_TAPE_VARS.store, vals, numVals * sizeof(double));
    delete[] vals;
    vals = 0;
  }

  /* write back */
  if (edfct->dp_x_changes)
    for (size_t i = 0; i < dim_x; ++i)
      xa[i].value(edfct->dp_x[i]);

  for (size_t i = 0; i < dim_y; ++i)
    ya[i].value(edfct->dp_y[i]);

  ADOLC_CURRENT_TAPE_INFOS.traceFlag = oldTraceFlag;
}

int call_ext_fct(ext_diff_fct *edfct, size_t dim_x, adouble *xa, size_t dim_y,
                 adouble *ya) {
  int ret;
  size_t oldTraceFlag, numVals = 0;
  double *vals = nullptr;

  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag)
    put_op(ext_diff);

  call_ext_fct_commonPrior(edfct, dim_x, xa, dim_y, ya, numVals, vals,
                           oldTraceFlag);
  ret = edfct->function(dim_x, edfct->dp_x, dim_y, edfct->dp_y);
  call_ext_fct_commonPost(edfct, dim_x, xa, dim_y, ya, numVals, vals,
                          oldTraceFlag);
  return ret;
}

int call_ext_fct(ext_diff_fct *edfct, size_t iArrLength, int *iArr,
                 size_t dim_x, adouble *xa, size_t dim_y, adouble *ya) {
  int ret;
  size_t oldTraceFlag, numVals = 0;
  double *vals = nullptr;

  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op_reserve(ext_diff_iArr, iArrLength + 2);
    ADOLC_PUT_LOCINT(iArrLength);

    for (size_t i = 0; i < iArrLength; ++i)
      ADOLC_PUT_LOCINT(iArr[i]);

    ADOLC_PUT_LOCINT(
        iArrLength); // do it again so we can read in either direction
  }
  call_ext_fct_commonPrior(edfct, dim_x, xa, dim_y, ya, numVals, vals,
                           oldTraceFlag);
  ret = edfct->function_iArr(iArrLength, iArr, dim_x, edfct->dp_x, dim_y,
                             edfct->dp_y);
  call_ext_fct_commonPost(edfct, dim_x, xa, dim_y, ya, numVals, vals,
                          oldTraceFlag);
  return ret;
}

ext_diff_fct *get_ext_diff_fct(int index) { return buffer.getElement(index); }

static int edfoo_wrapper_function(size_t dim_x, double *x, size_t dim_y,
                                  double *y) {
  ext_diff_fct *edf;
  EDFobject *ebase;
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  // figure out which edf
  edf = get_ext_diff_fct(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
  ebase = reinterpret_cast<EDFobject *>(edf->obj);
  return ebase->function(dim_x, x, dim_y, y);
}
static int edfoo_wrapper_zos_forward(size_t dim_x, double *x, size_t dim_y,
                                     double *y) {
  ext_diff_fct *edf;
  EDFobject *ebase;
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  // figure out which edf
  edf = get_ext_diff_fct(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
  ebase = reinterpret_cast<EDFobject *>(edf->obj);
  return ebase->zos_forward(dim_x, x, dim_y, y);
}

static int edfoo_wrapper_fos_forward(size_t dim_x, double *dp_x, double *dp_X,
                                     size_t dim_y, double *dp_y, double *dp_Y) {
  ext_diff_fct *edf;
  EDFobject *ebase;
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  // figure out which edf
  edf = get_ext_diff_fct(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
  ebase = reinterpret_cast<EDFobject *>(edf->obj);
  return ebase->fos_forward(dim_x, dp_x, dp_X, dim_y, dp_y, dp_Y);
}

static int edfoo_wrapper_fov_forward(size_t dim_x, double *dp_x,
                                     size_t num_dirs, double **dpp_X,
                                     size_t dim_y, double *dp_y,
                                     double **dpp_Y) {
  ext_diff_fct *edf;
  EDFobject *ebase;
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  // figure out which edf
  edf = get_ext_diff_fct(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
  ebase = reinterpret_cast<EDFobject *>(edf->obj);
  return ebase->fov_forward(dim_x, dp_x, num_dirs, dpp_X, dim_y, dp_y, dpp_Y);
}

static int edfoo_wrapper_fos_reverse(size_t dim_y, double *dp_U, size_t dim_x,
                                     double *dp_Z, double *dp_x, double *dp_y) {
  ext_diff_fct *edf;
  EDFobject *ebase;
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  // figure out which edf
  edf = get_ext_diff_fct(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
  ebase = reinterpret_cast<EDFobject *>(edf->obj);
  return ebase->fos_reverse(dim_y, dp_U, dim_x, dp_Z, dp_x, dp_y);
}
static int edfoo_wrapper_fov_reverse(size_t dim_y, size_t num_weights,
                                     double **dpp_U, size_t dim_x,
                                     double **dpp_Z, double *dp_x,
                                     double *dp_y) {
  ext_diff_fct *edf;
  EDFobject *ebase;
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  // figure out which edf
  edf = get_ext_diff_fct(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
  ebase = reinterpret_cast<EDFobject *>(edf->obj);
  return ebase->fov_reverse(dim_y, num_weights, dpp_U, dim_x, dpp_Z, dp_x,
                            dp_y);
}

void EDFobject::init_edf(EDFobject *ebase) {
  edf = buffer.append();
  edf->obj = reinterpret_cast<void *>(ebase);
  edf->function = edfoo_wrapper_function;
  edf->zos_forward = edfoo_wrapper_zos_forward;
  edf->fos_forward = edfoo_wrapper_fos_forward;
  edf->fov_forward = edfoo_wrapper_fov_forward;
  edf->fos_reverse = edfoo_wrapper_fos_reverse;
  edf->fov_reverse = edfoo_wrapper_fov_reverse;
}

static int edfoo_iarr_wrapper_function(size_t iArrLength, int *iArr,
                                       size_t dim_x, double *x, size_t dim_y,
                                       double *y) {
  ext_diff_fct *edf;
  EDFobject_iArr *ebase;
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  // figure out which edf
  edf = get_ext_diff_fct(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
  ebase = reinterpret_cast<EDFobject_iArr *>(edf->obj);
  return ebase->function(iArrLength, iArr, dim_x, x, dim_y, y);
}
static int edfoo_iarr_wrapper_zos_forward(size_t iArrLength, int *iArr,
                                          size_t dim_x, double *x, size_t dim_y,
                                          double *y) {
  ext_diff_fct *edf;
  EDFobject_iArr *ebase;
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  // figure out which edf
  edf = get_ext_diff_fct(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
  ebase = reinterpret_cast<EDFobject_iArr *>(edf->obj);
  return ebase->zos_forward(iArrLength, iArr, dim_x, x, dim_y, y);
}

static int edfoo_iarr_wrapper_fos_forward(size_t iArrLength, int *iArr,
                                          size_t dim_x, double *dp_x,
                                          double *dp_X, size_t dim_y,
                                          double *dp_y, double *dp_Y) {
  ext_diff_fct *edf;
  EDFobject_iArr *ebase;
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  // figure out which edf
  edf = get_ext_diff_fct(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
  ebase = reinterpret_cast<EDFobject_iArr *>(edf->obj);
  return ebase->fos_forward(iArrLength, iArr, dim_x, dp_x, dp_X, dim_y, dp_y,
                            dp_Y);
}

static int edfoo_iarr_wrapper_fov_forward(size_t iArrLength, int *iArr,
                                          size_t dim_x, double *dp_x,
                                          size_t num_dirs, double **dpp_X,
                                          size_t dim_y, double *dp_y,
                                          double **dpp_Y) {
  ext_diff_fct *edf;
  EDFobject_iArr *ebase;
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  // figure out which edf
  edf = get_ext_diff_fct(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
  ebase = reinterpret_cast<EDFobject_iArr *>(edf->obj);
  return ebase->fov_forward(iArrLength, iArr, dim_x, dp_x, num_dirs, dpp_X,
                            dim_y, dp_y, dpp_Y);
}

static int edfoo_iarr_wrapper_fos_reverse(size_t iArrLength, int *iArr,
                                          size_t dim_y, double *dp_U,
                                          size_t dim_x, double *dp_Z,
                                          double *dp_x, double *dp_y) {
  ext_diff_fct *edf;
  EDFobject_iArr *ebase;
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  // figure out which edf
  edf = get_ext_diff_fct(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
  ebase = reinterpret_cast<EDFobject_iArr *>(edf->obj);
  return ebase->fos_reverse(iArrLength, iArr, dim_y, dp_U, dim_x, dp_Z, dp_x,
                            dp_y);
}
static int edfoo_iarr_wrapper_fov_reverse(size_t iArrLength, int *iArr,
                                          size_t dim_y, size_t num_weights,
                                          double **dpp_U, size_t dim_x,
                                          double **dpp_Z, double *dp_x,
                                          double *dp_y) {
  ext_diff_fct *edf;
  EDFobject_iArr *ebase;
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  // figure out which edf
  edf = get_ext_diff_fct(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
  ebase = reinterpret_cast<EDFobject_iArr *>(edf->obj);
  return ebase->fov_reverse(iArrLength, iArr, dim_y, num_weights, dpp_U, dim_x,
                            dpp_Z, dp_x, dp_y);
}

void EDFobject_iArr::init_edf(EDFobject_iArr *ebase) {
  edf = buffer.append();
  edf->obj = reinterpret_cast<void *>(ebase);
  edf->function = edfoo_wrapper_function;
  edf->zos_forward_iArr = edfoo_iarr_wrapper_zos_forward;
  edf->fos_forward_iArr = edfoo_iarr_wrapper_fos_forward;
  edf->fov_forward_iArr = edfoo_iarr_wrapper_fov_forward;
  edf->fos_reverse_iArr = edfoo_iarr_wrapper_fos_reverse;
  edf->fov_reverse_iArr = edfoo_iarr_wrapper_fov_reverse;
}

/****************************************************************************/
/*                                                               THAT'S ALL */

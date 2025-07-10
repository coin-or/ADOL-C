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

#include <adolc/adalloc.h>
#include <adolc/adtb_types.h>
#include <adolc/edfclasses.h>
#include <adolc/externfcts.h>
#include <adolc/oplate.h>
#include <adolc/tape_interface.h>
#include <adolc/valuetape/valuetape.h>
#include <cstring>

/****************************************************************************/
/*                                    extern differentiated functions stuff */

void edf_zero(ext_diff_fct *edf) {}
ext_diff_fct *reg_ext_fct(short tapeId, short ext_tape_id,
                          ADOLC_ext_fct ext_fct) {

  // this call sets edf->index:
  ext_diff_fct *edf = findTape(tapeId).ext_diff_append();
  edf->function = ext_fct;
  edf->tapeId = tapeId;
  edf->ext_tape_id = ext_tape_id;
  return edf;
}

ext_diff_fct *reg_ext_fct(short tapeId, short ext_tape_id,
                          ADOLC_ext_fct_iArr ext_fct) {
  // this call sets  edf->index:
  ext_diff_fct *edf = findTape(tapeId).ext_diff_append();
  edf->function_iArr = ext_fct;
  edf->tapeId = tapeId;
  edf->ext_tape_id = ext_tape_id;
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
void check_input(ext_diff_fct *edfct, size_t dim_x, adouble *xa, size_t dim_y,
                 adouble *ya) {
  if (xa[dim_x - 1].loc() - xa[0].loc() != dim_x - 1 ||
      ya[dim_y - 1].loc() - ya[0].loc() != dim_y - 1)
    ADOLCError::fail(ADOLCError::ErrorType::EXT_DIFF_LOCATIONGAP,
                     CURRENT_LOCATION);
  if (!edfct)
    ADOLCError::fail(ADOLCError::ErrorType::EXT_DIFF_NULLPOINTER_STRUCT,
                     CURRENT_LOCATION);
}

void call_ext_fct_commonPrior(ext_diff_fct *edfct, size_t dim_x, adouble *xa,
                              size_t dim_y, adouble *ya, size_t &numVals,
                              double *&vals, size_t &oldTraceFlag) {

  check_input(edfct, dim_x, xa, dim_y, ya);

  ValueTape &tape = findTape(edfct->tapeId);
  if (tape.traceFlag()) {
    tape.put_loc(edfct->index);
    tape.put_loc(dim_x);
    tape.put_loc(dim_y);
    tape.put_loc(xa[0].loc());
    tape.put_loc(ya[0].loc());
    /* keep space for checkpointing index */
    tape.put_loc(0);

    oldTraceFlag = tape.traceFlag();
    tape.traceFlag(0);
  } else
    oldTraceFlag = 0;

  if (edfct->nestedAdolc) {
    vals = new double[tape.storeSize()];
    std::copy(tape.store(), tape.store() + tape.storeSize(), vals);
  }

  if (!edfct->user_allocated_mem)
    update_ext_fct_memory(edfct, dim_x, dim_y);

  /* update taylor buffer if keep != 0 ; possible double counting as in
   * adouble.cpp => correction in taping.cpp */

  if (oldTraceFlag != 0) {
    if (edfct->dp_x_changes)
      tape.add_numTays_Tape(dim_x);

    if (edfct->dp_y_priorRequired)
      tape.add_numTays_Tape(dim_y);

    if (tape.keepTaylors()) {
      if (edfct->dp_x_changes)
        for (size_t i = 0; i < dim_x; ++i)
          tape.write_scaylor(xa[i].value());

      if (edfct->dp_y_priorRequired)
        for (size_t i = 0; i < dim_y; ++i)
          tape.write_scaylor(ya[i].value());
    }
  }

  for (size_t i = 0; i < dim_x; ++i)
    edfct->dp_x[i] = xa[i].value();

  if (edfct->dp_y_priorRequired)
    for (size_t i = 0; i < dim_y; ++i)
      edfct->dp_y[i] = ya[i].value();

  tape.ext_diff_fct_index(edfct->index);
}

void call_ext_fct_commonPost(ext_diff_fct *edfct, size_t dim_x, adouble *xa,
                             size_t dim_y, adouble *ya, size_t &numVals,
                             double *&vals, size_t &oldTraceFlag) {

  ValueTape &tape = findTape(edfct->tapeId);
  if (edfct->nestedAdolc) {
    std::copy(vals, vals + tape.storeSize(), tape.store());
    delete[] vals;
    vals = nullptr;
  }

  /* write back */
  if (edfct->dp_x_changes)
    for (size_t i = 0; i < dim_x; ++i)
      xa[i].value(edfct->dp_x[i]);

  for (size_t i = 0; i < dim_y; ++i)
    ya[i].value(edfct->dp_y[i]);

  tape.traceFlag(oldTraceFlag);
}

int call_ext_fct(ext_diff_fct *edfct, size_t dim_x, adouble *xa, size_t dim_y,
                 adouble *ya) {
  int ret;
  size_t oldTraceFlag, numVals = 0;
  double *vals = nullptr;

  ValueTape &tape = findTape(edfct->tapeId);

  if (tape.traceFlag())
    tape.put_op(ext_diff);

  call_ext_fct_commonPrior(edfct, dim_x, xa, dim_y, ya, numVals, vals,
                           oldTraceFlag);
  ret = edfct->function(edfct->ext_tape_id, dim_x, edfct->dp_x, dim_y,
                        edfct->dp_y);
  call_ext_fct_commonPost(edfct, dim_x, xa, dim_y, ya, numVals, vals,
                          oldTraceFlag);
  return ret;
}

int call_ext_fct(ext_diff_fct *edfct, size_t iArrLength, int *iArr,
                 size_t dim_x, adouble *xa, size_t dim_y, adouble *ya) {
  int ret;
  size_t oldTraceFlag, numVals = 0;
  double *vals = nullptr;

  ValueTape &tape = findTape(edfct->tapeId);
  if (tape.traceFlag()) {
    tape.put_op(ext_diff_iArr, iArrLength + 2);
    tape.put_loc(iArrLength);

    for (size_t i = 0; i < iArrLength; ++i)
      tape.put_loc(iArr[i]);

    tape.put_loc(iArrLength); // do it again so we can read in either direction
  }
  call_ext_fct_commonPrior(edfct, dim_x, xa, dim_y, ya, numVals, vals,
                           oldTraceFlag);
  ret = edfct->function_iArr(edfct->ext_tape_id, iArrLength, iArr, dim_x,
                             edfct->dp_x, dim_y, edfct->dp_y);
  call_ext_fct_commonPost(edfct, dim_x, xa, dim_y, ya, numVals, vals,
                          oldTraceFlag);
  return ret;
}

ext_diff_fct *get_ext_diff_fct(short tapeId, int index) {
  ValueTape &tape = findTape(tapeId);
  return tape.ext_diff_getElement(index);
}

/**
 * EDFobject definitions
 *
 *
 */

static int edfoo_wrapper_function(short tapeId, size_t dim_x, double *x,
                                  size_t dim_y, double *y) {
  ext_diff_fct *edf;
  EDFobject *ebase;

  ValueTape &tape = findTape(tapeId);
  // figure out which edf
  edf = get_ext_diff_fct(tapeId, tape.ext_diff_fct_index());
  ebase = reinterpret_cast<EDFobject *>(edf->obj);
  return ebase->function(tapeId, dim_x, x, dim_y, y);
}

static int edfoo_wrapper_zos_forward(short tapeId, size_t dim_x, double *x,
                                     size_t dim_y, double *y) {
  ext_diff_fct *edf;
  EDFobject *ebase;

  ValueTape &tape = findTape(tapeId);
  // figure out which edf
  edf = get_ext_diff_fct(tapeId, tape.ext_diff_fct_index());
  ebase = reinterpret_cast<EDFobject *>(edf->obj);
  return ebase->zos_forward(tapeId, dim_x, x, dim_y, y);
}

static int edfoo_wrapper_fos_forward(short tapeId, size_t dim_x, double *dp_x,
                                     double *dp_X, size_t dim_y, double *dp_y,
                                     double *dp_Y) {
  ext_diff_fct *edf;
  EDFobject *ebase;

  ValueTape &tape = findTape(tapeId);
  // figure out which edf
  edf = get_ext_diff_fct(tapeId, tape.ext_diff_fct_index());
  ebase = reinterpret_cast<EDFobject *>(edf->obj);
  return ebase->fos_forward(tapeId, dim_x, dp_x, dp_X, dim_y, dp_y, dp_Y);
}

static int edfoo_wrapper_fov_forward(short tapeId, size_t dim_x, double *dp_x,
                                     size_t num_dirs, double **dpp_X,
                                     size_t dim_y, double *dp_y,
                                     double **dpp_Y) {
  ext_diff_fct *edf;
  EDFobject *ebase;

  ValueTape &tape = findTape(tapeId);
  // figure out which edf
  edf = get_ext_diff_fct(tapeId, tape.ext_diff_fct_index());
  ebase = reinterpret_cast<EDFobject *>(edf->obj);
  return ebase->fov_forward(tapeId, dim_x, dp_x, num_dirs, dpp_X, dim_y, dp_y,
                            dpp_Y);
}

static int edfoo_wrapper_fos_reverse(short tapeId, size_t dim_y, double *dp_U,
                                     size_t dim_x, double *dp_Z, double *dp_x,
                                     double *dp_y) {
  ext_diff_fct *edf;
  EDFobject *ebase;

  ValueTape &tape = findTape(tapeId);
  // figure out which edf
  edf = get_ext_diff_fct(tapeId, tape.ext_diff_fct_index());
  ebase = reinterpret_cast<EDFobject *>(edf->obj);
  return ebase->fos_reverse(tapeId, dim_y, dp_U, dim_x, dp_Z, dp_x, dp_y);
}
static int edfoo_wrapper_fov_reverse(short tapeId, size_t dim_y,
                                     size_t num_weights, double **dpp_U,
                                     size_t dim_x, double **dpp_Z, double *dp_x,
                                     double *dp_y) {
  ext_diff_fct *edf;
  EDFobject *ebase;

  ValueTape &tape = findTape(tapeId);
  // figure out which edf
  edf = get_ext_diff_fct(tapeId, tape.ext_diff_fct_index());
  ebase = reinterpret_cast<EDFobject *>(edf->obj);
  return ebase->fov_reverse(tapeId, dim_y, num_weights, dpp_U, dim_x, dpp_Z,
                            dp_x, dp_y);
}

void EDFobject::init_edf(EDFobject *ebase) {
  ValueTape &tape = currentTape();
  edf = tape.ext_diff_append();
  edf->obj = reinterpret_cast<void *>(ebase);
  edf->function = edfoo_wrapper_function;
  edf->zos_forward = edfoo_wrapper_zos_forward;
  edf->fos_forward = edfoo_wrapper_fos_forward;
  edf->fov_forward = edfoo_wrapper_fov_forward;
  edf->fos_reverse = edfoo_wrapper_fos_reverse;
  edf->fov_reverse = edfoo_wrapper_fov_reverse;
}

[[maybe_unused]] static int
edfoo_iarr_wrapper_function(short tapeId, size_t iArrLength, int *iArr,
                            size_t dim_x, double *x, size_t dim_y, double *y) {
  ext_diff_fct *edf;
  EDFobject_iArr *ebase;

  ValueTape &tape = findTape(tapeId);
  // figure out which edf
  edf = get_ext_diff_fct(tapeId, tape.ext_diff_fct_index());
  ebase = reinterpret_cast<EDFobject_iArr *>(edf->obj);
  return ebase->function(tapeId, iArrLength, iArr, dim_x, x, dim_y, y);
}
static int edfoo_iarr_wrapper_zos_forward(short tapeId, size_t iArrLength,
                                          int *iArr, size_t dim_x, double *x,
                                          size_t dim_y, double *y) {
  ext_diff_fct *edf;
  EDFobject_iArr *ebase;

  ValueTape &tape = findTape(tapeId);
  // figure out which edf
  edf = get_ext_diff_fct(tapeId, tape.ext_diff_fct_index());
  ebase = reinterpret_cast<EDFobject_iArr *>(edf->obj);
  return ebase->zos_forward(tapeId, iArrLength, iArr, dim_x, x, dim_y, y);
}

static int edfoo_iarr_wrapper_fos_forward(short tapeId, size_t iArrLength,
                                          int *iArr, size_t dim_x, double *dp_x,
                                          double *dp_X, size_t dim_y,
                                          double *dp_y, double *dp_Y) {
  ext_diff_fct *edf;
  EDFobject_iArr *ebase;

  ValueTape &tape = findTape(tapeId);
  // figure out which edf
  edf = get_ext_diff_fct(tapeId, tape.ext_diff_fct_index());
  ebase = reinterpret_cast<EDFobject_iArr *>(edf->obj);
  return ebase->fos_forward(tapeId, iArrLength, iArr, dim_x, dp_x, dp_X, dim_y,
                            dp_y, dp_Y);
}

static int edfoo_iarr_wrapper_fov_forward(short tapeId, size_t iArrLength,
                                          int *iArr, size_t dim_x, double *dp_x,
                                          size_t num_dirs, double **dpp_X,
                                          size_t dim_y, double *dp_y,
                                          double **dpp_Y) {
  ext_diff_fct *edf;
  EDFobject_iArr *ebase;

  ValueTape &tape = findTape(tapeId);
  // figure out which edf
  edf = get_ext_diff_fct(tapeId, tape.ext_diff_fct_index());
  ebase = reinterpret_cast<EDFobject_iArr *>(edf->obj);
  return ebase->fov_forward(tapeId, iArrLength, iArr, dim_x, dp_x, num_dirs,
                            dpp_X, dim_y, dp_y, dpp_Y);
}

static int edfoo_iarr_wrapper_fos_reverse(short tapeId, size_t iArrLength,
                                          int *iArr, size_t dim_y, double *dp_U,
                                          size_t dim_x, double *dp_Z,
                                          double *dp_x, double *dp_y) {
  ext_diff_fct *edf;
  EDFobject_iArr *ebase;

  ValueTape &tape = findTape(tapeId);
  // figure out which edf
  edf = get_ext_diff_fct(tapeId, tape.ext_diff_fct_index());
  ebase = reinterpret_cast<EDFobject_iArr *>(edf->obj);
  return ebase->fos_reverse(tapeId, iArrLength, iArr, dim_y, dp_U, dim_x, dp_Z,
                            dp_x, dp_y);
}
static int edfoo_iarr_wrapper_fov_reverse(short tapeId, size_t iArrLength,
                                          int *iArr, size_t dim_y,
                                          size_t num_weights, double **dpp_U,
                                          size_t dim_x, double **dpp_Z,
                                          double *dp_x, double *dp_y) {
  ext_diff_fct *edf;
  EDFobject_iArr *ebase;

  ValueTape &tape = findTape(tapeId);
  // figure out which edf
  edf = get_ext_diff_fct(tapeId, tape.ext_diff_fct_index());
  ebase = reinterpret_cast<EDFobject_iArr *>(edf->obj);
  return ebase->fov_reverse(tapeId, iArrLength, iArr, dim_y, num_weights, dpp_U,
                            dim_x, dpp_Z, dp_x, dp_y);
}

void EDFobject_iArr::init_edf(EDFobject_iArr *ebase) {
  ValueTape &tape = currentTape();
  edf = tape.ext_diff_append();
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

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
#include <adolc/internal/common.h>
#include <adolc/oplate.h>
#include <adolc/tape_interface.h>
#include <adolc/valuetape/valuetape.h>
#include <cstring>

/****************************************************************************/
/*                                    extern differentiated functions stuff */

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
 *preallocation with the maximum of the respective dimension values. The x
 *and y pointers have to be valid during both, the tracing phase and the
 *trace interpretation; all the other pointers are required to be valid only for
 *the trace interpretation.
 ****
 * Doing this now internally saves the user from doing it, as well as updating
 * when using multiple problem sizes.
 */
static void update_ext_fct_memory(ext_diff_fct *edfct, int n, int m) {
  if (edfct->max_n < to_size_t(n) || edfct->max_m < to_size_t(m)) {
    /* We need memory stored in the edfct x[n], X[n], z[n],
     * y[m], Y[m], u[m], Xp[n][n], Yp[m][n],
     * Uq[m][m], Zq[m][n]. We have no implementation for higher order
     * so leave it out.
     */
    size_t totalmem = (3 * n + 3 * m /*+ n*n + 2*n*m + m*m*/) * sizeof(double) +
                      (3 * m + n) * sizeof(double *);
    char *tmp;
    if (edfct->allmem != NULL)
      free(edfct->allmem);
    edfct->allmem = (char *)malloc(totalmem);
    memset(edfct->allmem, 0, totalmem);
    edfct->x = (double *)edfct->allmem;
    edfct->y = edfct->x + n;
    edfct->X = edfct->y + m;
    edfct->Y = edfct->X + n;
    edfct->u = edfct->Y + m;
    edfct->z = edfct->u + m;
    tmp = (char *)(edfct->z + n);
    edfct->Xp = (double **)tmp;
    edfct->Yp = edfct->Xp + n;
    edfct->Uq = edfct->Yp + m;
    edfct->Zq = edfct->Uq + m;
    /*
    tmp = populate_dpp(&edfct->Xp, tmp, n,n);
    tmp = populate_dpp(&edfct->Yp, tmp, m,n);
    tmp = populate_dpp(&edfct->Uq, tmp, m,m);
    tmp = populate_dpp(&edfct->Zq, tmp, m,n);
    */
  }

  edfct->max_n = (edfct->max_n < to_size_t(n)) ? to_size_t(n) : edfct->max_n;
  edfct->max_m = (edfct->max_m < to_size_t(m)) ? to_size_t(m) : edfct->max_m;
}
void check_input(ext_diff_fct *edfct, int n, adouble *xa, int m, adouble *ya) {
  if (xa[n - 1].loc() - xa[0].loc() != to_size_t(n - 1) ||
      ya[m - 1].loc() - ya[0].loc() != to_size_t(m - 1))
    ADOLCError::fail(ADOLCError::ErrorType::EXT_DIFF_LOCATIONGAP,
                     CURRENT_LOCATION);
  if (!edfct)
    ADOLCError::fail(ADOLCError::ErrorType::EXT_DIFF_NULLPOINTER_STRUCT,
                     CURRENT_LOCATION);
}

void call_ext_fct_commonPrior(ext_diff_fct *edfct, int n, adouble *xa, int m,
                              adouble *ya, double *&vals) {

  check_input(edfct, n, xa, m, ya);

  ValueTape &tape = findTape(edfct->tapeId);

  tape.put_loc(edfct->index);
  // The ext-diff shape is stored per taped operation, not on the edfct
  // itself. Reusing the same edfct for multiple calls on one tape must remain
  // correct even if different calls use different m/n.
  tape.put_loc(to_size_t(n));
  tape.put_loc(to_size_t(m));

  // store the index of the first location of input and output variables to
  // later find the right position to read (write) from (to) the taylor
  // coefficient buffers
  edfct->firstDepLocation = ya->loc();
  edfct->firstIndLocation = xa->loc();

  if (edfct->nestedAdolc) {
    vals = new double[tape.storeSize()];
    std::copy(tape.store(), tape.store() + tape.storeSize(), vals);
  }

  if (!edfct->user_allocated_mem)
    update_ext_fct_memory(edfct, n, m);

  /* update taylor buffer if keep != 0 ; possible double counting as in
   * adouble.cpp => correction in taping.cpp */

  if (edfct->dp_x_changes)
    tape.add_numTays_Tape(to_size_t(n));

  if (edfct->dp_y_priorRequired)
    tape.add_numTays_Tape(to_size_t(m));

  if (tape.keepTaylors()) {
    if (edfct->dp_x_changes)
      for (int i = 0; i < n; ++i)
        tape.write_scaylor(xa[i].value());

    if (edfct->dp_y_priorRequired)
      for (int i = 0; i < m; ++i)
        tape.write_scaylor(ya[i].value());
  }

  for (int i = 0; i < n; ++i)
    edfct->x[i] = xa[i].value();

  if (edfct->dp_y_priorRequired)
    for (int i = 0; i < m; ++i)
      edfct->y[i] = ya[i].value();

  tape.ext_diff_fct_index(edfct->index);
}

void call_ext_fct_commonPost(ext_diff_fct *edfct, int n, adouble *xa, int m,
                             adouble *ya, double *&vals) {

  ValueTape &tape = findTape(edfct->tapeId);
  if (edfct->nestedAdolc) {
    std::copy(vals, vals + tape.storeSize(), tape.store());
    delete[] vals;
    vals = nullptr;
  }

  /* write back */
  if (edfct->dp_x_changes)
    for (int i = 0; i < n; ++i)
      xa[i].value(edfct->x[i]);

  for (int i = 0; i < m; ++i)
    ya[i].value(edfct->y[i]);
}

int call_ext_fct(ext_diff_fct *edfct, int n, adouble *xa, int m, adouble *ya) {
  int ret;
  double *vals = nullptr;
  assert(n >= 0);
  assert(m >= 0);

  ValueTape &tape = findTape(edfct->tapeId);

  tape.put_op(ext_diff);

  call_ext_fct_commonPrior(edfct, n, xa, m, ya, vals);
  ret = edfct->function(edfct->ext_tape_id, m, n, edfct->x, edfct->y);
  call_ext_fct_commonPost(edfct, n, xa, m, ya, vals);
  return ret;
}

int call_ext_fct(ext_diff_fct *edfct, size_t iArrLength, size_t *iArr, int n,
                 adouble *xa, int m, adouble *ya) {
  int ret;
  double *vals = nullptr;
  assert(n >= 0);
  assert(m >= 0);

  ValueTape &tape = findTape(edfct->tapeId);
  tape.put_op(ext_diff_iArr, iArrLength + 2);
  tape.put_loc(iArrLength);

  for (size_t i = 0; i < iArrLength; ++i)
    tape.put_loc(iArr[i]);

  tape.put_loc(iArrLength); // do it again so we can read in either direction

  call_ext_fct_commonPrior(edfct, n, xa, m, ya, vals);
  ret = edfct->function_iArr(edfct->ext_tape_id, iArrLength, iArr, m, n,
                             edfct->x, edfct->y);
  call_ext_fct_commonPost(edfct, n, xa, m, ya, vals);
  return ret;
}

ext_diff_fct *get_ext_diff_fct(short tapeId, size_t index) {
  ValueTape &tape = findTape(tapeId);
  return tape.ext_diff_getElement(index);
}

/**
 * EDFobject definitions
 *
 *
 */

static int edfoo_wrapper_function(short tapeId, int m, int n, double *x,
                                  double *y) {
  ext_diff_fct *edf;
  EDFobject *ebase;

  ValueTape &tape = findTape(tapeId);
  // figure out which edf
  edf = get_ext_diff_fct(tapeId, tape.ext_diff_fct_index());
  ebase = reinterpret_cast<EDFobject *>(edf->obj);
  return ebase->function(tapeId, m, n, x, y);
}

static int edfoo_wrapper_zos_forward(short tapeId, int m, int n, double *x,
                                     double *y) {
  ext_diff_fct *edf;
  EDFobject *ebase;

  ValueTape &tape = findTape(tapeId);
  // figure out which edf
  edf = get_ext_diff_fct(tapeId, tape.ext_diff_fct_index());
  ebase = reinterpret_cast<EDFobject *>(edf->obj);
  return ebase->zos_forward(tapeId, m, n, x, y);
}

static int edfoo_wrapper_fos_forward(short tapeId, int m, int n, double *x,
                                     double *X, double *y, double *Y) {
  ext_diff_fct *edf;
  EDFobject *ebase;

  ValueTape &tape = findTape(tapeId);
  // figure out which edf
  edf = get_ext_diff_fct(tapeId, tape.ext_diff_fct_index());
  ebase = reinterpret_cast<EDFobject *>(edf->obj);
  return ebase->fos_forward(tapeId, m, n, x, X, y, Y);
}

static int edfoo_wrapper_fov_forward(short tapeId, int m, int n, int p,
                                     double *x, double **Xp, double *y,
                                     double **Yp) {
  ext_diff_fct *edf;
  EDFobject *ebase;

  ValueTape &tape = findTape(tapeId);
  // figure out which edf
  edf = get_ext_diff_fct(tapeId, tape.ext_diff_fct_index());
  ebase = reinterpret_cast<EDFobject *>(edf->obj);
  return ebase->fov_forward(tapeId, m, n, p, x, Xp, y, Yp);
}

static int edfoo_wrapper_fos_reverse(short tapeId, int m, int n, double *u,
                                     double *z) {
  ext_diff_fct *edf;
  EDFobject *ebase;

  ValueTape &tape = findTape(tapeId);
  // figure out which edf
  edf = get_ext_diff_fct(tapeId, tape.ext_diff_fct_index());
  ebase = reinterpret_cast<EDFobject *>(edf->obj);
  return ebase->fos_reverse(tapeId, m, n, u, z);
}
static int edfoo_wrapper_fov_reverse(short tapeId, int m, int n, int q,
                                     double **Uq, double **Zq) {
  ext_diff_fct *edf;
  EDFobject *ebase;

  ValueTape &tape = findTape(tapeId);
  // figure out which edf
  edf = get_ext_diff_fct(tapeId, tape.ext_diff_fct_index());
  ebase = reinterpret_cast<EDFobject *>(edf->obj);
  return ebase->fov_reverse(tapeId, m, n, q, Uq, Zq);
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
edfoo_iarr_wrapper_function(short tapeId, size_t iArrLength, size_t *iArr,
                            int m, int n, double *x, double *y) {
  ext_diff_fct *edf;
  EDFobject_iArr *ebase;

  ValueTape &tape = findTape(tapeId);
  // figure out which edf
  edf = get_ext_diff_fct(tapeId, tape.ext_diff_fct_index());
  ebase = reinterpret_cast<EDFobject_iArr *>(edf->obj);
  return ebase->function(tapeId, iArrLength, iArr, m, n, x, y);
}
static int edfoo_iarr_wrapper_zos_forward(short tapeId, size_t iArrLength,
                                          size_t *iArr, int m, int n, double *x,
                                          double *y) {
  ext_diff_fct *edf;
  EDFobject_iArr *ebase;

  ValueTape &tape = findTape(tapeId);
  // figure out which edf
  edf = get_ext_diff_fct(tapeId, tape.ext_diff_fct_index());
  ebase = reinterpret_cast<EDFobject_iArr *>(edf->obj);
  return ebase->zos_forward(tapeId, iArrLength, iArr, m, n, x, y);
}

static int edfoo_iarr_wrapper_fos_forward(short tapeId, size_t iArrLength,
                                          size_t *iArr, int m, int n, double *x,
                                          double *X, double *y, double *Y) {
  ext_diff_fct *edf;
  EDFobject_iArr *ebase;

  ValueTape &tape = findTape(tapeId);
  // figure out which edf
  edf = get_ext_diff_fct(tapeId, tape.ext_diff_fct_index());
  ebase = reinterpret_cast<EDFobject_iArr *>(edf->obj);
  return ebase->fos_forward(tapeId, iArrLength, iArr, m, n, x, X, y, Y);
}

static int edfoo_iarr_wrapper_fov_forward(short tapeId, size_t iArrLength,
                                          size_t *iArr, int m, int n, int p,
                                          double *x, double **Xp, double *y,
                                          double **Yp) {
  ext_diff_fct *edf;
  EDFobject_iArr *ebase;

  ValueTape &tape = findTape(tapeId);
  // figure out which edf
  edf = get_ext_diff_fct(tapeId, tape.ext_diff_fct_index());
  ebase = reinterpret_cast<EDFobject_iArr *>(edf->obj);
  return ebase->fov_forward(tapeId, iArrLength, iArr, m, n, p, x, Xp, y, Yp);
}

static int edfoo_iarr_wrapper_fos_reverse(short tapeId, size_t iArrLength,
                                          size_t *iArr, int m, int n, double *u,
                                          double *z) {
  ext_diff_fct *edf;
  EDFobject_iArr *ebase;

  ValueTape &tape = findTape(tapeId);
  // figure out which edf
  edf = get_ext_diff_fct(tapeId, tape.ext_diff_fct_index());
  ebase = reinterpret_cast<EDFobject_iArr *>(edf->obj);
  return ebase->fos_reverse(tapeId, iArrLength, iArr, m, n, u, z);
}
static int edfoo_iarr_wrapper_fov_reverse(short tapeId, size_t iArrLength,
                                          size_t *iArr, int m, int n, int q,
                                          double **Uq, double **Zq) {
  ext_diff_fct *edf;
  EDFobject_iArr *ebase;

  ValueTape &tape = findTape(tapeId);
  // figure out which edf
  edf = get_ext_diff_fct(tapeId, tape.ext_diff_fct_index());
  ebase = reinterpret_cast<EDFobject_iArr *>(edf->obj);
  return ebase->fov_reverse(tapeId, iArrLength, iArr, m, n, q, Uq, Zq);
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

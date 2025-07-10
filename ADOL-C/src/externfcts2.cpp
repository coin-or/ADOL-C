/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     externfcts2.cpp
 Revision: $Id$
 Contents: functions and data types for extern (differentiated) functions.

 Copyright (c) Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#include <adolc/adalloc.h>
#include <adolc/adtb_types.h>
#include <adolc/edfclasses.h>
#include <adolc/oplate.h>
#include <adolc/tape_interface.h>
#include <adolc/valuetape/valuetape.h>
#include <cstring>
/****************************************************************************/
/*                                    extern differentiated functions stuff */

void edf_zero(ext_diff_fct_v2 *edf) {}

ext_diff_fct_v2 *reg_ext_fct(short tapeId, short ext_tape_id,
                             ADOLC_ext_fct_v2 *ext_fct) {
  ValueTape &tape = findTape(tapeId);
  ext_diff_fct_v2 *edf = tape.ext_diff_v2_append();
  edf->function = ext_fct;
  edf->tapeId = tapeId;
  edf->ext_tape_id = ext_tape_id;
  return edf;
}

static void update_ext_fct_memory(ext_diff_fct_v2 *edfct, int nin, int nout,
                                  int *insz, int *outsz) {
  int m_isz = 0, m_osz = 0;
  for (auto i = 0; i < nin; i++)
    m_isz = (m_isz < insz[i]) ? insz[i] : m_isz;
  for (auto i = 0; i < nout; i++)
    m_osz = (m_osz < outsz[i]) ? outsz[i] : m_osz;
  if (edfct->max_nin < nin || edfct->max_nout < nout ||
      edfct->max_insz < m_isz || edfct->max_outsz < m_osz) {
    char *tmp;
    size_t q = nout * m_osz;
    size_t totalmem =
        (3 * nin * m_isz + 3 * nout * m_osz
         // + nin*m_isz*p + nout*m_osz*p
         // + q*nout*m_osz + q*nin*m_isz
         ) * sizeof(double) +
        (3 * nin + 3 * nout + nin * m_isz + nout * m_osz + q * nout + q * nin) *
            sizeof(double *) +
        (nin + nout + 2 * q) * sizeof(double **);
    if (edfct->allmem != NULL)
      free(edfct->allmem);
    edfct->allmem = (char *)malloc(totalmem);
    memset(edfct->allmem, 0, totalmem);
    tmp = edfct->allmem;
    tmp = populate_dpp(&edfct->x, tmp, nin, m_isz);
    tmp = populate_dpp(&edfct->y, tmp, nout, m_osz);
    tmp = populate_dpp(&edfct->xp, tmp, nin, m_isz);
    tmp = populate_dpp(&edfct->yp, tmp, nout, m_osz);
    tmp = populate_dpp(&edfct->up, tmp, nout, m_osz);
    tmp = populate_dpp(&edfct->zp, tmp, nin, m_isz);
    tmp = populate_dppp_nodata(&edfct->Xp, tmp, nin, m_isz);
    tmp = populate_dppp_nodata(&edfct->Yp, tmp, nout, m_osz);
    tmp = populate_dppp_nodata(&edfct->Up, tmp, nout, m_osz);
    tmp = populate_dppp_nodata(&edfct->Zp, tmp, nin, m_isz);
  }
  edfct->max_nin = (edfct->max_nin < nin) ? nin : edfct->max_nin;
  edfct->max_nout = (edfct->max_nout < nout) ? nout : edfct->max_nout;
  edfct->max_insz = (edfct->max_insz < m_isz) ? m_isz : edfct->max_insz;
  edfct->max_outsz = (edfct->max_outsz < m_osz) ? m_osz : edfct->max_outsz;
}

int call_ext_fct(ext_diff_fct_v2 *edfct, int iArrLen, int *iArr, int nin,
                 int nout, int *insz, adouble **x, int *outsz, adouble **y) {
  int ret;
  int oldTraceFlag;
  int i, j;
  double *vals;

  ValueTape &tape = findTape(edfct->tapeId);
  if (tape.traceFlag()) {
    tape.put_op(ext_diff_v2, 2 * (nin + nout) + iArrLen);
    tape.put_loc(edfct->index);
    tape.put_loc(iArrLen);
    for (i = 0; i < iArrLen; i++)
      tape.put_loc(iArr[i]);
    tape.put_loc(iArrLen);
    tape.put_loc(nin);
    tape.put_loc(nout);
    for (i = 0; i < nin; i++) {
      if (x[i][insz[i] - 1].loc() - x[i][0].loc() != (unsigned)insz[i] - 1)
        ADOLCError::fail(ADOLCError::ErrorType::EXT_DIFF_LOCATIONGAP,
                         CURRENT_LOCATION);
      tape.put_loc(insz[i]);
      tape.put_loc(x[i][0].loc());
    }
    for (i = 0; i < nout; i++) {
      if (y[i][outsz[i] - 1].loc() - y[i][0].loc() != (unsigned)outsz[i] - 1)
        ADOLCError::fail(ADOLCError::ErrorType::EXT_DIFF_LOCATIONGAP,
                         CURRENT_LOCATION);
      tape.put_loc(outsz[i]);
      tape.put_loc(y[i][0].loc());
    }
    tape.put_loc(nin);
    tape.put_loc(nout);
    oldTraceFlag = tape.traceFlag();
    tape.traceFlag(0);
  } else
    oldTraceFlag = 0;

  if (edfct->nestedAdolc) {
    vals = new double[tape.storeSize()];
    std::copy(tape.store(), tape.store() + tape.storeSize(), vals);
  }
  if (!edfct->user_allocated_mem)
    update_ext_fct_memory(edfct, nin, nout, insz, outsz);
  if (oldTraceFlag != 0) {
    if (edfct->dp_x_changes)
      for (i = 0; i < nin; i++)
        tape.add_numTays_Tape(insz[i]);
    if (edfct->dp_y_priorRequired)
      for (i = 0; i < nout; i++)
        tape.add_numTays_Tape(outsz[i]);
    if (tape.keepTaylors()) {
      if (edfct->dp_x_changes)
        for (i = 0; i < nin; i++)
          for (j = 0; j < insz[i]; j++)
            tape.write_scaylor(x[i][j].value());
      if (edfct->dp_y_priorRequired)
        for (i = 0; i < nout; i++)
          for (j = 0; j < outsz[i]; j++)
            tape.write_scaylor(y[i][j].value());
    }
  }

  for (i = 0; i < nin; i++)
    for (j = 0; j < insz[i]; j++)
      edfct->x[i][j] = x[i][j].value();

  if (edfct->dp_y_priorRequired)
    for (i = 0; i < nout; i++)
      for (j = 0; j < outsz[i]; j++)
        edfct->y[i][j] = y[i][j].value();

  tape.ext_diff_fct_index(edfct->index);
  ret = edfct->function(edfct->ext_tape_id, iArrLen, iArr, nin, nout, insz,
                        edfct->x, outsz, edfct->y, edfct->context);

  if (edfct->nestedAdolc) {
    std::copy(vals, vals + tape.storeSize(), tape.store());
    delete[] vals;
    vals = nullptr;
  }
  if (edfct->dp_x_changes)
    for (i = 0; i < nin; i++)
      for (j = 0; j < insz[i]; j++)
        x[i][j].value(edfct->x[i][j]);

  for (i = 0; i < nout; i++)
    for (j = 0; j < outsz[i]; j++)
      y[i][j].value(edfct->y[i][j]);

  tape.traceFlag(oldTraceFlag);
  return ret;
}

ext_diff_fct_v2 *get_ext_diff_fct_v2(short tapeId, int index) {
  ValueTape &tape = findTape(tapeId);
  return tape.ext_diff_v2_getElement(index);
}

static int edfoo_v2_wrapper_function(short tapeId, int iArrLen, int *iArr,
                                     int nin, int nout, int *insz, double **x,
                                     int *outsz, double **y, void *ctx) {
  ext_diff_fct_v2 *edf;
  EDFobject_v2 *ebase;

  ValueTape &tape = findTape(tapeId);
  // figure out which edf
  edf = get_ext_diff_fct_v2(tapeId, tape.ext_diff_fct_index());
  ebase = reinterpret_cast<EDFobject_v2 *>(edf->obj);
  return ebase->function(tapeId, iArrLen, iArr, nin, nout, insz, x, outsz, y,
                         ctx);
}
static int edfoo_v2_wrapper_zos_forward(short tapeId, int iArrLen, int *iArr,
                                        int nin, int nout, int *insz,
                                        double **x, int *outsz, double **y,
                                        void *ctx) {
  ext_diff_fct_v2 *edf;
  EDFobject_v2 *ebase;

  ValueTape &tape = findTape(tapeId);
  // figure out which edf
  edf = get_ext_diff_fct_v2(tapeId, tape.ext_diff_fct_index());
  ebase = reinterpret_cast<EDFobject_v2 *>(edf->obj);
  return ebase->zos_forward(tapeId, iArrLen, iArr, nin, nout, insz, x, outsz, y,
                            ctx);
}
static int edfoo_v2_wrapper_fos_forward(short tapeId, int iArrLen, int *iArr,
                                        int nin, int nout, int *insz,
                                        double **x, double **xp, int *outsz,
                                        double **y, double **yp, void *ctx) {
  ext_diff_fct_v2 *edf;
  EDFobject_v2 *ebase;

  ValueTape &tape = findTape(tapeId);
  // figure out which edf
  edf = get_ext_diff_fct_v2(tapeId, tape.ext_diff_fct_index());
  ebase = reinterpret_cast<EDFobject_v2 *>(edf->obj);
  return ebase->fos_forward(tapeId, iArrLen, iArr, nin, nout, insz, x, xp,
                            outsz, y, yp, ctx);
}
static int edfoo_v2_wrapper_fov_forward(short tapeId, int iArrLen, int *iArr,
                                        int nin, int nout, int *insz,
                                        double **x, int ndir, double ***Xp,
                                        int *outsz, double **y, double ***Yp,
                                        void *ctx) {
  ext_diff_fct_v2 *edf;
  EDFobject_v2 *ebase;

  ValueTape &tape = findTape(tapeId);
  // figure out which edf
  edf = get_ext_diff_fct_v2(tapeId, tape.ext_diff_fct_index());
  ebase = reinterpret_cast<EDFobject_v2 *>(edf->obj);
  return ebase->fov_forward(tapeId, iArrLen, iArr, nin, nout, insz, x, ndir, Xp,
                            outsz, y, Yp, ctx);
}
static int edfoo_v2_wrapper_fos_reverse(short tapeId, int iArrLen, int *iArr,
                                        int nout, int nin, int *outsz,
                                        double **up, int *insz, double **zp,
                                        double **x, double **y, void *ctx) {
  ext_diff_fct_v2 *edf;
  EDFobject_v2 *ebase;

  ValueTape &tape = findTape(tapeId);
  // figure out which edf
  edf = get_ext_diff_fct_v2(tapeId, tape.ext_diff_fct_index());
  ebase = reinterpret_cast<EDFobject_v2 *>(edf->obj);
  return ebase->fos_reverse(tapeId, iArrLen, iArr, nout, nin, outsz, up, insz,
                            zp, x, y, ctx);
}
static int edfoo_v2_wrapper_fov_reverse(short tapeId, int iArrLen, int *iArr,
                                        int nout, int nin, int *outsz, int dir,
                                        double ***Up, int *insz, double ***Zp,
                                        double **x, double **y, void *ctx) {
  ext_diff_fct_v2 *edf;
  EDFobject_v2 *ebase;

  ValueTape &tape = findTape(tapeId);
  // figure out which edf
  edf = get_ext_diff_fct_v2(tapeId, tape.ext_diff_fct_index());
  ebase = reinterpret_cast<EDFobject_v2 *>(edf->obj);
  return ebase->fov_reverse(tapeId, iArrLen, iArr, nout, nin, outsz, dir, Up,
                            insz, Zp, x, y, ctx);
}

void EDFobject_v2::init_edf(EDFobject_v2 *ebase) {
  ValueTape &tape = currentTape();
  edf = tape.ext_diff_v2_append();
  edf->obj = reinterpret_cast<void *>(ebase);
  edf->function = edfoo_v2_wrapper_function;
  edf->zos_forward = edfoo_v2_wrapper_zos_forward;
  edf->fos_forward = edfoo_v2_wrapper_fos_forward;
  edf->fov_forward = edfoo_v2_wrapper_fov_forward;
  edf->fos_reverse = edfoo_v2_wrapper_fos_reverse;
  edf->fov_reverse = edfoo_v2_wrapper_fov_reverse;
}

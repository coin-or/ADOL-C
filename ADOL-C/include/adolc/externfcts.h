/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     externfcts.h
 Revision: $Id$
 Contents: public functions and data types for extern (differentiated)
           functions.

 Copyright (c) Andreas Kowarz, Jean Utke

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#ifndef ADOLC_EXTERNFCTS_H
#define ADOLC_EXTERNFCTS_H

#include <adolc/adolcexport.h>
#include <adolc/internal/common.h>
#include <functional>

// ignore missing dll-interface of stl for the moment
// would require a bigger refactor
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4251) // STL members in exported classes
#endif

class adouble;

using ADOLC_ext_fct =
    std::function<int(short tapeId, int m, int n, double *x, double *y)>;
using ADOLC_ext_fct_zos_forward = std::function<int(
    short tapeId, int m, int n, int keep, double *x, double *y)>;
using ADOLC_ext_fct_fos_forward =
    std::function<int(short tapeId, int m, int n, int keep, double *x,
                      double *X, double *y, double *Y)>;
using ADOLC_ext_fct_fov_forward =
    std::function<int(short tapeId, int m, int n, int p, double *x, double **Xp,
                      double *y, double **Yp)>;
using ADOLC_ext_fct_hos_forward =
    std::function<int(short tapeId, int m, int n, int d, int keep, double *x,
                      double **Xd, double *y, double **Yd)>;
using ADOLC_ext_fct_hov_forward =
    std::function<int(short tapeId, int m, int n, int d, int p, double *x,
                      double ***Xpd, double *y, double ***Ypd)>;
using ADOLC_ext_fct_fos_reverse = std::function<int(
    short tapeId, int m, int n, double *u, double *z, double *x, double *y)>;
using ADOLC_ext_fct_fov_reverse =
    std::function<int(short tapeId, int m, int n, int q, double **Uq,
                      double **Zq, double *x, double *y)>;
using ADOLC_ext_fct_hos_reverse =
    std::function<int(short tapeId, int m, int n, int d, double *u, double **Zd,
                      double **Xd, double **Yd)>;
using ADOLC_ext_fct_hos_ti_reverse =
    std::function<int(short tapeId, int m, int n, int d, double **Ud,
                      double **Zd, double **Xd, double **Yd)>;
using ADOLC_ext_fct_hov_reverse =
    std::function<int(short tapeId, int m, int n, int d, int q, double **Uq,
                      double ***Zqd, short **nz, double **Xd, double **Yd)>;

/**
 * Integer-array callback variants store application metadata on the location
 * tape, for example a sparse solver's index arrays.
 */
using ADOLC_ext_fct_iArr =
    std::function<int(short tapeId, size_t iArrLength, size_t *iArr, int m,
                      int n, double *x, double *y)>;
using ADOLC_ext_fct_iArr_zos_forward =
    std::function<int(short tapeId, size_t iArrLength, size_t *iArr, int m,
                      int n, int keep, double *x, double *y)>;
using ADOLC_ext_fct_iArr_fos_forward = std::function<int(
    short tapeId, size_t iArrLength, size_t *iArr, int m, int n, int keep,
    double *x, double *X, double *y, double *Y)>;
using ADOLC_ext_fct_iArr_fov_forward = std::function<int(
    short tapeId, size_t iArrLength, size_t *iArr, int m, int n, int p,
    double *x, double **Xp, double *y, double **Yp)>;
using ADOLC_ext_fct_iArr_hos_forward = std::function<int(
    short tapeId, size_t iArrLength, size_t *iArr, int m, int n, int d,
    int keep, double *x, double **Xd, double *y, double **Yd)>;
using ADOLC_ext_fct_iArr_hov_forward = std::function<int(
    short tapeId, size_t iArrLength, size_t *iArr, int m, int n, int d, int p,
    double *x, double ***Xpd, double *y, double ***Ypd)>;
using ADOLC_ext_fct_iArr_fos_reverse =
    std::function<int(short tapeId, size_t iArrLength, size_t *iArr, int m,
                      int n, double *u, double *z, double *x, double *y)>;
using ADOLC_ext_fct_iArr_fov_reverse = std::function<int(
    short tapeId, size_t iArrLength, size_t *iArr, int m, int n, int q,
    double **Uq, double **Zq, double *x, double *y)>;
using ADOLC_ext_fct_iArr_hos_reverse = std::function<int(
    short tapeId, size_t iArrLength, size_t *iArr, int m, int n, int d,
    double *u, double **Zd, double **Xd, double **Yd)>;
using ADOLC_ext_fct_iArr_hov_reverse = std::function<int(
    short tapeId, size_t iArrLength, size_t *iArr, int m, int n, int d, int q,
    double **Uq, double ***Zqd, short **nz, double **Xd, double **Yd)>;

/**
 * @brief Callback descriptor for an externally differentiated function.
 *
 * `reg_ext_fct()` allocates the descriptor on the outer tape. Supply the
 * callbacks required by each sweep mode. `call_ext_fct()` invokes the primal
 * callback and records the external operation while tracing.
 * Use `ext_diff_fct_v2` for block-structured inputs and an opaque user context.
 */
struct ADOLC_API ext_diff_fct {
  // This is the id of the outer tape that calls the external differentiated
  // function later
  short tapeId{0};

  // tape that stores the external differentiated function.
  short extTapeId{0};

  // storage for the adouble locations to select the right locations to read and
  // write for the taylor buffer later on! note: We can not just use the
  // location of adp_y[0] etc. later, because the location might change while
  // evaluation of the ext function
  size_t firstDepLocation{0};
  size_t firstIndLocation{0};

  /** @brief Number of directions in vector forward sweeps. */
  int p{0};

  /** @brief Number of weights in vector reverse sweeps. */
  int q{0};

  /** @brief Primal callback set by `reg_ext_fct()`; do not replace it. */
  ADOLC_ext_fct function{nullptr};
  ADOLC_ext_fct_iArr function_iArr{nullptr};

  /** @brief Descriptor index set by `reg_ext_fct()`; do not modify it. */
  size_t index{0};

  size_t cp_index{0};

  /** @brief Evaluate the primal function, optionally saving Taylor data. */
  ADOLC_ext_fct_zos_forward zos_forward{nullptr};
  ADOLC_ext_fct_iArr_zos_forward zos_forward_iArr{nullptr};

  /** @brief Evaluate the primal and the direction `Y = J X`. */
  ADOLC_ext_fct_fos_forward fos_forward{nullptr};
  ADOLC_ext_fct_iArr_fos_forward fos_forward_iArr{nullptr};

  /** @brief Evaluate the primal and the directions `Yp = J Xp`. */
  ADOLC_ext_fct_fov_forward fov_forward{nullptr};
  ADOLC_ext_fct_iArr_fov_forward fov_forward_iArr{nullptr};
  /** @brief Reserved; higher-order scalar forward callbacks are unsupported. */
  ADOLC_ext_fct_hos_forward hos_forward{nullptr};
  ADOLC_ext_fct_iArr_hos_forward hos_forward_iArr{nullptr};
  /** @brief Reserved; higher-order vector forward callbacks are unsupported. */
  ADOLC_ext_fct_hov_forward hov_forward{nullptr};
  ADOLC_ext_fct_iArr_hov_forward hov_forward_iArr{nullptr};
  /** @brief Accumulate input adjoints: `z += J^T u`. */
  ADOLC_ext_fct_fos_reverse fos_reverse{nullptr};
  ADOLC_ext_fct_iArr_fos_reverse fos_reverse_iArr{nullptr};
  /** @brief Accumulate adjoints for each weight: `Zq += Uq J`. */
  ADOLC_ext_fct_fov_reverse fov_reverse{nullptr};
  ADOLC_ext_fct_iArr_fov_reverse fov_reverse_iArr{nullptr};
  /** @brief Reserved; higher-order scalar reverse callbacks are unsupported. */
  ADOLC_ext_fct_hos_reverse hos_reverse{nullptr};
  ADOLC_ext_fct_iArr_hos_reverse hos_reverse_iArr{nullptr};

  ADOLC_ext_fct_hos_ti_reverse hos_ti_reverse{nullptr};

  /** @brief Reserved; higher-order vector reverse callbacks are unsupported. */
  ADOLC_ext_fct_hov_reverse hov_reverse{nullptr};
  ADOLC_ext_fct_iArr_hov_reverse hov_reverse_iArr{nullptr};

  /** @brief Enable nested ADOL-C calls; zero avoids saving the outer active
   * store. */
  char nestedAdolc{1};

  /** @brief Save input values for reverse sweeps when the primal callback
   * changes x. */
  char dp_x_changes{1};

  /** @brief Save prior output values when reverse sweeps need them. */
  char dp_y_priorRequired{1};

  /** @brief Owning allocation for internal workspace. */
  char *allmem{nullptr};

  /** @brief Object pointer used by `EDFobject`; do not modify it. */
  void *obj{nullptr};
};

/****************************************************************************/
/*                                                          This is all C++ */

ADOLC_API ext_diff_fct *reg_ext_fct(short tapeId, short extTapeId,
                                    ADOLC_ext_fct ext_fct);
ADOLC_API ext_diff_fct *reg_ext_fct(short tapeId, short extTapeId,
                                    ADOLC_ext_fct_iArr ext_fct);

ADOLC_API ext_diff_fct *get_ext_diff_fct(short tapeId, size_t index);

ADOLC_API int call_ext_fct(ext_diff_fct *edfct, int n, adouble *xa, int m,
                           adouble *ya);
ADOLC_API int call_ext_fct(ext_diff_fct *edfct, size_t iArrLength, size_t *iArr,
                           int n, adouble *xa, int m, adouble *ya);

/****************************************************************************/
#endif // ADOLC_EXTERNFCTS_H

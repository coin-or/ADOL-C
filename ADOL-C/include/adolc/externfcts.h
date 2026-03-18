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
using ADOLC_ext_fct_fos_forward = std::function<int(
    short tapeId, int m, int n, double *x, double *X, double *y, double *Y)>;
using ADOLC_ext_fct_fov_forward =
    std::function<int(short tapeId, int m, int n, int p, double *x, double **Xp,
                      double *y, double **Yp)>;
using ADOLC_ext_fct_hos_forward =
    std::function<int(short tapeId, int m, int n, int d, double *x, double **Xd,
                      double *y, double **Yd)>;
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
 * we add a second set of function pointers with a signature expanded by a an
 * integer array iArr and a parameter iArrLength motivated by externalizing
 * sparse solvers where the sparsity format may be triples (i,j,A[i][j]) and a
 * number of nonzero entries nz where all these integers are to be packed into
 * iArr. Doing this will still allow the integers to be stored in the size_t
 * part of the tape. The alternative to doing this is the introduction of a
 * separate stack to contain the extra data but this would break the
 * self-containment of the tape.
 */
using ADOLC_ext_fct_iArr =
    std::function<int(short tapeId, size_t iArrLength, size_t *iArr, int m,
                      int n, double *x, double *y)>;
using ADOLC_ext_fct_iArr_fos_forward =
    std::function<int(short tapeId, size_t iArrLength, size_t *iArr, int m,
                      int n, double *x, double *X, double *y, double *Y)>;
using ADOLC_ext_fct_iArr_fov_forward = std::function<int(
    short tapeId, size_t iArrLength, size_t *iArr, int m, int n, int p,
    double *x, double **Xp, double *y, double **Yp)>;
using ADOLC_ext_fct_iArr_hos_forward = std::function<int(
    short tapeId, size_t iArrLength, size_t *iArr, int m, int n, int d,
    double *x, double **Xd, double *y, double **Yd)>;
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
 * A variable of this type has to be instantiated by reg_ext_fct (see below) and
 * a pointer to it is returned. Within reg_ext_fct the memberse function and
 * index are properly set. is likely to be wrong in this case. Use pointers
 * instead.
 */
struct ADOLC_API ext_diff_fct {
  // This is the id of the outer tape that calls the external differentiated
  // function later
  short tapeId{0};

  // tape that stores the external differentiated function.
  short ext_tape_id{0};

  // storage for the adouble locations to select the right locations to read and
  // write for the taylor buffer later on! note: We can not just use the
  // location of adp_y[0] etc. later, because the location might change while
  // evaluation of the ext function
  size_t firstDepLocation{0};
  size_t firstIndLocation{0};

  /**
   * Number of forward directions in fos/fov forward calls.
   */
  int p{0};

  /**
   * Number of reverse weight vectors in fov/hov reverse calls.
   */
  int q{0};

  /**
   * DO NOT touch - the function pointer is set through reg_ext_fct
   */
  ADOLC_ext_fct function{nullptr};
  ADOLC_ext_fct_iArr function_iArr{nullptr};

  /**
   * DO NOT touch - the index is set through reg_ext_fct
   */
  size_t index{0};

  size_t cp_index{0};

  /**
   * below are function pointers used for call back from the corresponding
   * ADOL-C trace interpreters; these function pointers are initialized to 0 by
   * reg_ext_fct; the  user needs to set eplicitly the function pointers for the
   * trace interpreters called in the application driver
   */

  /**
   * this points to a method implementing a forward execution of the externally
   * differentiated function y=f(x); the pointer would typically be set to
   * the same function pointer supplied in the call to reg_ext_fct, i.e.
   * zos_forward would be equal to function (above) but there are cases when it
   * makes sense for this to be different as illustrated in
   * examples/additional_examples/ext_diff_func/ext_diff_func.cpp
   */
  ADOLC_ext_fct zos_forward{nullptr};
  ADOLC_ext_fct_iArr zos_forward_iArr{nullptr};

  /**
   * this points to a  method implementing a forward execution of the externally
   * differentiated function y=f(x) and computing the projection
   * Y=Jacobian*X see also the explanation of the X/Y members below.
   */
  ADOLC_ext_fct_fos_forward fos_forward{nullptr};
  ADOLC_ext_fct_iArr_fos_forward fos_forward_iArr{nullptr};

  /**
   * this points to a  method implementing a forward execution of the externally
   * differentiated function y=f(x) and computing the projection
   * Yp=Jacobian*Xp see also the explanation of the Xp/Yp members below.
   */
  ADOLC_ext_fct_fov_forward fov_forward{nullptr};
  ADOLC_ext_fct_iArr_fov_forward fov_forward_iArr{nullptr};
  /**
   * higher order scalar forward for external functions  is currently not
   * implemented in uni5_for.cpp
   */
  ADOLC_ext_fct_hos_forward hos_forward{nullptr};
  ADOLC_ext_fct_iArr_hos_forward hos_forward_iArr{nullptr};
  /**
   * higher order vector forward for external functions  is currently not
   * implemented in uni5_for.cpp
   */
  ADOLC_ext_fct_hov_forward hov_forward{nullptr};
  ADOLC_ext_fct_iArr_hov_forward hov_forward_iArr{nullptr};
  /**
   * this points to a method computing the projection z=transpose(u) * Jacobian
   * see also the explanation of the u/z members below.
   */
  ADOLC_ext_fct_fos_reverse fos_reverse{nullptr};
  ADOLC_ext_fct_iArr_fos_reverse fos_reverse_iArr{nullptr};
  /**
   * this points to a method computing the projection Zq=transpose(Uq) *
   * Jacobian see also the explanation of the Uq/Zq members below.
   */
  ADOLC_ext_fct_fov_reverse fov_reverse{nullptr};
  ADOLC_ext_fct_iArr_fov_reverse fov_reverse_iArr{nullptr};
  /**
   * higher order scalar reverse for external functions  is currently not
   * implemented in ho_rev.cpp
   */
  ADOLC_ext_fct_hos_reverse hos_reverse{nullptr};
  ADOLC_ext_fct_iArr_hos_reverse hos_reverse_iArr{nullptr};

  ADOLC_ext_fct_hos_ti_reverse hos_ti_reverse{nullptr};

  /**
   * higher order vector reverse for external functions  is currently not
   * implemented in ho_rev.cpp
   */
  ADOLC_ext_fct_hov_reverse hov_reverse{nullptr};
  ADOLC_ext_fct_iArr_hov_reverse hov_reverse_iArr{nullptr};

  /**
   * make the call such that Adol-C may be used inside
   * of the externally differentiated function;
   * defaults to non-0;
   * this implies certain storage duplication that can
   * be avoided if no nested use of Adol-C takes place
   */
  char nestedAdolc{1};

  /**
   * if 0, then the 'function' does not change x;
   * defaults to non-0 which implies x values are saved in taylors
   */
  char dp_x_changes{1};

  /**
   * if 0, then the value of y prior to calling 'function'
   * is not required for reverse;
   * defaults to non-0 which implies y values are saved in taylors
   */
  char dp_y_priorRequired{1};

  /**
   * This is an all-memory pointer for allocating and deallocating
   * all other pointers can point to memory within here.
   */
  char *allmem{nullptr};

  /**
   * This is a reference to an object for the C++ object-oriented
   * implementation of the external function ** do not touch **
   */
  void *obj{nullptr};
};

/****************************************************************************/
/*                                                          This is all C++ */

ADOLC_API ext_diff_fct *reg_ext_fct(short tapeId, short ext_tape_id,
                                    ADOLC_ext_fct ext_fct);
ADOLC_API ext_diff_fct *reg_ext_fct(short tapeId, short ext_tape_id,
                                    ADOLC_ext_fct_iArr ext_fct);

ADOLC_API ext_diff_fct *get_ext_diff_fct(short tapeId, size_t index);

ADOLC_API int call_ext_fct(ext_diff_fct *edfct, int n, adouble *xa, int m,
                           adouble *ya);
ADOLC_API int call_ext_fct(ext_diff_fct *edfct, size_t iArrLength, size_t *iArr,
                           int n, adouble *xa, int m, adouble *ya);

/****************************************************************************/
#endif // ADOLC_EXTERNFCTS_H

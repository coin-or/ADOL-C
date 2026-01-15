/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     externfcts.h
 Revision: $Id$
 Contents: public functions and data types for extern (differentiated)
           functions.

 Copyright (c) Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#if !defined(ADOLC_EXTERNFCTS2_H)
#define ADOLC_EXTERNFCTS2_H 1

#include <adolc/adolcexport.h>
#include <adolc/internal/common.h>

using ADOLC_ext_fct_v2 = int(short tapeId, size_t iArrLen, size_t *iArr,
                             size_t nin, size_t nout, size_t *insz, double **x,
                             size_t *outsz, double **y, void *ctx);
using ADOLC_ext_fct_v2_fos_forward = int(short tapeId, size_t iArrLen,
                                         size_t *iArr, size_t nin, size_t nout,
                                         size_t *insz, double **x, double **xp,
                                         size_t *outsz, double **y, double **yp,
                                         void *ctx);
using ADOLC_ext_fct_v2_fov_forward = int(short tapeId, size_t iArrLen,
                                         size_t *iArr, size_t nin, size_t nout,
                                         size_t *insz, double **x, size_t ndir,
                                         double ***Xp, size_t *outsz,
                                         double **y, double ***Yp, void *ctx);
using ADOLC_ext_fct_v2_fos_reverse = int(short tapeId, size_t iArrLen,
                                         size_t *iArr, size_t nout, size_t nin,
                                         size_t *outsz, double **up,
                                         size_t *insz, double **zp, double **x,
                                         double **y, void *ctx);
using ADOLC_ext_fct_v2_fov_reverse = int(short tapeId, size_t iArrLen,
                                         size_t *iArr, size_t nout, size_t nin,
                                         size_t *outsz, size_t dir,
                                         double ***Up, size_t *insz,
                                         double ***Zp, double **x, double **y,
                                         void *ctx);

/* The following two aren't implemented */
using ADOLC_ext_fct_v2_hos_forward = int(short tapeId, size_t iArrLen,
                                         size_t *iArr, size_t nin, size_t nout,
                                         size_t *insz, double **x,
                                         size_t degree, double ***Xp,
                                         size_t *outsz, double **y,
                                         double ***Yp, void *ctx);
using ADOLC_ext_fct_v2_hov_forward = int(short tapeId, size_t iArrLen,
                                         size_t *iArr, size_t nin, size_t nout,
                                         size_t *insz, double **x,
                                         size_t degree, size_t ndir,
                                         double ****Xp, size_t *outsz,
                                         double **y, double ****Yp, void *ctx);

struct ADOLC_API ext_diff_fct_v2 {
  // This is the id of the outer tape that calls the external differentiated
  // function later
  short tapeId{0};

  // tape that stores the external differentiated function.
  short ext_tape_id{0};
  /**
   * DO NOT touch - the function pointer is set through reg_ext_fct
   */
  ADOLC_ext_fct_v2 *function;
  /**
   * DO NOT touch - the index is set through reg_ext_fct
   */
  locint index;

  /**
   * below are function pointers used for call back from the corresponding
   * ADOL-C trace interpreters; these function pointers are initialized to 0 by
   * reg_ext_fct; the  user needs to set eplicitly the function pointers for the
   * trace interpreters called in the application driver
   */

  /**
   * this points to a  method implementing a forward execution of the externally
   * differentiated function y=f(x); the pointer would typically be set to the
   * same function pointer supplied in the call to reg_ext_fct, i.e. zos_forward
   * would be equal to function (above) but there are cases when it makes sense
   * for this to be different as illustrated in
   * examples/additional_examples/ext_diff_func/ext_diff_func.cpp
   */
  ADOLC_ext_fct_v2 *zos_forward;

  /**
   * this points to a  method implementing a forward execution of the externally
   * differentiated function y=f(x) and computing the projection yp=Jacobian*xp
   * see also the explanation of the xp,yp members below.
   */
  ADOLC_ext_fct_v2_fos_forward *fos_forward;

  /**
   * this points to a  method implementing a forward execution of the externally
   * differentiated function y=f(x) and computing the projection Yp=Jacobian*Xp
   * see also the explanation of the Xp/Yp  members below.
   */
  ADOLC_ext_fct_v2_fov_forward *fov_forward;

  /**
   * this points to a  method computing the projection zp=transpose(zp) *
   * Jacobian see also the explanation of the up/zp  members below.
   */
  ADOLC_ext_fct_v2_fos_reverse *fos_reverse;

  /**
   * this points to a  method computing the projection Zp=transpose(Up) *
   * Jacobian see also the explanation of the Up/Zp  members below.
   */
  ADOLC_ext_fct_v2_fov_reverse *fov_reverse;

  /**
   * The names of the variables below correspond to the formal parameters names
   * in the call back functions above;
   */

  /**
   * function and all _forward calls: function argument, dimension
   * nin*insz[0..nin]
   */
  double **x;

  /**
   * fos_forward: tangent direction, dimension nin*insz[0..nin]
   */
  double **xp;

  /**
   * fov_forward: seed matrix for p directions, dimensions nin*insz[0..nin]*p
   * (p=nin*insz[0..nin])
   */
  double ***Xp;

  /**
   * function and all _forward calls: function result, dimension
   * nout*outsz[0..nout]
   */
  double **y;

  /**
   * fos_forward: Jacobian projection, dimension nout*outsz[0..nout]
   */
  double **yp;

  /**
   * fov_forward: Jacobian projection in p directions, dimension
   * nout*outsz[0..nout]*p (p=nin*insz[0..nin])
   */
  double ***Yp;

  /**
   * fos_reverse and hos_reverse:  weight vector, dimension nout*outsz[0..nout]
   */
  double **up;

  /**
   * fov_reverse and hov_reverse: q weight vectors, dimensions
   * (q=nout*outsz[0..nout]) q*nout*outsz[0..nout]
   */
  double ***Up;

  /**
   * fos_reverse: Jacobian projection, dimension nin*insz[0..nin]
   */
  double **zp;

  /**
   * fov_reverse: Jacobian projection for q weight vectors, dimensions
   * (q=nout*outsz[0..nout]) q*nin*insz[0..nin]
   */
  double ***Zp;

  /**
   * track maximal dimensions when function is invoked
   */
  size_t max_nin, max_nout, max_insz, max_outsz;

  /**
   * make the call such that Adol-C may be used inside
   * of the externally differentiated function;
   * defaults to 0;
   * this implies certain storage duplication that can
   * be avoided if no nested use of Adol-C takes place
   */
  char nestedAdolc;

  /**
   * if 0, then the 'function' does not change dp_x;
   * defaults to non-0 which implies dp_x values are saved in taylors
   */
  char dp_x_changes;

  /**
   * if 0, then the value of dp_y prior to calling 'function'
   * is not required for reverse;
   * defaults to non-0 which implies  dp_y values are saved in taylors
   */
  char dp_y_priorRequired;
  /**
   * This is a opaque context pointer that the user may set and use
   * in his implementation of the above functions
   */
  void *context;
  /**
   * This is an all-memory pointer for allocating and deallocating
   * all other pointers can point to memory within here.
   */
  char *allmem;
  /**
   * This is a reference to an object for the C++ object-oriented
   * implementation of the external function ** do not touch **
   */
  void *obj;
  /**
   * This flag indicates that user allocates memory and internally no
   * memory should be allocated
   */
  char user_allocated_mem;
};

ADOLC_API ext_diff_fct_v2 *reg_ext_fct(short tapeId, short ext_tape_id,
                                       ADOLC_ext_fct_v2 ext_fct);
ADOLC_API ext_diff_fct_v2 *get_ext_diff_fct_v2(short tapeId, size_t index);
ADOLC_API int call_ext_fct(ext_diff_fct_v2 *edfct, size_t iArrLen, size_t *iArr,
                           size_t nin, size_t out, size_t *insz, adouble **x,
                           size_t *outsz, adouble **y);

ADOLC_API void edf_zero(ext_diff_fct_v2 *edfct);

ADOLC_API inline void edf_set_opaque_context(ext_diff_fct_v2 *edfct,
                                             void *ctx) {
  edfct->context = ctx;
}

#endif

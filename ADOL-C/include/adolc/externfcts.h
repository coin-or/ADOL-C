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

#include <adolc/internal/common.h>

class adouble;

using ADOLC_ext_fct = int(short tapeId, size_t dim_x, double *x, size_t dim_y,
                          double *y);
using ADOLC_ext_fct_fos_forward = int(short tapeId, size_t n, double *dp_x,
                                      double *dp_X, size_t m, double *dp_y,
                                      double *dp_Y);
using ADOLC_ext_fct_fov_forward = int(short tapeId, size_t n, double *dp_x,
                                      size_t p, double **dpp_X, size_t m,
                                      double *dp_y, double **dpp_Y);
using ADOLC_ext_fct_hos_forward = int(short tapeId, size_t n, double *dp_x,
                                      size_t d, double **dpp_X, size_t m,
                                      double *dp_y, double **dpp_Y);
using ADOLC_ext_fct_hov_forward = int(short tapeId, size_t n, double *dp_x,
                                      size_t d, size_t p, double ***dppp_X,
                                      size_t m, double *dp_y, double ***dppp_Y);
using ADOLC_ext_fct_fos_reverse = int(short tapeId, size_t m, double *dp_U,
                                      size_t n, double *dp_Z, double *dp_x,
                                      double *dp_y);
using ADOLC_ext_fct_fov_reverse = int(short tapeId, size_t m, size_t p,
                                      double **dpp_U, size_t n, double **dpp_Z,
                                      double *dp_x, double *dp_y);
using ADOLC_ext_fct_hos_reverse = int(short tapeId, size_t m, double *dp_U,
                                      size_t n, size_t d, double **dpp_Z);
// dpp_x: {x_{0,0}, x{0,1}, ..., x_{0,keep}}, {x1,0, ..., x_1,keep}, ...} ; n
// Taylor polynomials of degree keep (i.e., array of size n * (keep+1))
using ADOLC_ext_fct_hos_ti_reverse = int(short tapeId, size_t m, double **dp_U,
                                         size_t n, size_t d, double **dpp_Z,
                                         double **dpp_x, double **dpp_y);
using ADOLC_ext_fct_hov_reverse = int(short tapeId, size_t m, size_t p,
                                      double **dpp_U, size_t n, size_t d,
                                      double ***dppp_Z, short **spp_nz);

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
using ADOLC_ext_fct_iArr = int(short tapeId, size_t iArrLength, int *iArr,
                               size_t n, double *x, size_t m, double *y);
using ADOLC_ext_fct_iArr_fos_forward = int(short tapeId, size_t iArrLength,
                                           int *iArr, size_t n, double *dp_x,
                                           double *dp_X, size_t m, double *dp_y,
                                           double *dp_Y);
using ADOLC_ext_fct_iArr_fov_forward = int(short tapeId, size_t iArrLength,
                                           int *iArr, size_t n, double *dp_x,
                                           size_t p, double **dpp_X, size_t m,
                                           double *dp_y, double **dpp_Y);
using ADOLC_ext_fct_iArr_hos_forward = int(short tapeId, size_t iArrLength,
                                           int *iArr, size_t n, double *dp_x,
                                           size_t d, double **dpp_X, size_t m,
                                           double *dp_y, double **dpp_Y);
using ADOLC_ext_fct_iArr_hov_forward = int(short tapeId, size_t iArrLength,
                                           int *iArr, size_t n, double *dp_x,
                                           size_t d, size_t p, double ***dppp_X,
                                           size_t m, double *dp_y,
                                           double ***dppp_Y);
using ADOLC_ext_fct_iArr_fos_reverse = int(short tapeId, size_t iArrLength,
                                           int *iArr, size_t m, double *dp_U,
                                           size_t n, double *dp_Z, double *dp_x,
                                           double *dp_y);
using ADOLC_ext_fct_iArr_fov_reverse = int(short tapeId, size_t iArrLength,
                                           int *iArr, size_t m, size_t p,
                                           double **dpp_U, size_t n,
                                           double **dpp_Z, double *dp_x,
                                           double *dp_y);
using ADOLC_ext_fct_iArr_hos_reverse = int(short tapeId, size_t iArrLength,
                                           int *iArr, size_t m, double *dp_U,
                                           size_t n, size_t d, double **dpp_Z);
using ADOLC_ext_fct_iArr_hov_reverse = int(short tapeId, size_t iArrLength,
                                           int *iArr, size_t m, size_t p,
                                           double **dpp_U, size_t n, size_t d,
                                           double ***dppp_Z, short **spp_nz);

/**
 * A variable of this type has to be instantiated by reg_ext_fct (see below) and
 * a pointer to it is returned. Within reg_ext_fct the memberse function and
 * index are properly set. is likely to be wrong in this case. Use pointers
 * instead.
 */
struct ext_diff_fct {
  // This is the id of the outer tape that calls the external differentiated
  // function later
  short tapeId{0};

  // tape that stores the external differentiated function.
  short ext_tape_id{0};
  /**
   * DO NOT touch - the function pointer is set through reg_ext_fct
   */
  ADOLC_ext_fct *function{nullptr};
  ADOLC_ext_fct_iArr *function_iArr{nullptr};

  /**
   * DO NOT touch - the index is set through reg_ext_fct
   */
  size_t index{0};

  /**
   * below are function pointers used for call back from the corresponding
   * ADOL-C trace interpreters; these function pointers are initialized to 0 by
   * reg_ext_fct; the  user needs to set eplicitly the function pointers for the
   * trace interpreters called in the application driver
   */

  /**
   * this points to a  method implementing a forward execution of the externally
   * differentiated function dp_y=f(dp_x); the pointer would typically be set to
   * the same function pointer supplied in the call to reg_ext_fct, i.e.
   * zos_forward would be equal to function (above) but there are cases when it
   * makes sense for this to be different as illustrated in
   * examples/additional_examples/ext_diff_func/ext_diff_func.cpp
   */
  ADOLC_ext_fct *zos_forward{nullptr};
  ADOLC_ext_fct_iArr *zos_forward_iArr{nullptr};

  /**
   * this points to a  method implementing a forward execution of the externally
   * differentiated function dp_y=f(dp_x) and computing the projection
   * dp_Y=Jacobian*dp_x see also the explanation of the dp_X/Y  members below.
   */
  ADOLC_ext_fct_fos_forward *fos_forward{nullptr};
  ADOLC_ext_fct_iArr_fos_forward *fos_forward_iArr{nullptr};

  /**
   * this points to a  method implementing a forward execution of the externally
   * differentiated function dp_y=f(dp_x) and computing the projection
   * dpp_Y=Jacobian*dpp_x see also the explanation of the dpp_X/Y  members
   * below.
   */
  ADOLC_ext_fct_fov_forward *fov_forward{nullptr};
  ADOLC_ext_fct_iArr_fov_forward *fov_forward_iArr{nullptr};
  /**
   * higher order scalar forward for external functions  is currently not
   * implemented in uni5_for.cpp
   */
  ADOLC_ext_fct_hos_forward *hos_forward{nullptr};
  ADOLC_ext_fct_iArr_hos_forward *hos_forward_iArr{nullptr};
  /**
   * higher order vector forward for external functions  is currently not
   * implemented in uni5_for.cpp
   */
  ADOLC_ext_fct_hov_forward *hov_forward{nullptr};
  ADOLC_ext_fct_iArr_hov_forward *hov_forward_iArr{nullptr};
  /**
   * this points to a  method computing the projection dp_Z=transpose(dp_U) *
   * Jacobian see also the explanation of the dp_U/Z  members below.
   */
  ADOLC_ext_fct_fos_reverse *fos_reverse{nullptr};
  ADOLC_ext_fct_iArr_fos_reverse *fos_reverse_iArr{nullptr};
  /**
   * this points to a  method computing the projection dpp_Z=transpose(dpp_U) *
   * Jacobian see also the explanation of the dpp_U/Z  members below.
   */
  ADOLC_ext_fct_fov_reverse *fov_reverse{nullptr};
  ADOLC_ext_fct_iArr_fov_reverse *fov_reverse_iArr{nullptr};
  /**
   * higher order scalar reverse for external functions  is currently not
   * implemented in ho_rev.cpp
   */
  ADOLC_ext_fct_hos_reverse *hos_reverse{nullptr};
  ADOLC_ext_fct_iArr_hos_reverse *hos_reverse_iArr{nullptr};

  ADOLC_ext_fct_hos_ti_reverse *hos_ti_reverse{nullptr};

  /**
   * higher order vector reverse for external functions  is currently not
   * implemented in ho_rev.cpp
   */
  ADOLC_ext_fct_hov_reverse *hov_reverse{nullptr};
  ADOLC_ext_fct_iArr_hov_reverse *hov_reverse_iArr{nullptr};

  /**
   * The names of the variables below correspond to the formal parameters names
   * in the call back functions above;
   */

  /**
   * function and all _forward calls: function argument, dimension [n]
   */
  double *dp_x{nullptr};

  /**
   * fos_forward: tangent direction, dimension [n]
   */
  double *dp_X{nullptr};

  /**
   * fov_forward: seed matrix for p directions, dimensions [n][p]
   * hos_forward: argument Taylor polynomial coefficients up to order d.
   * dimensions [n][d]
   */
  double **dpp_X{nullptr};

  /**
   * hov_forward: argument Taylor polynomial coefficients up to order d in p
   * directions. dimensions [n][p][d]
   */
  double ***dppp_X{nullptr};

  /**
   * function and all _forward calls: function result, dimension [m]
   */
  double *dp_y{nullptr};

  /**
   * fos_forward: Jacobian projection, dimension [m]
   */
  double *dp_Y{nullptr};

  /**
   * fov_forward: Jacobian projection in p directions, dimension [m][p]
   * hos_forward: result Taylor polynomial coefficients up to order d.
   * dimensions [m][d]
   */
  double **dpp_Y{nullptr};

  /**
   * hov_forward: result Taylor polynomial coefficients up to order d in p
   * directions. dimensions [m][p][d]
   */
  double ***dppp_Y{nullptr};

  /**
   * fos_reverse and hos_reverse:  weight vector, dimension [m]
   */
  double *dp_U{nullptr};

  /**
   * fov_reverse and hov_reverse: p weight vectors, dimensions [p][m]
   */
  double **dpp_U{nullptr};

  /**
   * fos_reverse: Jacobian projection, dimension [n]
   */
  double *dp_Z{nullptr};

  /**
   * fov_reverse: Jacobian projection for p weight vectors, dimensions [p][n]
   * hos_reverse: adjoint Taylor polynomial coefficients up to order d,
   * dimensions [n][d+1]
   */
  double **dpp_Z{nullptr};

  /**
   * hov_reverse:  adjoint Taylor polynomial coefficients up to order d for p
   * weight vectors, dimension [p][n][d+1]
   */
  double ***dppp_Z{nullptr};

  /**
   * hov_reverse: non-zero pattern of dppp_Z, dimension [p][n], see also the
   * hov_reverse ADOL-C driver
   */
  short **spp_nz{nullptr};

  /**
   * track maximal value of n when function is invoked
   */
  size_t max_n{0};

  /**
   * track maximal value of m when function is invoked
   */
  size_t max_m{0};

  /**
   * make the call such that Adol-C may be used inside
   * of the externally differentiated function;
   * defaults to non-0;
   * this implies certain storage duplication that can
   * be avoided if no nested use of Adol-C takes place
   */
  char nestedAdolc{1};

  /**
   * if 0, then the 'function' does not change dp_x;
   * defaults to non-0 which implies dp_x values are saved in taylors
   */
  char dp_x_changes{1};

  /**
   * if 0, then the value of dp_y prior to calling 'function'
   * is not required for reverse;
   * defaults to non-0 which implies  dp_y values are saved in taylors
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

  /**
   * This flag indicates that user allocates memory and internally no
   * memory should be allocated
   */
  char user_allocated_mem{0};
};

/****************************************************************************/
/*                                                          This is all C++ */

ext_diff_fct *reg_ext_fct(short tapeId, short ext_tape_id,
                          ADOLC_ext_fct ext_fct);
ext_diff_fct *reg_ext_fct(short tapeId, short ext_tape_id,
                          ADOLC_ext_fct_iArr ext_fct);

ext_diff_fct *get_ext_diff_fct(short tapeId, int index);

int call_ext_fct(ext_diff_fct *edfct, size_t dim_x, adouble *xa, size_t dim_y,
                 adouble *ya);
int call_ext_fct(ext_diff_fct *edfct, size_t iArrLength, int *iArr,
                 size_t dim_x, adouble *xa, size_t dim_y, adouble *ya);

/**
 * zeros out the edf pointers and sets bools to defaults
 */
void edf_zero(ext_diff_fct *edfct);
/****************************************************************************/
#endif // ADOLC_EXTERNFCTS_H

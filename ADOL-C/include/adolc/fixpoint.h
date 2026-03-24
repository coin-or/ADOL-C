/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     fixpoint.h
 Revision: $Id$
 Contents: all C functions directly accessing at least one of the four tapes
           (operations, locations, constants, value stack)

 Copyright (c) Andreas Kowarz, Sebastian Schlenkrich

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#ifndef ADOLC_FIXPOINT_H
#define ADOLC_FIXPOINT_H

#include <adolc/adolcerror.h>
#include <adolc/adolcexport.h>
#include <adolc/externfcts.h>
#include <adolc/internal/common.h>
#include <functional>

namespace ADOLC::FpIteration {

// define the function types, which are part of the public driver API
using double_F = std::function<int(double *, double *, double *, int, int)>;
using adouble_F = std::function<int(adouble *, adouble *, adouble *, int, int)>;
using norm_F = std::function<double(double *, int)>;
using norm_deriv_F = std::function<double(double *, int)>;

struct FpProblem {
  short tapeId;         /**< Id of the outer tape */
  short subTapeId;      /**< Subtape ID for active evaluation */
  short internalTapeId; /**< Id of tape required for higher-order derivatives */
  double_F double_func; /**< F(x,u) in passive (double) mode          */
  adouble_F adouble_func;       /**< F(x,u) in active (adouble) mode         */
  norm_F norm_func;             /**< norm() for convergence check (passive)  */
  norm_deriv_F norm_deriv_func; /**< norm() for derivative convergence     */
  double epsilon;               /**< tolerance for |x_{k+1}-x_k| (eq. (2))    */
  double
      epsilon_deriv;  /**< tolerance for |\dot{x}_{k+1}-\dot{x}_k| (eq. (5)) */
  size_t N_max;       /**< maximum iterations for value iteration   */
  size_t N_max_deriv; /**< maximum iterations for derivative iter.  */
  adouble *x_0;       /**< initial iterate of the fixed-point iteration */
  adouble *u;         /**< parameter of the fixed-point */
  adouble *x_fix;     /**< storage for the fixed-point: x_fix = F(x_fix, u) */
  int dim_x;          /**< dimension of x */
  int dim_u;          /**< dimension of u */
};

/**
 * @enum FpMode
 * @brief Indicates the mode of the fixed-point driver.
 */
enum class FpMode {
  firstOrder,
  secondOrder,
};

ADOLC_API void resetFpiStack();
template <FpMode mode> int fp_iteration(FpProblem problem);
template <>
ADOLC_API int fp_iteration<FpMode::firstOrder>(FpProblem problem);
template <>
ADOLC_API int fp_iteration<FpMode::secondOrder>(FpProblem problem);
/**
 * @brief Register and perform fixed-point iteration
 * with active (adouble) types, using advanced taping of subtapes.
 *
 * Wraps up: takes initialization of τ, uses the fixed-point φ, and gives the
 * result back to χ as in eq. (7) and Section 2.3.
 *
 * @param tapeId            Tape identifier for outer tape.
 * @param subTapeId      Tape identifier for active subtape.
 * @param double_func       Pointer to passive F(x,u).
 * @param adouble_func      Pointer to active F(x,u).
 * @param norm_func         Passive norm function.
 * @param norm_deriv_func   Derivative norm function.
 * @param epsilon           Convergence tolerance for values.
 * @param epsilon_deriv     Tolerance for derivative iteration.
 * @param N_max             Max iterations for values.
 * @param N_max_deriv       Max iterations for derivatives.
 * @param x_0               Initial x vector (active) (output of τ).
 * @param u                 Control vector (active) (output of τ).
 * @param x_fix             Output fixed-point (active).
 * @param dim_x             Dimension of x.
 * @param dim_u             Dimension of u.
 * @return Number of fixed-point iterations performed.
 */
template <FpMode mode = FpMode::firstOrder>
int fp_iteration(short tapeId, short subTapeId, double_F double_func,
                 adouble_F adouble_func, norm_F norm_func,
                 norm_deriv_F norm_deriv_func, double epsilon,
                 double epsilon_deriv, size_t N_max, size_t N_max_deriv,
                 adouble *x_0, adouble *u, adouble *x_fix, int dim_x,
                 int dim_u) {
  return fp_iteration<mode>(FpProblem{
      .tapeId = tapeId,
      .subTapeId = subTapeId,
      .internalTapeId = -1,
      .double_func = double_func,
      .adouble_func = adouble_func,
      .norm_func = norm_func,
      .norm_deriv_func = norm_deriv_func,
      .epsilon = epsilon,
      .epsilon_deriv = epsilon_deriv,
      .N_max = N_max,
      .N_max_deriv = N_max_deriv,
      .x_0 = x_0,
      .u = u,
      .x_fix = x_fix,
      .dim_x = dim_x,
      .dim_u = dim_u,
  });
}
}; // namespace ADOLC::FpIteration

#endif // ADOLC_FIXPOINT_H

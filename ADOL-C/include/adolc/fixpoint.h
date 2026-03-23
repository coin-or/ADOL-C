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

namespace ADOLC::FpIteration {

// define the function types, which are used internally
using double_F = std::function<int(double *, double *, double *, int, int)>;
using adouble_F = std::function<int(adouble *, adouble *, adouble *, int, int)>;
using norm_F = std::function<double(double *, int)>;
using norm_deriv_F = std::function<double(double *, int)>;

struct FixedPoint {
  size_t lastIter{0};    /**< number of laster iteration the point belongs to */
  std::vector<double> x; /**< x = F(x, u) */
  std::vector<double> u; /**< parameter of the fixed-point */
};

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
  FixedPoint fp;      /**< stores the fixed-point and last iteration */
  bool isInternal{
      false}; /**< flag to indicate if we use internalTape or subTape */
};

/**
 * @enum FpMode
 * @brief Indicates the mode of the fixed-point driver.
 */
enum class FpMode {
  firstOrder,
  secondOrder,
};

/**
 * @struct fpi_data
 * @brief Storage of fixed-point iteration parameters and function pointers.
 *
 * Holds the passive and active evaluation functions F (double/adouble),
 * norms, tolerances, and maximum iteration counts.
 * Corresponds to the description in Section 2.3 of the paper.
 */
struct fpi_data {
  size_t edfIdx; /**< External differentiated function index */
  FpProblem problem;
  ext_diff_fct *edfIteration;
};

/// Stack of active fixed-point configurations
inline static std::vector<fpi_data> &fpiStack() {
  static std::vector<fpi_data> fpi_stack;
  return fpi_stack;
}

ADOLC_API void resetFpiStack();
inline FpProblem &getFpProblem(short tapeId, size_t edfIdx) {
  // Locate iteration parameters
  auto fpiDataPtr =
      std::find_if(fpiStack().begin(), fpiStack().end(), [&](auto &&v) {
        const bool tapeMatches = v.problem.tapeId == tapeId ||
                                 v.problem.subTapeId == tapeId ||
                                 v.problem.internalTapeId == tapeId;
        return tapeMatches && v.edfIdx == edfIdx;
      });

  if (fpiDataPtr == fpiStack().end())
    ADOLCError::fail(ADOLCError::ErrorType::FP_NO_EDF, CURRENT_LOCATION);

  return fpiDataPtr->problem;
}

/**
 * @brief Passive fixed-point iteration x_{k+1} = F(x_k, u).
 *
 * Implements equation (2) of the paper:
 *   x_{k+1} = F(x_k, u)
 * and checks convergence:
 *   \|x_{k+1} - x_k\| < epsilon.
 *
 * @param tapeId       ADOL-C tape identifier (unused here).
 * @param dim_xu       Total dimension of [x; u].
 * @param xu           Input/output array: [x; u] passive values.
 * @param dim_x        Dimension of x.
 * @param x_fix        Output array to store converged x^*.
 * @return Number of iterations k when converged, or -1 if not.
 */
int iteration(short tapeId, int dim_x, int dim_xu, double *xu, double *x_fix);

/**
 * @brief Zero-Order Scalar (ZOS) forward. Does the same as `iteration`.
 *
 * @param tapeId       ADOL-C tape identifier (unused here).
 * @param dim_xu       Total dimension of [x; u].
 * @param xu           Input/output array: [x; u] passive values.
 * @param dim_x        Dimension of x.
 * @param x_fix        Output array to store converged x^*.
 * @return Number of iterations k when converged, or -1 if not.
 */
int fp_zos_forward(short tapeId, int dim_x, int dim_xu, double *xu,
                   double *x_fix);

/**
 * @brief First-order Scalar (FOS) forward for differentiating fixed-point
 * iterations.
 *
 * This function implements the forward mode differentiated fixed-point
 * iteration described in Equation (5) of the referenced paper:
 *
 *     \dot{x}_{k+1} = F'_x(x_*, u) \dot{x}_k + F'_u(x_*, u) \dot{u}
 *
 * It uses ADOL-C's `fos_forward` to evaluate the directional derivatives and
 * iterates until both primal and directional residuals fall below specified
 * tolerances or the maximal number of iterations is reached.
 *
 * @param tapeId         ID of the outer tape.
 * @param dim_xu         Dimension of total input vector [x; u].
 * @param xu             Input/output primal vector.
 * @param xu_dot         Input/output directional derivative vector.
 * @param dim_x          Dimension of x.
 * @param x_fix          Reference solution x_*.
 * @param x_fix_dot      Initial directional derivative of x (\dot{x}_0).
 *
 * @return Iteration count on success, -1 if convergence not reached.
 */
int fp_fos_forward(short tapeId, int dim_x, int dim_xu, double *xu,
                   double *xu_dot, double *x_fix, double *x_fix_dot);

/**
 * @brief First-Order Scalar (FOS) reverse mode for for differentiating
 * fixed-point iterations.
 *
 * Implements the fixed-point reverse mode derivative iteration from
 * Equation (6) in the paper:
 *
 *      \xeta^T_{k+1} = \xeta^T_k F_x(x_*, u) + \bar{x}^T,
 *      \bar{u} = \xeta^T_k F_u(x_*, u)
 *
 * This sweep accumulates adjoint contributions to the control inputs `u`
 * using reverse-mode directional derivatives via `fos_reverse`.
 * Convergence is determined using the norm of the residual in the adjoint
 * direction.
 *
 * @param tapeId      ID of the outer (value) tape.
 * @param dim_x       Dimension of primal state x.
 * @param x_fix_bar   Input adjoint of x_*.
 * @param dim_xu      Dimension of total input vector [x; u].
 * @param xu_bar      Output adjoint for [x; u], incremented in-place.
 * @param unused1     Unused.
 * @param unused2     Unused.
 *
 * @return Number of iterations to convergence, or -1 if not converged.
 */

int fp_fos_reverse(short tapeId, int dim_x, int dim_xu, double *x_fix_bar,
                   double *xu_bar, double * /*unused*/, double * /*unused*/);

int fp_hos_ti_reverse(short tapeId, int dim_x, int dim_xu, int d,
                      double **x_fix_bar, double **xu_bar, double **dpp_x,
                      double **dpp_y);

ext_diff_fct *registerFpIteration(const FpProblem &problem);

int firstOrderFp(FpProblem &problem);
int secondOrderFp(FpProblem &problem);
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
  return fp_iteration<mode>(
      FpProblem{.tapeId = tapeId,
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
                .fp = FixedPoint{.x = std::vector<double>(dim_x),
                                 .u = std::vector<double>(dim_u)}});
}

/// Ensures that the static_assertion is not evaluated until "mode" is known.
template <auto> inline constexpr bool is_dependent_v = false;

template <FpMode mode> int fp_iteration(FpProblem problem) {
  if constexpr (mode == FpMode::firstOrder)
    return firstOrderFp(problem);
  else if constexpr (mode == FpMode::secondOrder)
    return secondOrderFp(problem);
  else
    static_assert(is_dependent_v<mode>,
                  "Not implemented for Template parameter mode!");
}
}; // namespace ADOLC::FpIteration

#endif // ADOLC_FIXPOINT_H

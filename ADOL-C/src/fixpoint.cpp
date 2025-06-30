
/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     fixpoint.cpp
 Revision: $Id$

 This file implements fixed-point iterations and their derivative
 computations as described in "Differentiating Fixed Point Iterations
 with ADOL-C: Gradient Calculation for Fluid Dynamics" (Schlenkrich et al.,
 2008). The methods correspond to equations (1)–(6) in the paper.

  Copyright (c) Andreas Kowarz, Sebastian Schlenkrich

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#include <adolc/adolc.h>
#include <adolc/adolcerror.h>
#include <adolc/externfcts.h>
#include <adolc/fixpoint.h>
#include <adolc/valuetape/valuetape.h>
#include <algorithm>
#include <vector>

/**
 * @struct fpi_data
 * @brief Storage of fixed-point iteration parameters and function pointers.
 *
 * Holds the passive and active evaluation functions F (double/adouble),
 * norms, tolerances, and maximum iteration counts.
 * Corresponds to the description in Section 2.3 of the paper.
 */
struct fpi_data {
  size_t edf_index;             /**< External differentiated function index */
  size_t sub_tape_num;          /**< Subtape ID for active evaluation */
  double_F double_func;         /**< F(x,u) in passive (double) mode          */
  adouble_F adouble_func;       /**< F(x,u) in active (adouble) mode         */
  norm_F norm_func;             /**< norm() for convergence check (passive)  */
  norm_deriv_F norm_deriv_func; /**< norm() for derivative convergence     */
  double epsilon;               /**< tolerance for |x_{k+1}-x_k| (eq. (2))    */
  double epsilon_deriv;         /**< tolerance for |ẋ_{k+1}-ẋ_k| (eq. (5)) */
  size_t N_max;                 /**< maximum iterations for value iteration   */
  size_t N_max_deriv;           /**< maximum iterations for derivative iter.  */
};

/// Stack of active fixed-point configurations
static std::vector<fpi_data> fpi_stack;

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
static int iteration(short tapeId, size_t dim_xu, double *xu, size_t dim_x,
                     double *x_fix) {
  double err;
  const fpi_data &current = fpi_stack.back();

  // Initialize x_0 from xu[0..dim_x-1]
  for (size_t i = 0; i < dim_x; ++i)
    x_fix[i] = xu[i];

  // Main fixed-point loop (eq. (2))
  for (size_t k = 1; k <= current.N_max; ++k) {
    // copy x_fix to xu
    for (size_t i = 0; i < dim_x; ++i)
      xu[i] = x_fix[i];

    // passive call: x_{k+1} = F(x_k, u)
    current.double_func(xu, xu + dim_x, x_fix, dim_x,
                        static_cast<int>(dim_xu - dim_x));

    // residual: x_fix - F(x_k, u)
    for (size_t i = 0; i < dim_x; ++i)
      xu[i] = x_fix[i] - xu[i];

    // convergence check: ||xu|| < epsilon
    err = current.norm_func(xu, dim_x);
    if (err < current.epsilon)
      // this is bad... but the return type of all functions is fixed to "int"
      // so it is converted implicitly anyway
      return static_cast<int>(k);
  }
  return -1;
}

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
static int fp_zos_forward(short tapeId, size_t dim_xu, double *xu, size_t dim_x,
                          double *x_fix) {

  ValueTape &tape = findTape(tapeId);
  double err;
  const size_t edf_index = tape.ext_diff_fct_index();

  // Locate iteration parameters
  auto current =
      std::find_if(fpi_stack.begin(), fpi_stack.end(),
                   [&](auto &&v) { return v.edf_index == edf_index; });

  if (current == fpi_stack.end())
    ADOLCError::fail(ADOLCError::ErrorType::FP_NO_EDF, CURRENT_LOCATION);

  // identical to 'iteration' passive loop
  for (size_t i = 0; i < dim_x; ++i)
    x_fix[i] = xu[i];
  for (size_t k = 1; k <= current->N_max; ++k) {
    for (size_t i = 0; i < dim_x; ++i)
      xu[i] = x_fix[i];
    current->double_func(xu, xu + dim_x, x_fix, dim_x, dim_xu - dim_x);
    for (size_t i = 0; i < dim_x; ++i)
      xu[i] = x_fix[i] - xu[i];
    err = current->norm_func(xu, dim_x);
    if (err < current->epsilon)
      // this is bad... but the return type of all functions is fixed to "int"
      // so it is converted implicitly anyway
      return static_cast<int>(k);
  }
  return -1;
}

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
static int fp_fos_forward(short tapeId, size_t dim_xu, double *xu,
                          double *xu_dot, size_t dim_x, double *x_fix,
                          double *x_fix_dot) {

  double err = 0;
  double err_deriv = 0;
  ValueTape &tape = findTape(tapeId);
  const size_t edf_index = tape.ext_diff_fct_index();

  // locate iteration parameters
  auto current =
      std::find_if(fpi_stack.begin(), fpi_stack.end(),
                   [&](auto &&v) { return v.edf_index == edf_index; });
  if (current == fpi_stack.end())
    ADOLCError::fail(ADOLCError::ErrorType::FP_NO_EDF, CURRENT_LOCATION);

  const size_t maxIter = std::max(current->N_max_deriv, current->N_max);

  for (size_t k = 1; k < maxIter; ++k) {
    if (k > 1) {
      // copies x_*= f(x_*, u) and \dot{x}_{k+1} = F'(x_*, u)[\dot{x_k};
      // \dot{u}]
      for (size_t i = 0; i < dim_x; ++i) {
        xu[i] = x_fix[i];
        xu_dot[i] = x_fix_dot[i];
      }
    }

    // Compute F(x_*,u) and F'(x_*, u)[\dot{x_k}; \dot{u}] using forward-mode AD
    // type signatures of "external driver" and "normal" drivers are not aligned
    // thats why we have to cast...
    fos_forward(current->sub_tape_num, static_cast<int>(dim_x),
                static_cast<int>(dim_xu), 0, xu, xu_dot, x_fix, x_fix_dot);

    // Compute residuals in primal and tangent values
    for (size_t i = 0; i < dim_x; ++i) {
      xu[i] = x_fix[i] - xu[i];
      xu_dot[i] = x_fix_dot[i] - xu_dot[i];
    }
    err = current->norm_func(xu, dim_x);
    err_deriv = current->norm_deriv_func(xu_dot, dim_x);
    // check if converged
    if ((err < current->epsilon) && (err_deriv < current->epsilon_deriv)) {
      // this is bad... but the return type of all functions is fixed to "int"
      // so it is converted implicitly anyway
      return static_cast<int>(k);
    }
  }
  return -1;
}

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
static int fp_fos_reverse(short tapeId, size_t dim_x, double *x_fix_bar,
                          size_t dim_xu, double *xu_bar, double * /*unused*/,
                          double * /*unused*/) {
  double err;
  ValueTape &tape = findTape(tapeId);
  const size_t edf_index = tape.ext_diff_fct_index();

  // Find fpi_stack element with index 'edf_index'.
  auto current =
      std::find_if(fpi_stack.begin(), fpi_stack.end(),
                   [&](auto &&v) { return v.edf_index == edf_index; });

  if (current == fpi_stack.end())
    ADOLCError::fail(ADOLCError::ErrorType::FP_NO_EDF, CURRENT_LOCATION);

  std::vector<double> xeta_u(dim_xu);
  std::vector<double> xeta(dim_x);

  for (size_t k = 1; k < current->N_max_deriv; ++k) {
    for (size_t i = 0; i < dim_x; ++i)
      xeta[i] = xeta_u[i];

    // compute xeta_u = (\xeta^T_{k+1} - x_*^T, u_{k+1}) = (\xeta_k^T F_x,
    // \xeta_k^T F_u)
    // type signatures of "external driver" and "normal" drivers are not aligned
    // thats why we have to cast...
    fos_reverse(current->sub_tape_num, static_cast<int>(dim_x),
                static_cast<int>(dim_xu), xi, U);

    // compoute xeta_u[0, ... dim_x - 1] = \xeta_{k+1}^T and residual
    for (size_t i = 0; i < dim_x; ++i) {
      xeta_u[i] += x_fix_bar[i];     //  = \xeta_{k+1}^T
      xeta[i] = xeta_u[i] - xeta[i]; // residual
    }

    current->norm_deriv_func(xeta.data(), dim_x);
    // check convergence
    if (err < current->epsilon_deriv) {
      // copy u_* to xu[dim_x, ..., dim_u -1]
      for (size_t i = 0; i < dim_xu - dim_x; ++i) {
        xu_bar[dim_x + i] += xeta_u[dim_x + i];
      }
      // this is bad... but the return type of all functions is fixed to "int"
      // so it is converted implicitly anyway
      return static_cast<int>(k);
    }
  }
  // we are not converged in maxIter
  // copy our best value for u_*
  for (size_t i = 0; i < dim_xu - dim_x; ++i)
    xu_bar[dim_x + i] += xeta_u[dim_x + i];
  return -1;
}

/**
 * @brief Register and perform fixed-point iteration
 * with active (adouble) types, using advanced taping of subtapes.
 *
 * Wraps up: takes initialization of τ, uses the fixed-point φ, and gives the
 * result back to χ as in eq. (7) and Section 2.3.
 *
 * @param tapeId            Tape identifier for outer tape.
 * @param sub_tape_num      Tape identifier for active subtape.
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
int fp_iteration(short tapeId, size_t sub_tape_num, double_F double_func,
                 adouble_F adouble_func, norm_F norm_func,
                 norm_deriv_F norm_deriv_func, double epsilon,
                 double epsilon_deriv, size_t N_max, size_t N_max_deriv,
                 adouble *x_0, adouble *u, adouble *x_fix, size_t dim_x,
                 size_t dim_u) {

  // declare extern differentiated function using the fixed-point functions
  ext_diff_fct *edf_iteration = reg_ext_fct(tapeId, sub_tape_num, &iteration);
  edf_iteration->zos_forward = &fp_zos_forward;
  edf_iteration->fos_forward = &fp_fos_forward;
  edf_iteration->fos_reverse = &fp_fos_reverse;

  // add parameters of the fp iteration to the stack
  fpi_stack.emplace_back(fpi_data{
      edf_iteration->index,
      sub_tape_num,
      double_func,
      adouble_func,
      norm_func,
      norm_deriv_func,
      epsilon,
      epsilon_deriv,
      N_max,
      N_max_deriv,
  });

  std::vector<double> u_vals(dim_u);
  std::vector<double> x_vals(dim_x);
  size_t numIterations = 0;
  ValueTape &tape = findTape(tapeId);
  // ensure that the adoubles are contiguous
  tape.ensureContiguousLocations(dim_x + dim_u);
  // to reset old default later
  const short last_default_tape_id = currentTape().tapeId();
  // ensure that the "new" allocates the adoubles for the "tape"
  setCurrentTape(tape.tapeId());
  // scope of tape
  {
    // put x and u together for the iteration
    std::vector<adouble> xu(dim_x + dim_u);

    // initialize x_0
    for (size_t i = 0; i < dim_x; ++i)
      xu[i] = x_0[i];

    // initialize u
    for (size_t i = 0; i < dim_u; ++i)
      xu[dim_x + i] = u[i];

    numIterations =
        call_ext_fct(edf_iteration, dim_x + dim_u, xu.data(), dim_x, x_fix);

    // copy x_fix
    for (size_t i = 0; i < dim_x; ++i)
      x_vals[i] = x_fix[i].value();

    // copy u
    for (size_t i = 0; i < dim_u; ++i)
      u_vals[i] = xu[dim_x + i].value();

    setCurrentTape(sub_tape_num);
    // scope for sub_tape_num
    {
      currentTape().ensureContiguousLocations(2 * (dim_u + dim_x));
      std::vector<adouble> x_fix_new(dim_u + dim_x);
      std::vector<adouble> xu_sub_tape(dim_u + dim_x);

      // copy
      for (size_t i = 0; i < dim_x; ++i)
        x_fix_new[i] = x_vals[i];

      // tape the last fixed-point iteration
      trace_on(sub_tape_num, 1);

      for (size_t i = 0; i < dim_x; ++i)
        xu_sub_tape[i] <<= x_vals[i];

      for (size_t i = 0; i < dim_u; ++i)
        xu_sub_tape[dim_x + i] <<= u_vals[i];

      // IMPORTANT: Dont reuse x_fix here. The location of the x_fix's adoubles
      // could change and the old locations are already stored on the tape. This
      // would cause errors
      adouble_func(xu_sub_tape.data(), xu_sub_tape.data() + dim_x,
                   x_fix_new.data(), dim_x, dim_u);

      double dummy_out;
      for (size_t i = 0; i < dim_x; ++i)
        x_fix_new[i] >>= dummy_out;

      trace_off();
    }
    setCurrentTape(tapeId);
  }
  // reset default tape
  setCurrentTape(last_default_tape_id);

  return numIterations;
}

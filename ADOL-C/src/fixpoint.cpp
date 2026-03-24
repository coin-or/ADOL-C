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

namespace ADOLC::FpIteration {
namespace {
struct FixedPoint {
  size_t lastIter{0};
  std::vector<double> x;
  std::vector<double> u;
};

struct fpi_data {
  size_t edfIdx;
  FpProblem problem;
  ext_diff_fct *edfIteration;
  FixedPoint fp;
  bool isInternal{false};
};

int iteration(short tapeId, int dim_x, int dim_xu, double *xu, double *x_fix);
int fp_zos_forward(short tapeId, int dim_x, int dim_xu, double *xu,
                   double *x_fix);
int fp_fos_forward(short tapeId, int dim_x, int dim_xu, double *xu,
                   double *xu_dot, double *x_fix, double *x_fix_dot);
int fp_fos_reverse(short tapeId, int dim_x, int dim_xu, double *x_bar,
                   double *xi_u_bar, double *, double *);
int fp_hos_ti_reverse(short tapeId, int dim_x, int dim_xu, int d,
                      double **x_bar_ti, double **xu_bar, double **dpp_x,
                      double **dpp_y);
int firstOrderFp(FpProblem &problem);
int secondOrderFp(FpProblem &problem);

std::vector<fpi_data> &fpiStack() {
  static std::vector<fpi_data> fpi_stack;
  return fpi_stack;
}

fpi_data &getfpiData(short tapeId, size_t edfIdx) {
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

  return *fpiDataPtr;
}

ext_diff_fct *registerFpIData(const FpProblem &problem) {
  // declare extern differentiated function using the fixed-point functions
  ext_diff_fct *edfIteration =
      reg_ext_fct(problem.tapeId, problem.subTapeId, iteration);
  edfIteration->zos_forward = fp_zos_forward;
  edfIteration->fos_forward = fp_fos_forward;
  edfIteration->fos_reverse = fp_fos_reverse;
  edfIteration->hos_ti_reverse = fp_hos_ti_reverse;
  fpiStack().emplace_back(fpi_data{
      .edfIdx = edfIteration->index,
      .problem = problem,
      .edfIteration = edfIteration,
      .fp = FixedPoint{},
      .isInternal = false,
  });
  return edfIteration;
}

void prepareFixedPoint(ext_diff_fct *edfIteration, FpProblem &problem,
                       FixedPoint &fp) {
  fp.x.resize(static_cast<size_t>(problem.dim_x));
  fp.u.resize(static_cast<size_t>(problem.dim_u));

  // put x and u together for the iteration
  std::vector<adouble> xu(problem.dim_x + problem.dim_u);

  // initialize x_0
  for (int i = 0; i < problem.dim_x; ++i)
    xu[i] = problem.x_0[i];

  // initialize u
  for (int i = 0; i < problem.dim_u; ++i)
    xu[problem.dim_x + i] = problem.u[i];

  fp.lastIter = call_ext_fct(edfIteration, problem.dim_x + problem.dim_u,
                             xu.data(), problem.dim_x, problem.x_fix);

  // copy fixed-point of x for output
  for (int i = 0; i < problem.dim_x; ++i) {
    fp.x[i] = problem.x_fix[i].value();
  }

  // copy fixed-point of u for output
  for (int i = 0; i < problem.dim_u; ++i) {
    fp.u[i] = xu[problem.dim_x + i].value();
  }
}

void tapeLastFpIteration(short tapeId, FpProblem &problem,
                         const FixedPoint &fp) {
  currentTape().ensureContiguousLocations(problem.dim_u + (2 * problem.dim_x));
  std::vector<adouble> x_fix_new(problem.dim_x);
  std::vector<adouble> xu_sub_tape(problem.dim_u + problem.dim_x);
  // copy fixed-point
  for (int i = 0; i < problem.dim_x; ++i) {
    x_fix_new[i] = fp.x[i];
  }

  // tape the last fixed-point iteration and keep the result
  trace_on(tapeId, 1);
  for (int i = 0; i < problem.dim_x; ++i) {
    xu_sub_tape[i] <<= fp.x[i];
  }

  for (int i = 0; i < problem.dim_u; ++i) {
    xu_sub_tape[problem.dim_x + i] <<= fp.u[i];
  }

  // IMPORTANT: Dont reuse x_fix here. The location of the x_fix's
  // adoubles could change and the old locations are already stored on the
  // tape due to externa differentation. This would cause errors
  problem.adouble_func(xu_sub_tape.data(), xu_sub_tape.data() + problem.dim_x,
                       x_fix_new.data(), problem.dim_x, problem.dim_u);

  double dummy_out;
  for (int i = 0; i < problem.dim_x; ++i) {
    x_fix_new[i] >>= dummy_out;
  }
  trace_off();
}

int iteration(short tapeId, int dim_x, int dim_xu, double *xu, double *x_fix) {
  assert(dim_xu > dim_x && "Dimension mismatch, dim_xu <= dim_x");
  const fpi_data &data =
      getfpiData(tapeId, findTape(tapeId).ext_diff_fct_index());
  // Initialize x_0 from xu[0..dim_x-1]
  for (int i = 0; i < dim_x; ++i) {
    x_fix[i] = xu[i];
  }

  // Main fixed-point loop (eq. (2))
  for (size_t k = 1; k <= data.problem.N_max; ++k) {
    // copy x_fix to xu
    for (int i = 0; i < dim_x; ++i) {
      xu[i] = x_fix[i];
    }

    // passive call: x_{k+1} = F(x_k, u)
    data.problem.double_func(xu, xu + dim_x, x_fix, dim_x, (dim_xu - dim_x));

    // residual: x_fix - F(x_k, u)
    for (int i = 0; i < dim_x; ++i)
      xu[i] = x_fix[i] - xu[i];

    // convergence check: ||xu|| < epsilon
    const double err = data.problem.norm_func(xu, dim_x);
    assert(err >= 0 && "Error should not be negative");
    if (err < data.problem.epsilon) {
      // this is bad... but the return type of all functions is fixed to "int"
      // so it is converted implicitly anyway
      return static_cast<int>(k);
    }
  }
  return -1;
}

int fp_zos_forward(short tapeId, int dim_x, int dim_xu, double *xu,
                   double *x_fix) {
  assert(dim_xu > dim_x && "Dimension mismatch, dim_xu <= dim_x");
  double err = 0.0;
  const fpi_data &data =
      getfpiData(tapeId, findTape(tapeId).ext_diff_fct_index());

  for (int i = 0; i < dim_x; ++i) {
    x_fix[i] = data.fp.x[i];
  }
  for (size_t k = 1; k <= data.problem.N_max; ++k) {
    for (int i = 0; i < dim_x; ++i) {
      xu[i] = x_fix[i];
    }
    data.problem.double_func(xu, xu + dim_x, x_fix, dim_x, (dim_xu - dim_x));

    for (int i = 0; i < dim_x; ++i) {
      xu[i] = x_fix[i] - xu[i];
    }
    err = data.problem.norm_func(xu, dim_x);
    assert(err >= 0 && "Error should not be negative");
    if (err < data.problem.epsilon)
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
int fp_fos_forward(short tapeId, int dim_x, int dim_xu, double *xu,
                   double *xu_dot, double *x_fix, double *x_fix_dot) {
  std::cout << "add keep to fp_fos_forward signauture!" << std::endl;
  assert(dim_xu > dim_x && "Dimension mismatch, dim_xu <= dim_x");
  const fpi_data &data =
      getfpiData(tapeId, findTape(tapeId).ext_diff_fct_index());
  const size_t maxIter = std::max(data.problem.N_max_deriv, data.problem.N_max);
  // initialize with the x fixed point
  std::copy(data.fp.x.begin(), data.fp.x.end(), x_fix);
  std::vector<double> residual(dim_x);
  std::vector<double> residualDeriv(dim_x);
  double err = 0;
  double err_deriv = 0;
  for (size_t k = 1; k < maxIter; ++k) {
    // copies x_*= f(x_*, u) and \dot{x}_{k+1} = F'(x_*, u)\dot{x_k}
    std::copy(x_fix, x_fix + dim_x, xu);
    std::copy(x_fix_dot, x_fix_dot + dim_x, xu_dot);

    // Compute F(x_*,u) and F'(x_*, u)[\dot{x_k}; \dot{u}]
    fos_forward(data.problem.subTapeId, dim_x, dim_xu, 2, xu, xu_dot, x_fix,
                x_fix_dot);

    // Compute residuals in primal and tangent values
    for (int i = 0; i < dim_x; ++i) {
      residual[i] = x_fix[i] - xu[i];
      residualDeriv[i] = x_fix_dot[i] - xu_dot[i];
    }
    err = data.problem.norm_func(residual.data(), dim_x);
    err_deriv = data.problem.norm_deriv_func(residualDeriv.data(), dim_x);
    assert(err >= 0 && "Error should not be negative");
    assert(err_deriv >= 0 && "Error should not be negative");
    // check if converged
    if ((err < data.problem.epsilon) &&
        (err_deriv < data.problem.epsilon_deriv)) {
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
int fp_fos_reverse(short tapeId, int dim_x, int dim_xu, double *x_bar,
                   double *xi_u_bar, double *, double *) {
  assert(dim_xu > dim_x && "Dimension mismatch, dim_xu <= dim_x");
  double err = 0.0;

  const fpi_data &data =
      getfpiData(tapeId, findTape(tapeId).ext_diff_fct_index());
  const short tapeId_ =
      data.isInternal ? data.problem.internalTapeId : data.problem.subTapeId;

  std::vector<double> xi_u(dim_xu, 0.0);
  std::vector<double> xi(dim_x);
  std::vector<double> residual(dim_x);
  for (size_t k = 1; k < data.problem.N_max_deriv; ++k) {
    std::copy(xi_u.begin(), xi_u.begin() + dim_x, xi.begin());

    // do one fixed point iteration:
    // xi_u = [\xi_k^T F_x + \bar{x}, \xi_k^T F_u]
    fos_reverse(tapeId_, dim_x, dim_xu, xi.data(), xi_u.data());
    for (int i = 0; i < dim_x; i++) {
      xi_u[i] += x_bar[i];
    }

    // compute residual \xi_{k+1} - \xi_k
    for (int i = 0; i < dim_x; i++) {
      residual[i] = xi_u[i] - xi[i];
    }
    err = data.problem.norm_deriv_func(residual.data(), dim_x);
    assert(err >= 0 && "Error should not be negative");
    // check convergence
    if (err < data.problem.epsilon_deriv) {
      // add up the resulting adjoints \bar{x} and \bar{u}
      for (int i = 0; i < dim_xu; ++i) {
        xi_u_bar[i] += xi_u[i];
      }
      // this is bad... but the return type of all functions is fixed to "int"
      // so it is converted implicitly anyway
      return static_cast<int>(k);
    }
  }
  // we hit the maximal iteration before converging
  // add up the resulting adjoints \bar{x} and \bar{u}
  for (int i = 0; i < dim_xu; ++i) {
    xi_u_bar[i] += xi_u[i];
  }
  return -1;
}

int fp_hos_ti_reverse(short tapeId, int dim_x, int dim_xu, int d,
                      double **x_bar_ti, double **xu_bar, double **dpp_x,
                      double **dpp_y) {

  // assumption: x_N = Fx(x_N,u) \dot{x}_N + Fu(x_N,u)\dot{u} is taped!
  // assumption: x_N = F(x_N, u) is taped!

  std::cout << "TODO: add ADOLCError, document hos_ti_reverse" << std::endl;
  if (d > 1)
    std::cerr << "fp_hos_reverse is not defined for degree != 1" << std::endl;

  // reference because we set the "isInternal"
  fpi_data &data = getfpiData(tapeId, findTape(tapeId).ext_diff_fct_index());

  // 1. compute [\bar{xi_N}, \bar{u}] via fp_fos_reverse (line 8-10 in algo)
  std::vector<double> xi_u_bar(dim_xu, 0.0);

  // read out \bar{x} from the "ti" input
  std::vector<double> x_bar(dim_x);
  for (int i = 0; i < dim_x; ++i) {
    x_bar[i] = x_bar_ti[i][0];
  }

  // use internal Tape!
  data.isInternal = true;
  fp_fos_reverse(data.problem.tapeId, dim_x, dim_xu, x_bar.data(),
                 xi_u_bar.data(), nullptr, nullptr);
  data.isInternal = false;

  //  2. compute r and u via hos_reverse of subTapeNum, thus we have
  //  to keep the (line 13 and part of line 21) Here we need the
  //  taped fos_forward of the fp iteration -> keep = 2
  std::vector<double> xu(dim_xu);
  std::vector<double> xu_dot(dim_xu);
  std::vector<double> x_fix(dim_x);
  std::vector<double> x_fix_dot(dim_x);
  // store x fix and \dot
  for (int i = 0; i < dim_x; ++i) {
    xu[i] = data.fp.x[i];
    xu_dot[i] = dpp_y[i][1];
  }
  // store u and \dot{u}
  for (int i = 0; i < data.problem.dim_u; ++i) {
    xu[dim_x + i] = data.fp.u[i];
    xu_dot[dim_x + i] = dpp_x[dim_x + i][1];
  }
  fos_forward(data.problem.subTapeId, dim_x, dim_xu, 2, xu.data(),
              xu_dot.data(), x_fix.data(), x_fix_dot.data());

  std::vector<double> xi_bar(xi_u_bar.begin(), xi_u_bar.begin() + dim_x);
  hos_reverse(data.problem.subTapeId, dim_x, dim_xu, 1, xi_bar.data(), xu_bar);

  // We now have xu_bar[1] = [xi_k^T(Fxx \dot{x} + Fxu \dot{u}),
  // xi_k^T(Fux \dot{x} + Fuu \dot{u})]

  // 3. compute fp_fos_reverse with \bar{xi} and \bar{x} = \dot{\bar{r}} =
  // \bar{xu}[1][0]
  std::vector<double> xi_bar_dot(dim_xu);
  std::vector<double> r_bar_dot(dim_x);
  for (int i = 0; i < dim_x; ++i) {
    r_bar_dot[i] = xu_bar[i][1];
  }

  data.isInternal = true;
  fp_fos_reverse(data.problem.tapeId, dim_x, dim_xu, r_bar_dot.data(),
                 xi_bar_dot.data(), nullptr, nullptr);
  data.isInternal = false;
  // We now have xi_bar_dot = [xi_K^T Fx + \bar{r}, xi_K^T Fu]_K^T Fx + \bar{r},
  // xi_K^T Fu]

  // 4. return xi_k, u = u part of last fos_reverse + u part of hos_reverse
  for (int i = 0; i < data.problem.dim_u; ++i) {
    xu_bar[dim_x + i][1] += xi_bar_dot[dim_x + i];
  }

  return 0;
}

int firstOrderFp(FpProblem &problem) {
  std::cout << "TODO: document firstOrderFP" << std::endl;
  auto edfIteration = registerFpIData(problem);
  fpi_data &data = getfpiData(problem.tapeId, edfIteration->index);

  FixedPoint fp;

  ValueTape &tape = findTape(problem.tapeId);
  // ensure that the adoubles are contiguous
  tape.ensureContiguousLocations(problem.dim_x + problem.dim_u);
  // to reset old default later
  const short last_default_tape_id = currentTape().tapeId();
  // ensure that the adoubles are allocated the correct tape
  setCurrentTape(tape.tapeId());
  prepareFixedPoint(edfIteration, problem, fp);
  setCurrentTape(problem.subTapeId);
  tapeLastFpIteration(problem.subTapeId, problem, fp);

  // reset previous default tape
  setCurrentTape(last_default_tape_id);

  data.fp = fp;
  return static_cast<int>(fp.lastIter);
}

int secondOrderFp(FpProblem &problem) {
  std::cout << "TODO: document secondOrderFP" << std::endl;
  auto edfIteration = registerFpIData(problem);
  fpi_data &data = getfpiData(problem.tapeId, edfIteration->index);
  ValueTape &tape = findTape(problem.tapeId);

  FixedPoint fp;

  // ensure that the adoubles are contiguous
  tape.ensureContiguousLocations(problem.dim_x + problem.dim_u);
  // to reset old default later
  const short last_default_tape_id = currentTape().tapeId();
  // ensure that the adoubles are allocated for the correct tape
  setCurrentTape(tape.tapeId());
  prepareFixedPoint(edfIteration, problem, fp);
  setCurrentTape(problem.subTapeId);
  tapeLastFpIteration(problem.subTapeId, problem, fp);

  setCurrentTape(problem.internalTapeId);
  tapeLastFpIteration(problem.internalTapeId, problem, fp);
  setCurrentTape(last_default_tape_id);

  data.fp = fp;
  return static_cast<int>(fp.lastIter);
}
} // namespace

template <>
int fp_iteration<FpMode::firstOrder>(FpProblem problem) {
  return firstOrderFp(problem);
}

template <>
int fp_iteration<FpMode::secondOrder>(FpProblem problem) {
  return secondOrderFp(problem);
}

void resetFpiStack() { fpiStack().clear(); }
}; // namespace ADOLC::FpIteration

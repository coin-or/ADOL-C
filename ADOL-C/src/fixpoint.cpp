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

void prepareFixedPoint(ext_diff_fct *edfIteration, const FpProblem &problem,
                       FixedPoint &fp) {
  // put x and u together for the iteration
  std::vector<adouble> xu(problem.dim_x + problem.dim_u);

  // initialize x_0
  for (size_t i = 0; i < problem.dim_x; ++i)
    xu[i] = problem.x_0[i];

  // initialize u
  for (size_t i = 0; i < problem.dim_u; ++i)
    xu[problem.dim_x + i] = problem.u[i];

  fp.lastIter = call_ext_fct(edfIteration, problem.dim_x + problem.dim_u,
                             xu.data(), problem.dim_x, problem.x_fix);

  // copy fixed-point of x for output
  for (size_t i = 0; i < problem.dim_x; ++i)
    fp.x[i] = problem.x_fix[i].value();

  // copy fixed-point of u for output
  for (size_t i = 0; i < problem.dim_u; ++i)
    fp.u[i] = xu[problem.dim_x + i].value();
}

void tapeLastFpIteration(short tapeId, const FpProblem &problem,
                         FixedPoint &fp) {
  currentTape().ensureContiguousLocations(2 * (problem.dim_u + problem.dim_x));
  std::vector<adouble> x_fix_new(problem.dim_u + problem.dim_x);
  std::vector<adouble> xu_sub_tape(problem.dim_u + problem.dim_x);
  // copy fixed-point
  for (size_t i = 0; i < problem.dim_x; ++i) {
    x_fix_new[i] = fp.x[i];
  }

  // tape the last fixed-point iteration and keep the result
  trace_on(tapeId, 1);
  for (size_t i = 0; i < problem.dim_x; ++i)
    xu_sub_tape[i] <<= fp.x[i];

  for (size_t i = 0; i < problem.dim_u; ++i)
    xu_sub_tape[problem.dim_x + i] <<= fp.u[i];

  // IMPORTANT: Dont reuse x_fix here. The location of the x_fix's
  // adoubles could change and the old locations are already stored on the
  // tape due to externa differentation. This would cause errors
  problem.adouble_func(xu_sub_tape.data(), xu_sub_tape.data() + problem.dim_x,
                       x_fix_new.data(), static_cast<int>(problem.dim_x),
                       static_cast<int>(problem.dim_u));

  double dummy_out;
  for (size_t i = 0; i < problem.dim_x; ++i)
    x_fix_new[i] >>= dummy_out;

  trace_off();
}

int iteration(short tapeId, size_t dim_xu, double *xu, size_t dim_x,
              double *x_fix) {
  assert(dim_xu > dim_x && "Dimension mismatch, dim_xu <= dim_x");
  FpProblem problem = getFpProblem(findTape(tapeId).ext_diff_fct_index());
  // Initialize x_0 from xu[0..dim_x-1]
  for (size_t i = 0; i < dim_x; ++i)
    x_fix[i] = xu[i];

  // Main fixed-point loop (eq. (2))
  for (size_t k = 1; k <= problem.N_max; ++k) {
    // copy x_fix to xu
    for (size_t i = 0; i < dim_x; ++i)
      xu[i] = x_fix[i];

    // passive call: x_{k+1} = F(x_k, u)
    problem.double_func(xu, xu + dim_x, x_fix, static_cast<int>(dim_x),
                        static_cast<int>(dim_xu - dim_x));

    // residual: x_fix - F(x_k, u)
    for (size_t i = 0; i < dim_x; ++i)
      xu[i] = x_fix[i] - xu[i];

    // convergence check: ||xu|| < epsilon
    const double err = problem.norm_func(xu, static_cast<int>(dim_x));
    assert(err >= 0 && "Error should not be negative");
    if (err < problem.epsilon)
      // this is bad... but the return type of all functions is fixed to "int"
      // so it is converted implicitly anyway
      return static_cast<int>(k);
  }
  return -1;
}

int fp_zos_forward(short tapeId, size_t dim_xu, double *xu, size_t dim_x,
                   double *x_fix) {
  assert(dim_xu > dim_x && "Dimension mismatch, dim_xu <= dim_x");
  double err = 0.0;
  FpProblem problem = getFpProblem(findTape(tapeId).ext_diff_fct_index());

  // identical to 'iteration' passive loop
  for (size_t i = 0; i < dim_x; ++i)
    x_fix[i] = xu[i];
  for (size_t k = 1; k <= problem.N_max; ++k) {
    for (size_t i = 0; i < dim_x; ++i)
      xu[i] = x_fix[i];
    problem.double_func(xu, xu + dim_x, x_fix, static_cast<int>(dim_x),
                        static_cast<int>(dim_xu - dim_x));
    for (size_t i = 0; i < dim_x; ++i)
      xu[i] = x_fix[i] - xu[i];
    err = problem.norm_func(xu, static_cast<int>(dim_x));
    assert(err >= 0 && "Error should not be negative");
    if (err < problem.epsilon)
      // this is bad... but the return type of all functions is fixed to "int"
      // so it is converted implicitly anyway
      return static_cast<int>(k);
  }
  return -1;
}

int fp_fos_forward(short tapeId, size_t dim_xu, double *xu, double *xu_dot,
                   size_t dim_x, double *x_fix, double *x_fix_dot) {
  assert(dim_xu > dim_x && "Dimension mismatch, dim_xu <= dim_x");
  double err = 0;
  double err_deriv = 0;
  FpProblem problem = getFpProblem(findTape(tapeId).ext_diff_fct_index());
  const size_t maxIter = std::max(problem.N_max_deriv, problem.N_max);

  for (size_t k = 1; k < maxIter; ++k) {
    if (k > 1) {
      // copies x_*= f(x_*, u) and \dot{x}_{k+1} = F'(x_*, u)\dot{x_k}
      for (size_t i = 0; i < dim_x; ++i) {
        xu[i] = x_fix[i];
        xu_dot[i] = x_fix_dot[i];
      }
    }

    // Compute F(x_*,u) and F'(x_*, u)[\dot{x_k}; \dot{u}] using forward-mode AD
    // type signatures of "external driver" and "normal" drivers are not aligned
    // thats why we have to cast...
    fos_forward(problem.subTapeId, static_cast<int>(dim_x),
                static_cast<int>(dim_xu), 2, xu, xu_dot, x_fix, x_fix_dot);

    // Compute residuals in primal and tangent values
    for (size_t i = 0; i < dim_x; ++i) {
      xu[i] = x_fix[i] - xu[i];
      xu_dot[i] = x_fix_dot[i] - xu_dot[i];
    }
    err = problem.norm_func(xu, static_cast<int>(dim_x));
    err_deriv = problem.norm_deriv_func(xu_dot, static_cast<int>(dim_x));
    assert(err >= 0 && "Error should not be negative");
    assert(err_deriv >= 0 && "Error should not be negative");
    // check if converged
    if ((err < problem.epsilon) && (err_deriv < problem.epsilon_deriv)) {
      // this is bad... but the return type of all functions is fixed to "int"
      // so it is converted implicitly anyway
      return static_cast<int>(k);
    }
  }
  return -1;
}

int fp_fos_reverse(short tapeId, size_t dim_x, double *x_fix_bar, size_t dim_xu,
                   double *xu_bar, double * /*unused*/, double * /*unused*/) {
  assert(dim_xu > dim_x && "Dimension mismatch, dim_xu <= dim_x");
  double err = 0.0;

  FpProblem problem = getFpProblem(findTape(tapeId).ext_diff_fct_index());
  std::vector<double> xeta_u(dim_xu);
  std::fill(xeta_u.begin(), xeta_u.end(), 0.0);
  short tapeId_ =
      problem.isInternal ? problem.internalTapeId : problem.subTapeId;
  std::vector<double> xeta(dim_x);

  for (size_t k = 1; k < problem.N_max_deriv; ++k) {
    for (size_t i = 0; i < dim_x; ++i)
      xeta[i] = xeta_u[i];

    // compute xeta_u = (\xeta^T_{k+1} - x_*^T, u_{k+1}) = (\xeta_k^T F_x,
    // \xeta_k^T F_u)
    // type signatures of "external driver" and "normal" drivers are not aligned
    // thats why we have to cast...
    fos_reverse(tapeId_, static_cast<int>(dim_x), static_cast<int>(dim_xu),
                xeta.data(), xeta_u.data());

    // compoute xeta_u[0, ... dim_x - 1] = \xeta_{k+1}^T and residual
    for (size_t i = 0; i < dim_x; ++i) {
      xeta_u[i] += x_fix_bar[i];     //  = \xeta_{k+1}^T
      xeta[i] = xeta_u[i] - xeta[i]; // residual
    }

    err = problem.norm_deriv_func(xeta.data(), static_cast<int>(dim_x));
    assert(err >= 0 && "Error should not be negative");
    // check convergence
    if (err < problem.epsilon_deriv) {
      // copy u_* to xu[dim_x, ..., dim_u -1]
      for (size_t i = 0; i < dim_xu; ++i)
        xu_bar[i] += xeta_u[i];
      // this is bad... but the return type of all functions is fixed to "int"
      // so it is converted implicitly anyway
      return static_cast<int>(k);
    }
  }
  // we are not converged in maxIter
  // copy our best value for u_*
  for (size_t i = 0; i < dim_xu; ++i)
    xu_bar[i] += xeta_u[i];
  return -1;
}

int fp_hos_ti_reverse(short tapeId, size_t dim_x, double **x_fix_bar,
                      size_t dim_xu, size_t degree, double **xu_bar, double **,
                      double **) {

  // assumption: x_N = Fx(x_N,u) \dot{x}_N + Fu(x_N,u)\dot{u} is taped!
  // assumption: x_N = F(x_N, u) is taped!

  std::cout << "TODO: add ADOLCError, document hos_ti_reverse" << std::endl;
  if (degree > 1)
    std::cerr << "fp_hos_reverse is not defined for degree != 1" << std::endl;

  // reference because we set the "isInternal"
  FpProblem &problem = getFpProblem(findTape(tapeId).ext_diff_fct_index());

  // 1. compute [\bar{xi_N}, \bar{u}] via fp_fos_reverse (line 8-10 in algo)
  std::vector<double> xi_u_bar(dim_xu);
  std::fill(xi_u_bar.begin(), xi_u_bar.end(), 0.0);

  // use internal Tape!
  problem.isInternal = true;
  fp_fos_reverse(problem.tapeId, dim_x, x_fix_bar[0], dim_xu, xi_u_bar.data(),
                 nullptr, nullptr);
  problem.isInternal = false;

  // 2. compute r and u via hos_reverse of subTapeNum, thus we have
  // to keep the (line 13 and part of line 21) Here we need the
  // taped fos_forward of the fp iteration -> keep = 2
  std::vector<double> xi_bar(xi_u_bar.begin(), xi_u_bar.begin() + dim_x);
  hos_reverse(problem.subTapeId, static_cast<int>(dim_x),
              static_cast<int>(dim_xu), 1, xi_bar.data(), xu_bar);

  // We now have xu_bar[1] = [xi_k^T(Fxx \dot{x} + Fxu \dot{u}),
  // xi_k^T(Fux \dot{x} + Fuu \dot(u))]

  // 3. compute fp_fos_reverse with \bar{xi} and \bar{x} = \dot{\bar{r}} =
  // \bar{xu}[1][0]
  std::vector<double> xi_bar_dot(dim_xu);
  std::vector<double> r_bar_dot(dim_x);
  for (size_t i = 0; i < dim_x; ++i) {
    r_bar_dot[i] = xu_bar[i][1];
  }

  problem.isInternal = true;
  fp_fos_reverse(problem.tapeId, dim_x, r_bar_dot.data(), dim_xu,
                 xi_bar_dot.data(), nullptr, nullptr);
  problem.isInternal = false;
  // We now have xi_bar_dot = [xi_K^T Fx + \bar{r}, xi_K^T Fu]

  // 4. return xi_k, u = u part of last fos_reverse + u part of hos_reverse
  for (size_t i = 0; i < problem.dim_u; ++i)
    xu_bar[dim_x + i][1] += xi_bar_dot[dim_x + i];

  return 0;
}

int firstOrderFp(const FpProblem &problem) {
  std::cout << "TODO: document firstOrderFP" << std::endl;
  auto edfIteration = registerFpIteration(problem);

  FixedPoint fp{.x = std::vector<double>(problem.dim_x),
                .u = std::vector<double>(problem.dim_u)};

  ValueTape &tape = findTape(problem.tapeId);
  // ensure that the adoubles are contiguous
  tape.ensureContiguousLocations(problem.dim_x + problem.dim_u);
  // to reset old default later
  const short last_default_tape_id = currentTape().tapeId();
  // ensure that the "new" allocates the adoubles for the "tape"
  setCurrentTape(tape.tapeId());
  prepareFixedPoint(edfIteration, problem, fp);
  setCurrentTape(problem.subTapeId);
  tapeLastFpIteration(problem.subTapeId, problem, fp);

  // reset previous default tape
  setCurrentTape(last_default_tape_id);

  return static_cast<int>(fp.lastIter);
}

int secondOrderFp(const FpProblem &problem) {
  std::cout << "TODO: document secondOrderFP" << std::endl;
  auto edfIteration = registerFpIteration(problem);
  ValueTape &tape = findTape(problem.tapeId);

  FixedPoint fp{.x = std::vector<double>(problem.dim_x),
                .u = std::vector<double>(problem.dim_u)};

  // ensure that the adoubles are contiguous
  tape.ensureContiguousLocations(problem.dim_x + problem.dim_u);
  // to reset old default later
  const short last_default_tape_id = currentTape().tapeId();
  // ensure that the "new" allocates the adoubles for the "tape"
  setCurrentTape(tape.tapeId());
  prepareFixedPoint(edfIteration, problem, fp);
  setCurrentTape(problem.subTapeId);
  tapeLastFpIteration(problem.subTapeId, problem, fp);

  setCurrentTape(problem.internalTapeId);
  tapeLastFpIteration(problem.internalTapeId, problem, fp);

  setCurrentTape(last_default_tape_id);
  return static_cast<int>(fp.lastIter);
}
}; // namespace ADOLC::FpIteration

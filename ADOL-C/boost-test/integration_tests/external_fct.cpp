#include "../const.h"
#include <adolc/adolc.h>
#include <array>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <functional>
#include <span>
#include <vector>

/**
 * Integration tests for the v1 external-function facility.
 *
 * The file contains four scenarios:
 * 1. Euler parity: repeated external calls in a loop must match a fully
 *    taped implementation.
 * 2. Manual analytic callbacks: hand-coded driver callbacks for one external
 *    function.
 * 3. Standard driver parity: representative ADProblem instances must behave
 *    the same with and without call_ext_fct across the forward and reverse
 *    drivers.
 * 4. Nested reverse accumulation: the inner reverse sweep must accumulate to
 *    input adjoints that the outer tape has already computed. The negative
 *    cases intentionally show what happens if nestedReverseEval is disabled or
 *    toggled on the wrong tape.
 *
 */
namespace tt = boost::test_tools;

namespace {

enum class ReverseFlagTarget { Disabled, OuterTape, InnerTape };

/**
 * Per-test configuration for the shared external driver callbacks.
 *
 * Most tests use the correct flag and therefore enable
 * nestedReverseEval on the inner tape for first-order reverse callbacks. The
 * accumulation regression overrides this to keep the positive and negative
 * cases in one place instead of duplicating callback implementations.
 */
struct ExternalDriverConfig {
  ReverseFlagTarget fosReverseTarget{ReverseFlagTarget::InnerTape};
  ReverseFlagTarget fovReverseTarget{ReverseFlagTarget::InnerTape};
  bool enableHigherOrderInnerFlag{true};
};

/**
 * Trace variants for the generic ADProblem harness.
 *
 * Standard models an external-function workflow with pre/post
 * processing and a copied contiguous input block for call_ext_fct.
 * AccumulationWithPost and AccumulationExternalOnly build the scalar reference
 * traces used by the nested reverse regression.
 */
enum class ProblemTraceVariant {
  Standard,
  AccumulationWithPost,
  AccumulationExternalOnly,
};

struct TapeSet {
  short woExternTapeId;
  short wExternTapeId;
  short externalTapeId;
};

struct ADProblem {
  const char *name;
  int n;
  int m;
  std::vector<double> x;
  std::vector<double> xd;
  std::vector<double> xd_alt;
  std::vector<double> u;
  std::vector<double> u_alt;
  std::function<void(const adouble *, adouble *)> ad_func;
  std::function<void(const double *, double *)> double_func;
};

constexpr double h = 0.01;
constexpr int steps = 100;

void compareVector(const std::vector<double> &lhs,
                   const std::vector<double> &rhs) {
  BOOST_REQUIRE_EQUAL(lhs.size(), rhs.size());
  for (size_t i = 0; i < lhs.size(); ++i)
    BOOST_TEST(lhs[i] == rhs[i], tt::tolerance(tol));
}

template <size_t N>
void compareArray(const std::array<double, N> &lhs,
                  const std::array<double, N> &rhs) {
  for (size_t i = 0; i < N; ++i)
    BOOST_TEST(lhs[i] == rhs[i], tt::tolerance(tol));
}

template <size_t Rows, size_t Cols>
void compareArrayMatrix(const std::array<std::array<double, Cols>, Rows> &lhs,
                        const std::array<std::array<double, Cols>, Rows> &rhs) {
  for (size_t i = 0; i < Rows; ++i)
    compareArray(lhs[i], rhs[i]);
}

void compareMatrix(double **lhs, double **rhs, int rows, int cols) {
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      BOOST_TEST(lhs[i][j] == rhs[i][j], tt::tolerance(tol));
}

void setNestedReverseFlag(ReverseFlagTarget target, short outerTapeId,
                          short innerTapeId, bool enabled) {
  switch (target) {
  case ReverseFlagTarget::Disabled:
    break;
  case ReverseFlagTarget::OuterTape:
    findTape(outerTapeId).nestedReverseEval(enabled);
    break;
  case ReverseFlagTarget::InnerTape:
    findTape(innerTapeId).nestedReverseEval(enabled);
    break;
  }
}

/**
 * Register one external function and wire all driver callbacks used by this
 * file.
 */
ext_diff_fct *registerExternalFunction(short outerTapeId, short innerTapeId,
                                       ADOLC_ext_fct passiveFunction,
                                       ExternalDriverConfig config = {}) {
  ext_diff_fct *edf =
      reg_ext_fct(outerTapeId, innerTapeId, std::move(passiveFunction));

  edf->zos_forward = [innerTapeId](short, int m, int n, double *x, double *y) {
    return zos_forward(innerTapeId, m, n, 0, x, y);
  };

  edf->fos_forward = [innerTapeId](short, int m, int n, double *x, double *X,
                                   double *y, double *Y) {
    return fos_forward(innerTapeId, m, n, 0, x, X, y, Y);
  };

  edf->fov_forward = [innerTapeId](short, int m, int n, int p, double *x,
                                   double **Xp, double *y, double **Yp) {
    return fov_forward(innerTapeId, m, n, p, x, Xp, y, Yp);
  };

  edf->fos_reverse = [innerTapeId, config](short outerTapeId, int m, int n,
                                           double *u, double *z, double *x,
                                           double *y) {
    zos_forward(innerTapeId, m, n, 1, x, y);
    setNestedReverseFlag(config.fosReverseTarget, outerTapeId, innerTapeId,
                         true);
    const int rc = fos_reverse(innerTapeId, m, n, u, z);
    setNestedReverseFlag(config.fosReverseTarget, outerTapeId, innerTapeId,
                         false);
    return rc;
  };

  edf->fov_reverse = [innerTapeId, config](short outerTapeId, int m, int n,
                                           int q, double **Uq, double **Zq,
                                           double *x, double *y) {
    zos_forward(innerTapeId, m, n, 1, x, y);
    setNestedReverseFlag(config.fovReverseTarget, outerTapeId, innerTapeId,
                         true);
    const int rc = fov_reverse(innerTapeId, m, n, q, Uq, Zq);
    setNestedReverseFlag(config.fovReverseTarget, outerTapeId, innerTapeId,
                         false);
    return rc;
  };

  edf->hos_ti_reverse = [innerTapeId, config](short, int m, int n, int d,
                                              double **Uqd, double **Zd,
                                              double **, double **) {
    if (config.enableHigherOrderInnerFlag)
      findTape(innerTapeId).nestedReverseEval(true);
    const int rc = hos_ti_reverse(innerTapeId, m, n, d, Uqd, Zd);
    if (config.enableHigherOrderInnerFlag)
      findTape(innerTapeId).nestedReverseEval(false);
    return rc;
  };

  edf->hov_reverse = [innerTapeId, config](short, int m, int n, int d, int q,
                                           double **Uq, double ***Zqd,
                                           short **nz, double **, double **) {
    if (config.enableHigherOrderInnerFlag)
      findTape(innerTapeId).nestedReverseEval(true);
    const int rc = hov_reverse(innerTapeId, m, n, d, q, Uq, Zqd, nz);
    if (config.enableHigherOrderInnerFlag)
      findTape(innerTapeId).nestedReverseEval(false);
    return rc;
  };

  return edf;
}

void eulerStepAct(const adouble *yin, adouble *yout) {
  yout[0] = yin[0] + h * yin[0];
  yout[1] = yin[1] + h * 2.0 * yin[1];
}

int eulerStep(short, int, int, double *yin, double *yout) {
  yout[0] = yin[0] + h * yin[0];
  yout[1] = yin[1] + h * 2.0 * yin[1];
  return 1;
}

/**
 * Scenario 1: repeated external calls inside a time-stepping loop.
 *
 * This is intentionally separate from the generic ADProblem tests because it
 * stresses a different property: many external invocations on one outer tape
 * must still produce the same gradient as the fully taped loop.
 */
template <typename T> void eulerPreComputation(T *x) {
  const T x0 = x[0];
  const T x1 = x[1];
  x[0] = x0 + 2.0 * x1;
  x[1] = -x0 + 0.5 * x1;
}

template <typename T> void eulerPostComputation(const T *x, T *y) {
  y[0] = y[0] + x[0];
  y[1] = 2.0 * y[1] - x[1];
}

void traceEulerFullTape(short tapeId, const std::array<double, 2> &controls) {
  trace_on(tapeId);
  {
    std::array<adouble, 2> state{};
    std::array<adouble, 2> nextState{};
    std::array<adouble, 2> finalState{};
    std::array<adouble, 2> control{};

    control <<= controls;
    state = control;
    eulerPreComputation(state.data());

    for (int i = 0; i < steps; ++i) {
      eulerStepAct(state.data(), nextState.data());
      state = nextState;
    }

    finalState = state;
    eulerPostComputation(state.data(), finalState.data());

    adouble f = finalState[0] + finalState[1];
    double fOut = 0.0;
    f >>= fOut;
  }
  trace_off();
}

void traceEulerInnerTape(short tapeId, const std::array<double, 2> &controls) {
  trace_on(tapeId);
  {
    std::array<adouble, 2> y{};
    std::array<adouble, 2> ynew{};

    y <<= controls;

    eulerStepAct(y.data(), ynew.data());

    std::array<double, 2> dummy{};
    ynew >>= dummy;
  }
  trace_off();
}

void traceEulerOuterTape(short tapeId, ext_diff_fct *edf,
                         const std::array<double, 2> &controls) {
  trace_on(tapeId);
  {
    std::array<adouble, 2> state{};
    std::array<adouble, 2> nextState{};
    std::array<adouble, 2> finalState{};
    std::array<adouble, 2> extIn{};
    std::array<adouble, 2> extOut{};
    std::array<adouble, 2> control{};
    currentTape().ensureContiguousLocations(4);

    control <<= controls;
    state = control;
    eulerPreComputation(state.data());

    for (int i = 0; i < steps; ++i) {
      extIn = state;
      call_ext_fct(edf, 2, extIn.data(), 2, extOut.data());
      state = extOut;
    }

    finalState = state;
    eulerPostComputation(state.data(), finalState.data());

    adouble f = finalState[0] + finalState[1];
    double fOut = 0.0;
    f >>= fOut;
  }
  trace_off();
}

/**
 * Smoke test for repeated external calls.
 *
 * The same Euler step is traced once as a normal taped loop and once as a loop
 * around call_ext_fct. The test compares the resulting gradients and gives us
 * a compact sanity check that the shared callback wiring works in a repeated
 * nested setting.
 */
void checkEulerGradientParity() {
  const TapeSet tapes{createNewTape(), createNewTape(), createNewTape()};
  const std::array<double, 2> controls{1.0, 1.0};
  std::vector<double> gradWoExtern(2, 0.0);
  std::vector<double> gradWExtern(2, 0.0);

  traceEulerFullTape(tapes.woExternTapeId, controls);
  gradient(tapes.woExternTapeId, 2, controls.data(), gradWoExtern.data());

  traceEulerInnerTape(tapes.externalTapeId, controls);
  ext_diff_fct *edf = registerExternalFunction(
      tapes.wExternTapeId, tapes.externalTapeId, ADOLC_ext_fct(eulerStep));
  traceEulerOuterTape(tapes.wExternTapeId, edf, controls);
  gradient(tapes.wExternTapeId, 2, controls.data(), gradWExtern.data());

  compareVector(gradWoExtern, gradWExtern);
}

/**
 * Scenario 2: analytic external-function callbacks without a nested ADOL-C
 * tape.
 *
 * This tests the use case for external functions: the primal
 * evaluation and first-order derivatives are provided explicitly, so the outer
 * tape can reuse them directly instead of differentiating an inner tape.
 */
using ManualExternalPoint = std::array<double, 2>;
using ManualExternalMatrix = std::array<std::array<double, 2>, 2>;

ManualExternalPoint manualExternalValue(const ManualExternalPoint &x) {
  return {x[0] * x[1], std::sin(x[0]) + x[1] * x[1]};
}

ManualExternalMatrix manualExternalJacobian(const ManualExternalPoint &x) {
  return {{{x[1], x[0]}, {std::cos(x[0]), 2.0 * x[1]}}};
}

ManualExternalPoint
applyManualExternalJacobian(const ManualExternalMatrix &jacobian,
                            const ManualExternalPoint &direction) {
  return {jacobian[0][0] * direction[0] + jacobian[0][1] * direction[1],
          jacobian[1][0] * direction[0] + jacobian[1][1] * direction[1]};
}

ManualExternalPoint
applyManualExternalTranspose(const ManualExternalMatrix &jacobian,
                             const ManualExternalPoint &weights) {
  return {weights[0] * jacobian[0][0] + weights[1] * jacobian[1][0],
          weights[0] * jacobian[0][1] + weights[1] * jacobian[1][1]};
}

int manualExternalFunction(short, int m, int n, double *x, double *y) {
  if (m != 2 || n != 2)
    return -1;

  const ManualExternalPoint point{x[0], x[1]};
  const auto value = manualExternalValue(point);
  y[0] = value[0];
  y[1] = value[1];
  return 0;
}

/**
 * Register an external function whose first-order callbacks are hand-coded.
 *
 * The placeholder inner tape id is required by reg_ext_fct but is not used.
 * nestedAdolc is disabled because no inner ADOL-C driver is called here.
 */
ext_diff_fct *registerManualExternalFunction(short outerTapeId,
                                             short placeholderInnerTapeId) {
  ext_diff_fct *edf = reg_ext_fct(outerTapeId, placeholderInnerTapeId,
                                  ADOLC_ext_fct(manualExternalFunction));
  edf->nestedAdolc = 0;

  edf->zos_forward = [](short tapeId, int m, int n, double *x, double *y) {
    return manualExternalFunction(tapeId, m, n, x, y);
  };

  edf->fos_forward = [](short tapeId, int m, int n, double *x, double *X,
                        double *y, double *Y) {
    const int rc = manualExternalFunction(tapeId, m, n, x, y);
    if (rc != 0)
      return rc;

    const ManualExternalPoint point{x[0], x[1]};
    const ManualExternalPoint direction{X[0], X[1]};
    const auto projection =
        applyManualExternalJacobian(manualExternalJacobian(point), direction);
    Y[0] = projection[0];
    Y[1] = projection[1];
    return 0;
  };

  edf->fov_forward = [](short tapeId, int m, int n, int p, double *x,
                        double **Xp, double *y, double **Yp) {
    const int rc = manualExternalFunction(tapeId, m, n, x, y);
    if (rc != 0)
      return rc;

    const ManualExternalPoint point{x[0], x[1]};
    const auto jacobian = manualExternalJacobian(point);
    for (int direction = 0; direction < p; ++direction) {
      const ManualExternalPoint X{Xp[0][direction], Xp[1][direction]};
      const auto projection = applyManualExternalJacobian(jacobian, X);
      Yp[0][direction] = projection[0];
      Yp[1][direction] = projection[1];
    }
    return 0;
  };

  edf->fos_reverse = [](short, int m, int n, double *u, double *z, double *x,
                        double *) {
    if (m != 2 || n != 2)
      return -1;

    const ManualExternalPoint point{x[0], x[1]};
    const ManualExternalPoint weights{u[0], u[1]};
    const auto projection =
        applyManualExternalTranspose(manualExternalJacobian(point), weights);
    z[0] += projection[0];
    z[1] += projection[1];
    return 0;
  };

  edf->fov_reverse = [](short, int m, int n, int q, double **Uq, double **Zq,
                        double *x, double *) {
    if (m != 2 || n != 2)
      return -1;

    const ManualExternalPoint point{x[0], x[1]};
    const auto jacobian = manualExternalJacobian(point);
    for (int weight = 0; weight < q; ++weight) {
      const ManualExternalPoint U{Uq[weight][0], Uq[weight][1]};
      const auto projection = applyManualExternalTranspose(jacobian, U);
      Zq[weight][0] += projection[0];
      Zq[weight][1] += projection[1];
    }
    return 0;
  };

  return edf;
}

void traceManualExternalOutputTape(short tapeId, ext_diff_fct *edf,
                                   const ManualExternalPoint &x) {
  trace_on(tapeId, 1);
  {
    currentTape().ensureContiguousLocations(4);
    std::vector<adouble> args(4);
    std::span<adouble> ax(args.data(), 2);
    std::span<adouble> ay(args.data() + 2, 2);
    std::array<double, 2> out{0.0, 0.0};

    ax <<= std::span<const double>(x);
    call_ext_fct(edf, 2, ax.data(), 2, ay.data());
    ay >>= std::span<double>(out);
  }
  trace_off();
}

void traceManualExternalScalarTape(short tapeId, ext_diff_fct *edf,
                                   const ManualExternalPoint &x) {
  trace_on(tapeId, 1);
  {
    currentTape().ensureContiguousLocations(4);
    std::vector<adouble> args(4);
    std::span<adouble> ax(args.data(), 2);
    std::span<adouble> ay(args.data() + 2, 2);
    double out = 0.0;

    ax <<= std::span<const double>(x);
    call_ext_fct(edf, 2, ax.data(), 2, ay.data());

    adouble objective = 5.0 * ax[0] - 2.0 * ax[1] + ay[0] + 3.0 * ay[1];
    objective >>= out;
  }
  trace_off();
}

ManualExternalPoint
manualExternalObjectiveGradient(const ManualExternalPoint &x) {
  const ManualExternalPoint outerWeights{1.0, 3.0};
  auto gradient =
      applyManualExternalTranspose(manualExternalJacobian(x), outerWeights);
  gradient[0] += 5.0;
  gradient[1] -= 2.0;
  return gradient;
}

/**
 * Verifies a fully hand-coded external-function implementation.
 *
 * The outer tape sees the external call as a black box. All first-order driver
 * results are therefore produced by the hard-coded callback formulas below and
 * checked against explicit reference derivatives.
 */
void checkManualExternalAnalyticCallbacks() {
  const ManualExternalPoint x{0.7, -1.2};
  const short outputsTapeId = createNewTape();
  const short scalarTapeId = createNewTape();
  const short placeholderOutputsInnerTapeId = createNewTape();
  const short placeholderScalarInnerTapeId = createNewTape();

  ext_diff_fct *outputsEdf = registerManualExternalFunction(
      outputsTapeId, placeholderOutputsInnerTapeId);
  ext_diff_fct *scalarEdf = registerManualExternalFunction(
      scalarTapeId, placeholderScalarInnerTapeId);

  traceManualExternalOutputTape(outputsTapeId, outputsEdf, x);
  traceManualExternalScalarTape(scalarTapeId, scalarEdf, x);

  const auto expectedValue = manualExternalValue(x);
  const auto jacobian = manualExternalJacobian(x);

  std::vector<double> y(2, 0.0);
  int rc = zos_forward(outputsTapeId, 2, 2, 0, x.data(), y.data());
  BOOST_REQUIRE_EQUAL(rc, 0);
  compareArray(expectedValue, ManualExternalPoint{y[0], y[1]});

  ManualExternalPoint direction{1.0, -0.5};
  std::vector<double> dy(2, 0.0);
  rc = fos_forward(outputsTapeId, 2, 2, 0, x.data(), direction.data(), y.data(),
                   dy.data());
  BOOST_REQUIRE_EQUAL(rc, 0);
  compareArray(expectedValue, ManualExternalPoint{y[0], y[1]});
  compareArray(applyManualExternalJacobian(jacobian, direction),
               ManualExternalPoint{dy[0], dy[1]});

  const int p = 2;
  double **Xp = myalloc2(2, p);
  double **Yp = myalloc2(2, p);
  Xp[0][0] = 1.0;
  Xp[1][0] = -0.5;
  Xp[0][1] = 0.25;
  Xp[1][1] = 1.5;
  rc = fov_forward(outputsTapeId, 2, 2, p, x.data(), Xp, y.data(), Yp);
  BOOST_REQUIRE_EQUAL(rc, 0);
  compareArray(expectedValue, ManualExternalPoint{y[0], y[1]});
  ManualExternalMatrix expectedYp{
      {applyManualExternalJacobian(jacobian,
                                   ManualExternalPoint{Xp[0][0], Xp[1][0]}),
       applyManualExternalJacobian(jacobian,
                                   ManualExternalPoint{Xp[0][1], Xp[1][1]})}};
  ManualExternalMatrix actualYp{{{Yp[0][0], Yp[1][0]}, {Yp[0][1], Yp[1][1]}}};
  compareArrayMatrix(expectedYp, actualYp);
  myfree2(Yp);
  myfree2(Xp);

  ManualExternalPoint weights{1.25, -0.75};
  std::vector<double> z(2, 0.0);
  rc = fos_reverse(outputsTapeId, 2, 2, weights.data(), z.data());
  BOOST_REQUIRE_EQUAL(rc, 0);
  compareArray(applyManualExternalTranspose(jacobian, weights),
               ManualExternalPoint{z[0], z[1]});

  const int q = 2;
  double **Uq = myalloc2(q, 2);
  double **Zq = myalloc2(q, 2);
  Uq[0][0] = 1.25;
  Uq[0][1] = -0.75;
  Uq[1][0] = -0.5;
  Uq[1][1] = 2.0;
  rc = fov_reverse(outputsTapeId, 2, 2, q, Uq, Zq);
  BOOST_REQUIRE_EQUAL(rc, 0);
  ManualExternalMatrix expectedZq{
      {applyManualExternalTranspose(jacobian,
                                    ManualExternalPoint{Uq[0][0], Uq[0][1]}),
       applyManualExternalTranspose(jacobian,
                                    ManualExternalPoint{Uq[1][0], Uq[1][1]})}};
  ManualExternalMatrix actualZq{{{Zq[0][0], Zq[0][1]}, {Zq[1][0], Zq[1][1]}}};
  compareArrayMatrix(expectedZq, actualZq);
  myfree2(Zq);
  myfree2(Uq);

  std::vector<double> gradientResult(2, 0.0);
  gradient(scalarTapeId, 2, x.data(), gradientResult.data());
  compareArray(manualExternalObjectiveGradient(x),
               ManualExternalPoint{gradientResult[0], gradientResult[1]});
}

template <typename T> void preComputation(const ADProblem &problem, T *x) {
  if (problem.n == 2) {
    const T x0 = x[0];
    const T x1 = x[1];
    x[0] = x0 + 2.0 * x1;
    x[1] = -x0 + 0.5 * x1;
    return;
  }

  if (problem.n == 3) {
    const T x0 = x[0];
    const T x1 = x[1];
    const T x2 = x[2];
    x[0] = x0 + 2.0 * x1;
    x[1] = -x0 + 0.5 * x1;
    x[2] = x2 - x0;
  }
}

template <typename T>
void postComputation(const ADProblem &problem, const T *x, T *y) {
  if (problem.m == 1) {
    y[0] = y[0] + x[0] - 0.5 * x[1];
    return;
  }

  if (problem.m == 2) {
    y[0] = y[0] + x[0];
    y[1] = 2.0 * y[1] - x[1];
    return;
  }

  if (problem.m == 3) {
    y[0] = y[0] + x[0];
    y[1] = 2.0 * y[1] - x[1];
    y[2] = y[2] + x[0] - 0.25 * x[1];
  }
}

/**
 * Scenario 3 and 4 use a shared ADProblem harness.
 *
 * The standard workflow tests compare a fully taped trace with an outer tape
 * that calls into an externally taped inner function. The nested accumulation
 * regression reuses the same harness with scalar objectives that make overwrite
 * versus accumulate semantics visible in reverse mode.
 */
void traceProblemInnerTape(short tapeId, const ADProblem &problem) {
  trace_on(tapeId, 1);
  {
    std::vector<adouble> ax(problem.n);
    std::vector<adouble> ay(problem.m);
    std::vector<double> out(problem.m, 0.0);

    ax <<= problem.x;

    problem.ad_func(ax.data(), ay.data());

    ay >>= out;
  }
  trace_off();
}

template <typename T> T sumOutputs(const T *values, int count) {
  T sum = T(0.0);
  for (int i = 0; i < count; ++i)
    sum += values[i];
  return sum;
}

template <typename T> T sumPostOutputs(const ADProblem &problem, T *x) {
  std::vector<T> post(problem.m, T(0.0));
  postComputation(problem, x, post.data());
  return sumOutputs(post.data(), problem.m);
}

/**
 * Build the non-external reference trace for one ADProblem variant.
 *
 * Standard keeps the full pre/post structure and returns the original output
 * vector. The accumulation variants collapse the computation to a scalar
 * objective so that the reverse regression can compare accumulate-versus-
 * overwrite behavior against explicit references.
 */
void traceProblemWithoutExternal(short tapeId, const ADProblem &problem,
                                 ProblemTraceVariant variant) {
  trace_on(tapeId, 1);
  {
    if (variant == ProblemTraceVariant::Standard) {
      std::vector<adouble> ax(problem.n);
      std::vector<adouble> ay(problem.m);
      std::vector<double> out(problem.m, 0.0);

      ax <<= problem.x;

      preComputation(problem, ax.data());
      problem.ad_func(ax.data(), ay.data());
      postComputation(problem, ax.data(), ay.data());

      ay >>= out;
    } else {
      double out = 0.0;

      currentTape().ensureContiguousLocations(to_size_t(problem.n + problem.m));
      std::vector<adouble> args(problem.n + problem.m);
      std::span<adouble> ax(args.data(), problem.n);
      std::span<adouble> ay(args.data() + problem.n, problem.m);

      ax <<= std::span<const double>(problem.x);

      problem.ad_func(ax.data(), ay.data());
      adouble f = sumOutputs(ay.data(), problem.m);
      if (variant == ProblemTraceVariant::AccumulationWithPost)
        f += sumPostOutputs(problem, ax.data());

      f >>= out;
    }
  }
  trace_off();
}

/**
 * Build the outer tape that calls into the externally taped inner function.
 *
 * The Standard variant copies the processed inputs into a separate contiguous
 * block because that mirrors the common call_ext_fct workflow. The
 * accumulation variants intentionally pass the outer x block directly so that
 * both the outer post contribution and the nested reverse call write adjoints
 * into the same logical inputs.
 */
void traceProblemWithExternal(short tapeId, ext_diff_fct *edf,
                              const ADProblem &problem,
                              ProblemTraceVariant variant) {
  trace_on(tapeId, 1);
  {
    if (variant == ProblemTraceVariant::Standard) {
      std::vector<adouble> ax(problem.n);
      std::vector<double> out(problem.m, 0.0);

      ax <<= problem.x;

      preComputation(problem, ax.data());

      currentTape().ensureContiguousLocations(to_size_t(problem.n + problem.m));
      std::vector<adouble> args(problem.n + problem.m);
      std::span<adouble> axForExtern(args.data(), problem.n);
      std::span<adouble> ay(args.data() + problem.n, problem.m);
      std::copy(ax.begin(), ax.end(), axForExtern.begin());

      call_ext_fct(edf, problem.n, axForExtern.data(), problem.m, ay.data());
      postComputation(problem, ax.data(), ay.data());

      ay >>= std::span<double>(out);
    } else {
      double out = 0.0;

      currentTape().ensureContiguousLocations(to_size_t(problem.n + problem.m));
      std::vector<adouble> args(problem.n + problem.m);
      std::span<adouble> ax(args.data(), problem.n);
      std::span<adouble> ay(args.data() + problem.n, problem.m);

      ax <<= std::span<const double>(problem.x);

      call_ext_fct(edf, problem.n, ax.data(), problem.m, ay.data());

      adouble f = sumOutputs(ay.data(), problem.m);
      if (variant == ProblemTraceVariant::AccumulationWithPost)
        f += sumPostOutputs(problem, ax.data());

      f >>= out;
    }
  }
  trace_off();
}

/**
 * Create the three tapes used by the standard workflow parity checks:
 * a fully taped reference, an inner tape for the external function itself,
 * and an outer tape that calls it via call_ext_fct.
 */
TapeSet createStandardProblemTapes(const ADProblem &problem) {
  const TapeSet tapes{createNewTape(), createNewTape(), createNewTape()};

  traceProblemWithoutExternal(tapes.woExternTapeId, problem,
                              ProblemTraceVariant::Standard);
  traceProblemInnerTape(tapes.externalTapeId, problem);

  ext_diff_fct *edf = registerExternalFunction(
      tapes.wExternTapeId, tapes.externalTapeId,
      [problem](short, int, int, double *x, double *y) {
        problem.double_func(x, y);
        return 0;
      });
  traceProblemWithExternal(tapes.wExternTapeId, edf, problem,
                           ProblemTraceVariant::Standard);

  return tapes;
}

/**
 * Driver parity test for the regular external-function workflow.
 *
 * For one ADProblem instance, this builds both the fully taped computation and
 * the externalized version, then compares the driver results that we rely on
 * in this file: zos_forward, fos_forward, fov_forward, fos_reverse, and
 * fov_reverse. New representative problems should normally be added by calling
 * this helper from another BOOST test case.
 */
void checkStandardWorkflowParity(const ADProblem &problem) {
  const TapeSet tapes = createStandardProblemTapes(problem);

  std::vector<double> yWoExtern(problem.m, 0.0);
  std::vector<double> yWExtern(problem.m, 0.0);
  int rcWoExtern = zos_forward(tapes.woExternTapeId, problem.m, problem.n, 0,
                               problem.x.data(), yWoExtern.data());
  int rcWExtern = zos_forward(tapes.wExternTapeId, problem.m, problem.n, 0,
                              problem.x.data(), yWExtern.data());
  BOOST_REQUIRE_EQUAL(rcWoExtern, rcWExtern);
  compareVector(yWoExtern, yWExtern);

  std::vector<double> ydWoExtern(problem.m, 0.0);
  std::vector<double> ydWExtern(problem.m, 0.0);
  rcWoExtern = fos_forward(tapes.woExternTapeId, problem.m, problem.n, 0,
                           problem.x.data(), problem.xd.data(),
                           yWoExtern.data(), ydWoExtern.data());
  rcWExtern = fos_forward(tapes.wExternTapeId, problem.m, problem.n, 0,
                          problem.x.data(), problem.xd.data(), yWExtern.data(),
                          ydWExtern.data());
  BOOST_REQUIRE_EQUAL(rcWoExtern, rcWExtern);
  compareVector(yWoExtern, yWExtern);
  compareVector(ydWoExtern, ydWExtern);

  const int p = 2;
  double **XpWoExtern = myalloc2(problem.n, p);
  double **XpWExtern = myalloc2(problem.n, p);
  double **YpWoExtern = myalloc2(problem.m, p);
  double **YpWExtern = myalloc2(problem.m, p);
  for (int i = 0; i < problem.n; ++i) {
    XpWoExtern[i][0] = problem.xd[i];
    XpWoExtern[i][1] = problem.xd_alt[i];
    XpWExtern[i][0] = problem.xd[i];
    XpWExtern[i][1] = problem.xd_alt[i];
  }
  rcWoExtern =
      fov_forward(tapes.woExternTapeId, problem.m, problem.n, p,
                  problem.x.data(), XpWoExtern, yWoExtern.data(), YpWoExtern);
  rcWExtern =
      fov_forward(tapes.wExternTapeId, problem.m, problem.n, p,
                  problem.x.data(), XpWExtern, yWExtern.data(), YpWExtern);
  BOOST_REQUIRE_EQUAL(rcWoExtern, rcWExtern);
  compareVector(yWoExtern, yWExtern);
  compareMatrix(YpWoExtern, YpWExtern, problem.m, p);
  myfree2(YpWExtern);
  myfree2(YpWoExtern);
  myfree2(XpWExtern);
  myfree2(XpWoExtern);

  std::vector<double> zWoExtern(problem.n, 0.0);
  std::vector<double> zWExtern(problem.n, 0.0);
  rcWoExtern = fos_reverse(tapes.woExternTapeId, problem.m, problem.n,
                           problem.u.data(), zWoExtern.data());
  rcWExtern = fos_reverse(tapes.wExternTapeId, problem.m, problem.n,
                          problem.u.data(), zWExtern.data());
  BOOST_REQUIRE_EQUAL(rcWoExtern, rcWExtern);
  compareVector(zWoExtern, zWExtern);

  const int q = 2;
  double **UqWoExtern = myalloc2(q, problem.m);
  double **UqWExtern = myalloc2(q, problem.m);
  double **ZqWoExtern = myalloc2(q, problem.n);
  double **ZqWExtern = myalloc2(q, problem.n);
  for (int j = 0; j < problem.m; ++j) {
    UqWoExtern[0][j] = problem.u[j];
    UqWoExtern[1][j] = problem.u_alt[j];
    UqWExtern[0][j] = problem.u[j];
    UqWExtern[1][j] = problem.u_alt[j];
  }
  rcWoExtern = fov_reverse(tapes.woExternTapeId, problem.m, problem.n, q,
                           UqWoExtern, ZqWoExtern);
  rcWExtern = fov_reverse(tapes.wExternTapeId, problem.m, problem.n, q,
                          UqWExtern, ZqWExtern);
  BOOST_REQUIRE_EQUAL(rcWoExtern, rcWExtern);
  compareMatrix(ZqWoExtern, ZqWExtern, q, problem.n);
  myfree2(ZqWExtern);
  myfree2(ZqWoExtern);
  myfree2(UqWExtern);
  myfree2(UqWoExtern);
}

std::vector<double> gradientForTape(short tapeId, const ADProblem &problem) {
  std::vector<double> grad(problem.n, 0.0);
  gradient(tapeId, problem.n, problem.x.data(), grad.data());
  return grad;
}

std::vector<double>
gradientForProblemWithoutExternal(const ADProblem &problem,
                                  ProblemTraceVariant variant) {
  const short tapeId = createNewTape();
  traceProblemWithoutExternal(tapeId, problem, variant);
  return gradientForTape(tapeId, problem);
}

std::vector<double> gradientForExternalProblem(const ADProblem &problem,
                                               ProblemTraceVariant variant,
                                               ExternalDriverConfig config) {
  const short outerTapeId = createNewTape();
  const short innerTapeId = createNewTape();

  traceProblemInnerTape(innerTapeId, problem);
  ext_diff_fct *edf = registerExternalFunction(
      outerTapeId, innerTapeId,
      [problem](short, int, int, double *x, double *y) {
        problem.double_func(x, y);
        return 0;
      },
      config);
  traceProblemWithExternal(outerTapeId, edf, problem, variant);
  return gradientForTape(outerTapeId, problem);
}

/**
 * Regression test for nested reverse accumulation.
 *
 * The scalar objective is split into two contributions that both depend on the
 * same input block x: one comes from the external call and one is computed on
 * the outer tape after that call. During reverse mode, the outer contribution
 * reaches x first. The inner reverse sweep must therefore add its adjoints to
 * the existing z values instead of overwriting them.
 *
 * The assertions compare three external-driver configurations against explicit
 * references:
 * - InnerTape must match the fully taped accumulation reference.
 * - Disabled and OuterTape must both match the overwrite-style reference where
 *   only the external contribution survives.
 */
void checkNestedReverseAccumulation(const ADProblem &problem) {
  const auto gradFull = gradientForProblemWithoutExternal(
      problem, ProblemTraceVariant::AccumulationWithPost);
  const auto gradExternalOnly = gradientForProblemWithoutExternal(
      problem, ProblemTraceVariant::AccumulationExternalOnly);

  const auto gradDisabled = gradientForExternalProblem(
      problem, ProblemTraceVariant::AccumulationWithPost,
      ExternalDriverConfig{ReverseFlagTarget::Disabled,
                           ReverseFlagTarget::Disabled, true});
  const auto gradOuter = gradientForExternalProblem(
      problem, ProblemTraceVariant::AccumulationWithPost,
      ExternalDriverConfig{ReverseFlagTarget::OuterTape,
                           ReverseFlagTarget::OuterTape, true});
  const auto gradInner = gradientForExternalProblem(
      problem, ProblemTraceVariant::AccumulationWithPost,
      ExternalDriverConfig{ReverseFlagTarget::InnerTape,
                           ReverseFlagTarget::InnerTape, true});

  compareVector(gradInner, gradFull);
  compareVector(gradDisabled, gradExternalOnly);
  compareVector(gradOuter, gradExternalOnly);
}

/**
 * Representative problems for the shared workflow harness.
 *
 * Together they cover linear and nonlinear behavior as well as scalar and
 * vector output shapes without duplicating the driver comparison code.
 */
const ADProblem linearExample{"linear",
                              2,
                              2,
                              {0.4, -1.1},
                              {1.0, -0.5},
                              {0.25, 1.5},
                              {1.0, -0.75},
                              {0.5, 2.0},
                              [](const adouble *x, adouble *y) {
                                y[0] = 2.0 * x[0] - 3.0 * x[1];
                                y[1] = x[0] + 4.0 * x[1];
                              },
                              [](const double *x, double *y) {
                                y[0] = 2.0 * x[0] - 3.0 * x[1];
                                y[1] = x[0] + 4.0 * x[1];
                              }};

const ADProblem polynomialExample{"polynomial",
                                  3,
                                  2,
                                  {1.2, -0.3, 0.7},
                                  {1.0, 0.0, -0.5},
                                  {-0.25, 1.0, 0.75},
                                  {1.0, -0.4},
                                  {0.5, 1.2},
                                  [](const adouble *x, adouble *y) {
                                    y[0] = x[0] * x[0] + x[1] + x[2];
                                    y[1] = x[0] * x[2] + x[1] * x[1];
                                  },
                                  [](const double *x, double *y) {
                                    y[0] = x[0] * x[0] + x[1] + x[2];
                                    y[1] = x[0] * x[2] + x[1] * x[1];
                                  }};

const ADProblem bilinearExample{"bilinear",
                                2,
                                3,
                                {0.8, -1.4},
                                {1.0, 0.25},
                                {-0.5, 1.0},
                                {1.0, 0.0, -0.5},
                                {0.25, 1.5, 0.75},
                                [](const adouble *x, adouble *y) {
                                  y[0] = x[0] * x[1];
                                  y[1] = x[0] + x[1] * x[1];
                                  y[2] = x[0] * x[0] - x[1];
                                },
                                [](const double *x, double *y) {
                                  y[0] = x[0] * x[1];
                                  y[1] = x[0] + x[1] * x[1];
                                  y[2] = x[0] * x[0] - x[1];
                                }};

const ADProblem scalarExample{
    "scalar",
    3,
    1,
    {1.1, -0.6, 0.9},
    {1.0, -0.5, 0.25},
    {0.1, 1.0, -0.75},
    {1.0},
    {2.0},
    [](const adouble *x, adouble *y) {
      y[0] = x[0] * x[0] + x[0] * x[1] + 3.0 * x[2] * x[2];
    },
    [](const double *x, double *y) {
      y[0] = x[0] * x[0] + x[0] * x[1] + 3.0 * x[2] * x[2];
    }};

} // namespace

/**
 * High-level external-function regressions.
 *
 * These cases check the smoke-level repeated-call behavior, the explicit
 * analytic callback path, and the dedicated nested reverse accumulation
 * semantics that motivated nestedReverseEval.
 */
BOOST_AUTO_TEST_SUITE(ExternalFunctionTests)

/** Verifies repeated external Euler steps against the fully taped loop. */
BOOST_AUTO_TEST_CASE(CompareFullAndExternalGradients) {
  checkEulerGradientParity();
}

/** Verifies hand-coded external callbacks against analytic derivatives. */
BOOST_AUTO_TEST_CASE(ManualExternalCallbacksMatchAnalyticReference) {
  checkManualExternalAnalyticCallbacks();
}

/** Verifies that nestedReverseEval must be enabled on the inner tape. */
BOOST_AUTO_TEST_CASE(NestedReverseAccumulationRequiresInnerTapeFlag) {
  checkNestedReverseAccumulation(scalarExample);
}

BOOST_AUTO_TEST_SUITE_END()

/**
 * Representative workflow parity checks.
 *
 * Each case reuses the same harness with a different nonlinear problem so that
 * we cover scalar/vector inputs and outputs without duplicating the driver
 * comparison logic.
 */
BOOST_AUTO_TEST_SUITE(ExternalFunctionDriverWorkflowTests)

BOOST_AUTO_TEST_CASE(LinearExternalFunctionWorkflows) {
  checkStandardWorkflowParity(linearExample);
}

BOOST_AUTO_TEST_CASE(PolynomialExternalFunctionWorkflows) {
  checkStandardWorkflowParity(polynomialExample);
}

BOOST_AUTO_TEST_CASE(BilinearExternalFunctionWorkflows) {
  checkStandardWorkflowParity(bilinearExample);
}

BOOST_AUTO_TEST_CASE(ScalarExternalFunctionWorkflows) {
  checkStandardWorkflowParity(scalarExample);
}

BOOST_AUTO_TEST_SUITE_END()

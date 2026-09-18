
#include "adolc/adalloc.h"
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

#include <adolc/adolc.h>
#include <array>

#include "const.h"

BOOST_AUTO_TEST_SUITE(FixedPointBasicDriverTest)

/************************************************************/
/* Tests for automatic differentiation of fixed-point loops */
/************************************************************/

/* One iteration of Newton's method for finding zeros of f(x) = x^2 - z
 *
 * The solution is the square root of 'z'.
 * We are interested in derivatives with respect to 'z'.
 *
 * \tparam T The number type.  Both double and adouble versions are needed
 */
template <typename T> static int iteration(T *x, T *u, T *x_fix, int, int) {
  // Newton update: x = x - f(x)/f'(x) = x - (x*x-z) / 2x = x - x/2 + z/2x
  x_fix[0] = 0.5 * (x[0] + u[0] / x[0]);
  return 0;
}

static double norm(double *x, int dim) {
  double norm = 0;

  for (size_t i = 0; i < dim; i++)
    norm += x[i] * x[i];

  return std::sqrt(norm);
}

static double traceNewtonForSquareRoot(short tapeId, short sub_tape_id,
                                       double argument) {
  // ax1 = sqrt(ax1);
  setCurrentTape(tapeId);
  currentTape().ensureContiguousLocations(3);
  adouble x(2.5); // Initial iterate
  adouble x_fix;
  double out;
  trace_on(tapeId);
  adouble u;
  u <<= argument;

  ADOLC::FpIteration::fp_iteration(
      tapeId, sub_tape_id, iteration<double>, iteration<adouble>, norm,
      norm,   // Norm for the termination criterion for the adjoint
      1e-8,   // Termination threshold for fixed-point iteration
      1e-8,   // Termination threshold
      6,      // Maximum number of iterations
      6,      // Maximum number of adjoint iterations
      &x,     // [in] Initial iterate of fixed-point iteration
      &u,     // [in] The parameters: We compute the derivative wrt this
      &x_fix, // [out] Final state of the iteration
      1,      // Size of the vector x_0
      1);     // Number of parameters
  x_fix >>= out;
  trace_off();

  return out;
}

/* Check whether tracing works, and whether the value of the
 * square root function can be recovered from the tape.
 */
BOOST_AUTO_TEST_CASE(NewtonScalarFixedPoint_zos_forward) {
  const auto tapeId = createNewTape();
  const auto sub_tape_id = createNewTape();

  // Compute the square root of 2.0
  const double argument[1] = {2.0};
  double out = traceNewtonForSquareRoot(
      tapeId,       // tape number
      sub_tape_id,  // subtape number
      argument[0]); // Where to evaluate the square root function

  // Did taping really produce the correct value?
  BOOST_TEST(out == std::sqrt(argument[0]), tt::tolerance(tol));

  double value[1];

  zos_forward(tapeId,   // Tape number
              1,        // Number of dependent variables
              1,        // Number of indepdent variables
              0,        // Don't keep anything
              argument, // Where to evaluate the function
              value);   // Function value

  BOOST_TEST(value[0] == sqrt(argument[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(NewtonScalarFixedPoint_fos_forward) {
  // Compute the square root of 2.0
  const auto tapeId = createNewTape();
  const auto sub_tape_id = createNewTape();
  const double argument[1] = {2.0};
  double out = traceNewtonForSquareRoot(tapeId, sub_tape_id, argument[0]);

  // Did taping really produce the correct value?
  BOOST_TEST(out == std::sqrt(argument[0]), tt::tolerance(tol));

  double value[1];
  double derivative[1];

  /* Test first derivative using the scalar forward mode */

  const double tangent[1] = {1.0};

  fos_forward(tapeId,   // Tape number
              1,        // Number of dependent variables
              1,        // Number of independent variables,
              0,        // Don't keep anything
              argument, // Where to evalute the derivative
              tangent,
              value,       // The computed function value
              derivative); // The computed derivative

  double exactDerivative = 1.0 / (2 * sqrt(argument[0]));

  BOOST_TEST(value[0] == out, tt::tolerance(tol));
  BOOST_TEST(derivative[0] == exactDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(FixedPointSecondOrder1DTest)

namespace {
template <typename T> T f(T x, T u) {
  using std::cos;
  return (cos(u) * x) + 1.0;
}
double derivativeFx(double, double u) {
  using std::cos;
  return cos(u);
}
double derivativeFu(double x, double u) {
  using std::sin;
  return -sin(u) * x;
}
double fixedPoint(double u) {
  using std::cos;
  return 1.0 / (1.0 - cos(u));
}
double derivativeFP(double u) {
  using std::cos;
  using std::pow;
  using std::sin;
  return -std::sin(u) / std::pow(1.0 - std::cos(u), 2);
}

double secondDerivFP(double u) {
  using std::cos;
  using std::pow;
  using std::sin;
  return (2.0 * std::pow(std::sin(u), 2) / std::pow(1.0 - std::cos(u), 3)) -
         (std::cos(u) / std::pow(1.0 - std::cos(u), 2));
}
double norm(double *x, int dim) {
  double norm = 0.0;

  for (int i = 0; i < dim; i++) {
    norm += x[i] * x[i];
  }
  return std::sqrt(norm);
}

void tapeFP(short outerTapeId, short innerTapeId, std::span<double, 2> xu,
            int keep = 0) {
  findTape(outerTapeId).ensureContiguousLocations(3);
  trace_on(outerTapeId, keep);
  {
    adouble x = 0.0;
    adouble x_fix;
    adouble u;
    u <<= xu[1];

    auto f_double = [](double *x, double *u, double *x_fix, int, int) {
      x_fix[0] = f(x[0], u[0]);
      return 0;
    };
    auto f_adouble = [](adouble *x, adouble *u, adouble *x_fix, int, int) {
      x_fix[0] = f<adouble>(x[0], u[0]);
      return 0;
    };
    ADOLC::FpIteration::fp_iteration(
        outerTapeId, innerTapeId, f_double, f_adouble, norm,
        norm,   // Norm for the termination criterion for the adjoint
        1e-8,   // Termination threshold for fixed-point iteration
        1e-8,   // Termination threshold
        188,    // Maximum number of iterations
        188,    // Maximum number of adjoint iterations
        &x,     // [in] Initial iterate of fixed-point iteration
        &u,     // [in] The parameters: We compute the derivative wrt this
        &x_fix, // [out] Final state of the iteration
        1,      // Size of the vector x_0
        1);     // Number of parameters

    double out;
    x_fix >>= out;
  }
  trace_off();
}

void tapeSec(short outerTapeId, short innerTapeId, short secoInnerTape,
             std::span<double, 2> xu) {
  findTape(outerTapeId).ensureContiguousLocations(3);
  trace_on(outerTapeId);
  {
    adouble x = 0.0;
    adouble x_fix;
    adouble u;
    u <<= xu[1];

    auto f_double = [](double *x, double *u, double *x_fix, int, int) {
      x_fix[0] = f(x[0], u[0]);
      return 0;
    };
    auto f_adouble = [](adouble *x, adouble *u, adouble *x_fix, int, int) {
      x_fix[0] = f<adouble>(x[0], u[0]);
      return 0;
    };
    ADOLC::FpIteration::FpProblem problem{
        outerTapeId, innerTapeId, secoInnerTape, f_double, f_adouble, norm,
        norm,   // Norm for the termination criterion for the
                // adjoint
        1e-9,   // Termination threshold for fixed-point iteration
        1e-9,   // Termination threshold
        188,    // Maximum number of iterations
        188,    // Maximum number of adjoint iterations
        &x,     // [in] Initial iterate of fixed-point iteration
        &u,     // [in] The parameters: We compute the derivative wrt
                // this
        &x_fix, // [out] Final state of the iteration
        1,      // Size of the vector x_0
        1};
    ADOLC::FpIteration::fp_iteration<ADOLC::FpIteration::FpMode::secondOrder>(
        problem); // Number of parameters

    double out;
    x_fix >>= out;
  }
  trace_off();
}
} // namespace

BOOST_AUTO_TEST_CASE(zos_forward_) {
  ADOLC::FpIteration::resetFpiStack();
  const short outerTapeId = createNewTape();
  const short innerTapeId = createNewTape();
  std::array<double, 2> xu{0.0, 0.5};
  tapeFP(outerTapeId, innerTapeId, xu);
  std::array<double, 1> y{};
  zos_forward(outerTapeId, 1, 1, 0, xu.data() + 1, y.data());
  BOOST_TEST(fixedPoint(xu[1]) == y[0], tt::tolerance(tol));
}
BOOST_AUTO_TEST_CASE(fos_forward_) {
  ADOLC::FpIteration::resetFpiStack();
  const short outerTapeId = createNewTape();
  const short innerTapeId = createNewTape();
  std::array<double, 2> xu{0.0, 0.5};
  tapeFP(outerTapeId, innerTapeId, xu);
  std::array<double, 1> y{};
  std::array<double, 1> tangent{1.0};
  std::array<double, 1> Y{};
  fos_forward(outerTapeId, 1, 1, 0, xu.data() + 1, tangent.data(), y.data(),
              Y.data());
  BOOST_TEST(fixedPoint(xu[1]) == y[0], tt::tolerance(tol));
  BOOST_TEST(derivativeFP(xu[1]) == Y[0], tt::tolerance(tol));
}
BOOST_AUTO_TEST_CASE(fos_reverse_) {
  ADOLC::FpIteration::resetFpiStack();
  const short outerTapeId = createNewTape();
  const short innerTapeId = createNewTape();
  std::array<double, 2> xu{0.0, 0.5};
  tapeFP(outerTapeId, innerTapeId, xu, 1);
  std::array<double, 1> weight{1.0};
  std::array<double, 1> z{};

  fos_reverse(outerTapeId, 1, 1, weight.data(), z.data());
  BOOST_TEST(derivativeFP(xu[1]) == z[0], tt::tolerance(1e-07));
}
BOOST_AUTO_TEST_CASE(hos_reverse_) {
  ADOLC::FpIteration::resetFpiStack();
  const short outerTapeId = createNewTape();
  const short innerTapeId = createNewTape();
  const short secInnerTape = createNewTape();
  std::array<double, 2> xu{0.0, 0.5};
  tapeSec(outerTapeId, innerTapeId, secInnerTape, xu);
  std::array<double, 1> y{};
  std::array<double, 1> tangent{1.0};
  double Y[1];
  fos_forward(outerTapeId, 1, 1, 2, xu.data() + 1, tangent.data(), y.data(), Y);
  BOOST_TEST(fixedPoint(xu[1]) == y[0], tt::tolerance(tol));
  BOOST_TEST(derivativeFP(xu[1]) == Y[0], tt::tolerance(tol));
  std::array<double, 1> weight{1.0};
  double **Z = myalloc2(1, 2);
  hos_reverse(outerTapeId, 1, 1, 1, weight.data(), Z);
  BOOST_TEST(derivativeFP(xu[1]) == Z[0][0], tt::tolerance(tol));
  BOOST_TEST(secondDerivFP(xu[1]) == Z[0][1], tt::tolerance(tol));
  myfree2(Z);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(FixedPointCompositionTest)
namespace {
template <class T> int coupledIteration(T *x, T *u, T *y, int, int) {
  y[0] = 0.5 * x[0] + 0.125 * x[1] + u[0] * u[0];
  y[1] = 0.25 * x[1] + u[1];
  return 0;
}

double coupledNorm(double *x, int n) {
  double result = 0;
  for (int i = 0; i < n; ++i)
    result = std::max(result, std::abs(x[i]));
  return result;
}

int activeIterationCalls = 0;
short traceComposition(bool useFixedPoint, bool activeInitialGuess,
                       bool recordBranch = false) {
  const short tape = createNewTape();
  const short sub = createNewTape();
  const short internal = createNewTape();
  setCurrentTape(tape);
  currentTape().ensureContiguousLocations(6);
  {
    std::array<adouble, 2> u, x, y;
    trace_on(tape);
    u[0] <<= 0.7;
    u[1] <<= 0.4;
    x[0] = activeInitialGuess ? u[0] : adouble(0.0);
    x[1] = activeInitialGuess ? u[1] : adouble(0.0);
    if (useFixedPoint) {
      ADOLC::FpIteration::FpProblem problem{tape,
                                            sub,
                                            internal,
                                            [recordBranch](double *x, double *u,
                                                           double *y, int n, int m) {
                                              coupledIteration(x, u, y, n, m);
                                              if (recordBranch && u[0] > 1.0)
                                                y[0] += u[0];
                                              return 0;
                                            },
                                            [recordBranch](adouble *x, adouble *u,
                                                           adouble *y, int n, int m) {
                                              ++activeIterationCalls;
                                              coupledIteration(x, u, y, n, m);
                                              if (recordBranch && u[0] > 1.0)
                                                y[0] += u[0];
                                              return 0;
                                            },
                                            coupledNorm,
                                            coupledNorm,
                                            1e-13,
                                            1e-13,
                                            200,
                                            200,
                                            x.data(),
                                            u.data(),
                                            y.data(),
                                            2,
                                            2};
      BOOST_REQUIRE(ADOLC::FpIteration::fp_iteration<
                        ADOLC::FpIteration::FpMode::secondOrder>(problem) > 0);
    } else {
      y[0] = 2.0 * u[0] * u[0] + u[1] / 3.0;
      y[1] = 4.0 * u[1] / 3.0;
    }
    // Nonlinear postprocessing produces a nonzero first-order adjoint seed.
    // The direct parameter dependence also exercises adjoint accumulation.
    adouble result = y[0] * y[0] + y[0] * y[1] + u[0] * y[1] + u[1] * u[1];
    double value;
    result >>= value;
    trace_off();
  }
  return tape;
}

void checkComposition(bool activeInitialGuess, bool replay) {
  ADOLC::FpIteration::resetFpiStack();
  const short actualTape = traceComposition(true, activeInitialGuess);
  const short referenceTape = traceComposition(false, activeInitialGuess);
  std::array<double, 2> u = replay ? std::array<double, 2>{1.1, -0.2}
                                   : std::array<double, 2>{0.7, 0.4};
  double **actual = myalloc2(2, 2), **reference = myalloc2(2, 2);
  for (int direction = 0; direction < 2; ++direction) {
    std::array<double, 2> tangent{};
    tangent[direction] = 1.0;
    double value, derivative, referenceValue, referenceDerivative;
    fos_forward(actualTape, 1, 2, 2, u.data(), tangent.data(), &value,
                &derivative);
    fos_forward(referenceTape, 1, 2, 2, u.data(), tangent.data(),
                &referenceValue, &referenceDerivative);
    BOOST_TEST(value == referenceValue, tt::tolerance(1e-10));
    BOOST_TEST(derivative == referenceDerivative, tt::tolerance(1e-10));
    double weight = 1.0;
    hos_reverse(actualTape, 1, 2, 1, &weight, actual);
    hos_reverse(referenceTape, 1, 2, 1, &weight, reference);
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j)
        BOOST_TEST(actual[i][j] == reference[i][j], tt::tolerance(1e-10));
  }
  // A zero-order forward establishes the subtape point for reverse.
  // Use a different point from the preceding tangent sweeps to catch stale
  // retained values; reverse itself must not change the evaluation point.
  u = {0.8, 0.2};
  double value, weight = 1.0;
  std::array<double, 2> gradient{}, referenceGradient{};
  zos_forward(actualTape, 1, 2, 1, u.data(), &value);
  fos_reverse(actualTape, 1, 2, &weight, gradient.data());
  zos_forward(referenceTape, 1, 2, 1, u.data(), &value);
  fos_reverse(referenceTape, 1, 2, &weight, referenceGradient.data());
  for (int i = 0; i < 2; ++i)
    BOOST_TEST(gradient[i] == referenceGradient[i], tt::tolerance(1e-10));
  myfree2(actual);
  myfree2(reference);
}
} // namespace
BOOST_AUTO_TEST_CASE(composed_vector_fixed_point) {
  checkComposition(false, false);
}
BOOST_AUTO_TEST_CASE(active_initial_guess) { checkComposition(true, false); }
BOOST_AUTO_TEST_CASE(replay_at_new_parameters) {
  checkComposition(false, true);
}
BOOST_AUTO_TEST_CASE(replay_with_active_initial_guess) {
  checkComposition(true, true);
}
BOOST_AUTO_TEST_CASE(branch_switch_requires_user_retaping) {
  ADOLC::FpIteration::resetFpiStack();
  activeIterationCalls = 0;
  const short tape = traceComposition(true, false, true);
  const int recordedCalls = activeIterationCalls;
  std::array<double, 2> u{1.1, 0.4}, tangent{1.0, 0.0};
  double value, derivative;
  // The inner tape warns and returns -1. No active callback may be invoked
  // to silently record a replacement tape, including on a repeated replay.
  for (int replay = 0; replay < 2; ++replay) {
    BOOST_TEST(fos_forward(tape, 1, 2, 2, u.data(), tangent.data(), &value,
                           &derivative) == -1);
    BOOST_TEST(activeIterationCalls == recordedCalls);
  }
  BOOST_TEST(zos_forward(tape, 1, 2, 1, u.data(), &value) == -1);
  // Do not reverse after a failed forward sweep.
  BOOST_TEST(activeIterationCalls == recordedCalls);
}
BOOST_AUTO_TEST_SUITE_END()

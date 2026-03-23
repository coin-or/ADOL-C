
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

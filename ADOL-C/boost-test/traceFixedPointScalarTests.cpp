#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

#include <adolc/adolc.h>

#include "const.h"

BOOST_AUTO_TEST_SUITE(trace_fixed_point_scalar)

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
template <typename T>
static int iteration(T *x, T *u, T *y, int dim_x, int dim_u) {
  // Newton update: x = x - f(x)/f'(x) = x - (x*x-z) / 2x = x - x/2 + z/2x
  y[0] = 0.5 * (x[0] + u[0] / x[0]);
  return 0;
}

static double norm(double *x, int dim) {
  double norm = 0;

  for (int i = 0; i < dim; i++)
    norm += x[i] * x[i];

  return std::sqrt(norm);
}

static double traceNewtonForSquareRoot(int tapeId, int sub_tape_id,
                                       double argument) {
  // ax1 = sqrt(ax1);
  setCurrentTape(tapeId);
  currentTape().ensureContiguousLocations(3);
  adouble x(2.5); // Initial iterate
  adouble y;
  double out;
  trace_on(tapeId);
  adouble u;
  u <<= argument;

  fp_iteration(tapeId, sub_tape_id, iteration<double>, iteration<adouble>, norm,
               norm, // Norm for the termination criterion for the adjoint
               1e-8, // Termination threshold for fixed-point iteration
               1e-8, // Termination threshold
               6,    // Maximum number of iterations
               6,    // Maximum number of adjoint iterations
               &x,   // [in] Initial iterate of fixed-point iteration
               &u,   // [in] The parameters: We compute the derivative wrt this
               &y,   // [out] Final state of the iteration
               1,    // Size of the vector x_0
               1);   // Number of parameters
  y >>= out;
  trace_off();

  return out;
}

/* Check whether tracing works, and whether the value of the
 * square root function can be recovered from the tape.
 */
BOOST_AUTO_TEST_CASE(NewtonScalarFixedPoint_zos_forward) {
  const short tapeId = 8;
  const short sub_tape_id = 9;

  createNewTape(tapeId);
  createNewTape(sub_tape_id);

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
  const short tapeId = 1;
  const double argument[1] = {2.0};
  double out = traceNewtonForSquareRoot(tapeId, 2, argument[0]);

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

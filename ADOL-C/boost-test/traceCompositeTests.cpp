#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

#include <adolc/adolc.h>

#include "const.h"

BOOST_AUTO_TEST_SUITE(trace_composite)

/************************************/
/* Tests for trace vector mode with */
/* custom composite functions       */
/* Author: Philipp Schuette         */
/************************************/

/* The tests for ADOL-C trace vector mode in this file involve more
 * complicated custom compositions of elementary functions with
 * higher numbers of independent and possibly dependent variables.
 * The case of multiple independents is especially interesting in
 * forward mode, while the case of multiple dependents is more
 * interesting in reverse mode.
 *
 * Before every test, a short comment explains the structure of the
 * tested composite function and states the expected analytic
 * derivative.
 */
const short tapeId4 = 4;
struct TapeInitializer {
  TapeInitializer() { createNewTape(tapeId4); }
};

BOOST_GLOBAL_FIXTURE(TapeInitializer);
/* Tested function: sin(x1)*sin(x1) + cos(x1)*cos(x1) + x2
 * Gradient vector: (
 *                    0.0,
 *                    1.0
 *                  )
 */
BOOST_AUTO_TEST_CASE(CompositeTrig1_FOV_Forward) {
  double x1 = 0.289, x2 = 1.927, out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;

  trace_on(tapeId4);
  ax1 <<= x1;
  ax2 <<= x2;

  ax1 = sin(ax1) * sin(ax1) + cos(ax1) * cos(ax1) + ax2;

  ax1 >>= out;
  trace_off();

  double x1Derivative = 0.;
  double x2Derivative = 1.;
  x1 = std::sin(x1) * std::sin(x1) + std::cos(x1) * std::cos(x1) + x2;

  double *x = myalloc1(2);
  double **xd = myalloc2(2, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  /* Test partial derivative wrt x1 and x2. */
  x[0] = 0.289;
  x[1] = 1.927;

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      if (i == j)
        xd[i][j] = 1.;
      else
        xd[i][j] = 0.;
    }
  }

  fov_forward(tapeId4, 1, 2, 2, x, xd, y, yd);

  BOOST_TEST(*y == x1, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == x2Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(CompositeTrig1Operator_FOV_Reverse) {
  double x1 = 0.289, x2 = 1.927, out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;

  trace_on(tapeId4, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ax1 = sin(ax1) * sin(ax1) + cos(ax1) * cos(ax1) + ax2;

  ax1 >>= out;
  trace_off();

  double x1Derivative = 0.;
  double x2Derivative = 1.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 2);

  u[0][0] = 1.;
  u[1][0] = std::sqrt(2.);

  fov_reverse(tapeId4, 1, 2, 2, u, z);

  BOOST_TEST(z[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == std::sqrt(2.) * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == std::sqrt(2.) * x2Derivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

/* Tested function: 2*sin(cos(x1))*exp(x2) - pow(cos(x3), 2)*sin(x2)
 * Gradient vector: (
 *                    -2*cos(cos(x1))*exp(x2)*sin(x1),
 *                    2*sin(cos(x1))*exp(x2) - pow(cos(x3), 2)*cos(x2),
 *                    2*cos(x3)*sin(x3)*sin(x2)
 *                  )
 */
BOOST_AUTO_TEST_CASE(CompositeTrig2_FOV_Forward) {
  double x1 = 1.11, x2 = 2.22, x3 = 3.33, out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;

  trace_on(tapeId4);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ax1 = 2 * sin(cos(ax1)) * exp(ax2) - pow(cos(ax3), 2) * sin(ax2);

  ax1 >>= out;
  trace_off();

  double x1Derivative =
      -2 * std::cos(std::cos(x1)) * std::exp(x2) * std::sin(x1);
  double x2Derivative = 2 * std::sin(std::cos(x1)) * std::exp(x2) -
                        std::pow(std::cos(x3), 2) * std::cos(x2);
  double x3Derivative = 2 * std::cos(x3) * std::sin(x3) * std::sin(x2);
  x1 = 2 * std::sin(std::cos(x1)) * std::exp(x2) -
       std::pow(std::cos(x3), 2) * std::sin(x2);

  double *x = myalloc1(3);
  double **xd = myalloc2(3, 3);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 3);

  /* Test partial derivative wrt x1, x2 and x3. */
  x[0] = 1.11;
  x[1] = 2.22;
  x[2] = 3.33;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (i == j)
        xd[i][j] = 1.;
      else
        xd[i][j] = 0.;
    }
  }

  fov_forward(tapeId4, 1, 3, 3, x, xd, y, yd);

  BOOST_TEST(*y == x1, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][2] == x3Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(CompositeTrig2Operator_FOV_Reverse) {
  double x1 = 1.11, x2 = 2.22, x3 = 3.33, out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;

  trace_on(tapeId4, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ax1 = 2 * sin(cos(ax1)) * exp(ax2) - pow(cos(ax3), 2) * sin(ax2);

  ax1 >>= out;
  trace_off();

  double x1Derivative =
      -2 * std::cos(std::cos(x1)) * std::exp(x2) * std::sin(x1);
  double x2Derivative = 2 * std::sin(std::cos(x1)) * std::exp(x2) -
                        std::pow(std::cos(x3), 2) * std::cos(x2);
  double x3Derivative = 2 * std::cos(x3) * std::sin(x3) * std::sin(x2);

  double **u = myalloc2(3, 1);
  double **z = myalloc2(3, 3);

  u[0][0] = 1.;
  u[1][0] = std::exp(6.);
  u[2][0] = std::log(6.);

  fov_reverse(tapeId4, 1, 3, 3, u, z);

  BOOST_TEST(z[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][2] == x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == std::exp(6.) * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == std::exp(6.) * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][2] == std::exp(6.) * x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][0] == std::log(6.) * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][1] == std::log(6.) * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][2] == std::log(6.) * x3Derivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

/* Tested function: pow(sin(x1), cos(x1) - x2)*x3
 * Gradient vector: (
 *                    pow(sin(x1), cos(x1) - x2)*x3*(-sin(x1)*log(sin(x1))
 *                    + (cos(x1) - x2)*cos(x1)/sin(x1)),
 *                    -log(sin(x1))*pow(sin(x1), cos(x1) - x2)*x3,
 *                    pow(sin(x1), cos(x1) - x2)
 *                  )
 */
BOOST_AUTO_TEST_CASE(CompositeTrig3_FOV_Forward) {
  double x1 = 0.516, x2 = 9.89, x3 = 0.072, out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;

  trace_on(tapeId4);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ax1 = pow(sin(ax1), cos(ax1) - ax2) * ax3;

  ax1 >>= out;
  trace_off();

  double x1Derivative = std::pow(std::sin(x1), std::cos(x1) - x2) * x3 *
                        (-std::sin(x1) * std::log(std::sin(x1)) +
                         (std::cos(x1) - x2) * std::cos(x1) / std::sin(x1));
  double x2Derivative =
      -std::log(std::sin(x1)) * std::pow(std::sin(x1), std::cos(x1) - x2) * x3;
  double x3Derivative = std::pow(std::sin(x1), std::cos(x1) - x2);
  x1 = std::pow(std::sin(x1), std::cos(x1) - x2) * x3;

  double *x = myalloc1(3);
  double **xd = myalloc2(3, 3);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 3);

  /* Test partial derivative wrt x1, x2 and x3. */
  x[0] = 0.516;
  x[1] = 9.89;
  x[2] = 0.072;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (i == j)
        xd[i][j] = 1.;
      else
        xd[i][j] = 0.;
    }
  }

  fov_forward(tapeId4, 1, 3, 3, x, xd, y, yd);

  BOOST_TEST(*y == x1, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][2] == x3Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(CompositeTrig3Operator_FOV_Reverse) {
  double x1 = 0.516, x2 = 9.89, x3 = 0.072, out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;

  trace_on(tapeId4, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ax1 = pow(sin(ax1), cos(ax1) - ax2) * ax3;

  ax1 >>= out;
  trace_off();

  double x1Derivative = std::pow(std::sin(x1), std::cos(x1) - x2) * x3 *
                        (-std::sin(x1) * std::log(std::sin(x1)) +
                         (std::cos(x1) - x2) * std::cos(x1) / std::sin(x1));
  double x2Derivative =
      -std::log(std::sin(x1)) * std::pow(std::sin(x1), std::cos(x1) - x2) * x3;
  double x3Derivative = std::pow(std::sin(x1), std::cos(x1) - x2);

  double **u = myalloc2(3, 1);
  double **z = myalloc2(3, 3);

  u[0][0] = 1.;
  u[1][0] = std::pow(10., 6.);
  u[2][0] = std::pow(6., 10.);

  fov_reverse(tapeId4, 1, 3, 3, u, z);

  BOOST_TEST(z[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][2] == x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == std::pow(10., 6.) * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == std::pow(10., 6.) * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][2] == std::pow(10., 6.) * x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][0] == std::pow(6., 10.) * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][1] == std::pow(6., 10.) * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][2] == std::pow(6., 10.) * x3Derivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

/* Tested function: atan(tan(x1))*exp(x2)
 * Gradient vector: (
 *                    exp(x2)
 *                    x1*exp(x2)
 *                  )
 */
BOOST_AUTO_TEST_CASE(CompositeTrig4_FOV_Forward) {
  double x1 = 1.56, x2 = 8.99, out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;

  trace_on(tapeId4);
  ax1 <<= x1;
  ax2 <<= x2;

  ax1 = atan(tan(ax1)) * exp(ax2);

  ax1 >>= out;
  trace_off();

  double x1Derivative = std::exp(x2);
  double x2Derivative = x1 * std::exp(x2);
  x1 = x1 * exp(x2);

  double *x = myalloc1(3);
  double **xd = myalloc2(3, 3);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 3);

  /* Test partial derivative wrt x1, x2 and x3. */
  x[0] = 1.56;
  x[1] = 8.99;

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      if (i == j)
        xd[i][j] = 1.;
      else
        xd[i][j] = 0.;
    }
  }

  fov_forward(tapeId4, 1, 2, 2, x, xd, y, yd);

  BOOST_TEST(*y == x1, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == x2Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(CompositeTrig4Operator_FOV_Reverse) {
  double x1 = 1.56, x2 = 8.99, out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;

  trace_on(tapeId4, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ax1 = atan(tan(ax1)) * exp(ax2);

  ax1 >>= out;
  trace_off();

  double x1Derivative = std::exp(x2);
  double x2Derivative = x1 * std::exp(x2);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 2);

  u[0][0] = 1.;
  u[1][0] = -1.;

  fov_reverse(tapeId4, 1, 2, 2, u, z);

  BOOST_TEST(z[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == -1. * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == -1. * x2Derivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

/* Tested function: x1 + x2 - x3 + pow(x1, 2) - 10 + sqrt(x4*x5)
 * Gradient vector: (
 *                    1.0 + 2*x1,
 *                    1.0,
 *                    -1.0,
 *                    0.5 * sqrt(x5/x4),
 *                    0.5 * sqrt(x4/x5)
 *                  )
 */
BOOST_AUTO_TEST_CASE(LongSum_FOV_Forward) {
  double x1 = 0.11, x2 = -2.27, x3 = 81.7, x4 = 0.444, x5 = 4.444, out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ax4;
  adouble ax5;

  trace_on(tapeId4);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;
  ax4 <<= x4;
  ax5 <<= x5;

  ax1 = ax1 + ax2 - ax3 + pow(ax1, 2) - 10 + sqrt(ax4 * ax5);

  ax1 >>= out;
  trace_off();

  double x1Derivative = 1. + 2 * x1;
  double x2Derivative = 1.;
  double x3Derivative = -1.;
  double x4Derivative = 0.5 * std::sqrt(x5 / x4);
  double x5Derivative = 0.5 * std::sqrt(x4 / x5);
  x1 = x1 + x2 - x3 + std::pow(x1, 2) - 10 + std::sqrt(x4 * x5);

  double *x = myalloc1(5);
  double **xd = myalloc2(5, 5);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 5);

  /* Test partial derivative wrt x1, x2, x3, x4 and x5. */
  x[0] = 0.11;
  x[1] = -2.27;
  x[2] = 81.7;
  x[3] = 0.444;
  x[4] = 4.444;

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      if (i == j)
        xd[i][j] = 1.;
      else
        xd[i][j] = 0.;
    }
  }

  fov_forward(tapeId4, 1, 5, 5, x, xd, y, yd);

  BOOST_TEST(*y == x1, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][2] == x3Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][3] == x4Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][4] == x5Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(LongSumOperator_FOV_Reverse) {
  double x1 = 0.11, x2 = -2.27, x3 = 81.7, x4 = 0.444, x5 = 4.444, out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ax4;
  adouble ax5;

  trace_on(tapeId4, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;
  ax4 <<= x4;
  ax5 <<= x5;

  ax1 = ax1 + ax2 - ax3 + pow(ax1, 2) - 10 + sqrt(ax4 * ax5);

  ax1 >>= out;
  trace_off();

  double x1Derivative = 1. + 2 * x1;
  double x2Derivative = 1.;
  double x3Derivative = -1.;
  double x4Derivative = 0.5 * std::sqrt(x5 / x4);
  double x5Derivative = 0.5 * std::sqrt(x4 / x5);

  double **u = myalloc2(5, 1);
  double **z = myalloc2(5, 5);

  u[0][0] = 1.;
  u[1][0] = 2.;
  u[2][0] = 3.;
  u[3][0] = 4.;
  u[4][0] = 5.;

  fov_reverse(tapeId4, 1, 5, 5, u, z);

  BOOST_TEST(z[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][2] == x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][3] == x4Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][4] == x5Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == 2. * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == 2. * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][2] == 2. * x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][3] == 2. * x4Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][4] == 2. * x5Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][0] == 3. * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][1] == 3. * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][2] == 3. * x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][3] == 3. * x4Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][4] == 3. * x5Derivative, tt::tolerance(tol));
  BOOST_TEST(z[3][0] == 4. * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[3][1] == 4. * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[3][2] == 4. * x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[3][3] == 4. * x4Derivative, tt::tolerance(tol));
  BOOST_TEST(z[3][4] == 4. * x5Derivative, tt::tolerance(tol));
  BOOST_TEST(z[4][0] == 5. * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[4][1] == 5. * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[4][2] == 5. * x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[4][3] == 5. * x4Derivative, tt::tolerance(tol));
  BOOST_TEST(z[4][4] == 5. * x5Derivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

/* Tested function: sqrt(pow(x1, 2))*x2
 * Gradient vector: (
 *                    x2,
 *                    x1
 *                  )
 */
BOOST_AUTO_TEST_CASE(InverseFunc_FOV_Forward) {
  double x1 = 3.77, x2 = -21.12, out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;

  trace_on(tapeId4);
  ax1 <<= x1;
  ax2 <<= x2;

  ax1 = sqrt(pow(ax1, 2)) * ax2;

  ax1 >>= out;
  trace_off();

  double x1Derivative = x2;
  double x2Derivative = x1;
  x1 = std::sqrt(std::pow(x1, 2)) * x2;

  double *x = myalloc1(2);
  double **xd = myalloc2(2, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  /* Test partial derivative wrt x1 and x2. */
  x[0] = 3.77;
  x[1] = -21.12;

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      if (i == j)
        xd[i][j] = 1.;
      else
        xd[i][j] = 0.;
    }
  }

  fov_forward(tapeId4, 1, 2, 2, x, xd, y, yd);

  BOOST_TEST(*y == x1, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == x2Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(InverseFuncOperator_FOV_Reverse) {
  double x1 = 3.77, x2 = -21.12, out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;

  trace_on(tapeId4, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ax1 = sqrt(pow(ax1, 2)) * ax2;

  ax1 >>= out;
  trace_off();

  double x1Derivative = x2;
  double x2Derivative = x1;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 2);

  u[0][0] = 1.;
  u[1][0] = std::cos(2.);

  fov_reverse(tapeId4, 1, 2, 2, u, z);

  BOOST_TEST(z[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == std::cos(2.) * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == std::cos(2.) * x2Derivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

/* Tested function: exp(x1 + exp(x2 + x3))*pow(x1 + x2, x3)
 * Gradient vector: (
 *                    exp(x1 + exp(x2 + x3))*pow(x1 + x2, x3)
 *                    + exp(x1 + exp(x2 + x3))*x3*pow(x1 + x2, x3 - 1),
 *                    exp(x1 + exp(x2 + x3))*exp(x2 + x3)*pow(x1 + x2, x3)
 *                    + exp(x1 + exp(x2 + x3))*x3*pow(x1 + x2, x3 - 1),
 *                    exp(x1 + exp(x2 + x3))*exp(x2 + x3)*pow(x1 + x2, x3)
 *                    + exp(x1 + exp(x2 + x3))*pow(x1 + x2, x3)*log(x1 + x2)
 *                  )
 */
BOOST_AUTO_TEST_CASE(ExpPow_FOV_Forward) {
  double x1 = 0.642, x2 = 6.42, x3 = 0.528, out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;

  trace_on(tapeId4);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ax1 = exp(ax1 + exp(ax2 + ax3)) * pow(ax1 + ax2, ax3);

  ax1 >>= out;
  trace_off();

  double x1Derivative =
      std::exp(x1 + std::exp(x2 + x3)) * std::pow(x1 + x2, x3) +
      std::exp(x1 + std::exp(x2 + x3)) * x3 * std::pow(x1 + x2, x3 - 1);
  double x2Derivative =
      std::exp(x1 + std::exp(x2 + x3)) * std::exp(x2 + x3) *
          std::pow(x1 + x2, x3) +
      std::exp(x1 + std::exp(x2 + x3)) * x3 * std::pow(x1 + x2, x3 - 1);
  double x3Derivative = std::exp(x1 + std::exp(x2 + x3)) * std::exp(x2 + x3) *
                            std::pow(x1 + x2, x3) +
                        std::exp(x1 + std::exp(x2 + x3)) *
                            std::pow(x1 + x2, x3) * std::log(x1 + x2);
  x1 = std::exp(x1 + std::exp(x2 + x3)) * std::pow(x1 + x2, x3);

  double *x = myalloc1(3);
  double **xd = myalloc2(3, 3);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 3);

  /* Test partial derivative wrt x1, x2 and x3. */
  x[0] = 0.642;
  x[1] = 6.42;
  x[2] = 0.528;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (i == j)
        xd[i][j] = 1.;
      else
        xd[i][j] = 0.;
    }
  }

  fov_forward(tapeId4, 1, 3, 3, x, xd, y, yd);

  BOOST_TEST(*y == x1, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][2] == x3Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(ExpPowOperator_FOV_Reverse) {
  double x1 = 1., x2 = 2., x3 = 3., out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;

  trace_on(tapeId4, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ax1 = exp(ax1 + exp(ax2 + ax3)) * pow(ax1 + ax2, ax3);

  ax1 >>= out;
  trace_off();

  double x1Derivative =
      std::exp(x1 + std::exp(x2 + x3)) * std::pow(x1 + x2, x3) +
      std::exp(x1 + std::exp(x2 + x3)) * x3 * std::pow(x1 + x2, x3 - 1);
  double x2Derivative =
      std::exp(x1 + std::exp(x2 + x3)) * std::exp(x2 + x3) *
          std::pow(x1 + x2, x3) +
      std::exp(x1 + std::exp(x2 + x3)) * x3 * std::pow(x1 + x2, x3 - 1);
  double x3Derivative = std::exp(x1 + std::exp(x2 + x3)) * std::exp(x2 + x3) *
                            std::pow(x1 + x2, x3) +
                        std::exp(x1 + std::exp(x2 + x3)) *
                            std::pow(x1 + x2, x3) * std::log(x1 + x2);

  double **u = myalloc2(3, 1);
  double **z = myalloc2(3, 3);

  u[0][0] = 1.;
  u[1][0] = -1.;
  u[2][0] = -2.;

  fov_reverse(tapeId4, 1, 3, 3, u, z);

  BOOST_TEST(z[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][2] == x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == -1. * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == -1. * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][2] == -1. * x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][0] == -2. * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][1] == -2. * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][2] == -2. * x3Derivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

/* Tested function: sqrt(sqrt(x1*x2 + 2*x3))*x4
 * Gradient vector: (
 *                    0.25*pow(x1*x2 + 2*x3, -0.75)*x2*x4,
 *                    0.25*pow(x1*x2 + 2*x3, -0.75)*x1*x4,
 *                    0.25*pow(x1*x2 + 2*x3, -0.75)*2*x4,
 *                    pow(x1*x2 + 2*x3, 0.25)
 *                  )
 */
BOOST_AUTO_TEST_CASE(CompositeSqrt_FOV_Forward) {
  double x1 = -2.14, x2 = -2.22, x3 = 50.05, x4 = 0.104, out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ax4;

  trace_on(tapeId4);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;
  ax4 <<= x4;

  ax1 = sqrt(sqrt(ax1 * ax2 + 2 * ax3)) * ax4;

  ax1 >>= out;
  trace_off();

  double x1Derivative = 0.25 * std::pow(x1 * x2 + 2 * x3, -0.75) * x2 * x4;
  double x2Derivative = 0.25 * std::pow(x1 * x2 + 2 * x3, -0.75) * x1 * x4;
  double x3Derivative = 0.25 * std::pow(x1 * x2 + 2 * x3, -0.75) * 2.0 * x4;
  double x4Derivative = std::pow(x1 * x2 + 2 * x3, 0.25);
  x1 = std::sqrt(std::sqrt(x1 * x2 + 2 * x3)) * x4;

  double *x = myalloc1(4);
  double **xd = myalloc2(4, 4);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 4);

  /* Test partial derivative wrt x1, x2, x3 and x4. */
  x[0] = -2.14;
  x[1] = -2.22;
  x[2] = 50.05;
  x[3] = 0.104;

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      if (i == j)
        xd[i][j] = 1.;
      else
        xd[i][j] = 0.;
    }
  }

  fov_forward(tapeId4, 1, 4, 4, x, xd, y, yd);

  BOOST_TEST(*y == x1, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][2] == x3Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][3] == x4Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(CompositeSqrtOperator_FOV_Reverse) {
  double x1 = -2.14, x2 = -2.22, x3 = 50.05, x4 = 0.104, out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ax4;

  trace_on(tapeId4, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;
  ax4 <<= x4;

  ax1 = sqrt(sqrt(ax1 * ax2 + 2 * ax3)) * ax4;

  ax1 >>= out;
  trace_off();

  double x1Derivative = 0.25 * std::pow(x1 * x2 + 2 * x3, -0.75) * x2 * x4;
  double x2Derivative = 0.25 * std::pow(x1 * x2 + 2 * x3, -0.75) * x1 * x4;
  double x3Derivative = 0.25 * std::pow(x1 * x2 + 2 * x3, -0.75) * 2.0 * x4;
  double x4Derivative = std::pow(x1 * x2 + 2 * x3, 0.25);

  double **u = myalloc2(4, 1);
  double **z = myalloc2(4, 4);

  u[0][0] = 1.;
  u[1][0] = -1.;
  u[2][0] = 2.;
  u[3][0] = -2.;

  fov_reverse(tapeId4, 1, 4, 4, u, z);

  BOOST_TEST(z[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][2] == x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][3] == x4Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == -1. * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == -1. * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][2] == -1. * x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][3] == -1. * x4Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][0] == 2. * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][1] == 2. * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][2] == 2. * x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][3] == 2. * x4Derivative, tt::tolerance(tol));
  BOOST_TEST(z[3][0] == -2. * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[3][1] == -2. * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[3][2] == -2. * x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[3][3] == -2. * x4Derivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

/* Tested function: tanh(acos(pow(x1, 2) + 1)*sin(x2))*x3 + exp(cosh(x4))
 * Gradient vector: (
 *                    (1 - pow(tanh(acos(pow(x1, 2) + 1)*sin(x2)), 2))
 *                    * x3 * sin(x2) * 2. * x1 / sqrt(1. - pow(x1, 4)),
 *                    (1 - pow(tanh(acos(pow(x1, 2) + 1)*sin(x2)), 2))
 *                    * x3 * acos(pow(x1, 2) + 1) * cos(x2),
 *                    tanh(acos(pow(x1, 2) + 1)*sin(x2)),
 *                    exp(cosh(x4)) * sinh(x4)
 *                  )
 */
BOOST_AUTO_TEST_CASE(CompositeHyperbOperator_FOV_Forward) {
  double x1 = 0.1, x2 = 5.099, x3 = 5.5, x4 = 4.73, out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ax4;

  trace_on(tapeId4);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;
  ax4 <<= x4;

  ax1 = tanh(acos(pow(ax1, 2) + 0.5) * sin(ax2)) * ax3 + exp(cosh(ax4));

  ax1 >>= out;
  trace_off();

  double x1Derivative =
      -(1 - std::pow(std::tanh(std::acos(std::pow(x1, 2) + 0.5) * std::sin(x2)),
                     2)) *
      x3 * std::sin(x2) * 2. * x1 /
      (std::sqrt(1. - std::pow(std::pow(x1, 2) + 0.5, 2)));
  double x2Derivative =
      (1 - std::pow(std::tanh(std::acos(std::pow(x1, 2) + 0.5) * std::sin(x2)),
                    2)) *
      x3 * std::acos(std::pow(x1, 2) + 0.5) * std::cos(x2);
  double x3Derivative =
      std::tanh(std::acos(std::pow(x1, 2) + 0.5) * std::sin(x2));
  double x4Derivative = std::exp(std::cosh(x4)) * std::sinh(x4);
  x1 = std::tanh(std::acos(std::pow(x1, 2) + 0.5) * std::sin(x2)) * x3 +
       std::exp(std::cosh(x4));

  double *x = myalloc1(4);
  double **xd = myalloc2(4, 4);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 4);

  /* Test partial derivative wrt x1, x2, x3 and x4. */
  x[0] = 0.1;
  x[1] = 5.099;
  x[2] = 5.5;
  x[3] = 4.73;

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      if (i == j)
        xd[i][j] = 1.;
      else
        xd[i][j] = 0.;
    }
  }

  fov_forward(tapeId4, 1, 4, 4, x, xd, y, yd);

  BOOST_TEST(*y == x1, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][2] == x3Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][3] == x4Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(CompositeHyperbOperator_FOV_Reverse) {
  double x1 = 0.1, x2 = 5.099, x3 = 5.5, x4 = 4.73, out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ax4;

  trace_on(tapeId4, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;
  ax4 <<= x4;

  ax1 = tanh(acos(pow(ax1, 2) + 0.5) * sin(ax2)) * ax3 + exp(cosh(ax4));

  ax1 >>= out;
  trace_off();

  double x1Derivative =
      -(1 - std::pow(std::tanh(std::acos(std::pow(x1, 2) + 0.5) * std::sin(x2)),
                     2)) *
      x3 * std::sin(x2) * 2. * x1 /
      (std::sqrt(1. - std::pow(std::pow(x1, 2) + 0.5, 2)));
  double x2Derivative =
      (1 - std::pow(std::tanh(std::acos(std::pow(x1, 2) + 0.5) * std::sin(x2)),
                    2)) *
      x3 * std::acos(std::pow(x1, 2) + 0.5) * std::cos(x2);
  double x3Derivative =
      std::tanh(std::acos(std::pow(x1, 2) + 0.5) * std::sin(x2));
  double x4Derivative = std::exp(std::cosh(x4)) * std::sinh(x4);

  double **u = myalloc2(4, 1);
  double **z = myalloc2(4, 4);

  u[0][0] = 1.;
  u[1][0] = std::exp(1.);
  u[2][0] = -2.;
  u[3][0] = std::exp(2.);

  fov_reverse(tapeId4, 1, 4, 4, u, z);

  BOOST_TEST(z[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][2] == x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][3] == x4Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == std::exp(1.) * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == std::exp(1.) * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][2] == std::exp(1.) * x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][3] == std::exp(1.) * x4Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][0] == -2. * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][1] == -2. * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][2] == -2. * x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][3] == -2. * x4Derivative, tt::tolerance(tol));
  BOOST_TEST(z[3][0] == std::exp(2.) * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[3][1] == std::exp(2.) * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[3][2] == std::exp(2.) * x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[3][3] == std::exp(2.) * x4Derivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

/* Tested function: fmax(x1*pow(x3, 2), x2*pow(x3, 2))*exp(x3)
 * Gradient vector: (
 *                    pow(x3, 2)*exp(x3),
 *                    0.0,
 *                    2.0*x1*x3*exp(x3) + x1*pow(x3, 2)*exp(x3)
 *                  )
 */
BOOST_AUTO_TEST_CASE(CompositeFmaxOperator_FOV_Forward) {
  double x1 = 2.31, x2 = 1.32, x3 = 3.21, out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;

  trace_on(tapeId4);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ax1 = fmax(ax1 * pow(ax3, 2), ax2 * pow(ax3, 2)) * exp(ax3);

  ax1 >>= out;
  trace_off();

  double x1Derivative = std::pow(x3, 2) * std::exp(x3);
  double x2Derivative = 0.0;
  double x3Derivative =
      2.0 * x1 * x3 * std::exp(x3) + x1 * std::pow(x3, 2) * std::exp(x3);
  x1 = std::fmax(x1 * std::pow(x3, 2), x2 * std::pow(x3, 2)) * std::exp(x3);

  double *x = myalloc1(3);
  double **xd = myalloc2(3, 3);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 3);

  /* Test partial derivative wrt x1, x2, and x3. */
  x[0] = 2.31;
  x[1] = 1.32;
  x[2] = 3.21;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (i == j)
        xd[i][j] = 1.;
      else
        xd[i][j] = 0.;
    }
  }

  fov_forward(tapeId4, 1, 3, 3, x, xd, y, yd);

  BOOST_TEST(*y == x1, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][2] == x3Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

/* Tested function: max(x1*pow(x3, 2), x2*pow(x3, 2))*exp(x3)
 * Gradient vector: (
 *                    pow(x3, 2)*exp(x3),
 *                    0.0,
 *                    2.0*x1*x3*exp(x3) + x1*pow(x3, 2)*exp(x3)
 *                  )
 */
BOOST_AUTO_TEST_CASE(CompositeMaxOperator_FOV_Forward) {
  double x1 = 2.31, x2 = 1.32, x3 = 3.21, out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;

  trace_on(tapeId4);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ax1 = max(ax1 * pow(ax3, 2), ax2 * pow(ax3, 2)) * exp(ax3);

  ax1 >>= out;
  trace_off();

  double x1Derivative = std::pow(x3, 2) * std::exp(x3);
  double x2Derivative = 0.0;
  double x3Derivative =
      2.0 * x1 * x3 * std::exp(x3) + x1 * std::pow(x3, 2) * std::exp(x3);
  x1 = std::max(x1 * std::pow(x3, 2), x2 * std::pow(x3, 2)) * std::exp(x3);

  double *x = myalloc1(3);
  double **xd = myalloc2(3, 3);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 3);

  /* Test partial derivative wrt x1, x2, and x3. */
  x[0] = 2.31;
  x[1] = 1.32;
  x[2] = 3.21;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (i == j)
        xd[i][j] = 1.;
      else
        xd[i][j] = 0.;
    }
  }

  fov_forward(tapeId4, 1, 3, 3, x, xd, y, yd);

  BOOST_TEST(*y == x1, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][2] == x3Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(CompositeFmaxOperator_FOV_Reverse) {
  double x1 = 2.31, x2 = 1.32, x3 = 3.21, out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;

  trace_on(tapeId4, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ax1 = fmax(ax1 * pow(ax3, 2), ax2 * pow(ax3, 2)) * exp(ax3);

  ax1 >>= out;
  trace_off();

  double x1Derivative = std::pow(x3, 2) * std::exp(x3);
  double x2Derivative = 0.0;
  double x3Derivative =
      2.0 * x1 * x3 * std::exp(x3) + x1 * std::pow(x3, 2) * std::exp(x3);

  double **u = myalloc2(3, 1);
  double **z = myalloc2(3, 3);

  u[0][0] = 1.;
  u[1][0] = std::sqrt(5.);
  u[2][0] = -std::sqrt(10.);

  fov_reverse(tapeId4, 1, 3, 3, u, z);

  BOOST_TEST(z[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][2] == x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == std::sqrt(5.) * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == std::sqrt(5.) * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][2] == std::sqrt(5.) * x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][0] == -std::sqrt(10.) * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][1] == -std::sqrt(10.) * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][2] == -std::sqrt(10.) * x3Derivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(CompositeMaxOperator_FOV_Reverse) {
  double x1 = 2.31, x2 = 1.32, x3 = 3.21, out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;

  trace_on(tapeId4, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ax1 = max(ax1 * pow(ax3, 2), ax2 * pow(ax3, 2)) * exp(ax3);

  ax1 >>= out;
  trace_off();

  double x1Derivative = std::pow(x3, 2) * std::exp(x3);
  double x2Derivative = 0.0;
  double x3Derivative =
      2.0 * x1 * x3 * std::exp(x3) + x1 * std::pow(x3, 2) * std::exp(x3);

  double **u = myalloc2(3, 1);
  double **z = myalloc2(3, 3);

  u[0][0] = 1.;
  u[1][0] = std::sqrt(5.);
  u[2][0] = -std::sqrt(10.);

  fov_reverse(tapeId4, 1, 3, 3, u, z);

  BOOST_TEST(z[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][2] == x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == std::sqrt(5.) * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == std::sqrt(5.) * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][2] == std::sqrt(5.) * x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][0] == -std::sqrt(10.) * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][1] == -std::sqrt(10.) * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][2] == -std::sqrt(10.) * x3Derivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

/* Tested function: fmin(x1*pow(x3, 2), x2*pow(x3, 2))*exp(x3)
 * Gradient vector: (
 *                    0.0,
 *                    pow(x3, 2)*exp(x3),
 *                    2.0*x2*x3*exp(x3) + x2*pow(x3, 2)*exp(x3)
 *                  )
 */
BOOST_AUTO_TEST_CASE(CompositeFminOperator_FOV_Forward) {
  double x1 = 2.31, x2 = 1.32, x3 = 3.21, out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;

  trace_on(tapeId4);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ax1 = fmin(ax1 * pow(ax3, 2), ax2 * pow(ax3, 2)) * exp(ax3);

  ax1 >>= out;
  trace_off();

  double x1Derivative = 0.0;
  double x2Derivative = std::pow(x3, 2) * std::exp(x3);
  double x3Derivative =
      2.0 * x2 * x3 * std::exp(x3) + x2 * std::pow(x3, 2) * std::exp(x3);
  x1 = std::fmin(x1 * std::pow(x3, 2), x2 * std::pow(x3, 2)) * std::exp(x3);

  double *x = myalloc1(3);
  double **xd = myalloc2(3, 3);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 3);

  /* Test partial derivative wrt x1, x2, and x3. */
  x[0] = 2.31;
  x[1] = 1.32;
  x[2] = 3.21;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (i == j)
        xd[i][j] = 1.;
      else
        xd[i][j] = 0.;
    }
  }

  fov_forward(tapeId4, 1, 3, 3, x, xd, y, yd);

  BOOST_TEST(*y == x1, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][2] == x3Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

/* Tested function: fmin(x1*pow(x3, 2), x2*pow(x3, 2))*exp(x3)
 * Gradient vector: (
 *                    0.0,
 *                    pow(x3, 2)*exp(x3),
 *                    2.0*x2*x3*exp(x3) + x2*pow(x3, 2)*exp(x3)
 *                  )
 */
BOOST_AUTO_TEST_CASE(CompositeMinOperator_FOV_Forward) {
  double x1 = 2.31, x2 = 1.32, x3 = 3.21, out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;

  trace_on(tapeId4);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ax1 = min(ax1 * pow(ax3, 2), ax2 * pow(ax3, 2)) * exp(ax3);

  ax1 >>= out;
  trace_off();

  double x1Derivative = 0.0;
  double x2Derivative = std::pow(x3, 2) * std::exp(x3);
  double x3Derivative =
      2.0 * x2 * x3 * std::exp(x3) + x2 * std::pow(x3, 2) * std::exp(x3);
  x1 = std::min(x1 * std::pow(x3, 2), x2 * std::pow(x3, 2)) * std::exp(x3);

  double *x = myalloc1(3);
  double **xd = myalloc2(3, 3);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 3);

  /* Test partial derivative wrt x1, x2, and x3. */
  x[0] = 2.31;
  x[1] = 1.32;
  x[2] = 3.21;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (i == j)
        xd[i][j] = 1.;
      else
        xd[i][j] = 0.;
    }
  }

  fov_forward(tapeId4, 1, 3, 3, x, xd, y, yd);

  BOOST_TEST(*y == x1, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][2] == x3Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(CompositeFminOperator_FOV_Reverse) {
  double x1 = 2.31, x2 = 1.32, x3 = 3.21, out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;

  trace_on(tapeId4, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ax1 = fmin(ax1 * pow(ax3, 2), ax2 * pow(ax3, 2)) * exp(ax3);

  ax1 >>= out;
  trace_off();

  double x1Derivative = 0.0;
  double x2Derivative = std::pow(x3, 2) * std::exp(x3);
  double x3Derivative =
      2.0 * x2 * x3 * std::exp(x3) + x2 * std::pow(x3, 2) * std::exp(x3);

  double **u = myalloc2(3, 1);
  double **z = myalloc2(3, 3);

  u[0][0] = 1.;
  u[1][0] = std::sqrt(6.);
  u[2][0] = -std::sqrt(3.);

  fov_reverse(tapeId4, 1, 3, 3, u, z);

  BOOST_TEST(z[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][2] == x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == std::sqrt(6.) * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == std::sqrt(6.) * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][2] == std::sqrt(6.) * x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][0] == -std::sqrt(3.) * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][1] == -std::sqrt(3.) * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][2] == -std::sqrt(3.) * x3Derivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(CompositeMinOperator_FOV_Reverse) {
  double x1 = 2.31, x2 = 1.32, x3 = 3.21, out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;

  trace_on(tapeId4, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ax1 = min(ax1 * pow(ax3, 2), ax2 * pow(ax3, 2)) * exp(ax3);

  ax1 >>= out;
  trace_off();

  double x1Derivative = 0.0;
  double x2Derivative = std::pow(x3, 2) * std::exp(x3);
  double x3Derivative =
      2.0 * x2 * x3 * std::exp(x3) + x2 * std::pow(x3, 2) * std::exp(x3);

  double **u = myalloc2(3, 1);
  double **z = myalloc2(3, 3);

  u[0][0] = 1.;
  u[1][0] = std::sqrt(6.);
  u[2][0] = -std::sqrt(3.);

  fov_reverse(tapeId4, 1, 3, 3, u, z);

  BOOST_TEST(z[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][2] == x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == std::sqrt(6.) * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == std::sqrt(6.) * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][2] == std::sqrt(6.) * x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][0] == -std::sqrt(3.) * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][1] == -std::sqrt(3.) * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][2] == -std::sqrt(3.) * x3Derivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

/* Tested function: erf(fabs(x1 - x2)*sinh(x3 - x4))*sin(x5)
 * Gradient vector: (
 *                    -2./sqrt(acos(-1.)) * exp(-pow(fabs(x1 - x2)
 *                    * sinh(x3 - x4), 2)) * sin(x5) * sinh(x3 - x4),
 *                    2./sqrt(acos(-1.)) * exp(-pow(fabs(x1 - x2)
 *                    * sinh(x3 - x4), 2)) * sin(x5) * sinh(x3 - x4),
 *                    2./sqrt(acos(-1.)) * exp(-pow(fabs(x1 - x2)
 *                    * sinh(x3 - x4), 2)) * sin(x5) * fabs(x1 - x2)
 *                    * cosh(x3 - x4),
 *                    -2./sqrt(acos(-1.)) * exp(-pow(fabs(x1 - x2)
 *                    * sinh(x3 - x4), 2)) * sin(x5) * fabs(x1 - x2)
 *                    * cosh(x3 - x4),
 *                    erf(fabs(x1 - x2)*sinh(x3 - x4))*cos(x5)
 *                  )
 */
BOOST_AUTO_TEST_CASE(CompositeErfFabs_FOV_Forward) {
  double x1 = 4.56, x2 = 5.46, x3 = 4.65, x4 = 6.54, x5 = 6.45, out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ax4;
  adouble ax5;

  trace_on(tapeId4);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;
  ax4 <<= x4;
  ax5 <<= x5;

  ax1 = erf(fabs(ax1 - ax2) * sinh(ax3 - ax4)) * sin(ax5);

  ax1 >>= out;
  trace_off();

  double x1Derivative =
      -2. / std::sqrt(std::acos(-1.)) *
      std::exp(-std::pow(std::fabs(x1 - x2) * std::sinh(x3 - x4), 2)) *
      std::sin(x5) * std::sinh(x3 - x4);
  double x2Derivative =
      2. / std::sqrt(std::acos(-1.)) *
      std::exp(-std::pow(std::fabs(x1 - x2) * std::sinh(x3 - x4), 2)) *
      std::sin(x5) * std::sinh(x3 - x4);
  double x3Derivative =
      2. / std::sqrt(std::acos(-1.)) *
      std::exp(-std::pow(std::fabs(x1 - x2) * std::sinh(x3 - x4), 2)) *
      std::sin(x5) * std::fabs(x1 - x2) * std::cosh(x3 - x4);
  double x4Derivative =
      -2. / std::sqrt(std::acos(-1.)) *
      std::exp(-std::pow(std::fabs(x1 - x2) * std::sinh(x3 - x4), 2)) *
      std::sin(x5) * std::fabs(x1 - x2) * std::cosh(x3 - x4);
  double x5Derivative =
      std::erf(std::fabs(x1 - x2) * std::sinh(x3 - x4)) * std::cos(x5);
  x1 = std::erf(std::fabs(x1 - x2) * std::sinh(x3 - x4)) * std::sin(x5);

  double *x = myalloc1(5);
  double **xd = myalloc2(5, 5);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 5);

  /* Test partial derivative wrt x1, x2, x3, x4 and x5. */
  x[0] = 4.56;
  x[1] = 5.46;
  x[2] = 4.65;
  x[3] = 6.54;
  x[4] = 6.45;

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      if (i == j)
        xd[i][j] = 1.;
      else
        xd[i][j] = 0.;
    }
  }

  fov_forward(tapeId4, 1, 5, 5, x, xd, y, yd);

  BOOST_TEST(*y == x1, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][2] == x3Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][3] == x4Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][4] == x5Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

/* Tested function: erf(fabs(x1 - x2)*sinh(x3 - x4))*sin(x5)
 * Gradient vector: (
 *                    -2./sqrt(acos(-1.)) * exp(-pow(fabs(x1 - x2)
 *                    * sinh(x3 - x4), 2)) * sin(x5) * sinh(x3 - x4),
 *                    2./sqrt(acos(-1.)) * exp(-pow(fabs(x1 - x2)
 *                    * sinh(x3 - x4), 2)) * sin(x5) * sinh(x3 - x4),
 *                    2./sqrt(acos(-1.)) * exp(-pow(fabs(x1 - x2)
 *                    * sinh(x3 - x4), 2)) * sin(x5) * fabs(x1 - x2)
 *                    * cosh(x3 - x4),
 *                    -2./sqrt(acos(-1.)) * exp(-pow(fabs(x1 - x2)
 *                    * sinh(x3 - x4), 2)) * sin(x5) * fabs(x1 - x2)
 *                    * cosh(x3 - x4),
 *                    erf(fabs(x1 - x2)*sinh(x3 - x4))*cos(x5)
 *                  )
 */
BOOST_AUTO_TEST_CASE(CompositeErfAbs_FOV_Forward) {
  double x1 = 4.56, x2 = 5.46, x3 = 4.65, x4 = 6.54, x5 = 6.45, out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ax4;
  adouble ax5;

  trace_on(tapeId4);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;
  ax4 <<= x4;
  ax5 <<= x5;

  ax1 = erf(abs(ax1 - ax2) * sinh(ax3 - ax4)) * sin(ax5);

  ax1 >>= out;
  trace_off();

  double x1Derivative =
      -2. / std::sqrt(std::acos(-1.)) *
      std::exp(-std::pow(std::abs(x1 - x2) * std::sinh(x3 - x4), 2)) *
      std::sin(x5) * std::sinh(x3 - x4);
  double x2Derivative =
      2. / std::sqrt(std::acos(-1.)) *
      std::exp(-std::pow(std::abs(x1 - x2) * std::sinh(x3 - x4), 2)) *
      std::sin(x5) * std::sinh(x3 - x4);
  double x3Derivative =
      2. / std::sqrt(std::acos(-1.)) *
      std::exp(-std::pow(std::abs(x1 - x2) * std::sinh(x3 - x4), 2)) *
      std::sin(x5) * std::abs(x1 - x2) * std::cosh(x3 - x4);
  double x4Derivative =
      -2. / std::sqrt(std::acos(-1.)) *
      std::exp(-std::pow(std::abs(x1 - x2) * std::sinh(x3 - x4), 2)) *
      std::sin(x5) * std::abs(x1 - x2) * std::cosh(x3 - x4);
  double x5Derivative =
      std::erf(std::abs(x1 - x2) * std::sinh(x3 - x4)) * std::cos(x5);
  x1 = std::erf(std::abs(x1 - x2) * std::sinh(x3 - x4)) * std::sin(x5);

  double *x = myalloc1(5);
  double **xd = myalloc2(5, 5);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 5);

  /* Test partial derivative wrt x1, x2, x3, x4 and x5. */
  x[0] = 4.56;
  x[1] = 5.46;
  x[2] = 4.65;
  x[3] = 6.54;
  x[4] = 6.45;

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      if (i == j)
        xd[i][j] = 1.;
      else
        xd[i][j] = 0.;
    }
  }

  fov_forward(tapeId4, 1, 5, 5, x, xd, y, yd);

  BOOST_TEST(*y == x1, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][2] == x3Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][3] == x4Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][4] == x5Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(CompositeErfFabsOperator_FOV_Reverse) {
  double x1 = 4.56, x2 = 5.46, x3 = 4.65, x4 = 6.54, x5 = 6.45, out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ax4;
  adouble ax5;

  trace_on(tapeId4, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;
  ax4 <<= x4;
  ax5 <<= x5;

  ax1 = erf(fabs(ax1 - ax2) * sinh(ax3 - ax4)) * sin(ax5);

  ax1 >>= out;
  trace_off();

  double x1Derivative =
      -2. / std::sqrt(std::acos(-1.)) *
      std::exp(-std::pow(std::fabs(x1 - x2) * std::sinh(x3 - x4), 2)) *
      std::sin(x5) * std::sinh(x3 - x4);
  double x2Derivative =
      2. / std::sqrt(std::acos(-1.)) *
      std::exp(-std::pow(std::fabs(x1 - x2) * std::sinh(x3 - x4), 2)) *
      std::sin(x5) * std::sinh(x3 - x4);
  double x3Derivative =
      2. / std::sqrt(std::acos(-1.)) *
      std::exp(-std::pow(std::fabs(x1 - x2) * std::sinh(x3 - x4), 2)) *
      std::sin(x5) * std::fabs(x1 - x2) * std::cosh(x3 - x4);
  double x4Derivative =
      -2. / std::sqrt(std::acos(-1.)) *
      std::exp(-std::pow(std::fabs(x1 - x2) * std::sinh(x3 - x4), 2)) *
      std::sin(x5) * std::fabs(x1 - x2) * std::cosh(x3 - x4);
  double x5Derivative =
      std::erf(std::fabs(x1 - x2) * std::sinh(x3 - x4)) * std::cos(x5);

  double **u = myalloc2(5, 1);
  double **z = myalloc2(5, 5);

  u[0][0] = 1.;
  u[1][0] = std::sqrt(5.);
  u[2][0] = -std::sqrt(2.);
  u[3][0] = 7.;
  u[4][0] = -9.;

  fov_reverse(tapeId4, 1, 5, 5, u, z);

  BOOST_TEST(z[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][2] == x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][3] == x4Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][4] == x5Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == std::sqrt(5.) * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == std::sqrt(5.) * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][2] == std::sqrt(5.) * x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][3] == std::sqrt(5.) * x4Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][4] == std::sqrt(5.) * x5Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][0] == -std::sqrt(2.) * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][1] == -std::sqrt(2.) * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][2] == -std::sqrt(2.) * x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][3] == -std::sqrt(2.) * x4Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][4] == -std::sqrt(2.) * x5Derivative, tt::tolerance(tol));
  BOOST_TEST(z[3][0] == 7. * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[3][1] == 7. * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[3][2] == 7. * x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[3][3] == 7. * x4Derivative, tt::tolerance(tol));
  BOOST_TEST(z[3][4] == 7. * x5Derivative, tt::tolerance(tol));
  BOOST_TEST(z[4][0] == -9. * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[4][1] == -9. * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[4][2] == -9. * x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[4][3] == -9. * x4Derivative, tt::tolerance(tol));
  BOOST_TEST(z[4][4] == -9. * x5Derivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(CompositeErfAbsOperator_FOV_Reverse) {
  double x1 = 4.56, x2 = 5.46, x3 = 4.65, x4 = 6.54, x5 = 6.45, out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ax4;
  adouble ax5;

  trace_on(tapeId4, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;
  ax4 <<= x4;
  ax5 <<= x5;

  ax1 = erf(abs(ax1 - ax2) * sinh(ax3 - ax4)) * sin(ax5);

  ax1 >>= out;
  trace_off();

  double x1Derivative =
      -2. / std::sqrt(std::acos(-1.)) *
      std::exp(-std::pow(std::abs(x1 - x2) * std::sinh(x3 - x4), 2)) *
      std::sin(x5) * std::sinh(x3 - x4);
  double x2Derivative =
      2. / std::sqrt(std::acos(-1.)) *
      std::exp(-std::pow(std::abs(x1 - x2) * std::sinh(x3 - x4), 2)) *
      std::sin(x5) * std::sinh(x3 - x4);
  double x3Derivative =
      2. / std::sqrt(std::acos(-1.)) *
      std::exp(-std::pow(std::abs(x1 - x2) * std::sinh(x3 - x4), 2)) *
      std::sin(x5) * std::abs(x1 - x2) * std::cosh(x3 - x4);
  double x4Derivative =
      -2. / std::sqrt(std::acos(-1.)) *
      std::exp(-std::pow(std::abs(x1 - x2) * std::sinh(x3 - x4), 2)) *
      std::sin(x5) * std::abs(x1 - x2) * std::cosh(x3 - x4);
  double x5Derivative =
      std::erf(std::abs(x1 - x2) * std::sinh(x3 - x4)) * std::cos(x5);

  double **u = myalloc2(5, 1);
  double **z = myalloc2(5, 5);

  u[0][0] = 1.;
  u[1][0] = std::sqrt(5.);
  u[2][0] = -std::sqrt(2.);
  u[3][0] = 7.;
  u[4][0] = -9.;

  fov_reverse(tapeId4, 1, 5, 5, u, z);

  BOOST_TEST(z[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][2] == x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][3] == x4Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][4] == x5Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == std::sqrt(5.) * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == std::sqrt(5.) * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][2] == std::sqrt(5.) * x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][3] == std::sqrt(5.) * x4Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][4] == std::sqrt(5.) * x5Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][0] == -std::sqrt(2.) * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][1] == -std::sqrt(2.) * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][2] == -std::sqrt(2.) * x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][3] == -std::sqrt(2.) * x4Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][4] == -std::sqrt(2.) * x5Derivative, tt::tolerance(tol));
  BOOST_TEST(z[3][0] == 7. * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[3][1] == 7. * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[3][2] == 7. * x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[3][3] == 7. * x4Derivative, tt::tolerance(tol));
  BOOST_TEST(z[3][4] == 7. * x5Derivative, tt::tolerance(tol));
  BOOST_TEST(z[4][0] == -9. * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[4][1] == -9. * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[4][2] == -9. * x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[4][3] == -9. * x4Derivative, tt::tolerance(tol));
  BOOST_TEST(z[4][4] == -9. * x5Derivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

/* Tested function: 5.*exp(sin(x1)*cos(x1))*pow(sqrt(x2), x3)
 * Gradient vector: (
 *                    5. * exp(sin(x1)*cos(x1)) * (cos(x1)*cos(x1)
 *                    - sin(x1)*sin(x1)) * pow(sqrt(x2), x3),
 *                    5. * exp(sin(x1)*cos(x1)) * x3
 *                    * pow(sqrt(x2), x3 - 1.) * (1. / 2.*sqrt(x2)),
 *                    5. * exp(sin(x1)*cos(x1)) * pow(sqrt(x2), x3)
 *                    * log(sqrt(x2))
 *                  )
 */
BOOST_AUTO_TEST_CASE(ExpTrigSqrt_FOV_Forward) {
  double x1 = 2.1, x2 = 1.2, x3 = 0.12, out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;

  trace_on(tapeId4);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ax1 = 5. * exp(sin(ax1) * cos(ax1)) * pow(sqrt(ax2), ax3);

  ax1 >>= out;
  trace_off();

  double x1Derivative =
      5. * std::exp(std::sin(x1) * std::cos(x1)) *
      (std::cos(x1) * std::cos(x1) - std::sin(x1) * std::sin(x1)) *
      std::pow(std::sqrt(x2), x3);
  double x2Derivative = 5. * std::exp(std::sin(x1) * std::cos(x1)) * x3 *
                        std::pow(std::sqrt(x2), x3 - 1.) / (2. * std::sqrt(x2));
  double x3Derivative = 5. * std::exp(std::sin(x1) * std::cos(x1)) *
                        std::pow(std::sqrt(x2), x3) * std::log(std::sqrt(x2));
  x1 = 5. * std::exp(std::sin(x1) * std::cos(x1)) * std::pow(std::sqrt(x2), x3);

  double *x = myalloc1(3);
  double **xd = myalloc2(3, 3);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 3);

  /* Test partial derivative wrt x1, x2, and x3. */
  x[0] = 2.1;
  x[1] = 1.2;
  x[2] = 0.12;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (i == j)
        xd[i][j] = 1.;
      else
        xd[i][j] = 0.;
    }
  }

  fov_forward(tapeId4, 1, 3, 3, x, xd, y, yd);

  BOOST_TEST(*y == x1, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][2] == x3Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(ExpTrigSqrtFabsOperator_FOV_Reverse) {
  double x1 = 2.1, x2 = 1.2, x3 = 0.12, out;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;

  trace_on(tapeId4, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ax1 = 5. * exp(sin(ax1) * cos(ax1)) * pow(sqrt(ax2), ax3);

  ax1 >>= out;
  trace_off();

  double x1Derivative =
      5. * std::exp(std::sin(x1) * std::cos(x1)) *
      (std::cos(x1) * std::cos(x1) - std::sin(x1) * std::sin(x1)) *
      std::pow(std::sqrt(x2), x3);
  double x2Derivative = 5. * std::exp(std::sin(x1) * std::cos(x1)) * x3 *
                        std::pow(std::sqrt(x2), x3 - 1.) / (2. * std::sqrt(x2));
  double x3Derivative = 5. * std::exp(std::sin(x1) * std::cos(x1)) *
                        std::pow(std::sqrt(x2), x3) * std::log(std::sqrt(x2));

  double **u = myalloc2(3, 1);
  double **z = myalloc2(3, 3);

  u[0][0] = 1.;
  u[1][0] = 3.;
  u[2][0] = -5.;

  fov_reverse(tapeId4, 1, 3, 3, u, z);

  BOOST_TEST(z[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][2] == x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == 3. * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == 3. * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][2] == 3. * x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][0] == -5. * x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][1] == -5. * x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][2] == -5. * x3Derivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

/* The next test functions emphasize the number of dependents being
 * greater than 1, which yields more interesting tests for reverse mode.
 */

/* Tested function: y1 = sqrt(x1*x1 + x2*x2 + x3*x3)
 *                  y2 = atan(sqrt(x1*x1 + x2*x2)/x3)
 *                  y3 = atan(x2/x1)
 * Jacobian matrix: (
 *                    (x1/sqrt(x1*x1 + x2*x2 + x3*x3),
 *                     x2/sqrt(x1*x1 + x2*x2 + x3*x3),
 *                     x3/sqrt(x1*x1 + x2*x2 + x3*x3)),
 *                    (x1*x3/((x1*x1 + x2*x2 + x3*x3)*sqrt(x1*x1 + x2*x2)),
 *                     x2*x3/((x1*x1 + x2*x2 + x3*x3)*sqrt(x1*x1 + x2*x2)),
 *                     -sqrt(x1*x1 + x2*x2)/(x1*x1 + x2*x2 + x3*x3)),
 *                    (-x2/(x1*x1 + x2*x2), x1/(x1*x1 + x2*x2), 0.0)
 *                  )
 */
BOOST_AUTO_TEST_CASE(PolarCoord_FOV_Forward) {
  double x1 = 8.17, x2 = -3.41, x3 = 10.01, out1, out2, out3;
  double y1, y2, y3;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ay1;
  adouble ay2;
  adouble ay3;

  trace_on(tapeId4);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ay1 = sqrt(ax1 * ax1 + ax2 * ax2 + ax3 * ax3);
  ay2 = atan(sqrt(ax1 * ax1 + ax2 * ax2) / ax3);
  ay3 = atan(ax2 / ax1);

  ay1 >>= out1;
  ay2 >>= out2;
  ay3 >>= out3;
  trace_off();

  /* The obvious naming convention is applied here:  The derivative of
   * component yi in the direction xj is saved in yixjDerivative.
   */
  double y1x1Derivative = x1 / std::sqrt(x1 * x1 + x2 * x2 + x3 * x3);
  double y1x2Derivative = x2 / std::sqrt(x1 * x1 + x2 * x2 + x3 * x3);
  double y1x3Derivative = x3 / std::sqrt(x1 * x1 + x2 * x2 + x3 * x3);
  double y2x1Derivative =
      x1 * x3 / ((x1 * x1 + x2 * x2 + x3 * x3) * std::sqrt(x1 * x1 + x2 * x2));
  double y2x2Derivative =
      x2 * x3 / ((x1 * x1 + x2 * x2 + x3 * x3) * std::sqrt(x1 * x1 + x2 * x2));
  double y2x3Derivative =
      -std::sqrt(x1 * x1 + x2 * x2) / (x1 * x1 + x2 * x2 + x3 * x3);
  double y3x1Derivative = -x2 / (x1 * x1 + x2 * x2);
  double y3x2Derivative = x1 / (x1 * x1 + x2 * x2);
  double y3x3Derivative = 0.0;

  y1 = std::sqrt(x1 * x1 + x2 * x2 + x3 * x3);
  y2 = std::atan(std::sqrt(x1 * x1 + x2 * x2) / x3);
  y3 = std::atan(x2 / x1);

  double *x = myalloc1(3);
  double **xd = myalloc2(3, 3);
  double *y = myalloc1(3);
  double **yd = myalloc2(3, 3);

  /* Test all entries of Jacobian matrix. */
  x[0] = 8.17;
  x[1] = -3.41;
  x[2] = 10.01;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (i == j)
        xd[i][j] = 1.;
      else
        xd[i][j] = 0.;
    }
  }

  fov_forward(tapeId4, 3, 3, 3, x, xd, y, yd);

  BOOST_TEST(y[0] == y1, tt::tolerance(tol));
  BOOST_TEST(y[1] == y2, tt::tolerance(tol));
  BOOST_TEST(y[2] == y3, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][2] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[1][0] == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[1][1] == y2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[1][2] == y2x3Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[2][0] == y3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[2][1] == y3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[2][2] == y3x3Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(PolarCoordOperator_FOV_Reverse) {
  double x1 = 8.17, x2 = -3.41, x3 = 10.01, out1, out2, out3;
  double y1, y2, y3;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ay1;
  adouble ay2;
  adouble ay3;

  trace_on(tapeId4, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ay1 = sqrt(ax1 * ax1 + ax2 * ax2 + ax3 * ax3);
  ay2 = atan(sqrt(ax1 * ax1 + ax2 * ax2) / ax3);
  ay3 = atan(ax2 / ax1);

  ay1 >>= out1;
  ay2 >>= out2;
  ay3 >>= out3;
  trace_off();

  /* The obvious naming convention is applied here:  The derivative of
   * component yi in the direction xj is saved in yixjDerivative.
   */
  double y1x1Derivative = x1 / std::sqrt(x1 * x1 + x2 * x2 + x3 * x3);
  double y1x2Derivative = x2 / std::sqrt(x1 * x1 + x2 * x2 + x3 * x3);
  double y1x3Derivative = x3 / std::sqrt(x1 * x1 + x2 * x2 + x3 * x3);
  double y2x1Derivative =
      x1 * x3 / ((x1 * x1 + x2 * x2 + x3 * x3) * std::sqrt(x1 * x1 + x2 * x2));
  double y2x2Derivative =
      x2 * x3 / ((x1 * x1 + x2 * x2 + x3 * x3) * std::sqrt(x1 * x1 + x2 * x2));
  double y2x3Derivative =
      -std::sqrt(x1 * x1 + x2 * x2) / (x1 * x1 + x2 * x2 + x3 * x3);
  double y3x1Derivative = -x2 / (x1 * x1 + x2 * x2);
  double y3x2Derivative = x1 / (x1 * x1 + x2 * x2);
  double y3x3Derivative = 0.0;

  double **u = myalloc2(3, 3);
  double **z = myalloc2(3, 3);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (i == j)
        u[i][j] = 1.;
      else
        u[i][j] = 0.;
    }
  }

  fov_reverse(tapeId4, 3, 3, 3, u, z);

  BOOST_TEST(z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][2] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == y2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][2] == y2x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][0] == y3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][1] == y3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][2] == y3x3Derivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

/* Tested function: y1 = x2*x3
 *                  y2 = x1*x3
 *                  y3 = x1*x2
 * Jacobian matrix: (
 *                    (0.0, x3, x2),
 *                    (x3, 0.0, x1),
 *                    (x2, x1, 0.0)
 *                  )
 */
BOOST_AUTO_TEST_CASE(SimpleProd_FOV_Forward) {
  double x1 = 2.52, x2 = 5.22, x3 = -2.25, out1, out2, out3;
  double y1, y2, y3;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ay1;
  adouble ay2;
  adouble ay3;

  trace_on(tapeId4);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ay1 = ax2 * ax3;
  ay2 = ax1 * ax3;
  ay3 = ax1 * ax2;

  ay1 >>= out1;
  ay2 >>= out2;
  ay3 >>= out3;
  trace_off();

  double y1x1Derivative = 0.0;
  double y1x2Derivative = x3;
  double y1x3Derivative = x2;
  double y2x1Derivative = x3;
  double y2x2Derivative = 0.0;
  double y2x3Derivative = x1;
  double y3x1Derivative = x2;
  double y3x2Derivative = x1;
  double y3x3Derivative = 0.0;

  y1 = x2 * x3;
  y2 = x1 * x3;
  y3 = x1 * x2;

  double *x = myalloc1(3);
  double **xd = myalloc2(3, 3);
  double *y = myalloc1(3);
  double **yd = myalloc2(3, 3);

  x[0] = 2.52;
  x[1] = 5.22;
  x[2] = -2.25;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (i == j)
        xd[i][j] = 1.;
      else
        xd[i][j] = 0.;
    }
  }

  fov_forward(tapeId4, 3, 3, 3, x, xd, y, yd);

  BOOST_TEST(y[0] == y1, tt::tolerance(tol));
  BOOST_TEST(y[1] == y2, tt::tolerance(tol));
  BOOST_TEST(y[2] == y3, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][2] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[1][0] == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[1][1] == y2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[1][2] == y2x3Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[2][0] == y3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[2][1] == y3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[2][2] == y3x3Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(SimpleProdOperator_FOV_Reverse) {
  double x1 = 2.52, x2 = 5.22, x3 = -2.25, out1, out2, out3;
  double y1, y2, y3;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ay1;
  adouble ay2;
  adouble ay3;

  trace_on(tapeId4, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ay1 = ax2 * ax3;
  ay2 = ax1 * ax3;
  ay3 = ax1 * ax2;

  ay1 >>= out1;
  ay2 >>= out2;
  ay3 >>= out3;
  trace_off();

  double y1x1Derivative = 0.0;
  double y1x2Derivative = x3;
  double y1x3Derivative = x2;
  double y2x1Derivative = x3;
  double y2x2Derivative = 0.0;
  double y2x3Derivative = x1;
  double y3x1Derivative = x2;
  double y3x2Derivative = x1;
  double y3x3Derivative = 0.0;

  double **u = myalloc2(3, 3);
  double **z = myalloc2(3, 3);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (i == j)
        u[i][j] = 1.;
      else
        u[i][j] = 0.;
    }
  }

  fov_reverse(tapeId4, 3, 3, 3, u, z);

  BOOST_TEST(z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][2] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == y2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][2] == y2x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][0] == y3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][1] == y3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][2] == y3x3Derivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

/* Tested function: y1 = x2 + x3
 *                  y2 = x1 + x3
 *                  y3 = x1 + x2
 * Jacobian matrix: (
 *                    (0.0, 1.0, 1.0),
 *                    (1.0, 0.0, 1.0),
 *                    (1.0, 1.0, 0.0)
 *                  )
 */
BOOST_AUTO_TEST_CASE(SimpleSum_FOV_Forward) {
  double x1 = 2.52, x2 = 5.22, x3 = -2.25, out1, out2, out3;
  double y1, y2, y3;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ay1;
  adouble ay2;
  adouble ay3;

  trace_on(tapeId4);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ay1 = ax2 + ax3;
  ay2 = ax1 + ax3;
  ay3 = ax1 + ax2;

  ay1 >>= out1;
  ay2 >>= out2;
  ay3 >>= out3;
  trace_off();

  double y1x1Derivative = 0.0;
  double y1x2Derivative = 1.0;
  double y1x3Derivative = 1.0;
  double y2x1Derivative = 1.0;
  double y2x2Derivative = 0.0;
  double y2x3Derivative = 1.0;
  double y3x1Derivative = 1.0;
  double y3x2Derivative = 1.0;
  double y3x3Derivative = 0.0;

  y1 = x2 + x3;
  y2 = x1 + x3;
  y3 = x1 + x2;

  double *x = myalloc1(3);
  double **xd = myalloc2(3, 3);
  double *y = myalloc1(3);
  double **yd = myalloc2(3, 3);

  x[0] = 2.52;
  x[1] = 5.22;
  x[2] = -2.25;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (i == j)
        xd[i][j] = 1.;
      else
        xd[i][j] = 0.;
    }
  }

  fov_forward(tapeId4, 3, 3, 3, x, xd, y, yd);

  BOOST_TEST(y[0] == y1, tt::tolerance(tol));
  BOOST_TEST(y[1] == y2, tt::tolerance(tol));
  BOOST_TEST(y[2] == y3, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][2] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[1][0] == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[1][1] == y2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[1][2] == y2x3Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[2][0] == y3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[2][1] == y3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[2][2] == y3x3Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(SimpleSumOperator_FOV_Reverse) {
  double x1 = 2.52, x2 = 5.22, x3 = -2.25, out1, out2, out3;
  double y1, y2, y3;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ay1;
  adouble ay2;
  adouble ay3;

  trace_on(tapeId4, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ay1 = ax2 + ax3;
  ay2 = ax1 + ax3;
  ay3 = ax1 + ax2;

  ay1 >>= out1;
  ay2 >>= out2;
  ay3 >>= out3;
  trace_off();

  double y1x1Derivative = 0.0;
  double y1x2Derivative = 1.0;
  double y1x3Derivative = 1.0;
  double y2x1Derivative = 1.0;
  double y2x2Derivative = 0.0;
  double y2x3Derivative = 1.0;
  double y3x1Derivative = 1.0;
  double y3x2Derivative = 1.0;
  double y3x3Derivative = 0.0;

  double **u = myalloc2(3, 3);
  double **z = myalloc2(3, 3);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (i == j)
        u[i][j] = 1.;
      else
        u[i][j] = 0.;
    }
  }

  fov_reverse(tapeId4, 3, 3, 3, u, z);

  BOOST_TEST(z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][2] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == y2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][2] == y2x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][0] == y3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][1] == y3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][2] == y3x3Derivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

/* Tested function: y1 = x1*cos(x2) + sqrt(x3*x4)
 *                  y2 = x4
 *                  y3 = x1*x2*x3*x4
 *                  y4 = atan((x1 + x2)/(x3 + x4))
 * Jacobian matrix: (
 *                    (cos(x2), -x1*sin(x2), 0.5*sqrt(x4/x3),
 *                     0.5*sqrt(x3/x4)),
 *                    (0.0, 0.0, 0.0, 1.0),
 *                    (x2*x3*x4, x1*x3*x4, x1*x2*x4, x1*x2*x3),
 *                    ((x3 + x4)/(pow(x1 + x2, 2) + pow(x3 + x4, 2)),
 *                     (x3 + x4)/(pow(x1 + x2, 2) + pow(x3 + x4, 2)),
 *                     -(x1 + x2)/(pow(x1 + x2, 2) + pow(x3 + x4, 2)),
 *                     -(x1 + x2)/(pow(x1 + x2, 2) + pow(x3 + x4, 2)))
 *                  )
 */
BOOST_AUTO_TEST_CASE(TrigProd_FOV_Forward) {
  double x1 = 5.5, x2 = 0.5, x3 = 5.55, x4 = 2.33, out1, out2, out3, out4;
  double y1, y2, y3, y4;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ax4;
  adouble ay1;
  adouble ay2;
  adouble ay3;
  adouble ay4;

  trace_on(tapeId4);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;
  ax4 <<= x4;

  ay1 = ax1 * cos(ax2) + sqrt(ax3 * ax4);
  ay2 = ax4;
  ay3 = ax1 * ax2 * ax3 * ax4;
  ay4 = atan((ax1 + ax2) / (ax3 + ax4));

  ay1 >>= out1;
  ay2 >>= out2;
  ay3 >>= out3;
  ay4 >>= out4;
  trace_off();

  double y1x1Derivative = std::cos(x2);
  double y1x2Derivative = -x1 * std::sin(x2);
  double y1x3Derivative = 0.5 * std::sqrt(x4 / x3);
  double y1x4Derivative = 0.5 * std::sqrt(x3 / x4);
  double y2x1Derivative = 0.0;
  double y2x2Derivative = 0.0;
  double y2x3Derivative = 0.0;
  double y2x4Derivative = 1.0;
  double y3x1Derivative = x2 * x3 * x4;
  double y3x2Derivative = x1 * x3 * x4;
  double y3x3Derivative = x1 * x2 * x4;
  double y3x4Derivative = x1 * x2 * x3;
  double y4x1Derivative =
      (x3 + x4) / (std::pow(x1 + x2, 2) + std::pow(x3 + x4, 2));
  double y4x2Derivative =
      (x3 + x4) / (std::pow(x1 + x2, 2) + std::pow(x3 + x4, 2));
  double y4x3Derivative =
      -(x1 + x2) / (std::pow(x1 + x2, 2) + std::pow(x3 + x4, 2));
  double y4x4Derivative =
      -(x1 + x2) / (std::pow(x1 + x2, 2) + std::pow(x3 + x4, 2));

  y1 = x1 * std::cos(x2) + std::sqrt(x3 * x4);
  y2 = x4;
  y3 = x1 * x2 * x3 * x4;
  y4 = std::atan((x1 + x2) / (x3 + x4));

  double *x = myalloc1(4);
  double **xd = myalloc2(4, 4);
  double *y = myalloc1(4);
  double **yd = myalloc2(4, 4);

  x[0] = 5.5;
  x[1] = 0.5;
  x[2] = 5.55;
  x[3] = 2.33;

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      if (i == j)
        xd[i][j] = 1.;
      else
        xd[i][j] = 0.;
    }
  }

  fov_forward(tapeId4, 4, 4, 4, x, xd, y, yd);

  BOOST_TEST(y[0] == y1, tt::tolerance(tol));
  BOOST_TEST(y[1] == y2, tt::tolerance(tol));
  BOOST_TEST(y[2] == y3, tt::tolerance(tol));
  BOOST_TEST(y[3] == y4, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][2] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][3] == y1x4Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[1][0] == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[1][1] == y2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[1][2] == y2x3Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[1][3] == y2x4Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[2][0] == y3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[2][1] == y3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[2][2] == y3x3Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[2][3] == y3x4Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[3][0] == y4x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[3][1] == y4x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[3][2] == y4x3Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[3][3] == y4x4Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(TrigProdOperator_FOV_Reverse) {
  double x1 = 5.5, x2 = 0.5, x3 = 5.55, x4 = 2.33, out1, out2, out3, out4;
  double y1, y2, y3, y4;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ax4;
  adouble ay1;
  adouble ay2;
  adouble ay3;
  adouble ay4;

  trace_on(tapeId4, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;
  ax4 <<= x4;

  ay1 = ax1 * cos(ax2) + sqrt(ax3 * ax4);
  ay2 = ax4;
  ay3 = ax1 * ax2 * ax3 * ax4;
  ay4 = atan((ax1 + ax2) / (ax3 + ax4));

  ay1 >>= out1;
  ay2 >>= out2;
  ay3 >>= out3;
  ay4 >>= out4;
  trace_off();

  double y1x1Derivative = std::cos(x2);
  double y1x2Derivative = -x1 * std::sin(x2);
  double y1x3Derivative = 0.5 * std::sqrt(x4 / x3);
  double y1x4Derivative = 0.5 * std::sqrt(x3 / x4);
  double y2x1Derivative = 0.0;
  double y2x2Derivative = 0.0;
  double y2x3Derivative = 0.0;
  double y2x4Derivative = 1.0;
  double y3x1Derivative = x2 * x3 * x4;
  double y3x2Derivative = x1 * x3 * x4;
  double y3x3Derivative = x1 * x2 * x4;
  double y3x4Derivative = x1 * x2 * x3;
  double y4x1Derivative =
      (x3 + x4) / (std::pow(x1 + x2, 2) + std::pow(x3 + x4, 2));
  double y4x2Derivative =
      (x3 + x4) / (std::pow(x1 + x2, 2) + std::pow(x3 + x4, 2));
  double y4x3Derivative =
      -(x1 + x2) / (std::pow(x1 + x2, 2) + std::pow(x3 + x4, 2));
  double y4x4Derivative =
      -(x1 + x2) / (std::pow(x1 + x2, 2) + std::pow(x3 + x4, 2));

  double **u = myalloc2(4, 4);
  double **z = myalloc2(4, 4);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      if (i == j)
        u[i][j] = 1.;
      else
        u[i][j] = 0.;
    }
  }

  fov_reverse(tapeId4, 4, 4, 4, u, z);

  BOOST_TEST(z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][2] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][3] == y1x4Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == y2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][2] == y2x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][3] == y2x4Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][0] == y3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][1] == y3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][2] == y3x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][3] == y3x4Derivative, tt::tolerance(tol));
  BOOST_TEST(z[3][0] == y4x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[3][1] == y4x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[3][2] == y4x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[3][3] == y4x4Derivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

/* Tested function: y1 = x1*cos(x2)*sin(x3)
 *                  y2 = x1*sin(x2)*sin(x3)
 *                  y3 = x1*cos(x3)
 * Jacobian matrix: (
 *                    (cos(x2)*sin(x3), -x1*sin(x2)*sin(x3),
 *                     x1*cos(x2)*cos(x3)),
 *                    (sin(x2)*sin(x3), x1*cos(x2)*sin(x3),
 *                     x1*sin(x2)*cos(x3)),
 *                    (cos(x3), 0.0, -x1*sin(x3)),
 *                  )
 */
BOOST_AUTO_TEST_CASE(PolarCoordInv_FOV_Forward) {
  double x1 = 4.21, x2 = -0.98, x3 = 3.02, out1, out2, out3;
  double y1, y2, y3;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ay1;
  adouble ay2;
  adouble ay3;

  trace_on(tapeId4);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ay1 = ax1 * cos(ax2) * sin(ax3);
  ay2 = ax1 * sin(ax2) * sin(ax3);
  ay3 = ax1 * cos(ax3);

  ay1 >>= out1;
  ay2 >>= out2;
  ay3 >>= out3;
  trace_off();

  double y1x1Derivative = std::cos(x2) * std::sin(x3);
  double y1x2Derivative = -x1 * std::sin(x2) * std::sin(x3);
  double y1x3Derivative = x1 * std::cos(x2) * std::cos(x3);
  double y2x1Derivative = std::sin(x2) * std::sin(x3);
  double y2x2Derivative = x1 * std::cos(x2) * std::sin(x3);
  double y2x3Derivative = x1 * std::sin(x2) * std::cos(x3);
  double y3x1Derivative = std::cos(x3);
  double y3x2Derivative = 0.0;
  double y3x3Derivative = -x1 * std::sin(x3);

  y1 = x1 * std::cos(x2) * std::sin(x3);
  y2 = x1 * std::sin(x2) * std::sin(x3);
  y3 = x1 * std::cos(x3);

  double *x = myalloc1(3);
  double **xd = myalloc2(3, 3);
  double *y = myalloc1(3);
  double **yd = myalloc2(3, 3);

  x[0] = 4.21;
  x[1] = -0.98;
  x[2] = 3.02;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (i == j)
        xd[i][j] = 1.;
      else
        xd[i][j] = 0.;
    }
  }

  fov_forward(tapeId4, 3, 3, 3, x, xd, y, yd);

  BOOST_TEST(y[0] == y1, tt::tolerance(tol));
  BOOST_TEST(y[1] == y2, tt::tolerance(tol));
  BOOST_TEST(y[2] == y3, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][2] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[1][0] == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[1][1] == y2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[1][2] == y2x3Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[2][0] == y3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[2][1] == y3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[2][2] == y3x3Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(PolarCoordInvProdOperator_FOV_Reverse) {
  double x1 = 4.21, x2 = -0.98, x3 = 3.02, out1, out2, out3;
  double y1, y2, y3;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ay1;
  adouble ay2;
  adouble ay3;

  trace_on(tapeId4, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ay1 = ax1 * cos(ax2) * sin(ax3);
  ay2 = ax1 * sin(ax2) * sin(ax3);
  ay3 = ax1 * cos(ax3);

  ay1 >>= out1;
  ay2 >>= out2;
  ay3 >>= out3;
  trace_off();

  double y1x1Derivative = std::cos(x2) * std::sin(x3);
  double y1x2Derivative = -x1 * std::sin(x2) * std::sin(x3);
  double y1x3Derivative = x1 * std::cos(x2) * std::cos(x3);
  double y2x1Derivative = std::sin(x2) * std::sin(x3);
  double y2x2Derivative = x1 * std::cos(x2) * std::sin(x3);
  double y2x3Derivative = x1 * std::sin(x2) * std::cos(x3);
  double y3x1Derivative = std::cos(x3);
  double y3x2Derivative = 0.0;
  double y3x3Derivative = -x1 * std::sin(x3);

  double **u = myalloc2(3, 3);
  double **z = myalloc2(3, 3);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (i == j)
        u[i][j] = 1.;
      else
        u[i][j] = 0.;
    }
  }

  fov_reverse(tapeId4, 3, 3, 3, u, z);

  BOOST_TEST(z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][2] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == y2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][2] == y2x3Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][0] == y3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][1] == y3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][2] == y3x3Derivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

/* Tested function: y1 = sinh(x1*x1)*cosh(x2*x2*x2)
 *                  y2 = pow(cosh(pow(x1, 4)), 2)
 *                       - pow(cosh(pow(x1, 4)), 2)
 *                  y3 = -cosh(sqrt(x1)*x2)*x2
 *                  y4 = cosh(x1)/sinh(x2)
 * Jacobian matrix: (
 *                    (2*x1*cosh(x1*x1)*cosh(x2*x2*x2),
 *                     3*x2*x2*sinh(x1*x1)*sinh(x2*x2*x2)),
 *                    (0.0, 0.0),
 *                    (-0.5*sinh(sqrt(x1)*x2)*x2*x2/sqrt(x1),
 *                     -sinh(sqrt(x1)*x2)*sqrt(x1)*x2 - cosh(sqrt(x1)*x2)),
 *                    (sinh(x1)/sinh(x2), -cosh(x1)/(sinh(x2)*cosh(x2)))
 *                  )
 */
BOOST_AUTO_TEST_CASE(MultiHyperb_FOV_Forward) {
  double x1 = 1., x2 = 0.1, out1, out2, out3, out4;
  double y1, y2, y3, y4;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ay1;
  adouble ay2;
  adouble ay3;
  adouble ay4;

  trace_on(tapeId4);
  ax1 <<= x1;
  ax2 <<= x2;

  ay1 = sinh(ax1 * ax1) * cosh(ax2 * ax2 * ax2);
  ay2 = pow(cosh(pow(ax1, 4.)), 2.) - pow(cosh(pow(ax1, 4.)), 2.);
  ay3 = -cosh(sqrt(ax1) * ax2) * ax2;
  ay4 = cosh(ax1) / sinh(ax2);

  ay1 >>= out1;
  ay2 >>= out2;
  ay3 >>= out3;
  ay4 >>= out4;
  trace_off();

  double y1x1Derivative =
      2. * x1 * std::cosh(x1 * x1) * std::cosh(x2 * x2 * x2);
  double y1x2Derivative =
      3. * x2 * x2 * std::sinh(x1 * x1) * std::sinh(x2 * x2 * x2);
  double y2x1Derivative = 0.0;
  double y2x2Derivative = 0.0;
  double y3x1Derivative =
      -0.5 * std::sinh(std::sqrt(x1) * x2) * x2 * x2 / std::sqrt(x1);
  double y3x2Derivative = -std::sinh(std::sqrt(x1) * x2) * std::sqrt(x1) * x2 -
                          std::cosh(std::sqrt(x1) * x2);
  double y4x1Derivative = std::sinh(x1) / std::sinh(x2);
  double y4x2Derivative =
      -std::cosh(x1) * std::cosh(x2) / std::pow(std::sinh(x2), 2.);

  y1 = std::sinh(x1 * x1) * std::cosh(x2 * x2 * x2);
  y2 = std::pow(std::cosh(std::pow(x1, 4)), 2) -
       std::pow(std::cosh(std::pow(x1, 4)), 2);
  y3 = -std::cosh(std::sqrt(x1) * x2) * x2;
  y4 = std::cosh(x1) / std::sinh(x2);

  double *x = myalloc1(2);
  double **xd = myalloc2(2, 2);
  double *y = myalloc1(4);
  double **yd = myalloc2(4, 2);

  x[0] = 1.;
  x[1] = 0.1;

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      if (i == j)
        xd[i][j] = 1.;
      else
        xd[i][j] = 0.;
    }
  }

  fov_forward(tapeId4, 4, 2, 2, x, xd, y, yd);

  BOOST_TEST(y[0] == y1, tt::tolerance(tol));
  BOOST_TEST(y[1] == y2, tt::tolerance(tol));
  BOOST_TEST(y[2] == y3, tt::tolerance(tol));
  BOOST_TEST(y[3] == y4, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[1][0] == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[1][1] == y2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[2][0] == y3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[2][1] == y3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[3][0] == y4x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[3][1] == y4x2Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(MultiHyperbProdOperator_FOV_Reverse) {
  double x1 = 1., x2 = 0.1, out1, out2, out3, out4;
  double y1, y2, y3, y4;
  const short tapeId4 = 1;

  setCurrentTape(tapeId4);

  adouble ax1;
  adouble ax2;
  adouble ay1;
  adouble ay2;
  adouble ay3;
  adouble ay4;

  trace_on(tapeId4, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay1 = sinh(ax1 * ax1) * cosh(ax2 * ax2 * ax2);
  ay2 = pow(cosh(pow(ax1, 4.)), 2.) - pow(cosh(pow(ax1, 4.)), 2.);
  ay3 = -cosh(sqrt(ax1) * ax2) * ax2;
  ay4 = cosh(ax1) / sinh(ax2);

  ay1 >>= out1;
  ay2 >>= out2;
  ay3 >>= out3;
  ay4 >>= out4;
  trace_off();

  double y1x1Derivative =
      2. * x1 * std::cosh(x1 * x1) * std::cosh(x2 * x2 * x2);
  double y1x2Derivative =
      3. * x2 * x2 * std::sinh(x1 * x1) * std::sinh(x2 * x2 * x2);
  double y2x1Derivative = 0.0;
  double y2x2Derivative = 0.0;
  double y3x1Derivative =
      -0.5 * std::sinh(std::sqrt(x1) * x2) * x2 * x2 / std::sqrt(x1);
  double y3x2Derivative = -std::sinh(std::sqrt(x1) * x2) * std::sqrt(x1) * x2 -
                          std::cosh(std::sqrt(x1) * x2);
  double y4x1Derivative = std::sinh(x1) / std::sinh(x2);
  double y4x2Derivative =
      -std::cosh(x1) * std::cosh(x2) / std::pow(std::sinh(x2), 2.);

  double **u = myalloc2(4, 4);
  double **z = myalloc2(4, 2);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      if (i == j)
        u[i][j] = 1.;
      else
        u[i][j] = 0.;
    }
  }

  fov_reverse(tapeId4, 4, 2, 4, u, z);

  BOOST_TEST(z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == y2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][0] == y3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[2][1] == y3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(z[3][0] == y4x1Derivative, tt::tolerance(tol));
  BOOST_TEST(z[3][1] == y4x2Derivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_SUITE_END()

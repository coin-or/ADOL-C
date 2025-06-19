#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

#include <adolc/adolc.h>

#include "const.h"

BOOST_AUTO_TEST_SUITE(trace_sec_order)

/**************************************/
/* Tests for ADOL-C trace scalar mode */
/* drivers hos_forward, hessian and   */
/* hos_reverse                        */
/* Author: Philipp Schuette           */
/**************************************/

/* This file contains custom tests for the higher order derivative
 * evaluation drivers hos_forward, 'hessian'.  As 'hessian' uses
 * hos_reverse internally (after a call to fos_foward), this effectively
 * tests higher order derivative evaluation in both forward and reverse
 * mode.  In this file, only second derivatives are tests.
 *
 * As for the trace/traceless first order derivative tests, the custom
 * functions are described, together with their derivatives, before
 * the actual test implementation.
 */

const short tapeId3 = 3;
struct TapeInitializer {
  TapeInitializer() { createNewTape(tapeId3); }
};
BOOST_GLOBAL_FIXTURE(TapeInitializer);
/* Tested function: 2.*x*x*x
 * First derivative: 2.*3.*x*x
 * Second derivative: 2.*3.*2.*x
 */
BOOST_AUTO_TEST_CASE(CustomCube_HOS) {

  setCurrentTape(tapeId3);

  double x = 3.;
  adouble ax;
  double y;
  adouble ay;

  trace_on(tapeId3, 1);
  ax <<= x;

  ay = 2. * ax * ax * ax;

  ay >>= y;
  trace_off();

  // Calculate primitive value analytically for testing.
  double yprim = 2. * x * x * x;
  // Calculate first and second derivative analytically for testing.
  double **yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = 2. * 3. * x * x;
  yDerivative[0][1] = 2. * 3. * x * x + 0.5 * (2. * 3. * 2. * x);

  double **X;
  X = myalloc2(1, 2);
  X[0][0] = 1.;
  X[0][1] = 1.;

  double **Y;
  Y = myalloc2(1, 2);

  // Signature: hos_forward(tag, m, n, d, keep, x[n], X[n][d], y[m], Y[m][d])
  hos_forward(tapeId3, 1, 1, 2, 1, &x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double **H;
  H = myalloc2(1, 1);

  // Calculate Hessian matrix analytically:
  double yxxDerivative = 2. * 3. * 2. * x;

  hessian(tapeId3, 1, &x, H);

  BOOST_TEST(yxxDerivative == H[0][0], tt::tolerance(tol));

  myfree2(yDerivative);
  myfree2(X);
  myfree2(Y);
  myfree2(H);
}

/* Tested function: cos(x1)*sin(x2)
 * First derivatives: (-sin(x1)*sin(x2), cos(x1)*cos(x2))
 * Second derivatives: (-cos(x1)*sin(x2), -sin(x1)*cos(x2),
 *                      -sin(x1)*cos(x2), -cos(x1)*sin(x2))
 */
BOOST_AUTO_TEST_CASE(CustomTrigProd_HOS) {
  double x1 = 1.3, x2 = 3.1;

  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  double y;
  adouble ay;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay = cos(ax1) * sin(ax2);

  ay >>= y;
  trace_off();

  double yprim = std::cos(x1) * std::sin(x2);

  double **yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = -std::sin(x1) * std::sin(x2);
  yDerivative[0][1] =
      std::cos(x1) * std::cos(x2) + 0.5 * (-std::cos(x1)) * std::sin(x2);

  double *x;
  x = myalloc1(2);
  x[0] = x1;
  x[1] = x2;

  double **X;
  X = myalloc2(2, 2);
  X[0][0] = 1.;
  X[0][1] = 0.;
  X[1][0] = 0.;
  X[1][1] = 1.;

  double **Y;
  Y = myalloc2(1, 2);

  hos_forward(tapeId3, 1, 2, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double **H;
  H = myalloc2(2, 2);

  double yx1x1Derivative = -std::cos(x1) * std::sin(x2);
  double yx1x2Derivative = -std::sin(x1) * std::cos(x2);
  double yx2x2Derivative = -std::cos(x1) * std::sin(x2);

  hessian(tapeId3, 2, x, H);

  BOOST_TEST(yx1x1Derivative == H[0][0], tt::tolerance(tol));
  BOOST_TEST(yx1x2Derivative == H[1][0], tt::tolerance(tol));
  BOOST_TEST(yx2x2Derivative == H[1][1], tt::tolerance(tol));

  myfree1(x);
  myfree2(yDerivative);
  myfree2(X);
  myfree2(Y);
  myfree2(H);
}

/* Tested function: pow(x1, x2)*exp(2.*x3)
 * First derivatives: (x2*pow(x1, x2 - 1)*exp(2.*x3),
 *                     pow(x1, x2)*log(x1)*exp(2.*x3),
 *                     2.*pow(x1, x2)*exp(2.*x3))
 * Second derivatives: ((x2*(x2 - 1)*pow(x1, x2 - 2)*exp(2.*x3),
 *                       x2*pow(x1, x2 - 1)*log(x1)*exp(2.*x3)
 *                       + pow(x1, x2 - 1)*exp(2.*x3),
 *                       2.*x2*pow(x1, x2 - 1)*exp(2.*x3)),
 *                      (pow(x1, x2)*exp(2.*x3)/x1
 *                       + x2*pow(x1, x2 - 1)*log(x1)*exp(2.*x3),
 *                       pow(x1, x2)*log(x1)*log(x1)*exp(2.*x3),
 *                       2.*pow(x1, x2)*log(x1)*exp(2.*x3)),
 *                      (2.*x2*pow(x1, x2 - 1)*exp(2.*x3),
 *                       2.*pow(x1, x2)*log(x1)*exp(2.*x3)
 *                       4.*pow(x1, x2)*exp(2.*x3)))
 */
BOOST_AUTO_TEST_CASE(CustomTrigPow_HOS) {
  double x1 = 1.1, x2 = 4.53, x3 = -3.03;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  adouble ax3;
  double y;
  adouble ay;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ay = pow(ax1, ax2) * exp(2. * ax3);

  ay >>= y;
  trace_off();

  double yprim = std::pow(x1, x2) * std::exp(2. * x3);

  double **yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = x2 * std::pow(x1, x2 - 1) * std::exp(2. * x3) +
                      0.1 * std::pow(x1, x2) * std::log(x1) * std::exp(2. * x3);
  yDerivative[0][1] =
      std::pow(x1, x2) * std::log(x1) * std::exp(2. * x3) +
      0.2 * 2. * std::pow(x1, x2) * std::exp(2. * x3) +
      0.5 *
          (x2 * (x2 - 1) * std::pow(x1, x2 - 2) * std::exp(2. * x3) +
           0.1 * x2 * std::pow(x1, x2 - 1) * std::log(x1) * std::exp(2. * x3) +
           0.1 * std::pow(x1, x2 - 1) * std::exp(2. * x3) +
           0.1 * std::pow(x1, x2 - 1) * std::exp(2. * x3) +
           0.1 * x2 * std::pow(x1, x2 - 1) * std::log(x1) * std::exp(2. * x3) +
           0.01 * std::pow(x1, x2) * std::pow(std::log(x1), 2) *
               std::exp(2. * x3));

  double *x;
  x = myalloc1(3);
  x[0] = x1;
  x[1] = x2;
  x[2] = x3;

  double **X;
  X = myalloc2(3, 2);
  X[0][0] = 1.;
  X[1][0] = 0.1;
  X[2][0] = 0.;
  X[0][1] = 0.;
  X[1][1] = 1.;
  X[2][1] = 0.2;

  double **Y;
  Y = myalloc2(1, 2);

  hos_forward(tapeId3, 1, 3, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double **H;
  H = myalloc2(3, 3);

  double yx1x1Derivative =
      x2 * (x2 - 1) * std::pow(x1, x2 - 2) * std::exp(2. * x3);
  double yx2x1Derivative =
      x2 * std::pow(x1, x2 - 1) * std::log(x1) * std::exp(2. * x3) +
      std::pow(x1, x2 - 1) * std::exp(2. * x3);
  double yx3x1Derivative = 2. * x2 * std::pow(x1, x2 - 1) * std::exp(2. * x3);
  double yx2x2Derivative =
      std::pow(x1, x2) * std::pow(std::log(x1), 2) * std::exp(2. * x3);
  double yx3x2Derivative =
      2. * std::pow(x1, x2) * std::log(x1) * std::exp(2. * x3);
  double yx3x3Derivative = 4. * std::pow(x1, x2) * std::exp(2. * x3);

  hessian(tapeId3, 3, x, H);

  BOOST_TEST(yx1x1Derivative == H[0][0], tt::tolerance(tol));
  BOOST_TEST(yx2x1Derivative == H[1][0], tt::tolerance(tol));
  BOOST_TEST(yx2x2Derivative == H[1][1], tt::tolerance(tol));
  BOOST_TEST(yx3x1Derivative == H[2][0], tt::tolerance(tol));
  BOOST_TEST(yx3x2Derivative == H[2][1], tt::tolerance(tol));
  BOOST_TEST(yx3x3Derivative == H[2][2], tt::tolerance(tol));

  myfree1(x);
  myfree2(yDerivative);
  myfree2(X);
  myfree2(Y);
  myfree2(H);
}

/* Tested function: cosh(2.*x1)*sinh(3.*x2)
 * First derivatives: (2.*sinh(2.*x1)*sinh(3.*x2), cosh(2.*x1)*3.*cosh(3.*x2))
 * Second derivatives: (4.*cosh(2.*x1)*sinh(3.*x2), 6.*sinh(2.*x1)*cosh(3.*x2),
 *                      6.*sinh(2.*x1)*cosh(3.*x2), 9.*cosh(2.*x1)*sinh(3.*x2))
 */
BOOST_AUTO_TEST_CASE(CustomHyperbProd_HOS) {
  double x1 = 2.22, x2 = -2.22;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  double y;
  adouble ay;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay = cosh(2. * ax1) * sinh(3. * ax2);

  ay >>= y;
  trace_off();

  double yprim = std::cosh(2. * x1) * std::sinh(3. * x2);

  double **yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = 2. * std::sinh(2. * x1) * std::sinh(3. * x2);
  yDerivative[0][1] = 3. * std::cosh(2. * x1) * std::cosh(3. * x2) +
                      0.5 * 4. * std::cosh(2. * x1) * std::sinh(3. * x2);

  double *x;
  x = myalloc1(2);
  x[0] = x1;
  x[1] = x2;

  double **X;
  X = myalloc2(2, 2);
  X[0][0] = 1.;
  X[0][1] = 0.;
  X[1][0] = 0.;
  X[1][1] = 1.;

  double **Y;
  Y = myalloc2(1, 2);

  hos_forward(tapeId3, 1, 2, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double **H;
  H = myalloc2(2, 2);

  double yx1x1Derivative = 4. * std::cosh(2. * x1) * std::sinh(3. * x2);
  double yx1x2Derivative = 6. * std::sinh(2. * x1) * std::cosh(3. * x2);
  double yx2x2Derivative = 9. * std::cosh(2. * x1) * std::sinh(3. * x2);

  hessian(tapeId3, 2, x, H);

  BOOST_TEST(yx1x1Derivative == H[0][0], tt::tolerance(tol));
  BOOST_TEST(yx1x2Derivative == H[1][0], tt::tolerance(tol));
  BOOST_TEST(yx2x2Derivative == H[1][1], tt::tolerance(tol));

  myfree1(x);
  myfree2(yDerivative);
  myfree2(X);
  myfree2(Y);
  myfree2(H);
}

/* Tested function: pow(sin(x1), cos(x2))
 * First derivatives: (pow(sin(x1), cos(x2))*cos(x2)*cos(x1)/sin(x1),
 *                     -pow(sin(x1), cos(x2))*sin(x2)*log(sin(x1))
 *                    )
 * Second derivatives: (pow(sin(x1), cos(x2))*cos(x2)
 *                      * (-1 + pow(cos(x1)/sin(x1), 2)*(cos(x2) - 1)),
 *                      -pow(sin(x1), cos(x2))*sin(x2)
 *                      * (cos(x1)/sin(x1) + log(sin(x1))*cos(x2)
 *                         *cos(x1)/sin(x1)),
 *                      -pow(sin(x1), cos(x2))*sin(x2)
 *                      * (cos(x1)/sin(x1) + log(sin(x1))*cos(x2)
 *                         *cos(x1)/sin(x1)),
 *                      pow(sin(x1), cos(x2))*log(sin(x1))
 *                      * (-cos(x2) + pow(sin(x2), 2)*log(sin(x1)))
 *                     )
 */
BOOST_AUTO_TEST_CASE(CustomPowTrig_HOS) {
  double x1 = 0.531, x2 = 3.12;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  double y;
  adouble ay;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay = pow(sin(ax1), cos(ax2));

  ay >>= y;
  trace_off();

  double yprim = std::pow(std::sin(x1), std::cos(x2));

  double **yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = std::pow(std::sin(x1), std::cos(x2)) * std::cos(x2) *
                      std::cos(x1) / std::sin(x1);
  yDerivative[0][1] =
      -std::pow(std::sin(x1), std::cos(x2)) * std::sin(x2) *
          std::log(std::sin(x1)) +
      0.5 * (std::pow(std::sin(x1), std::cos(x2)) * std::cos(x2) *
             (-1 +
              std::pow(std::cos(x1) / std::sin(x1), 2) * (std::cos(x2) - 1)));

  double *x;
  x = myalloc1(2);
  x[0] = x1;
  x[1] = x2;

  double **X;
  X = myalloc2(2, 2);
  X[0][0] = 1.;
  X[0][1] = 0.;
  X[1][0] = 0.;
  X[1][1] = 1.;

  double **Y;
  Y = myalloc2(1, 2);

  hos_forward(tapeId3, 1, 2, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double **H;
  H = myalloc2(2, 2);

  double yx1x1Derivative =
      std::pow(std::sin(x1), std::cos(x2)) * std::cos(x2) *
      (-1 + std::pow(std::cos(x1) / std::sin(x1), 2) * (std::cos(x2) - 1));
  double yx1x2Derivative =
      -std::pow(std::sin(x1), std::cos(x2)) * std::sin(x2) *
      (std::cos(x1) / std::sin(x1) +
       std::log(std::sin(x1)) * std::cos(x2) * std::cos(x1) / std::sin(x1));
  double yx2x2Derivative =
      std::pow(std::sin(x1), std::cos(x2)) * std::log(std::sin(x1)) *
      (-std::cos(x2) + std::pow(std::sin(x2), 2) * std::log(std::sin(x1)));

  hessian(tapeId3, 2, x, H);

  BOOST_TEST(yx1x1Derivative == H[0][0], tt::tolerance(tol));
  BOOST_TEST(yx1x2Derivative == H[1][0], tt::tolerance(tol));
  BOOST_TEST(yx2x2Derivative == H[1][1], tt::tolerance(tol));

  myfree1(x);
  myfree2(yDerivative);
  myfree2(X);
  myfree2(Y);
  myfree2(H);
}

/* Tested function: pow(x1, x2)
 * First derivatives: (x2*pow(x1, x2 - 1), pow(x1, x2)*log(x1)
 *                    )
 * Second derivatives: (x2*(x2 - 1)*pow(x1, x2 - 2),
 *                      pow(x1, x2 - 1)*(1 + x2*log(x1)),
 *                      pow(x1, x2 - 1)*(1 + x2*log(x1)),
 *                      pow(x1, x2)*pow(log(x1), 2)
 *                     )
 */
BOOST_AUTO_TEST_CASE(CustomPow_HOS) {
  double x1 = 1.04, x2 = -2.01;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  double y;
  adouble ay;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay = pow(ax1, ax2);

  ay >>= y;
  trace_off();

  double yprim = std::pow(x1, x2);

  double **yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = x2 * std::pow(x1, x2 - 1);
  yDerivative[0][1] = std::pow(x1, x2) * std::log(x1) +
                      0.5 * x2 * (x2 - 1) * std::pow(x1, x2 - 2);

  double *x;
  x = myalloc1(2);
  x[0] = x1;
  x[1] = x2;

  double **X;
  X = myalloc2(2, 2);
  X[0][0] = 1.;
  X[0][1] = 0.;
  X[1][0] = 0.;
  X[1][1] = 1.;

  double **Y;
  Y = myalloc2(1, 2);

  hos_forward(tapeId3, 1, 2, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double **H;
  H = myalloc2(2, 2);

  double yx1x1Derivative = x2 * (x2 - 1) * pow(x1, x2 - 2);
  double yx1x2Derivative = std::pow(x1, x2 - 1) * (1 + x2 * std::log(x1));
  double yx2x2Derivative = std::pow(x1, x2) * std::pow(std::log(x1), 2);

  hessian(tapeId3, 2, x, H);

  BOOST_TEST(yx1x1Derivative == H[0][0], tt::tolerance(tol));
  BOOST_TEST(yx1x2Derivative == H[1][0], tt::tolerance(tol));
  BOOST_TEST(yx2x2Derivative == H[1][1], tt::tolerance(tol));

  myfree1(x);
  myfree2(yDerivative);
  myfree2(X);
  myfree2(Y);
  myfree2(H);
}

/* Tested function: exp(x1 + 3.*x2 +5.*x3 + 7.*x4)
 * First derivatives: (exp(x1 + 3.*x2 +5.*x3 + 7.*x4),
 *                     3.*exp(x1 + 3.*x2 +5.*x3 + 7.*x4)
 *                     5.*exp(x1 + 3.*x2 +5.*x3 + 7.*x4)
 *                     7.*exp(x1 + 3.*x2 +5.*x3 + 7.*x4)
 *                    )
 * Second derivatives: (exp(x1 + 3.*x2 +5.*x3 + 7.*x4),
 *                      3.*exp(x1 + 3.*x2 +5.*x3 + 7.*x4),
 *                      5.*exp(x1 + 3.*x2 +5.*x3 + 7.*x4),
 *                      7.*exp(x1 + 3.*x2 +5.*x3 + 7.*x4),
 *                      3.*exp(x1 + 3.*x2 +5.*x3 + 7.*x4),
 *                      9.*exp(x1 + 3.*x2 +5.*x3 + 7.*x4),
 *                      15.*exp(x1 + 3.*x2 +5.*x3 + 7.*x4),
 *                      21.*exp(x1 + 3.*x2 +5.*x3 + 7.*x4),
 *                      5.*exp(x1 + 3.*x2 +5.*x3 + 7.*x4),
 *                      15.*exp(x1 + 3.*x2 +5.*x3 + 7.*x4),
 *                      25.*exp(x1 + 3.*x2 +5.*x3 + 7.*x4),
 *                      35.*exp(x1 + 3.*x2 +5.*x3 + 7.*x4),
 *                      7.*exp(x1 + 3.*x2 +5.*x3 + 7.*x4),
 *                      21.*exp(x1 + 3.*x2 +5.*x3 + 7.*x4),
 *                      35.*exp(x1 + 3.*x2 +5.*x3 + 7.*x4),
 *                      49.*exp(x1 + 3.*x2 +5.*x3 + 7.*x4)
 *                     )
 */
BOOST_AUTO_TEST_CASE(CustomExpSum_HOS) {
  double x1 = -1.1, x2 = -4.53, x3 = 3.03, x4 = 0.;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ax4;
  double y;
  adouble ay;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;
  ax4 <<= x4;

  ay = exp(ax1 + 3. * ax2 + 5. * ax3 + 7. * ax4);

  ay >>= y;
  trace_off();

  double yprim = std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4);

  double **yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] =
      std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4) * (1. + 0.1 * 3. - 0.01 * 7.);
  yDerivative[0][1] = std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4) *
                      (3. + 0.2 * 5. +
                       0.5 * (1. + 0.3 - 0.07 + 0.1 * (3. + 0.9 - 0.21) -
                              0.01 * (7. + 2.1 - 0.49)));

  double *x;
  x = myalloc1(4);
  x[0] = x1;
  x[1] = x2;
  x[2] = x3;
  x[3] = x4;

  double **X;
  X = myalloc2(4, 2);
  X[0][0] = 1.;
  X[1][0] = 0.1;
  X[2][0] = 0.;
  X[3][0] = -0.01;
  X[0][1] = 0.;
  X[1][1] = 1.;
  X[2][1] = 0.2;
  X[3][1] = 0.;

  double **Y;
  Y = myalloc2(1, 2);

  hos_forward(tapeId3, 1, 4, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double **H;
  H = myalloc2(4, 4);

  double yx1x1Derivative = std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4);
  double yx2x1Derivative = 3. * std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4);
  double yx3x1Derivative = 5. * std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4);
  double yx4x1Derivative = 7. * std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4);
  double yx2x2Derivative = 9. * std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4);
  double yx3x2Derivative = 15. * std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4);
  double yx4x2Derivative = 21. * std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4);
  double yx3x3Derivative = 25. * std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4);
  double yx4x3Derivative = 35. * std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4);
  double yx4x4Derivative = 49. * std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4);

  hessian(tapeId3, 4, x, H);

  BOOST_TEST(yx1x1Derivative == H[0][0], tt::tolerance(tol));
  BOOST_TEST(yx2x1Derivative == H[1][0], tt::tolerance(tol));
  BOOST_TEST(yx2x2Derivative == H[1][1], tt::tolerance(tol));
  BOOST_TEST(yx3x1Derivative == H[2][0], tt::tolerance(tol));
  BOOST_TEST(yx3x2Derivative == H[2][1], tt::tolerance(tol));
  BOOST_TEST(yx3x3Derivative == H[2][2], tt::tolerance(tol));
  BOOST_TEST(yx4x1Derivative == H[3][0], tt::tolerance(tol));
  BOOST_TEST(yx4x2Derivative == H[3][1], tt::tolerance(tol));
  BOOST_TEST(yx4x3Derivative == H[3][2], tt::tolerance(tol));
  BOOST_TEST(yx4x4Derivative == H[3][3], tt::tolerance(tol));

  myfree1(x);
  myfree2(yDerivative);
  myfree2(X);
  myfree2(Y);
  myfree2(H);
}

/* Tested function: exp(tanh(x1)*erf(x2))
 * First derivatives: (exp(tanh(x1)*erf(x2))*(1 - pow(tanh(x1), 2))*erf(x2),
 *                     exp(tanh(x1)*erf(x2))*tanh(x1)*exp(-x2*x2)
 *                     *2/sqrt(acos(-1))
 *                    )
 * Second derivatives: (exp(tanh(x1)*erf(x2))*(1 - pow(tanh(x1), 2))*erf(x2)
 *                      *((1 - pow(tanh(x1), 2))*erf(x2) - 2*tanh(x1)),
 *                      exp(tanh(x1)*erf(x2))*exp(-x2*x2)
 *                      *(1 - pow(tanh(x1), 2))*2/sqrt(acos(-1))
 *                      *(1 + tanh(x1)*erf(x2)),
 *                      exp(tanh(x1)*erf(x2))*exp(-x2*x2)
 *                      *(1 - pow(tanh(x2), 2))*2/sqrt(acos(-1))
 *                      *(1 + tanh(x1)*erf(x2)),
 *                      exp(tanh(x1)*erf(x2))*tanh(x1)*exp(-2*x2*x2)
 *                      *(4*tanh(x1)/acos(-1) - 4*x2*exp(x2*x2)/sqrt(acos(-1)))
 *                     )
 */
BOOST_AUTO_TEST_CASE(CustomHypErf_HOS) {
  double x1 = 5.55, x2 = 9.99;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  double y;
  adouble ay;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay = exp(tanh(ax1) * erf(ax2));

  ay >>= y;
  trace_off();

  double yprim = std::exp(std::tanh(x1) * std::erf(x2));

  double **yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = std::exp(std::tanh(x1) * std::erf(x2)) *
                      (1 - std::pow(std::tanh(x1), 2)) * std::erf(x2);
  yDerivative[0][1] =
      std::exp(std::tanh(x1) * std::erf(x2)) * std::tanh(x1) *
          std::exp(-x2 * x2) * 2 / std::sqrt(std::acos(-1)) +
      0.5 * std::exp(std::tanh(x1) * std::erf(x2)) *
          (1 - std::pow(std::tanh(x1), 2)) * std::erf(x2) *
          ((1 - std::pow(std::tanh(x1), 2)) * std::erf(x2) - 2 * std::tanh(x1));

  double *x;
  x = myalloc1(2);
  x[0] = x1;
  x[1] = x2;

  double **X;
  X = myalloc2(2, 2);
  X[0][0] = 1.;
  X[0][1] = 0.;
  X[1][0] = 0.;
  X[1][1] = 1.;

  double **Y;
  Y = myalloc2(1, 2);

  hos_forward(tapeId3, 1, 2, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double **H;
  H = myalloc2(2, 2);

  double yx1x1Derivative =
      std::exp(std::tanh(x1) * std::erf(x2)) *
      (1 - std::pow(std::tanh(x1), 2)) * std::erf(x2) *
      ((1 - std::pow(std::tanh(x1), 2)) * std::erf(x2) - 2 * std::tanh(x1));
  double yx1x2Derivative =
      std::exp(std::tanh(x1) * std::erf(x2)) * std::exp(-x2 * x2) *
      (1 - std::pow(std::tanh(x1), 2)) * 2 / std::sqrt(std::acos(-1)) *
      (1 + std::tanh(x1) * std::erf(x2));
  double yx2x2Derivative =
      std::exp(std::tanh(x1) * std::erf(x2)) * std::tanh(x1) *
      std::exp(-2 * x2 * x2) *
      (4 * std::tanh(x1) / std::acos(-1) -
       4 * x2 * std::exp(x2 * x2) / std::sqrt(std::acos(-1)));

  hessian(tapeId3, 2, x, H);

  BOOST_TEST(yx1x1Derivative == H[0][0], tt::tolerance(tol));
  BOOST_TEST(yx1x2Derivative == H[1][0], tt::tolerance(tol));
  BOOST_TEST(yx2x2Derivative == H[1][1], tt::tolerance(tol));

  myfree1(x);
  myfree2(yDerivative);
  myfree2(X);
  myfree2(Y);
  myfree2(H);
}

/* Tested function: (pow(cosh(x1), 2) - pow(sinh(x1), 2))*atan(x2)
 * First derivatives: (0, 1./(1. + x2*x2)
 *                    )
 * Second derivatives: (0, 0,
 *                      0, -2.*x2/pow(1. + x2*x2, 2)
 *                     )
 */
BOOST_AUTO_TEST_CASE(CustomHypAtan_HOS) {
  double x1 = 7.19, x2 = -4.32;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  double y;
  adouble ay;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay = (pow(cosh(ax1), 2) - pow(sinh(ax1), 2)) * atan(ax2);

  ay >>= y;
  trace_off();

  double yprim =
      (std::pow(std::cosh(x1), 2) - std::pow(std::sinh(x1), 2)) * std::atan(x2);

  double **yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = 0.;
  yDerivative[0][1] = 1. / (1. + x2 * x2) + 0.5 * 0.;

  double *x;
  x = myalloc1(2);
  x[0] = x1;
  x[1] = x2;

  double **X;
  X = myalloc2(2, 2);
  X[0][0] = 1.;
  X[0][1] = 0.;
  X[1][0] = 0.;
  X[1][1] = 1.;

  double **Y;
  Y = myalloc2(1, 2);

  hos_forward(tapeId3, 1, 2, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double **H;
  H = myalloc2(2, 2);

  double yx1x1Derivative = 0.;
  double yx1x2Derivative = 0.;
  double yx2x2Derivative = -2. * x2 / std::pow(1. + x2 * x2, 2);

  hessian(tapeId3, 2, x, H);

  BOOST_TEST(yx1x1Derivative == H[0][0], tt::tolerance(tol));
  BOOST_TEST(yx1x2Derivative == H[1][0], tt::tolerance(tol));
  BOOST_TEST(yx2x2Derivative == H[1][1], tt::tolerance(tol));

  myfree1(x);
  myfree2(yDerivative);
  myfree2(X);
  myfree2(Y);
  myfree2(H);
}

/* Tested function: 1. + x1 + x1*x1 + x2*x2 + x2*x2*x2 + x3*x3*x3 + x3*x3*x3*x3
 * First derivatives: (1. + 2.*x1, 2.*x2 + 3.*x2*x2, 3.*x3*x3 + 4.*x3*x3*x3
 *                    )
 * Second derivatives: (2., 0., 0.,
 *                      0., 2. + 6.*x2, 0.,
 *                      0., 0., 6.*x3 + 12.*x3*x3
 *                     )
 */
BOOST_AUTO_TEST_CASE(CustomLongSum_HOS) {
  double x1 = 99.99, x2 = std::exp(-0.44), x3 = std::sqrt(2);
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  adouble ax3;
  double y;
  adouble ay;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ay = 1. + ax1 + ax1 * ax1 + ax2 * ax2 + ax2 * ax2 * ax2 + ax3 * ax3 * ax3 +
       ax3 * ax3 * ax3 * ax3;

  ay >>= y;
  trace_off();

  double yprim = 1. + x1 + x1 * x1 + x2 * x2 + x2 * x2 * x2 + x3 * x3 * x3 +
                 x3 * x3 * x3 * x3;

  double **yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = 1. + 2. * x1 - 0.01 * (3. * x3 * x3 + 4. * x3 * x3 * x3);
  yDerivative[0][1] = 0.3 * (2. * x2 + 3. * x2 * x2) +
                      0.5 * (2. + 0.01 * 0.01 * (6. * x3 + 12. * x3 * x3));

  double *x;
  x = myalloc1(3);
  x[0] = x1;
  x[1] = x2;
  x[2] = x3;

  double **X;
  X = myalloc2(3, 2);
  X[0][0] = 1.;
  X[1][0] = 0.;
  X[2][0] = -0.01;
  X[0][1] = 0.;
  X[1][1] = 0.3;
  X[2][1] = 0.;

  double **Y;
  Y = myalloc2(1, 2);

  hos_forward(tapeId3, 1, 3, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double **H;
  H = myalloc2(3, 3);

  double yx1x1Derivative = 2.;
  double yx2x1Derivative = 0.;
  double yx3x1Derivative = 0.;
  double yx2x2Derivative = 2. + 6. * x2;
  double yx3x2Derivative = 0.;
  double yx3x3Derivative = 6. * x3 + 12. * x3 * x3;

  hessian(tapeId3, 3, x, H);

  BOOST_TEST(yx1x1Derivative == H[0][0], tt::tolerance(tol));
  BOOST_TEST(yx2x1Derivative == H[1][0], tt::tolerance(tol));
  BOOST_TEST(yx2x2Derivative == H[1][1], tt::tolerance(tol));
  BOOST_TEST(yx3x1Derivative == H[2][0], tt::tolerance(tol));
  BOOST_TEST(yx3x2Derivative == H[2][1], tt::tolerance(tol));
  BOOST_TEST(yx3x3Derivative == H[2][2], tt::tolerance(tol));

  myfree1(x);
  myfree2(yDerivative);
  myfree2(X);
  myfree2(Y);
  myfree2(H);
}

/* Tested function: exp(x1)*sqrt(2.*x2)*pow(x3, 2)
 * First derivatives: (exp(x1)*sqrt(2.*x2)*pow(x3, 2),
 *                     exp(x1)*pow(x3, 2)/sqrt(2.*x2),
 *                     exp(x1)*sqrt(2.*x2)*2.*x3
 *                    )
 * Second derivatives: (exp(x1)*sqrt(2.*x2)*pow(x3, 2),
 *                      exp(x1)*pow(x3, 2)/sqrt(2.*x2),
 *                      exp(x1)*sqrt(2.*x2)*2.*x3,
 *                      exp(x1)*pow(x3, 2)/sqrt(2.*x2),
 *                      -exp(x1)*pow(x3, 2)/pow(sqrt(2.*x2, 3)),
 *                      exp(x1)*2.*x3/sqrt(2.*x2),
 *                      exp(x1)*sqrt(2.*x2)*2.*x3,
 *                      exp(x1)*2.*x3/sqrt(2.*x2),
 *                      2.*exp(x1)*sqrt(2.*x2)
 *                     )
 */
BOOST_AUTO_TEST_CASE(CustomExpSqrt_HOS) {
  double x1 = -0.77, x2 = 10.01, x3 = 0.99;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  adouble ax3;
  double y;
  adouble ay;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ay = exp(ax1) * sqrt(2. * ax2) * pow(ax3, 2);

  ay >>= y;
  trace_off();

  double yprim = std::exp(x1) * std::sqrt(2. * x2) * std::pow(x3, 2);

  double **yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = std::exp(x1) * std::sqrt(2. * x2) * std::pow(x3, 2);
  yDerivative[0][1] = 2. * exp(x1) * pow(x3, 2) / sqrt(2. * x2) +
                      0.5 * exp(x1) * sqrt(2. * x2) * pow(x3, 2);

  double *x;
  x = myalloc1(3);
  x[0] = x1;
  x[1] = x2;
  x[2] = x3;

  double **X;
  X = myalloc2(3, 2);
  X[0][0] = 1.;
  X[1][0] = 0.;
  X[2][0] = 0.;
  X[0][1] = 0.;
  X[1][1] = 2.;
  X[2][1] = 0.;

  double **Y;
  Y = myalloc2(1, 2);

  hos_forward(tapeId3, 1, 3, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double **H;
  H = myalloc2(3, 3);

  double yx1x1Derivative = std::exp(x1) * std::sqrt(2. * x2) * std::pow(x3, 2);
  double yx2x1Derivative = std::exp(x1) * std::pow(x3, 2) / std::sqrt(2. * x2);
  double yx3x1Derivative = std::exp(x1) * std::sqrt(2. * x2) * 2. * x3;
  double yx2x2Derivative =
      -std::exp(x1) * std::pow(x3, 2) / std::pow(std::sqrt(2. * x2), 3);
  double yx3x2Derivative = std::exp(x1) * 2. * x3 / std::sqrt(2. * x2);
  double yx3x3Derivative = 2. * std::exp(x1) * std::sqrt(2. * x2);

  hessian(tapeId3, 3, x, H);

  BOOST_TEST(yx1x1Derivative == H[0][0], tt::tolerance(tol));
  BOOST_TEST(yx2x1Derivative == H[1][0], tt::tolerance(tol));
  BOOST_TEST(yx2x2Derivative == H[1][1], tt::tolerance(tol));
  BOOST_TEST(yx3x1Derivative == H[2][0], tt::tolerance(tol));
  BOOST_TEST(yx3x2Derivative == H[2][1], tt::tolerance(tol));
  BOOST_TEST(yx3x3Derivative == H[2][2], tt::tolerance(tol));

  myfree1(x);
  myfree2(yDerivative);
  myfree2(X);
  myfree2(Y);
  myfree2(H);
}

/* Tested function: 2.*acosh(cosh(x1*x1))*atanh(x2)
 * First derivatives: (4.*x1*atanh(x2), 2.*x1*x1/(1. - x2*x2)
 *                    )
 * Second derivatives: (4.*atanh(x2), 4.*x1/(1. - x2*x2),
 *                      4.*x1/(1. - x2*x2), 4.*x1*x1*x2/pow(1. - x2*x2, 2)
 *                     )
 */
BOOST_AUTO_TEST_CASE(CustomInvHyperb_HOS) {
  double x1 = -3.03, x2 = 0.11;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  double y;
  adouble ay;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay = 2. * acosh(cosh(ax1 * ax1)) * atanh(ax2);

  ay >>= y;
  trace_off();

  double yprim = 2. * std::acosh(std::cosh(x1 * x1)) * std::atanh(x2);

  double **yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = 4. * x1 * std::atanh(x2);
  yDerivative[0][1] = 3. * x1 * x1 / (1. - x2 * x2) + 2. * std::atanh(x2);

  double *x;
  x = myalloc1(2);
  x[0] = x1;
  x[1] = x2;

  double **X;
  X = myalloc2(2, 2);
  X[0][0] = 1.;
  X[0][1] = 0.;
  X[1][0] = 0.;
  X[1][1] = 1.5;

  double **Y;
  Y = myalloc2(1, 2);

  hos_forward(tapeId3, 1, 2, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double **H;
  H = myalloc2(2, 2);

  double yx1x1Derivative = 4. * std::atanh(x2);
  double yx1x2Derivative = 4. * x1 / (1. - x2 * x2);
  double yx2x2Derivative = 4. * x1 * x1 * x2 / std::pow(1. - x2 * x2, 2);

  hessian(tapeId3, 2, x, H);

  BOOST_TEST(yx1x1Derivative == H[0][0], tt::tolerance(tol));
  BOOST_TEST(yx1x2Derivative == H[1][0], tt::tolerance(tol));
  BOOST_TEST(yx2x2Derivative == H[1][1], tt::tolerance(tol));

  myfree1(x);
  myfree2(yDerivative);
  myfree2(X);
  myfree2(Y);
  myfree2(H);
}

/* Tested function: fmax(fmin(x1, x2), fabs(x3))*x4
 * First derivatives: (0., 0., -x4, -x3)
 * Second derivatives: (0., 0., 0., 0.,
 *                      0., 0., 0., 0.,
 *                      0., 0., 0., -1.,
 *                      0., 0., -1., 0.)
 */
BOOST_AUTO_TEST_CASE(CustomFminFmaxFabs_HOS) {
  double x1 = 1., x2 = 2.5, x3 = -4.5, x4 = -1.;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ax4;
  double y;
  adouble ay;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;
  ax4 <<= x4;

  ay = fmax(fmin(ax1, ax2), fabs(ax3)) * ax4;

  ay >>= y;
  trace_off();

  double yprim = std::fmax(std::fmin(x1, x2), std::fabs(x3)) * x4;

  double **yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = 0.01 * x3;
  yDerivative[0][1] = -0.2 * x4 + 0.5 * 0.;

  double *x;
  x = myalloc1(4);
  x[0] = x1;
  x[1] = x2;
  x[2] = x3;
  x[3] = x4;

  double **X;
  X = myalloc2(4, 2);
  X[0][0] = 1.;
  X[1][0] = 0.1;
  X[2][0] = 0.;
  X[3][0] = -0.01;
  X[0][1] = 0.;
  X[1][1] = 1.;
  X[2][1] = 0.2;
  X[3][1] = 0.;

  double **Y;
  Y = myalloc2(1, 2);

  hos_forward(tapeId3, 1, 4, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double **H;
  H = myalloc2(4, 4);

  double yx1x1Derivative = 0.;
  double yx2x1Derivative = 0.;
  double yx3x1Derivative = 0.;
  double yx4x1Derivative = 0.;
  double yx2x2Derivative = 0.;
  double yx3x2Derivative = 0.;
  double yx4x2Derivative = 0.;
  double yx3x3Derivative = 0.;
  double yx4x3Derivative = -1.;
  double yx4x4Derivative = 0.;

  hessian(tapeId3, 4, x, H);

  BOOST_TEST(yx1x1Derivative == H[0][0], tt::tolerance(tol));
  BOOST_TEST(yx2x1Derivative == H[1][0], tt::tolerance(tol));
  BOOST_TEST(yx2x2Derivative == H[1][1], tt::tolerance(tol));
  BOOST_TEST(yx3x1Derivative == H[2][0], tt::tolerance(tol));
  BOOST_TEST(yx3x2Derivative == H[2][1], tt::tolerance(tol));
  BOOST_TEST(yx3x3Derivative == H[2][2], tt::tolerance(tol));
  BOOST_TEST(yx4x1Derivative == H[3][0], tt::tolerance(tol));
  BOOST_TEST(yx4x2Derivative == H[3][1], tt::tolerance(tol));
  BOOST_TEST(yx4x3Derivative == H[3][2], tt::tolerance(tol));
  BOOST_TEST(yx4x4Derivative == H[3][3], tt::tolerance(tol));

  myfree1(x);
  myfree2(yDerivative);
  myfree2(X);
  myfree2(Y);
  myfree2(H);
}

/* Tested function: max(min(x1, x2), abs(x3))*x4
 * First derivatives: (0., 0., -x4, -x3)
 * Second derivatives: (0., 0., 0., 0.,
 *                      0., 0., 0., 0.,
 *                      0., 0., 0., -1.,
 *                      0., 0., -1., 0.)
 */
BOOST_AUTO_TEST_CASE(CustomMinMaxAbs_HOS) {
  double x1 = 1., x2 = 2.5, x3 = -4.5, x4 = -1.;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ax4;
  double y;
  adouble ay;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;
  ax4 <<= x4;

  ay = max(min(ax1, ax2), abs(ax3)) * ax4;

  ay >>= y;
  trace_off();

  double yprim = std::max(std::min(x1, x2), std::abs(x3)) * x4;

  double **yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = 0.01 * x3;
  yDerivative[0][1] = -0.2 * x4 + 0.5 * 0.;

  double *x;
  x = myalloc1(4);
  x[0] = x1;
  x[1] = x2;
  x[2] = x3;
  x[3] = x4;

  double **X;
  X = myalloc2(4, 2);
  X[0][0] = 1.;
  X[1][0] = 0.1;
  X[2][0] = 0.;
  X[3][0] = -0.01;
  X[0][1] = 0.;
  X[1][1] = 1.;
  X[2][1] = 0.2;
  X[3][1] = 0.;

  double **Y;
  Y = myalloc2(1, 2);

  hos_forward(tapeId3, 1, 4, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double **H;
  H = myalloc2(4, 4);

  double yx1x1Derivative = 0.;
  double yx2x1Derivative = 0.;
  double yx3x1Derivative = 0.;
  double yx4x1Derivative = 0.;
  double yx2x2Derivative = 0.;
  double yx3x2Derivative = 0.;
  double yx4x2Derivative = 0.;
  double yx3x3Derivative = 0.;
  double yx4x3Derivative = -1.;
  double yx4x4Derivative = 0.;

  hessian(tapeId3, 4, x, H);

  BOOST_TEST(yx1x1Derivative == H[0][0], tt::tolerance(tol));
  BOOST_TEST(yx2x1Derivative == H[1][0], tt::tolerance(tol));
  BOOST_TEST(yx2x2Derivative == H[1][1], tt::tolerance(tol));
  BOOST_TEST(yx3x1Derivative == H[2][0], tt::tolerance(tol));
  BOOST_TEST(yx3x2Derivative == H[2][1], tt::tolerance(tol));
  BOOST_TEST(yx3x3Derivative == H[2][2], tt::tolerance(tol));
  BOOST_TEST(yx4x1Derivative == H[3][0], tt::tolerance(tol));
  BOOST_TEST(yx4x2Derivative == H[3][1], tt::tolerance(tol));
  BOOST_TEST(yx4x3Derivative == H[3][2], tt::tolerance(tol));
  BOOST_TEST(yx4x4Derivative == H[3][3], tt::tolerance(tol));

  myfree1(x);
  myfree2(yDerivative);
  myfree2(X);
  myfree2(Y);
  myfree2(H);
}

/* Tested function: 3.*asin(sin(x1 + x2))*sin(x3)*cos(x4)
 * First derivatives: (3.*sin(x3)*cos(x4), 3.*sin(x3)*cos(x4),
 *                     3.*(x1 + x2)*cos(x3)*cos(x4),
 *                     -3.*(x1 + x2)*sin(x3)*sin(x4)
 *                    )
 * Second derivatives: (0., 0., 3*cos(x3)*cos(x4), -3.*sin(x3)*sin(x4),
 *                      0., 0., 3*cos(x3)*cos(x4), -3.*sin(x3)*sin(x4),
 *                      3.*cos(x3)*cos(x4), 3.*cos(x3)*cos(x4),
 *                      -3.*(x1 + x2)*sin(x3)*cos(x4),
 *                      -3.*(x1 + x2)*cos(x3)*sin(x4),
 *                      -3.*sin(x3)*sin(x4), -3.*sin(x3)*sin(x4),
 *                      -3.*(x1 + x2)*cos(x3)*sin(x4),
 *                      -3.*(x1 + x2)*sin(x3)*cos(x4)
 *                     )
 */
BOOST_AUTO_TEST_CASE(CustomInvTrig_HOS) {
  double x1 = 0.11, x2 = 0.33, x3 = 0.1 * std::acos(0.), x4 = std::exp(-1.);
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ax4;
  double y;
  adouble ay;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;
  ax4 <<= x4;

  ay = 3. * asin(sin(ax1 + ax2)) * sin(ax3) * cos(ax4);

  ay >>= y;
  trace_off();

  double yprim =
      3. * std::asin(std::sin(x1 + x2)) * std::sin(x3) * std::cos(x4);

  double **yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = 3. * std::sin(x3) * std::cos(x4) +
                      0.1 * 3. * std::sin(x3) * std::cos(x4) +
                      0.01 * 3. * (x1 + x2) * std::sin(x3) * std::sin(x4);
  yDerivative[0][1] = 3. * std::sin(x3) * std::cos(x4) +
                      0.2 * 3. * (x1 + x2) * std::cos(x3) * std::cos(x4) -
                      0.5 * 0.01 *
                          (-2. * 3. * std::sin(x3) * std::sin(x4) -
                           0.2 * 3. * std::sin(x3) * std::sin(x4) +
                           0.01 * 3. * (x1 + x2) * std::sin(x3) * std::cos(x4));

  double *x;
  x = myalloc1(4);
  x[0] = x1;
  x[1] = x2;
  x[2] = x3;
  x[3] = x4;

  double **X;
  X = myalloc2(4, 2);
  X[0][0] = 1.;
  X[1][0] = 0.1;
  X[2][0] = 0.;
  X[3][0] = -0.01;
  X[0][1] = 0.;
  X[1][1] = 1.;
  X[2][1] = 0.2;
  X[3][1] = 0.;

  double **Y;
  Y = myalloc2(1, 2);

  hos_forward(tapeId3, 1, 4, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double **H;
  H = myalloc2(4, 4);

  double yx1x1Derivative = 0.;
  double yx2x1Derivative = 0.;
  double yx3x1Derivative = 3 * std::cos(x3) * std::cos(x4);
  double yx4x1Derivative = -3. * sin(x3) * sin(x4);
  double yx2x2Derivative = 0.;
  double yx3x2Derivative = 3. * std::cos(x3) * std::cos(x4);
  double yx4x2Derivative = -3. * std::sin(x3) * std::sin(x4);
  double yx3x3Derivative = -3. * (x1 + x2) * std::sin(x3) * std::cos(x4);
  double yx4x3Derivative = -3. * (x1 + x2) * std::cos(x3) * std::sin(x4);
  double yx4x4Derivative = -3. * (x1 + x2) * std::sin(x3) * std::cos(x4);

  hessian(tapeId3, 4, x, H);

  BOOST_TEST(yx1x1Derivative == H[0][0], tt::tolerance(tol));
  BOOST_TEST(yx2x1Derivative == H[1][0], tt::tolerance(tol));
  BOOST_TEST(yx2x2Derivative == H[1][1], tt::tolerance(tol));
  BOOST_TEST(yx3x1Derivative == H[2][0], tt::tolerance(tol));
  BOOST_TEST(yx3x2Derivative == H[2][1], tt::tolerance(tol));
  BOOST_TEST(yx3x3Derivative == H[2][2], tt::tolerance(tol));
  BOOST_TEST(yx4x1Derivative == H[3][0], tt::tolerance(tol));
  BOOST_TEST(yx4x2Derivative == H[3][1], tt::tolerance(tol));
  BOOST_TEST(yx4x3Derivative == H[3][2], tt::tolerance(tol));
  BOOST_TEST(yx4x4Derivative == H[3][3], tt::tolerance(tol));

  myfree1(x);
  myfree2(yDerivative);
  myfree2(X);
  myfree2(Y);
  myfree2(H);
}

/* Tested function: atan(x1)*asin(x2)
 * First derivatives: (asin(x2)/(1. + x1*x1), atan(x1)/sqrt(1. - x2*x2)
 *                    )
 * Second derivatives: (-2.*x1*asin(x2)/pow(1. + x1*x1, 2),
 *                      1./((1. + x1*x1)*sqrt(1. - x2*x2)),
 *                      1./((1. + x1*x1)*sqrt(1. - x2*x2)),
 *                      atan(x1)*x2/pow(sqrt(1. - x2*x2), 3)
 *                     )
 */
BOOST_AUTO_TEST_CASE(CustomInvTrig2_HOS) {
  double x1 = 0.53, x2 = -0.01;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  double y;
  adouble ay;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay = atan(ax1) * asin(ax2);

  ay >>= y;
  trace_off();

  double yprim = std::atan(x1) * std::asin(x2);

  double **yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = std::asin(x2) / (1. + x1 * x1);
  yDerivative[0][1] = 1.5 * std::atan(x1) / std::sqrt(1. - x2 * x2) -
                      0.5 * 2. * x1 * std::asin(x2) / std::pow(1. + x1 * x1, 2);

  double *x;
  x = myalloc1(2);
  x[0] = x1;
  x[1] = x2;

  double **X;
  X = myalloc2(2, 2);
  X[0][0] = 1.;
  X[0][1] = 0.;
  X[1][0] = 0.;
  X[1][1] = 1.5;

  double **Y;
  Y = myalloc2(1, 2);

  hos_forward(tapeId3, 1, 2, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double **H;
  H = myalloc2(2, 2);

  double yx1x1Derivative = -2. * x1 * std::asin(x2) / std::pow(1. + x1 * x1, 2);
  double yx1x2Derivative = 1. / ((1. + x1 * x1) * std::sqrt(1. - x2 * x2));
  double yx2x2Derivative =
      std::atan(x1) * x2 / std::pow(std::sqrt(1. - x2 * x2), 3);

  hessian(tapeId3, 2, x, H);

  BOOST_TEST(yx1x1Derivative == H[0][0], tt::tolerance(tol));
  BOOST_TEST(yx1x2Derivative == H[1][0], tt::tolerance(tol));
  BOOST_TEST(yx2x2Derivative == H[1][1], tt::tolerance(tol));

  myfree1(x);
  myfree2(yDerivative);
  myfree2(X);
  myfree2(Y);
  myfree2(H);
}

/* Tested function: fmax(fabs(x1*x1), fabs(x2*x2))
 * First derivatives: (2.*x1, 0.
 *                    )
 * Second derivatives: (2., 0., 0., 0.
 *                     )
 */
BOOST_AUTO_TEST_CASE(CustomFabsFmax_HOS) {
  double x1 = 9.9, x2 = -4.7;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  double y;
  adouble ay;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay = fmax(fabs(ax1 * ax1), fabs(ax2 * ax2));

  ay >>= y;
  trace_off();

  double yprim = std::fmax(std::fabs(x1 * x1), std::fabs(x2 * x2));

  double **yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = 2. * x1;
  yDerivative[0][1] = 0.5 * 2.;

  double *x;
  x = myalloc1(2);
  x[0] = x1;
  x[1] = x2;

  double **X;
  X = myalloc2(2, 2);
  X[0][0] = 1.;
  X[0][1] = 0.;
  X[1][0] = 0.;
  X[1][1] = 1.5;

  double **Y;
  Y = myalloc2(1, 2);

  hos_forward(tapeId3, 1, 2, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double **H;
  H = myalloc2(2, 2);

  double yx1x1Derivative = 2.;
  double yx1x2Derivative = 0.;
  double yx2x2Derivative = 0.;

  hessian(tapeId3, 2, x, H);

  BOOST_TEST(yx1x1Derivative == H[0][0], tt::tolerance(tol));
  BOOST_TEST(yx1x2Derivative == H[1][0], tt::tolerance(tol));
  BOOST_TEST(yx2x2Derivative == H[1][1], tt::tolerance(tol));

  myfree1(x);
  myfree2(yDerivative);
  myfree2(X);
  myfree2(Y);
  myfree2(H);
}

/* Tested function: fmax(fabs(x1*x1), fabs(x2*x2))
 * First derivatives: (2.*x1, 0.
 *                    )
 * Second derivatives: (2., 0., 0., 0.
 *                     )
 */
BOOST_AUTO_TEST_CASE(CustomAbsMax_HOS) {
  double x1 = 9.9, x2 = -4.7;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  double y;
  adouble ay;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay = max(abs(ax1 * ax1), abs(ax2 * ax2));

  ay >>= y;
  trace_off();

  double yprim = std::max(std::abs(x1 * x1), std::abs(x2 * x2));

  double **yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = 2. * x1;
  yDerivative[0][1] = 0.5 * 2.;

  double *x;
  x = myalloc1(2);
  x[0] = x1;
  x[1] = x2;

  double **X;
  X = myalloc2(2, 2);
  X[0][0] = 1.;
  X[0][1] = 0.;
  X[1][0] = 0.;
  X[1][1] = 1.5;

  double **Y;
  Y = myalloc2(1, 2);

  hos_forward(tapeId3, 1, 2, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double **H;
  H = myalloc2(2, 2);

  double yx1x1Derivative = 2.;
  double yx1x2Derivative = 0.;
  double yx2x2Derivative = 0.;

  hessian(tapeId3, 2, x, H);

  BOOST_TEST(yx1x1Derivative == H[0][0], tt::tolerance(tol));
  BOOST_TEST(yx1x2Derivative == H[1][0], tt::tolerance(tol));
  BOOST_TEST(yx2x2Derivative == H[1][1], tt::tolerance(tol));

  myfree1(x);
  myfree2(yDerivative);
  myfree2(X);
  myfree2(Y);
  myfree2(H);
}

/* Tested function: fmin(fabs(x1*x1), fabs(x2*x2))
 * First derivatives: (0., 2.*x2
 *                    )
 * Second derivatives: (0., 0., 0., 2.
 *                     )
 */
BOOST_AUTO_TEST_CASE(CustomFabsFmin_HOS) {
  double x1 = 9.9, x2 = -4.7;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  double y;
  adouble ay;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay = fmin(fabs(ax1 * ax1), fabs(ax2 * ax2));

  ay >>= y;
  trace_off();

  double yprim = std::fmin(std::fabs(x1 * x1), std::fabs(x2 * x2));

  double **yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = 0.;
  yDerivative[0][1] = 1.5 * 2. * x2;

  double *x;
  x = myalloc1(2);
  x[0] = x1;
  x[1] = x2;

  double **X;
  X = myalloc2(2, 2);
  X[0][0] = 1.;
  X[0][1] = 0.;
  X[1][0] = 0.;
  X[1][1] = 1.5;

  double **Y;
  Y = myalloc2(1, 2);

  hos_forward(tapeId3, 1, 2, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double **H;
  H = myalloc2(2, 2);

  double yx1x1Derivative = 0.;
  double yx1x2Derivative = 0.;
  double yx2x2Derivative = 2.;

  hessian(tapeId3, 2, x, H);

  BOOST_TEST(yx1x1Derivative == H[0][0], tt::tolerance(tol));
  BOOST_TEST(yx1x2Derivative == H[1][0], tt::tolerance(tol));
  BOOST_TEST(yx2x2Derivative == H[1][1], tt::tolerance(tol));

  myfree1(x);
  myfree2(yDerivative);
  myfree2(X);
  myfree2(Y);
  myfree2(H);
}

/* Tested function: fmax(x1*x1*cos(x2), sin(x1)*cos(x2)*exp(x2))
 * First derivatives: (2.*x1*cos(x2), -x1*x1*sin(x2)
 *                    )
 * Second derivatives: (2.*cos(x2), -2.*x1*sin(x2),
 *                      -2.*x1*sin(x2), -x1*x1*cos(x2)
 *                     )
 */
BOOST_AUTO_TEST_CASE(CustomFmaxTrigExp_HOS) {
  double x1 = 21.07, x2 = 1.5;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  double y;
  adouble ay;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay = fmax(ax1 * ax1 * cos(ax2), sin(ax1) * cos(ax2) * exp(ax2));

  ay >>= y;
  trace_off();

  double yprim = std::fmax(x1 * x1 * std::cos(x2),
                           std::sin(x1) * std::cos(x2) * std::exp(x2));

  double **yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = 2. * x1 * std::cos(x2);
  yDerivative[0][1] = -1.5 * x1 * x1 * std::sin(x2) + 0.5 * 2. * std::cos(x2);

  double *x;
  x = myalloc1(2);
  x[0] = x1;
  x[1] = x2;

  double **X;
  X = myalloc2(2, 2);
  X[0][0] = 1.;
  X[0][1] = 0.;
  X[1][0] = 0.;
  X[1][1] = 1.5;

  double **Y;
  Y = myalloc2(1, 2);

  hos_forward(tapeId3, 1, 2, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double **H;
  H = myalloc2(2, 2);

  double yx1x1Derivative = 2. * std::cos(x2);
  double yx1x2Derivative = -2. * x1 * std::sin(x2);
  double yx2x2Derivative = -x1 * x1 * std::cos(x2);

  hessian(tapeId3, 2, x, H);

  BOOST_TEST(yx1x1Derivative == H[0][0], tt::tolerance(tol));
  BOOST_TEST(yx1x2Derivative == H[1][0], tt::tolerance(tol));
  BOOST_TEST(yx2x2Derivative == H[1][1], tt::tolerance(tol));

  myfree1(x);
  myfree2(yDerivative);
  myfree2(X);
  myfree2(Y);
  myfree2(H);
}

BOOST_AUTO_TEST_CASE(CustomMaxTrigExp_HOS) {
  double x1 = 21.07, x2 = 1.5;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  double y;
  adouble ay;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay = max(ax1 * ax1 * cos(ax2), sin(ax1) * cos(ax2) * exp(ax2));

  ay >>= y;
  trace_off();

  double yprim = std::max(x1 * x1 * std::cos(x2),
                          std::sin(x1) * std::cos(x2) * std::exp(x2));

  double **yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = 2. * x1 * std::cos(x2);
  yDerivative[0][1] = -1.5 * x1 * x1 * std::sin(x2) + 0.5 * 2. * std::cos(x2);

  double *x;
  x = myalloc1(2);
  x[0] = x1;
  x[1] = x2;

  double **X;
  X = myalloc2(2, 2);
  X[0][0] = 1.;
  X[0][1] = 0.;
  X[1][0] = 0.;
  X[1][1] = 1.5;

  double **Y;
  Y = myalloc2(1, 2);

  hos_forward(tapeId3, 1, 2, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double **H;
  H = myalloc2(2, 2);

  double yx1x1Derivative = 2. * std::cos(x2);
  double yx1x2Derivative = -2. * x1 * std::sin(x2);
  double yx2x2Derivative = -x1 * x1 * std::cos(x2);

  hessian(tapeId3, 2, x, H);

  BOOST_TEST(yx1x1Derivative == H[0][0], tt::tolerance(tol));
  BOOST_TEST(yx1x2Derivative == H[1][0], tt::tolerance(tol));
  BOOST_TEST(yx2x2Derivative == H[1][1], tt::tolerance(tol));

  myfree1(x);
  myfree2(yDerivative);
  myfree2(X);
  myfree2(Y);
  myfree2(H);
}

/* Tested function: fmin(x1*x1*cos(x2), sin(x1)*cos(x2*exp(x2)))
 * First derivatives: (cos(x1)*cos(x2*exp(x2)),
 *                     -sin(x1)*sin(x2*exp(x2))*(1. + x2)*exp(x2)
 *                    )
 * Second derivatives: (-sin(x1)*cos(x2*exp(x2)),
 *                      -cos(x1)*sin(x2*exp(x2))*(1. + x2)*exp(x2),
 *                      -cos(x1)*sin(x2*exp(x2))*(1. + x2)*exp(x2),
 *                      -sin(x1)*cos(x2*exp(x2))*pow((1. + x2)*exp(x2), 2)
 *                      - sin(x1)*sin(x2*exp(x2))*(2. + x2)*exp(x2)
 *                     )
 */
BOOST_AUTO_TEST_CASE(CustomFminTrigExp_HOS) {
  double x1 = 21.07, x2 = 1.5;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  double y;
  adouble ay;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay = fmin(ax1 * ax1 * cos(ax2), sin(ax1) * cos(ax2 * exp(ax2)));

  ay >>= y;
  trace_off();

  double yprim = std::fmin(x1 * x1 * std::cos(x2),
                           std::sin(x1) * std::cos(x2 * std::exp(x2)));

  double **yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = std::cos(x1) * std::cos(x2 * std::exp(x2));
  yDerivative[0][1] = -1.5 * sin(x1) * sin(x2 * exp(x2)) * (1. + x2) * exp(x2) -
                      0.5 * std::sin(x1) * std::cos(x2 * std::exp(x2));

  double *x;
  x = myalloc1(2);
  x[0] = x1;
  x[1] = x2;

  double **X;
  X = myalloc2(2, 2);
  X[0][0] = 1.;
  X[0][1] = 0.;
  X[1][0] = 0.;
  X[1][1] = 1.5;

  double **Y;
  Y = myalloc2(1, 2);

  hos_forward(tapeId3, 1, 2, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double **H;
  H = myalloc2(2, 2);

  double yx1x1Derivative = -std::sin(x1) * std::cos(x2 * std::exp(x2));
  double yx1x2Derivative =
      -std::cos(x1) * std::sin(x2 * std::exp(x2)) * (1. + x2) * std::exp(x2);
  double yx2x2Derivative =
      -std::sin(x1) * std::cos(x2 * std::exp(x2)) *
          std::pow((1. + x2) * std::exp(x2), 2) -
      std::sin(x1) * std::sin(x2 * std::exp(x2)) * (2. + x2) * std::exp(x2);

  hessian(tapeId3, 2, x, H);

  BOOST_TEST(yx1x1Derivative == H[0][0], tt::tolerance(tol));
  BOOST_TEST(yx1x2Derivative == H[1][0], tt::tolerance(tol));
  BOOST_TEST(yx2x2Derivative == H[1][1], tt::tolerance(tol));

  myfree1(x);
  myfree2(yDerivative);
  myfree2(X);
  myfree2(Y);
  myfree2(H);
}

BOOST_AUTO_TEST_CASE(CustomMinTrigExp_HOS) {
  double x1 = 21.07, x2 = 1.5;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  double y;
  adouble ay;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay = min(ax1 * ax1 * cos(ax2), sin(ax1) * cos(ax2 * exp(ax2)));

  ay >>= y;
  trace_off();

  double yprim = std::min(x1 * x1 * std::cos(x2),
                          std::sin(x1) * std::cos(x2 * std::exp(x2)));

  double **yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = std::cos(x1) * std::cos(x2 * std::exp(x2));
  yDerivative[0][1] = -1.5 * sin(x1) * sin(x2 * exp(x2)) * (1. + x2) * exp(x2) -
                      0.5 * std::sin(x1) * std::cos(x2 * std::exp(x2));

  double *x;
  x = myalloc1(2);
  x[0] = x1;
  x[1] = x2;

  double **X;
  X = myalloc2(2, 2);
  X[0][0] = 1.;
  X[0][1] = 0.;
  X[1][0] = 0.;
  X[1][1] = 1.5;

  double **Y;
  Y = myalloc2(1, 2);

  hos_forward(tapeId3, 1, 2, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double **H;
  H = myalloc2(2, 2);

  double yx1x1Derivative = -std::sin(x1) * std::cos(x2 * std::exp(x2));
  double yx1x2Derivative =
      -std::cos(x1) * std::sin(x2 * std::exp(x2)) * (1. + x2) * std::exp(x2);
  double yx2x2Derivative =
      -std::sin(x1) * std::cos(x2 * std::exp(x2)) *
          std::pow((1. + x2) * std::exp(x2), 2) -
      std::sin(x1) * std::sin(x2 * std::exp(x2)) * (2. + x2) * std::exp(x2);

  hessian(tapeId3, 2, x, H);

  BOOST_TEST(yx1x1Derivative == H[0][0], tt::tolerance(tol));
  BOOST_TEST(yx1x2Derivative == H[1][0], tt::tolerance(tol));
  BOOST_TEST(yx2x2Derivative == H[1][1], tt::tolerance(tol));

  myfree1(x);
  myfree2(yDerivative);
  myfree2(X);
  myfree2(Y);
  myfree2(H);
}

/* Tested function: pow(x1, 3)*pow(x2, 4)*exp(tan(x3)) + x3 + sqrt(11)
 * First derivatives: (3.*pow(x1, 2)*pow(x2, 4)*exp(tan(x3)),
 *                     4.*pow(x1, 3)*pow(x2, 3)*exp(tan(x3)),
 *                     pow(x1, 3)*pow(x2, 4)*exp(tan(x3))
 *                     *(1. + pow(tan(x3), 2)) + 1.
 *                    )
 * Second derivatives: (6.*x1*pow(x2, 4)*exp(tan(x3)),
 *                      12.*pow(x1, 2)*pow(x2, 3)*exp(tan(x3)),
 *                      3.*pow(x1, 2)*pow(x2, 4)*exp(tan(x3))
 *                      *(1. + pow(tan(x3), 2)),
 *                      12.*pow(x1, 2)*pow(x2, 3)*exp(tan(x3)),
 *                      12.*pow(x1, 3)*pow(x2, 2)*exp(tan(x3)),
 *                      4.*pow(x1, 3)*pow(x2, 3)*exp(tan(x3))
 *                      *(1. + pow(tan(x3), 2)),
 *                      3.*pow(x1, 2)*pow(x2, 4)*exp(tan(x3))
 *                      *(1. + pow(tan(x3), 2)),
 *                      4.*pow(x1, 3)*pow(x2, 3)*exp(tan(x3))
 *                      *(1. + pow(tan(x3), 2)),
 *                      pow(x1, 3)*pow(x2, 4)*exp(tan(x3))
 *                      *(1. + pow(tan(x3), 2))
 *                      *(1. + 2.*tan(x3) + pow(tan(x3), 2))
 *                     )
 */
BOOST_AUTO_TEST_CASE(CustomPowExpTan_HOS) {
  double x1 = -5.2, x2 = 1.1, x3 = 5.4;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  adouble ax3;
  double y;
  adouble ay;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ay = pow(ax1, 3) * pow(ax2, 4) * exp(tan(ax3)) + ax3 + sqrt(11);

  ay >>= y;
  trace_off();

  double yprim = std::pow(x1, 3) * std::pow(x2, 4) * std::exp(std::tan(x3)) +
                 x3 + sqrt(11);

  double **yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] =
      0.6 * std::pow(x1, 2) * std::pow(x2, 4) * std::exp(std::tan(x3));
  yDerivative[0][1] =
      4. * std::pow(x1, 3) * std::pow(x2, 3) * std::exp(std::tan(x3)) +
      0.1 * std::pow(x1, 3) * std::pow(x2, 4) * std::exp(std::tan(x3)) *
          (1. + std::pow(std::tan(x3), 2)) +
      0.1 + 0.12 * x1 * std::pow(x2, 4) * std::exp(std::tan(x3));

  double *x;
  x = myalloc1(3);
  x[0] = x1;
  x[1] = x2;
  x[2] = x3;

  double **X;
  X = myalloc2(3, 2);
  X[0][0] = 0.2;
  X[1][0] = 0.;
  X[2][0] = 0.;
  X[0][1] = 0.;
  X[1][1] = 1.;
  X[2][1] = 0.1;

  double **Y;
  Y = myalloc2(1, 2);

  hos_forward(tapeId3, 1, 3, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double **H;
  H = myalloc2(3, 3);

  double yx1x1Derivative = 6. * x1 * std::pow(x2, 4) * std::exp(std::tan(x3));
  double yx2x1Derivative =
      12. * std::pow(x1, 2) * std::pow(x2, 3) * std::exp(std::tan(x3));
  double yx3x1Derivative = 3. * std::pow(x1, 2) * std::pow(x2, 4) *
                           std::exp(std::tan(x3)) *
                           (1. + std::pow(std::tan(x3), 2));
  double yx2x2Derivative =
      12. * std::pow(x1, 3) * std::pow(x2, 2) * std::exp(std::tan(x3));
  double yx3x2Derivative = 4. * std::pow(x1, 3) * std::pow(x2, 3) *
                           std::exp(std::tan(x3)) *
                           (1. + std::pow(std::tan(x3), 2));
  double yx3x3Derivative = std::pow(x1, 3) * std::pow(x2, 4) *
                           std::exp(std::tan(x3)) *
                           (1. + std::pow(std::tan(x3), 2)) *
                           (1. + 2. * std::tan(x3) + std::pow(std::tan(x3), 2));

  hessian(tapeId3, 3, x, H);

  BOOST_TEST(yx1x1Derivative == H[0][0], tt::tolerance(tol));
  BOOST_TEST(yx2x1Derivative == H[1][0], tt::tolerance(tol));
  BOOST_TEST(yx2x2Derivative == H[1][1], tt::tolerance(tol));
  BOOST_TEST(yx3x1Derivative == H[2][0], tt::tolerance(tol));
  BOOST_TEST(yx3x2Derivative == H[2][1], tt::tolerance(tol));
  BOOST_TEST(yx3x3Derivative == H[2][2], tt::tolerance(tol));

  myfree1(x);
  myfree2(yDerivative);
  myfree2(X);
  myfree2(Y);
  myfree2(H);
}

/* Tested function: 0.5*(x1*x1 + x2*x2 + x3*x3 + x4*x4 + x5*x5 + x6*x6)
 * First derivatives: (x1, x2, x3, x4, x5, x6
 *                    )
 * Second derivatives: (1., 0., 0., 0., 0., 0.,
 *                      0., 1., 0., 0., 0., 0.,
 *                      0., 0., 1., 0., 0., 0.,
 *                      0., 0., 0., 1., 0., 0.,
 *                      0., 0., 0., 0., 1., 0.,
 *                      0., 0., 0., 0., 0., 1.
 *                     )
 */
BOOST_AUTO_TEST_CASE(CustomManyVariabl_HOS) {
  double x1 = 1.5, x2 = -1.5, x3 = 3., x4 = -3., x5 = 4.5, x6 = -4.5;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ax4;
  adouble ax5;
  adouble ax6;
  double y;
  adouble ay;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;
  ax4 <<= x4;
  ax5 <<= x5;
  ax6 <<= x6;

  ay = 0.5 *
       (ax1 * ax1 + ax2 * ax2 + ax3 * ax3 + ax4 * ax4 + ax5 * ax5 + ax6 * ax6);

  ay >>= y;
  trace_off();

  double yprim =
      0.5 * (x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4 + x5 * x5 + x6 * x6);

  double **yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = x1;
  yDerivative[0][1] = x3 + 0.5 * 1.;

  double *x;
  x = myalloc1(6);
  x[0] = x1;
  x[1] = x2;
  x[2] = x3;
  x[3] = x4;
  x[4] = x5;
  x[5] = x6;

  double **X;
  X = myalloc2(6, 2);
  X[0][0] = 1.;
  X[1][0] = 0.;
  X[2][0] = 0.;
  X[3][0] = 0.;
  X[4][0] = 0.;
  X[5][0] = 0.;
  X[0][1] = 0.;
  X[1][1] = 0.;
  X[2][1] = 1.;
  X[3][1] = 0.;
  X[4][1] = 0.;
  X[5][1] = 0.;

  double **Y;
  Y = myalloc2(1, 2);

  hos_forward(tapeId3, 1, 6, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double **H;
  H = myalloc2(6, 6);

  hessian(tapeId3, 6, x, H);

  BOOST_TEST(H[0][0] == 1., tt::tolerance(tol));
  BOOST_TEST(H[1][0] == 0., tt::tolerance(tol));
  BOOST_TEST(H[2][0] == 0., tt::tolerance(tol));
  BOOST_TEST(H[3][0] == 0., tt::tolerance(tol));
  BOOST_TEST(H[4][0] == 0., tt::tolerance(tol));
  BOOST_TEST(H[5][0] == 0., tt::tolerance(tol));
  BOOST_TEST(H[1][1] == 1., tt::tolerance(tol));
  BOOST_TEST(H[2][1] == 0., tt::tolerance(tol));
  BOOST_TEST(H[3][1] == 0., tt::tolerance(tol));
  BOOST_TEST(H[4][1] == 0., tt::tolerance(tol));
  BOOST_TEST(H[5][1] == 0., tt::tolerance(tol));
  BOOST_TEST(H[2][2] == 1., tt::tolerance(tol));
  BOOST_TEST(H[3][2] == 0., tt::tolerance(tol));
  BOOST_TEST(H[4][2] == 0., tt::tolerance(tol));
  BOOST_TEST(H[5][2] == 0., tt::tolerance(tol));
  BOOST_TEST(H[3][3] == 1., tt::tolerance(tol));
  BOOST_TEST(H[4][3] == 0., tt::tolerance(tol));
  BOOST_TEST(H[5][3] == 0., tt::tolerance(tol));
  BOOST_TEST(H[4][4] == 1., tt::tolerance(tol));
  BOOST_TEST(H[5][4] == 0., tt::tolerance(tol));
  BOOST_TEST(H[5][5] == 1., tt::tolerance(tol));

  myfree1(x);
  myfree2(yDerivative);
  myfree2(X);
  myfree2(Y);
  myfree2(H);
}

/* Tested function: 0.001001
 * First derivatives: (0., 0., 0., 0., 0., 0.
 *                    )
 * Second derivatives: (0., 0., 0., 0., 0., 0.,
 *                      0., 0., 0., 0., 0., 0.,
 *                      0., 0., 0., 0., 0., 0.,
 *                      0., 0., 0., 0., 0., 0.,
 *                      0., 0., 0., 0., 0., 0.,
 *                      0., 0., 0., 0., 0., 0.
 *                     )
 */
BOOST_AUTO_TEST_CASE(CustomConstant_HOS) {
  double x1 = 1., x2 = -1., x3 = 3.5, x4 = -3.5, x5 = 4., x6 = -4.;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ax4;
  adouble ax5;
  adouble ax6;
  double y;
  adouble ay;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;
  ax4 <<= x4;
  ax5 <<= x5;
  ax6 <<= x6;

  ay = 0.001001;

  ay >>= y;
  trace_off();

  double yprim = 0.001001;

  double **yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = 0.;
  yDerivative[0][1] = 0.;

  double *x;
  x = myalloc1(6);
  x[0] = x1;
  x[1] = x2;
  x[2] = x3;
  x[3] = x4;
  x[4] = x5;
  x[5] = x6;

  double **X;
  X = myalloc2(6, 2);
  X[0][0] = 1.;
  X[1][0] = 3.1;
  X[2][0] = -0.381;
  X[3][0] = 1000.;
  X[4][0] = -100.;
  X[5][0] = -0.0001;
  X[0][1] = 5.23;
  X[1][1] = 3.25;
  X[2][1] = 1.2;
  X[3][1] = 0.2;
  X[4][1] = 9.91;
  X[5][1] = 2.345;

  double **Y;
  Y = myalloc2(1, 2);

  hos_forward(tapeId3, 1, 6, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double **H;
  H = myalloc2(6, 6);

  hessian(tapeId3, 6, x, H);

  BOOST_TEST(H[0][0] == 0., tt::tolerance(tol));
  BOOST_TEST(H[1][0] == 0., tt::tolerance(tol));
  BOOST_TEST(H[2][0] == 0., tt::tolerance(tol));
  BOOST_TEST(H[3][0] == 0., tt::tolerance(tol));
  BOOST_TEST(H[4][0] == 0., tt::tolerance(tol));
  BOOST_TEST(H[5][0] == 0., tt::tolerance(tol));
  BOOST_TEST(H[1][1] == 0., tt::tolerance(tol));
  BOOST_TEST(H[2][1] == 0., tt::tolerance(tol));
  BOOST_TEST(H[3][1] == 0., tt::tolerance(tol));
  BOOST_TEST(H[4][1] == 0., tt::tolerance(tol));
  BOOST_TEST(H[5][1] == 0., tt::tolerance(tol));
  BOOST_TEST(H[2][2] == 0., tt::tolerance(tol));
  BOOST_TEST(H[3][2] == 0., tt::tolerance(tol));
  BOOST_TEST(H[4][2] == 0., tt::tolerance(tol));
  BOOST_TEST(H[5][2] == 0., tt::tolerance(tol));
  BOOST_TEST(H[3][3] == 0., tt::tolerance(tol));
  BOOST_TEST(H[4][3] == 0., tt::tolerance(tol));
  BOOST_TEST(H[5][3] == 0., tt::tolerance(tol));
  BOOST_TEST(H[4][4] == 0., tt::tolerance(tol));
  BOOST_TEST(H[5][4] == 0., tt::tolerance(tol));
  BOOST_TEST(H[5][5] == 0., tt::tolerance(tol));

  myfree1(x);
  myfree2(yDerivative);
  myfree2(X);
  myfree2(Y);
  myfree2(H);
}

/* Next, tests for the ADOL-C driver hos_reverse are implemented.  They
 * are stated separately in order not to clutter the hos_forward and
 * hessian driver test cases.  Analytical derivatives are not state
 * separately, as the following tests require several different partial
 * derivatives.  The derivative values can instead be found in the
 * variables yixj for first order partial derivatives and yixjxk for
 * second order partial derivatives.
 *
 * Before calling hos_reverse, one needs to call fos_forward or
 * hos_forward with appropriate keep parameter.
 */

BOOST_AUTO_TEST_CASE(customSimpleSum_HOS_Reverse) {
  double x1 = 1., x2 = -1., x3 = 0.;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  adouble ax3;
  double y1, y2;
  adouble ay1;
  adouble ay2;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ay1 = ax1 * exp(ax2 + ax3);
  ay2 = ax1 + ax2 + ax3;

  ay1 >>= y1;
  ay2 >>= y2;
  trace_off();

  double y1x1Derivative = std::exp(x2 + x3);
  double y1x2Derivative = x1 * std::exp(x2 + x3);
  double y1x3Derivative = x1 * std::exp(x2 + x3);
  double y2x1Derivative = 1.;
  double y2x2Derivative = 1.;
  double y2x3Derivative = 1.;

  double y1x1x1Derivative = 0.;
  double y1x1x2Derivative = std::exp(x2 + x3);
  double y1x1x3Derivative = std::exp(x2 + x3);
  double y2x1x1Derivative = 0.;
  double y2x1x2Derivative = 0.;
  double y2x1x3Derivative = 0.;
  double y1x2x1Derivative = std::exp(x2 + x3);
  double y1x2x2Derivative = x1 * std::exp(x2 + x3);
  double y1x2x3Derivative = x1 * std::exp(x2 + x3);
  double y2x2x1Derivative = 0.;
  double y2x2x2Derivative = 0.;
  double y2x2x3Derivative = 0.;
  double y1x3x1Derivative = std::exp(x2 + x3);
  double y1x3x2Derivative = x1 * std::exp(x2 + x3);
  double y1x3x3Derivative = x1 * std::exp(x2 + x3);
  double y2x3x1Derivative = 0.;
  double y2x3x2Derivative = 0.;
  double y2x3x3Derivative = 0.;

  double *x = myalloc1(3);
  double *xd = myalloc1(3);
  double *y = myalloc1(2);
  double *yd = myalloc1(2);

  x[0] = 1.;
  x[1] = -1.;
  x[2] = 0.;
  xd[0] = 1.;
  xd[1] = 0.;
  xd[2] = 0.;

  fos_forward(tapeId3, 2, 3, 2, x, xd, y, yd);

  double *u = myalloc1(2);
  double **Z = myalloc2(3, 2);

  u[0] = 1.;
  u[1] = 0.;

  hos_reverse(tapeId3, 2, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));

  /* The second derivative tensor is tested against u from the left and
   * xd from the right!
   */

  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x1x3Derivative, tt::tolerance(tol));

  u[0] = 0.;
  u[1] = 1.;

  hos_reverse(tapeId3, 2, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y2x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y2x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y2x1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y2x1x3Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;
  xd[2] = 0.;

  fos_forward(tapeId3, 2, 3, 2, x, xd, y, yd);

  u[0] = 1.;
  u[1] = 0.;

  hos_reverse(tapeId3, 2, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x2x3Derivative, tt::tolerance(tol));

  u[0] = 0.;
  u[1] = 1.;

  hos_reverse(tapeId3, 2, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y2x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y2x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y2x2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y2x2x3Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 0.;
  xd[2] = 1.;

  fos_forward(tapeId3, 2, 3, 2, x, xd, y, yd);

  u[0] = 1.;
  u[1] = 0.;

  hos_reverse(tapeId3, 2, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x3x3Derivative, tt::tolerance(tol));

  u[0] = 0.;
  u[1] = 1.;

  hos_reverse(tapeId3, 2, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y2x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y2x3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y2x3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y2x3x3Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_CASE(customTrigExp_HOS_Reverse) {
  double x1 = 1.78, x2 = -7.81, x3 = 0.03;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  adouble ax3;
  double y1, y2;
  adouble ay1;
  adouble ay2;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ay1 = ax1 * exp(sin(ax2) * cos(ax3) + sqrt(2) + 3. * ax3);
  ay2 = 2. * ax1 * ax1 + 3. * ax2 * ax2 + 4. * ax3 * ax3 + ax1 * ax2 * ax3;

  ay1 >>= y1;
  ay2 >>= y2;
  trace_off();

  double y1x1Derivative =
      std::exp(std::sin(x2) * std::cos(x3) + std::sqrt(2) + 3. * x3);
  double y1x2Derivative =
      x1 * std::exp(std::sin(x2) * std::cos(x3) + std::sqrt(2) + 3. * x3) *
      std::cos(x2) * std::cos(x3);
  double y1x3Derivative =
      x1 * std::exp(std::sin(x2) * std::cos(x3) + std::sqrt(2) + 3. * x3) *
      (3. - std::sin(x2) * std::sin(x3));
  double y2x1Derivative = 4. * x1 + x2 * x3;
  double y2x2Derivative = 6. * x2 + x1 * x3;
  double y2x3Derivative = 8. * x3 + x1 * x2;

  double y1x1x1Derivative = 0.;
  double y1x1x2Derivative =
      std::exp(std::sin(x2) * std::cos(x3) + std::sqrt(2) + 3. * x3) *
      std::cos(x2) * std::cos(x3);
  double y1x1x3Derivative =
      std::exp(std::sin(x2) * std::cos(x3) + std::sqrt(2) + 3. * x3) *
      (3. - std::sin(x2) * std::sin(x3));
  double y2x1x1Derivative = 4.;
  double y2x1x2Derivative = x3;
  double y2x1x3Derivative = x2;
  double y1x2x1Derivative =
      std::exp(std::sin(x2) * std::cos(x3) + std::sqrt(2) + 3. * x3) *
      std::cos(x2) * std::cos(x3);
  double y1x2x2Derivative =
      x1 * std::exp(std::sin(x2) * std::cos(x3) + std::sqrt(2) + 3. * x3) *
      (std::cos(x2) * std::cos(x3) * std::cos(x2) * std::cos(x3) -
       std::sin(x2) * std::cos(x3));
  double y1x2x3Derivative =
      x1 * std::exp(std::sin(x2) * std::cos(x3) + std::sqrt(2) + 3. * x3) *
      (std::cos(x2) * std::cos(x3) * (-std::sin(x2) * std::sin(x3) + 3.) -
       std::cos(x2) * std::sin(x3));
  double y2x2x1Derivative = x3;
  double y2x2x2Derivative = 6.;
  double y2x2x3Derivative = x1;
  double y1x3x1Derivative =
      std::exp(std::sin(x2) * std::cos(x3) + std::sqrt(2) + 3. * x3) *
      (3. - std::sin(x2) * std::sin(x3));
  double y1x3x2Derivative =
      x1 * std::exp(std::sin(x2) * std::cos(x3) + std::sqrt(2) + 3. * x3) *
      (std::cos(x2) * std::cos(x3) * (-std::sin(x2) * std::sin(x3) + 3.) -
       std::cos(x2) * std::sin(x3));
  double y1x3x3Derivative =
      x1 * std::exp(std::sin(x2) * std::cos(x3) + std::sqrt(2) + 3. * x3) *
      ((3. - std::sin(x2) * std::sin(x3)) * (3. - std::sin(x2) * std::sin(x3)) -
       std::sin(x2) * std::cos(x3));

  double y2x3x1Derivative = x2;
  double y2x3x2Derivative = x1;
  double y2x3x3Derivative = 8.;

  double *x = myalloc1(3);
  double *xd = myalloc1(3);
  double *y = myalloc1(2);
  double *yd = myalloc1(2);

  x[0] = 1.78;
  x[1] = -7.81;
  x[2] = 0.03;
  xd[0] = 1.;
  xd[1] = 0.;
  xd[2] = 0.;

  fos_forward(tapeId3, 2, 3, 2, x, xd, y, yd);

  double *u = myalloc1(2);
  double **Z = myalloc2(3, 2);

  u[0] = 1.;
  u[1] = 0.;

  hos_reverse(tapeId3, 2, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x1x3Derivative, tt::tolerance(tol));

  u[0] = 0.;
  u[1] = 1.;

  hos_reverse(tapeId3, 2, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y2x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y2x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y2x1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y2x1x3Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;
  xd[2] = 0.;

  fos_forward(tapeId3, 2, 3, 2, x, xd, y, yd);

  u[0] = 1.;
  u[1] = 0.;

  hos_reverse(tapeId3, 2, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x2x3Derivative, tt::tolerance(tol));

  u[0] = 0.;
  u[1] = 1.;

  hos_reverse(tapeId3, 2, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y2x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y2x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y2x2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y2x2x3Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 0.;
  xd[2] = 1.;

  fos_forward(tapeId3, 2, 3, 2, x, xd, y, yd);

  u[0] = 1.;
  u[1] = 0.;

  hos_reverse(tapeId3, 2, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x3x3Derivative, tt::tolerance(tol));

  u[0] = 0.;
  u[1] = 1.;

  hos_reverse(tapeId3, 2, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y2x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y2x3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y2x3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y2x3x3Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_CASE(customPowPow_HOS_Reverse) {
  double x1 = 2.35, x2 = 5.6, x3 = 2.66;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  adouble ax3;
  double y1, y2;
  adouble ay1;
  adouble ay2;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ay1 = pow(ax1, ax2) + pow(ax3, 9.2);
  ay2 = pow(ax1, -2.4) * pow(ax2, 2.4) * pow(ax3, 3.);

  ay1 >>= y1;
  ay2 >>= y2;
  trace_off();

  double y1x1Derivative = x2 * std::pow(x1, x2 - 1.);
  double y1x2Derivative = std::pow(x1, x2) * std::log(x1);
  double y1x3Derivative = 9.2 * std::pow(x3, 8.2);
  double y2x1Derivative =
      -2.4 * std::pow(x1, -3.4) * std::pow(x2, 2.4) * std::pow(x3, 3.);
  double y2x2Derivative =
      2.4 * std::pow(x1, -2.4) * std::pow(x2, 1.4) * std::pow(x3, 3.);
  double y2x3Derivative =
      3. * std::pow(x1, -2.4) * std::pow(x2, 2.4) * std::pow(x3, 2.);

  double y1x1x1Derivative = x2 * (x2 - 1.) * std::pow(x1, x2 - 2.);
  double y1x1x2Derivative =
      std::pow(x1, x2 - 1.) + x2 * std::pow(x1, x2 - 1.) * std::log(x1);
  double y1x1x3Derivative = 0.;
  double y2x1x1Derivative =
      2.4 * 3.4 * std::pow(x1, -4.4) * std::pow(x2, 2.4) * std::pow(x3, 3.);
  double y2x1x2Derivative =
      -2.4 * 2.4 * std::pow(x1, -3.4) * std::pow(x2, 1.4) * std::pow(x3, 3.);
  double y2x1x3Derivative =
      -2.4 * 3. * std::pow(x1, -3.4) * std::pow(x2, 2.4) * std::pow(x3, 2.);
  double y1x2x1Derivative =
      std::pow(x1, x2 - 1.) + x2 * std::pow(x1, x2 - 1.) * std::log(x1);
  double y1x2x2Derivative = std::pow(x1, x2) * std::pow(std::log(x1), 2.);
  double y1x2x3Derivative = 0.;
  double y2x2x1Derivative =
      -2.4 * 2.4 * std::pow(x1, -3.4) * std::pow(x2, 1.4) * std::pow(x3, 3.);
  double y2x2x2Derivative =
      2.4 * 1.4 * std::pow(x1, -2.4) * std::pow(x2, 0.4) * std::pow(x3, 3.);
  double y2x2x3Derivative =
      2.4 * 3. * std::pow(x1, -2.4) * std::pow(x2, 1.4) * std::pow(x3, 2.);
  double y1x3x1Derivative = 0.;
  double y1x3x2Derivative = 0.;
  double y1x3x3Derivative = 9.2 * 8.2 * std::pow(x3, 7.2);
  double y2x3x1Derivative =
      -2.4 * 3. * std::pow(x1, -3.4) * std::pow(x2, 2.4) * std::pow(x3, 2.);
  double y2x3x2Derivative =
      2.4 * 3. * std::pow(x1, -2.4) * std::pow(x2, 1.4) * std::pow(x3, 2.);
  double y2x3x3Derivative =
      3. * 2. * std::pow(x1, -2.4) * std::pow(x2, 2.4) * std::pow(x3, 1.);

  double *x = myalloc1(3);
  double *xd = myalloc1(3);
  double *y = myalloc1(2);
  double *yd = myalloc1(2);

  x[0] = 2.35;
  x[1] = 5.6;
  x[2] = 2.66;
  xd[0] = 1.;
  xd[1] = 0.;
  xd[2] = 0.;

  fos_forward(tapeId3, 2, 3, 2, x, xd, y, yd);

  double *u = myalloc1(2);
  double **Z = myalloc2(3, 2);

  u[0] = 1.;
  u[1] = 0.;

  hos_reverse(tapeId3, 2, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x1x3Derivative, tt::tolerance(tol));

  u[0] = 0.;
  u[1] = 1.;

  hos_reverse(tapeId3, 2, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y2x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y2x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y2x1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y2x1x3Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;
  xd[2] = 0.;

  fos_forward(tapeId3, 2, 3, 2, x, xd, y, yd);

  u[0] = 1.;
  u[1] = 0.;

  hos_reverse(tapeId3, 2, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x2x3Derivative, tt::tolerance(tol));

  u[0] = 0.;
  u[1] = 1.;

  hos_reverse(tapeId3, 2, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y2x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y2x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y2x2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y2x2x3Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 0.;
  xd[2] = 1.;

  fos_forward(tapeId3, 2, 3, 2, x, xd, y, yd);

  u[0] = 1.;
  u[1] = 0.;

  hos_reverse(tapeId3, 2, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x3x3Derivative, tt::tolerance(tol));

  u[0] = 0.;
  u[1] = 1.;

  hos_reverse(tapeId3, 2, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y2x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y2x3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y2x3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y2x3x3Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_CASE(customCube_HOS_Reverse) {
  double x1 = 3.;
  setCurrentTape(tapeId3);
  adouble ax1;
  double y1;
  adouble ay1;

  trace_on(tapeId3, 1);
  ax1 <<= x1;

  ay1 = 2. * ax1 * ax1 * ax1;

  ay1 >>= y1;
  trace_off();

  double y1x1Derivative = 6. * x1 * x1;

  double y1x1x1Derivative = 12. * x1;

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  x[0] = 3.;
  xd[0] = 1.;

  fos_forward(tapeId3, 1, 1, 2, x, xd, y, yd);

  double *u = myalloc1(1);
  double **Z = myalloc2(1, 2);

  u[0] = 1.;

  hos_reverse(tapeId3, 1, 1, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_CASE(customTrigProd_HOS_Reverse) {
  double x1 = 1.3, x2 = 3.1;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  double y1;
  adouble ay1;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay1 = cos(ax1) * sin(ax2);

  ay1 >>= y1;
  trace_off();

  double y1x1Derivative = -std::sin(x1) * std::sin(x2);
  double y1x2Derivative = std::cos(x1) * std::cos(x2);

  double y1x1x1Derivative = -std::cos(x1) * std::sin(x2);
  double y1x1x2Derivative = -std::sin(x1) * std::cos(x2);
  double y1x2x1Derivative = -std::sin(x1) * std::cos(x2);
  double y1x2x2Derivative = -std::cos(x1) * std::sin(x2);

  double *x = myalloc1(2);
  double *xd = myalloc1(2);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  x[0] = 1.3;
  x[1] = 3.1;
  xd[0] = 1.;
  xd[1] = 0.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  double *u = myalloc1(1);
  double **Z = myalloc2(2, 2);

  u[0] = 1.;

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x1x2Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x2x2Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_CASE(customTrigPow_HOS_Reverse) {
  double x1 = 1.1, x2 = 4.53, x3 = -3.03;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  adouble ax3;
  double y1;
  adouble ay1;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ay1 = pow(ax1, ax2) * exp(2. * ax3);

  ay1 >>= y1;
  trace_off();

  double y1x1Derivative = x2 * pow(x1, x2 - 1) * exp(2. * x3);
  double y1x2Derivative = pow(x1, x2) * log(x1) * exp(2. * x3);
  double y1x3Derivative = 2. * pow(x1, x2) * exp(2. * x3);

  double y1x1x1Derivative =
      x2 * (x2 - 1) * std::pow(x1, x2 - 2) * std::exp(2. * x3);
  double y1x1x2Derivative =
      x2 * std::pow(x1, x2 - 1) * std::log(x1) * std::exp(2. * x3) +
      std::pow(x1, x2 - 1) * exp(2. * x3);
  double y1x1x3Derivative = 2. * x2 * std::pow(x1, x2 - 1) * std::exp(2. * x3);
  double y1x2x1Derivative =
      std::pow(x1, x2) * std::exp(2. * x3) / x1 +
      x2 * std::pow(x1, x2 - 1) * std::log(x1) * std::exp(2. * x3);
  double y1x2x2Derivative =
      std::pow(x1, x2) * std::log(x1) * std::log(x1) * std::exp(2. * x3);
  double y1x2x3Derivative =
      2. * std::pow(x1, x2) * std::log(x1) * std::exp(2. * x3);
  double y1x3x1Derivative = 2. * x2 * std::pow(x1, x2 - 1) * std::exp(2. * x3);
  double y1x3x2Derivative =
      2. * std::pow(x1, x2) * std::log(x1) * std::exp(2. * x3);
  double y1x3x3Derivative = 4. * std::pow(x1, x2) * std::exp(2. * x3);

  double *x = myalloc1(3);
  double *xd = myalloc1(3);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  x[0] = 1.1;
  x[1] = 4.53;
  x[2] = -3.03;
  xd[0] = 1.;
  xd[1] = 0.;
  xd[2] = 0.;

  fos_forward(tapeId3, 1, 3, 2, x, xd, y, yd);

  double *u = myalloc1(1);
  double **Z = myalloc2(3, 2);

  u[0] = 1.;

  hos_reverse(tapeId3, 1, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x1x3Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;
  xd[2] = 0.;

  fos_forward(tapeId3, 1, 3, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x2x3Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 0.;
  xd[2] = 1.;

  fos_forward(tapeId3, 1, 3, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x3x3Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_CASE(customHyperbProd_HOS_Reverse) {
  double x1 = 2.22, x2 = -2.22;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  double y1;
  adouble ay1;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay1 = cosh(2. * ax1) * sinh(3. * ax2);

  ay1 >>= y1;
  trace_off();

  double y1x1Derivative = 2. * std::sinh(2. * x1) * std::sinh(3. * x2);
  double y1x2Derivative = std::cosh(2. * x1) * 3. * std::cosh(3. * x2);

  double y1x1x1Derivative = 4. * std::cosh(2. * x1) * std::sinh(3. * x2);
  double y1x1x2Derivative = 6. * std::sinh(2. * x1) * std::cosh(3. * x2);
  double y1x2x1Derivative = 6. * std::sinh(2. * x1) * std::cosh(3. * x2);
  double y1x2x2Derivative = 9. * std::cosh(2. * x1) * std::sinh(3. * x2);

  double *x = myalloc1(2);
  double *xd = myalloc1(2);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  x[0] = 2.22;
  x[1] = -2.22;
  xd[0] = 1.;
  xd[1] = 0.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  double *u = myalloc1(1);
  double **Z = myalloc2(2, 2);

  u[0] = 1.;

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x1x2Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x2x2Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_CASE(customPowTrig_HOS_Reverse) {
  double x1 = 0.531, x2 = 3.12;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  double y1;
  adouble ay1;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay1 = pow(sin(ax1), cos(ax2));

  ay1 >>= y1;
  trace_off();

  double y1x1Derivative = std::pow(std::sin(x1), std::cos(x2)) * std::cos(x2) *
                          std::cos(x1) / std::sin(x1);
  double y1x2Derivative = -std::pow(std::sin(x1), std::cos(x2)) * std::sin(x2) *
                          std::log(std::sin(x1));

  double y1x1x1Derivative =
      std::pow(std::sin(x1), std::cos(x2)) * std::cos(x2) *
      (-1 + std::pow(std::cos(x1) / std::sin(x1), 2) * (std::cos(x2) - 1));
  double y1x1x2Derivative =
      -std::pow(std::sin(x1), std::cos(x2)) * std::sin(x2) *
      (std::cos(x1) / std::sin(x1) +
       std::log(std::sin(x1)) * std::cos(x2) * std::cos(x1) / std::sin(x1));
  double y1x2x1Derivative =
      -std::pow(std::sin(x1), std::cos(x2)) * std::sin(x2) *
      (std::cos(x1) / std::sin(x1) +
       std::log(std::sin(x1)) * std::cos(x2) * std::cos(x1) / std::sin(x1));
  double y1x2x2Derivative =
      std::pow(std::sin(x1), std::cos(x2)) * std::log(std::sin(x1)) *
      (-std::cos(x2) + std::pow(std::sin(x2), 2) * std::log(std::sin(x1)));

  double *x = myalloc1(2);
  double *xd = myalloc1(2);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  x[0] = 0.531;
  x[1] = 3.12;
  xd[0] = 1.;
  xd[1] = 0.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  double *u = myalloc1(1);
  double **Z = myalloc2(2, 2);

  u[0] = 1.;

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x1x2Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x2x2Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_CASE(customPow_HOS_Reverse) {
  double x1 = 1.04, x2 = -2.01;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  double y1;
  adouble ay1;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay1 = pow(ax1, ax2);

  ay1 >>= y1;
  trace_off();

  double y1x1Derivative = x2 * std::pow(x1, x2 - 1);
  double y1x2Derivative = std::pow(x1, x2) * std::log(x1);

  double y1x1x1Derivative = x2 * (x2 - 1) * std::pow(x1, x2 - 2);
  double y1x1x2Derivative = std::pow(x1, x2 - 1) * (1 + x2 * std::log(x1));
  double y1x2x1Derivative = std::pow(x1, x2 - 1) * (1 + x2 * std::log(x1));
  double y1x2x2Derivative = std::pow(x1, x2) * std::pow(std::log(x1), 2);

  double *x = myalloc1(2);
  double *xd = myalloc1(2);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  x[0] = 1.04;
  x[1] = -2.01;
  xd[0] = 1.;
  xd[1] = 0.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  double *u = myalloc1(1);
  double **Z = myalloc2(2, 2);

  u[0] = 1.;

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x1x2Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x2x2Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_CASE(customExpSum_HOS_Reverse) {
  double x1 = -1.1, x2 = -4.53, x3 = 3.03, x4 = 0.;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ax4;
  double y1;
  adouble ay1;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;
  ax4 <<= x4;

  ay1 = exp(ax1 + 3. * ax2 + 5. * ax3 + 7. * ax4);

  ay1 >>= y1;
  trace_off();

  double y1x1Derivative = std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4);
  double y1x2Derivative = 3. * std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4);
  double y1x3Derivative = 5. * std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4);
  double y1x4Derivative = 7. * std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4);

  double y1x1x1Derivative = std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4);
  double y1x1x2Derivative = 3. * std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4);
  double y1x1x3Derivative = 5. * std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4);
  double y1x1x4Derivative = 7. * std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4);
  double y1x2x1Derivative = 3. * std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4);
  double y1x2x2Derivative = 9. * std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4);
  double y1x2x3Derivative = 15. * std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4);
  double y1x2x4Derivative = 21. * std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4);
  double y1x3x1Derivative = 5. * std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4);
  double y1x3x2Derivative = 15. * std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4);
  double y1x3x3Derivative = 25. * std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4);
  double y1x3x4Derivative = 35. * std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4);
  double y1x4x1Derivative = 7. * std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4);
  double y1x4x2Derivative = 21. * std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4);
  double y1x4x3Derivative = 35. * std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4);
  double y1x4x4Derivative = 49. * std::exp(x1 + 3. * x2 + 5. * x3 + 7. * x4);

  double *x = myalloc1(4);
  double *xd = myalloc1(4);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  x[0] = -1.1;
  x[1] = -4.53;
  x[2] = 3.03;
  x[3] = 0.;
  xd[0] = 1.;
  xd[1] = 0.;
  xd[2] = 0.;
  xd[3] = 0.;

  fos_forward(tapeId3, 1, 4, 2, x, xd, y, yd);

  double *u = myalloc1(1);
  double **Z = myalloc2(4, 2);

  u[0] = 1.;

  hos_reverse(tapeId3, 1, 4, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][0] == y1x4Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][1] == y1x1x4Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;
  xd[2] = 0.;
  xd[3] = 0.;

  fos_forward(tapeId3, 1, 4, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 4, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][0] == y1x4Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x2x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][1] == y1x2x4Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 0.;
  xd[2] = 1.;
  xd[3] = 0.;

  fos_forward(tapeId3, 1, 4, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 4, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][0] == y1x4Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x3x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][1] == y1x3x4Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 0.;
  xd[2] = 0.;
  xd[3] = 1.;

  fos_forward(tapeId3, 1, 4, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 4, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][0] == y1x4Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x4x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x4x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x4x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][1] == y1x4x4Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_CASE(customHypErf_HOS_Reverse) {
  double x1 = 5.55, x2 = 9.99;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  double y1;
  adouble ay1;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay1 = exp(tanh(ax1) * erf(ax2));

  ay1 >>= y1;
  trace_off();

  double y1x1Derivative = std::exp(std::tanh(x1) * std::erf(x2)) *
                          (1. - std::pow(std::tanh(x1), 2)) * std::erf(x2);
  double y1x2Derivative = std::exp(std::tanh(x1) * std::erf(x2)) *
                          std::tanh(x1) * std::exp(-x2 * x2) * 2. /
                          std::sqrt(std::acos(-1));

  double y1x1x1Derivative =
      std::exp(std::tanh(x1) * std::erf(x2)) *
      (1 - std::pow(std::tanh(x1), 2)) * std::erf(x2) *
      ((1 - std::pow(std::tanh(x1), 2)) * std::erf(x2) - 2 * std::tanh(x1));
  double y1x1x2Derivative =
      std::exp(std::tanh(x1) * std::erf(x2)) * std::exp(-x2 * x2) *
      (1 - std::pow(std::tanh(x1), 2)) * 2 / std::sqrt(std::acos(-1)) *
      (1 + std::tanh(x1) * std::erf(x2));
  double y1x2x1Derivative =
      std::exp(std::tanh(x1) * std::erf(x2)) * std::exp(-x2 * x2) *
      (1 - std::pow(std::tanh(x1), 2)) * 2 / std::sqrt(std::acos(-1)) *
      (1 + std::tanh(x1) * std::erf(x2));
  double y1x2x2Derivative =
      std::exp(std::tanh(x1) * std::erf(x2)) * std::exp(-x2 * x2) *
      (4 * std::pow(std::tanh(x1), 2) / std::acos(-1) * std::exp(-x2 * x2) -
       4. * x2 * std::tanh(x1) / std::sqrt(std::acos(-1)));

  double *x = myalloc1(2);
  double *xd = myalloc1(2);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  x[0] = 5.55;
  x[1] = 9.99;
  xd[0] = 1.;
  xd[1] = 0.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  double *u = myalloc1(1);
  double **Z = myalloc2(2, 2);

  u[0] = 1.;

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x1x2Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x2x2Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_CASE(customHypAtan_HOS_Reverse) {
  double x1 = 7.19, x2 = -4.32;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  double y1;
  adouble ay1;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay1 = (pow(cosh(ax1), 2) - pow(sinh(ax1), 2)) * atan(ax2);

  ay1 >>= y1;
  trace_off();

  double y1x1Derivative = 0;
  double y1x2Derivative = 1. / (1. + x2 * x2);

  double y1x1x1Derivative = 0.;
  double y1x1x2Derivative = 0.;
  double y1x2x1Derivative = 0.;
  double y1x2x2Derivative = -2. * x2 / std::pow(1. + x2 * x2, 2);

  double *x = myalloc1(2);
  double *xd = myalloc1(2);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  x[0] = 7.19;
  x[1] = -4.32;
  xd[0] = 1.;
  xd[1] = 0.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  double *u = myalloc1(1);
  double **Z = myalloc2(2, 2);

  u[0] = 1.;

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x1x2Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x2x2Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_CASE(customLongSum_HOS_Reverse) {
  double x1 = 99.99, x2 = std::exp(-0.44), x3 = std::sqrt(2);
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  adouble ax3;
  double y1;
  adouble ay1;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ay1 = 1. + ax1 + ax1 * ax1 + ax2 * ax2 + ax2 * ax2 * ax2 + ax3 * ax3 * ax3 +
        ax3 * ax3 * ax3 * ax3;

  ay1 >>= y1;
  trace_off();

  double y1x1Derivative = 1. + 2. * x1;
  double y1x2Derivative = 2. * x2 + 3. * x2 * x2;
  double y1x3Derivative = 3. * x3 * x3 + 4. * x3 * x3 * x3;

  double y1x1x1Derivative = 2.;
  double y1x1x2Derivative = 0.;
  double y1x1x3Derivative = 0.;
  double y1x2x1Derivative = 0.;
  double y1x2x2Derivative = 2. + 6. * x2;
  double y1x2x3Derivative = 0.;
  double y1x3x1Derivative = 0.;
  double y1x3x2Derivative = 0.;
  double y1x3x3Derivative = 6. * x3 + 12. * x3 * x3;

  double *x = myalloc1(3);
  double *xd = myalloc1(3);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  x[0] = 99.99;
  x[1] = std::exp(-0.44);
  x[2] = std::sqrt(2);
  xd[0] = 1.;
  xd[1] = 0.;
  xd[2] = 0.;

  fos_forward(tapeId3, 1, 3, 2, x, xd, y, yd);

  double *u = myalloc1(1);
  double **Z = myalloc2(3, 2);

  u[0] = 1.;

  hos_reverse(tapeId3, 1, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x1x3Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;
  xd[2] = 0.;

  fos_forward(tapeId3, 1, 3, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x2x3Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 0.;
  xd[2] = 1.;

  fos_forward(tapeId3, 1, 3, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x3x3Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_CASE(customExpSqrt_HOS_Reverse) {
  double x1 = -0.77, x2 = 10.01, x3 = 0.99;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  adouble ax3;
  double y1;
  adouble ay1;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ay1 = exp(ax1) * sqrt(2. * ax2) * pow(ax3, 2);

  ay1 >>= y1;
  trace_off();

  double y1x1Derivative = std::exp(x1) * std::sqrt(2. * x2) * std::pow(x3, 2);
  double y1x2Derivative = std::exp(x1) * std::pow(x3, 2) / std::sqrt(2. * x2);
  double y1x3Derivative = std::exp(x1) * std::sqrt(2. * x2) * 2. * x3;

  double y1x1x1Derivative = std::exp(x1) * std::sqrt(2. * x2) * std::pow(x3, 2);
  double y1x1x2Derivative = std::exp(x1) * std::pow(x3, 2) / std::sqrt(2. * x2);
  double y1x1x3Derivative = std::exp(x1) * std::sqrt(2. * x2) * 2. * x3;
  double y1x2x1Derivative = std::exp(x1) * std::pow(x3, 2) / std::sqrt(2. * x2);
  double y1x2x2Derivative =
      -std::exp(x1) * std::pow(x3, 2) / std::pow(std::sqrt(2. * x2), 3);
  double y1x2x3Derivative = std::exp(x1) * 2. * x3 / std::sqrt(2. * x2);
  double y1x3x1Derivative = std::exp(x1) * std::sqrt(2. * x2) * 2. * x3;
  double y1x3x2Derivative = std::exp(x1) * 2. * x3 / std::sqrt(2. * x2);
  double y1x3x3Derivative = 2. * std::exp(x1) * std::sqrt(2. * x2);

  double *x = myalloc1(3);
  double *xd = myalloc1(3);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  x[0] = -0.77;
  x[1] = 10.01;
  x[2] = 0.99;
  xd[0] = 1.;
  xd[1] = 0.;
  xd[2] = 0.;

  fos_forward(tapeId3, 1, 3, 2, x, xd, y, yd);

  double *u = myalloc1(1);
  double **Z = myalloc2(3, 2);

  u[0] = 1.;

  hos_reverse(tapeId3, 1, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x1x3Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;
  xd[2] = 0.;

  fos_forward(tapeId3, 1, 3, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x2x3Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 0.;
  xd[2] = 1.;

  fos_forward(tapeId3, 1, 3, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x3x3Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_CASE(customInvHyperb_HOS_Reverse) {
  double x1 = -3.03, x2 = 0.11;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  double y1;
  adouble ay1;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay1 = 2. * acosh(cosh(ax1 * ax1)) * atanh(ax2);

  ay1 >>= y1;
  trace_off();

  double y1x1Derivative = 4. * x1 * std::atanh(x2);
  double y1x2Derivative = 2. * x1 * x1 / (1. - x2 * x2);

  double y1x1x1Derivative = 4. * std::atanh(x2);
  double y1x1x2Derivative = 4. * x1 / (1. - x2 * x2);
  double y1x2x1Derivative = 4. * x1 / (1. - x2 * x2);
  double y1x2x2Derivative = 4. * x1 * x1 * x2 / std::pow(1. - x2 * x2, 2);

  double *x = myalloc1(2);
  double *xd = myalloc1(2);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  x[0] = -3.03;
  x[1] = 0.11;
  xd[0] = 1.;
  xd[1] = 0.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  double *u = myalloc1(1);
  double **Z = myalloc2(2, 2);

  u[0] = 1.;

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x1x2Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x2x2Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_CASE(customFminFmaxFabs_HOS_Reverse) {
  double x1 = 1., x2 = 2.5, x3 = -4.5, x4 = -1.;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ax4;
  double y1;
  adouble ay1;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;
  ax4 <<= x4;

  ay1 = fmax(fmin(ax1, ax2), fabs(ax3)) * ax4;

  ay1 >>= y1;
  trace_off();

  double y1x1Derivative = 0.;
  double y1x2Derivative = 0.;
  double y1x3Derivative = -x4;
  double y1x4Derivative = -x3;

  double y1x1x1Derivative = 0.;
  double y1x1x2Derivative = 0.;
  double y1x1x3Derivative = 0.;
  double y1x1x4Derivative = 0.;
  double y1x2x1Derivative = 0.;
  double y1x2x2Derivative = 0.;
  double y1x2x3Derivative = 0.;
  double y1x2x4Derivative = 0.;
  double y1x3x1Derivative = 0.;
  double y1x3x2Derivative = 0.;
  double y1x3x3Derivative = 0.;
  double y1x3x4Derivative = -1.;
  double y1x4x1Derivative = 0.;
  double y1x4x2Derivative = 0.;
  double y1x4x3Derivative = -1.;
  double y1x4x4Derivative = 0.;

  double *x = myalloc1(4);
  double *xd = myalloc1(4);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  x[0] = 1.;
  x[1] = 2.5;
  x[2] = -4.5;
  x[3] = -1.;
  xd[0] = 1.;
  xd[1] = 0.;
  xd[2] = 0.;
  xd[3] = 0.;

  fos_forward(tapeId3, 1, 4, 2, x, xd, y, yd);

  double *u = myalloc1(1);
  double **Z = myalloc2(4, 2);

  u[0] = 1.;

  hos_reverse(tapeId3, 1, 4, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][0] == y1x4Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][1] == y1x1x4Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;
  xd[2] = 0.;
  xd[3] = 0.;

  fos_forward(tapeId3, 1, 4, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 4, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][0] == y1x4Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x2x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][1] == y1x2x4Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 0.;
  xd[2] = 1.;
  xd[3] = 0.;

  fos_forward(tapeId3, 1, 4, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 4, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][0] == y1x4Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x3x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][1] == y1x3x4Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 0.;
  xd[2] = 0.;
  xd[3] = 1.;

  fos_forward(tapeId3, 1, 4, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 4, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][0] == y1x4Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x4x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x4x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x4x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][1] == y1x4x4Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_CASE(customMinMaxAbs_HOS_Reverse) {
  double x1 = 1., x2 = 2.5, x3 = -4.5, x4 = -1.;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ax4;
  double y1;
  adouble ay1;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;
  ax4 <<= x4;

  ay1 = max(min(ax1, ax2), abs(ax3)) * ax4;

  ay1 >>= y1;
  trace_off();

  double y1x1Derivative = 0.;
  double y1x2Derivative = 0.;
  double y1x3Derivative = -x4;
  double y1x4Derivative = -x3;

  double y1x1x1Derivative = 0.;
  double y1x1x2Derivative = 0.;
  double y1x1x3Derivative = 0.;
  double y1x1x4Derivative = 0.;
  double y1x2x1Derivative = 0.;
  double y1x2x2Derivative = 0.;
  double y1x2x3Derivative = 0.;
  double y1x2x4Derivative = 0.;
  double y1x3x1Derivative = 0.;
  double y1x3x2Derivative = 0.;
  double y1x3x3Derivative = 0.;
  double y1x3x4Derivative = -1.;
  double y1x4x1Derivative = 0.;
  double y1x4x2Derivative = 0.;
  double y1x4x3Derivative = -1.;
  double y1x4x4Derivative = 0.;

  double *x = myalloc1(4);
  double *xd = myalloc1(4);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  x[0] = 1.;
  x[1] = 2.5;
  x[2] = -4.5;
  x[3] = -1.;
  xd[0] = 1.;
  xd[1] = 0.;
  xd[2] = 0.;
  xd[3] = 0.;

  fos_forward(tapeId3, 1, 4, 2, x, xd, y, yd);

  double *u = myalloc1(1);
  double **Z = myalloc2(4, 2);

  u[0] = 1.;

  hos_reverse(tapeId3, 1, 4, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][0] == y1x4Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][1] == y1x1x4Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;
  xd[2] = 0.;
  xd[3] = 0.;

  fos_forward(tapeId3, 1, 4, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 4, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][0] == y1x4Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x2x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][1] == y1x2x4Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 0.;
  xd[2] = 1.;
  xd[3] = 0.;

  fos_forward(tapeId3, 1, 4, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 4, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][0] == y1x4Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x3x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][1] == y1x3x4Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 0.;
  xd[2] = 0.;
  xd[3] = 1.;

  fos_forward(tapeId3, 1, 4, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 4, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][0] == y1x4Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x4x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x4x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x4x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][1] == y1x4x4Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_CASE(customInvTrig_HOS_Reverse) {
  double x1 = 0.11, x2 = 0.33, x3 = 0.1 * std::acos(0.), x4 = std::exp(-1.);
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ax4;
  double y1;
  adouble ay1;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;
  ax4 <<= x4;

  ay1 = 3. * asin(sin(ax1 + ax2)) * sin(ax3) * cos(ax4);

  ay1 >>= y1;
  trace_off();

  double y1x1Derivative = 3. * std::sin(x3) * std::cos(x4);
  double y1x2Derivative = 3. * std::sin(x3) * std::cos(x4);
  double y1x3Derivative = 3. * (x1 + x2) * std::cos(x3) * std::cos(x4);
  double y1x4Derivative = -3. * (x1 + x2) * std::sin(x3) * std::sin(x4);

  double y1x1x1Derivative = 0.;
  double y1x1x2Derivative = 0.;
  double y1x1x3Derivative = 3 * std::cos(x3) * std::cos(x4);
  double y1x1x4Derivative = -3. * std::sin(x3) * std::sin(x4);
  double y1x2x1Derivative = 0.;
  double y1x2x2Derivative = 0.;
  double y1x2x3Derivative = 3 * std::cos(x3) * std::cos(x4);
  double y1x2x4Derivative = -3. * std::sin(x3) * std::sin(x4);
  double y1x3x1Derivative = 3. * std::cos(x3) * std::cos(x4);
  double y1x3x2Derivative = 3. * std::cos(x3) * std::cos(x4);
  double y1x3x3Derivative = -3. * (x1 + x2) * std::sin(x3) * std::cos(x4);
  double y1x3x4Derivative = -3. * (x1 + x2) * std::cos(x3) * std::sin(x4);
  double y1x4x1Derivative = -3. * std::sin(x3) * std::sin(x4);
  double y1x4x2Derivative = -3. * std::sin(x3) * std::sin(x4);
  double y1x4x3Derivative = -3. * (x1 + x2) * std::cos(x3) * std::sin(x4);
  double y1x4x4Derivative = -3. * (x1 + x2) * std::sin(x3) * std::cos(x4);

  double *x = myalloc1(4);
  double *xd = myalloc1(4);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  x[0] = 0.11;
  x[1] = 0.33;
  x[2] = 0.1 * std::acos(0.);
  x[3] = std::exp(-1.);
  xd[0] = 1.;
  xd[1] = 0.;
  xd[2] = 0.;
  xd[3] = 0.;

  fos_forward(tapeId3, 1, 4, 2, x, xd, y, yd);

  double *u = myalloc1(1);
  double **Z = myalloc2(4, 2);

  u[0] = 1.;

  hos_reverse(tapeId3, 1, 4, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][0] == y1x4Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][1] == y1x1x4Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;
  xd[2] = 0.;
  xd[3] = 0.;

  fos_forward(tapeId3, 1, 4, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 4, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][0] == y1x4Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x2x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][1] == y1x2x4Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 0.;
  xd[2] = 1.;
  xd[3] = 0.;

  fos_forward(tapeId3, 1, 4, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 4, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][0] == y1x4Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x3x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][1] == y1x3x4Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 0.;
  xd[2] = 0.;
  xd[3] = 1.;

  fos_forward(tapeId3, 1, 4, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 4, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][0] == y1x4Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x4x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x4x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x4x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][1] == y1x4x4Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_CASE(customInvTrig2_HOS_Reverse) {
  double x1 = 0.53, x2 = -0.01;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  double y1;
  adouble ay1;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay1 = atan(ax1) * asin(ax2);

  ay1 >>= y1;
  trace_off();

  double y1x1Derivative = std::asin(x2) / (1. + x1 * x1);
  double y1x2Derivative = std::atan(x1) / std::sqrt(1. - x2 * x2);

  double y1x1x1Derivative =
      -2. * x1 * std::asin(x2) / std::pow(1. + x1 * x1, 2);
  double y1x1x2Derivative = 1. / ((1. + x1 * x1) * std::sqrt(1. - x2 * x2));
  double y1x2x1Derivative = 1. / ((1. + x1 * x1) * std::sqrt(1. - x2 * x2));
  double y1x2x2Derivative =
      std::atan(x1) * x2 / std::pow(std::sqrt(1. - x2 * x2), 3);

  double *x = myalloc1(2);
  double *xd = myalloc1(2);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  x[0] = 0.53;
  x[1] = -0.01;
  xd[0] = 1.;
  xd[1] = 0.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  double *u = myalloc1(1);
  double **Z = myalloc2(2, 2);

  u[0] = 1.;

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x1x2Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x2x2Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_CASE(customFabsFmax_HOS_Reverse) {
  double x1 = 9.9, x2 = -4.7;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  double y1;
  adouble ay1;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay1 = fmax(fabs(ax1 * ax1), fabs(ax2 * ax2));

  ay1 >>= y1;
  trace_off();

  double y1x1Derivative = 2. * x1;
  double y1x2Derivative = 0.;

  double y1x1x1Derivative = 2.;
  double y1x1x2Derivative = 0.;
  double y1x2x1Derivative = 0.;
  double y1x2x2Derivative = 0.;

  double *x = myalloc1(2);
  double *xd = myalloc1(2);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  x[0] = 9.9;
  x[1] = -4.7;
  xd[0] = 1.;
  xd[1] = 0.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  double *u = myalloc1(1);
  double **Z = myalloc2(2, 2);

  u[0] = 1.;

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x1x2Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x2x2Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_CASE(customAbsMax_HOS_Reverse) {
  double x1 = 9.9, x2 = -4.7;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  double y1;
  adouble ay1;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay1 = max(abs(ax1 * ax1), abs(ax2 * ax2));

  ay1 >>= y1;
  trace_off();

  double y1x1Derivative = 2. * x1;
  double y1x2Derivative = 0.;

  double y1x1x1Derivative = 2.;
  double y1x1x2Derivative = 0.;
  double y1x2x1Derivative = 0.;
  double y1x2x2Derivative = 0.;

  double *x = myalloc1(2);
  double *xd = myalloc1(2);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  x[0] = 9.9;
  x[1] = -4.7;
  xd[0] = 1.;
  xd[1] = 0.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  double *u = myalloc1(1);
  double **Z = myalloc2(2, 2);

  u[0] = 1.;

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x1x2Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x2x2Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_CASE(customFabsFmin_HOS_Reverse) {
  double x1 = 9.9, x2 = -4.7;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  double y1;
  adouble ay1;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay1 = fmin(fabs(ax1 * ax1), fabs(ax2 * ax2));

  ay1 >>= y1;
  trace_off();

  double y1x1Derivative = 0.;
  double y1x2Derivative = 2. * x2;

  double y1x1x1Derivative = 0.;
  double y1x1x2Derivative = 0.;
  double y1x2x1Derivative = 0.;
  double y1x2x2Derivative = 2.;

  double *x = myalloc1(2);
  double *xd = myalloc1(2);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  x[0] = 9.9;
  x[1] = -4.7;
  xd[0] = 1.;
  xd[1] = 0.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  double *u = myalloc1(1);
  double **Z = myalloc2(2, 2);

  u[0] = 1.;

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x1x2Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x2x2Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_CASE(customAbsMin_HOS_Reverse) {
  double x1 = 9.9, x2 = -4.7;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  double y1;
  adouble ay1;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay1 = min(abs(ax1 * ax1), abs(ax2 * ax2));

  ay1 >>= y1;
  trace_off();

  double y1x1Derivative = 0.;
  double y1x2Derivative = 2. * x2;

  double y1x1x1Derivative = 0.;
  double y1x1x2Derivative = 0.;
  double y1x2x1Derivative = 0.;
  double y1x2x2Derivative = 2.;

  double *x = myalloc1(2);
  double *xd = myalloc1(2);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  x[0] = 9.9;
  x[1] = -4.7;
  xd[0] = 1.;
  xd[1] = 0.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  double *u = myalloc1(1);
  double **Z = myalloc2(2, 2);

  u[0] = 1.;

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x1x2Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x2x2Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_CASE(customFmaxTrig_HOS_Reverse) {
  double x1 = 21.07, x2 = 1.5;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  double y1;
  adouble ay1;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay1 = fmax(ax1 * ax1 * cos(ax2), sin(ax1) * cos(ax2) * exp(ax2));

  ay1 >>= y1;
  trace_off();

  double y1x1Derivative = 2. * x1 * std::cos(x2);
  double y1x2Derivative = -x1 * x1 * std::sin(x2);

  double y1x1x1Derivative = 2. * std::cos(x2);
  double y1x1x2Derivative = -2. * x1 * std::sin(x2);
  double y1x2x1Derivative = -2. * x1 * std::sin(x2);
  double y1x2x2Derivative = -x1 * x1 * std::cos(x2);

  double *x = myalloc1(2);
  double *xd = myalloc1(2);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  x[0] = 21.07;
  x[1] = 1.5;
  xd[0] = 1.;
  xd[1] = 0.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  double *u = myalloc1(1);
  double **Z = myalloc2(2, 2);

  u[0] = 1.;

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x1x2Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x2x2Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_CASE(customMaxTrig_HOS_Reverse) {
  double x1 = 21.07, x2 = 1.5;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  double y1;
  adouble ay1;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay1 = max(ax1 * ax1 * cos(ax2), sin(ax1) * cos(ax2) * exp(ax2));

  ay1 >>= y1;
  trace_off();

  double y1x1Derivative = 2. * x1 * std::cos(x2);
  double y1x2Derivative = -x1 * x1 * std::sin(x2);

  double y1x1x1Derivative = 2. * std::cos(x2);
  double y1x1x2Derivative = -2. * x1 * std::sin(x2);
  double y1x2x1Derivative = -2. * x1 * std::sin(x2);
  double y1x2x2Derivative = -x1 * x1 * std::cos(x2);

  double *x = myalloc1(2);
  double *xd = myalloc1(2);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  x[0] = 21.07;
  x[1] = 1.5;
  xd[0] = 1.;
  xd[1] = 0.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  double *u = myalloc1(1);
  double **Z = myalloc2(2, 2);

  u[0] = 1.;

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x1x2Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x2x2Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_CASE(customFminTrig_HOS_Reverse) {
  double x1 = 21.07, x2 = 1.5;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  double y1;
  adouble ay1;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay1 = fmin(ax1 * ax1 * cos(ax2), sin(ax1) * cos(ax2 * exp(ax2)));

  ay1 >>= y1;
  trace_off();

  double y1x1Derivative = std::cos(x1) * std::cos(x2 * std::exp(x2));
  double y1x2Derivative =
      -std::sin(x1) * std::sin(x2 * std::exp(x2)) * (1. + x2) * std::exp(x2);

  double y1x1x1Derivative = -std::sin(x1) * std::cos(x2 * std::exp(x2));
  double y1x1x2Derivative =
      -std::cos(x1) * std::sin(x2 * std::exp(x2)) * (1. + x2) * std::exp(x2);
  double y1x2x1Derivative =
      -std::cos(x1) * std::sin(x2 * std::exp(x2)) * (1. + x2) * std::exp(x2);
  double y1x2x2Derivative =
      -std::sin(x1) * std::cos(x2 * std::exp(x2)) *
          std::pow((1. + x2) * std::exp(x2), 2) -
      std::sin(x1) * std::sin(x2 * exp(x2)) * (2. + x2) * std::exp(x2);

  double *x = myalloc1(2);
  double *xd = myalloc1(2);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  x[0] = 21.07;
  x[1] = 1.5;
  xd[0] = 1.;
  xd[1] = 0.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  double *u = myalloc1(1);
  double **Z = myalloc2(2, 2);

  u[0] = 1.;

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x1x2Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x2x2Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_CASE(customMinTrig_HOS_Reverse) {
  double x1 = 21.07, x2 = 1.5;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  double y1;
  adouble ay1;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay1 = min(ax1 * ax1 * cos(ax2), sin(ax1) * cos(ax2 * exp(ax2)));

  ay1 >>= y1;
  trace_off();

  double y1x1Derivative = std::cos(x1) * std::cos(x2 * std::exp(x2));
  double y1x2Derivative =
      -std::sin(x1) * std::sin(x2 * std::exp(x2)) * (1. + x2) * std::exp(x2);

  double y1x1x1Derivative = -std::sin(x1) * std::cos(x2 * std::exp(x2));
  double y1x1x2Derivative =
      -std::cos(x1) * std::sin(x2 * std::exp(x2)) * (1. + x2) * std::exp(x2);
  double y1x2x1Derivative =
      -std::cos(x1) * std::sin(x2 * std::exp(x2)) * (1. + x2) * std::exp(x2);
  double y1x2x2Derivative =
      -std::sin(x1) * std::cos(x2 * std::exp(x2)) *
          std::pow((1. + x2) * std::exp(x2), 2) -
      std::sin(x1) * std::sin(x2 * exp(x2)) * (2. + x2) * std::exp(x2);

  double *x = myalloc1(2);
  double *xd = myalloc1(2);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  x[0] = 21.07;
  x[1] = 1.5;
  xd[0] = 1.;
  xd[1] = 0.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  double *u = myalloc1(1);
  double **Z = myalloc2(2, 2);

  u[0] = 1.;

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x1x2Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;

  fos_forward(tapeId3, 1, 2, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 2, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x2x2Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_CASE(customPowExpTan_HOS_Reverse) {
  double x1 = -5.2, x2 = 1.1, x3 = 5.4;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  adouble ax3;
  double y1;
  adouble ay1;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ay1 = pow(ax1, 3) * pow(ax2, 4) * exp(tan(ax3)) + ax3 + sqrt(11);

  ay1 >>= y1;
  trace_off();

  double y1x1Derivative =
      3. * std::pow(x1, 2) * std::pow(x2, 4) * std::exp(std::tan(x3));
  double y1x2Derivative =
      4. * std::pow(x1, 3) * std::pow(x2, 3) * std::exp(std::tan(x3));
  double y1x3Derivative = std::pow(x1, 3) * std::pow(x2, 4) *
                              std::exp(std::tan(x3)) *
                              (1. + std::pow(std::tan(x3), 2)) +
                          1.;

  double y1x1x1Derivative = 6. * x1 * std::pow(x2, 4) * std::exp(std::tan(x3));
  double y1x1x2Derivative =
      12. * std::pow(x1, 2) * std::pow(x2, 3) * std::exp(std::tan(x3));
  double y1x1x3Derivative = 3. * std::pow(x1, 2) * std::pow(x2, 4) *
                            std::exp(std::tan(x3)) *
                            (1. + std::pow(std::tan(x3), 2));
  double y1x2x1Derivative = 12. * pow(x1, 2) * pow(x2, 3) * exp(tan(x3));
  double y1x2x2Derivative = 12. * pow(x1, 3) * pow(x2, 2) * exp(tan(x3));
  double y1x2x3Derivative = 4. * std::pow(x1, 3) * std::pow(x2, 3) *
                            std::exp(std::tan(x3)) *
                            (1. + std::pow(std::tan(x3), 2));
  double y1x3x1Derivative = 3. * std::pow(x1, 2) * std::pow(x2, 4) *
                            std::exp(std::tan(x3)) *
                            (1. + std::pow(std::tan(x3), 2));
  double y1x3x2Derivative = 4. * std::pow(x1, 3) * std::pow(x2, 3) *
                            std::exp(std::tan(x3)) *
                            (1. + std::pow(std::tan(x3), 2));
  double y1x3x3Derivative =
      std::pow(x1, 3) * std::pow(x2, 4) * std::exp(std::tan(x3)) *
      (1. + std::pow(std::tan(x3), 2)) *
      (1. + 2. * std::tan(x3) + std::pow(std::tan(x3), 2));

  double *x = myalloc1(3);
  double *xd = myalloc1(3);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  x[0] = -5.2;
  x[1] = 1.1;
  x[2] = 5.4;
  xd[0] = 1.;
  xd[1] = 0.;
  xd[2] = 0.;

  fos_forward(tapeId3, 1, 3, 2, x, xd, y, yd);

  double *u = myalloc1(1);
  double **Z = myalloc2(3, 2);

  u[0] = 1.;

  hos_reverse(tapeId3, 1, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x1x3Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;
  xd[2] = 0.;

  fos_forward(tapeId3, 1, 3, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x2x3Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 0.;
  xd[2] = 1.;

  fos_forward(tapeId3, 1, 3, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x3x3Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_CASE(customManyVariabl_HOS_Reverse) {
  double x1 = 1.5, x2 = -1.5, x3 = 3., x4 = -3., x5 = 4.5, x6 = -4.5;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ax4;
  adouble ax5;
  adouble ax6;
  double y1;
  adouble ay1;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;
  ax4 <<= x4;
  ax5 <<= x5;
  ax6 <<= x6;

  ay1 = 0.5 *
        (ax1 * ax1 + ax2 * ax2 + ax3 * ax3 + ax4 * ax4 + ax5 * ax5 + ax6 * ax6);

  ay1 >>= y1;
  trace_off();

  double y1x1Derivative = x1;
  double y1x2Derivative = x2;
  double y1x3Derivative = x3;
  double y1x4Derivative = x4;
  double y1x5Derivative = x5;
  double y1x6Derivative = x6;

  double y1x1x1Derivative = 1.;
  double y1x1x2Derivative = 0.;
  double y1x1x3Derivative = 0.;
  double y1x1x4Derivative = 0.;
  double y1x1x5Derivative = 0.;
  double y1x1x6Derivative = 0.;
  double y1x2x1Derivative = 0.;
  double y1x2x2Derivative = 1.;
  double y1x2x3Derivative = 0.;
  double y1x2x4Derivative = 0.;
  double y1x2x5Derivative = 0.;
  double y1x2x6Derivative = 0.;
  double y1x3x1Derivative = 0.;
  double y1x3x2Derivative = 0.;
  double y1x3x3Derivative = 1.;
  double y1x3x4Derivative = 0.;
  double y1x3x5Derivative = 0.;
  double y1x3x6Derivative = 0.;
  double y1x4x1Derivative = 0.;
  double y1x4x2Derivative = 0.;
  double y1x4x3Derivative = 0.;
  double y1x4x4Derivative = 1.;
  double y1x4x5Derivative = 0.;
  double y1x4x6Derivative = 0.;
  double y1x5x1Derivative = 0.;
  double y1x5x2Derivative = 0.;
  double y1x5x3Derivative = 0.;
  double y1x5x4Derivative = 0.;
  double y1x5x5Derivative = 1.;
  double y1x5x6Derivative = 0.;
  double y1x6x1Derivative = 0.;
  double y1x6x2Derivative = 0.;
  double y1x6x3Derivative = 0.;
  double y1x6x4Derivative = 0.;
  double y1x6x5Derivative = 0.;
  double y1x6x6Derivative = 1.;

  double *x = myalloc1(6);
  double *xd = myalloc1(6);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  x[0] = 1.5;
  x[1] = -1.5;
  x[2] = 3.;
  x[3] = -3.;
  x[4] = 4.5;
  x[5] = -4.5;
  xd[0] = 1.;
  xd[1] = 0.;
  xd[2] = 0.;
  xd[3] = 0.;
  xd[4] = 0.;
  xd[5] = 0.;

  fos_forward(tapeId3, 1, 6, 2, x, xd, y, yd);

  double *u = myalloc1(1);
  double **Z = myalloc2(6, 2);

  u[0] = 1.;

  hos_reverse(tapeId3, 1, 6, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][0] == y1x4Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[4][0] == y1x5Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[5][0] == y1x6Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][1] == y1x1x4Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[4][1] == y1x1x5Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[5][1] == y1x1x6Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;
  xd[2] = 0.;
  xd[3] = 0.;
  xd[4] = 0.;
  xd[5] = 0.;

  fos_forward(tapeId3, 1, 6, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 6, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][0] == y1x4Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[4][0] == y1x5Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[5][0] == y1x6Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x2x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][1] == y1x2x4Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[4][1] == y1x2x5Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[5][1] == y1x2x6Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 0.;
  xd[2] = 1.;
  xd[3] = 0.;
  xd[4] = 0.;
  xd[5] = 0.;

  fos_forward(tapeId3, 1, 6, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 6, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][0] == y1x4Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[4][0] == y1x5Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[5][0] == y1x6Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x3x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][1] == y1x3x4Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[4][1] == y1x3x5Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[5][1] == y1x3x6Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 0.;
  xd[2] = 0.;
  xd[3] = 1.;
  xd[4] = 0.;
  xd[5] = 0.;

  fos_forward(tapeId3, 1, 6, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 6, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][0] == y1x4Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[4][0] == y1x5Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[5][0] == y1x6Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x4x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x4x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x4x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][1] == y1x4x4Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[4][1] == y1x4x5Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[5][1] == y1x4x6Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 0.;
  xd[2] = 0.;
  xd[3] = 0.;
  xd[4] = 1.;
  xd[5] = 0.;

  fos_forward(tapeId3, 1, 6, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 6, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][0] == y1x4Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[4][0] == y1x5Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[5][0] == y1x6Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x5x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x5x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x5x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][1] == y1x5x4Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[4][1] == y1x5x5Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[5][1] == y1x5x6Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 0.;
  xd[2] = 0.;
  xd[3] = 0.;
  xd[4] = 0.;
  xd[5] = 1.;

  fos_forward(tapeId3, 1, 6, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 6, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][0] == y1x4Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[4][0] == y1x5Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[5][0] == y1x6Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x6x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x6x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x6x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][1] == y1x6x4Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[4][1] == y1x6x5Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[5][1] == y1x6x6Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_CASE(customConstant_HOS_Reverse) {
  double x1 = 1.5, x2 = -1.5, x3 = 3., x4 = -3., x5 = 4.5, x6 = -4.5;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  adouble ax3;
  adouble ax4;
  adouble ax5;
  adouble ax6;
  double y1;
  adouble ay1;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;
  ax4 <<= x4;
  ax5 <<= x5;
  ax6 <<= x6;

  ay1 = 0.001001;

  ay1 >>= y1;
  trace_off();

  double y1x1Derivative = 0.;
  double y1x2Derivative = 0.;
  double y1x3Derivative = 0.;
  double y1x4Derivative = 0.;
  double y1x5Derivative = 0.;
  double y1x6Derivative = 0.;

  double y1x1x1Derivative = 0.;
  double y1x1x2Derivative = 0.;
  double y1x1x3Derivative = 0.;
  double y1x1x4Derivative = 0.;
  double y1x1x5Derivative = 0.;
  double y1x1x6Derivative = 0.;
  double y1x2x1Derivative = 0.;
  double y1x2x2Derivative = 0.;
  double y1x2x3Derivative = 0.;
  double y1x2x4Derivative = 0.;
  double y1x2x5Derivative = 0.;
  double y1x2x6Derivative = 0.;
  double y1x3x1Derivative = 0.;
  double y1x3x2Derivative = 0.;
  double y1x3x3Derivative = 0.;
  double y1x3x4Derivative = 0.;
  double y1x3x5Derivative = 0.;
  double y1x3x6Derivative = 0.;
  double y1x4x1Derivative = 0.;
  double y1x4x2Derivative = 0.;
  double y1x4x3Derivative = 0.;
  double y1x4x4Derivative = 0.;
  double y1x4x5Derivative = 0.;
  double y1x4x6Derivative = 0.;
  double y1x5x1Derivative = 0.;
  double y1x5x2Derivative = 0.;
  double y1x5x3Derivative = 0.;
  double y1x5x4Derivative = 0.;
  double y1x5x5Derivative = 0.;
  double y1x5x6Derivative = 0.;
  double y1x6x1Derivative = 0.;
  double y1x6x2Derivative = 0.;
  double y1x6x3Derivative = 0.;
  double y1x6x4Derivative = 0.;
  double y1x6x5Derivative = 0.;
  double y1x6x6Derivative = 0.;

  double *x = myalloc1(6);
  double *xd = myalloc1(6);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  x[0] = 1.5;
  x[1] = -1.5;
  x[2] = 3.;
  x[3] = -3.;
  x[4] = 4.5;
  x[5] = -4.5;
  xd[0] = 1.;
  xd[1] = 0.;
  xd[2] = 0.;
  xd[3] = 0.;
  xd[4] = 0.;
  xd[5] = 0.;

  fos_forward(tapeId3, 1, 6, 2, x, xd, y, yd);

  double *u = myalloc1(1);
  double **Z = myalloc2(6, 2);

  u[0] = 1.;

  hos_reverse(tapeId3, 1, 6, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][0] == y1x4Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[4][0] == y1x5Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[5][0] == y1x6Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][1] == y1x1x4Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[4][1] == y1x1x5Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[5][1] == y1x1x6Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;
  xd[2] = 0.;
  xd[3] = 0.;
  xd[4] = 0.;
  xd[5] = 0.;

  fos_forward(tapeId3, 1, 6, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 6, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][0] == y1x4Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[4][0] == y1x5Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[5][0] == y1x6Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x2x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][1] == y1x2x4Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[4][1] == y1x2x5Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[5][1] == y1x2x6Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 0.;
  xd[2] = 1.;
  xd[3] = 0.;
  xd[4] = 0.;
  xd[5] = 0.;

  fos_forward(tapeId3, 1, 6, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 6, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][0] == y1x4Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[4][0] == y1x5Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[5][0] == y1x6Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x3x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][1] == y1x3x4Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[4][1] == y1x3x5Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[5][1] == y1x3x6Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 0.;
  xd[2] = 0.;
  xd[3] = 1.;
  xd[4] = 0.;
  xd[5] = 0.;

  fos_forward(tapeId3, 1, 6, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 6, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][0] == y1x4Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[4][0] == y1x5Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[5][0] == y1x6Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x4x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x4x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x4x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][1] == y1x4x4Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[4][1] == y1x4x5Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[5][1] == y1x4x6Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 0.;
  xd[2] = 0.;
  xd[3] = 0.;
  xd[4] = 1.;
  xd[5] = 0.;

  fos_forward(tapeId3, 1, 6, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 6, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][0] == y1x4Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[4][0] == y1x5Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[5][0] == y1x6Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x5x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x5x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x5x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][1] == y1x5x4Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[4][1] == y1x5x5Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[5][1] == y1x5x6Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 0.;
  xd[2] = 0.;
  xd[3] = 0.;
  xd[4] = 0.;
  xd[5] = 1.;

  fos_forward(tapeId3, 1, 6, 2, x, xd, y, yd);

  hos_reverse(tapeId3, 1, 6, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][0] == y1x4Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[4][0] == y1x5Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[5][0] == y1x6Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x6x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x6x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x6x3Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[3][1] == y1x6x4Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[4][1] == y1x6x5Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[5][1] == y1x6x6Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_CASE(customSphereCoord_HOS_Reverse) {
  double x1 = 21.87, x2 = std::acos(0) - 0.01, x3 = 0.5 * std::acos(0);
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  adouble ax3;
  double y1, y2, y3;
  adouble ay1;
  adouble ay2;
  adouble ay3;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ay1 = ax1 * cos(ax2) * sin(ax3);
  ay2 = ax1 * sin(ax2) * sin(ax3);
  ay3 = ax1 * cos(ax3);

  ay1 >>= y1;
  ay2 >>= y2;
  ay3 >>= y3;
  trace_off();

  double y1x1Derivative = std::cos(x2) * std::sin(x3);
  double y1x2Derivative = -x1 * std::sin(x2) * std::sin(x3);
  double y1x3Derivative = x1 * std::cos(x2) * std::cos(x3);
  double y2x1Derivative = std::sin(x2) * std::sin(x3);
  double y2x2Derivative = x1 * std::cos(x2) * std::sin(x3);
  double y2x3Derivative = x1 * std::sin(x2) * std::cos(x3);
  double y3x1Derivative = std::cos(x3);
  double y3x2Derivative = 0.;
  double y3x3Derivative = -x1 * std::sin(x3);

  double y1x1x1Derivative = 0.;
  double y1x1x2Derivative = -std::sin(x2) * std::sin(x3);
  double y1x1x3Derivative = std::cos(x2) * std::cos(x3);
  double y2x1x1Derivative = 0.;
  double y2x1x2Derivative = std::cos(x2) * std::sin(x3);
  double y2x1x3Derivative = std::sin(x2) * std::cos(x3);
  double y3x1x1Derivative = 0.;
  double y3x1x2Derivative = 0.;
  double y3x1x3Derivative = -std::sin(x3);
  double y1x2x1Derivative = -std::sin(x2) * std::sin(x3);
  double y1x2x2Derivative = -x1 * std::cos(x2) * std::sin(x3);
  double y1x2x3Derivative = -x1 * std::sin(x2) * std::cos(x3);
  double y2x2x1Derivative = std::cos(x2) * std::sin(x3);
  double y2x2x2Derivative = -x1 * std::sin(x2) * std::sin(x3);
  double y2x2x3Derivative = x1 * std::cos(x2) * std::cos(x3);
  double y3x2x1Derivative = 0.;
  double y3x2x2Derivative = 0.;
  double y3x2x3Derivative = 0.;
  double y1x3x1Derivative = std::cos(x2) * std::cos(x3);
  double y1x3x2Derivative = -x1 * std::sin(x2) * std::cos(x3);
  double y1x3x3Derivative = -x1 * std::cos(x2) * std::sin(x3);
  double y2x3x1Derivative = std::sin(x2) * std::cos(x3);
  double y2x3x2Derivative = x1 * std::cos(x2) * std::cos(x3);
  double y2x3x3Derivative = -x1 * std::sin(x2) * std::sin(x3);
  double y3x3x1Derivative = -std::sin(x3);
  double y3x3x2Derivative = 0.;
  double y3x3x3Derivative = -x1 * std::cos(x3);

  double *x = myalloc1(3);
  double *xd = myalloc1(3);
  double *y = myalloc1(3);
  double *yd = myalloc1(3);

  x[0] = 21.87;
  x[1] = std::acos(0) - 0.01;
  x[2] = 0.5 * std::acos(0);
  xd[0] = 1.;
  xd[1] = 0.;
  xd[2] = 0.;

  fos_forward(tapeId3, 3, 3, 2, x, xd, y, yd);

  double *u = myalloc1(3);
  double **Z = myalloc2(3, 2);

  u[0] = 1.;
  u[1] = 0.;
  u[2] = 0.;

  hos_reverse(tapeId3, 3, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x1x3Derivative, tt::tolerance(tol));

  u[0] = 0.;
  u[1] = 1.;
  u[2] = 0.;

  hos_reverse(tapeId3, 3, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y2x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y2x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y2x1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y2x1x3Derivative, tt::tolerance(tol));

  u[0] = 0.;
  u[1] = 0.;
  u[2] = 1.;

  hos_reverse(tapeId3, 3, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y3x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y3x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y3x1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y3x1x3Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;
  xd[2] = 0.;

  fos_forward(tapeId3, 3, 3, 2, x, xd, y, yd);

  u[0] = 1.;
  u[1] = 0.;
  u[2] = 0.;

  hos_reverse(tapeId3, 3, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x2x3Derivative, tt::tolerance(tol));

  u[0] = 0.;
  u[1] = 1.;
  u[2] = 0.;

  hos_reverse(tapeId3, 3, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y2x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y2x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y2x2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y2x2x3Derivative, tt::tolerance(tol));

  u[0] = 0.;
  u[1] = 0.;
  u[2] = 1.;

  hos_reverse(tapeId3, 3, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y3x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y3x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y3x2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y3x2x3Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 0.;
  xd[2] = 1.;

  fos_forward(tapeId3, 3, 3, 2, x, xd, y, yd);

  u[0] = 1.;
  u[1] = 0.;
  u[2] = 0.;

  hos_reverse(tapeId3, 3, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x3x3Derivative, tt::tolerance(tol));

  u[0] = 0.;
  u[1] = 1.;
  u[2] = 0.;

  hos_reverse(tapeId3, 3, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y2x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y2x3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y2x3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y2x3x3Derivative, tt::tolerance(tol));

  u[0] = 0.;
  u[1] = 0.;
  u[2] = 1.;

  hos_reverse(tapeId3, 3, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y3x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y3x3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y3x3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y3x3x3Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_CASE(customCylinderCoord_HOS_Reverse) {
  double x1 = 21.87, x2 = std::acos(0) - 0.01, x3 = 105.05;
  setCurrentTape(tapeId3);
  adouble ax1;
  adouble ax2;
  adouble ax3;
  double y1, y2, y3;
  adouble ay1;
  adouble ay2;
  adouble ay3;

  trace_on(tapeId3, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ay1 = ax1 * cos(ax2);
  ay2 = ax1 * sin(ax2);
  ay3 = ax3;

  ay1 >>= y1;
  ay2 >>= y2;
  ay3 >>= y3;
  trace_off();

  double y1x1Derivative = std::cos(x2);
  double y1x2Derivative = -x1 * std::sin(x2);
  double y1x3Derivative = 0.;
  double y2x1Derivative = std::sin(x2);
  double y2x2Derivative = x1 * std::cos(x2);
  double y2x3Derivative = 0.;
  double y3x1Derivative = 0.;
  double y3x2Derivative = 0.;
  double y3x3Derivative = 1.;

  double y1x1x1Derivative = 0.;
  double y1x1x2Derivative = -std::sin(x2);
  double y1x1x3Derivative = 0.;
  double y2x1x1Derivative = 0.;
  double y2x1x2Derivative = std::cos(x2);
  double y2x1x3Derivative = 0.;
  double y3x1x1Derivative = 0.;
  double y3x1x2Derivative = 0.;
  double y3x1x3Derivative = 0.;
  double y1x2x1Derivative = -std::sin(x2);
  double y1x2x2Derivative = -x1 * std::cos(x2);
  double y1x2x3Derivative = 0.;
  double y2x2x1Derivative = std::cos(x2);
  double y2x2x2Derivative = -x1 * std::sin(x2);
  double y2x2x3Derivative = 0.;
  double y3x2x1Derivative = 0.;
  double y3x2x2Derivative = 0.;
  double y3x2x3Derivative = 0.;
  double y1x3x1Derivative = 0.;
  double y1x3x2Derivative = 0.;
  double y1x3x3Derivative = 0.;
  double y2x3x1Derivative = 0.;
  double y2x3x2Derivative = 0.;
  double y2x3x3Derivative = 0.;
  double y3x3x1Derivative = 0.;
  double y3x3x2Derivative = 0.;
  double y3x3x3Derivative = 0.;

  double *x = myalloc1(3);
  double *xd = myalloc1(3);
  double *y = myalloc1(3);
  double *yd = myalloc1(3);

  x[0] = 21.87;
  x[1] = std::acos(0) - 0.01;
  x[2] = 105.05;
  xd[0] = 1.;
  xd[1] = 0.;
  xd[2] = 0.;

  fos_forward(tapeId3, 3, 3, 2, x, xd, y, yd);

  double *u = myalloc1(3);
  double **Z = myalloc2(3, 2);

  u[0] = 1.;
  u[1] = 0.;
  u[2] = 0.;

  hos_reverse(tapeId3, 3, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x1x3Derivative, tt::tolerance(tol));

  u[0] = 0.;
  u[1] = 1.;
  u[2] = 0.;

  hos_reverse(tapeId3, 3, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y2x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y2x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y2x1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y2x1x3Derivative, tt::tolerance(tol));

  u[0] = 0.;
  u[1] = 0.;
  u[2] = 1.;

  hos_reverse(tapeId3, 3, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y3x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y3x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y3x1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y3x1x3Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;
  xd[2] = 0.;

  fos_forward(tapeId3, 3, 3, 2, x, xd, y, yd);

  u[0] = 1.;
  u[1] = 0.;
  u[2] = 0.;

  hos_reverse(tapeId3, 3, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x2x3Derivative, tt::tolerance(tol));

  u[0] = 0.;
  u[1] = 1.;
  u[2] = 0.;

  hos_reverse(tapeId3, 3, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y2x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y2x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y2x2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y2x2x3Derivative, tt::tolerance(tol));

  u[0] = 0.;
  u[1] = 0.;
  u[2] = 1.;

  hos_reverse(tapeId3, 3, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y3x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y3x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y3x2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y3x2x3Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 0.;
  xd[2] = 1.;

  fos_forward(tapeId3, 3, 3, 2, x, xd, y, yd);

  u[0] = 1.;
  u[1] = 0.;
  u[2] = 0.;

  hos_reverse(tapeId3, 3, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y1x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y1x3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y1x3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y1x3x3Derivative, tt::tolerance(tol));

  u[0] = 0.;
  u[1] = 1.;
  u[2] = 0.;

  hos_reverse(tapeId3, 3, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y2x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y2x3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y2x3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y2x3x3Derivative, tt::tolerance(tol));

  u[0] = 0.;
  u[1] = 0.;
  u[2] = 1.;

  hos_reverse(tapeId3, 3, 3, 1, u, Z);

  BOOST_TEST(Z[0][0] == y3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == y3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][0] == y3x3Derivative, tt::tolerance(tol));

  BOOST_TEST(Z[0][1] == y3x3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == y3x3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[2][1] == y3x3x3Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree1(u);
  myfree2(Z);
}

BOOST_AUTO_TEST_SUITE_END()

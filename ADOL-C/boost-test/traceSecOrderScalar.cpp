#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

#include <adolc/adolc.h>

#include "const.h"

BOOST_AUTO_TEST_SUITE( trace_sec_order )


/**************************************/
/* Tests for ADOL-C trace scalar mode */
/* drivers hos_forward, hessian       */
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

/* Tested function: 2.*x*x*x
 * First derivative: 2.*3.*x*x
 * Second derivative: 2.*3.*2.*x
 */
BOOST_AUTO_TEST_CASE(CustomCube_HOS)
{
  double x = 3.;
  adouble ax;
  double y;
  adouble ay;

  trace_on(1, 1);
  ax <<= x;

  ay = 2.*ax*ax*ax;

  ay >>= y;
  trace_off();

  // Calculate primitive value analytically for testing.
  double yprim = 2.*x*x*x;
  // Calculate first and second derivative analytically for testing.
  double** yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = 2.*3.*x*x;
  yDerivative[0][1] = 2.*3.*x*x + 0.5*(2.*3.*2.*x);

  double** X;
  X = myalloc2(1, 2);
  X[0][0] = 1.;
  X[0][1] = 1.;

  double** Y;
  Y = myalloc2(1, 2);

  // Signature: hos_forward(tag, m, n, d, keep, x[n], X[n][d], y[m], Y[m][d])
  hos_forward(1, 1, 1, 2, 1, &x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double** H;
  H = myalloc2(1, 1);

  // Calculate Hessian matrix analytically:
  double yxxDerivative = 2.*3.*2.*x;

  hessian(1, 1, &x, H);

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
BOOST_AUTO_TEST_CASE(CustomTrigProd_HOS)
{
  double x1 = 1.3, x2 = 3.1;
  adouble ax1, ax2;
  double y;
  adouble ay;

  trace_on(1, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay = cos(ax1)*sin(ax2);

  ay >>= y;
  trace_off();

  double yprim = std::cos(x1)*std::sin(x2);

  double** yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = -std::sin(x1)*std::sin(x2);
  yDerivative[0][1] = std::cos(x1)*std::cos(x2)
                      + 0.5*(-std::cos(x1))*std::sin(x2);

  double* x;
  x = myalloc1(2);
  x[0] = x1;
  x[1] = x2;

  double** X;
  X = myalloc2(2, 2);
  X[0][0] = 1.;
  X[0][1] = 0.;
  X[1][0] = 0.;
  X[1][1] = 1.;

  double** Y;
  Y = myalloc2(1, 2);

  hos_forward(1, 1, 2, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double** H;
  H = myalloc2(2, 2);

  double yx1x1Derivative = -std::cos(x1)*std::sin(x2);
  double yx1x2Derivative = -std::sin(x1)*std::cos(x2);
  double yx2x2Derivative = -std::cos(x1)*std::sin(x2);

  hessian(1, 2, x, H);

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
BOOST_AUTO_TEST_CASE(CustomTrigPow_HOS)
{
  double x1 = 1.1, x2 = 4.53, x3 = -3.03;
  adouble ax1, ax2, ax3;
  double y;
  adouble ay;

  trace_on(1, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ay = pow(ax1, ax2)*exp(2.*ax3);

  ay >>= y;
  trace_off();

  double yprim = std::pow(x1, x2)*std::exp(2.*x3);

  double** yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = x2*std::pow(x1, x2 - 1)*std::exp(2.*x3)
                      + 0.1*std::pow(x1, x2)*std::log(x1)*std::exp(2.*x3);
  yDerivative[0][1] = std::pow(x1, x2)*std::log(x1)*std::exp(2.*x3)
                      + 0.2*2.*std::pow(x1, x2)*std::exp(2.*x3)
                      + 0.5*(x2*(x2 - 1)*std::pow(x1, x2 - 2)*std::exp(2.*x3)
                             + 0.1*x2*std::pow(x1, x2 - 1)*std::log(x1)
                               *std::exp(2.*x3)
                             + 0.1*std::pow(x1, x2 - 1)*std::exp(2.*x3)
                             + 0.1*std::pow(x1, x2 - 1)*std::exp(2.*x3)
                             + 0.1*x2*std::pow(x1, x2 - 1)*std::log(x1)
                               *std::exp(2.*x3)
                             + 0.01*std::pow(x1, x2)
                               *std::pow(std::log(x1), 2)*std::exp(2.*x3));

  double* x;
  x = myalloc1(3);
  x[0] = x1;
  x[1] = x2;
  x[2] = x3;

  double** X;
  X = myalloc2(3, 2);
  X[0][0] = 1.;
  X[1][0] = 0.1;
  X[2][0] = 0.;
  X[0][1] = 0.;
  X[1][1] = 1.;
  X[2][1] = 0.2;

  double** Y;
  Y = myalloc2(1, 2);

  hos_forward(1, 1, 3, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double** H;
  H = myalloc2(3, 3);

  double yx1x1Derivative = x2*(x2 - 1)*std::pow(x1, x2 - 2)*std::exp(2.*x3);
  double yx2x1Derivative = x2*std::pow(x1, x2 - 1)*std::log(x1)*std::exp(2.*x3)
                           + std::pow(x1, x2 - 1)*std::exp(2.*x3);
  double yx3x1Derivative = 2.*x2*std::pow(x1, x2 - 1)*std::exp(2.*x3);
  double yx2x2Derivative = std::pow(x1, x2)*std::pow(std::log(x1), 2)
                           *std::exp(2.*x3);
  double yx3x2Derivative = 2.*std::pow(x1, x2)*std::log(x1)*std::exp(2.*x3);
  double yx3x3Derivative = 4.*std::pow(x1, x2)*std::exp(2.*x3);

  hessian(1, 3, x, H);

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
BOOST_AUTO_TEST_CASE(CustomHyperbProd_HOS)
{
  double x1 = 2.22, x2 = -2.22;
  adouble ax1, ax2;
  double y;
  adouble ay;

  trace_on(1, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay = cosh(2.*ax1)*sinh(3.*ax2);

  ay >>= y;
  trace_off();

  double yprim = std::cosh(2.*x1)*std::sinh(3.*x2);

  double** yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = 2.*std::sinh(2.*x1)*std::sinh(3.*x2);
  yDerivative[0][1] = 3.*std::cosh(2.*x1)*std::cosh(3.*x2)
                      + 0.5*4.*std::cosh(2.*x1)*std::sinh(3.*x2);

  double* x;
  x = myalloc1(2);
  x[0] = x1;
  x[1] = x2;

  double** X;
  X = myalloc2(2, 2);
  X[0][0] = 1.;
  X[0][1] = 0.;
  X[1][0] = 0.;
  X[1][1] = 1.;

  double** Y;
  Y = myalloc2(1, 2);

  hos_forward(1, 1, 2, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double** H;
  H = myalloc2(2, 2);

  double yx1x1Derivative = 4.*std::cosh(2.*x1)*std::sinh(3.*x2);
  double yx1x2Derivative = 6.*std::sinh(2.*x1)*std::cosh(3.*x2);
  double yx2x2Derivative = 9.*std::cosh(2.*x1)*std::sinh(3.*x2);

  hessian(1, 2, x, H);

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
BOOST_AUTO_TEST_CASE(CustomPowTrig_HOS)
{
  double x1 = 0.531, x2 = 3.12;
  adouble ax1, ax2;
  double y;
  adouble ay;

  trace_on(1, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay = pow(sin(ax1), cos(ax2));

  ay >>= y;
  trace_off();

  double yprim = std::pow(std::sin(x1), std::cos(x2));

  double** yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = std::pow(std::sin(x1), std::cos(x2))
                      * std::cos(x2)*std::cos(x1)/std::sin(x1);
  yDerivative[0][1] = -std::pow(std::sin(x1), std::cos(x2))
                      *std::sin(x2)*std::log(std::sin(x1))
                      + 0.5*(std::pow(std::sin(x1), std::cos(x2))*std::cos(x2)
                             *(-1 + std::pow(std::cos(x1)/std::sin(x1), 2)
                               *(std::cos(x2) - 1)));

  double* x;
  x = myalloc1(2);
  x[0] = x1;
  x[1] = x2;

  double** X;
  X = myalloc2(2, 2);
  X[0][0] = 1.;
  X[0][1] = 0.;
  X[1][0] = 0.;
  X[1][1] = 1.;

  double** Y;
  Y = myalloc2(1, 2);

  hos_forward(1, 1, 2, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double** H;
  H = myalloc2(2, 2);

  double yx1x1Derivative = std::pow(std::sin(x1), std::cos(x2))*std::cos(x2)
                           *(-1 + std::pow(std::cos(x1)/std::sin(x1), 2)
                             *(std::cos(x2) - 1));
  double yx1x2Derivative = -std::pow(std::sin(x1), std::cos(x2))*std::sin(x2)
                           *(std::cos(x1)/std::sin(x1) + std::log(std::sin(x1))
                             *std::cos(x2)*std::cos(x1)/std::sin(x1));
  double yx2x2Derivative = std::pow(std::sin(x1), std::cos(x2))
                           *std::log(std::sin(x1))*(-std::cos(x2)
                            +std::pow(std::sin(x2), 2)*std::log(std::sin(x1)));

  hessian(1, 2, x, H);

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
BOOST_AUTO_TEST_CASE(CustomPow_HOS)
{
  double x1 = 1.04, x2 = -2.01;
  adouble ax1, ax2;
  double y;
  adouble ay;

  trace_on(1, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay = pow(ax1, ax2);

  ay >>= y;
  trace_off();

  double yprim = std::pow(x1, x2);

  double** yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = x2*std::pow(x1, x2 - 1);
  yDerivative[0][1] = std::pow(x1, x2)*std::log(x1)
                      + 0.5*x2*(x2 - 1)*std::pow(x1, x2 - 2);

  double* x;
  x = myalloc1(2);
  x[0] = x1;
  x[1] = x2;

  double** X;
  X = myalloc2(2, 2);
  X[0][0] = 1.;
  X[0][1] = 0.;
  X[1][0] = 0.;
  X[1][1] = 1.;

  double** Y;
  Y = myalloc2(1, 2);

  hos_forward(1, 1, 2, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double** H;
  H = myalloc2(2, 2);

  double yx1x1Derivative = x2*(x2 - 1)*pow(x1, x2 - 2);
  double yx1x2Derivative = std::pow(x1, x2 - 1)*(1 + x2*std::log(x1));
  double yx2x2Derivative = std::pow(x1, x2)*std::pow(std::log(x1), 2);

  hessian(1, 2, x, H);

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
BOOST_AUTO_TEST_CASE(CustomExpSum_HOS)
{
  double x1 = -1.1, x2 = -4.53, x3 = 3.03, x4 = 0.;
  adouble ax1, ax2, ax3, ax4;
  double y;
  adouble ay;

  trace_on(1, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;
  ax4 <<= x4;

  ay = exp(ax1 + 3.*ax2 + 5.*ax3 + 7.*ax4);

  ay >>= y;
  trace_off();

  double yprim = std::exp(x1 + 3.*x2 + 5.*x3 + 7.*x4);

  double** yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = std::exp(x1 + 3.*x2 +5.*x3 + 7.*x4)
                      *(1. + 0.1*3. -0.01*7.);
  yDerivative[0][1] = std::exp(x1 + 3.*x2 +5.*x3 + 7.*x4)
                      *(3. + 0.2*5. + 0.5*(1. + 0.3 - 0.07
                        + 0.1*(3. + 0.9 - 0.21) - 0.01*(7. + 2.1 - 0.49)));

  double* x;
  x = myalloc1(4);
  x[0] = x1;
  x[1] = x2;
  x[2] = x3;
  x[3] = x4;

  double** X;
  X = myalloc2(4, 2);
  X[0][0] = 1.;
  X[1][0] = 0.1;
  X[2][0] = 0.;
  X[3][0] = -0.01;
  X[0][1] = 0.;
  X[1][1] = 1.;
  X[2][1] = 0.2;
  X[3][1] = 0.;

  double** Y;
  Y = myalloc2(1, 2);

  hos_forward(1, 1, 4, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double** H;
  H = myalloc2(4, 4);

  double yx1x1Derivative = std::exp(x1 + 3.*x2 +5.*x3 + 7.*x4);
  double yx2x1Derivative = 3.*std::exp(x1 + 3.*x2 +5.*x3 + 7.*x4);
  double yx3x1Derivative = 5.*std::exp(x1 + 3.*x2 +5.*x3 + 7.*x4);
  double yx4x1Derivative = 7.*std::exp(x1 + 3.*x2 +5.*x3 + 7.*x4);
  double yx2x2Derivative = 9.*std::exp(x1 + 3.*x2 +5.*x3 + 7.*x4);
  double yx3x2Derivative = 15.*std::exp(x1 + 3.*x2 +5.*x3 + 7.*x4);
  double yx4x2Derivative = 21.*std::exp(x1 + 3.*x2 +5.*x3 + 7.*x4);
  double yx3x3Derivative = 25.*std::exp(x1 + 3.*x2 +5.*x3 + 7.*x4);
  double yx4x3Derivative = 35.*std::exp(x1 + 3.*x2 +5.*x3 + 7.*x4);
  double yx4x4Derivative = 49.*std::exp(x1 + 3.*x2 +5.*x3 + 7.*x4);

  hessian(1, 4, x, H);

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

#if defined(ATRIG_ERF)
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
 *                      exp(tanh(x1)*erf(x2))*tanh(x1)*exp(-x2*x2)
 *                      *(4*tanh(x1)/acos(-1) - 4*x2/sqrt(acos(-1)))
 *                     )
 */
BOOST_AUTO_TEST_CASE(CustomHypErf_HOS)
{
  double x1 = 5.55, x2 = 9.99;
  adouble ax1, ax2;
  double y;
  adouble ay;

  trace_on(1, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay = exp(tanh(ax1)*erf(ax2));

  ay >>= y;
  trace_off();

  double yprim = std::exp(std::tanh(x1)*std::erf(x2));

  double** yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = std::exp(std::tanh(x1)*std::erf(x2))
                      *(1 - std::pow(std::tanh(x1), 2))*std::erf(x2);
  yDerivative[0][1] = std::exp(std::tanh(x1)*std::erf(x2))*std::tanh(x1)
                      *std::exp(-x2*x2)*2/std::sqrt(std::acos(-1))
                      + 0.5*std::exp(std::tanh(x1)*std::erf(x2))
                      *(1 - std::pow(std::tanh(x1), 2))*std::erf(x2)
                      *((1 - std::pow(std::tanh(x1), 2))*std::erf(x2)
                        - 2*std::tanh(x1));

  double* x;
  x = myalloc1(2);
  x[0] = x1;
  x[1] = x2;

  double** X;
  X = myalloc2(2, 2);
  X[0][0] = 1.;
  X[0][1] = 0.;
  X[1][0] = 0.;
  X[1][1] = 1.;

  double** Y;
  Y = myalloc2(1, 2);

  hos_forward(1, 1, 2, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double** H;
  H = myalloc2(2, 2);

  double yx1x1Derivative = std::exp(std::tanh(x1)*std::erf(x2))
                           *(1 - std::pow(std::tanh(x1), 2))*std::erf(x2)
                           *((1 - std::pow(std::tanh(x1), 2))*std::erf(x2)
                             - 2*std::tanh(x1));
  double yx1x2Derivative = std::exp(std::tanh(x1)*std::erf(x2))
                           *std::exp(-x2*x2)*(1 - std::pow(std::tanh(x1), 2))
                           *2/std::sqrt(std::acos(-1))
                           *(1 + std::tanh(x1)*std::erf(x2));
  double yx2x2Derivative = std::exp(std::tanh(x1)*std::erf(x2))*std::tanh(x1)
                           *std::exp(-2*x2*x2)*(4*std::tanh(x1)/std::acos(-1)
                             - 4*x2*std::exp(x2*x2)/std::sqrt(std::acos(-1)));

  hessian(1, 2, x, H);

  BOOST_TEST(yx1x1Derivative == H[0][0], tt::tolerance(tol));
  BOOST_TEST(yx1x2Derivative == H[1][0], tt::tolerance(tol));
  BOOST_TEST(yx2x2Derivative == H[1][1], tt::tolerance(tol));

  myfree1(x);
  myfree2(yDerivative);
  myfree2(X);
  myfree2(Y);
  myfree2(H);
}
#endif

/* Tested function: (pow(cosh(x1), 2) - pow(sinh(x1), 2))*atan(x2)
 * First derivatives: (0, 1./(1. + x2*x2)
 *                    )
 * Second derivatives: (0, 0,
 *                      0, -2.*x2/pow(1. + x2*x2, 2)
 *                     )
 */
BOOST_AUTO_TEST_CASE(CustomHypAtan_HOS)
{
  double x1 = 7.19, x2 = -4.32;
  adouble ax1, ax2;
  double y;
  adouble ay;

  trace_on(1, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay = (pow(cosh(ax1), 2) - pow(sinh(ax1), 2))*atan(ax2);

  ay >>= y;
  trace_off();

  double yprim = (std::pow(std::cosh(x1), 2) - std::pow(std::sinh(x1), 2))
                 *std::atan(x2);

  double** yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = 0.;
  yDerivative[0][1] = 1./(1. + x2*x2) + 0.5*0.;

  double* x;
  x = myalloc1(2);
  x[0] = x1;
  x[1] = x2;

  double** X;
  X = myalloc2(2, 2);
  X[0][0] = 1.;
  X[0][1] = 0.;
  X[1][0] = 0.;
  X[1][1] = 1.;

  double** Y;
  Y = myalloc2(1, 2);

  hos_forward(1, 1, 2, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double** H;
  H = myalloc2(2, 2);

  double yx1x1Derivative = 0.;
  double yx1x2Derivative = 0.;
  double yx2x2Derivative = -2.*x2/std::pow(1. + x2*x2, 2);

  hessian(1, 2, x, H);

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
BOOST_AUTO_TEST_CASE(CustomLongSum_HOS)
{
  double x1 = 99.99, x2 = std::exp(-0.44), x3 = std::sqrt(2);
  adouble ax1, ax2, ax3;
  double y;
  adouble ay;

  trace_on(1, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ay = 1. + ax1 + ax1*ax1 + ax2*ax2 + ax2*ax2*ax2
       + ax3*ax3*ax3 + ax3*ax3*ax3*ax3;

  ay >>= y;
  trace_off();

  double yprim = 1. + x1 + x1*x1 + x2*x2 + x2*x2*x2 + x3*x3*x3 + x3*x3*x3*x3;

  double** yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = 1. + 2.*x1 - 0.01*(3.*x3*x3 + 4.*x3*x3*x3);
  yDerivative[0][1] = 0.3*(2.*x2 + 3.*x2*x2)
                      + 0.5*(2. + 0.01*0.01*(6.*x3 + 12.*x3*x3));

  double* x;
  x = myalloc1(3);
  x[0] = x1;
  x[1] = x2;
  x[2] = x3;

  double** X;
  X = myalloc2(3, 2);
  X[0][0] = 1.;
  X[1][0] = 0.;
  X[2][0] = -0.01;
  X[0][1] = 0.;
  X[1][1] = 0.3;
  X[2][1] = 0.;

  double** Y;
  Y = myalloc2(1, 2);

  hos_forward(1, 1, 3, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double** H;
  H = myalloc2(3, 3);

  double yx1x1Derivative = 2.;
  double yx2x1Derivative = 0.;
  double yx3x1Derivative = 0.;
  double yx2x2Derivative = 2. + 6.*x2;
  double yx3x2Derivative = 0.;
  double yx3x3Derivative = 6.*x3 + 12.*x3*x3;

  hessian(1, 3, x, H);

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
BOOST_AUTO_TEST_CASE(CustomExpSqrt_HOS)
{
  double x1 = -0.77, x2 = 10.01, x3 = 0.99;
  adouble ax1, ax2, ax3;
  double y;
  adouble ay;

  trace_on(1, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ay = exp(ax1)*sqrt(2.*ax2)*pow(ax3, 2);

  ay >>= y;
  trace_off();

  double yprim = std::exp(x1)*std::sqrt(2.*x2)*std::pow(x3, 2);

  double** yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = std::exp(x1)*std::sqrt(2.*x2)*std::pow(x3, 2);
  yDerivative[0][1] = 2.*exp(x1)*pow(x3, 2)/sqrt(2.*x2)
                      + 0.5*exp(x1)*sqrt(2.*x2)*pow(x3, 2);

  double* x;
  x = myalloc1(3);
  x[0] = x1;
  x[1] = x2;
  x[2] = x3;

  double** X;
  X = myalloc2(3, 2);
  X[0][0] = 1.;
  X[1][0] = 0.;
  X[2][0] = 0.;
  X[0][1] = 0.;
  X[1][1] = 2.;
  X[2][1] = 0.;

  double** Y;
  Y = myalloc2(1, 2);

  hos_forward(1, 1, 3, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double** H;
  H = myalloc2(3, 3);

  double yx1x1Derivative = std::exp(x1)*std::sqrt(2.*x2)*std::pow(x3, 2);
  double yx2x1Derivative = std::exp(x1)*std::pow(x3, 2)/std::sqrt(2.*x2);
  double yx3x1Derivative = std::exp(x1)*std::sqrt(2.*x2)*2.*x3;
  double yx2x2Derivative = -std::exp(x1)*std::pow(x3, 2)
                           /std::pow(std::sqrt(2.*x2), 3);
  double yx3x2Derivative = std::exp(x1)*2.*x3/std::sqrt(2.*x2);
  double yx3x3Derivative = 2.*std::exp(x1)*std::sqrt(2.*x2);

  hessian(1, 3, x, H);

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

#if defined(ATRIG_ERF)
/* Tested function: 2.*acosh(cosh(x1*x1))*atanh(x2)
 * First derivatives: (4.*x1*atanh(x2), 2.*x1*x1/(1. - x2*x2)
 *                    )
 * Second derivatives: (4.*atanh(x2), 4.*x1/(1. - x2*x2),
 *                      4.*x1/(1. - x2*x2), 4.*x1*x1*x2/pow(1. - x2*x2, 2)
 *                     )
 */
BOOST_AUTO_TEST_CASE(CustomInvHyperb_HOS)
{
  double x1 = -3.03, x2 = 0.11;
  adouble ax1, ax2;
  double y;
  adouble ay;

  trace_on(1, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay = 2.*acosh(cosh(ax1*ax1))*atanh(ax2);

  ay >>= y;
  trace_off();

  double yprim = 2.*std::acosh(std::cosh(x1*x1))*std::atanh(x2);

  double** yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = 4.*x1*std::atanh(x2);
  yDerivative[0][1] = 3.*x1*x1/(1. - x2*x2) + 2.*std::atanh(x2);

  double* x;
  x = myalloc1(2);
  x[0] = x1;
  x[1] = x2;

  double** X;
  X = myalloc2(2, 2);
  X[0][0] = 1.;
  X[0][1] = 0.;
  X[1][0] = 0.;
  X[1][1] = 1.5;

  double** Y;
  Y = myalloc2(1, 2);

  hos_forward(1, 1, 2, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double** H;
  H = myalloc2(2, 2);

  double yx1x1Derivative = 4.*std::atanh(x2);
  double yx1x2Derivative = 4.*x1/(1. - x2*x2);
  double yx2x2Derivative = 4.*x1*x1*x2/std::pow(1. - x2*x2, 2);

  hessian(1, 2, x, H);

  BOOST_TEST(yx1x1Derivative == H[0][0], tt::tolerance(tol));
  BOOST_TEST(yx1x2Derivative == H[1][0], tt::tolerance(tol));
  BOOST_TEST(yx2x2Derivative == H[1][1], tt::tolerance(tol));

  myfree1(x);
  myfree2(yDerivative);
  myfree2(X);
  myfree2(Y);
  myfree2(H);
}
#endif

/* Tested function: fmax(fmin(x1, x2), fabs(x3))*x4
 * First derivatives: (0., 0., -x4, -x3)
 * Second derivatives: (0., 0., 0., 0.,
 *                      0., 0., 0., 0.,
 *                      0., 0., 0., -1.,
 *                      0., 0., -1., 0.)
 */
BOOST_AUTO_TEST_CASE(CustomFminFmaxFabs_HOS)
{
  double x1 = 1., x2 = 2.5, x3 = -4.5, x4 = -1.;
  adouble ax1, ax2, ax3, ax4;
  double y;
  adouble ay;

  trace_on(1, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;
  ax4 <<= x4;

  ay = fmax(fmin(ax1, ax2), fabs(ax3))*ax4;

  ay >>= y;
  trace_off();

  double yprim = std::fmax(std::fmin(x1, x2), std::fabs(x3))*x4;

  double** yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = 0.01*x3;
  yDerivative[0][1] = -0.2*x4 + 0.5*0.;

  double* x;
  x = myalloc1(4);
  x[0] = x1;
  x[1] = x2;
  x[2] = x3;
  x[3] = x4;

  double** X;
  X = myalloc2(4, 2);
  X[0][0] = 1.;
  X[1][0] = 0.1;
  X[2][0] = 0.;
  X[3][0] = -0.01;
  X[0][1] = 0.;
  X[1][1] = 1.;
  X[2][1] = 0.2;
  X[3][1] = 0.;

  double** Y;
  Y = myalloc2(1, 2);

  hos_forward(1, 1, 4, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double** H;
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

  hessian(1, 4, x, H);

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
BOOST_AUTO_TEST_CASE(CustomInvTrig_HOS)
{
  double x1 = 0.11, x2 = 0.33, x3 = 0.1*std::acos(0.), x4 = std::exp(-1.);
  adouble ax1, ax2, ax3, ax4;
  double y;
  adouble ay;

  trace_on(1, 1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;
  ax4 <<= x4;

  ay = 3.*asin(sin(ax1 + ax2))*sin(ax3)*cos(ax4);

  ay >>= y;
  trace_off();

  double yprim = 3.*std::asin(std::sin(x1 + x2))*std::sin(x3)*std::cos(x4);

  double** yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = 3.*std::sin(x3)*std::cos(x4)
                      + 0.1*3.*std::sin(x3)*std::cos(x4)
                      + 0.01*3.*(x1 + x2)*std::sin(x3)*std::sin(x4);
  yDerivative[0][1] = 3.*std::sin(x3)*std::cos(x4)
                      + 0.2*3.*(x1 + x2)*std::cos(x3)*std::cos(x4)
                      - 0.5*0.01*(-2.*3.*std::sin(x3)*std::sin(x4)
                        - 0.2*3.*std::sin(x3)*std::sin(x4)
                        + 0.01*3.*(x1 + x2)*std::sin(x3)*std::cos(x4));

  double* x;
  x = myalloc1(4);
  x[0] = x1;
  x[1] = x2;
  x[2] = x3;
  x[3] = x4;

  double** X;
  X = myalloc2(4, 2);
  X[0][0] = 1.;
  X[1][0] = 0.1;
  X[2][0] = 0.;
  X[3][0] = -0.01;
  X[0][1] = 0.;
  X[1][1] = 1.;
  X[2][1] = 0.2;
  X[3][1] = 0.;

  double** Y;
  Y = myalloc2(1, 2);

  hos_forward(1, 1, 4, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double** H;
  H = myalloc2(4, 4);

  double yx1x1Derivative = 0.;
  double yx2x1Derivative = 0.;
  double yx3x1Derivative = 3*std::cos(x3)*std::cos(x4);
  double yx4x1Derivative = -3.*sin(x3)*sin(x4);
  double yx2x2Derivative = 0.;
  double yx3x2Derivative = 3.*std::cos(x3)*std::cos(x4);
  double yx4x2Derivative = -3.*std::sin(x3)*std::sin(x4);
  double yx3x3Derivative = -3.*(x1 + x2)*std::sin(x3)*std::cos(x4);
  double yx4x3Derivative = -3.*(x1 + x2)*std::cos(x3)*std::sin(x4);
  double yx4x4Derivative = -3.*(x1 + x2)*std::sin(x3)*std::cos(x4);

  hessian(1, 4, x, H);

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
BOOST_AUTO_TEST_CASE(CustomInvTrig2_HOS)
{
  double x1 = 0.53, x2 = -0.01;
  adouble ax1, ax2;
  double y;
  adouble ay;

  trace_on(1, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay = atan(ax1)*asin(ax2);

  ay >>= y;
  trace_off();

  double yprim = std::atan(x1)*std::asin(x2);

  double** yDerivative;
  yDerivative = myalloc2(1, 2);
  yDerivative[0][0] = std::asin(x2)/(1. + x1*x1);
  yDerivative[0][1] = 1.5*std::atan(x1)/std::sqrt(1. - x2*x2)
                      - 0.5*2.*x1*std::asin(x2)/std::pow(1. + x1*x1, 2);

  double* x;
  x = myalloc1(2);
  x[0] = x1;
  x[1] = x2;

  double** X;
  X = myalloc2(2, 2);
  X[0][0] = 1.;
  X[0][1] = 0.;
  X[1][0] = 0.;
  X[1][1] = 1.5;

  double** Y;
  Y = myalloc2(1, 2);

  hos_forward(1, 1, 2, 2, 1, x, X, &y, Y);

  BOOST_TEST(y == yprim, tt::tolerance(tol));
  BOOST_TEST(Y[0][0] == yDerivative[0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == yDerivative[0][1], tt::tolerance(tol));

  double** H;
  H = myalloc2(2, 2);

  double yx1x1Derivative = -2.*x1*std::asin(x2)/std::pow(1. + x1*x1, 2);
  double yx1x2Derivative = 1./((1. + x1*x1)*std::sqrt(1. - x2*x2));
  double yx2x2Derivative = std::atan(x1)*x2/std::pow(std::sqrt(1. - x2*x2), 3);

  hessian(1, 2, x, H);

  BOOST_TEST(yx1x1Derivative == H[0][0], tt::tolerance(tol));
  BOOST_TEST(yx1x2Derivative == H[1][0], tt::tolerance(tol));
  BOOST_TEST(yx2x2Derivative == H[1][1], tt::tolerance(tol));

  myfree1(x);
  myfree2(yDerivative);
  myfree2(X);
  myfree2(Y);
  myfree2(H);
}

/* TODO: combine trig, invtrig, hyperb, invhypber, fmax, fmin, fabs! */


BOOST_AUTO_TEST_SUITE_END()





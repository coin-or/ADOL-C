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



/* TODO */


BOOST_AUTO_TEST_SUITE_END()





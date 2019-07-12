#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

#include <adolc/adolc.h>

#include "const.h"

BOOST_AUTO_TEST_SUITE( trace_sec_order_vec )


/**************************************/
/* Tests for ADOL-C trace vector mode */
/* drivers hov_forward, hov_reverse   */
/* Author: Philipp Schuette           */
/**************************************/


/* This file contains custom tests for the higher order derivative
 * evaluation drivers hov_forward, hov_reverse.
 *
 * As for the trace/traceless first order derivative tests, the custom
 * functions are described, together with their derivatives, before
 * the actual test implementation.
 *
 * Every test-function is first used to test forward mode and then to
 * test reverse mode.
 */

/* Tested function: 2.*x*x*x
 * First derivative: 2.*3.*x*x
 * Second derivative: 2.*3.*2.*x
 */
BOOST_AUTO_TEST_CASE(CustomCube_HOV_Forward)
{
  double x1 = 3.;
  adouble ax1;
  double y1;
  adouble ay1;

  trace_on(1, 1);
  ax1 <<= x1;

  ay1 = 2.*ax1*ax1*ax1;

  ay1 >>= y1;
  trace_off();

  double* yprim;
  yprim = myalloc1(1);
  yprim[0] = 2.*x1*x1*x1;

  double*** yDerivative;
  yDerivative = myalloc3(1, 3, 2);
  yDerivative[0][0][0] = 2.*3.*x1*x1;
  yDerivative[0][0][1] = 2.*3.*x1*x1 + 0.5*(2.*3.*2.*x1);
  yDerivative[0][1][0] = 2.*2.*3.*x1*x1;
  yDerivative[0][1][1] = 2.*2.*3.*x1*x1 + 0.5*(2.*3.*2.*x1)*2.*2.;
  yDerivative[0][2][0] = 3.*2.*3.*x1*x1;
  yDerivative[0][2][1] = 3.*2.*3.*x1*x1 + 0.5*(2.*3.*2.*x1)*3.*3.;

  double* x;
  x = myalloc1(1);
  x[0] = 3.;

  double*** X;
  X = myalloc3(1, 3, 2);
  X[0][0][0] = 1.;
  X[0][1][0] = 2.;
  X[0][2][0] = 3.;
  X[0][0][1] = 1.;
  X[0][1][1] = 2.;
  X[0][2][1] = 3.;

  double* y;
  y = myalloc1(1);

  double*** Y;
  Y = myalloc3(1, 3, 2);

  hov_forward(1, 1, 1, 2, 3, x, X, y, Y);

  BOOST_TEST(y[0] == yprim[0], tt::tolerance(tol));
  BOOST_TEST(Y[0][0][0] == yDerivative[0][0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][0][1] == yDerivative[0][0][1], tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] == yDerivative[0][1][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] == yDerivative[0][1][1], tt::tolerance(tol));
  BOOST_TEST(Y[0][2][0] == yDerivative[0][2][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][2][1] == yDerivative[0][2][1], tt::tolerance(tol));

  myfree3(yDerivative);
  myfree1(x);
  myfree1(y);
  myfree3(X);
  myfree3(Y);
}

BOOST_AUTO_TEST_CASE(customCube_HOV_Reverse)
{
  double x1 = 3.;
  adouble ax1;
  double y1;
  adouble ay1;

  trace_on(1, 1);
  ax1 <<= x1;

  ay1 = 2.*ax1*ax1*ax1;

  ay1 >>= y1;
  trace_off();

  double y1x1Derivative = 6.*x1*x1;

  double y1x1x1Derivative = 12.*x1;

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  x[0] = 3.;
  xd[0] = 1.;

  fos_forward(1, 1, 1, 2, x, xd, y, yd);

  double** U = myalloc2(1, 1);
  double*** Z = myalloc3(1, 1, 2);
  short int** nz = (short int**)malloc(sizeof(short int*)*1);

  nz[0] = (short int*)malloc(sizeof(short int)*1);
  nz[0][0] = 1;
  U[0][0] = 1.;

  hov_reverse(1, 1, 1, 1, 1, U, Z, nz);

  BOOST_TEST(Z[0][0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[0][0][1] == y1x1x1Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
  myfree2(U);
  myfree3(Z);
}




BOOST_AUTO_TEST_SUITE_END()





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
BOOST_AUTO_TEST_CASE(CustomCube_HOS_Forward)
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

/* TODO */


BOOST_AUTO_TEST_SUITE_END()





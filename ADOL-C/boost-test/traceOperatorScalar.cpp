#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

#include <adolc/adolc.h>

#include "const.h"

BOOST_AUTO_TEST_SUITE(trace_scalar)

/**********************************************/
/* Tests for ADOL-C trace scalar mode drivers */
/* zos_forward, fos_forward and fos_reverse   */
/* Author: Philipp Schuette                   */
/**********************************************/

/* Naming convention for test cases:  Operatorname_Operator_ZOS_Forward for
 * primal value (= zero order derivative), Operatorname_Operator_FOS_Forward
 * for forward derivative evaluation and Operatorname_Operator_FOS_Reverse for
 * reverse mode derivative evaluation.
 *
 * For documentation of a concrete test implementation, check the traceless scalar
 * mode test implementation.  The testing order is consistent with that file
 * as well.
 */

BOOST_AUTO_TEST_CASE(ExpOperator_ZOS_Forward) {
  double a = 2., aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = exp(ad);

  ad >>= aout;
  trace_off();

  a = std::exp(a);

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 2.;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(ExpOperator_FOS_Forward) {
  double a = 2., aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = exp(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = std::exp(a);
  a = std::exp(a);

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 2.;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(ExpOperator_FOS_Reverse) {
  double a = 2., aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = exp(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = std::exp(a);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(MultOperator_ZOS_Forward) {
  double a = 2., b = 3.5, out;
  adouble ad, bd;

  trace_on(1);
  ad <<= a;
  bd <<= b;

  ad = ad * bd;

  ad >>= out;
  trace_off();

  a = a * b;

  BOOST_TEST(out == a, tt::tolerance(tol));

  double *x = myalloc1(2);
  double *y = myalloc1(1);

  // x[0] = 2.;
  *x = 2.;
  *(x + 1) = 3.5;

  zos_forward(1, 1, 2, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(MultOperator_FOS_Forward) {
  double a = 2., b = 3.5, out;
  adouble ad, bd;

  trace_on(1);
  ad <<= a;
  bd <<= b;

  ad = ad * bd;

  ad >>= out;
  trace_off();

  double aDerivative = b;
  double bDerivative = a;
  a = a * b;

  double *x = myalloc1(2);
  double *xd = myalloc1(2);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  /* Test partial derivative wrt a. */
  *x = 2.;
  *(x + 1) = 3.5;
  *xd = 1.;
  *(xd + 1) = 0.;

  fos_forward(1, 1, 2, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  /* Test partial derivative wrt b. */
  *xd = 0.;
  *(xd + 1) = 1.;

  fos_forward(1, 1, 2, 0, x, xd, y, yd);

  BOOST_TEST(*yd == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(MultOperator_FOS_Reverse) {
  double a = 2., b = 3.5, out;
  adouble ad, bd;

  trace_on(1, 1);
  ad <<= a;
  bd <<= b;

  ad = ad * bd;

  ad >>= out;
  trace_off();

  double aDerivative = b;
  double bDerivative = a;

  double *u = myalloc1(1);
  double *z = myalloc1(2);

  *u = 1.;

  fos_reverse(1, 1, 2, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));
  BOOST_TEST(*(z + 1) == bDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(AddOperator_ZOS_Forward) {
  double a = 2.5, b = 3., out;
  adouble ad, bd;

  trace_on(1);
  ad <<= a;
  bd <<= b;

  ad = ad + bd;

  ad >>= out;
  trace_off();

  a = a + b;

  BOOST_TEST(out == a, tt::tolerance(tol));

  double *x = myalloc1(2);
  double *y = myalloc1(1);

  *x = 2.5;
  *(x + 1) = 3.;

  zos_forward(1, 1, 2, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(AddOperator_FOS_Forward) {
  double a = 2.5, b = 3., out;
  adouble ad, bd;

  trace_on(1);
  ad <<= a;
  bd <<= b;

  ad = ad + bd;

  ad >>= out;
  trace_off();

  double aDerivative = 1.;
  double bDerivative = 1.;
  a = a + b;

  double *x = myalloc1(2);
  double *xd = myalloc1(2);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  /* Test partial derivative wrt a. */
  *x = 2.5;
  *(x + 1) = 3.;
  *xd = 1.;
  *(xd + 1) = 0.;

  fos_forward(1, 1, 2, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  /* Test partial derivative wrt b. */
  *xd = 0.;
  *(xd + 1) = 1.;

  fos_forward(1, 1, 2, 0, x, xd, y, yd);

  BOOST_TEST(*yd == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(AddOperator_FOS_Reverse) {
  double a = 2.5, b = 3., out;
  adouble ad, bd;

  trace_on(1, 1);
  ad <<= a;
  bd <<= b;

  ad = ad + bd;

  ad >>= out;
  trace_off();

  double aDerivative = 1.;
  double bDerivative = 1.;

  double *u = myalloc1(1);
  double *z = myalloc1(2);

  *u = 1.;

  fos_reverse(1, 1, 2, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));
  BOOST_TEST(*(z + 1) == bDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(SubOperator_ZOS_Forward) {
  double a = 1.5, b = 3.2, out;
  adouble ad, bd;

  trace_on(1);
  ad <<= a;
  bd <<= b;

  ad = ad - bd;

  ad >>= out;
  trace_off();

  a = a - b;

  BOOST_TEST(out == a, tt::tolerance(tol));

  double *x = myalloc1(2);
  double *y = myalloc1(1);

  *x = 1.5;
  *(x + 1) = 3.2;

  zos_forward(1, 1, 2, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(SubOperator_FOS_Forward) {
  double a = 1.5, b = 3.2, out;
  adouble ad, bd;

  trace_on(1);
  ad <<= a;
  bd <<= b;

  ad = ad - bd;

  ad >>= out;
  trace_off();

  double aDerivative = 1.;
  double bDerivative = -1.;
  a = a - b;

  double *x = myalloc1(2);
  double *xd = myalloc1(2);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  /* Test partial derivative wrt a. */
  *x = 1.5;
  *(x + 1) = 3.2;
  *xd = 1.;
  *(xd + 1) = 0.;

  fos_forward(1, 1, 2, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  /* Test partial derivative wrt b. */
  *xd = 0.;
  *(xd + 1) = 1.;

  fos_forward(1, 1, 2, 0, x, xd, y, yd);

  BOOST_TEST(*yd == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(SubOperator_FOS_Reverse) {
  double a = 1.5, b = 3.2, out;
  adouble ad, bd;

  trace_on(1, 1);
  ad <<= a;
  bd <<= b;

  ad = ad - bd;

  ad >>= out;
  trace_off();

  double aDerivative = 1.;
  double bDerivative = -1.;

  double *u = myalloc1(1);
  double *z = myalloc1(2);

  *u = 1.;

  fos_reverse(1, 1, 2, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));
  BOOST_TEST(*(z + 1) == bDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(DivOperator_ZOS_Forward) {
  double a = 0.5, b = 4.5, out;
  adouble ad, bd;

  trace_on(1);
  ad <<= a;
  bd <<= b;

  ad = ad / bd;

  ad >>= out;
  trace_off();

  a = a / b;

  BOOST_TEST(out == a, tt::tolerance(tol));

  double *x = myalloc1(2);
  double *y = myalloc1(1);

  *x = 0.5;
  *(x + 1) = 4.5;

  zos_forward(1, 1, 2, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(DivOperator_FOS_Forward) {
  double a = 0.5, b = 4.5, out;
  adouble ad, bd;

  trace_on(1);
  ad <<= a;
  bd <<= b;

  ad = ad / bd;

  ad >>= out;
  trace_off();

  double aDerivative = 1. / b;
  double bDerivative = -a / (b * b);
  a = a / b;

  double *x = myalloc1(2);
  double *xd = myalloc1(2);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  /* Test partial derivative wrt a. */
  *x = 0.5;
  *(x + 1) = 4.5;
  *xd = 1.;
  *(xd + 1) = 0.;

  fos_forward(1, 1, 2, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  /* Test partial derivative wrt b. */
  *xd = 0.;
  *(xd + 1) = 1.;

  fos_forward(1, 1, 2, 0, x, xd, y, yd);

  BOOST_TEST(*yd == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(DivOperator_FOS_Reverse) {
  double a = 0.5, b = 4.5, out;
  adouble ad, bd;

  trace_on(1, 1);
  ad <<= a;
  bd <<= b;

  ad = ad / bd;

  ad >>= out;
  trace_off();

  double aDerivative = 1. / b;
  double bDerivative = -a / (b * b);

  double *u = myalloc1(1);
  double *z = myalloc1(2);

  *u = 1.;

  fos_reverse(1, 1, 2, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));
  BOOST_TEST(*(z + 1) == bDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(TanOperator_ZOS_Forward) {
  double a = 0.7, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = tan(ad);

  ad >>= aout;
  trace_off();

  a = std::tan(a);

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 0.7;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(TanOperator_FOS_Forward) {
  double a = 0.7, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = tan(ad);

  ad >>= aout;
  trace_off();

  a = std::tan(a);
  double aDerivative = 1. + a * a;

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 0.7;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(TanOperator_FOS_Reverse) {
  double a = 0.7, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = tan(ad);

  ad >>= aout;
  trace_off();

  a = std::tan(a);
  double aDerivative = 1. + a * a;

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(SinOperator_ZOS_Forward) {
  double a = 1.2, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = sin(ad);

  ad >>= aout;
  trace_off();

  a = std::sin(a);

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 1.2;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(SinOperator_FOS_Forward) {
  double a = 1.2, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = sin(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = std::cos(a);
  a = std::sin(a);

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 1.2;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(SinOperator_FOS_Reverse) {
  double a = 1.2, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = sin(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = std::cos(a);
  a = std::sin(a);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(CosOperator_ZOS_Forward) {
  double a = 1.2, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = cos(ad);

  ad >>= aout;
  trace_off();

  a = std::cos(a);

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 1.2;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(CosOperator_FOS_Forward) {
  double a = 1.2, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = cos(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = -std::sin(a);
  a = std::cos(a);

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 1.2;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(CosOperator_FOS_Reverse) {
  double a = 1.2, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = cos(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = -std::sin(a);
  a = std::cos(a);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(SqrtOperator_ZOS_Forward) {
  double a = 2.2, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = sqrt(ad);

  ad >>= aout;
  trace_off();

  a = std::sqrt(a);

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 2.2;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(SqrtOperator_FOS_Forward) {
  double a = 2.2, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = sqrt(ad);

  ad >>= aout;
  trace_off();

  a = std::sqrt(a);
  double aDerivative = 1. / (2 * a);

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 2.2;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(SqrtOperator_FOS_Reverse) {
  double a = 2.2, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = sqrt(ad);

  ad >>= aout;
  trace_off();

  a = std::sqrt(a);
  double aDerivative = 1. / (2 * a);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(LogOperator_ZOS_Forward) {
  double a = 4.9, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = log(ad);

  ad >>= aout;
  trace_off();

  a = std::log(a);

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 4.9;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(LogOperator_FOS_Forward) {
  double a = 4.9, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = log(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / a;
  a = std::log(a);

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 4.9;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(LogOperator_FOS_Reverse) {
  double a = 4.9, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = log(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / a;
  a = std::log(a);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(SinhOperator_ZOS_Forward) {
  double a = 4., aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = sinh(ad);

  ad >>= aout;
  trace_off();

  a = std::sinh(a);

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 4.;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(SinhOperator_FOS_Forward) {
  double a = 4., aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = sinh(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = std::cosh(a);
  a = std::sinh(a);

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 4.;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(SinhOperator_FOS_Reverse) {
  double a = 4., aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = sinh(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = std::cosh(a);
  a = std::sinh(a);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(CoshOperator_ZOS_Forward) {
  double a = 4., aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = cosh(ad);

  ad >>= aout;
  trace_off();

  a = std::cosh(a);

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 4.;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(CoshOperator_FOS_Forward) {
  double a = 4., aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = cosh(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = std::sinh(a);
  a = std::cosh(a);

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 4.;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(CoshOperator_FOS_Reverse) {
  double a = 4., aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = cosh(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = std::sinh(a);
  a = std::cosh(a);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(TanhOperator_ZOS_Forward) {
  double a = 4., aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = tanh(ad);

  ad >>= aout;
  trace_off();

  a = std::tanh(a);

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 4.;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(TanhOperator_FOS_Forward) {
  double a = 4., aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = tanh(ad);

  ad >>= aout;
  trace_off();

  a = std::tanh(a);
  double aDerivative = 1. - a * a;

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 4.;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(TanhOperator_FOS_Reverse) {
  double a = 4., aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = tanh(ad);

  ad >>= aout;
  trace_off();

  a = std::tanh(a);
  double aDerivative = 1. - a * a;

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(AsinOperator_ZOS_Forward) {
  double a = 0.9, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = asin(ad);

  ad >>= aout;
  trace_off();

  a = std::asin(a);

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 0.9;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(AsinOperator_FOS_Forward) {
  double a = 0.9, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = asin(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (std::sqrt(1. - a * a));
  a = std::asin(a);

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 0.9;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(AsinOperator_FOS_Reverse) {
  double a = 0.9, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = asin(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (std::sqrt(1. - a * a));
  a = std::asin(a);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(AcosOperator_ZOS_Forward) {
  double a = 0.8, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = acos(ad);

  ad >>= aout;
  trace_off();

  a = std::acos(a);

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 0.8;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(AcosOperator_FOS_Forward) {
  double a = 0.8, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = acos(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = -1. / (std::sqrt(1. - a * a));
  a = std::acos(a);

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 0.8;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(AcosOperator_FOS_Reverse) {
  double a = 0.8, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = acos(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = -1. / (std::sqrt(1. - a * a));
  a = std::acos(a);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(AtanOperator_ZOS_Forward) {
  double a = 9.8, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = atan(ad);

  ad >>= aout;
  trace_off();

  a = std::atan(a);

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 9.8;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(AtanOperator_FOS_Forward) {
  double a = 9.8, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = atan(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (1. + a * a);
  a = std::atan(a);

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 9.8;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(AtanOperator_FOS_Reverse) {
  double a = 9.8, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = atan(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (1. + a * a);
  a = std::atan(a);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(Log10Operator_ZOS_Forward) {
  double a = 12.3, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = log10(ad);

  ad >>= aout;
  trace_off();

  a = std::log10(a);

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 12.3;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(Log10Operator_FOS_Forward) {
  double a = 12.3, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = log10(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (a * std::log(10));
  a = std::log10(a);

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 12.3;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(Log10Operator_FOS_Reverse) {
  double a = 12.3, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = log10(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (a * std::log(10));
  a = std::log10(a);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(AsinhOperator_ZOS_Forward) {
  double a = 0.6, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = asinh(ad);

  ad >>= aout;
  trace_off();

  a = std::asinh(a);

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 0.6;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(AsinhOperator_FOS_Forward) {
  double a = 0.6, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = asinh(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (std::sqrt(a * a + 1.));
  a = std::asinh(a);

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 0.6;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(AsinhOperator_FOS_Reverse) {
  double a = 0.6, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = asinh(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (std::sqrt(a * a + 1.));
  a = std::asinh(a);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(AcoshOperator_ZOS_Forward) {
  double a = 1.7, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = acosh(ad);

  ad >>= aout;
  trace_off();

  a = std::acosh(a);

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 1.7;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(AcoshOperator_FOS_Forward) {
  double a = 1.7, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = acosh(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (std::sqrt(a * a - 1));
  a = std::acosh(a);

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 1.7;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(AcoshOperator_FOS_Reverse) {
  double a = 1.7, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = acosh(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (std::sqrt(a * a - 1.));
  a = std::acosh(a);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(AtanhOperator_ZOS_Forward) {
  double a = 0.6, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = atanh(ad);

  ad >>= aout;
  trace_off();

  a = std::atanh(a);

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 0.6;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(AtanhOperator_FOS_Forward) {
  double a = 0.6, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = atanh(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (1. - a * a);
  a = std::atanh(a);

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 0.6;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(AtanhOperator_FOS_Reverse) {
  double a = 0.6, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = atanh(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (1. - a * a);
  a = std::atanh(a);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(InclOperator_ZOS_Forward) {
  double a = 5., aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = ++ad;

  ad >>= aout;
  trace_off();

  a = ++a;

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 5.;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(InclOperator_FOS_Forward) {
  double a = 5., aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = ++ad;

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;
  a = ++a;

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 5.;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(InclOperator_FOS_Reverse) {
  double a = 5., aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = ++ad;

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;
  a = ++a;

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(DeclOperator_ZOS_Forward) {
  double a = 5., aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = --ad;

  ad >>= aout;
  trace_off();

  a = --a;

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 5.;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(DeclOperator_FOS_Forward) {
  double a = 5., aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = --ad;

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;
  a = --a;

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 5.;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(DeclOperator_FOS_Reverse) {
  double a = 5., aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = --ad;

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;
  a = --a;

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(SignPlusOperator_ZOS_Forward) {
  double a = 1.5, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = +ad;

  ad >>= aout;
  trace_off();

  a = +a;

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 1.5;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(SignPlusOperator_FOS_Forward) {
  double a = 1.5, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = +ad;

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;
  a = +a;

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 1.5;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(SignPlusOperator_FOS_Reverse) {
  double a = 1.5, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = +ad;

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;
  a = +a;

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(SignMinusOperator_ZOS_Forward) {
  double a = 1.5, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = -ad;

  ad >>= aout;
  trace_off();

  a = -a;

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 1.5;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(SignMinusOperator_FOS_Forward) {
  double a = 1.5, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = -ad;

  ad >>= aout;
  trace_off();

  double aDerivative = -1.;
  a = -a;

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 1.5;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(SignMinusOperator_FOS_Reverse) {
  double a = 1.5, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = -ad;

  ad >>= aout;
  trace_off();

  double aDerivative = -1.;
  a = -a;

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(Atan2Operator_ZOS_Forward) {
  double a = 12.3, b = 2.1, out;
  adouble ad, bd;

  trace_on(1);
  ad <<= a;
  bd <<= b;

  ad = atan2(ad, bd);

  ad >>= out;
  trace_off();

  a = std::atan2(a, b);

  BOOST_TEST(out == a, tt::tolerance(tol));

  double *x = myalloc1(2);
  double *y = myalloc1(1);

  *x = 12.3;
  *(x + 1) = 2.1;

  zos_forward(1, 1, 2, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(Atan2Operator_FOS_Forward) {
  double a = 12.3, b = 2.1, out;
  adouble ad, bd;

  trace_on(1);
  ad <<= a;
  bd <<= b;

  ad = atan2(ad, bd);

  ad >>= out;
  trace_off();

  double aDerivative = b / (a * a + b * b);
  double bDerivative = -a / (a * a + b * b);
  a = std::atan2(a, b);

  double *x = myalloc1(2);
  double *xd = myalloc1(2);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  /* Test partial derivative wrt a. */
  *x = 12.3;
  *(x + 1) = 2.1;
  *xd = 1.;
  *(xd + 1) = 0.;

  fos_forward(1, 1, 2, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  /* Test partial derivative wrt b. */
  *xd = 0.;
  *(xd + 1) = 1.;

  fos_forward(1, 1, 2, 0, x, xd, y, yd);

  BOOST_TEST(*yd == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(Atan2Operator_FOS_Reverse) {
  double a = 12.3, b = 2.1, out;
  adouble ad, bd;

  trace_on(1, 1);
  ad <<= a;
  bd <<= b;

  ad = atan2(ad, bd);

  ad >>= out;
  trace_off();

  double aDerivative = b / (a * a + b * b);
  double bDerivative = -a / (a * a + b * b);

  double *u = myalloc1(1);
  double *z = myalloc1(2);

  *u = 1.;

  fos_reverse(1, 1, 2, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));
  BOOST_TEST(*(z + 1) == bDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(PowOperator_ZOS_Forward_1) {
  double a = 2.3, e = 3.5, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = pow(ad, e);

  ad >>= aout;
  trace_off();

  a = std::pow(a, e);

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 2.3;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(PowOperator_FOS_Forward_1) {
  double a = 2.3, e = 3.5, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = pow(ad, e);

  ad >>= aout;
  trace_off();

  double aDerivative = e * std::pow(a, e - 1.);
  a = std::pow(a, e);

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 2.3;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}
/*
BOOST_AUTO_TEST_CASE(PowOperator_FOS_Reverse_1)
{
  double a = 2.3, e = 3.5, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = pow(ad, e);

  ad >>= aout;
  trace_off();

  double aDerivative = e * std::pow(a, e - 1.);
  a = std::pow(a, e);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}
*/
BOOST_AUTO_TEST_CASE(PowOperator_ZOS_Forward_2) {
  double a = 2.3, b = 3.5, out;
  adouble ad, bd;

  trace_on(1);
  ad <<= a;
  bd <<= b;

  ad = pow(ad, bd);

  ad >>= out;
  trace_off();

  a = std::pow(a, b);

  BOOST_TEST(out == a, tt::tolerance(tol));

  double *x = myalloc1(2);
  double *y = myalloc1(1);

  *x = 2.3;
  *(x + 1) = 3.5;

  zos_forward(1, 1, 2, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(PowOperator_FOS_Forward_2) {
  double a = 2.3, b = 3.5, out;
  adouble ad, bd;

  trace_on(1);
  ad <<= a;
  bd <<= b;

  ad = pow(ad, bd);

  ad >>= out;
  trace_off();

  double aDerivative = b * std::pow(a, b - 1.);
  double bDerivative = std::log(a) * std::pow(a, b);
  a = std::pow(a, b);

  double *x = myalloc1(2);
  double *xd = myalloc1(2);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  /* Test partial derivative wrt a. */
  *x = 2.3;
  *(x + 1) = 3.5;
  *xd = 1.;
  *(xd + 1) = 0.;

  fos_forward(1, 1, 2, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  /* Test partial derivative wrt b. */
  *xd = 0.;
  *(xd + 1) = 1.;

  fos_forward(1, 1, 2, 0, x, xd, y, yd);

  BOOST_TEST(*yd == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(PowOperator_FOS_Reverse_2) {
  double a = 2.3, b = 3.5, out;
  adouble ad, bd;

  trace_on(1, 1);
  ad <<= a;
  bd <<= b;

  ad = pow(ad, bd);

  ad >>= out;
  trace_off();

  double aDerivative = b * std::pow(a, b - 1.);
  double bDerivative = std::log(a) * std::pow(a, b);

  double *u = myalloc1(1);
  double *z = myalloc1(2);

  *u = 1.;

  fos_reverse(1, 1, 2, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));
  BOOST_TEST(*(z + 1) == bDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(PowOperator_ZOS_Forward_3) {
  double a = 2.3, e = 3.5, eout;
  adouble ed;

  trace_on(1);
  ed <<= e;

  ed = pow(a, ed);

  ed >>= eout;
  trace_off();

  e = std::pow(a, e);

  BOOST_TEST(eout == e, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 3.5;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == e, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(PowOperator_FOS_Forward_3) {
  double a = 2.3, e = 3.5, eout;
  adouble ed;

  trace_on(1);
  ed <<= e;

  ed = pow(a, ed);

  ed >>= eout;
  trace_off();

  double eDerivative = std::log(a) * std::pow(a, e);
  e = std::pow(a, e);

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 3.5;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == e, tt::tolerance(tol));
  BOOST_TEST(*yd == eDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(PowOperator_FOS_Reverse_3) {
  double a = 2.3, e = 3.5, eout;
  adouble ed;

  trace_on(1, 1);
  ed <<= e;

  ed = pow(a, ed);

  ed >>= eout;
  trace_off();

  double eDerivative = std::log(a) * std::pow(a, e);
  e = std::pow(a, e);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == eDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

/* Frexp does not need to be tested, because it is non-differentiable. */
/*
BOOST_AUTO_TEST_CASE(LdexpOperator_ZOS_Forward_1)
{
  double a = 4., b = 3., out;
  adouble ad, bd;

  trace_on(1);
  ad <<= a;
  bd <<= b;

  ad = ldexp(ad, bd);

  ad >>= out;
  trace_off();

  a = a * std::pow(2., b);

  BOOST_TEST(out == a, tt::tolerance(tol));

  double *x = myalloc1(2);
  double *y = myalloc1(1);

  *x = 4.;
  *(x + 1) = 3.;

  zos_forward(1, 1, 2, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(LdexpOperator_FOS_Forward_1)
{
  double a = 4., b = 3., out;
  adouble ad, bd;

  trace_on(1);
  ad <<= a;
  bd <<= b;

  ad = ldexp(ad, bd);

  ad >>= out;
  trace_off();

  double aDerivative = std::pow(2., b);
  double bDerivative = a * std::log(2.) * std::pow(2., b);
  a = a * std::pow(2., b);

  double *x = myalloc1(2);
  double *xd = myalloc1(2);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 4.;
  *(x + 1) = 3.;
  *xd = 1.;
  *(xd + 1) = 0.;

  fos_forward(1, 1, 2, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  *xd = 0.;
  *(xd + 1) = 1.;

  fos_forward(1, 1, 2, 0, x, xd, y, yd);

  BOOST_TEST(*yd == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(LdexpOperator_FOS_Reverse_1)
{
  double a = 4., b = 3., out;
  adouble ad, bd;

  trace_on(1, 1);
  ad <<= a;
  bd <<= b;

  ad = ldexp(ad, bd);

  ad >>= out;
  trace_off();

  double aDerivative = std::pow(2., b);
  double bDerivative = a * std::log(2.) * std::pow(2., b);

  double *u = myalloc1(1);
  double *z = myalloc1(2);

  *u = 1.;

  fos_reverse(1, 1, 2, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));
  BOOST_TEST(*(z + 1) == bDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}
*/
BOOST_AUTO_TEST_CASE(LdexpOperator_ZOS_Forward_2) {
  double a = 4., e = 3., aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = ldexp(ad, e);

  ad >>= aout;
  trace_off();

  a = a * std::pow(2., e);

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 4.;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(LdexpOperator_FOS_Forward_2) {
  double a = 4., e = 3., aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = ldexp(ad, e);

  ad >>= aout;
  trace_off();

  double aDerivative = std::pow(2., e);
  a = a * std::pow(2., e);

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 4.;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(LdexpOperator_FOS_Reverse_2) {
  double a = 4., e = 3., aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = ldexp(ad, e);

  ad >>= aout;
  trace_off();

  double aDerivative = std::pow(2., e);
  a = a * std::pow(2., e);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}
/*
BOOST_AUTO_TEST_CASE(LdexpOperator_ZOS_Forward_3)
{
  double a = 4., e = 3., eout;
  adouble ed;

  trace_on(1);
  ed <<= e;

  ed = ldexp(a, ed);

  ed >>= eout;
  trace_off();

  e = a * std::pow(2., e);

  BOOST_TEST(eout == e, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 3.;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == e, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(LdexpOperator_FOS_Forward_3)
{
  double a = 4., e = 3., eout;
  adouble ed;

  trace_on(1);
  ed <<= e;

  ed = ldexp(a, ed);

  ed >>= eout;
  trace_off();

  double eDerivative = a * std::log(2.) * std::pow(2., e);
  e = std::pow(a, e);

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 3.;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == e, tt::tolerance(tol));
  BOOST_TEST(*yd == eDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(LdexpOperator_FOS_Reverse_3)
{
  double a = 4., e = 3., eout;
  adouble ed;

  trace_on(1, 1);
  ed <<= e;

  ed = ldexp(a, ed);

  ed >>= eout;
  trace_off();

  double eDerivative = a * std::log(2.) * std::pow(2., e);
  e = a * std::pow(2., e);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == eDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}
*/

BOOST_AUTO_TEST_CASE(FabsOperator_ZOS_Forward) {
  double a = 1.4, b = -5., c = 0., aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = fabs(ad);

  ad >>= aout;
  trace_off();

  a = std::fabs(a);
  b = std::fabs(b);
  c = std::fabs(c);

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x1 = myalloc1(1);
  double *y1 = myalloc1(1);
  double *x2 = myalloc1(1);
  double *y2 = myalloc1(1);
  double *x3 = myalloc1(1);
  double *y3 = myalloc1(1);

  *x1 = 1.4;
  *x2 = -5.;
  *x3 = 0.;

  zos_forward(1, 1, 1, 0, x1, y1);
  zos_forward(1, 1, 1, 0, x2, y2);
  zos_forward(1, 1, 1, 0, x3, y3);

  BOOST_TEST(*y1 == a, tt::tolerance(tol));
  BOOST_TEST(*y2 == b, tt::tolerance(tol));
  BOOST_TEST(*y3 == c, tt::tolerance(tol));

  myfree1(x1);
  myfree1(y1);
  myfree1(x2);
  myfree1(y2);
  myfree1(x3);
  myfree1(y3);
}

BOOST_AUTO_TEST_CASE(FabsOperator_FOS_Forward) {
  double a = 1.4, b = -5., c = 0., aout, bout, cout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = fabs(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;
  double bDerivative = -1.;
  a = std::fabs(a);
  b = std::fabs(b);

  double *x1 = myalloc1(1);
  double *xd1 = myalloc1(1);
  double *y1 = myalloc1(1);
  double *yd1 = myalloc1(1);
  double *x2 = myalloc1(1);
  double *xd2 = myalloc1(1);
  double *y2 = myalloc1(1);
  double *yd2 = myalloc1(1);

  *x1 = 1.4;
  *xd1 = 1.;
  *x2 = -5.;
  *xd2 = 1.;

  fos_forward(1, 1, 1, 0, x1, xd1, y1, yd1);
  fos_forward(1, 1, 1, 0, x2, xd2, y2, yd2);

  BOOST_TEST(*y1 == a, tt::tolerance(tol));
  BOOST_TEST(*yd1 == aDerivative, tt::tolerance(tol));
  BOOST_TEST(*y2 == b, tt::tolerance(tol));
  BOOST_TEST(*yd2 == bDerivative, tt::tolerance(tol));

  double *x3 = myalloc1(1);
  double *xd3_1 = myalloc1(1);
  double *xd3_2 = myalloc1(1);
  double *y3 = myalloc1(1);
  double *yd3 = myalloc1(1);

  *x3 = 0.0;
  *xd3_1 = 2.5;
  *xd3_2 = -3.5;

  fos_forward(1, 1, 1, 0, x3, xd3_1, y3, yd3);

  BOOST_TEST(*y3 == 0.0, tt::tolerance(tol));
  BOOST_TEST(*yd3 == 2.5, tt::tolerance(tol));

  fos_forward(1, 1, 1, 0, x3, xd3_2, y3, yd3);

  BOOST_TEST(*y3 == 0.0, tt::tolerance(tol));
  BOOST_TEST(*yd3 == 3.5, tt::tolerance(tol));

  myfree1(x1);
  myfree1(xd1);
  myfree1(y1);
  myfree1(yd1);
  myfree1(x2);
  myfree1(xd2);
  myfree1(y2);
  myfree1(yd2);
  myfree1(x3);
  myfree1(xd3_1);
  myfree1(xd3_2);
  myfree1(y3);
  myfree1(yd3);
}

BOOST_AUTO_TEST_CASE(FabsOperator_FOS_Reverse_Pos) {
  double a = 1.4, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = fabs(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(FabsOperator_FOS_Reverse_Neg) {
  double a = -5., aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = fabs(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = -1.;

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(FabsOperator_FOS_Reverse_Zero) {
  double a = 0., aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = fabs(ad);

  ad >>= aout;
  trace_off();

  /* In reverse mode, the derivative at zero is calculated to be zero. */
  double aDerivative = 0.;

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(CeilOperator_ZOS_Forward) {
  double a = 3.573, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = ceil(ad);

  ad >>= aout;
  trace_off();

  a = std::ceil(a);

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 3.573;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(CeilOperator_FOS_Forward) {
  double a = 3.573, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = ceil(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 0.;
  a = std::ceil(a);

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 3.573;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(CeilOperator_FOS_Reverse) {
  double a = 3.573, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = ceil(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 0.;

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(FloorOperator_ZOS_Forward) {
  double a = 4.483, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = floor(ad);

  ad >>= aout;
  trace_off();

  a = std::floor(a);

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 4.483;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(FloorOperator_FOS_Forward) {
  double a = 4.483, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = floor(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 0.;
  a = std::floor(a);

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 4.483;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(FloorOperator_FOS_Reverse) {
  double a = 4.483, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = ceil(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 0.;

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(FmaxOperator_ZOS_Forward_1) {
  double a = 4., b = 3.2, out;
  adouble ad, bd;

  trace_on(1);
  ad <<= a;
  bd <<= b;

  ad = fmax(ad, bd);

  ad >>= out;
  trace_off();

  a = std::fmax(a, b);

  BOOST_TEST(out == a, tt::tolerance(tol));

  double *x = myalloc1(2);
  double *y = myalloc1(1);

  *x = 4.;
  *(x + 1) = 3.2;

  zos_forward(1, 1, 2, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(FmaxOperator_FOS_Forward_1) {
  double a = 4., b = 3.2, out;
  adouble ad, bd;

  trace_on(1);
  ad <<= a;
  bd <<= b;

  ad = fmax(ad, bd);

  ad >>= out;
  trace_off();

  double aDerivative = 1.;
  double bDerivative = 0.;
  a = std::fmax(a, b);

  double *x = myalloc1(2);
  double *xd = myalloc1(2);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  /* Test partial derivative wrt a. */
  *x = 4.;
  *(x + 1) = 3.2;
  *xd = 1.;
  *(xd + 1) = 0.;

  fos_forward(1, 1, 2, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  /* Test partial derivative wrt b. */
  *xd = 0.;
  *(xd + 1) = 1.;

  fos_forward(1, 1, 2, 0, x, xd, y, yd);

  BOOST_TEST(*yd == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);

  /* Test derivative value for a = b. */
  double *x1 = myalloc1(2);
  double *xd1 = myalloc1(2);
  double *y1 = myalloc1(1);
  double *yd1 = myalloc1(1);

  *x1 = 2.5;
  *(x1 + 1) = 2.5;
  *xd1 = 1.3;
  *(xd1 + 1) = 3.7;

  fos_forward(1, 1, 2, 0, x1, xd1, y1, yd1);

  BOOST_TEST(*y1 == 2.5, tt::tolerance(tol));
  BOOST_TEST(*yd1 == 3.7, tt::tolerance(tol));

  myfree1(x1);
  myfree1(xd1);
  myfree1(y1);
  myfree1(yd1);
}

BOOST_AUTO_TEST_CASE(FmaxOperator_FOS_Reverse_1) {
  double a = 4., b = 3.2, out;
  adouble ad, bd;

  trace_on(1, 1);
  ad <<= a;
  bd <<= b;

  ad = fmax(ad, bd);

  ad >>= out;
  trace_off();

  double aDerivative = 1.;
  double bDerivative = 0.;

  double *u = myalloc1(1);
  double *z = myalloc1(2);

  *u = 1.;

  fos_reverse(1, 1, 2, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));
  BOOST_TEST(*(z + 1) == bDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);

  /* Test reverse derivative value for a = b. */
  double a1 = 2.5, b1 = 2.5, out1;
  adouble ad1, bd1;

  trace_on(1, 1);
  ad1 <<= a1;
  bd1 <<= b1;

  ad1 = fmax(ad1, bd1);

  ad1 >>= out1;
  trace_off();

  double *u1 = myalloc1(1);
  double *z1 = myalloc1(2);

  *u1 = 1.;

  fos_reverse(1, 1, 2, u1, z1);

  BOOST_TEST(*z1 == 0.5, tt::tolerance(tol));
  BOOST_TEST(*(z1 + 1) == 0.5, tt::tolerance(tol));

  myfree1(u1);
  myfree1(z1);
}

BOOST_AUTO_TEST_CASE(FmaxOperator_ZOS_Forward_2) {
  double a = 4., b = 3.2, bout;
  adouble bd;

  trace_on(1);
  bd <<= b;

  bd = fmax(a, bd);

  bd >>= bout;
  trace_off();

  b = std::fmax(a, b);

  BOOST_TEST(bout == b, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 3.2;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == b, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(FmaxOperator_FOS_Forward_2) {
  double a = 4., b = 3.2, bout;
  adouble bd;

  trace_on(1);
  bd <<= b;

  bd = fmax(a, bd);

  bd >>= bout;
  trace_off();

  /* Derivative value is 0.0, as the active variable is smaller than the passive
   * one. */
  double bDerivative = 0.;
  b = std::fmax(a, b);

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 3.2;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == b, tt::tolerance(tol));
  BOOST_TEST(*yd == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);

  /* Test derivative calculation for a = b. */
  double a1 = 2.5, b1 = 2.5, bout1;
  adouble bd1;

  trace_on(1);
  bd1 <<= b1;

  bd1 = fmax(a1, bd1);

  bd1 >>= bout1;
  trace_off();

  double b1Derivative = 0.;
  b1 = std::fmax(a1, b1);

  double *x1 = myalloc1(1);
  double *xd1 = myalloc1(1);
  double *y1 = myalloc1(1);
  double *yd1 = myalloc1(1);

  *x1 = 2.5;
  *xd1 = -1.3;

  fos_forward(1, 1, 1, 0, x1, xd1, y1, yd1);

  BOOST_TEST(*y1 == b1, tt::tolerance(tol));
  BOOST_TEST(*yd1 == b1Derivative, tt::tolerance(tol));

  *xd1 = 3.7;
  b1Derivative = 3.7;

  fos_forward(1, 1, 1, 0, x1, xd1, y1, yd1);

  BOOST_TEST(*y1 == b1, tt::tolerance(tol));
  BOOST_TEST(*yd1 == b1Derivative, tt::tolerance(tol));

  myfree1(x1);
  myfree1(xd1);
  myfree1(y1);
  myfree1(yd1);
}

BOOST_AUTO_TEST_CASE(FmaxOperator_FOS_Reverse_2) {
  double a = 4., b = 3.2, bout;
  adouble bd;

  trace_on(1, 1);
  bd <<= b;

  bd = fmax(a, bd);

  bd >>= bout;
  trace_off();

  double bDerivative = 0.;
  b = std::fmax(a, b);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == bDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);

  /* Test derivative calculation for a = b. */
  double a1 = 2.5, b1 = 2.5, bout1;
  adouble bd1;

  trace_on(1, 1);
  bd1 <<= b1;

  bd1 = fmax(a1, bd1);

  bd1 >>= bout1;
  trace_off();

  double b1Derivative = 0.5;
  b1 = std::fmax(a1, b1);

  double *u1 = myalloc1(1);
  double *z1 = myalloc1(1);

  *u1 = 1.;

  fos_reverse(1, 1, 1, u1, z1);

  BOOST_TEST(*z1 == b1Derivative, tt::tolerance(tol));

  myfree1(u1);
  myfree1(z1);
}

BOOST_AUTO_TEST_CASE(FmaxOperator_ZOS_Forward_3) {
  double a = 4., b = 3.2, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = fmax(ad, b);

  ad >>= aout;
  trace_off();

  a = std::fmax(a, b);

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 4.;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(FmaxOperator_FOS_Forward_3) {
  double a = 4., b = 3.2, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = fmax(ad, b);

  ad >>= aout;
  trace_off();

  /* Derivative value is 1.0, as the active variable is grater than the passive
   * one. */
  double aDerivative = 1.;
  a = std::fmax(a, b);

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 4.;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);

  /* Test derivative calculation for a = b. */
  double a1 = 2.5, b1 = 2.5, aout1;
  adouble ad1;

  trace_on(1);
  ad1 <<= a1;

  ad1 = fmax(ad1, b1);

  ad1 >>= aout1;
  trace_off();

  double a1Derivative = 0.;
  a1 = std::fmax(a1, b1);

  double *x1 = myalloc1(1);
  double *xd1 = myalloc1(1);
  double *y1 = myalloc1(1);
  double *yd1 = myalloc1(1);

  *x1 = 2.5;
  *xd1 = -1.3;

  fos_forward(1, 1, 1, 0, x1, xd1, y1, yd1);

  BOOST_TEST(*y1 == a1, tt::tolerance(tol));
  BOOST_TEST(*yd1 == a1Derivative, tt::tolerance(tol));

  *xd1 = 3.7;
  a1Derivative = 3.7;

  fos_forward(1, 1, 1, 0, x1, xd1, y1, yd1);

  BOOST_TEST(*y1 == a1, tt::tolerance(tol));
  BOOST_TEST(*yd1 == a1Derivative, tt::tolerance(tol));

  myfree1(x1);
  myfree1(xd1);
  myfree1(y1);
  myfree1(yd1);
}

BOOST_AUTO_TEST_CASE(FmaxOperator_FOS_Reverse_3) {
  double a = 4., b = 3.2, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = fmax(ad, b);

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;
  a = std::fmax(a, b);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);

  /* Test derivative calculation for a = b. */
  double a1 = 2.5, b1 = 2.5, aout1;
  adouble ad1;

  trace_on(1, 1);
  ad1 <<= a1;

  ad1 = fmax(ad1, b1);

  ad1 >>= aout1;
  trace_off();

  double a1Derivative = 0.5;
  a1 = std::fmax(a1, b1);

  double *u1 = myalloc1(1);
  double *z1 = myalloc1(1);

  *u1 = 1.;

  fos_reverse(1, 1, 1, u1, z1);

  BOOST_TEST(*z1 == a1Derivative, tt::tolerance(tol));

  myfree1(u1);
  myfree1(z1);
}

BOOST_AUTO_TEST_CASE(FminOperator_ZOS_Forward_1) {
  double a = 4., b = 3.2, out;
  adouble ad, bd;

  trace_on(1);
  ad <<= a;
  bd <<= b;

  ad = fmin(ad, bd);

  ad >>= out;
  trace_off();

  a = std::fmin(a, b);

  BOOST_TEST(out == a, tt::tolerance(tol));

  double *x = myalloc1(2);
  double *y = myalloc1(1);

  *x = 4.;
  *(x + 1) = 3.2;

  zos_forward(1, 1, 2, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(FminOperator_FOS_Forward_1) {
  double a = 4., b = 3.2, out;
  adouble ad, bd;

  trace_on(1);
  ad <<= a;
  bd <<= b;

  ad = fmin(ad, bd);

  ad >>= out;
  trace_off();

  double aDerivative = 0.;
  double bDerivative = 1.;
  a = std::fmin(a, b);

  double *x = myalloc1(2);
  double *xd = myalloc1(2);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  /* Test partial derivative wrt a. */
  *x = 4.;
  *(x + 1) = 3.2;
  *xd = 1.;
  *(xd + 1) = 0.;

  fos_forward(1, 1, 2, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  /* Test partial derivative wrt b. */
  *xd = 0.;
  *(xd + 1) = 1.;

  fos_forward(1, 1, 2, 0, x, xd, y, yd);

  BOOST_TEST(*yd == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);

  /* Test derivative value for a = b. */
  double *x1 = myalloc1(2);
  double *xd1 = myalloc1(2);
  double *y1 = myalloc1(1);
  double *yd1 = myalloc1(1);

  *x1 = 2.5;
  *(x1 + 1) = 2.5;
  *xd1 = 1.3;
  *(xd1 + 1) = 3.7;

  fos_forward(1, 1, 2, 0, x1, xd1, y1, yd1);

  BOOST_TEST(*y1 == 2.5, tt::tolerance(tol));
  BOOST_TEST(*yd1 == 1.3, tt::tolerance(tol));

  myfree1(x1);
  myfree1(xd1);
  myfree1(y1);
  myfree1(yd1);
}

BOOST_AUTO_TEST_CASE(FminOperator_FOS_Reverse_1) {
  double a = 4., b = 3.2, out;
  adouble ad, bd;

  trace_on(1, 1);
  ad <<= a;
  bd <<= b;

  ad = fmin(ad, bd);

  ad >>= out;
  trace_off();

  double aDerivative = 0.;
  double bDerivative = 1.;

  double *u = myalloc1(1);
  double *z = myalloc1(2);

  *u = 1.;

  fos_reverse(1, 1, 2, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));
  BOOST_TEST(*(z + 1) == bDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);

  /* Test reverse derivative value for a = b. */
  double a1 = 2.5, b1 = 2.5, out1;
  adouble ad1, bd1;

  trace_on(1, 1);
  ad1 <<= a1;
  bd1 <<= b1;

  ad1 = fmin(ad1, bd1);

  ad1 >>= out1;
  trace_off();

  double *u1 = myalloc1(1);
  double *z1 = myalloc1(2);

  *u1 = 1.;

  fos_reverse(1, 1, 2, u1, z1);

  BOOST_TEST(*z1 == 0.5, tt::tolerance(tol));
  BOOST_TEST(*(z1 + 1) == 0.5, tt::tolerance(tol));

  myfree1(u1);
  myfree1(z1);
}

BOOST_AUTO_TEST_CASE(FminOperator_ZOS_Forward_2) {
  double a = 4., b = 3.2, bout;
  adouble bd;

  trace_on(1);
  bd <<= b;

  bd = fmin(a, bd);

  bd >>= bout;
  trace_off();

  b = std::fmin(a, b);

  BOOST_TEST(bout == b, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 3.2;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == b, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(FminOperator_FOS_Forward_2) {
  double a = 4., b = 3.2, bout;
  adouble bd;

  trace_on(1);
  bd <<= b;

  bd = fmin(a, bd);

  bd >>= bout;
  trace_off();

  /* Derivative value is 1.0, as the active variable is smaller than the passive
   * one. */
  double bDerivative = 1.;
  b = std::fmin(a, b);

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 3.2;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == b, tt::tolerance(tol));
  BOOST_TEST(*yd == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);

  /* Test derivative calculation for a = b. */
  double a1 = 2.5, b1 = 2.5, bout1;
  adouble bd1;

  trace_on(1);
  bd1 <<= b1;

  bd1 = fmin(a1, bd1);

  bd1 >>= bout1;
  trace_off();

  double b1Derivative = -1.3;
  b1 = std::fmin(a1, b1);

  double *x1 = myalloc1(1);
  double *xd1 = myalloc1(1);
  double *y1 = myalloc1(1);
  double *yd1 = myalloc1(1);

  *x1 = 2.5;
  *xd1 = -1.3;

  fos_forward(1, 1, 1, 0, x1, xd1, y1, yd1);

  BOOST_TEST(*y1 == b1, tt::tolerance(tol));
  BOOST_TEST(*yd1 == b1Derivative, tt::tolerance(tol));

  *xd1 = 3.7;
  b1Derivative = 0.;

  fos_forward(1, 1, 1, 0, x1, xd1, y1, yd1);

  BOOST_TEST(*y1 == b1, tt::tolerance(tol));
  BOOST_TEST(*yd1 == b1Derivative, tt::tolerance(tol));

  myfree1(x1);
  myfree1(xd1);
  myfree1(y1);
  myfree1(yd1);
}

BOOST_AUTO_TEST_CASE(FminOperator_FOS_Reverse_2) {
  double a = 4., b = 3.2, bout;
  adouble bd;

  trace_on(1, 1);
  bd <<= b;

  bd = fmin(a, bd);

  bd >>= bout;
  trace_off();

  double bDerivative = 1.;
  b = std::fmin(a, b);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == bDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);

  /* Test derivative calculation for a = b. */
  double a1 = 2.5, b1 = 2.5, bout1;
  adouble bd1;

  trace_on(1, 1);
  bd1 <<= b1;

  bd1 = fmin(a1, bd1);

  bd1 >>= bout1;
  trace_off();

  double b1Derivative = 0.5;
  b1 = std::fmin(a1, b1);

  double *u1 = myalloc1(1);
  double *z1 = myalloc1(1);

  *u1 = 1.;

  fos_reverse(1, 1, 1, u1, z1);

  BOOST_TEST(*z1 == b1Derivative, tt::tolerance(tol));

  myfree1(u1);
  myfree1(z1);
}

BOOST_AUTO_TEST_CASE(FminOperator_ZOS_Forward_3) {
  double a = 4., b = 3.2, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = fmin(ad, b);

  ad >>= aout;
  trace_off();

  a = std::fmin(a, b);

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 4.;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(FminOperator_FOS_Forward_3) {
  double a = 4., b = 3.2, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = fmin(ad, b);

  ad >>= aout;
  trace_off();

  /* Derivative value is 0.0, as the active variable is grater than the passive
   * one. */
  double aDerivative = 0.;
  a = std::fmin(a, b);

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 4.;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);

  /* Test derivative calculation for a = b. */
  double a1 = 2.5, b1 = 2.5, aout1;
  adouble ad1;

  trace_on(1);
  ad1 <<= a1;

  ad1 = fmin(ad1, b1);

  ad1 >>= aout1;
  trace_off();

  double a1Derivative = -1.3;
  a1 = std::fmin(a1, b1);

  double *x1 = myalloc1(1);
  double *xd1 = myalloc1(1);
  double *y1 = myalloc1(1);
  double *yd1 = myalloc1(1);

  *x1 = 2.5;
  *xd1 = -1.3;

  fos_forward(1, 1, 1, 0, x1, xd1, y1, yd1);

  BOOST_TEST(*y1 == a1, tt::tolerance(tol));
  BOOST_TEST(*yd1 == a1Derivative, tt::tolerance(tol));

  *xd1 = 3.7;
  a1Derivative = 0.;

  fos_forward(1, 1, 1, 0, x1, xd1, y1, yd1);

  BOOST_TEST(*y1 == a1, tt::tolerance(tol));
  BOOST_TEST(*yd1 == a1Derivative, tt::tolerance(tol));

  myfree1(x1);
  myfree1(xd1);
  myfree1(y1);
  myfree1(yd1);
}

BOOST_AUTO_TEST_CASE(FminOperator_FOS_Reverse_3) {
  double a = 4., b = 3.2, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = fmin(ad, b);

  ad >>= aout;
  trace_off();

  double aDerivative = 0.;
  a = std::fmin(a, b);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);

  /* Test derivative calculation for a = b. */
  double a1 = 2.5, b1 = 2.5, aout1;
  adouble ad1;

  trace_on(1, 1);
  ad1 <<= a1;

  ad1 = fmin(ad1, b1);

  ad1 >>= aout1;
  trace_off();

  double a1Derivative = 0.5;
  a1 = std::fmin(a1, b1);

  double *u1 = myalloc1(1);
  double *z1 = myalloc1(1);

  *u1 = 1.;

  fos_reverse(1, 1, 1, u1, z1);

  BOOST_TEST(*z1 == a1Derivative, tt::tolerance(tol));

  myfree1(u1);
  myfree1(z1);
}

BOOST_AUTO_TEST_CASE(ErfOperator_ZOS_Forward) {
  double a = 7.1, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = erf(ad);

  ad >>= aout;
  trace_off();

  a = std::erf(a);

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 7.1;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(ErfOperator_FOS_Forward) {
  double a = 7.1, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = erf(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 2. / std::sqrt(std::acos(-1.)) * std::exp(-a * a);
  a = std::erf(a);

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 7.1;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(ErfOperator_FOS_Reverse) {
  double a = 7.1, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = erf(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 2. / std::sqrt(std::acos(-1.)) * std::exp(-a * a);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(ErfcOperator_ZOS_Forward) {
  double a = 7.1, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = erfc(ad);

  ad >>= aout;
  trace_off();

  a = std::erfc(a);

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 7.1;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(ErfcOperator_FOS_Forward) {
  double a = 7.1, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = erfc(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = -2. / std::sqrt(std::acos(-1.)) * std::exp(-a * a);
  a = std::erfc(a);

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 7.1;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(ErfcOperator_FOS_Reverse) {
  double a = 7.1, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = erfc(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = -2. / std::sqrt(std::acos(-1.)) * std::exp(-a * a);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(EqOperator_ZOS_Forward) {
  double a = 10.01, aout;
  adouble ad, bd;

  trace_on(1);
  ad <<= a;

  bd = ad;

  bd >>= aout;
  trace_off();

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 10.01;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(EqOperator_FOS_Forward) {
  double a = 10.01, aout;
  adouble ad, bd;

  trace_on(1);
  ad <<= a;

  bd = ad;

  bd >>= aout;
  trace_off();

  double aDerivative = 1.;

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 10.01;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(EqOperator_FOS_Reverse) {
  double a = 10.01, aout;
  adouble ad, bd;

  trace_on(1, 1);
  ad <<= a;

  bd = ad;

  bd >>= aout;
  trace_off();

  double aDerivative = 1.;

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(EqPlusOperator_ZOS_Forward) {
  double a = 5.132, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad += 5.2;

  ad >>= aout;
  trace_off();

  a += 5.2;

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 5.132;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(EqPlusOperator_FOS_Forward) {
  double a = 5.132, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad += 5.2;

  ad >>= aout;
  trace_off();

  a += 5.2;
  double aDerivative = 1.;

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 5.132;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(EqPlusOperator_FOS_Reverse) {
  double a = 5.132, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad += 5.2;

  ad >>= aout;
  trace_off();

  a += 5.2;
  double aDerivative = 1.;

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(EqMinusOperator_ZOS_Forward) {
  double a = 5.132, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad -= 5.2;

  ad >>= aout;
  trace_off();

  a -= 5.2;

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 5.132;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(EqMinusOperator_FOS_Forward) {
  double a = 5.132, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad -= 5.2;

  ad >>= aout;
  trace_off();

  a -= 5.2;
  double aDerivative = 1.;

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 5.132;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(EqMinusOperator_FOS_Reverse) {
  double a = 5.132, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad -= 5.2;

  ad >>= aout;
  trace_off();

  a -= 5.2;
  double aDerivative = 1.;

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(EqTimesOperator_ZOS_Forward_1) {
  double a = 5.132, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad *= 5.2;

  ad >>= aout;
  trace_off();

  a *= 5.2;

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 5.132;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(EqTimesOperator_FOS_Forward_1) {
  double a = 5.132, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad *= 5.2;

  ad >>= aout;
  trace_off();

  a *= 5.2;
  double aDerivative = 5.2;

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 5.132;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(EqTimesOperator_FOS_Reverse_1) {
  double a = 5.132, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad *= 5.2;

  ad >>= aout;
  trace_off();

  a *= 5.2;
  double aDerivative = 5.2;

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(EqTimesOperator_ZOS_Forward_2) {
  double a = 5.132, b = 11.1, out;
  adouble ad, bd;

  trace_on(1);
  ad <<= a;
  bd <<= b;

  ad *= bd;

  ad >>= out;
  trace_off();

  a *= b;

  BOOST_TEST(out == a, tt::tolerance(tol));

  double *x = myalloc1(2);
  double *y = myalloc1(1);

  *x = 5.132;
  *(x + 1) = 11.1;

  zos_forward(1, 1, 2, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(EqTimesOperator_FOS_Forward_2) {
  double a = 5.132, b = 11.1, out;
  adouble ad, bd;

  trace_on(1);
  ad <<= a;
  bd <<= b;

  ad *= bd;

  ad >>= out;
  trace_off();

  double aDerivative = b;
  double bDerivative = a;
  a *= b;

  double *x = myalloc1(2);
  double *xd = myalloc1(2);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  /* Test partial derivative wrt a. */
  *x = 5.132;
  *(x + 1) = 11.1;
  *xd = 1.;
  *(xd + 1) = 0.;

  fos_forward(1, 1, 2, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  /* Test partial derivative wrt b. */
  *xd = 0.;
  *(xd + 1) = 1.;

  fos_forward(1, 1, 2, 0, x, xd, y, yd);

  BOOST_TEST(*yd == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(EqTimesOperator_FOS_Reverse_2) {
  double a = 5.132, b = 11.1, out;
  adouble ad, bd;

  trace_on(1, 1);
  ad <<= a;
  bd <<= b;

  ad *= bd;

  ad >>= out;
  trace_off();

  double aDerivative = b;
  double bDerivative = a;

  double *u = myalloc1(1);
  double *z = myalloc1(2);

  *u = 1.;

  fos_reverse(1, 1, 2, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));
  BOOST_TEST(*(z + 1) == bDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(EqDivOperator_ZOS_Forward_1) {
  double a = 5.132, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad /= 5.2;

  ad >>= aout;
  trace_off();

  a /= 5.2;

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 5.132;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(EqDivOperator_FOS_Forward_1) {
  double a = 5.132, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad /= 5.2;

  ad >>= aout;
  trace_off();

  a /= 5.2;
  double aDerivative = 1. / 5.2;

  double *x = myalloc1(1);
  double *xd = myalloc1(1);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 5.132;
  *xd = 1.;

  fos_forward(1, 1, 1, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(EqDivOperator_FOS_Reverse_1) {
  double a = 5.132, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad /= 5.2;

  ad >>= aout;
  trace_off();

  a /= 5.2;
  double aDerivative = 1. / 5.2;

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;

  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(EqDivOperator_ZOS_Forward_2) {
  double a = 5.132, b = 11.1, out;
  adouble ad, bd;

  trace_on(1);
  ad <<= a;
  bd <<= b;

  ad /= bd;

  ad >>= out;
  trace_off();

  a /= b;

  BOOST_TEST(out == a, tt::tolerance(tol));

  double *x = myalloc1(2);
  double *y = myalloc1(1);

  *x = 5.132;
  *(x + 1) = 11.1;

  zos_forward(1, 1, 2, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(EqDivOperator_FOS_Forward_2) {
  double a = 5.132, b = 11.1, out;
  adouble ad, bd;

  trace_on(1);
  ad <<= a;
  bd <<= b;

  ad /= bd;

  ad >>= out;
  trace_off();

  double aDerivative = 1. / b;
  double bDerivative = -a / (b * b);
  a /= b;

  double *x = myalloc1(2);
  double *xd = myalloc1(2);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  /* Test partial derivative wrt a. */
  *x = 5.132;
  *(x + 1) = 11.1;
  *xd = 1.;
  *(xd + 1) = 0.;

  fos_forward(1, 1, 2, 0, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(*yd == aDerivative, tt::tolerance(tol));

  /* Test partial derivative wrt b. */
  *xd = 0.;
  *(xd + 1) = 1.;

  fos_forward(1, 1, 2, 0, x, xd, y, yd);

  BOOST_TEST(*yd == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}

BOOST_AUTO_TEST_CASE(EqDivOperator_FOS_Reverse_2) {
  double a = 5.132, b = 11.1, out;
  adouble ad, bd;

  trace_on(1, 1);
  ad <<= a;
  bd <<= b;

  ad /= bd;

  ad >>= out;
  trace_off();

  double aDerivative = 1. / b;
  double bDerivative = -a / (b * b);

  double *u = myalloc1(1);
  double *z = myalloc1(2);

  *u = 1.;

  fos_reverse(1, 1, 2, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));
  BOOST_TEST(*(z + 1) == bDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(CondassignOperator_ZOS_Forward) {
  double out;
  adouble cond, arg1, arg2;
  adouble p;

  trace_on(1);
  cond <<= 1.;
  arg1 <<= 3.5;
  arg2 <<= 5.3;

  condassign(p, cond, arg1, arg2);

  p >>= out;
  trace_off();

  BOOST_TEST(out == 3.5, tt::tolerance(tol));

  double *x = myalloc1(3);
  double *y = myalloc1(1);

  *x = 1.;
  *(x + 1) = 3.5;
  *(x + 2) = 5.3;

  zos_forward(1, 1, 3, 0, x, y);

  BOOST_TEST(*y == 3.5, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(CondassignOperator_FOS_Forward) {
  double out;
  adouble cond, arg1, arg2;
  adouble p;

  trace_on(1);
  cond <<= 1.;
  arg1 <<= 3.5;
  arg2 <<= 5.3;

  condassign(p, cond, arg1, arg2);

  p >>= out;
  trace_off();

  BOOST_TEST(out == 3.5, tt::tolerance(tol));

  double *x = myalloc1(3);
  double *xd = myalloc1(3);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 1.;
  *(x + 1) = 3.5;
  *(x + 2) = 5.3;
  *xd = 0.;
  *(xd + 1) = 0.1;
  *(xd + 2) = 0.2;

  fos_forward(1, 1, 3, 0, x, xd, y, yd);

  BOOST_TEST(*y == 3.5, tt::tolerance(tol));
  BOOST_TEST(*yd == 0.1, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}
/*
BOOST_AUTO_TEST_CASE(CondassignOperator_FOS_Reverse)
{
  double out;
  adouble cond, arg1, arg2;
  adouble p;

  trace_on(1);
  cond <<= 1.;
  arg1 <<= 3.5;
  arg2 <<= 5.3;

  condassign(p, cond, arg1, arg2);

  p >>= out;
  trace_off();

  double aDerivative = 1.;
  double bDerivative = 0.;

  double *u = myalloc1(1);
  double *z = myalloc1(3);

  *u = 1.;

  fos_reverse(1, 1, 3, u, z);

  BOOST_TEST(*z == 0., tt::tolerance(tol));
  BOOST_TEST(*(z + 1) == aDerivative, tt::tolerance(tol));
  BOOST_TEST(*(z + 2) == bDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}
*/
BOOST_AUTO_TEST_CASE(CondeqassignOperator_ZOS_Forward) {
  double out;
  adouble cond, arg1, arg2;
  adouble p;

  trace_on(1);
  cond <<= 1.;
  arg1 <<= 3.5;
  arg2 <<= 5.3;

  condeqassign(p, cond, arg1, arg2);

  p >>= out;
  trace_off();

  BOOST_TEST(out == 3.5, tt::tolerance(tol));

  double *x = myalloc1(3);
  double *y = myalloc1(1);

  *x = 1.;
  *(x + 1) = 3.5;
  *(x + 2) = 5.3;

  zos_forward(1, 1, 3, 0, x, y);

  BOOST_TEST(*y == 3.5, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(CondeqassignOperator_FOS_Forward) {
  double out;
  adouble cond, arg1, arg2;
  adouble p;

  trace_on(1);
  cond <<= 1.;
  arg1 <<= 3.5;
  arg2 <<= 5.3;

  condeqassign(p, cond, arg1, arg2);

  p >>= out;
  trace_off();

  BOOST_TEST(out == 3.5, tt::tolerance(tol));

  double *x = myalloc1(3);
  double *xd = myalloc1(3);
  double *y = myalloc1(1);
  double *yd = myalloc1(1);

  *x = 1.;
  *(x + 1) = 3.5;
  *(x + 2) = 5.3;
  *xd = 0.;
  *(xd + 1) = 0.1;
  *(xd + 2) = 0.2;

  fos_forward(1, 1, 3, 0, x, xd, y, yd);

  BOOST_TEST(*y == 3.5, tt::tolerance(tol));
  BOOST_TEST(*yd == 0.1, tt::tolerance(tol));

  myfree1(x);
  myfree1(xd);
  myfree1(y);
  myfree1(yd);
}
/*
BOOST_AUTO_TEST_CASE(CondeqassignOperator_FOS_Reverse)
{
  double out;
  adouble cond, arg1, arg2;
  adouble p;

  trace_on(1);
  cond <<= 1.;
  arg1 <<= 3.5;
  arg2 <<= 5.3;

  condeqassign(p, cond, arg1, arg2);

  p >>= out;
  trace_off();

  double aDerivative = 1.;
  double bDerivative = 0.;

  double *u = myalloc1(1);
  double *z = myalloc1(3);

  *u = 1.;

  fos_reverse(1, 1, 3, u, z);

  BOOST_TEST(*z == 0., tt::tolerance(tol));
  BOOST_TEST(*(z + 1) == aDerivative, tt::tolerance(tol));
  BOOST_TEST(*(z + 2) == bDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}
*/

/* Boolean operations only alter the trace, if advanced branching is
 * activated.  This advanced branching is not tested here.
 */

BOOST_AUTO_TEST_CASE(TraceNotOperatorPrimal) {
  double a = 1.0;
  adouble ad = a;

  BOOST_TEST(!a == 0.0, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(TraceCompNeqOperatorPrimal) {
  double a = 1.5, b = 0.5;
  adouble ad = a, bd = b;

  int n = (ad != bd);
  int m = (a != b);

  BOOST_TEST(n == m, tt::tolerance(tol));

  int k = (ad != a);
  int l = (a != a);

  BOOST_TEST(k == l, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(TraceCompEqOperatorPrimal) {
  double a = 0.5, b = 1.5;
  adouble ad = a, bd = b;

  int n = (ad == bd);
  int m = (a == b);

  BOOST_TEST(n == m, tt::tolerance(tol));

  int k = (ad == a);
  int l = (a == a);

  BOOST_TEST(k == l, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(TraceCompLeqOperatorPrimal) {
  double a = 1.0, b = 0.99;
  adouble ad = a, bd = b;

  int n = (ad <= bd);
  int m = (a <= b);

  BOOST_TEST(n == m, tt::tolerance(tol));

  int k = (ad <= a);
  int l = (a <= a);

  BOOST_TEST(k == l, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(TraceCompGeqOperatorPrimal) {
  double a = 1.2, b = 2.5;
  adouble ad = a, bd = b;

  int n = (ad >= bd);
  int m = (a >= b);

  BOOST_TEST(n == m, tt::tolerance(tol));

  int k = (ad >= a);
  int l = (a >= a);

  BOOST_TEST(k == l, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(TraceCompLeOperatorPrimal) {
  double a = 1.1, b = 1.1;
  adouble ad = a, bd = b;

  int n = (ad < bd);
  int m = (a < b);

  BOOST_TEST(n == m, tt::tolerance(tol));

  int k = (ad < a);
  int l = (a < a);

  BOOST_TEST(k == l, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(TraceCompGeOperatorPrimal) {
  double a = 1.7, b = 7.5;
  adouble ad = a, bd = b;

  int n = (ad > bd);
  int m = (a > b);

  BOOST_TEST(n == m, tt::tolerance(tol));

  int k = (ad > a);
  int l = (a > a);

  BOOST_TEST(k == l, tt::tolerance(tol));
}

/* Implementation of PowOperator_FOS_Reverse_1 does not work.  Why?
 * Apparently, PowOperator_FOS_Reverse_3 works fine (for some reason...).
 * Also, the implementations for LdexpOperator_1 and LdexpOperator_3 do
 * not work. It seems, that no implementations of ldexp(double, adouble)
 * and ldexp(adouble, adouble) exist.
 * Implementation for CondeqassignOperator_FOS_Reverse and
 * CondassignOperator_FOS_Reverse do not work either (same error message)
 * as for PowOperator_FOS_Reverse_1...).
 */

/* What does reverse mode do for fmax(), fmin() with a = b? */

BOOST_AUTO_TEST_SUITE_END()

#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

#include <adolc/adolc.h>

#include "const.h"

BOOST_AUTO_TEST_SUITE( trace_scalar )


/**********************************************/
/* Tests for ADOL-C trace scalar mode drivers */
/* zos_forward, fos_forward and fos_reverse   */
/* Author: Philipp Schuette                   */
/**********************************************/


/* Naming convention for test cases:  Operatorname_Operator_ZOS_Forward for
 * primal value (= zero order derivative), Operatorname_Operator_FOS_Forward
 * for foward derivative evulation and Operatorname_Operator_FOS_Reverse for
 * reverse mode derivative evalution.
 *
 * For documentation of concrete test implementation, check traceless scalar
 * mode test implementation.  The testing order is consistent with that file
 * as well.
 */

BOOST_AUTO_TEST_CASE(ExpOperator_ZOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(ExpOperator_FOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(ExpOperator_FOS_Reverse)
{
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

BOOST_AUTO_TEST_CASE(MultOperator_ZOS_Forward)
{
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

  *x = 2.;
  *(x + 1) = 3.5;
  
  zos_forward(1, 1, 2, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_CASE(MultOperator_FOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(MultOperator_FOS_Reverse)
{
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

BOOST_AUTO_TEST_CASE(AddOperator_ZOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(AddOperator_FOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(AddOperator_FOS_Reverse)
{
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

BOOST_AUTO_TEST_CASE(SubOperator_ZOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(SubOperator_FOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(SubOperator_FOS_Reverse)
{
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

BOOST_AUTO_TEST_CASE(DivOperator_ZOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(DivOperator_FOS_Forward)
{
  double a = 0.5, b = 4.5, out;
  adouble ad, bd;

  trace_on(1);
  ad <<= a;
  bd <<= b;

  ad = ad / bd;

  ad >>= out;
  trace_off();

  double aDerivative = 1. / b;
  double bDerivative = -a / (b*b);
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

BOOST_AUTO_TEST_CASE(DivOperator_FOS_Reverse)
{
  double a = 0.5, b = 4.5, out;
  adouble ad, bd;

  trace_on(1, 1);
  ad <<= a;
  bd <<= b;

  ad = ad / bd;

  ad >>= out;
  trace_off();

  double aDerivative = 1. / b;
  double bDerivative = -a / (b*b);

  double *u = myalloc1(1);
  double *z = myalloc1(2);

  *u = 1.;
  
  fos_reverse(1, 1, 2, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));
  BOOST_TEST(*(z + 1) == bDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(TanOperator_ZOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(TanOperator_FOS_Forward)
{
  double a = 0.7, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = tan(ad);

  ad >>= aout;
  trace_off();

  a = std::tan(a);
  double aDerivative = 1. + a*a;

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

BOOST_AUTO_TEST_CASE(TanOperator_FOS_Reverse)
{
  double a = 0.7, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = tan(ad);

  ad >>= aout;
  trace_off();

  a = std::tan(a);
  double aDerivative = 1. + a*a;

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;
  
  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(SinOperator_ZOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(SinOperator_FOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(SinOperator_FOS_Reverse)
{
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

BOOST_AUTO_TEST_CASE(CosOperator_ZOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(CosOperator_FOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(CosOperator_FOS_Reverse)
{
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

BOOST_AUTO_TEST_CASE(SqrtOperator_ZOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(SqrtOperator_FOS_Forward)
{
  double a = 2.2, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = sqrt(ad);

  ad >>= aout;
  trace_off();

  a = std::sqrt(a);
  double aDerivative = 1. / (2*a);

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

BOOST_AUTO_TEST_CASE(SqrtOperator_FOS_Reverse)
{
  double a = 2.2, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = sqrt(ad);

  ad >>= aout;
  trace_off();

  a = std::sqrt(a);
  double aDerivative = 1. / (2*a);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;
  
  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(LogOperator_ZOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(LogOperator_FOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(LogOperator_FOS_Reverse)
{
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

BOOST_AUTO_TEST_CASE(SinhOperator_ZOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(SinhOperator_FOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(SinhOperator_FOS_Reverse)
{
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

BOOST_AUTO_TEST_CASE(CoshOperator_ZOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(CoshOperator_FOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(CoshOperator_FOS_Reverse)
{
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

BOOST_AUTO_TEST_CASE(TanhOperator_ZOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(TanhOperator_FOS_Forward)
{
  double a = 4., aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = tanh(ad);

  ad >>= aout;
  trace_off();

  a = std::tanh(a);
  double aDerivative = 1. - a*a;

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

BOOST_AUTO_TEST_CASE(TanhOperator_FOS_Reverse)
{
  double a = 4., aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = tanh(ad);

  ad >>= aout;
  trace_off();

  a = std::tanh(a);
  double aDerivative = 1. - a*a;

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;
  
  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(AsinOperator_ZOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(AsinOperator_FOS_Forward)
{
  double a = 0.9, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = asin(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (std::sqrt(1. - a*a));
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

BOOST_AUTO_TEST_CASE(AsinOperator_FOS_Reverse)
{
  double a = 0.9, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = asin(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (std::sqrt(1. - a*a));
  a = std::asin(a);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;
  
  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(AcosOperator_ZOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(AcosOperator_FOS_Forward)
{
  double a = 0.8, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = acos(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = -1. / (std::sqrt(1. - a*a));
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

BOOST_AUTO_TEST_CASE(AcosOperator_FOS_Reverse)
{
  double a = 0.8, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = acos(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = -1. / (std::sqrt(1. - a*a));
  a = std::acos(a);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;
  
  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(AtanOperator_ZOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(AtanOperator_FOS_Forward)
{
  double a = 9.8, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = atan(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (1. + a*a);
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

BOOST_AUTO_TEST_CASE(AtanOperator_FOS_Reverse)
{
  double a = 9.8, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = atan(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (1. + a*a);
  a = std::atan(a);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;
  
  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(Log10Operator_ZOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(Log10Operator_FOS_Forward)
{
  double a = 12.3, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = log10(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (a*std::log(10));
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

BOOST_AUTO_TEST_CASE(Log10Operator_FOS_Reverse)
{
  double a = 12.3, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = log10(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (a*std::log(10));
  a = std::log10(a);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;
  
  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

#if defined(ATRIG_ERF)
BOOST_AUTO_TEST_CASE(AsinhOperator_ZOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(AsinhOperator_FOS_Forward)
{
  double a = 0.6, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = asinh(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (std::sqrt(a*a + 1.));
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

BOOST_AUTO_TEST_CASE(AsinhOperator_FOS_Reverse)
{
  double a = 0.6, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = asinh(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (std::sqrt(a*a + 1.));
  a = std::asinh(a);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;
  
  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(AcoshOperator_ZOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(AcoshOperator_FOS_Forward)
{
  double a = 1.7, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = acosh(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (std::sqrt(a*a - 1));
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

BOOST_AUTO_TEST_CASE(AcoshOperator_FOS_Reverse)
{
  double a = 1.7, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = acosh(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (std::sqrt(a*a - 1.));
  a = std::acosh(a);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;
  
  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}

BOOST_AUTO_TEST_CASE(AtanhOperator_ZOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(AtanhOperator_FOS_Forward)
{
  double a = 0.6, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = atanh(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (1. - a*a);
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

BOOST_AUTO_TEST_CASE(AtanhOperator_FOS_Reverse)
{
  double a = 0.6, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = atanh(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (1. - a*a);
  a = std::atanh(a);

  double *u = myalloc1(1);
  double *z = myalloc1(1);

  *u = 1.;
  
  fos_reverse(1, 1, 1, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}
#endif

BOOST_AUTO_TEST_CASE(InclOperator_ZOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(InclOperator_FOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(InclOperator_FOS_Reverse)
{
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

BOOST_AUTO_TEST_CASE(DeclOperator_ZOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(DeclOperator_FOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(DeclOperator_FOS_Reverse)
{
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

BOOST_AUTO_TEST_CASE(SignPlusOperator_ZOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(SignPlusOperator_FOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(SignPlusOperator_FOS_Reverse)
{
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

BOOST_AUTO_TEST_CASE(SignMinusOperator_ZOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(SignMinusOperator_FOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(SignMinusOperator_FOS_Reverse)
{
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

BOOST_AUTO_TEST_CASE(Atan2Operator_ZOS_Forward)
{
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

BOOST_AUTO_TEST_CASE(Atan2Operator_FOS_Forward)
{
  double a = 12.3, b = 2.1, out;
  adouble ad, bd;

  trace_on(1);
  ad <<= a;
  bd <<= b;

  ad = atan2(ad, bd);

  ad >>= out;
  trace_off();

  double aDerivative = b / (a*a + b*b);
  double bDerivative = -a / (a*a + b*b);
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

BOOST_AUTO_TEST_CASE(Atan2Operator_FOS_Reverse)
{
  double a = 12.3, b = 2.1, out;
  adouble ad, bd;

  trace_on(1, 1);
  ad <<= a;
  bd <<= b;

  ad = atan2(ad, bd);

  ad >>= out;
  trace_off();

  double aDerivative = b / (a*a + b*b);
  double bDerivative = -a / (a*a + b*b);

  double *u = myalloc1(1);
  double *z = myalloc1(2);

  *u = 1.;
  
  fos_reverse(1, 1, 2, u, z);

  BOOST_TEST(*z == aDerivative, tt::tolerance(tol));
  BOOST_TEST(*(z + 1) == bDerivative, tt::tolerance(tol));

  myfree1(u);
  myfree1(z);
}




BOOST_AUTO_TEST_SUITE_END()





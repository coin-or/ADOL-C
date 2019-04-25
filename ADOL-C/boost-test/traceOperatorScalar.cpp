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




BOOST_AUTO_TEST_SUITE_END()





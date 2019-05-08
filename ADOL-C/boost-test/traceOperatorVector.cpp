#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

#include <adolc/adolc.h>

#include "const.h"

BOOST_AUTO_TEST_SUITE( trace_vector )


/***********************************/
/* Tests for traceless scalar mode */
/* Author: Philipp Schuette        */
/***********************************/


/* Naming convention for test cases:  Operatorname_Operator_FOV_Forward for
 * foward derivative evulation in vector mode.
 *
 * For documentation of concrete test implementation, check traceless scalar
 * mode test implementation.  The testing order is consistent with that file
 * as well.
 */

BOOST_AUTO_TEST_CASE(MultOperator_FOV_Forward)
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
  double **xd = myalloc2(2, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  /* Test partial derivative wrt a and b. */
  x[0] = 2.;
  x[1] = 3.5;
  
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      if (i == j)
        xd[i][j] = 1.;
      else
        xd[i][j] = 0.;
    }
  }

  fov_forward(1, 1, 2, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(AddOperator_FOV_Forward)
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
  double **xd = myalloc2(2, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  /* Test partial derivative wrt a and b. */
  x[0] = 2.5;
  x[1] = 3.;
  
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      if (i == j)
        xd[i][j] = 1.;
      else
        xd[i][j] = 0.;
    }
  }

  fov_forward(1, 1, 2, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(SubOperator_FOV_Forward)
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
  double **xd = myalloc2(2, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  /* Test partial derivative wrt a and b. */
  x[0] = 1.5;
  x[1] = 3.2;
  
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      if (i == j)
        xd[i][j] = 1.;
      else
        xd[i][j] = 0.;
    }
  }

  fov_forward(1, 1, 2, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(DivOperator_FOV_Forward)
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
  double **xd = myalloc2(2, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  /* Test partial derivative wrt a and b. */
  x[0] = 0.5;
  x[1] = 4.5;
  
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      if (i == j)
        xd[i][j] = 1.;
      else
        xd[i][j] = 0.;
    }
  }

  fov_forward(1, 1, 2, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(TanOperator_FOV_Forward)
{
  double a = 0.7, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = tan(ad);

  ad >>= aout;
  trace_off();

  a = tan(a);
  double aDerivative = 1. + a*a;

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  /* Test partial derivative wrt a and b. */
  x[0] = 0.7;
  
  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. + std::pow(2, i);
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative * (1. + std::pow(2, 0)), tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. + std::pow(2, 1)), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(SinOperator_FOV_Forward)
{
  double a = 1.2, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = sin(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = std::cos(a);
  a = sin(a);

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  /* Test partial derivative wrt a and b. */
  x[0] = 1.2;
  
  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. + i * (-2.0);
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative , tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (-1.), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(CosOperator_FOV_Forward)
{
  double a = 1.2, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = cos(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = -std::sin(a);
  a = cos(a);

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  /* Test partial derivative wrt a and b. */
  x[0] = 1.2;
  
  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. + i * 2.;
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * 3., tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(SqrtOperator_FOV_Forward)
{
  double a = 2.2, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = sqrt(ad);

  ad >>= aout;
  trace_off();

  a = std::sqrt(a);
  double aDerivative = 1. / (2.*a);

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  /* Test partial derivative wrt a and b. */
  x[0] = 2.2;
  
  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. * std::pow(2, i);
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * 2., tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(LogOperator_FOV_Forward)
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
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  /* Test partial derivative wrt a and b. */
  x[0] = 4.9;
  
  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. + i * 5.5;
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * 6.5, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(SinhOperator_FOV_Forward)
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
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  /* Test partial derivative wrt a and b. */
  x[0] = 4.;
  
  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - std::sqrt(2.*i);
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. - std::sqrt(2.)), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(CoshOperator_FOV_Forward)
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
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  /* Test partial derivative wrt a and b. */
  x[0] = 4.;
  
  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. + i * 3.2;
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * 4.2, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(TanhOperator_FOV_Forward)
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
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  /* Test partial derivative wrt a and b. */
  x[0] = 4.;
  
  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - 1.3*i;
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. - 1.3), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}




BOOST_AUTO_TEST_SUITE_END()


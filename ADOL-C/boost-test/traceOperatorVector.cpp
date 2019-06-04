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

BOOST_AUTO_TEST_CASE(ExpOperator_FOV_Forward)
{
  double a = 2., aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = exp(ad);

  ad >>= aout;
  trace_off();

  a = std::exp(a);
  double aDerivative = a;

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 2.0;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. + i;
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

BOOST_AUTO_TEST_CASE(ExpOperator_FOV_Reverse)
{
  double a = 2., aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = exp(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = std::exp(a);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = std::exp(3.);

  fov_reverse(1, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative*std::exp(3.), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

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

BOOST_AUTO_TEST_CASE(MultOperator_FOV_Reverse)
{
  double a = 2., b = 3.5, aout;
  adouble ad, bd;

  trace_on(1, 1);
  ad <<= a;
  bd <<= b;

  ad = ad * bd;

  ad >>= aout;
  trace_off();

  double aDerivative = b;
  double bDerivative = a;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 2);

  u[0][0] = 1.;
  u[1][0] = 2.;

  fov_reverse(1, 1, 2, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == 2.*aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == 2.*bDerivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
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

BOOST_AUTO_TEST_CASE(AddOperator_FOV_Reverse)
{
  double a = 2.5, b = 3., aout;
  adouble ad, bd;

  trace_on(1, 1);
  ad <<= a;
  bd <<= b;

  ad = ad + bd;

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;
  double bDerivative = 1.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 2);

  u[0][0] = 1.;
  u[1][0] = 9.;

  fov_reverse(1, 1, 2, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == 9.*aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == 9.*bDerivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
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

BOOST_AUTO_TEST_CASE(SubOperator_FOV_Reverse)
{
  double a = 1.5, b = 3.2, aout;
  adouble ad, bd;

  trace_on(1, 1);
  ad <<= a;
  bd <<= b;

  ad = ad - bd;

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;
  double bDerivative = -1.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 2);

  u[0][0] = 1.;
  u[1][0] = std::sqrt(2);

  fov_reverse(1, 1, 2, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == std::sqrt(2)*aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == std::sqrt(2)*bDerivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
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

BOOST_AUTO_TEST_CASE(DivOperator_FOV_Reverse)
{
  double a = 0.5, b = 4.5, aout;
  adouble ad, bd;

  trace_on(1, 1);
  ad <<= a;
  bd <<= b;

  ad = ad / bd;

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / b;
  double bDerivative = -a / (b*b);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 2);

  u[0][0] = 1.;
  u[1][0] = 0.9;

  fov_reverse(1, 1, 2, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == 0.9*aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == 0.9*bDerivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
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

BOOST_AUTO_TEST_CASE(TanOperator_FOV_Reverse)
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

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 1.1;

  fov_reverse(1, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative*1.1, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
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

BOOST_AUTO_TEST_CASE(SinOperator_FOV_Reverse)
{
  double a = 1.2, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = sin(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = std::cos(a);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = std::tan(4.4);

  fov_reverse(1, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative*std::tan(4.4), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
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

BOOST_AUTO_TEST_CASE(CosOperator_FOV_Reverse)
{
  double a = 1.2, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = cos(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = -std::sin(a);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = std::log(2.);

  fov_reverse(1, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative*std::log(2.), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
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

BOOST_AUTO_TEST_CASE(SqrtOperator_FOV_Reverse)
{
  double a = 2.2, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = sqrt(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (2.*std::sqrt(a));

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = std::exp(2.);

  fov_reverse(1, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative*std::exp(2.), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
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

BOOST_AUTO_TEST_CASE(LogOperator_FOV_Reverse)
{
  double a = 4.9, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = log(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / a;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = std::exp(-1.);

  fov_reverse(1, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative*std::exp(-1.), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
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

BOOST_AUTO_TEST_CASE(SinhOperator_FOV_Reverse)
{
  double a = 4., aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = sinh(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = std::cosh(a);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = std::cosh(3.5);

  fov_reverse(1, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative*std::cosh(3.5), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
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

BOOST_AUTO_TEST_CASE(CoshOperator_FOV_Reverse)
{
  double a = 4., aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = cosh(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = std::sinh(a);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = std::sinh(3.5);

  fov_reverse(1, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative*std::sinh(3.5), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
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

BOOST_AUTO_TEST_CASE(TanhOperator_FOV_Reverse)
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

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 5.4;

  fov_reverse(1, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == 5.4*aDerivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(AsinOperator_FOV_Forward)
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
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 0.9;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. + i*(i + 1.7)*4.3;
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative*(1. + 2.7*4.3), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(AsinOperator_FOV_Reverse)
{
  double a = 0.9, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = asin(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (std::sqrt(1. - a*a));

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 1. + 2.7*4.3;

  fov_reverse(1, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative*(1. + 2.7*4.3), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(AcosOperator_FOV_Forward)
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
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 0.8;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i*(i + 0.7)*3.4;
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative*(1. - 1.7*3.4), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(AcosOperator_FOV_Reverse)
{
  double a = 0.8, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = acos(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = -1. / (std::sqrt(1. - a*a));

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 1. - 1.7*3.4;

  fov_reverse(1, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative*(1. - 1.7*3.4), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(AtanOperator_FOV_Forward)
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
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 9.8;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i*(i - 0.3)*4.3;
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative*(1. - 0.7*4.3), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(Atanperator_FOV_Reverse)
{
  double a = 9.8, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = atan(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (1. + a*a);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 1. - 0.7*4.3;

  fov_reverse(1, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative*(1. - 0.7*4.3), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(Log10Operator_FOV_Forward)
{
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
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 12.3;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. + i*9.9;
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. + 9.9), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(Log10perator_FOV_Reverse)
{
  double a = 12.3, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = log10(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (a*std::log(10));

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 1. + 9.9;

  fov_reverse(1, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative*(1. + 9.9), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

#if defined(ATRIG_ERF)
BOOST_AUTO_TEST_CASE(AsinhOperator_FOV_Forward)
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
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 0.6;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i*6.2;
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. - 6.2), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(Asinhperator_FOV_Reverse)
{
  double a = 0.6, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = asinh(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (std::sqrt(a*a + 1.));

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 1. - 6.1;

  fov_reverse(1, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative*(1. - 6.1), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(AcoshOperator_FOV_Forward)
{
  double a = 1.7, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = acosh(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (std::sqrt(a*a - 1.));
  a = std::acosh(a);

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 1.7;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. + i*3.1;
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. + 3.1), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(Acoshperator_FOV_Reverse)
{
  double a = 1.6, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = acosh(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (std::sqrt(a*a - 1.));

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 1. + 3.1;

  fov_reverse(1, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative*(1. + 3.1), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(AtanhOperator_FOV_Forward)
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
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 0.6;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. + i*2.2;
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. + 2.2), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(Atanhperator_FOV_Reverse)
{
  double a = 0.6, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = atanh(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (1. - a*a);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 1. + 2.2;

  fov_reverse(1, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative*(1. + 2.2), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}
#endif

BOOST_AUTO_TEST_CASE(InclOperator_FOV_Forward)
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
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 5.;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i*4.2;
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. - 4.2), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(Inclperator_FOV_Reverse)
{
  double a = 5., aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = ++ad;

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = std::sqrt(5);

  fov_reverse(1, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative*std::sqrt(5), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(DeclOperator_FOV_Forward)
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
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 5.;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i*4.2;
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. - 4.2), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(Declperator_FOV_Reverse)
{
  double a = 5., aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = --ad;

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = std::sqrt(5);

  fov_reverse(1, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative*std::sqrt(5), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(SignPlusOperator_FOV_Forward)
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
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 1.5;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. + i*0.8;
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. + 0.8), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(SignPlusOperator_FOV_Reverse)
{
  double a = 1.5, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = +ad;

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = std::sqrt(3);

  fov_reverse(1, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative*std::sqrt(3), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(SignMinusOperator_FOV_Forward)
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
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 1.5;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. + i*0.8;
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. + 0.8), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(SignMinusOperator_FOV_Reverse)
{
  double a = 1.5, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = -ad;

  ad >>= aout;
  trace_off();

  double aDerivative = -1.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = std::sqrt(3);

  fov_reverse(1, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative*std::sqrt(3), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(Atan2Operator_FOV_Forward)
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
  double **xd = myalloc2(2, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  /* Test partial derivative wrt a and b. */
  x[0] = 12.3;
  x[1] = 2.1;

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

BOOST_AUTO_TEST_CASE(Atan2Operator_FOV_Reverse)
{
  double a = 12.3, b = 2.1, aout;
  adouble ad, bd;

  trace_on(1, 1);
  ad <<= a;
  bd <<= b;

  ad = atan2(ad, bd);

  ad >>= aout;
  trace_off();

  double aDerivative = b / (a*a + b*b);
  double bDerivative = -a / (a*a + b*b);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 2);

  u[0][0] = 1.;
  u[1][0] = std::exp(1.);

  fov_reverse(1, 1, 2, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative*std::exp(1.), tt::tolerance(tol));
  BOOST_TEST(z[1][1] == bDerivative*std::exp(1.), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(PowOperator_FOV_Forward_1)
{
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
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 2.3;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. + i*0.5;
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. + 0.5), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}
/*
BOOST_AUTO_TEST_CASE(PowOperator_FOV_Reverse_1)
{
  double a = 2.3, e = 3.5, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = pow(ad, e);

  ad >>= aout;
  trace_off();

  double aDerivative = e * std::pow(a, e - 1.);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = -1.1;

  fov_reverse(1, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == -1.1*aDerivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}
*/
BOOST_AUTO_TEST_CASE(PowOperator_FOV_Forward_2)
{
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
  double **xd = myalloc2(2, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  /* Test partial derivative wrt a and b. */
  x[0] = 2.3;
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

BOOST_AUTO_TEST_CASE(PowOperator_FOV_Reverse_2)
{
  double a = 2.3, b = 3.5, aout;
  adouble ad, bd;

  trace_on(1, 1);
  ad <<= a;
  bd <<= b;

  ad = pow(ad, bd);

  ad >>= aout;
  trace_off();

  double aDerivative = b * std::pow(a, b - 1.);
  double bDerivative = std::pow(a, b)*std::log(a);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 2);

  u[0][0] = 1.;
  u[1][0] = 2.;

  fov_reverse(1, 1, 2, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == 2.*aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == 2.*bDerivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(PowOperator_FOV_Forward_3)
{
  double a = 2.3, e = 3.5, eout;
  adouble ed;

  trace_on(1);
  ed <<= e;

  ed = pow(a, ed);

  ed >>= eout;
  trace_off();

  double eDerivative = std::log(a) * std::pow(a, e);
  a = std::pow(a, e);

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 3.5;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. + i*0.5;
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == eDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == eDerivative * (1. + 0.5), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(PowOperator_FOV_Reverse_3)
{
  double a = 2.3, e = 3.4, eout;
  adouble ed;

  trace_on(1, 1);
  ed <<= e;

  ed = pow(a, ed);

  ed >>= eout;
  trace_off();

  double eDerivative = std::pow(a, e)*std::log(a);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = -1.1;

  fov_reverse(1, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == eDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == -1.1*eDerivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

/* Frexp operator is not differentiable and does not have to be tested. */

BOOST_AUTO_TEST_CASE(LdexpOperator_FOV_Forward_1)
{
  double a = 4., b = 3., out;
  adouble ad, bd;

  trace_on(1);
  ad <<= a;
  bd <<= b;

  ad = ad * pow(2., bd);

  ad >>= out;
  trace_off();

  double aDerivative = std::pow(2., b);
  double bDerivative = a * std::log(2.) * std::pow(2., b);
  a = a * std::pow(2., b);

  double *x = myalloc1(2);
  double **xd = myalloc2(2, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  /* Test partial derivative wrt a and b. */
  x[0] = 4.;
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

BOOST_AUTO_TEST_CASE(LdexpOperator_FOV_Reverse_1)
{
  double a = 4., b = 3., aout;
  adouble ad, bd;

  trace_on(1, 1);
  ad <<= a;
  bd <<= b;

  ad = ad * pow(2., bd);

  ad >>= aout;
  trace_off();

  double aDerivative = std::pow(2., b);
  double bDerivative = a * std::pow(2., b)*std::log(2.);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 2);

  u[0][0] = 1.;
  u[1][0] = -2.;

  fov_reverse(1, 1, 2, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == -2.*aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == -2.*bDerivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(LdexpOperator_FOV_Forward_2)
{
  double a = 4., e = 3., aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = ldexp(ad, e);

  ad >>= aout;
  trace_off();

  double aDerivative = std::pow(2., e);
  a = std::ldexp(a, e);

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 4.;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. + i;
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

BOOST_AUTO_TEST_CASE(LdexpOperator_FOV_Reverse_2)
{
  double a = 4., e = 3., aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = ldexp(ad, e);

  ad >>= aout;
  trace_off();

  double aDerivative = std::pow(2., e);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = std::exp(std::log(10.));

  fov_reverse(1, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(
    z[1][0] == std::exp(std::log(10.))*aDerivative, tt::tolerance(tol)
  );

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(LdexpOperator_FOV_Forward_3)
{
  double a = 4., e = 3., eout;
  adouble ed;

  trace_on(1);
  ed <<= e;

  ed = a * pow(2., ed);

  ed >>= eout;
  trace_off();

  double eDerivative = a * std::log(2.) * std::pow(2., e);
  e = std::ldexp(a, e);

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 3.;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. + i;
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == e, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == eDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == eDerivative * 2., tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(LdexpOperator_FOV_Reverse_3)
{
  double a = 4., e = 3., eout;
  adouble ed;

  trace_on(1, 1);
  ed <<= e;

  ed = a * pow(2., ed);

  ed >>= eout;
  trace_off();

  double eDerivative = a * std::pow(2., e) * std::log(2.);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 2.2;

  fov_reverse(1, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == eDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == 2.2*eDerivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(FabsOperator_FOV_Forward)
{
  double a = 1.4, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = fabs(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;
  a = std::fabs(a);

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 1.4;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i*1.5;
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. - 1.5), tt::tolerance(tol));

  x[0] = -5.;

  a = std::fabs(-5.);
  aDerivative = -1.;

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. - 1.5), tt::tolerance(tol));

  x[0] = 0.;

  xd[0][0] = 2.5;
  xd[0][1] = -3.5;

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == 0., tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == 2.5, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == 3.5, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(FabsOperator_FOV_Reverse)
{
  double a = 1.4, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = fabs(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 3.3;

  fov_reverse(1, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == 3.3*aDerivative, tt::tolerance(tol));

  a = -5.;

  trace_on(1, 1);
  ad <<= a;

  ad = fabs(ad);

  ad >>= aout;
  trace_off();

  aDerivative = -1.;

  u[0][0] = 1.;
  u[1][0] = 3.3;

  fov_reverse(1, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == 3.3*aDerivative, tt::tolerance(tol));

  a = 0.;

  trace_on(1, 1);
  ad <<= a;

  ad = fabs(ad);

  ad >>= aout;
  trace_off();

  u[0][0] = 2.5;
  u[1][0] = -3.5;

  fov_reverse(1, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == 0., tt::tolerance(tol));
  BOOST_TEST(z[1][0] == 0., tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(CeilOperator_FOV_Forward)
{
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
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 3.573;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. + i*5.8;
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * 6.8, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(CeilOperator_FOV_Reverse)
{
  double a = 3.573, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = ceil(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 0.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 6.8;

  fov_reverse(1, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * 6.8, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(FloorOperator_FOV_Forward)
{
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
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 4.483;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i*5.8;
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (-4.8), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(FloorOperator_FOV_Reverse)
{
  double a = 4.483, aout;
  adouble ad;

  trace_on(1, 1);
  ad <<= a;

  ad = floor(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 0.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 6.8;

  fov_reverse(1, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * 6.8, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(FmaxOperator_FOV_Forward_1)
{
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
  double **xd = myalloc2(2, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  /* Test partial derivative wrt a and b. */
  x[0] = 4.;
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

  x[0] = 2.5;
  x[1] = 2.5;

  xd[0][0] = 1.3;
  xd[0][1] = 3.7;
  xd[1][0] = -1.3;
  xd[1][0] = -3.7;

  a = std::fmax(2.5, 2.5);
  aDerivative = 1.3;
  bDerivative = 1.3;

  fov_forward(1, 1, 2, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(FmaxOperator_FOV_Forward_2)
{
  double a = 4., b = 3.2, bout;
  adouble bd;

  trace_on(1);
  bd <<= b;

  bd = fmax(a, bd);

  bd >>= bout;
  trace_off();

  double bDerivative = 0.;
  b = std::fmax(a, b);

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 3.2;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i*2.1;
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == b, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative * (1. - 2.1), tt::tolerance(tol));

  x[0] = 5.2;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i*2.1;
  }

  b = std::fmax(a, x[0]);
  bDerivative = 1.;

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == b, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative * (1. - 2.1), tt::tolerance(tol));

  x[0] = 4.5;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i*2.1;
  }

  b = std::fmax(a, x[0]);
  bDerivative = 1.;

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == b, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative * (1. - 2.1), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(FmaxOperator_FOV_Forward_3)
{
  double a = 4., b = 3.2, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = fmax(ad, b);

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;
  a = std::fmax(a, b);

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 4.;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i*2.1;
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. - 2.1), tt::tolerance(tol));

  x[0] = 2.3;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i*2.1;
  }

  a = std::fmax(x[0], b);
  aDerivative = 0.;

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. - 2.1), tt::tolerance(tol));

  x[0] = 3.2;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i*2.1;
  }

  a = std::fmax(x[0], b);
  aDerivative = 1.;

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(FminOperator_FOV_Forward_1)
{
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
  double **xd = myalloc2(2, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  /* Test partial derivative wrt a and b. */
  x[0] = 4.;
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

  x[0] = 2.5;
  x[1] = 2.5;

  xd[0][0] = 1.3;
  xd[0][1] = 3.7;
  xd[1][0] = -1.3;
  xd[1][0] = -3.7;

  a = std::fmax(2.5, 2.5);
  aDerivative = -3.7;
  bDerivative = -3.7;

  fov_forward(1, 1, 2, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(FminOperator_FOV_Forward_2)
{
  double a = 4., b = 3.2, bout;
  adouble bd;

  trace_on(1);
  bd <<= b;

  bd = fmin(a, bd);

  bd >>= bout;
  trace_off();

  double bDerivative = 1.;
  b = std::fmin(a, b);

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 3.2;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i*2.1;
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == b, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative * (1. - 2.1), tt::tolerance(tol));

  x[0] = 5.2;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i*2.1;
  }

  b = std::fmin(a, x[0]);
  bDerivative = 0.;

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == b, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative * (1. - 2.1), tt::tolerance(tol));

  x[0] = 4.5;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i*2.1;
  }

  b = std::fmin(a, x[0]);
  bDerivative = 0.;

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == b, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative * (1. - 2.1), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(FminOperator_FOV_Forward_3)
{
  double a = 4., b = 3.2, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad = fmin(ad, b);

  ad >>= aout;
  trace_off();

  double aDerivative = 0.;
  a = std::fmin(a, b);

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 4.;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i*2.1;
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. - 2.1), tt::tolerance(tol));

  x[0] = 2.3;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i*2.1;
  }

  a = std::fmin(x[0], b);
  aDerivative = 1.;

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. - 2.1), tt::tolerance(tol));

  x[0] = 3.2;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i*2.1;
  }

  a = std::fmin(x[0], b);
  aDerivative = 0.;

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

#if defined(ATRIG_ERF)
BOOST_AUTO_TEST_CASE(ErfOperator_FOV_Forward)
{
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
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 7.1;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = std::pow(3., i*2.);
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * std::pow(3., 2.), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}
#endif

BOOST_AUTO_TEST_CASE(EqPlusOperator_FOV_Forward)
{
  double a = 5.132, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad += 5.2;

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;
  a += 5.2;

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 5.132;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = std::pow(4., i*1.5);
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * std::pow(4., 1.5), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(EqMinusOperator_FOV_Forward)
{
  double a = 5.132, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad -= 5.2;

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;
  a -= 5.2;

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 5.132;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = std::pow(4., i*1.5);
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * std::pow(4., 1.5), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(EqTimesOperator_FOV_Forward)
{
  double a = 5.132, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad *= 5.2;

  ad >>= aout;
  trace_off();

  double aDerivative = 5.2;
  a *= 5.2;

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 5.132;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = std::pow(4., i*1.5);
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * std::pow(4., 1.5), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(EqDivOperator_FOV_Forward)
{
  double a = 5.132, aout;
  adouble ad;

  trace_on(1);
  ad <<= a;

  ad /= 5.2;

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / 5.2;
  a /= 5.2;

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 5.132;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = std::pow(4., i*1.5);
  }

  fov_forward(1, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * std::pow(4., 1.5), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(CondassignOperator_FOV_Forward)
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

  double *x = myalloc1(3);
  double **xd = myalloc2(3, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 1.;
  x[1] = 3.5;
  x[2] = 5.3;
  xd[0][0] = 0.;
  xd[0][1] = 0.;
  xd[1][0] = 0.1;
  xd[1][1] = 1.;
  xd[2][0] = 0.2;
  xd[2][1] = 2.;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. + i;
  }

  fov_forward(1, 1, 3, 2, x, xd, y, yd);

  BOOST_TEST(*y == 3.5, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == 0.1, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == 1., tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(CondeqassignOperator_FOV_Forward)
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

  double *x = myalloc1(3);
  double **xd = myalloc2(3, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 1.;
  x[1] = 3.5;
  x[2] = 5.3;
  xd[0][0] = 0.;
  xd[0][1] = 0.;
  xd[1][0] = 0.1;
  xd[1][1] = 1.;
  xd[2][0] = 0.2;
  xd[2][1] = 2.;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. + i;
  }

  fov_forward(1, 1, 3, 2, x, xd, y, yd);

  BOOST_TEST(*y == 3.5, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == 0.1, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == 1., tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

/* This concludes the tests for elementary operations. The next tests
 * involve more complicated compositions of elementary functions with
 * higher numbers of independent (and possibly dependent) variables.
 *
 * Before every test, a short comment explains the structure of the
 * tested composite function and states the expected analytic
 * derivative.
 */

/* Tested function: sin(x1)*sin(x1) + cos(x1)*cos(x1) + x2
 * Gradient vector: (
 *                    0.0,
 *                    1.0
 *                  )
 */
BOOST_AUTO_TEST_CASE(CompositeTrig1_FOV_Forward)
{
  double x1 = 0.289, x2 = 1.927, out;
  adouble ax1, ax2;

  trace_on(1);
  ax1 <<= x1;
  ax2 <<= x2;

  ax1 = sin(ax1)*sin(ax1) + cos(ax1)*cos(ax1) + ax2;

  ax1 >>= out;
  trace_off();

  double x1Derivative = 0.;
  double x2Derivative = 1.;
  x1 = std::sin(x1)*std::sin(x1) + std::cos(x1)*std::cos(x1) + x2;

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

  fov_forward(1, 1, 2, 2, x, xd, y, yd);

  BOOST_TEST(*y == x1, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == x2Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

/* Tested function: 2*sin(cos(x1))*exp(x2) - pow(cos(x3), 2)*sin(x2)
 * Gradient vector: (
 *                    -2*cos(cos(x1))*exp(x2)*sin(x1),
 *                    2*sin(cos(x1))*exp(x2) - pow(cos(x3), 2)*cos(x2),
 *                    2*cos(x3)*sin(x3)*sin(x2)
 *                  )
 */
BOOST_AUTO_TEST_CASE(CompositeTrig2_FOV_Forward)
{
  double x1 = 1.11, x2 = 2.22, x3 = 3.33, out;
  adouble ax1, ax2, ax3;

  trace_on(1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ax1 = 2*sin(cos(ax1))*exp(ax2) - pow(cos(ax3), 2)*sin(ax2);

  ax1 >>= out;
  trace_off();

  double x1Derivative = -2*std::cos(std::cos(x1))*std::exp(x2)*std::sin(x1);
  double x2Derivative = 2*std::sin(std::cos(x1))*std::exp(x2)
                        - std::pow(std::cos(x3), 2)*std::cos(x2);
  double x3Derivative = 2*std::cos(x3)*std::sin(x3)*std::sin(x2);
  x1 = 2*std::sin(std::cos(x1))*std::exp(x2)
       - std::pow(std::cos(x3), 2)*std::sin(x2);

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

  fov_forward(1, 1, 3, 3, x, xd, y, yd);

  BOOST_TEST(*y == x1, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][2] == x3Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

/* Tested function: pow(sin(x1), cos(x1) - x2)*x3
 * Gradient vector: (
 *                    pow(sin(x1), cos(x1) - x2)*x3*(-sin(x1)*log(sin(x1))
 *                    + (cos(x1) - x2)*cos(x1)/sin(x1)),
 *                    -log(sin(x1))*pow(sin(x1), cos(x1) - x2)*x3,
 *                    pow(sin(x1), cos(x1) - x2)
 *                  )
 */
BOOST_AUTO_TEST_CASE(CompositeTrig3_FOV_Forward)
{
  double x1 = 0.516, x2 = 9.89, x3 = 0.072, out;
  adouble ax1, ax2, ax3;

  trace_on(1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ax1 = pow(sin(ax1), cos(ax1) - ax2)*ax3;

  ax1 >>= out;
  trace_off();

  double x1Derivative = std::pow(std::sin(x1), std::cos(x1) - x2) * x3
                        * (-std::sin(x1)*std::log(std::sin(x1))
                        + (std::cos(x1) - x2)*std::cos(x1)/std::sin(x1));
  double x2Derivative = -std::log(std::sin(x1))
                        * std::pow(std::sin(x1), std::cos(x1) - x2) * x3;
  double x3Derivative = std::pow(std::sin(x1), std::cos(x1) - x2);
  x1 = std::pow(std::sin(x1), std::cos(x1) - x2)*x3;

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

  fov_forward(1, 1, 3, 3, x, xd, y, yd);

  BOOST_TEST(*y == x1, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][2] == x3Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

/* Tested function: atan(tan(x1))*exp(x2)
 * Gradient vector: (
 *                    exp(x2)
 *                    x1*exp(x2)
 *                  )
 */
BOOST_AUTO_TEST_CASE(CompositeTrig4_FOV_Forward)
{
  double x1 = 1.56, x2 = 8.99, out;
  adouble ax1, ax2;

  trace_on(1);
  ax1 <<= x1;
  ax2 <<= x2;

  ax1 = atan(tan(ax1))*exp(ax2);

  ax1 >>= out;
  trace_off();

  double x1Derivative = std::exp(x2);
  double x2Derivative = x1*std::exp(x2);
  x1 = x1*exp(x2);

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

  fov_forward(1, 1, 2, 2, x, xd, y, yd);

  BOOST_TEST(*y == x1, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == x2Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
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
BOOST_AUTO_TEST_CASE(LongSum_FOV_Forward)
{
  double x1 = 0.11, x2 = -2.27, x3 = 81.7, x4 = 0.444, x5 = 4.444, out;
  adouble ax1, ax2, ax3, ax4, ax5;

  trace_on(1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;
  ax4 <<= x4;
  ax5 <<= x5;

  ax1 = ax1 + ax2 - ax3 + pow(ax1, 2) - 10 + sqrt(ax4*ax5);

  ax1 >>= out;
  trace_off();

  double x1Derivative = 1. + 2*x1;
  double x2Derivative = 1.;
  double x3Derivative = -1.;
  double x4Derivative = 0.5 * std::sqrt(x5/x4);
  double x5Derivative = 0.5 * std::sqrt(x4/x5);
  x1 = x1 + x2 - x3 + std::pow(x1, 2) - 10 + std::sqrt(x4*x5);

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

  fov_forward(1, 1, 5, 5, x, xd, y, yd);

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

/* Tested function: sqrt(pow(x1, 2))*x2
 * Gradient vector: (
 *                    x2,
 *                    x1
 *                  )
 */
BOOST_AUTO_TEST_CASE(InverseFunc_FOV_Forward)
{
  double x1 = 3.77, x2 = -21.12, out;
  adouble ax1, ax2;

  trace_on(1);
  ax1 <<= x1;
  ax2 <<= x2;

  ax1 = sqrt(pow(ax1, 2))*ax2;

  ax1 >>= out;
  trace_off();

  double x1Derivative = x2;
  double x2Derivative = x1;
  x1 = std::sqrt(std::pow(x1, 2))*x2;

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

  fov_forward(1, 1, 2, 2, x, xd, y, yd);

  BOOST_TEST(*y == x1, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == x2Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
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
BOOST_AUTO_TEST_CASE(ExpPow_FOV_Forward)
{
  double x1 = 0.642, x2 = 6.42, x3 = 0.528, out;
  adouble ax1, ax2, ax3;

  trace_on(1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ax1 = exp(ax1 + exp(ax2 + ax3))*pow(ax1 + ax2, ax3);

  ax1 >>= out;
  trace_off();

  double x1Derivative = std::exp(x1 + std::exp(x2 + x3))*std::pow(x1 + x2, x3)
                        + std::exp(x1 + std::exp(x2 + x3))
                        * x3 * std::pow(x1 + x2, x3 - 1);
  double x2Derivative = std::exp(x1 + std::exp(x2 + x3))*std::exp(x2 + x3)
                        * std::pow(x1 + x2, x3)
                        + std::exp(x1 + std::exp(x2 + x3))
                        * x3 * std::pow(x1 + x2, x3 - 1);
  double x3Derivative = std::exp(x1 + std::exp(x2 + x3))*std::exp(x2 + x3)
                        * std::pow(x1 + x2, x3)
                        + std::exp(x1 + std::exp(x2 + x3))
                        * std::pow(x1 + x2, x3)*std::log(x1 + x2);
  x1 = std::exp(x1 + std::exp(x2 + x3))*std::pow(x1 + x2, x3);

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

  fov_forward(1, 1, 3, 3, x, xd, y, yd);

  BOOST_TEST(*y == x1, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][2] == x3Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

/* Tested function: sqrt(sqrt(x1*x2 + 2*x3))*x4
 * Gradient vector: (
 *                    0.25*pow(x1*x2 + 2*x3, -0.75)*x2*x4,
 *                    0.25*pow(x1*x2 + 2*x3, -0.75)*x1*x4,
 *                    0.25*pow(x1*x2 + 2*x3, -0.75)*2*x4,
 *                    pow(x1*x2 + 2*x3, 0.25)
 *                  )
 */
BOOST_AUTO_TEST_CASE(CompositeSqrt_FOV_Forward)
{
  double x1 = -2.14, x2 = -2.22, x3 = 50.05, x4 = 0.104, out;
  adouble ax1, ax2, ax3, ax4;

  trace_on(1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;
  ax4 <<= x4;

  ax1 = sqrt(sqrt(ax1*ax2 + 2*ax3))*ax4;

  ax1 >>= out;
  trace_off();

  double x1Derivative = 0.25*std::pow(x1*x2 + 2*x3, -0.75)*x2*x4;
  double x2Derivative = 0.25*std::pow(x1*x2 + 2*x3, -0.75)*x1*x4;
  double x3Derivative = 0.25*std::pow(x1*x2 + 2*x3, -0.75)*2.0*x4;
  double x4Derivative = std::pow(x1*x2 + 2*x3, 0.25);
  x1 = std::sqrt(std::sqrt(x1*x2 + 2*x3))*x4;

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

  fov_forward(1, 1, 4, 4, x, xd, y, yd);

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
BOOST_AUTO_TEST_CASE(CompositeHyperb_FOV_Forward)
{
  double x1 = 0.0, x2 = 5.099, x3 = 5.5, x4 = 4.73, out;
  adouble ax1, ax2, ax3, ax4;

  trace_on(1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;
  ax4 <<= x4;

  ax1 = tanh(acos(pow(ax1, 2) + 1.)*sin(ax2))*ax3 + exp(cosh(ax4));

  ax1 >>= out;
  trace_off();

  double x1Derivative = (1 - std::pow(std::tanh(std::acos(std::pow(x1, 2) + 1.)
                        * std::sin(x2)), 2)) * x3 * std::sin(x2) * 2.
                        * x1 / std::sqrt(1. - std::pow(x1, 4));
  double x2Derivative = (1 - std::pow(std::tanh(std::acos(std::pow(x1, 2) + 1)
                        * std::sin(x2)), 2)) * x3 * std::acos(std::pow(x1, 2)
                        + 1) * std::cos(x2);
  double x3Derivative = std::tanh(std::acos(std::pow(x1, 2) + 1)*std::sin(x2));
  double x4Derivative = std::exp(std::cosh(x4))*std::sinh(x4);
  x1 = std::tanh(std::acos(std::pow(x1, 2) + 1.)*std::sin(x2))*x3
       + std::exp(std::cosh(x4));

  double *x = myalloc1(4);
  double **xd = myalloc2(4, 4);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 4);

  /* Test partial derivative wrt x1, x2, x3 and x4. */
  x[0] = 0.0;
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

  fov_forward(1, 1, 4, 4, x, xd, y, yd);

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

/* Tested function: fmax(x1*pow(x3, 2), x2*pow(x3, 2))*exp(x3)
 * Gradient vector: (
 *                    pow(x3, 2)*exp(x3),
 *                    0.0,
 *                    2.0*x1*x3*exp(x3) + x1*pow(x3, 2)*exp(x3)
 *                  )
 */
BOOST_AUTO_TEST_CASE(CompositeFmax_FOV_Forward)
{
  double x1 = 2.31, x2 = 1.32, x3 = 3.21, out;
  adouble ax1, ax2, ax3;

  trace_on(1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ax1 = fmax(ax1*pow(ax3, 2), ax2*pow(ax3, 2))*exp(ax3);

  ax1 >>= out;
  trace_off();

  double x1Derivative = std::pow(x3, 2)*std::exp(x3);
  double x2Derivative = 0.0;
  double x3Derivative = 2.0*x1*x3*std::exp(x3)
                        + x1*std::pow(x3, 2)*std::exp(x3);
  x1 = std::fmax(x1*std::pow(x3, 2), x2*std::pow(x3, 2))*std::exp(x3);

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

  fov_forward(1, 1, 3, 3, x, xd, y, yd);

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
BOOST_AUTO_TEST_CASE(CompositeFmin_FOV_Forward)
{
  double x1 = 2.31, x2 = 1.32, x3 = 3.21, out;
  adouble ax1, ax2, ax3;

  trace_on(1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;

  ax1 = fmin(ax1*pow(ax3, 2), ax2*pow(ax3, 2))*exp(ax3);

  ax1 >>= out;
  trace_off();

  double x1Derivative = 0.0;
  double x2Derivative = std::pow(x3, 2)*std::exp(x3);
  double x3Derivative = 2.0*x2*x3*std::exp(x3)
                        + x2*std::pow(x3, 2)*std::exp(x3);
  x1 = std::fmin(x1*std::pow(x3, 2), x2*std::pow(x3, 2))*std::exp(x3);

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

  fov_forward(1, 1, 3, 3, x, xd, y, yd);

  BOOST_TEST(*y == x1, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == x2Derivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][2] == x3Derivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

#if defined(ATRIG_ERF)
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
BOOST_AUTO_TEST_CASE(CompositeErfFabs_FOV_Forward)
{
  double x1 = 4.56, x2 = 5.46, x3 = 4.65, x4 = 6.54, x5 = 6.45, out;
  adouble ax1, ax2, ax3, ax4, ax5;

  trace_on(1);
  ax1 <<= x1;
  ax2 <<= x2;
  ax3 <<= x3;
  ax4 <<= x4;
  ax5 <<= x5;

  ax1 = erf(fabs(ax1 - ax2)*sinh(ax3 - ax4))*sin(ax5);

  ax1 >>= out;
  trace_off();

  double x1Derivative = -2./std::sqrt(std::acos(-1.))
                        * std::exp(-std::pow(std::fabs(x1 - x2)
                        * std::sinh(x3 - x4), 2)) * std::sin(x5)
                        * std::sinh(x3 - x4);
  double x2Derivative = 2./std::sqrt(std::acos(-1.))
                        * std::exp(-std::pow(std::fabs(x1 - x2)
                        * std::sinh(x3 - x4), 2)) * std::sin(x5)
                        * std::sinh(x3 - x4);
  double x3Derivative = 2./std::sqrt(std::acos(-1.))
                        * std::exp(-std::pow(std::fabs(x1 - x2)
                        * std::sinh(x3 - x4), 2)) * std::sin(x5)
                        * std::fabs(x1 - x2) * std::cosh(x3 - x4);
  double x4Derivative = -2./std::sqrt(std::acos(-1.))
                        * std::exp(-std::pow(std::fabs(x1 - x2)
                        * std::sinh(x3 - x4), 2)) * std::sin(x5)
                        * std::fabs(x1 - x2) * std::cosh(x3 - x4);
  double x5Derivative = std::erf(std::fabs(x1 - x2)*std::sinh(x3 - x4))
                        * std::cos(x5);
  x1 = std::erf(std::fabs(x1 - x2)*std::sinh(x3 - x4))*std::sin(x5);

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

  fov_forward(1, 1, 5, 5, x, xd, y, yd);

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
#endif


/* The test PowOperator_FOV_Reverse_1 is not working, but gives an error
 * 'memory access violation: no mapping at fault adress'.  What is the
 * problem here?
 */


BOOST_AUTO_TEST_SUITE_END()


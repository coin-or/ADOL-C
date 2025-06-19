#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

#include <adolc/adolc.h>

#include "const.h"

BOOST_AUTO_TEST_SUITE(trace_vector)

/***********************************/
/* Tests for trace vector mode     */
/* Author: Philipp Schuette        */
/***********************************/

/* Naming convention for test cases:  Operatorname_Operator_FOV_Forward for
 * forward derivative evaluation in vector mode.
 *
 * For documentation of concrete test implementation, check traceless scalar
 * mode test implementation.  The testing order is consistent with that file
 * as well.
 */

const short tapeId = 2;
struct TapeInitializer {
  TapeInitializer() { createNewTape(tapeId); }
};

BOOST_GLOBAL_FIXTURE(TapeInitializer);

BOOST_AUTO_TEST_CASE(ExpOperator_FOV_Forward) {
  double a = 2., aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
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

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * 2., tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(ExpOperator_FOV_Reverse) {
  double a = 2., aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = exp(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = std::exp(a);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = std::exp(3.);

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * std::exp(3.), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(MultOperator_FOV_Forward) {
  double a = 2., b = 3.5, out;

  setCurrentTape(tapeId);

  adouble ad;
  adouble bd;

  trace_on(tapeId);
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

  fov_forward(tapeId, 1, 2, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(MultOperator_FOV_Reverse) {
  double a = 2., b = 3.5, aout;

  setCurrentTape(tapeId);

  adouble ad;
  adouble bd;

  trace_on(tapeId, 1);
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

  fov_reverse(tapeId, 1, 2, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == 2. * aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == 2. * bDerivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(AddOperator_FOV_Forward) {
  double a = 2.5, b = 3., out;

  setCurrentTape(tapeId);

  adouble ad;
  adouble bd;

  trace_on(tapeId);
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

  fov_forward(tapeId, 1, 2, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(AddOperator_FOV_Reverse) {
  double a = 2.5, b = 3., aout;

  setCurrentTape(tapeId);

  adouble ad;
  adouble bd;

  trace_on(tapeId, 1);
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

  fov_reverse(tapeId, 1, 2, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == 9. * aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == 9. * bDerivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(SubOperator_FOV_Forward) {
  double a = 1.5, b = 3.2, out;

  setCurrentTape(tapeId);

  adouble ad;
  adouble bd;

  trace_on(tapeId);
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

  fov_forward(tapeId, 1, 2, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(SubOperator_FOV_Reverse) {
  double a = 1.5, b = 3.2, aout;

  setCurrentTape(tapeId);

  adouble ad;
  adouble bd;

  trace_on(tapeId, 1);
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

  fov_reverse(tapeId, 1, 2, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == std::sqrt(2) * aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == std::sqrt(2) * bDerivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(DivOperator_FOV_Forward) {
  double a = 0.5, b = 4.5, out;

  setCurrentTape(tapeId);

  adouble ad;
  adouble bd;

  trace_on(tapeId);
  ad <<= a;
  bd <<= b;

  ad = ad / bd;

  ad >>= out;
  trace_off();

  double aDerivative = 1. / b;
  double bDerivative = -a / (b * b);
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

  fov_forward(tapeId, 1, 2, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(DivOperator_FOV_Reverse) {
  double a = 0.5, b = 4.5, aout;

  setCurrentTape(tapeId);

  adouble ad;
  adouble bd;

  trace_on(tapeId, 1);
  ad <<= a;
  bd <<= b;

  ad = ad / bd;

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / b;
  double bDerivative = -a / (b * b);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 2);

  u[0][0] = 1.;
  u[1][0] = 0.9;

  fov_reverse(tapeId, 1, 2, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == 0.9 * aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == 0.9 * bDerivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(TanOperator_FOV_Forward) {
  double a = 0.7, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
  ad <<= a;

  ad = tan(ad);

  ad >>= aout;
  trace_off();

  a = tan(a);
  double aDerivative = 1. + a * a;

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 0.7;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. + std::pow(2, i);
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative * (1. + std::pow(2, 0)),
             tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. + std::pow(2, 1)),
             tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(TanOperator_FOV_Reverse) {
  double a = 0.7, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = tan(ad);

  ad >>= aout;
  trace_off();

  a = std::tan(a);
  double aDerivative = 1. + a * a;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 1.1;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * 1.1, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(SinOperator_FOV_Forward) {
  double a = 1.2, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
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

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (-1.), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(SinOperator_FOV_Reverse) {
  double a = 1.2, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = sin(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = std::cos(a);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = std::tan(4.4);

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * std::tan(4.4), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(CosOperator_FOV_Forward) {
  double a = 1.2, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
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

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * 3., tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(CosOperator_FOV_Reverse) {
  double a = 1.2, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = cos(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = -std::sin(a);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = std::log(2.);

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * std::log(2.), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(SqrtOperator_FOV_Forward) {
  double a = 2.2, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
  ad <<= a;

  ad = sqrt(ad);

  ad >>= aout;
  trace_off();

  a = std::sqrt(a);
  double aDerivative = 1. / (2. * a);

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 2.2;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. * std::pow(2, i);
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * 2., tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(SqrtOperator_FOV_Reverse) {
  double a = 2.2, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = sqrt(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (2. * std::sqrt(a));

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = std::exp(2.);

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * std::exp(2.), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(CbrtOperator_FOV_Forward) {
  double a = 2.2, aout;

  setCurrentTape(tapeId);

  adouble ad;

  // cbrt(a)
  trace_on(tapeId);
  ad <<= a;

  ad = cbrt(ad);

  ad >>= aout;
  trace_off();

  // cbrt(a)
  const double out = std::cbrt(a);

  // 1 / 3 * a^(-2 / 3)
  const double aDerivative = 1.0 / (3.0 * std::pow(a, 2.0 / 3.0));

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 2.2;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. * std::pow(2, i);
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == out, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * 2., tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(CbrtOperator_FOV_Reverse) {
  double a = 2.2, aout;

  setCurrentTape(tapeId);

  adouble ad;

  // cbrt(a)
  trace_on(tapeId, 1);
  ad <<= a;

  ad = cbrt(ad);

  ad >>= aout;
  trace_off();

  // 1 / 3 * a^(-2 / 3)
  const double aDerivative = 1. / (3.0 * std::pow(a, 2.0 / 3.0));

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = std::exp(2.);

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * std::exp(2.), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(LogOperator_FOV_Forward) {
  double a = 4.9, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
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

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * 6.5, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(LogOperator_FOV_Reverse) {
  double a = 4.9, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = log(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / a;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = std::exp(-1.);

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * std::exp(-1.), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(SinhOperator_FOV_Forward) {
  double a = 4., aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
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
    xd[0][i] = 1. - std::sqrt(2. * i);
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. - std::sqrt(2.)),
             tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(SinhOperator_FOV_Reverse) {
  double a = 4., aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = sinh(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = std::cosh(a);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = std::cosh(3.5);

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * std::cosh(3.5), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(CoshOperator_FOV_Forward) {
  double a = 4., aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
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

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * 4.2, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(CoshOperator_FOV_Reverse) {
  double a = 4., aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = cosh(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = std::sinh(a);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = std::sinh(3.5);

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * std::sinh(3.5), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(TanhOperator_FOV_Forward) {
  double a = 4., aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
  ad <<= a;

  ad = tanh(ad);

  ad >>= aout;
  trace_off();

  a = std::tanh(a);
  double aDerivative = 1. - a * a;

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 4.;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - 1.3 * i;
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. - 1.3), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(TanhOperator_FOV_Reverse) {
  double a = 4., aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = tanh(ad);

  ad >>= aout;
  trace_off();

  a = std::tanh(a);
  double aDerivative = 1. - a * a;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 5.4;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == 5.4 * aDerivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(AsinOperator_FOV_Forward) {
  double a = 0.9, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
  ad <<= a;

  ad = asin(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (std::sqrt(1. - a * a));
  a = std::asin(a);

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 0.9;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. + i * (i + 1.7) * 4.3;
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. + 2.7 * 4.3), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(AsinOperator_FOV_Reverse) {
  double a = 0.9, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = asin(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (std::sqrt(1. - a * a));

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 1. + 2.7 * 4.3;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * (1. + 2.7 * 4.3), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(AcosOperator_FOV_Forward) {
  double a = 0.8, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
  ad <<= a;

  ad = acos(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = -1. / (std::sqrt(1. - a * a));
  a = std::acos(a);

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 0.8;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i * (i + 0.7) * 3.4;
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. - 1.7 * 3.4), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(AcosOperator_FOV_Reverse) {
  double a = 0.8, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = acos(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = -1. / (std::sqrt(1. - a * a));

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 1. - 1.7 * 3.4;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * (1. - 1.7 * 3.4), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(AtanOperator_FOV_Forward) {
  double a = 9.8, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
  ad <<= a;

  ad = atan(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (1. + a * a);
  a = std::atan(a);

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 9.8;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i * (i - 0.3) * 4.3;
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. - 0.7 * 4.3), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(Atanperator_FOV_Reverse) {
  double a = 9.8, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = atan(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (1. + a * a);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 1. - 0.7 * 4.3;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * (1. - 0.7 * 4.3), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(Log10Operator_FOV_Forward) {
  double a = 12.3, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
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
    xd[0][i] = 1. + i * 9.9;
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. + 9.9), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(Log10perator_FOV_Reverse) {
  double a = 12.3, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = log10(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (a * std::log(10));

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 1. + 9.9;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * (1. + 9.9), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(AsinhOperator_FOV_Forward) {
  double a = 0.6, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
  ad <<= a;

  ad = asinh(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (std::sqrt(a * a + 1.));
  a = std::asinh(a);

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 0.6;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i * 6.2;
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. - 6.2), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(Asinhperator_FOV_Reverse) {
  double a = 0.6, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = asinh(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (std::sqrt(a * a + 1.));

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 1. - 6.1;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * (1. - 6.1), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(AcoshOperator_FOV_Forward) {
  double a = 1.7, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
  ad <<= a;

  ad = acosh(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (std::sqrt(a * a - 1.));
  a = std::acosh(a);

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 1.7;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. + i * 3.1;
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. + 3.1), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(Acoshperator_FOV_Reverse) {
  double a = 1.6, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = acosh(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (std::sqrt(a * a - 1.));

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 1. + 3.1;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * (1. + 3.1), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(AtanhOperator_FOV_Forward) {
  double a = 0.6, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
  ad <<= a;

  ad = atanh(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (1. - a * a);
  a = std::atanh(a);

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 0.6;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. + i * 2.2;
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. + 2.2), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(Atanhperator_FOV_Reverse) {
  double a = 0.6, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = atanh(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / (1. - a * a);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 1. + 2.2;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * (1. + 2.2), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(InclOperator_FOV_Forward) {
  double a = 5., aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
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
    xd[0][i] = 1. - i * 4.2;
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. - 4.2), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(Inclperator_FOV_Reverse) {
  double a = 5., aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = ++ad;

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = std::sqrt(5);

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * std::sqrt(5), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(DeclOperator_FOV_Forward) {
  double a = 5., aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
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
    xd[0][i] = 1. - i * 4.2;
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. - 4.2), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(Declperator_FOV_Reverse) {
  double a = 5., aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = --ad;

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = std::sqrt(5);

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * std::sqrt(5), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(SignPlusOperator_FOV_Forward) {
  double a = 1.5, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
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
    xd[0][i] = 1. + i * 0.8;
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. + 0.8), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(SignPlusOperator_FOV_Reverse) {
  double a = 1.5, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = +ad;

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = std::sqrt(3);

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * std::sqrt(3), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(SignMinusOperator_FOV_Forward) {
  double a = 1.5, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
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
    xd[0][i] = 1. + i * 0.8;
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. + 0.8), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(SignMinusOperator_FOV_Reverse) {
  double a = 1.5, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = -ad;

  ad >>= aout;
  trace_off();

  double aDerivative = -1.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = std::sqrt(3);

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * std::sqrt(3), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(Atan2Operator_FOV_Forward) {
  double a = 12.3, b = 2.1, out;

  setCurrentTape(tapeId);

  adouble ad;
  adouble bd;

  trace_on(tapeId);
  ad <<= a;
  bd <<= b;

  ad = atan2(ad, bd);

  ad >>= out;
  trace_off();

  double aDerivative = b / (a * a + b * b);
  double bDerivative = -a / (a * a + b * b);
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

  fov_forward(tapeId, 1, 2, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(Atan2Operator_FOV_Reverse) {
  double a = 12.3, b = 2.1, aout;

  setCurrentTape(tapeId);

  adouble ad;
  adouble bd;

  trace_on(tapeId, 1);
  ad <<= a;
  bd <<= b;

  ad = atan2(ad, bd);

  ad >>= aout;
  trace_off();

  double aDerivative = b / (a * a + b * b);
  double bDerivative = -a / (a * a + b * b);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 2);

  u[0][0] = 1.;
  u[1][0] = std::exp(1.);

  fov_reverse(tapeId, 1, 2, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * std::exp(1.), tt::tolerance(tol));
  BOOST_TEST(z[1][1] == bDerivative * std::exp(1.), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(PowOperator_FOV_Forward_1) {
  double a = 2.3, e = 3.5, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
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
    xd[0][i] = 1. + i * 0.5;
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. + 0.5), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(PowOperator_FOV_Reverse_1) {
  double a = 2.3, e = 3.5, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = pow(ad, e);

  ad >>= aout;
  trace_off();

  double aDerivative = e * std::pow(a, e - 1.);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = -1.1;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == -1.1 * aDerivative, tt::tolerance(tol));
  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(PowOperator_FOV_Forward_2) {
  double a = 2.3, b = 3.5, out;

  setCurrentTape(tapeId);

  adouble ad;
  adouble bd;

  trace_on(tapeId);
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

  fov_forward(tapeId, 1, 2, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(PowOperator_FOV_Reverse_2) {
  double a = 2.3, b = 3.5, aout;

  setCurrentTape(tapeId);

  adouble ad;
  adouble bd;

  trace_on(tapeId, 1);
  ad <<= a;
  bd <<= b;

  ad = pow(ad, bd);

  ad >>= aout;
  trace_off();

  double aDerivative = b * std::pow(a, b - 1.);
  double bDerivative = std::pow(a, b) * std::log(a);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 2);

  u[0][0] = 1.;
  u[1][0] = 2.;

  fov_reverse(tapeId, 1, 2, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == 2. * aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == 2. * bDerivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(PowOperator_FOV_Forward_3) {
  double a = 2.3, e = 3.5, eout;

  setCurrentTape(tapeId);

  adouble ed;

  trace_on(tapeId);
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
    xd[0][i] = 1. + i * 0.5;
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == eDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == eDerivative * (1. + 0.5), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(PowOperator_FOV_Reverse_3) {
  double a = 2.3, e = 3.4, eout;

  setCurrentTape(tapeId);

  adouble ed;

  trace_on(tapeId, 1);
  ed <<= e;

  ed = pow(a, ed);

  ed >>= eout;
  trace_off();

  double eDerivative = std::pow(a, e) * std::log(a);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = -1.1;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == eDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == -1.1 * eDerivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

/* Frexp operator is not differentiable and does not have to be tested. */

BOOST_AUTO_TEST_CASE(LdexpOperator_FOV_Forward_1) {
  double a = 4., b = 3., out;

  setCurrentTape(tapeId);

  adouble ad;
  adouble bd;

  trace_on(tapeId);
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

  fov_forward(tapeId, 1, 2, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(LdexpOperator_FOV_Reverse_1) {
  double a = 4., b = 3., aout;

  setCurrentTape(tapeId);

  adouble ad;
  adouble bd;

  trace_on(tapeId, 1);
  ad <<= a;
  bd <<= b;

  ad = ad * pow(2., bd);

  ad >>= aout;
  trace_off();

  double aDerivative = std::pow(2., b);
  double bDerivative = a * std::pow(2., b) * std::log(2.);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 2);

  u[0][0] = 1.;
  u[1][0] = -2.;

  fov_reverse(tapeId, 1, 2, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == -2. * aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == -2. * bDerivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(LdexpOperator_FOV_Forward_2) {
  double a = 4., e = 3., aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
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

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * 2., tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(LdexpOperator_FOV_Reverse_2) {
  double a = 4., e = 3., aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = ldexp(ad, e);

  ad >>= aout;
  trace_off();

  double aDerivative = std::pow(2., e);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = std::exp(std::log(10.));

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == std::exp(std::log(10.)) * aDerivative,
             tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(LdexpOperator_FOV_Forward_3) {
  double a = 4., e = 3., eout;

  setCurrentTape(tapeId);

  adouble ed;

  trace_on(tapeId);
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

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == e, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == eDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == eDerivative * 2., tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(LdexpOperator_FOV_Reverse_3) {
  double a = 4., e = 3., eout;

  setCurrentTape(tapeId);

  adouble ed;

  trace_on(tapeId, 1);
  ed <<= e;

  ed = a * pow(2., ed);

  ed >>= eout;
  trace_off();

  double eDerivative = a * std::pow(2., e) * std::log(2.);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 2.2;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == eDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == 2.2 * eDerivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(FabsOperator_FOV_Forward) {
  double a = 1.4, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
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
    xd[0][i] = 1. - i * 1.5;
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. - 1.5), tt::tolerance(tol));

  x[0] = -5.;

  a = std::fabs(-5.);
  aDerivative = -1.;

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. - 1.5), tt::tolerance(tol));

  x[0] = 0.;

  xd[0][0] = 2.5;
  xd[0][1] = -3.5;

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == 0., tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == 2.5, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == 3.5, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(FabsOperator_FOV_Reverse) {
  double a = 1.4, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = fabs(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 3.3;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == 3.3 * aDerivative, tt::tolerance(tol));
  a = -5.;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = fabs(ad);

  ad >>= aout;
  trace_off();

  aDerivative = -1.;

  u[0][0] = 1.;
  u[1][0] = 3.3;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == 3.3 * aDerivative, tt::tolerance(tol));

  a = 0.;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = fabs(ad);

  ad >>= aout;
  trace_off();

  u[0][0] = 2.5;
  u[1][0] = -3.5;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == 0., tt::tolerance(tol));
  BOOST_TEST(z[1][0] == 0., tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(AbsOperator_FOV_Forward) {
  double a = 1.4, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
  ad <<= a;

  ad = abs(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;
  a = std::abs(a);

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 1.4;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i * 1.5;
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. - 1.5), tt::tolerance(tol));

  x[0] = -5.;

  a = std::abs(-5.);
  aDerivative = -1.;

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. - 1.5), tt::tolerance(tol));

  x[0] = 0.;

  xd[0][0] = 2.5;
  xd[0][1] = -3.5;

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == 0., tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == 2.5, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == 3.5, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(AbsOperator_FOV_Reverse) {
  double a = 1.4, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = abs(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 3.3;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == 3.3 * aDerivative, tt::tolerance(tol));
  a = -5.;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = abs(ad);

  ad >>= aout;
  trace_off();

  aDerivative = -1.;

  u[0][0] = 1.;
  u[1][0] = 3.3;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == 3.3 * aDerivative, tt::tolerance(tol));

  a = 0.;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = abs(ad);

  ad >>= aout;
  trace_off();

  u[0][0] = 2.5;
  u[1][0] = -3.5;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == 0., tt::tolerance(tol));
  BOOST_TEST(z[1][0] == 0., tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(CeilOperator_FOV_Forward) {
  double a = 3.573, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
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
    xd[0][i] = 1. + i * 5.8;
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * 6.8, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(CeilOperator_FOV_Reverse) {
  double a = 3.573, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = ceil(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 0.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 6.8;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * 6.8, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(FloorOperator_FOV_Forward) {
  double a = 4.483, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
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
    xd[0][i] = 1. - i * 5.8;
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (-4.8), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(FloorOperator_FOV_Reverse) {
  double a = 4.483, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = floor(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 0.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 6.8;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * 6.8, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(FmaxOperator_FOV_Forward_1) {
  double a = 4., b = 3.2, out;

  setCurrentTape(tapeId);

  adouble ad;
  adouble bd;

  trace_on(tapeId);
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

  fov_forward(tapeId, 1, 2, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative, tt::tolerance(tol));

  x[0] = 2.5;
  x[1] = 2.5;

  xd[0][0] = 1.3;
  xd[0][1] = 3.7;
  xd[1][0] = -1.3;
  xd[1][1] = -3.7;

  a = std::fmax(2.5, 2.5);
  aDerivative = std::fmax(xd[0][0], xd[1][0]);
  bDerivative = std::fmax(xd[0][1], xd[1][1]);

  fov_forward(tapeId, 1, 2, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(FmaxOperator_FOV_Reverse_1) {
  double a = 4., b = 3.2, aout;

  setCurrentTape(tapeId);

  adouble ad;
  adouble bd;

  trace_on(tapeId, 1);
  ad <<= a;
  bd <<= b;

  ad = fmax(ad, bd);

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;
  double bDerivative = 0.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 2);

  u[0][0] = 1.;
  u[1][0] = 2.;

  fov_reverse(tapeId, 1, 2, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == 2. * aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == 2. * bDerivative, tt::tolerance(tol));

  a = 2.5, b = 2.5;

  trace_on(tapeId, 1);
  ad <<= a;
  bd <<= b;

  ad = fmax(ad, bd);

  ad >>= aout;
  trace_off();

  u[0][0] = 1.;
  u[1][0] = -1.;

  fov_reverse(tapeId, 1, 2, 2, u, z);

  BOOST_TEST(z[0][0] == 0.5, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == 0.5, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == -0.5, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == -0.5, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(FmaxOperator_FOV_Forward_2) {
  double a = 4., b = 3.2, bout;

  setCurrentTape(tapeId);

  adouble bd;

  trace_on(tapeId);
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
    xd[0][i] = 1. - i * 2.1;
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == b, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative * (1. - 2.1), tt::tolerance(tol));

  x[0] = 5.2;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i * 2.1;
  }

  b = std::fmax(a, x[0]);
  bDerivative = 1.;

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == b, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative * (1. - 2.1), tt::tolerance(tol));

  x[0] = 4.5;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i * 2.1;
  }

  b = std::fmax(a, x[0]);
  bDerivative = 1.;

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == b, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative * (1. - 2.1), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(FmaxOperator_FOV_Reverse_2) {
  double a = 4., b = 3.2, bout;

  setCurrentTape(tapeId);

  adouble bd;

  trace_on(tapeId, 1);
  bd <<= b;

  bd = fmax(a, bd);

  bd >>= bout;
  trace_off();

  double bDerivative = 0.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 6.8;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == bDerivative * 6.8, tt::tolerance(tol));

  b = 5.2;

  trace_on(tapeId, 1);
  bd <<= b;

  bd = fmax(a, bd);

  bd >>= bout;
  trace_off();

  bDerivative = 1.;

  u[0][0] = 1.;
  u[1][0] = 6.8;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == bDerivative * 6.8, tt::tolerance(tol));

  b = 4.;

  trace_on(tapeId, 1);
  bd <<= b;

  bd = fmax(a, bd);

  bd >>= bout;
  trace_off();

  bDerivative = 0.5;

  u[0][0] = 1.;
  u[1][0] = 6.8;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == bDerivative * 6.8, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(FmaxOperator_FOV_Forward_3) {
  double a = 4., b = 3.2, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
  ad <<= a;

  ad = fmax(ad, b);

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;
  double bDerivative = 1.;
  a = std::fmax(a, b);

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 4.;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i * 2.1;
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. - 2.1), tt::tolerance(tol));

  x[0] = 2.3;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i * 2.1;
  }

  a = std::fmax(x[0], b);
  aDerivative = 0.;

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. - 2.1), tt::tolerance(tol));

  x[0] = 3.2;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i * 2.1;
  }

  a = std::fmax(x[0], b);
  aDerivative = std::fmax(xd[0][0], 0.0);
  bDerivative = std::fmax(xd[0][1], 0.0);

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(FmaxOperator_FOV_Reverse_3) {
  double a = 4., b = 3.2, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = fmax(ad, b);

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 6.8;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * 6.8, tt::tolerance(tol));

  a = 2.5;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = fmax(ad, b);

  ad >>= aout;
  trace_off();

  aDerivative = 0.;

  u[0][0] = 1.;
  u[1][0] = 6.8;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * 6.8, tt::tolerance(tol));

  a = 3.2;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = fmax(ad, b);

  ad >>= aout;
  trace_off();

  aDerivative = 0.5;

  u[0][0] = 1.;
  u[1][0] = 6.8;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * 6.8, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(MaxOperator_FOV_Forward_1) {
  double a = 4., b = 3.2, out;

  setCurrentTape(tapeId);

  adouble ad;
  adouble bd;

  trace_on(tapeId);
  ad <<= a;
  bd <<= b;

  ad = max(ad, bd);

  ad >>= out;
  trace_off();

  double aDerivative = 1.;
  double bDerivative = 0.;
  a = std::max(a, b);

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

  fov_forward(tapeId, 1, 2, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative, tt::tolerance(tol));

  x[0] = 2.5;
  x[1] = 2.5;

  xd[0][0] = 1.3;
  xd[0][1] = 3.7;
  xd[1][0] = -1.3;
  xd[1][1] = -3.7;

  a = std::max(2.5, 2.5);
  aDerivative = std::max(xd[0][0], xd[1][0]);
  bDerivative = std::max(xd[0][1], xd[1][1]);

  fov_forward(tapeId, 1, 2, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(MaxOperator_FOV_Reverse_1) {
  double a = 4., b = 3.2, aout;

  setCurrentTape(tapeId);

  adouble ad;
  adouble bd;

  trace_on(tapeId, 1);
  ad <<= a;
  bd <<= b;

  ad = max(ad, bd);

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;
  double bDerivative = 0.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 2);

  u[0][0] = 1.;
  u[1][0] = 2.;

  fov_reverse(tapeId, 1, 2, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == 2. * aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == 2. * bDerivative, tt::tolerance(tol));

  a = 2.5, b = 2.5;

  trace_on(tapeId, 1);
  ad <<= a;
  bd <<= b;

  ad = max(ad, bd);

  ad >>= aout;
  trace_off();

  u[0][0] = 1.;
  u[1][0] = -1.;

  fov_reverse(tapeId, 1, 2, 2, u, z);

  BOOST_TEST(z[0][0] == 0.5, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == 0.5, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == -0.5, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == -0.5, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(MaxOperator_FOV_Forward_2) {
  double a = 4., b = 3.2, bout;

  setCurrentTape(tapeId);

  adouble bd;

  trace_on(tapeId);
  bd <<= b;

  bd = max(a, bd);

  bd >>= bout;
  trace_off();

  double bDerivative = 0.;
  b = std::max(a, b);

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 3.2;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i * 2.1;
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == b, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative * (1. - 2.1), tt::tolerance(tol));

  x[0] = 5.2;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i * 2.1;
  }

  b = std::max(a, x[0]);
  bDerivative = 1.;

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == b, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative * (1. - 2.1), tt::tolerance(tol));

  x[0] = 4.5;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i * 2.1;
  }

  b = std::max(a, x[0]);
  bDerivative = 1.;

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == b, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative * (1. - 2.1), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(MaxOperator_FOV_Reverse_2) {
  double a = 4., b = 3.2, bout;

  setCurrentTape(tapeId);

  adouble bd;

  trace_on(tapeId, 1);
  bd <<= b;

  bd = max(a, bd);

  bd >>= bout;
  trace_off();

  double bDerivative = 0.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 6.8;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == bDerivative * 6.8, tt::tolerance(tol));

  b = 5.2;

  trace_on(tapeId, 1);
  bd <<= b;

  bd = max(a, bd);

  bd >>= bout;
  trace_off();

  bDerivative = 1.;

  u[0][0] = 1.;
  u[1][0] = 6.8;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == bDerivative * 6.8, tt::tolerance(tol));

  b = 4.;

  trace_on(tapeId, 1);
  bd <<= b;

  bd = max(a, bd);

  bd >>= bout;
  trace_off();

  bDerivative = 0.5;

  u[0][0] = 1.;
  u[1][0] = 6.8;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == bDerivative * 6.8, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(MaxOperator_FOV_Forward_3) {
  double a = 4., b = 3.2, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
  ad <<= a;

  ad = max(ad, b);

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;
  double bDerivative = 1.;
  a = std::max(a, b);

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 4.;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i * 2.1;
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. - 2.1), tt::tolerance(tol));

  x[0] = 2.3;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i * 2.1;
  }

  a = std::max(x[0], b);
  aDerivative = 0.;

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. - 2.1), tt::tolerance(tol));

  x[0] = 3.2;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i * 2.1;
  }

  a = std::max(x[0], b);
  aDerivative = std::max(xd[0][0], 0.0);
  bDerivative = std::max(xd[0][1], 0.0);

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(MaxOperator_FOV_Reverse_3) {
  double a = 4., b = 3.2, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = max(ad, b);

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 6.8;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * 6.8, tt::tolerance(tol));

  a = 2.5;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = max(ad, b);

  ad >>= aout;
  trace_off();

  aDerivative = 0.;

  u[0][0] = 1.;
  u[1][0] = 6.8;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * 6.8, tt::tolerance(tol));

  a = 3.2;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = max(ad, b);

  ad >>= aout;
  trace_off();

  aDerivative = 0.5;

  u[0][0] = 1.;
  u[1][0] = 6.8;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * 6.8, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(FminOperator_FOV_Forward_1) {
  double a = 4., b = 3.2, out;

  setCurrentTape(tapeId);

  adouble ad;
  adouble bd;

  trace_on(tapeId);
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

  fov_forward(tapeId, 1, 2, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative, tt::tolerance(tol));

  x[0] = 2.5;
  x[1] = 2.5;

  xd[0][0] = 1.3;
  xd[0][1] = 3.7;
  xd[1][0] = -1.3;
  xd[1][1] = -3.7;

  a = std::fmin(2.5, 2.5);
  aDerivative = std::fmin(xd[0][0], xd[1][0]);
  bDerivative = std::fmin(xd[0][1], xd[1][1]);

  fov_forward(tapeId, 1, 2, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(FminOperator_FOV_Reverse_1) {
  double a = 4., b = 3.2, aout;

  setCurrentTape(tapeId);

  adouble ad;
  adouble bd;

  trace_on(tapeId, 1);
  ad <<= a;
  bd <<= b;

  ad = fmin(ad, bd);

  ad >>= aout;
  trace_off();

  double aDerivative = 0.;
  double bDerivative = 1.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 2);

  u[0][0] = 1.;
  u[1][0] = 2.;

  fov_reverse(tapeId, 1, 2, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == 2. * aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == 2. * bDerivative, tt::tolerance(tol));

  a = 2.5, b = 2.5;

  trace_on(tapeId, 1);
  ad <<= a;
  bd <<= b;

  ad = fmin(ad, bd);

  ad >>= aout;
  trace_off();

  u[0][0] = 1.;
  u[1][0] = -1.;

  fov_reverse(tapeId, 1, 2, 2, u, z);

  BOOST_TEST(z[0][0] == 0.5, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == 0.5, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == -0.5, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == -0.5, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(FminOperator_FOV_Forward_2) {
  double a = 4., b = 3.2, bout;

  setCurrentTape(tapeId);

  adouble bd;

  trace_on(tapeId);
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
    xd[0][i] = 1. - i * 2.1;
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == b, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative * (1. - 2.1), tt::tolerance(tol));

  x[0] = 5.2;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i * 2.1;
  }

  b = std::fmin(a, x[0]);
  bDerivative = 0.;

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == b, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative * (1. - 2.1), tt::tolerance(tol));

  x[0] = 4.5;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i * 2.1;
  }

  b = std::fmin(a, x[0]);
  bDerivative = 0.;

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == b, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative * (1. - 2.1), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(FminOperator_FOV_Reverse_2) {
  double a = 4., b = 3.2, bout;

  setCurrentTape(tapeId);

  adouble bd;

  trace_on(tapeId, 1);
  bd <<= b;

  bd = fmin(a, bd);

  bd >>= bout;
  trace_off();

  double bDerivative = 1.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 6.8;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == bDerivative * 6.8, tt::tolerance(tol));

  b = 5.2;

  trace_on(tapeId, 1);
  bd <<= b;

  bd = fmin(a, bd);

  bd >>= bout;
  trace_off();

  bDerivative = 0.;

  u[0][0] = 1.;
  u[1][0] = 6.8;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == bDerivative * 6.8, tt::tolerance(tol));

  b = 4.;

  trace_on(tapeId, 1);
  bd <<= b;

  bd = fmin(a, bd);

  bd >>= bout;
  trace_off();

  bDerivative = 0.5;

  u[0][0] = 1.;
  u[1][0] = 6.8;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == bDerivative * 6.8, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(FminOperator_FOV_Forward_3) {
  double a = 4., b = 3.2, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
  ad <<= a;

  ad = fmin(ad, b);

  ad >>= aout;
  trace_off();

  double aDerivative = 0.;
  double bDerivative = 0.;
  a = std::fmin(a, b);

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 4.;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i * 2.1;
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. - 2.1), tt::tolerance(tol));

  x[0] = 2.3;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i * 2.1;
  }

  a = std::fmin(x[0], b);
  aDerivative = 1.;

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. - 2.1), tt::tolerance(tol));

  x[0] = 3.2;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i * 2.1;
  }

  a = std::fmin(x[0], b);
  aDerivative = std::fmin(xd[0][0], 0.0);
  bDerivative = std::fmin(xd[0][1], 0.0);

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(FminOperator_FOV_Reverse_3) {
  double a = 4., b = 3.2, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = fmin(ad, b);

  ad >>= aout;
  trace_off();

  double aDerivative = 0.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 6.8;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * 6.8, tt::tolerance(tol));

  a = 2.5;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = fmin(ad, b);

  ad >>= aout;
  trace_off();

  aDerivative = 1.;

  u[0][0] = 1.;
  u[1][0] = 6.8;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * 6.8, tt::tolerance(tol));

  a = 3.2;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = fmin(ad, b);

  ad >>= aout;
  trace_off();

  aDerivative = 0.5;

  u[0][0] = 1.;
  u[1][0] = 6.8;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * 6.8, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(MinOperator_FOV_Forward_1) {
  double a = 4., b = 3.2, out;

  setCurrentTape(tapeId);

  adouble ad;
  adouble bd;

  trace_on(tapeId);
  ad <<= a;
  bd <<= b;

  ad = min(ad, bd);

  ad >>= out;
  trace_off();

  double aDerivative = 0.;
  double bDerivative = 1.;
  a = std::min(a, b);

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

  fov_forward(tapeId, 1, 2, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative, tt::tolerance(tol));

  x[0] = 2.5;
  x[1] = 2.5;

  xd[0][0] = 1.3;
  xd[0][1] = 3.7;
  xd[1][0] = -1.3;
  xd[1][1] = -3.7;

  a = std::min(2.5, 2.5);
  aDerivative = std::min(xd[0][0], xd[1][0]);
  bDerivative = std::min(xd[0][1], xd[1][1]);

  fov_forward(tapeId, 1, 2, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(MinOperator_FOV_Reverse_1) {
  double a = 4., b = 3.2, aout;

  setCurrentTape(tapeId);

  adouble ad;
  adouble bd;

  trace_on(tapeId, 1);
  ad <<= a;
  bd <<= b;

  ad = min(ad, bd);

  ad >>= aout;
  trace_off();

  double aDerivative = 0.;
  double bDerivative = 1.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 2);

  u[0][0] = 1.;
  u[1][0] = 2.;

  fov_reverse(tapeId, 1, 2, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == 2. * aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == 2. * bDerivative, tt::tolerance(tol));

  a = 2.5, b = 2.5;

  trace_on(tapeId, 1);
  ad <<= a;
  bd <<= b;

  ad = min(ad, bd);

  ad >>= aout;
  trace_off();

  u[0][0] = 1.;
  u[1][0] = -1.;

  fov_reverse(tapeId, 1, 2, 2, u, z);

  BOOST_TEST(z[0][0] == 0.5, tt::tolerance(tol));
  BOOST_TEST(z[0][1] == 0.5, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == -0.5, tt::tolerance(tol));
  BOOST_TEST(z[1][1] == -0.5, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(MinOperator_FOV_Forward_2) {
  double a = 4., b = 3.2, bout;

  setCurrentTape(tapeId);

  adouble bd;

  trace_on(tapeId);
  bd <<= b;

  bd = min(a, bd);

  bd >>= bout;
  trace_off();

  double bDerivative = 1.;
  b = std::min(a, b);

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 3.2;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i * 2.1;
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == b, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative * (1. - 2.1), tt::tolerance(tol));

  x[0] = 5.2;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i * 2.1;
  }

  b = std::min(a, x[0]);
  bDerivative = 0.;

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == b, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative * (1. - 2.1), tt::tolerance(tol));

  x[0] = 4.5;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i * 2.1;
  }

  b = std::min(a, x[0]);
  bDerivative = 0.;

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == b, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative * (1. - 2.1), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(MinOperator_FOV_Reverse_2) {
  double a = 4., b = 3.2, bout;

  setCurrentTape(tapeId);

  adouble bd;

  trace_on(tapeId, 1);
  bd <<= b;

  bd = min(a, bd);

  bd >>= bout;
  trace_off();

  double bDerivative = 1.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 6.8;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == bDerivative * 6.8, tt::tolerance(tol));

  b = 5.2;

  trace_on(tapeId, 1);
  bd <<= b;

  bd = min(a, bd);

  bd >>= bout;
  trace_off();

  bDerivative = 0.;

  u[0][0] = 1.;
  u[1][0] = 6.8;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == bDerivative * 6.8, tt::tolerance(tol));

  b = 4.;

  trace_on(tapeId, 1);
  bd <<= b;

  bd = min(a, bd);

  bd >>= bout;
  trace_off();

  bDerivative = 0.5;

  u[0][0] = 1.;
  u[1][0] = 6.8;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == bDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == bDerivative * 6.8, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(MinOperator_FOV_Forward_3) {
  double a = 4., b = 3.2, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
  ad <<= a;

  ad = min(ad, b);

  ad >>= aout;
  trace_off();

  double aDerivative = 0.;
  double bDerivative = 0.;
  a = std::min(a, b);

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 4.;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i * 2.1;
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. - 2.1), tt::tolerance(tol));

  x[0] = 2.3;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i * 2.1;
  }

  a = std::min(x[0], b);
  aDerivative = 1.;

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * (1. - 2.1), tt::tolerance(tol));

  x[0] = 3.2;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = 1. - i * 2.1;
  }

  a = std::min(x[0], b);
  aDerivative = std::min(xd[0][0], 0.0);
  bDerivative = std::min(xd[0][1], 0.0);

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == bDerivative, tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(MinOperator_FOV_Reverse_3) {
  double a = 4., b = 3.2, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = min(ad, b);

  ad >>= aout;
  trace_off();

  double aDerivative = 0.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = 6.8;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * 6.8, tt::tolerance(tol));

  a = 2.5;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = min(ad, b);

  ad >>= aout;
  trace_off();

  aDerivative = 1.;

  u[0][0] = 1.;
  u[1][0] = 6.8;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * 6.8, tt::tolerance(tol));

  a = 3.2;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = min(ad, b);

  ad >>= aout;
  trace_off();

  aDerivative = 0.5;

  u[0][0] = 1.;
  u[1][0] = 6.8;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * 6.8, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(ErfOperator_FOV_Forward) {
  double a = 7.1, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
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
    xd[0][i] = std::pow(3., i * 2.);
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * std::pow(3., 2.), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(ErfOperator_FOV_Reverse) {
  double a = 7.1, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = erf(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = 2. / std::sqrt(std::acos(-1.)) * std::exp(-a * a);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = -1.1;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == -1.1 * aDerivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(ErfcOperator_FOV_Forward) {
  double a = 7.1, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
  ad <<= a;

  ad = erfc(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = -2. / std::sqrt(std::acos(-1.)) * std::exp(-a * a);
  a = std::erfc(a);

  double *x = myalloc1(1);
  double **xd = myalloc2(1, 2);
  double *y = myalloc1(1);
  double **yd = myalloc2(1, 2);

  x[0] = 7.1;

  for (int i = 0; i < 2; i++) {
    xd[0][i] = std::pow(3., i * 2.);
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * std::pow(3., 2.), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(ErfcOperator_FOV_Reverse) {
  double a = 7.1, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad = erfc(ad);

  ad >>= aout;
  trace_off();

  double aDerivative = -2. / std::sqrt(std::acos(-1.)) * std::exp(-a * a);

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = -1.1;

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == -1.1 * aDerivative, tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(EqPlusOperator_FOV_Forward) {
  double a = 5.132, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
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
    xd[0][i] = std::pow(4., i * 1.5);
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * std::pow(4., 1.5), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(EqPlusOperator_FOV_Reverse) {
  double a = 5.132, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad += 5.2;

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = std::pow(2., -1.1);

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * std::pow(2., -1.1), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(EqMinusOperator_FOV_Forward) {
  double a = 5.132, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
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
    xd[0][i] = std::pow(4., i * 1.5);
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * std::pow(4., 1.5), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(EqMinusOperator_FOV_Reverse) {
  double a = 5.132, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad -= 5.2;

  ad >>= aout;
  trace_off();

  double aDerivative = 1.;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = std::pow(2., -1.1);

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * std::pow(2., -1.1), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(EqTimesOperator_FOV_Forward) {
  double a = 5.132, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
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
    xd[0][i] = std::pow(4., i * 1.5);
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * std::pow(4., 1.5), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(EqTimesOperator_FOV_Reverse) {
  double a = 5.132, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad *= 5.2;

  ad >>= aout;
  trace_off();

  double aDerivative = 5.2;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = std::pow(2., -1.1);

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * std::pow(2., -1.1), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(EqDivOperator_FOV_Forward) {
  double a = 5.132, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId);
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
    xd[0][i] = std::pow(4., i * 1.5);
  }

  fov_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  BOOST_TEST(*y == a, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == aDerivative * std::pow(4., 1.5), tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(EqDivOperator_FOV_Reverse) {
  double a = 5.132, aout;

  setCurrentTape(tapeId);

  adouble ad;

  trace_on(tapeId, 1);
  ad <<= a;

  ad /= 5.2;

  ad >>= aout;
  trace_off();

  double aDerivative = 1. / 5.2;

  double **u = myalloc2(2, 1);
  double **z = myalloc2(2, 1);

  u[0][0] = 1.;
  u[1][0] = std::pow(2., -1.1);

  fov_reverse(tapeId, 1, 1, 2, u, z);

  BOOST_TEST(z[0][0] == aDerivative, tt::tolerance(tol));
  BOOST_TEST(z[1][0] == aDerivative * std::pow(2., -1.1), tt::tolerance(tol));

  myfree2(u);
  myfree2(z);
}

BOOST_AUTO_TEST_CASE(CondassignOperator_FOV_Forward) {
  double out;

  setCurrentTape(tapeId);

  adouble cond;
  adouble arg1;
  adouble arg2;
  adouble p;

  trace_on(tapeId);
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

  fov_forward(tapeId, 1, 3, 2, x, xd, y, yd);

  BOOST_TEST(*y == 3.5, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == 0.1, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == 1., tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_CASE(CondeqassignOperator_FOV_Forward) {
  double out;

  setCurrentTape(tapeId);

  adouble cond;
  adouble arg1;
  adouble arg2;
  adouble p;

  trace_on(tapeId);
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

  fov_forward(tapeId, 1, 3, 2, x, xd, y, yd);

  BOOST_TEST(*y == 3.5, tt::tolerance(tol));
  BOOST_TEST(yd[0][0] == 0.1, tt::tolerance(tol));
  BOOST_TEST(yd[0][1] == 1., tt::tolerance(tol));

  myfree1(x);
  myfree2(xd);
  myfree1(y);
  myfree2(yd);
}

BOOST_AUTO_TEST_SUITE_END()

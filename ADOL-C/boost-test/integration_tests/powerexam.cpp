#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

#include "../const.h"
#include <adolc/adolc.h> /* use of ALL ADOL-C interfaces */
#include <array>
#include <numeric>

adouble power(adouble x, int n) {
  adouble z = 1;

  if (n > 0) /* Recursion and branches */
  {
    int nh = n / 2;   /* that do not depend on  */
    z = power(x, nh); /* adoubles are fine !!!! */
    z *= z;
    if (2 * nh != n)
      z *= x;
    return z;
  } /* end if */
  else {
    if (n == 0) /* The local adouble z dies */
      return z; /* as it goes out of scope. */
    else
      return 1 / power(x, -n);
  } /* end else */
} /* end power */

BOOST_AUTO_TEST_SUITE(test_power_example)
BOOST_AUTO_TEST_CASE(PowerExampTest) {
  const short tapeId = 57;
  createNewTape(tapeId);
  setCurrentTape(tapeId);
  int n = 4;

  /* allocations and initializations */
  double **X;
  double **Y;
  X = myalloc2(1, n + 4);
  Y = myalloc2(1, n + 4);
  X[0][0] = 0.5; /* function value = 0. coefficient */
  X[0][1] = 1.0; /* first derivative = 1. coefficient */
  for (auto i = 0; i < n + 2; i++)
    X[0][i + 2] = 0;      /* further coefficients */
  double **Z;             /* used for checking consistency */
  Z = myalloc2(1, n + 2); /* between forward and reverse */

  adouble y, x; /* declare active variables */
  /* beginning of active section */
  trace_on(tapeId); /* tapeId = 1 and keep = 0 */
  x <<= X[0][0];    /* only one independent var */
  y = power(x, n);  /* actual function call */
  y >>= Y[0][0];    /* only one dependent adouble */
  trace_off();      /* no global adouble has died */
  /* end of active section */

  double u[1];                     /* weighting vector */
  u[0] = 1;                        /* for reverse call */
  for (auto i = 0; i < n + 2; i++) /* note that keep = i+1 in call */
  {
    forward(tapeId, 1, 1, i, i + 1, X, Y); /* evaluate the i-the derivative */
    if (i == 0)
      BOOST_TEST(Y[0][i] == y.value(), tt::tolerance(tol));
    else {
      Z[0][i] = Z[0][i - 1] / i; /* scale derivative to Taylorcoeff. */
      BOOST_TEST(Y[0][i] == Z[0][i], tt::tolerance(tol));
    }
    reverse(tapeId, 1, 1, i, u, Z); /* evaluate the (i+1)-st deriv. */
  } /* end for */
  myfree2(X);
  myfree2(Y);
  myfree2(Z);
}

BOOST_AUTO_TEST_SUITE_END()

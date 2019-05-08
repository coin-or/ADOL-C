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




BOOST_AUTO_TEST_SUITE_END()


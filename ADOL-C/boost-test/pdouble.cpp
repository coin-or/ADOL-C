
/*
File for explicit testing functions from uni5_for.cpp file.
*/

#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

#include <adolc/adolc.h>

#include "const.h"

BOOST_AUTO_TEST_CASE(ExpOperator_ZOS_Forward) {
  double a = 2., aout;
  double p = 2.0;
  adouble ad;

  trace_on(1);
  ad <<= a;

  pdouble pd(2.0);
  std::cout << pd.loc() << std::endl;
  ad = ad * exp(pd);

  ad >>= aout;
  trace_off();

  a = a * std::exp(p);

  BOOST_TEST(aout == a, tt::tolerance(tol));

  double *x = myalloc1(1);
  double *y = myalloc1(1);

  *x = 2.;

  zos_forward(1, 1, 1, 0, x, y);

  BOOST_TEST(*y == a, tt::tolerance(tol));

  myfree1(x);
  myfree1(y);
}

BOOST_AUTO_TEST_SUITE(test_pdouble)
BOOST_AUTO_TEST_SUITE_END()
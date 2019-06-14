#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

#include <adolc/adtl.h>
typedef adtl::adouble adouble;

#include "const.h"

//number of directions = 2
const size_t numDir = adtl::getNumDir();

BOOST_AUTO_TEST_SUITE( traceless_composite )


/************************************/
/* Tests for traceless mode with    */
/* custom composite functions       */
/* Author: Philipp Schuette         */
/************************************/

/* Most of the tests in this file are implemented similarly for trace
 * mode tests in traceCompositeTests.cpp.  Exhaustive descriptions of
 * test functions and derivatives can be found there.
 */

BOOST_AUTO_TEST_CASE(CompositeTrig1_Traceless)
{
  double x1 = 0.289, x2 = 1.927, out;
  adouble ax1 = x1, ax2 = x2;

  ax1.setADValue(0, 1.);
  ax2.setADValue(1, 1.);

  ax1 = adtl::sin(ax1)*adtl::sin(ax1) + adtl::cos(ax1)*adtl::cos(ax1) + ax2;

  double x1Derivative = 0.;
  double x2Derivative = 1.;

  x1 = std::sin(x1)*std::sin(x1) + std::cos(x1)*std::cos(x1) + x2;

  BOOST_TEST(ax1.getValue() == x1, tt::tolerance(tol));
  BOOST_TEST(ax1.getADValue(0) == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(ax1.getADValue(1) == x2Derivative, tt::tolerance(tol));
}




BOOST_AUTO_TEST_SUITE_END()


#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

#include <adolc/adtl.h>
typedef adtl::adouble adouble;

#include "const.h"

//number of directions = 2
const size_t numDir = adtl::getNumDir();

BOOST_AUTO_TEST_SUITE( traceless_vector )

/***********************************/
/* tests for traceless vector mode */
/***********************************/

BOOST_AUTO_TEST_CASE(ExpOperatorDerivativeVectorMode) {
  double a = 2.;
  adouble ad = a;

  for(size_t j = 0; j < numDir; j++) {
    ad.setADValue(j,1.);
  }

  double aDerivative = std::exp(a);
  ad = adtl::exp(ad);

  for(size_t j = 0; j < numDir; j++) { 
    BOOST_TEST(ad.getADValue(j) == aDerivative, tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(MultOperatorDerivativeVectorMode) {
  double a = 2.5;
  double b = 4.;
  adouble ad = a;
  adouble bd = b;

  ad.setADValue(0,1.);
  bd.setADValue(1,1.);

  double aDerivative = b;
  double bDerivative = a;

  adouble cd = ad * bd;

  BOOST_TEST(cd.getADValue(0) == aDerivative, tt::tolerance(tol));
  BOOST_TEST(cd.getADValue(1) == bDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_SUITE_END()

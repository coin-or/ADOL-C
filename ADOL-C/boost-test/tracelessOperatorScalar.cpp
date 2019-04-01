#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

#include <adolc/adtl.h>
typedef adtl::adouble adouble;

#include "const.h"

BOOST_AUTO_TEST_SUITE( traceless_scalar )

/***********************************/
/* tests for traceless scalar mode */
/***********************************/

BOOST_AUTO_TEST_CASE(ExpOperatorPrimal) {
  double a = 2.;
  adouble ad = a;

  a = std::exp(a);
  ad = adtl::exp(ad);

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(ExpOperatorDerivative) {
  double a = 2.;
  adouble ad = a;

  ad.setADValue(0,1.);

  double aDerivative = std::exp(a);
  ad = adtl::exp(ad);

  BOOST_TEST(ad.getADValue(0) == aDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(MultOperatorPrimal) {
  double a = 2., b = 3.5;
  adouble ad = a, bd = b;

  double c = a * b;
  adouble cd = ad * bd;

  BOOST_TEST(cd.getValue() == c, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(MultOperatorDerivativeWrtA) {
  double a = 2., b = 3.5;
  adouble ad = a, bd = b;

  double cDerivative = 3.5;
  
  ad.setADValue(0,1.);
  adouble cd = ad * bd;

  BOOST_TEST(cd.getADValue(0) == cDerivative, tt::tolerance(tol)); 
}

BOOST_AUTO_TEST_CASE(MultOperatorDerivativeWrtB) {
  double a = 2., b = 3.5;
  adouble ad = a, bd = b;

  double cDerivative = 2.;
  
  bd.setADValue(0,1.);
  adouble cd = ad * bd;

  BOOST_TEST(cd.getADValue(0) == cDerivative, tt::tolerance(tol)); 
}

BOOST_AUTO_TEST_SUITE_END()

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
/* Tests for traceless vector mode */
/* Author: Philipp Schuette        */
/***********************************/


/* For consistency, the primitive traceless mode functions of ADOL-C
 * are tested in vector mode in the same order as in scalar mode.
 */

BOOST_AUTO_TEST_CASE(ExpOperatorDerivativeVectorMode)
{
  double a = 2.;
  adouble ad = a;

  for(size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + j*2.);
  }

  double aDerivative = std::exp(a);
  ad = adtl::exp(ad);

  for(size_t j = 0; j < numDir; j++) { 
    BOOST_TEST(ad.getADValue(j) == aDerivative*(1. + j*2.), tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(MultOperatorDerivativeVectorMode)
{
  double a = 2.5, b = 4.;
  adouble ad = a, bd = b;

  ad.setADValue(0, 1.);
  bd.setADValue(1, 1.);

  double aDerivative = b;
  double bDerivative = a;

  adouble cd = ad * bd;

  BOOST_TEST(cd.getADValue(0) == aDerivative, tt::tolerance(tol));
  BOOST_TEST(cd.getADValue(1) == bDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(AddOperatorDerivativeVectorMode)
{
  double a = 1.2, b = 2.1;
  adouble ad = a, bd = b;

  ad.setADValue(0, 1.);
  bd.setADValue(1, 1.);

  double aDerivative = 1.;
  double bDerivative = 1.;

  adouble cd = ad + bd;

  BOOST_TEST(cd.getADValue(0) == aDerivative, tt::tolerance(tol));
  BOOST_TEST(cd.getADValue(1) == bDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(SubOperatorDerivativeVectorMode)
{
  double a = 3.2, b = 1.5;
  adouble ad = a, bd = b;

  ad.setADValue(0, 1.);
  bd.setADValue(1, 1.);

  double aDerivative = 1.;
  double bDerivative = -1.;

  adouble cd = ad - bd;

  BOOST_TEST(cd.getADValue(0) == aDerivative, tt::tolerance(tol));
  BOOST_TEST(cd.getADValue(1) == bDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(DivOperatorDerivativeVectorMode)
{
  double a = 0.6, b = 4.4;
  adouble ad = a, bd = b;

  ad.setADValue(0, 1.);
  bd.setADValue(1, 1.);

  double aDerivative = 1. / b;
  double bDerivative = -a / (b*b);

  adouble cd = ad / bd;

  BOOST_TEST(cd.getADValue(0) == aDerivative, tt::tolerance(tol));
  BOOST_TEST(cd.getADValue(1) == bDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(TanOperatorDerivativeVectorMode)
{
  double a = 0.8;
  adouble ad = a;

  for(size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + j*1.1);
  }

  a = std::tan(a);
  double aDerivative = 1 + a*a;

  ad = adtl::tan(ad);

  for(size_t j = 0; j < numDir; j++) { 
    BOOST_TEST(ad.getADValue(j) == aDerivative*(1. + j*1.1), tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(SinOperatorDerivativeVectorMode)
{
  double a = 0.72;
  adouble ad = a;

  for(size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + j*(-1.1));
  }

  double aDerivative = std::cos(a);
  ad = adtl::sin(ad);

  for(size_t j = 0; j < numDir; j++) { 
    BOOST_TEST(ad.getADValue(j) == aDerivative*(1. + j*(-1.1)), tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(CosOperatorDerivativeVectorMode)
{
  double a = -1.12;
  adouble ad = a;

  for(size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + j*3.2);
  }

  double aDerivative = -std::sin(a);
  ad = adtl::cos(ad);

  for(size_t j = 0; j < numDir; j++) { 
    BOOST_TEST(ad.getADValue(j) == aDerivative*(1. + j*3.2), tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(SqrtOperatorDerivativeVectorMode)
{
  double a = 6.1;
  adouble ad = a;

  for(size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + j*0.8);
  }

  double aDerivative = 1. / (2*std::sqrt(a));
  ad = adtl::sqrt(ad);

  for(size_t j = 0; j < numDir; j++) { 
    BOOST_TEST(ad.getADValue(j) == aDerivative*(1. + j*0.8), tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(LogOperatorDerivativeVectorMode)
{
  
}


BOOST_AUTO_TEST_SUITE_END()





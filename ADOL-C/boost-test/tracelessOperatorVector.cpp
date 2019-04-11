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
  double a = 9.4;
  adouble ad = a;

  for(size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. - j*0.7);
  }

  double aDerivative = 1. / a;
  ad = adtl::log(ad);

  for(size_t j = 0; j < numDir; j++) { 
    BOOST_TEST(ad.getADValue(j) == aDerivative*(1. - j*0.7), tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(SinhOperatorDerivativeVectorMode)
{
  double a = 40.;
  adouble ad = a;

  for(size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + j*1.5);
  }

  double aDerivative = std::cosh(a);
  ad = adtl::sinh(ad);

  for(size_t j = 0; j < numDir; j++) { 
    BOOST_TEST(ad.getADValue(j) == aDerivative*(1. + j*1.5), tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(CoshOperatorDerivativeVectorMode)
{
  double a = 40.;
  adouble ad = a;

  for(size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + j*1.5);
  }

  double aDerivative = std::sinh(a);
  ad = adtl::cosh(ad);

  for(size_t j = 0; j < numDir; j++) { 
    BOOST_TEST(ad.getADValue(j) == aDerivative*(1. + j*1.5), tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(TanhOperatorDerivativeVectorMode)
{
  double a = 40.0;
  adouble ad = a;

  for(size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + j*1.5);
  }

  a = std::tanh(a);
  double aDerivative = 1. - a*a;
  ad = adtl::tanh(ad);

  for(size_t j = 0; j < numDir; j++) { 
    BOOST_TEST(ad.getADValue(j) == aDerivative*(1. + j*1.5), tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(AsinOperatorDerivativeVectorMode)
{
  double a = 0.91;
  adouble ad = a;

  for(size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. - j*8.0);
  }

  double aDerivative = 1. / (std::sqrt(1. - a*a));
  ad = adtl::asin(ad);

  for(size_t j = 0; j < numDir; j++) { 
    BOOST_TEST(ad.getADValue(j) == aDerivative*(1. - j*8.0), tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(AcosOperatorDerivativeVectorMode)
{
  double a = 0.35;
  adouble ad = a;

  for(size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + j*0.6);
  }

  double aDerivative = -1. / (std::sqrt(1. - a*a));
  ad = adtl::acos(ad);

  for(size_t j = 0; j < numDir; j++) { 
    BOOST_TEST(ad.getADValue(j) == aDerivative*(1. + j*0.6), tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(AtanOperatorDerivativeVectorMode)
{
  double a = 16.3;
  adouble ad = a;

  for(size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + j*0.5);
  }

  double aDerivative = 1. / (1. + a*a);
  ad = adtl::atan(ad);

  for(size_t j = 0; j < numDir; j++) { 
    BOOST_TEST(ad.getADValue(j) == aDerivative*(1. + j*0.5), tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(Log10OperatorDerivativeVectorMode)
{
  double a = 13.2;
  adouble ad = a;

  for(size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. - j*1.8);
  }

  double aDerivative = 1. / (a * std::log(10));
  ad = adtl::log10(ad);

  for(size_t j = 0; j < numDir; j++) { 
    BOOST_TEST(ad.getADValue(j) == aDerivative*(1. - j*1.8), tt::tolerance(tol));
  }
}

#if defined(ATRIG_ERF)
BOOST_AUTO_TEST_CASE(AsinhOperatorDerivativeVectorMode)
{
  double a = 1.6;
  adouble ad = a;

  for(size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + j*0.6);
  }

  double aDerivative = 1. / (std::sqrt(a*a + 1.));
  ad = adtl::asinh(ad);

  for(size_t j = 0; j < numDir; j++) { 
    BOOST_TEST(ad.getADValue(j) == aDerivative*(1. + j*0.6), tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(AcoshOperatorDerivativeVectorMode)
{
  double a = 1.7;
  adouble ad = a;

  for(size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. - j*2.1);
  }

  double aDerivative = 1. / (std::sqrt(a*a - 1.));
  ad = adtl::acosh(ad);

  for(size_t j = 0; j < numDir; j++) { 
    BOOST_TEST(ad.getADValue(j) == aDerivative*(1. - j*2.1), tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(AtanhOperatorDerivativeVectorMode)
{
  double a = 0.75;
  adouble ad = a;

  for(size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + j*1.3);
  }

  double aDerivative = 1. / (1. - a*a);
  ad = adtl::atanh(ad);

  for(size_t j = 0; j < numDir; j++) { 
    BOOST_TEST(ad.getADValue(j) == aDerivative*(1. + j*1.3), tt::tolerance(tol));
  }
}
#endif

BOOST_AUTO_TEST_CASE(InclOperatorDerivativeVectorMode)
{
  double a = 5.5;
  adouble ad = a;

  for(size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + j*1.);
  }

  double aDerivative = 1.;
  ad = ++ad;

  for(size_t j = 0; j < numDir; j++) { 
    BOOST_TEST(ad.getADValue(j) == aDerivative*(1. + j*1.), tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(DeclOperatorDerivativeVectorMode)
{
  double a = 5.;
  adouble ad = a;

  for(size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + j*1.5);
  }

  double aDerivative = 1.;
  ad = --ad;

  for(size_t j = 0; j < numDir; j++) { 
    BOOST_TEST(ad.getADValue(j) == aDerivative*(1. + j*1.5), tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(SignPlusOperatorDerivativeVectorMode)
{
  double a = 1.6;
  adouble ad = a;

  for(size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. - j*0.4);
  }

  double aDerivative = 1.;
  ad = +ad;

  for(size_t j = 0; j < numDir; j++) { 
    BOOST_TEST(ad.getADValue(j) == aDerivative*(1. - j*0.4), tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(SignMinusOperatorDerivativeVectorMode)
{
  double a = 1.6;
  adouble ad = a;

  for(size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + j*0.25);
  }

  double aDerivative = -1.;
  ad = -ad;

  for(size_t j = 0; j < numDir; j++) { 
    BOOST_TEST(ad.getADValue(j) == aDerivative*(1. + j*0.25), tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(Atan2OperatorDerivativeVectorMode)
{
  double a = 13.2, b = 1.2;
  adouble ad = a, bd = b;

  ad.setADValue(0, 1.);
  bd.setADValue(1, 1.);

  double aDerivative = b / (a*a + b*b);
  double bDerivative = -a / (a*a + b*b);

  adouble cd = adtl::atan2(ad, bd);

  BOOST_TEST(cd.getADValue(0) == aDerivative, tt::tolerance(tol));
  BOOST_TEST(cd.getADValue(1) == bDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(PowOperatorDerivativeVectorMode_1)
{
  double a = 2.3, e = 5.3;
  adouble ad = a;

  for(size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. - j*0.35);
  }

  double aDerivative = e * std::pow(a, e - 1.);
  ad = adtl::pow(ad, e);

  for(size_t j = 0; j < numDir; j++) { 
    BOOST_TEST(ad.getADValue(j) == aDerivative*(1. - j*0.35), tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(PowOperatorDerivativeVectorMode_2)
{
  double a = 2.3, e = 5.3;
  adouble ad = a, ed = e;

  ad.setADValue(0, 1.);
  ed.setADValue(1, 1.);

  double aDerivative = e * std::pow(a, e - 1.);
  double eDerivative = std::log(a) * std::pow(a, e);

  adouble cd = adtl::pow(ad, ed);

  BOOST_TEST(cd.getADValue(0) == aDerivative, tt::tolerance(tol));
  BOOST_TEST(cd.getADValue(1) == eDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(PowOperatorDerivativeVectorMode_3)
{
  double a = 2.3, e = 5.3;
  adouble ed = e;

  for(size_t j = 0; j < numDir; j++) {
    ed.setADValue(j, 1. - j*0.35);
  }

  double eDerivative = std::log(a) * std::pow(a, e);
  ed = adtl::pow(a, ed);

  for(size_t j = 0; j < numDir; j++) { 
    BOOST_TEST(ed.getADValue(j) == eDerivative*(1. - j*0.35), tt::tolerance(tol));
  }
}

/* Frexp is not differentiable and so does not need to be tested. */




BOOST_AUTO_TEST_SUITE_END()





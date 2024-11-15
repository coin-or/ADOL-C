#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

#include <adolc/adtl.h>
typedef adtl::adouble adouble;

#include "const.h"

BOOST_AUTO_TEST_SUITE(traceless_scalar)

/***********************************/
/* Tests for traceless scalar mode */
/* Author: Philipp Schuette        */
/***********************************/

/* Naming convention for test cases:  Operatorname_Operator_Primal for primal
 * function value.  Operatorname_Operator_Derivative(_WrtX) for function
 * derivative (or partial derivative wrt variable X).
 */

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

  /* Derivative value is exp(a)*adot. */
  double aDerivative = std::exp(a);

  /* Fast syntax for setting the 0th derivate direction; in scalar mode this
   * is the only one required.  Default derivative value is 0.0.
   */
  ad.setADValue(0, 1.);
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

/* First case with multiple active input variables.  All partial
 * derivatives are tested.
 */

BOOST_AUTO_TEST_CASE(MultOperatorDerivativeWrtA) {
  double a = 2., b = 3.5;
  adouble ad = a, bd = b;

  double cDerivative = 3.5;

  ad.setADValue(0, 1.);
  adouble cd = ad * bd;

  /* The expected derivative value is cdot = a*bdot + adot*b = b, because
   * adot = 1., bdot = 0. (by default).
   */
  BOOST_TEST(cd.getADValue(0) == cDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(MultOperatorDerivativeWrtB) {
  double a = 2., b = 3.5;
  adouble ad = a, bd = b;

  double cDerivative = 2.;

  bd.setADValue(0, 1.);
  adouble cd = ad * bd;

  BOOST_TEST(cd.getADValue(0) == cDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(AddOperatorPrimal) {
  double a = 2.5, b = 3.;
  adouble ad = a, bd = b;

  double c = a + b;
  adouble cd = ad + bd;

  BOOST_TEST(cd.getValue() == c, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(AddOperatorDerivativeWrtA) {
  double a = 2.5, b = 3.;
  adouble ad = a, bd = b;

  /* Expected derivate value is cdot = 1.*adot + 1.*bdot. */
  double cDerivative = 1.;

  ad.setADValue(0, 1.);
  adouble cd = ad + bd;

  BOOST_TEST(cd.getADValue(0) == cDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(AddOperatorDerivativeWrtB) {
  double a = 2.5, b = 3.;
  adouble ad = a, bd = b;

  double cDerivative = 1.;

  bd.setADValue(0, 1.);
  adouble cd = ad + bd;

  BOOST_TEST(cd.getADValue(0) == cDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(AddOperatorDerivativeWrtAB) {
  double a = 2.5, b = 3.;
  adouble ad = a, bd = b;

  /* For diversity, check derivative calculation with adot = 4., bdot = 5.5. */
  double cDerivative = 1. * 4. + 1. * 5.5;

  ad.setADValue(0, 4.);
  bd.setADValue(0, 5.5);
  adouble cd = ad + bd;

  BOOST_TEST(cd.getADValue(0) == cDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(SubOperatorPrimal) {
  double a = 1.5, b = 3.2;
  adouble ad = a, bd = b;

  double c = a - b;
  adouble cd = ad - bd;

  BOOST_TEST(cd.getValue() == c, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(SubOperatorDerivateWrtA) {
  double a = 1.5, b = 3.2;
  adouble ad = a, bd = b;

  /* Derivate value is cdot = 1.*adot - 1.*bdot. */
  double cDerivative = 1.;

  ad.setADValue(0, 1.);
  adouble cd = ad - bd;

  BOOST_TEST(cd.getADValue(0) == cDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(SubOperatorDerivateWrtB) {
  double a = 1.5, b = 3.2;
  adouble ad = a, bd = b;

  double cDerivative = -1.;

  bd.setADValue(0, 1.);
  adouble cd = ad - bd;

  BOOST_TEST(cd.getADValue(0) == cDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(SubOperatorDerivateWrtAB) {
  double a = 1.5, b = 3.2;
  adouble ad = a, bd = b;

  double cDerivative = 1. * 7.3 - 1. * 1.1;

  ad.setADValue(0, 7.3);
  bd.setADValue(0, 1.1);
  adouble cd = ad - bd;

  BOOST_TEST(cd.getADValue(0) == cDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(DivOperatorPrimal) {
  double a = 0.5, b = 4.5;
  adouble ad = a, bd = b;

  double c = a / b;
  adouble cd = ad / bd;

  BOOST_TEST(cd.getValue() == c, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(DivOperatorDerivativeWrtA) {
  double a = 0.5, b = 4.5;
  adouble ad = a, bd = b;

  /* Derivative value cdot = (adot/b) - (c/b)*bdot. */
  double cDerivative = 1. / b;

  ad.setADValue(0, 1.);
  adouble cd = ad / bd;

  BOOST_TEST(cd.getADValue(0) == cDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(DivOperatorDerivativeWrtB) {
  double a = 0.5, b = 4.5;
  adouble ad = a, bd = b;

  double c = a / b;
  double cDerivative = -(c / b);

  bd.setADValue(0, 1.);
  adouble cd = ad / bd;

  BOOST_TEST(cd.getADValue(0) == cDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(DivOperatorDerivativeWrtAB) {
  double a = 0.5, b = 4.5;
  adouble ad = a, bd = b;

  double c = a / b;
  double cDerivative = (2. / b) - (c / b) * 3.;

  ad.setADValue(0, 2.);
  bd.setADValue(0, 3.);
  adouble cd = ad / bd;

  BOOST_TEST(cd.getADValue(0) == cDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(TanOperatorPrimal) {
  double a = 0.7;
  adouble ad = a;

  a = std::tan(a);
  ad = adtl::tan(ad);

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(TanOperatorDerivative) {
  double a = 0.7;
  adouble ad = a;

  a = std::tan(a);
  /* Derivative value: tan = sin/cos --> tan_prime = 1./(cos*cos) = 1 + tan*tan.
   */
  double aDerivative = (1. + a * a);

  ad.setADValue(0, 1.);
  ad = adtl::tan(ad);

  BOOST_TEST(ad.getADValue(0) == aDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(SinOperatorPrimal) {
  double a = 1.2;
  adouble ad = a;

  a = std::sin(a);
  ad = adtl::sin(ad);

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(SinOperatorDerivative) {
  double a = 1.2;
  adouble ad = a;

  /* Derivative of sin is cos. */
  double aDerivative = std::cos(a);

  ad.setADValue(0, 1.);
  ad = adtl::sin(ad);

  BOOST_TEST(ad.getADValue(0) == aDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CosOperatorPrimal) {
  double a = 1.2;
  adouble ad = a;

  a = std::cos(a);
  ad = adtl::cos(ad);

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CosOperatorDerivative) {
  double a = 1.2;
  adouble ad = a;

  /* Derivative of cos is -sin. */
  double aDerivative = -1. * std::sin(a);

  ad.setADValue(0, 1.);
  ad = adtl::cos(ad);

  BOOST_TEST(ad.getADValue(0) == aDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(SqrtOperatorPrimal) {
  double a = 2.2;
  adouble ad = a;

  a = std::sqrt(a);
  ad = adtl::sqrt(ad);

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(SqrtOperatorDerivative) {
  double a = 2.2;
  adouble ad = a;

  a = std::sqrt(a);
  /* Derivative value is 1./(2*sqrt). */
  double aDerivative = 1. / (2 * a);

  ad.setADValue(0, 1.);
  ad = adtl::sqrt(ad);

  BOOST_TEST(ad.getADValue(0) == aDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(LogOperatorPrimal) {
  double a = 4.9;
  adouble ad = a;

  a = std::log(a);
  ad = adtl::log(ad);

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(LogOperatorDerivative) {
  double a = 4.9;
  adouble ad = a;

  /* Derivative value is 1./a. */
  double aDerivative = 1. / a;

  ad.setADValue(0, 1.);
  ad = adtl::log(ad);

  BOOST_TEST(ad.getADValue(0) == aDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(SinhOperatorPrimal) {
  double a = 4.;
  adouble ad = a;

  a = std::sinh(a);
  ad = adtl::sinh(ad);

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(SinhOperatorDerivative) {
  double a = 4.;
  adouble ad = a;

  /* Derivative of sinh is cosh. */
  double aDerivative = std::cosh(a);

  ad.setADValue(0, 1.);
  ad = adtl::sinh(ad);

  BOOST_TEST(ad.getADValue(0) == aDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CoshOperatorPrimal) {
  double a = 4.;
  adouble ad = a;

  a = std::cosh(a);
  ad = adtl::cosh(ad);

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CoshOperatorDerivative) {
  double a = 4.;
  adouble ad = a;

  /* Derivative of cosh is sinh. */
  double aDerivative = std::sinh(a);

  ad.setADValue(0, 1.);
  ad = adtl::cosh(ad);

  BOOST_TEST(ad.getADValue(0) == aDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(TanhOperatorPrimal) {
  double a = 4.;
  adouble ad = a;

  a = std::tanh(a);
  ad = adtl::tanh(ad);

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(TanhOperatorDerivative) {
  double a = 4.;
  adouble ad = a;

  a = std::tanh(a);
  /* Derivative value 1./(cosh*cosh) = 1 - tanh*tanh. */
  double aDerivative = 1 - a * a;

  ad.setADValue(0, 1.);
  ad = adtl::tanh(ad);

  BOOST_TEST(ad.getADValue(0) == aDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(AsinOperatorPrimal) {
  double a = 0.9;
  adouble ad = a;

  a = std::asin(a);
  ad = adtl::asin(ad);

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(AsinOperatorDerivative) {
  double a = 0.9;
  adouble ad = a;

  /* Derivative value 1. / sqrt(1. - a*a). */
  double aDerivative = 1. / (std::sqrt(1. - a * a));

  ad.setADValue(0, 1.);
  ad = adtl::asin(ad);

  BOOST_TEST(ad.getADValue(0) == aDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(AcosOperatorPrimal) {
  double a = 0.8;
  adouble ad = a;

  a = std::acos(a);
  ad = adtl::acos(ad);

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(AcosOperatorDerivative) {
  double a = 0.8;
  adouble ad = a;

  /* Derivative value -1. / sqrt(1. - a*a). */
  double aDerivative = -1. / (std::sqrt(1. - a * a));

  ad.setADValue(0, 1.);
  ad = adtl::acos(ad);

  BOOST_TEST(ad.getADValue(0) == aDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(AtanOperatorPrimal) {
  double a = 9.8;
  adouble ad = a;

  a = std::atan(a);
  ad = adtl::atan(ad);

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(AtanOperatorDerivative) {
  double a = 9.8;
  adouble ad = a;

  /* Derivative value 1./(1. + a*a). */
  double aDerivative = 1. / (1. + a * a);

  ad.setADValue(0, 1.);
  ad = adtl::atan(ad);

  BOOST_TEST(ad.getADValue(0) == aDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Log10OperatorPrimal) {
  double a = 12.3;
  adouble ad = a;

  a = std::log10(a);
  ad = adtl::log10(ad);

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Log10OperatorDerivative) {
  double a = 12.3;
  adouble ad = a;

  /* Derivative value 1. / (log(10)*a), because log10(a) = log(a)/log(10). */
  double aDerivative = 1. / (a * std::log(10));

  ad.setADValue(0, 1.);
  ad = adtl::log10(ad);

  BOOST_TEST(ad.getADValue(0) == aDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(AsinhOperatorPrimal) {
  double a = 0.6;
  adouble ad = a;

  a = std::asinh(a);
  ad = adtl::asinh(ad);

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(AsinhOperatorDerivative) {
  double a = 0.6;
  adouble ad = a;

  double aDerivative = 1. / (std::sqrt(a * a + 1.));

  ad.setADValue(0, 1.);
  ad = adtl::asinh(ad);

  BOOST_TEST(ad.getADValue(0) == aDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(AcoshOperatorPrimal) {
  double a = 1.7;
  adouble ad = a;

  a = std::acosh(a);
  ad = adtl::acosh(ad);

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(AcoshOperatorDerivative) {
  double a = 1.7;
  adouble ad = a;

  double aDerivative = 1. / (std::sqrt(a * a - 1.));

  ad.setADValue(0, 1.);
  ad = adtl::acosh(ad);

  BOOST_TEST(ad.getADValue(0) == aDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(AtanhOperatorPrimal) {
  double a = 0.6;
  adouble ad = a;

  a = std::atanh(a);
  ad = adtl::atanh(ad);

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(AtanhOperatorDerivative) {
  double a = 0.6;
  adouble ad = a;

  double aDerivative = 1. / (1. - a * a);

  ad.setADValue(0, 1.);
  ad = adtl::atanh(ad);

  BOOST_TEST(ad.getADValue(0) == aDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(InclOperatorPrimal) {
  double a = 5.;
  adouble ad = a;

  ++a;
  ++ad;

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(InclOperatorDerivative) {
  double a = 5.;
  adouble ad = a;

  double aDerivative = 1.;

  ad.setADValue(0, 1.);
  ++ad;

  BOOST_TEST(ad.getADValue(0) == aDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(DeclOperatorPrimal) {
  double a = 5.;
  adouble ad = a;

  --a;
  --ad;

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(DeclOperatorDerivative) {
  double a = 5.;
  adouble ad = a;

  double aDerivative = 1.;

  ad.setADValue(0, 1.);
  --ad;

  BOOST_TEST(ad.getADValue(0) == aDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(SignPlusOperatorPrimal) {
  double a = 1.5;
  adouble ad = a;

  a = +a;
  ad = +ad;

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(SignPlusOperatorDerivative) {
  double a = 1.5;
  adouble ad = a;

  double aDerivative = 1.;

  ad.setADValue(0, 1.);
  ad = +ad;

  BOOST_TEST(ad.getADValue(0) == aDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(SignMinusOperatorPrimal) {
  double a = 1.5;
  adouble ad = a;

  a = -a;
  ad = -ad;

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(SignMinusOperatorDerivative) {
  double a = 1.5;
  adouble ad = a;

  double aDerivative = -1.;

  ad.setADValue(0, 1.);
  ad = -ad;

  BOOST_TEST(ad.getADValue(0) == aDerivative, tt::tolerance(tol));
}

/* The atan2 operator is defined as atan2(a, b) = atan(a/b). */
BOOST_AUTO_TEST_CASE(Atan2OperatorPrimal) {
  double a = 12.3, b = 2.1;
  adouble ad = a, bd = b;

  double c = std::atan2(a, b);
  adouble cd = adtl::atan2(ad, bd);

  BOOST_TEST(cd.getValue() == c, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Atan2OperatorDerivativeWrtA) {
  double a = 12.3, b = 2.1;
  adouble ad = a, bd = b;

  /* Partial derivative wrt a of atan2(a, b) is b/(a*a + b*b). */
  double c = b / (a * a + b * b);

  ad.setADValue(0, 1.);
  adouble cd = adtl::atan2(ad, bd);

  BOOST_TEST(cd.getADValue(0) == c, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Atan2OperatorDerivativeWrtB) {
  double a = 12.3, b = 2.1;
  adouble ad = a, bd = b;

  /* Partial derivative wrt b of atan2(a, b) is -a/(a*a + b*b). */
  double c = -a / (a * a + b * b);

  bd.setADValue(0, 1.);
  adouble cd = adtl::atan2(ad, bd);

  BOOST_TEST(cd.getADValue(0) == c, tt::tolerance(tol));
}

/* For the pow operator (pow(x, n) = x^n), ADOL-C provides three different
 * options with the following signatures:
 *
 * (1) adouble pow(adouble, double)  --> can be differentiated wrt the base
 * adouble (2) adouble pow(adouble, adouble) --> can be differentiated wrt both
 * base and exponent (3) adouble pow(double, adouble)  --> can be differentiated
 * wrt the exponent adouble
 *
 * tests for these three are implemented separately for both primal and
 * derivative value.
 */

BOOST_AUTO_TEST_CASE(PowOperatorPrimal_1) {
  double a = 2.3;
  adouble ad = a;

  /* Common exponent for both std:: and adtl:: */
  double e = 3.5;

  a = std::pow(a, e);
  ad = adtl::pow(ad, e);

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(PowOperatorDerivative_1) {
  double a = 2.3;
  adouble ad = a;

  /* Common exponent for both std:: and adtl:: */
  double e = 3.5;

  double aDerivative = e * std::pow(a, e - 1.);

  ad.setADValue(0, 1.);
  ad = adtl::pow(ad, e);

  BOOST_TEST(ad.getADValue(0) == aDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(PowOperatorPrimal_2) {
  double a = 2.3, e = 3.5;
  adouble ad = a, ed = e;

  a = std::pow(a, e);
  ad = adtl::pow(ad, ed);

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(PowOperatorDerivative_2_WrtA) {
  double a = 2.3, e = 3.5;
  adouble ad = a, ed = e;

  double aDerivative = e * std::pow(a, e - 1.);

  ad.setADValue(0, 1.);
  ad = adtl::pow(ad, ed);

  BOOST_TEST(ad.getADValue(0) == aDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(PowOperatorDerivative_2_WrtB) {
  double a = 2.3, e = 3.5;
  adouble ad = a, ed = e;

  /* For derivative calculation use pow(a, e) = exp(e*log(a)). */
  double eDerivative = std::log(a) * std::pow(a, e);

  ed.setADValue(0, 1.);
  ed = adtl::pow(ad, ed);

  BOOST_TEST(ed.getADValue(0) == eDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(PowOperatorPrimal_3) {
  double e = 3.5;
  adouble ed = e;

  /* Common base for both std:: and adtl:: */
  double a = 2.3;

  e = std::pow(a, e);
  ed = adtl::pow(a, ed);

  BOOST_TEST(ed.getValue() == e, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(PowOperatorDerivative_3) {
  double e = 3.5;
  adouble ed = e;

  /* Common base for both std:: and adtl:: */
  double a = 2.3;

  double eDerivative = std::log(a) * std::pow(a, e);

  ed.setADValue(0, 1.);
  ed = adtl::pow(a, ed);

  BOOST_TEST(ed.getADValue(0) == eDerivative, tt::tolerance(tol));
}

/* Only the primitive value of frexp has to be tested, as this function
 * is not differentiable.
 */
BOOST_AUTO_TEST_CASE(FrexpOperatorPrimal_Derivative) {
  double a = 4.348;
  adouble ad = a;

  int m = 1;
  int *n;
  n = &m;

  double aValue = std::frexp(a, n);

  ad.setADValue(0, 1.);
  ad = adtl::frexp(ad, n);

  BOOST_TEST(ad.getValue() == aValue, tt::tolerance(tol));
  /* ADValue is 0., because the output of frexp() is double, not adouble. */
  BOOST_TEST(ad.getADValue(0) == 0., tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(LdexpOperatorPrimal_1) {
  double a = 4., b = 3.;
  adouble ad = a, bd = b;

  a = a * std::pow(2., b);
  ad = adtl::ldexp(ad, bd);

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(LdexpOperatorDerivative_1_WrtA) {
  double a = 4., b = 3.;
  adouble ad = a, bd = b;

  double aDerivative = std::pow(2., b);

  ad.setADValue(0, 1.);
  ad = adtl::ldexp(ad, bd);

  BOOST_TEST(ad.getADValue(0) == aDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(LdexpOperatorDerivative_1_WrtB) {
  double a = 4., b = 3.;
  adouble ad = a, bd = b;

  double bDerivative = a * std::log(2.) * std::pow(2., b);

  bd.setADValue(0, 1.);
  bd = adtl::ldexp(ad, bd);

  BOOST_TEST(bd.getADValue(0) == bDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(LdexpOperatorPrimal_2) {
  double a = 4., b = 3.;
  adouble ad = a;

  a = a * std::pow(2., b);
  ad = adtl::ldexp(ad, b);

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(LdexpOperatorDerivative_2) {
  double a = 4., b = 3.;
  adouble ad = a;

  double aDerivative = std::pow(2., b);

  ad.setADValue(0, 1.);
  ad = adtl::ldexp(ad, b);

  BOOST_TEST(ad.getADValue(0) == aDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(LdexpOperatorPrimal_3) {
  double a = 4., b = 3.;
  adouble bd = b;

  b = a * std::pow(2., b);
  bd = adtl::ldexp(a, bd);

  BOOST_TEST(bd.getValue() == b, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(LdexpOperatorDerivative_3) {
  double a = 4., b = 3.;
  adouble bd = b;

  double bDerivative = a * std::log(2.) * std::pow(2., b);

  bd.setADValue(0, 1.);
  bd = adtl::ldexp(a, bd);

  BOOST_TEST(bd.getADValue(0) == bDerivative, tt::tolerance(tol));
}

/* For the absolute value operator, the following test cases are implemented:
 *
 * (1) Primal value is tested on positive, negative and zero;
 * (2) Derivative value is tested on positive value, negative value;
 * (3) Derivative value is tested on zero in positive direction, negative
 * direction.
 */

BOOST_AUTO_TEST_CASE(FabsOperatorPrimal) {
  double a = 1.4, b = -5.;
  adouble ad = a, bd = b;

  a = std::fabs(a);
  b = std::fabs(b);
  ad = adtl::fabs(ad);
  bd = adtl::fabs(bd);

  adouble cd = 0;
  cd = adtl::fabs(cd);

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
  BOOST_TEST(bd.getValue() == b, tt::tolerance(tol));
  BOOST_TEST(cd.getValue() == 0, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(FabsOperatorDerivativeAtNonZero) {
  double a = 1.4, b = -5.;
  adouble ad = a, bd = b;

  double aDerivative = 1.;

  ad.setADValue(0, 1.);
  ad = adtl::fabs(ad);

  double bDerivative = -1.;

  bd.setADValue(0, 1.);
  bd = adtl::fabs(bd);

  BOOST_TEST(ad.getADValue(0) == aDerivative, tt::tolerance(tol));
  BOOST_TEST(bd.getADValue(0) == bDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(FabsOperatorDerivativeAtZero) {
  adouble ad = 0;
  adouble bd = 0;

  ad.setADValue(0, 2.5);
  bd.setADValue(0, -3.5);

  double posDerivative = 2.5; /* ADValue * (+1.) */
  double negDerivative = 3.5; /* ADValue * (-1.) */

  ad = adtl::fabs(ad);
  bd = adtl::fabs(bd);

  BOOST_TEST(ad.getADValue(0) == posDerivative, tt::tolerance(tol));
  BOOST_TEST(bd.getADValue(0) == negDerivative, tt::tolerance(tol));
}

/* The ceil operator is implemented as to 'destroy' any derivative calculations:
 * Its derivative is set to 0 regardless of Value/ADValue.
 */
BOOST_AUTO_TEST_CASE(CeilOperatorPrimal) {
  double a = 3.573;
  adouble ad = a;

  a = std::ceil(a);
  ad = adtl::ceil(ad);

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CeilOperatorDerivative) {
  double a = 3.573;
  adouble ad = a;

  double aDerivative = 0.0;

  /* Slight variation in derivative direction value. */
  ad.setADValue(0, 1.1);
  ad = adtl::ceil(ad);

  BOOST_TEST(ad.getADValue(0) == aDerivative, tt::tolerance(tol));
}

/* To the floor operator comments similar to those regarding ceil apply. */
BOOST_AUTO_TEST_CASE(FloorOperatorPrimal) {
  double a = 4.483;
  adouble ad = a;

  a = std::floor(a);
  ad = adtl::floor(ad);

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(FloorOperatorDerivative) {
  double a = 4.483;
  adouble ad = a;

  double aDerivative = 0.0;

  ad.setADValue(0, -1.7);
  ad = adtl::ceil(ad);

  BOOST_TEST(ad.getADValue(0) == aDerivative, tt::tolerance(tol));
}

/* For the fmax and fmin operators, ADOL-C provides three different options with
 * the following signatures:
 *
 * (1) adouble fmax(adouble, adouble)  --> can be differentiated wrt both
 * variables (2) adouble fmax(double, adouble) --> can be differentiated wrt the
 * second variable (3) adouble fmax(adouble, double)  --> can be differentiated
 * wrt the first variable
 *
 * tests for these three are implemented separately for both primal and
 * derivative value.
 */

BOOST_AUTO_TEST_CASE(FmaxOperatorPrimal_1) {
  double a = 4., b = 3.2;
  adouble ad = a, bd = b;

  double c = std::fmax(a, b);
  adouble cd = adtl::fmax(ad, bd);

  BOOST_TEST(cd.getValue() == c, tt::tolerance(tol));
}

/* If a > b, then the derivative is calculated as fmax(a, b) = a;  similarly
 * for b > a.  If a = b, then the greater derivative direction is adopted as
 * the derivative value.  Thus, the cases a > b and a = b are tested.
 */

BOOST_AUTO_TEST_CASE(FmaxOperatorDerivative_1) {
  /* First partial derivative, a > b. */
  double a1 = 4., b1 = 3.2;
  adouble a1d = a1, b1d = b1;

  double derivative1 = 1.;

  a1d.setADValue(0, 1.);
  adouble c1d = adtl::fmax(a1d, b1d);

  BOOST_TEST(c1d.getADValue(0) == derivative1, tt::tolerance(tol));

  /* Second partial derivative, a > b. */
  double a2 = 4., b2 = 3.2;
  adouble a2d = a2, b2d = b2;

  double derivative2 = 0.;

  b2d.setADValue(0, 1.);
  adouble c2d = adtl::fmax(a2d, b2d);

  BOOST_TEST(c2d.getADValue(0) == derivative2, tt::tolerance(tol));

  /* Derivative for a = b, with a.ADVal < b.ADVal. */
  double a3 = 2.5, b3 = 2.5;
  adouble a3d = a3, b3d = b3;

  double derivative3 = 3.7;

  a3d.setADValue(0, 1.3);
  b3d.setADValue(0, 3.7);
  adouble c3d = adtl::fmax(a3d, b3d);

  BOOST_TEST(c3d.getADValue(0) == derivative3, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(FmaxOperatorPrimal_2) {
  double a = 4., b = 3.2;
  adouble bd = b;

  double c = std::fmax(a, b);
  adouble cd = adtl::fmax(a, bd);

  BOOST_TEST(cd.getValue() == c, tt::tolerance(tol));
}

/* If a < b, then fmax(double a, adouble b) behaves like case (1);
 * if a > b, then the derivative is set to 0.0 (a is not active!).
 * If a = b, fmax carries through a positive derivative value of b and
 * introduces 0.0 as derivative value for negative b.ADVal.
 */

BOOST_AUTO_TEST_CASE(FmaxOperatorDerivative_2) {
  /* Case a > b. */
  double a1 = 4., b1 = 3.2;
  adouble b1d = b1;

  double derivative1 = 0.;

  b1d.setADValue(0, 1.);
  adouble c1d = adtl::fmax(a1, b1d);

  BOOST_TEST(c1d.getADValue(0) == derivative1, tt::tolerance(tol));

  /* Case a < b. */
  double a2 = 4., b2 = 5.2;
  adouble b2d = b2;

  double derivative2 = 1.;

  b2d.setADValue(0, 1.);
  adouble c2d = adtl::fmax(a2, b2d);

  BOOST_TEST(c2d.getADValue(0) == derivative2, tt::tolerance(tol));

  /* Case a = b. */
  double a3 = 4.5, b3 = 4.5;
  adouble b3d = b3;

  double derivative3 = 0.;

  b3d.setADValue(0, -7.3);
  adouble c3d = adtl::fmax(a3, b3d);

  BOOST_TEST(c3d.getADValue(0) == derivative3, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(FmaxOperatorPrimal_3) {
  double a = 4., b = 3.2;
  adouble ad = a;

  double c = std::fmax(a, b);
  adouble cd = adtl::fmax(ad, b);

  BOOST_TEST(cd.getValue() == c, tt::tolerance(tol));
}

/* Signature fmax(adouble, double) works basically the same as
 * signature fmax(double, adouble).
 */

BOOST_AUTO_TEST_CASE(FmaxOperatorDerivative_3) {
  /* Case a > b. */
  double a1 = 4., b1 = 3.2;
  adouble a1d = a1;

  double derivative1 = 1.;

  a1d.setADValue(0, 1.);
  adouble c1d = adtl::fmax(a1d, b1);

  BOOST_TEST(c1d.getADValue(0) == derivative1, tt::tolerance(tol));

  /* Case a < b. */
  double a2 = 4., b2 = 5.2;
  adouble a2d = a2;

  double derivative2 = 0.;

  a2d.setADValue(0, 1.);
  adouble c2d = adtl::fmax(a2d, b2);

  BOOST_TEST(c2d.getADValue(0) == derivative2, tt::tolerance(tol));

  /* Case a = b. */
  double a3 = 4.5, b3 = 4.5;
  adouble a3d = a3;

  double derivative3 = 7.3;

  a3d.setADValue(0, 7.3);
  adouble c3d = adtl::fmax(a3d, b3);

  BOOST_TEST(c3d.getADValue(0) == derivative3, tt::tolerance(tol));
}

/* The procedure for fmin() is completely analogous to that for fmax(). */

BOOST_AUTO_TEST_CASE(FminOperatorPrimal_1) {
  double a = 4., b = 3.2;
  adouble ad = a, bd = b;

  double c = std::fmin(a, b);
  adouble cd = adtl::fmin(ad, bd);

  BOOST_TEST(cd.getValue() == c, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(FminOperatorDerivative_1) {
  /* First partial derivative, a > b. */
  double a1 = 4., b1 = 3.2;
  adouble a1d = a1, b1d = b1;

  double derivative1 = 0.;

  a1d.setADValue(0, 1.);
  adouble c1d = adtl::fmin(a1d, b1d);

  BOOST_TEST(c1d.getADValue(0) == derivative1, tt::tolerance(tol));

  /* Second partial derivative, a > b. */
  double a2 = 4., b2 = 3.2;
  adouble a2d = a2, b2d = b2;

  double derivative2 = 1.;

  b2d.setADValue(0, 1.);
  adouble c2d = adtl::fmin(a2d, b2d);

  BOOST_TEST(c2d.getADValue(0) == derivative2, tt::tolerance(tol));

  /* Derivative for a = b, with a.ADVal < b.ADVal. */
  double a3 = 2.5, b3 = 2.5;
  adouble a3d = a3, b3d = b3;

  double derivative3 = 1.3;

  a3d.setADValue(0, 1.3);
  b3d.setADValue(0, 3.7);
  adouble c3d = adtl::fmin(a3d, b3d);

  BOOST_TEST(c3d.getADValue(0) == derivative3, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(FminOperatorPrimal_2) {
  double a = 4., b = 3.2;
  adouble bd = b;

  double c = std::fmin(a, b);
  adouble cd = adtl::fmin(a, bd);

  BOOST_TEST(cd.getValue() == c, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(FminOperatorDerivative_2) {
  /* Case a > b. */
  double a1 = 4., b1 = 3.2;
  adouble b1d = b1;

  double derivative1 = 1.;

  b1d.setADValue(0, 1.);
  adouble c1d = adtl::fmin(a1, b1d);

  BOOST_TEST(c1d.getADValue(0) == derivative1, tt::tolerance(tol));

  /* Case a < b. */
  double a2 = 4., b2 = 5.2;
  adouble b2d = b2;

  double derivative2 = 0.;

  b2d.setADValue(0, 1.);
  adouble c2d = adtl::fmin(a2, b2d);

  BOOST_TEST(c2d.getADValue(0) == derivative2, tt::tolerance(tol));

  /* Case a = b. */
  double a3 = 4.5, b3 = 4.5;
  adouble b3d = b3;

  double derivative3 = -7.3;

  b3d.setADValue(0, -7.3);
  adouble c3d = adtl::fmin(a3, b3d);

  BOOST_TEST(c3d.getADValue(0) == derivative3, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(FminOperatorPrimal_3) {
  double a = 4., b = 3.2;
  adouble ad = a;

  double c = std::fmin(a, b);
  adouble cd = adtl::fmin(ad, b);

  BOOST_TEST(cd.getValue() == c, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(FminOperatorDerivative_3) {
  /* Case a > b. */
  double a1 = 4., b1 = 3.2;
  adouble a1d = a1;

  double derivative1 = 0.;

  a1d.setADValue(0, 1.);
  adouble c1d = adtl::fmin(a1d, b1);

  BOOST_TEST(c1d.getADValue(0) == derivative1, tt::tolerance(tol));

  /* Case a < b. */
  double a2 = 4., b2 = 5.2;
  adouble a2d = a2;

  double derivative2 = 1.;

  a2d.setADValue(0, 1.);
  adouble c2d = adtl::fmin(a2d, b2);

  BOOST_TEST(c2d.getADValue(0) == derivative2, tt::tolerance(tol));

  /* Case a = b. */
  double a3 = 4.5, b3 = 4.5;
  adouble a3d = a3;

  double derivative3 = 0.;

  a3d.setADValue(0, 7.3);
  adouble c3d = adtl::fmin(a3d, b3);

  BOOST_TEST(c3d.getADValue(0) == derivative3, tt::tolerance(tol));
}

/* The error function erf(a) is defined as
 * 2. / sqrt(pi) * int_{0, a} (exp(- t^2))dt.
 */
BOOST_AUTO_TEST_CASE(ErfOperatorPrimal) {
  double a = 7.1;
  adouble ad = a;

  a = std::erf(a);
  ad = adtl::erf(ad);

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(ErfOperatorDerivative) {
  double a = 7.1;
  adouble ad = a;

  /* The inverse cosine is used for pi so that no additional math library
   * import is necessary, see also the implementation in adtl.h.
   */
  double aDerivative = 2. / std::sqrt(std::acos(-1.)) * std::exp(-a * a);

  ad.setADValue(0, 1.);
  ad = adtl::erf(ad);

  BOOST_TEST(ad.getADValue(0) == aDerivative, tt::tolerance(tol));
}

/* The complementary error function erfc(a) is defined as
 * 1.0 - erf(a).
 */
BOOST_AUTO_TEST_CASE(ErfcOperatorPrimal) {
  double a = 7.1;
  adouble ad = a;

  a = std::erfc(a);
  ad = adtl::erfc(ad);

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(ErfcOperatorDerivative) {
  double a = 7.1;
  adouble ad = a;

  /* The inverse cosine is used for pi so that no additional math library
   * import is necessary, see also the implementation in adtl.h.
   */
  double aDerivative = -2. / std::sqrt(std::acos(-1.)) * std::exp(-a * a);

  ad.setADValue(0, 1.);
  ad = adtl::erfc(ad);

  BOOST_TEST(ad.getADValue(0) == aDerivative, tt::tolerance(tol));
}

/* Test the primitive non-temporary operations =, +=, -=, *=, /=. */

BOOST_AUTO_TEST_CASE(EqOperatorPrimal_Derivative) {
  double a = 10.01;
  adouble ad = a;

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
  BOOST_TEST(ad.getADValue(0) == 0.0, tt::tolerance(tol));

  ad.setValue(5.147);
  ad.setADValue(0, 9.919);

  BOOST_TEST(ad.getValue() == 5.147, tt::tolerance(tol));
  BOOST_TEST(ad.getADValue(0) == 9.919, tt::tolerance(tol));

  adouble bd = ad;

  BOOST_TEST(ad.getValue() == bd.getValue(), tt::tolerance(tol));
  BOOST_TEST(ad.getADValue(0) == bd.getADValue(0), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(EqPlusOperatorPrimal_Derivative) {
  double a = 5.132;
  adouble ad = a;

  a += 5.2;
  ad.setADValue(0, 2.1);
  ad += 5.2;

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
  BOOST_TEST(ad.getADValue(0) == 2.1, tt::tolerance(tol));

  adouble bd;
  bd.setValue(1.1);
  bd.setADValue(0, 11.1);

  ad += bd;
  double Value = (5.132 + 5.2 + 1.1);
  double ADValue = (2.1 + 11.1);
  BOOST_TEST(ad.getValue() == Value, tt::tolerance(tol));
  BOOST_TEST(ad.getADValue(0) == ADValue, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(EqMinusOperatorPrimal_Derivative) {
  double a = 5.132;
  adouble ad = a;

  a -= 5.2;
  ad.setADValue(0, 2.1);
  ad -= 5.2;

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
  BOOST_TEST(ad.getADValue(0) == 2.1, tt::tolerance(tol));

  adouble bd;
  bd.setValue(1.1);
  bd.setADValue(0, 11.1);

  ad -= bd;
  double Value = (5.132 - 5.2 - 1.1);
  double ADValue = (2.1 - 11.1);
  BOOST_TEST(ad.getValue() == Value, tt::tolerance(tol));
  BOOST_TEST(ad.getADValue(0) == ADValue, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(EqTimesOperatorPrimal_Derivative) {
  double a = 5.132;
  adouble ad = a;

  a *= 5.2;
  ad.setADValue(0, 2.1);
  ad *= 5.2;

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
  BOOST_TEST(ad.getADValue(0) == (2.1 * 5.2), tt::tolerance(tol));

  adouble bd;
  bd.setValue(1.1);
  bd.setADValue(0, 11.1);

  ad *= bd;
  double Value = (5.132 * 5.2 * 1.1);
  double ADValue = (2.1 * 5.2 * 1.1) + (5.132 * 5.2 * 11.1);
  BOOST_TEST(ad.getValue() == Value, tt::tolerance(tol));
  BOOST_TEST(ad.getADValue(0) == ADValue, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(EqDivOperatorPrimal_Derivative) {
  double a = 5.132;
  adouble ad = a;

  a /= 5.2;
  ad.setADValue(0, 2.1);
  ad /= 5.2;

  BOOST_TEST(ad.getValue() == a, tt::tolerance(tol));
  BOOST_TEST(ad.getADValue(0) == (2.1 / 5.2), tt::tolerance(tol));

  adouble bd;
  bd.setValue(1.1);
  bd.setADValue(0, 11.1);

  ad /= bd;
  double Value = ((5.132 / 5.2) / 1.1);
  double ADValue = (2.1 / 5.2) / 1.1 - (5.132 / 5.2) * 11.1 / (1.1 * 1.1);
  BOOST_TEST(ad.getValue() == Value, tt::tolerance(tol));
  BOOST_TEST(ad.getADValue(0) == ADValue, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(NotOperatorPrimal) {
  double a = 1.0;
  adouble ad = a;

  BOOST_TEST(!a == 0.0, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CompNeqOperatorPrimal) {
  double a = 1.5, b = 0.5;
  adouble ad = a, bd = b;

  int n = (ad != bd);
  int m = (a != b);

  BOOST_TEST(n == m, tt::tolerance(tol));

  int k = (ad != a);
  int l = (a != a);

  BOOST_TEST(k == l, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CompEqOperatorPrimal) {
  double a = 0.5, b = 1.5;
  adouble ad = a, bd = b;

  int n = (ad == bd);
  int m = (a == b);

  BOOST_TEST(n == m, tt::tolerance(tol));

  int k = (ad == a);
  int l = (a == a);

  BOOST_TEST(k == l, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CompLeqOperatorPrimal) {
  double a = 1.0, b = 0.99;
  adouble ad = a, bd = b;

  int n = (ad <= bd);
  int m = (a <= b);

  BOOST_TEST(n == m, tt::tolerance(tol));

  int k = (ad <= a);
  int l = (a <= a);

  BOOST_TEST(k == l, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CompGeqOperatorPrimal) {
  double a = 1.2, b = 2.5;
  adouble ad = a, bd = b;

  int n = (ad >= bd);
  int m = (a >= b);

  BOOST_TEST(n == m, tt::tolerance(tol));

  int k = (ad >= a);
  int l = (a >= a);

  BOOST_TEST(k == l, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CompLeOperatorPrimal) {
  double a = 1.1, b = 1.1;
  adouble ad = a, bd = b;

  int n = (ad < bd);
  int m = (a < b);

  BOOST_TEST(n == m, tt::tolerance(tol));

  int k = (ad < a);
  int l = (a < a);

  BOOST_TEST(k == l, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CompGeOperatorPrimal) {
  double a = 1.7, b = 7.5;
  adouble ad = a, bd = b;

  int n = (ad > bd);
  int m = (a > b);

  BOOST_TEST(n == m, tt::tolerance(tol));

  int k = (ad > a);
  int l = (a > a);

  BOOST_TEST(k == l, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CondassignOperatorPrimal) {
  adouble cond = 1., arg1 = 3.5, arg2 = 5.3;
  adouble p;

  arg1.setADValue(0, 3.5);
  arg2.setADValue(0, 5.3);

  condassign(p, cond, arg1, arg2);

  BOOST_TEST(p.getValue() == arg1.getValue(), tt::tolerance(tol));
  BOOST_TEST(p.getADValue(0) == arg1.getADValue(0), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CondeqassignOperatorPrimal) {
  adouble cond = 0., arg1 = 3.5, arg2 = 5.3;
  adouble p;

  arg1.setADValue(0, 3.5);
  arg2.setADValue(0, 5.3);

  condeqassign(p, cond, arg1, arg2);

  BOOST_TEST(p.getValue() == arg1.getValue(), tt::tolerance(tol));
  BOOST_TEST(p.getADValue(0) == arg1.getADValue(0), tt::tolerance(tol));
}

BOOST_AUTO_TEST_SUITE_END()

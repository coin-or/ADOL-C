#define BOOST_TEST_DYN_LINK

#include <vector>

#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

#include <adolc/adtl.h>
typedef adtl::adouble adouble;

#include "const.h"

BOOST_AUTO_TEST_SUITE(traceless_vector)

/***********************************/
/* Tests for traceless vector mode */
/* Author: Philipp Schuette        */
/***********************************/

/* For consistency, the primitive traceless mode functions of ADOL-C
 * are tested in vector mode in the same order as in scalar mode.  For
 * implementation details, please refer to the scalar mode tests.
 */

BOOST_AUTO_TEST_CASE(ExpOperatorDerivativeVectorMode) {
  const size_t numDir = adtl::getNumDir();
  double a = 2.;
  adouble ad = a;

  for (size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + j * 2.);
  }

  double aDerivative = std::exp(a);
  ad = adtl::exp(ad);

  for (size_t j = 0; j < numDir; j++) {
    BOOST_TEST(ad.getADValue(j) == aDerivative * (1. + j * 2.),
               tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(MultOperatorDerivativeVectorMode) {
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

BOOST_AUTO_TEST_CASE(AddOperatorDerivativeVectorMode) {
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

BOOST_AUTO_TEST_CASE(SubOperatorDerivativeVectorMode) {
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

BOOST_AUTO_TEST_CASE(DivOperatorDerivativeVectorMode) {
  double a = 0.6, b = 4.4;
  adouble ad = a, bd = b;

  ad.setADValue(0, 1.);
  bd.setADValue(1, 1.);

  double aDerivative = 1. / b;
  double bDerivative = -a / (b * b);

  adouble cd = ad / bd;

  BOOST_TEST(cd.getADValue(0) == aDerivative, tt::tolerance(tol));
  BOOST_TEST(cd.getADValue(1) == bDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(TanOperatorDerivativeVectorMode) {
  const size_t numDir = adtl::getNumDir();
  double a = 0.8;
  adouble ad = a;

  for (size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + j * 1.1);
  }

  a = std::tan(a);
  double aDerivative = 1 + a * a;

  ad = adtl::tan(ad);

  for (size_t j = 0; j < numDir; j++) {
    BOOST_TEST(ad.getADValue(j) == aDerivative * (1. + j * 1.1),
               tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(SinOperatorDerivativeVectorMode) {
  const size_t numDir = adtl::getNumDir();
  double a = 0.72;
  adouble ad = a;

  for (size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + j * (-1.1));
  }

  double aDerivative = std::cos(a);
  ad = adtl::sin(ad);

  for (size_t j = 0; j < numDir; j++) {
    BOOST_TEST(ad.getADValue(j) == aDerivative * (1. + j * (-1.1)),
               tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(CosOperatorDerivativeVectorMode) {
  const size_t numDir = adtl::getNumDir();
  double a = -1.12;
  adouble ad = a;

  for (size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + j * 3.2);
  }

  double aDerivative = -std::sin(a);
  ad = adtl::cos(ad);

  for (size_t j = 0; j < numDir; j++) {
    BOOST_TEST(ad.getADValue(j) == aDerivative * (1. + j * 3.2),
               tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(SqrtOperatorDerivativeVectorMode) {
  const size_t numDir = adtl::getNumDir();
  double a = 6.1;
  adouble ad = a;

  for (size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + j * 0.8);
  }

  double aDerivative = 1. / (2 * std::sqrt(a));
  ad = adtl::sqrt(ad);

  for (size_t j = 0; j < numDir; j++) {
    BOOST_TEST(ad.getADValue(j) == aDerivative * (1. + j * 0.8),
               tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(LogOperatorDerivativeVectorMode) {
  const size_t numDir = adtl::getNumDir();
  double a = 9.4;
  adouble ad = a;

  for (size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. - j * 0.7);
  }

  double aDerivative = 1. / a;
  ad = adtl::log(ad);

  for (size_t j = 0; j < numDir; j++) {
    BOOST_TEST(ad.getADValue(j) == aDerivative * (1. - j * 0.7),
               tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(SinhOperatorDerivativeVectorMode) {
  const size_t numDir = adtl::getNumDir();
  double a = 40.;
  adouble ad = a;

  for (size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + j * 1.5);
  }

  double aDerivative = std::cosh(a);
  ad = adtl::sinh(ad);

  for (size_t j = 0; j < numDir; j++) {
    BOOST_TEST(ad.getADValue(j) == aDerivative * (1. + j * 1.5),
               tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(CoshOperatorDerivativeVectorMode) {
  const size_t numDir = adtl::getNumDir();
  double a = 40.;
  adouble ad = a;

  for (size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + j * 1.5);
  }

  double aDerivative = std::sinh(a);
  ad = adtl::cosh(ad);

  for (size_t j = 0; j < numDir; j++) {
    BOOST_TEST(ad.getADValue(j) == aDerivative * (1. + j * 1.5),
               tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(TanhOperatorDerivativeVectorMode) {
  const size_t numDir = adtl::getNumDir();
  double a = 40.0;
  adouble ad = a;

  for (size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + j * 1.5);
  }

  a = std::tanh(a);
  double aDerivative = 1. - a * a;
  ad = adtl::tanh(ad);

  for (size_t j = 0; j < numDir; j++) {
    BOOST_TEST(ad.getADValue(j) == aDerivative * (1. + j * 1.5),
               tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(AsinOperatorDerivativeVectorMode) {
  const size_t numDir = adtl::getNumDir();
  double a = 0.91;
  adouble ad = a;

  for (size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. - j * 8.0);
  }

  double aDerivative = 1. / (std::sqrt(1. - a * a));
  ad = adtl::asin(ad);

  for (size_t j = 0; j < numDir; j++) {
    BOOST_TEST(ad.getADValue(j) == aDerivative * (1. - j * 8.0),
               tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(AcosOperatorDerivativeVectorMode) {
  const size_t numDir = adtl::getNumDir();
  double a = 0.35;
  adouble ad = a;

  for (size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + j * 0.6);
  }

  double aDerivative = -1. / (std::sqrt(1. - a * a));
  ad = adtl::acos(ad);

  for (size_t j = 0; j < numDir; j++) {
    BOOST_TEST(ad.getADValue(j) == aDerivative * (1. + j * 0.6),
               tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(AtanOperatorDerivativeVectorMode) {
  const size_t numDir = adtl::getNumDir();
  double a = 16.3;
  adouble ad = a;

  for (size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + j * 0.5);
  }

  double aDerivative = 1. / (1. + a * a);
  ad = adtl::atan(ad);

  for (size_t j = 0; j < numDir; j++) {
    BOOST_TEST(ad.getADValue(j) == aDerivative * (1. + j * 0.5),
               tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(Log10OperatorDerivativeVectorMode) {
  const size_t numDir = adtl::getNumDir();
  double a = 13.2;
  adouble ad = a;

  for (size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. - j * 1.8);
  }

  double aDerivative = 1. / (a * std::log(10));
  ad = adtl::log10(ad);

  for (size_t j = 0; j < numDir; j++) {
    BOOST_TEST(ad.getADValue(j) == aDerivative * (1. - j * 1.8),
               tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(AsinhOperatorDerivativeVectorMode) {
  const size_t numDir = adtl::getNumDir();
  double a = 1.6;
  adouble ad = a;

  for (size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + j * 0.6);
  }

  double aDerivative = 1. / (std::sqrt(a * a + 1.));
  ad = adtl::asinh(ad);

  for (size_t j = 0; j < numDir; j++) {
    BOOST_TEST(ad.getADValue(j) == aDerivative * (1. + j * 0.6),
               tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(AcoshOperatorDerivativeVectorMode) {
  const size_t numDir = adtl::getNumDir();
  double a = 1.7;
  adouble ad = a;

  for (size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. - j * 2.1);
  }

  double aDerivative = 1. / (std::sqrt(a * a - 1.));
  ad = adtl::acosh(ad);

  for (size_t j = 0; j < numDir; j++) {
    BOOST_TEST(ad.getADValue(j) == aDerivative * (1. - j * 2.1),
               tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(AtanhOperatorDerivativeVectorMode) {
  const size_t numDir = adtl::getNumDir();
  double a = 0.75;
  adouble ad = a;

  for (size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + j * 1.3);
  }

  double aDerivative = 1. / (1. - a * a);
  ad = adtl::atanh(ad);

  for (size_t j = 0; j < numDir; j++) {
    BOOST_TEST(ad.getADValue(j) == aDerivative * (1. + j * 1.3),
               tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(InclOperatorDerivativeVectorMode) {
  const size_t numDir = adtl::getNumDir();
  double a = 5.5;
  adouble ad = a;

  for (size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + j * 1.);
  }

  double aDerivative = 1.;
  ad = ++ad;

  for (size_t j = 0; j < numDir; j++) {
    BOOST_TEST(ad.getADValue(j) == aDerivative * (1. + j * 1.),
               tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(DeclOperatorDerivativeVectorMode) {
  const size_t numDir = adtl::getNumDir();
  double a = 5.;
  adouble ad = a;

  for (size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + j * 1.5);
  }

  double aDerivative = 1.;
  ad = --ad;

  for (size_t j = 0; j < numDir; j++) {
    BOOST_TEST(ad.getADValue(j) == aDerivative * (1. + j * 1.5),
               tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(SignPlusOperatorDerivativeVectorMode) {
  const size_t numDir = adtl::getNumDir();
  double a = 1.6;
  adouble ad = a;

  for (size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. - j * 0.4);
  }

  double aDerivative = 1.;
  ad = +ad;

  for (size_t j = 0; j < numDir; j++) {
    BOOST_TEST(ad.getADValue(j) == aDerivative * (1. - j * 0.4),
               tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(SignMinusOperatorDerivativeVectorMode) {
  const size_t numDir = adtl::getNumDir();
  double a = 1.6;
  adouble ad = a;

  for (size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + j * 0.25);
  }

  double aDerivative = -1.;
  ad = -ad;

  for (size_t j = 0; j < numDir; j++) {
    BOOST_TEST(ad.getADValue(j) == aDerivative * (1. + j * 0.25),
               tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(Atan2OperatorDerivativeVectorMode) {
  double a = 13.2, b = 1.2;
  adouble ad = a, bd = b;

  ad.setADValue(0, 1.);
  bd.setADValue(1, 1.);

  double aDerivative = b / (a * a + b * b);
  double bDerivative = -a / (a * a + b * b);

  adouble cd = adtl::atan2(ad, bd);

  BOOST_TEST(cd.getADValue(0) == aDerivative, tt::tolerance(tol));
  BOOST_TEST(cd.getADValue(1) == bDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(PowOperatorDerivativeVectorMode_1) {
  const size_t numDir = adtl::getNumDir();
  double a = 2.3, e = 5.3;
  adouble ad = a;

  for (size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. - j * 0.35);
  }

  double aDerivative = e * std::pow(a, e - 1.);
  ad = adtl::pow(ad, e);

  for (size_t j = 0; j < numDir; j++) {
    BOOST_TEST(ad.getADValue(j) == aDerivative * (1. - j * 0.35),
               tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(PowOperatorDerivativeVectorMode_2) {
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

BOOST_AUTO_TEST_CASE(PowOperatorDerivativeVectorMode_3) {
  const size_t numDir = adtl::getNumDir();
  double a = 2.3, e = 5.3;
  adouble ed = e;

  for (size_t j = 0; j < numDir; j++) {
    ed.setADValue(j, 1. - j * 0.35);
  }

  double eDerivative = std::log(a) * std::pow(a, e);
  ed = adtl::pow(a, ed);

  for (size_t j = 0; j < numDir; j++) {
    BOOST_TEST(ed.getADValue(j) == eDerivative * (1. - j * 0.35),
               tt::tolerance(tol));
  }
}

/* Frexp is not differentiable and so does not need to be tested. */

BOOST_AUTO_TEST_CASE(LdexpOperatorDerivativeVectorMode_1) {
  double a = 3., b = 4.;
  adouble ad = a, bd = b;

  ad.setADValue(0, 1.);
  bd.setADValue(1, 1.);

  double aDerivative = std::pow(2., b);
  double bDerivative = a * std::log(2.) * std::pow(2., b);

  adouble cd = adtl::ldexp(ad, bd);

  BOOST_TEST(cd.getADValue(0) == aDerivative, tt::tolerance(tol));
  BOOST_TEST(cd.getADValue(1) == bDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(LdexpOperatorDerivativeVectorMode_2) {
  const size_t numDir = adtl::getNumDir();
  double a = 3., b = 4.;
  adouble ad = a;

  for (size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + j * 1.);
  }

  double aDerivative = std::pow(2., b);
  ad = adtl::ldexp(ad, b);

  for (size_t j = 0; j < numDir; j++) {
    BOOST_TEST(ad.getADValue(j) == aDerivative * (1. + j * 1.),
               tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(LdexpOperatorDerivativeVectorMode_3) {
  const size_t numDir = adtl::getNumDir();
  double a = 3., b = 4.;
  adouble bd = b;

  for (size_t j = 0; j < numDir; j++) {
    bd.setADValue(j, 1. - j * 1.);
  }

  double bDerivative = a * std::log(2.) * std::pow(2., b);
  bd = adtl::ldexp(a, bd);

  for (size_t j = 0; j < numDir; j++) {
    BOOST_TEST(bd.getADValue(j) == bDerivative * (1. - j * 1.),
               tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(FabsOperatorDerivativeVectorMode_NonZero) {
  const size_t numDir = adtl::getNumDir();
  double a = 1.4, b = -5.;
  adouble ad = a, bd = b;

  for (size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, 1. + std::pow(2., j));
  }
  for (size_t j = 0; j < numDir; j++) {
    bd.setADValue(j, 1. + std::pow(3., j));
  }

  double aDerivative = 1.;
  double bDerivative = -1.;

  ad = adtl::fabs(ad);
  bd = adtl::fabs(bd);

  for (size_t j = 0; j < numDir; j++) {
    BOOST_TEST(ad.getADValue(j) == aDerivative * (1. + std::pow(2., j)),
               tt::tolerance(tol));
  }
  for (size_t j = 0; j < numDir; j++) {
    BOOST_TEST(bd.getADValue(j) == bDerivative * (1. + std::pow(3., j)),
               tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(FabsOperatorDerivativeVectorMode_AtZero) {
  double c = 0.;
  adouble cd = c;

  cd.setADValue(0, 2.5);
  cd.setADValue(1, -3.5);

  double posDerivative = 2.5;
  double negDerivative = 3.5;

  cd = adtl::fabs(cd);

  BOOST_TEST(cd.getADValue(0) == posDerivative, tt::tolerance(tol));
  BOOST_TEST(cd.getADValue(1) == negDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CeilOperatorDerivativeVectorMode) {
  const size_t numDir = adtl::getNumDir();
  double a = 4.617;
  adouble ad = a;

  for (size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, std::pow(j, 10.));
  }

  double aDerivative = 0.;
  ad = adtl::ceil(ad);

  for (size_t j = 0; j < numDir; j++) {
    BOOST_TEST(ad.getADValue(j) == aDerivative, tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(FloorOperatorDerivativeVectorMode) {
  const size_t numDir = adtl::getNumDir();
  double a = 9.989;
  adouble ad = a;

  for (size_t j = 0; j < numDir; j++) {
    ad.setADValue(j, std::pow(j, 5.5));
  }

  double aDerivative = 0.;
  ad = adtl::floor(ad);

  for (size_t j = 0; j < numDir; j++) {
    BOOST_TEST(ad.getADValue(j) == aDerivative, tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(FmaxOperatorDerivativeVectorMode_1) {
  /* Derivative for a proper maximum of two input values. */
  double a = 4., b = 3.2;
  adouble ad = a, bd = b;

  double aDerivative = 1.;
  double bDerivative = 0.;

  ad.setADValue(0, 1.);
  bd.setADValue(1, 1.);

  adouble cd = adtl::fmax(ad, bd);

  BOOST_TEST(cd.getADValue(0) == aDerivative, tt::tolerance(tol));
  BOOST_TEST(cd.getADValue(1) == bDerivative, tt::tolerance(tol));

  /* Derivative for equal input values. */
  a = 2.5, b = 2.5;
  ad = a, bd = b;

  double Derivative1 = 3.7;
  double Derivative2 = 2.1;

  ad.setADValue(0, 1.3);
  bd.setADValue(0, 3.7);
  ad.setADValue(1, 2.1);
  bd.setADValue(1, 1.1);
  adouble ed = adtl::fmax(ad, bd);

  BOOST_TEST(ed.getADValue(0) == Derivative1, tt::tolerance(tol));
  BOOST_TEST(ed.getADValue(1) == Derivative2, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(FmaxOperatorDerivativeVectorMode_2) {
  /* Derivative for a proper maximum of two input values. */
  double a = 4., b = 3.2;
  adouble bd = b;

  double bDerivative = 0.;

  bd.setADValue(0, 1.);
  bd.setADValue(1, 3.21);

  adouble cd = adtl::fmax(a, bd);

  BOOST_TEST(cd.getADValue(0) == bDerivative * 1.0, tt::tolerance(tol));
  BOOST_TEST(cd.getADValue(1) == bDerivative * 3.21, tt::tolerance(tol));

  /* Derivative for equal input values. */
  a = 2.5, b = 2.5;
  bd = b;

  double Derivative1 = 3.7;
  double Derivative2 = 0.0;

  bd.setADValue(0, 3.7);
  bd.setADValue(1, -1.1);
  adouble ed = adtl::fmax(a, bd);

  BOOST_TEST(ed.getADValue(0) == Derivative1, tt::tolerance(tol));
  BOOST_TEST(ed.getADValue(1) == Derivative2, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(FmaxOperatorDerivativeVectorMode_3) {
  /* Derivative for a proper maximum of two input values. */
  double a = 4., b = 3.2;
  adouble ad = a;

  double aDerivative = 1.;

  ad.setADValue(0, 1.);
  ad.setADValue(1, 3.21);

  adouble cd = adtl::fmax(ad, b);

  BOOST_TEST(cd.getADValue(0) == aDerivative * 1.0, tt::tolerance(tol));
  BOOST_TEST(cd.getADValue(1) == aDerivative * 3.21, tt::tolerance(tol));

  /* Derivative for equal input values. */
  a = 2.5, b = 2.5;
  ad = a;

  double Derivative1 = 3.7;
  double Derivative2 = 0.0;

  ad.setADValue(0, 3.7);
  ad.setADValue(1, -1.1);
  adouble ed = adtl::fmax(ad, b);

  BOOST_TEST(ed.getADValue(0) == Derivative1, tt::tolerance(tol));
  BOOST_TEST(ed.getADValue(1) == Derivative2, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(FminOperatorDerivativeVectorMode_1) {
  /* Derivative for a proper minimum of two input values. */
  double a = 4., b = 3.2;
  adouble ad = a, bd = b;

  double aDerivative = 0.;
  double bDerivative = 1.;

  ad.setADValue(0, 1.);
  bd.setADValue(1, 1.);

  adouble cd = adtl::fmin(ad, bd);

  BOOST_TEST(cd.getADValue(0) == aDerivative, tt::tolerance(tol));
  BOOST_TEST(cd.getADValue(1) == bDerivative, tt::tolerance(tol));

  /* Derivative for equal input values. */
  a = 2.5, b = 2.5;
  ad = a, bd = b;

  double Derivative1 = 1.3;
  double Derivative2 = 1.1;

  ad.setADValue(0, 1.3);
  bd.setADValue(0, 3.7);
  ad.setADValue(1, 2.1);
  bd.setADValue(1, 1.1);
  adouble ed = adtl::fmin(ad, bd);

  BOOST_TEST(ed.getADValue(0) == Derivative1, tt::tolerance(tol));
  BOOST_TEST(ed.getADValue(1) == Derivative2, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(FminOperatorDerivativeVectorMode_2) {
  /* Derivative for a proper minimum of two input values. */
  double a = 4., b = 3.2;
  adouble bd = b;

  double bDerivative = 1.;

  bd.setADValue(0, 1.);
  bd.setADValue(1, 3.21);

  adouble cd = adtl::fmin(a, bd);

  BOOST_TEST(cd.getADValue(0) == bDerivative * 1.0, tt::tolerance(tol));
  BOOST_TEST(cd.getADValue(1) == bDerivative * 3.21, tt::tolerance(tol));

  /* Derivative for equal input values. */
  a = 2.5, b = 2.5;
  bd = b;

  double Derivative1 = -7.3;
  double Derivative2 = 0.0;

  bd.setADValue(0, -7.3);
  bd.setADValue(1, 5.2);
  adouble ed = adtl::fmin(a, bd);

  BOOST_TEST(ed.getADValue(0) == Derivative1, tt::tolerance(tol));
  BOOST_TEST(ed.getADValue(1) == Derivative2, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(FminOperatorDerivativeVectorMode_3) {
  /* Derivative for a proper minimum of two input values. */
  double a = 4., b = 3.2;
  adouble ad = a;

  double aDerivative = 0.;

  ad.setADValue(0, 1.);
  ad.setADValue(1, 3.21);

  adouble cd = adtl::fmin(ad, b);

  BOOST_TEST(cd.getADValue(0) == aDerivative * 1.0, tt::tolerance(tol));
  BOOST_TEST(cd.getADValue(1) == aDerivative * 3.21, tt::tolerance(tol));

  /* Derivative for equal input values. */
  a = 2.5, b = 2.5;
  ad = a;

  double Derivative1 = -7.3;
  double Derivative2 = 0.0;

  ad.setADValue(0, -7.3);
  ad.setADValue(1, 5.2);
  adouble ed = adtl::fmin(ad, b);

  BOOST_TEST(ed.getADValue(0) == Derivative1, tt::tolerance(tol));
  BOOST_TEST(ed.getADValue(1) == Derivative2, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(ErfOperatorDerivativeVectorMode) {
  double a = 7.1;
  adouble ad = a;

  double aDerivative = 2. / std::sqrt(std::acos(-1.)) * std::exp(-a * a);

  ad.setADValue(0, 1.);
  ad.setADValue(1, -2.5);
  ad = adtl::erf(ad);

  BOOST_TEST(ad.getADValue(0) == aDerivative, tt::tolerance(tol));
  BOOST_TEST(ad.getADValue(1) == aDerivative * (-2.5), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(ErfcOperatorDerivativeVectorMode) {
  double a = 7.1;
  adouble ad = a;

  double aDerivative = -2. / std::sqrt(std::acos(-1.)) * std::exp(-a * a);

  ad.setADValue(0, 1.);
  ad.setADValue(1, -2.5);
  ad = adtl::erfc(ad);

  BOOST_TEST(ad.getADValue(0) == aDerivative, tt::tolerance(tol));
  BOOST_TEST(ad.getADValue(1) == aDerivative * (-2.5), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(EqOperatorDerivativeVectorMode) {
  double a = 10.01;
  adouble ad = a;

  BOOST_TEST(ad.getADValue(0) == 0.0, tt::tolerance(tol));
  BOOST_TEST(ad.getADValue(1) == 0.0, tt::tolerance(tol));

  ad.setADValue(0, 5.147);
  ad.setADValue(1, -9.919);

  BOOST_TEST(ad.getADValue(0) == 5.147, tt::tolerance(tol));
  BOOST_TEST(ad.getADValue(1) == -9.919, tt::tolerance(tol));

  adouble bd = ad;

  BOOST_TEST(ad.getADValue(0) == bd.getADValue(0), tt::tolerance(tol));
  BOOST_TEST(ad.getADValue(1) == bd.getADValue(1), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(EqPlusOperatorDerivativeVectorMode) {
  double a = 5.132;
  adouble ad = a;

  ad.setADValue(0, 1.0);
  ad.setADValue(1, 2.1);
  ad += 5.2;

  BOOST_TEST(ad.getADValue(0) == 1.0, tt::tolerance(tol));
  BOOST_TEST(ad.getADValue(1) == 2.1, tt::tolerance(tol));

  adouble bd;
  bd.setValue(1.1);
  bd.setADValue(0, 11.1);
  bd.setADValue(1, -2.6);

  ad += bd;
  BOOST_TEST(ad.getADValue(0) == (1.0 + 11.1), tt::tolerance(tol));
  BOOST_TEST(ad.getADValue(1) == (2.1 - 2.6), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(EqMinusOperatorDerivativeVectorMode) {
  double a = 5.132;
  adouble ad = a;

  ad.setADValue(0, 1.0);
  ad.setADValue(1, 2.1);
  ad -= 5.2;

  BOOST_TEST(ad.getADValue(0) == 1.0, tt::tolerance(tol));
  BOOST_TEST(ad.getADValue(1) == 2.1, tt::tolerance(tol));

  adouble bd;
  bd.setValue(9.312);
  bd.setADValue(0, 11.1);
  bd.setADValue(1, -2.6);

  ad -= bd;
  BOOST_TEST(ad.getADValue(0) == (1.0 - 11.1), tt::tolerance(tol));
  BOOST_TEST(ad.getADValue(1) == (2.1 - (-2.6)), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(EqTimesOperatorDerivativeVectorMode) {
  double a = 5.132;
  adouble ad = a;

  ad.setADValue(0, 1.0);
  ad.setADValue(1, -7.1);
  ad *= 5.2;

  BOOST_TEST(ad.getADValue(0) == (5.2 * 1.0), tt::tolerance(tol));
  BOOST_TEST(ad.getADValue(1) == (5.2 * (-7.1)), tt::tolerance(tol));

  adouble bd;
  bd.setValue(1.1);
  bd.setADValue(0, 11.1);
  bd.setADValue(1, -2.1);

  ad *= bd;
  BOOST_TEST(ad.getADValue(0) == (5.132 * 5.2 * 11.1 + 1.1 * 5.2 * 1.0),
             tt::tolerance(tol));
  BOOST_TEST(ad.getADValue(1) == (5.132 * 5.2 * (-2.1) + 1.1 * 5.2 * (-7.1)),
             tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(EqDivOperatorDerivativeVectorMode) {
  double a = 5.132;
  adouble ad = a;

  a /= 5.2;
  ad.setADValue(0, 1.0);
  ad.setADValue(1, -2.1);
  ad /= 5.2;

  BOOST_TEST(ad.getADValue(0) == (1.0 / 5.2) * 1.0, tt::tolerance(tol));
  BOOST_TEST(ad.getADValue(1) == (1.0 / 5.2) * (-2.1), tt::tolerance(tol));

  adouble bd;
  bd.setValue(1.1);
  bd.setADValue(0, 1.0);
  bd.setADValue(1, 11.1);

  ad /= bd;
  double Derivative1 =
      (1.0 / 1.1) * (1.0 / 5.2) - (5.132 / (5.2 * 1.1 * 1.1) * 1.0);
  double Derivative2 =
      (1.0 / 1.1) * (-2.1 / 5.2) - (5.132 / (5.2 * 1.1 * 1.1) * 11.1);
  BOOST_TEST(ad.getADValue(0) == Derivative1, tt::tolerance(tol));
  BOOST_TEST(ad.getADValue(1) == Derivative2, tt::tolerance(tol));
}

/* The not, neq and eq comparison operators have no effect on the derivatives,
 * so the scalar tests suffice for them.  The same holds for leq, geq, greater
 * and less.
 */

BOOST_AUTO_TEST_CASE(CondassignOperatorDerivativeVectorMode) {
  adouble cond = 1., arg1 = 3.5, arg2 = 5.3;
  adouble p;

  arg1.setADValue(0, 3.5);
  arg2.setADValue(1, -7.0);
  arg2.setADValue(0, 5.3);
  arg2.setADValue(1, -10.6);

  condassign(p, cond, arg1, arg2);

  BOOST_TEST(p.getADValue(0) == arg1.getADValue(0), tt::tolerance(tol));
  BOOST_TEST(p.getADValue(1) == arg1.getADValue(1), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CondeqassignOperatorDerivativeVectorMode) {
  adouble cond = 1., arg1 = 3.5, arg2 = 5.3;
  adouble p;

  arg1.setADValue(0, 3.5);
  arg1.setADValue(1, -7.0);
  arg2.setADValue(0, 5.3);
  arg2.setADValue(1, -10.6);

  condeqassign(p, cond, arg1, arg2);

  BOOST_TEST(p.getADValue(0) == arg1.getADValue(0), tt::tolerance(tol));
  BOOST_TEST(p.getADValue(1) == arg1.getADValue(1), tt::tolerance(tol));
}

/* Test the derivative value assignment by pointer. */
BOOST_AUTO_TEST_CASE(SetADValueOperatorVectorMode) {
  const size_t numDir = adtl::getNumDir();
  adouble ad = 0.0;
  std::vector<double> aDerivative(numDir);

  for (size_t i = 0; i < numDir; i++) {
    aDerivative[i] = 1.0 + std::exp(10 * i) * std::sqrt(2.5 * i);
  }

  ad.setADValue(aDerivative.data());

  for (size_t j = 0; j < numDir; j++) {
    BOOST_TEST(ad.getADValue(j) == aDerivative[j], tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_SUITE_END()

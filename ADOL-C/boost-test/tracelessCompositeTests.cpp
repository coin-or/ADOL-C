#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

#include <adolc/adtl.h>
typedef adtl::adouble adouble;

#include "const.h"

BOOST_AUTO_TEST_SUITE(traceless_composite)

/************************************/
/* Tests for traceless mode with    */
/* custom composite functions       */
/* Author: Philipp Schuette         */
/************************************/

/* Most of the tests in this file are implemented similarly for trace
 * mode tests in traceCompositeTests.cpp.  Exhaustive descriptions of
 * test functions and derivatives can be found there.
 */

BOOST_AUTO_TEST_CASE(CompositeTrig1_Traceless) {
  double x1 = 0.289, x2 = 1.927;
  adouble ax1 = x1, ax2 = x2;

  ax1.setADValue(0, 1.);
  ax2.setADValue(1, 1.);

  ax1 = adtl::sin(ax1) * adtl::sin(ax1) + adtl::cos(ax1) * adtl::cos(ax1) + ax2;

  double x1Derivative = 0.;
  double x2Derivative = 1.;

  x1 = std::sin(x1) * std::sin(x1) + std::cos(x1) * std::cos(x1) + x2;

  BOOST_TEST(ax1.getValue() == x1, tt::tolerance(tol));
  BOOST_TEST(ax1.getADValue(0) == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(ax1.getADValue(1) == x2Derivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CompositeTrig2_Traceless) {
  double x1 = 1.11, x2 = 2.22;
  adouble ax1 = x1, ax2 = x2;

  ax1.setADValue(0, 1.);
  ax2.setADValue(1, 1.);

  ax1 = 2. * adtl::sin(adtl::cos(ax1)) * adtl::exp(ax2) - 2. * adtl::sin(ax2);

  double x1Derivative =
      -2. * std::cos(std::cos(x1)) * std::sin(x1) * std::exp(x2);
  double x2Derivative =
      2. * std::sin(std::cos(x1)) * std::exp(x2) - 2. * std::cos(x2);

  x1 = 2. * std::sin(std::cos(x1)) * std::exp(x2) - 2. * std::sin(x2);

  BOOST_TEST(ax1.getValue() == x1, tt::tolerance(tol));
  BOOST_TEST(ax1.getADValue(0) == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(ax1.getADValue(1) == x2Derivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CompositeTrig3_Traceless) {
  double x1 = 0.516, x2 = 9.89;
  adouble ax1 = x1, ax2 = x2;

  ax1.setADValue(0, 1.);
  ax2.setADValue(1, 1.);

  ax1 = adtl::pow(adtl::sin(ax1), adtl::cos(ax1) - ax2);

  double x1Derivative = std::pow(std::sin(x1), std::cos(x1) - x2) *
                        (-std::sin(x1) * std::log(std::sin(x1)) +
                         (std::cos(x1) - x2) * std::cos(x1) / std::sin(x1));
  double x2Derivative =
      -std::log(std::sin(x1)) * std::pow(std::sin(x1), std::cos(x1) - x2);

  x1 = std::pow(std::sin(x1), std::cos(x1) - x2);

  BOOST_TEST(ax1.getValue() == x1, tt::tolerance(tol));
  BOOST_TEST(ax1.getADValue(0) == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(ax1.getADValue(1) == x2Derivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CompositeTrig4_Traceless) {
  double x1 = 1.56, x2 = 8.99;
  adouble ax1 = x1, ax2 = x2;

  ax1.setADValue(0, 1.);
  ax2.setADValue(1, 1.);

  ax1 = adtl::atan(adtl::tan(ax1)) * adtl::exp(ax2);

  double x1Derivative = std::exp(x2);
  double x2Derivative = x1 * std::exp(x2);

  x1 = std::atan(std::tan(x1)) * std::exp(x2);

  BOOST_TEST(ax1.getValue() == x1, tt::tolerance(tol));
  BOOST_TEST(ax1.getADValue(0) == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(ax1.getADValue(1) == x2Derivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(LongSum_Traceless) {
  double x1 = 6.091, x2 = -0.004;
  adouble ax1 = x1, ax2 = x2;

  ax1.setADValue(0, 1.);
  ax2.setADValue(1, 1.);

  ax1 = std::pow(2., 10.) * ax1 + ax2 + std::sqrt(2.) * ax1 + (-3.5) * ax1 -
        9.9 * ax2 - 10000.001;

  double x1Derivative = std::pow(2., 10.) + std::sqrt(2.) - 3.5;
  double x2Derivative = 1. - 9.9;

  x1 = std::pow(2., 10.) * x1 + x2 + std::sqrt(2.) * x1 + (-3.5) * x1 -
       9.9 * x2 - 10000.001;

  BOOST_TEST(ax1.getValue() == x1, tt::tolerance(tol));
  BOOST_TEST(ax1.getADValue(0) == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(ax1.getADValue(1) == x2Derivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(InverseFunc_Traceless) {
  double x1 = 3.77, x2 = -21.12;
  adouble ax1 = x1, ax2 = x2;

  ax1.setADValue(0, 1.);
  ax2.setADValue(1, 1.);

  ax1 = adtl::sqrt(adtl::pow(ax1, 2.)) * ax2;

  double x1Derivative = x2;
  double x2Derivative = x1;

  x1 = x1 * x2;

  BOOST_TEST(ax1.getValue() == x1, tt::tolerance(tol));
  BOOST_TEST(ax1.getADValue(0) == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(ax1.getADValue(1) == x2Derivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(ExpPow_Traceless) {
  double x1 = 0.642, x2 = 6.42;
  adouble ax1 = x1, ax2 = x2;

  ax1.setADValue(0, 1.);
  ax2.setADValue(1, 1.);

  ax1 = adtl::exp(ax1 + adtl::exp(ax2)) * pow(ax1, ax2);

  double x1Derivative = std::exp(x1 + std::exp(x2)) * std::pow(x1, x2) +
                        std::exp(x1 + std::exp(x2)) * x2 * std::pow(x1, x2 - 1);
  double x2Derivative =
      std::exp(x1 + std::exp(x2)) * std::exp(x2) * std::pow(x1, x2) +
      std::exp(x1 + std::exp(x2)) * std::pow(x1, x2) * std::log(x1);

  x1 = std::exp(x1 + std::exp(x2)) * std::pow(x1, x2);

  BOOST_TEST(ax1.getValue() == x1, tt::tolerance(tol));
  BOOST_TEST(ax1.getADValue(0) == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(ax1.getADValue(1) == x2Derivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CompositeSqrt_Traceless) {
  double x1 = 2.22, x2 = -2.14;
  adouble ax1 = x1, ax2 = x2;

  ax1.setADValue(0, 1.);
  ax2.setADValue(1, 1.);

  ax1 = adtl::sqrt(adtl::sqrt(ax1 * adtl::exp(ax2)));

  double x1Derivative =
      0.25 * std::pow(x1 * std::exp(x2), -0.75) * std::exp(x2);
  double x2Derivative =
      0.25 * std::pow(x1 * std::exp(x2), -0.75) * x1 * std::exp(x2);

  x1 = std::sqrt(std::sqrt(x1 * std::exp(x2)));

  BOOST_TEST(ax1.getValue() == x1, tt::tolerance(tol));
  BOOST_TEST(ax1.getADValue(0) == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(ax1.getADValue(1) == x2Derivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CompositeHyper_Traceless) {
  double x1 = 0.1, x2 = 5.099;
  adouble ax1 = x1, ax2 = x2;

  ax1.setADValue(0, 1.);
  ax2.setADValue(1, 1.);

  ax1 = adtl::tanh(adtl::acos(adtl::pow(ax1, 2) + 0.5) * adtl::sin(ax2));

  double x1Derivative =
      -(1. -
        std::pow(std::tanh(std::acos(std::pow(x1, 2) + 0.5) * std::sin(x2)),
                 2)) *
      std::sin(x2) * 2. * x1 /
      (std::sqrt(1. - std::pow(std::pow(x1, 2) + 0.5, 2)));
  double x2Derivative =
      (1. - std::pow(std::tanh(std::acos(std::pow(x1, 2) + 0.5) * std::sin(x2)),
                     2)) *
      std::cos(x2) * std::acos(std::pow(x1, 2) + 0.5);

  x1 = std::tanh(std::acos(std::pow(x1, 2) + 0.5) * std::sin(x2));

  BOOST_TEST(ax1.getValue() == x1, tt::tolerance(tol));
  BOOST_TEST(ax1.getADValue(0) == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(ax1.getADValue(1) == x2Derivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CompositeFmax_Traceless) {
  double x1 = 2.31, x2 = 1.32;
  adouble ax1 = x1, ax2 = x2;

  ax1.setADValue(0, 1.);
  ax2.setADValue(1, 1.);

  ax1 = adtl::fmax(ax1 * std::sqrt(2.), ax2 * std::sqrt(3.));

  double x1Derivative = std::sqrt(2.);
  double x2Derivative = 0.;

  x1 = std::fmax(x1 * std::sqrt(2.), x2 * std::sqrt(2.));

  BOOST_TEST(ax1.getValue() == x1, tt::tolerance(tol));
  BOOST_TEST(ax1.getADValue(0) == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(ax1.getADValue(1) == x2Derivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CompositeFmin_Traceless) {
  double x1 = 2.31, x2 = 1.32;
  adouble ax1 = x1, ax2 = x2;

  ax1.setADValue(0, 1.);
  ax2.setADValue(1, 1.);

  ax1 = adtl::fmin(ax1 * std::sqrt(2.), ax2 * std::sqrt(3.));

  double x1Derivative = 0.;
  double x2Derivative = std::sqrt(3.);

  x1 = std::fmin(x1 * std::sqrt(2.), x2 * std::sqrt(3.));

  BOOST_TEST(ax1.getValue() == x1, tt::tolerance(tol));
  BOOST_TEST(ax1.getADValue(0) == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(ax1.getADValue(1) == x2Derivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CompositeErfFabs_Traceless) {
  double x1 = 4.56, x2 = 6.45;
  adouble ax1 = x1, ax2 = x2;

  ax1.setADValue(0, 1.);
  ax2.setADValue(1, 1.);

  ax1 = adtl::erf(adtl::fabs(ax1 - 2.) * adtl::sinh(ax2));

  double x1Derivative =
      2. / std::sqrt(std::acos(-1.)) *
      std::exp(-std::pow(std::fabs(x1 - 2.) * std::sinh(x2), 2)) *
      std::sinh(x2);
  double x2Derivative =
      2. / std::sqrt(std::acos(-1.)) *
      std::exp(-std::pow(std::fabs(x1 - 2.) * std::sinh(x2), 2)) *
      std::fabs(x1 - 2.) * std::cosh(x2);

  x1 = std::erf(std::fabs(x1 - 2.) * std::sinh(x2));

  BOOST_TEST(ax1.getValue() == x1, tt::tolerance(tol));
  BOOST_TEST(ax1.getADValue(0) == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(ax1.getADValue(1) == x2Derivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CompositeErfcFabs_Traceless) {
  double x1 = 4.56, x2 = 6.45;
  adouble ax1 = x1, ax2 = x2;

  ax1.setADValue(0, 1.);
  ax2.setADValue(1, 1.);

  ax1 = adtl::erfc(adtl::fabs(ax1 - 2.) * adtl::sinh(ax2));

  double x1Derivative =
      -2. / std::sqrt(std::acos(-1.)) *
      std::exp(-std::pow(std::fabs(x1 - 2.) * std::sinh(x2), 2)) *
      std::sinh(x2);
  double x2Derivative =
      -2. / std::sqrt(std::acos(-1.)) *
      std::exp(-std::pow(std::fabs(x1 - 2.) * std::sinh(x2), 2)) *
      std::fabs(x1 - 2.) * std::cosh(x2);

  x1 = std::erfc(std::fabs(x1 - 2.) * std::sinh(x2));

  BOOST_TEST(ax1.getValue() == x1, tt::tolerance(tol));
  BOOST_TEST(ax1.getADValue(0) == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(ax1.getADValue(1) == x2Derivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(ExpTrigSqrt_Traceless) {
  double x1 = 1.2, x2 = 2.1;
  adouble ax1 = x1, ax2 = x2;

  ax1.setADValue(0, 1.);
  ax2.setADValue(1, 1.);

  ax1 = 5. * adtl::exp(adtl::sin(ax1) * adtl::cos(ax1)) *
        adtl::pow(adtl::sqrt(ax2), 5.);

  double x1Derivative =
      5. * std::exp(std::sin(x1) * std::cos(x1)) *
      (std::cos(x1) * std::cos(x1) - std::sin(x1) * std::sin(x1)) *
      std::pow(std::sqrt(x2), 5.);
  double x2Derivative = 5. * std::exp(std::sin(x1) * std::cos(x1)) * 5. *
                        std::pow(std::sqrt(x2), 4.) / (2. * std::sqrt(x2));

  x1 = 5. * std::exp(std::sin(x1) * std::cos(x1)) * std::pow(std::sqrt(x2), 5.);

  BOOST_TEST(ax1.getValue() == x1, tt::tolerance(tol));
  BOOST_TEST(ax1.getADValue(0) == x1Derivative, tt::tolerance(tol));
  BOOST_TEST(ax1.getADValue(1) == x2Derivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(PolarCoord_Traceless) {
  double x1 = 8.17, x2 = -0.42;
  adouble ax1 = x1, ax2 = x2;
  double y1, y2;
  adouble ay1, ay2;

  ax1.setADValue(0, 1.);
  ax2.setADValue(1, 1.);

  ay1 = ax1 * adtl::cos(ax2);
  ay2 = ax1 * adtl::sin(ax2);

  double y1x1Derivative = std::cos(x2);
  double y1x2Derivative = -x1 * std::sin(x2);
  double y2x1Derivative = std::sin(x2);
  double y2x2Derivative = x1 * std::cos(x2);

  y1 = x1 * std::cos(x2);
  y2 = x1 * std::sin(x2);

  BOOST_TEST(ay1.getValue() == y1, tt::tolerance(tol));
  BOOST_TEST(ay1.getADValue(0) == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(ay1.getADValue(1) == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(ay2.getValue() == y2, tt::tolerance(tol));
  BOOST_TEST(ay2.getADValue(0) == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(ay2.getADValue(1) == y2x2Derivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(SimpleProd_Traceless) {
  double x1 = -5.25, x2 = 2.52;
  adouble ax1 = x1, ax2 = x2;
  double y1, y2;
  adouble ay1, ay2;

  ax1.setADValue(0, 1.);
  ax2.setADValue(1, 1.);

  ay1 = ax2 * ax2;
  ay2 = ax1 * ax1;

  double y1x1Derivative = 0.;
  double y1x2Derivative = 2. * x2;
  double y2x1Derivative = 2. * x1;
  double y2x2Derivative = 0.;

  y1 = x2 * x2;
  y2 = x1 * x1;

  BOOST_TEST(ay1.getValue() == y1, tt::tolerance(tol));
  BOOST_TEST(ay1.getADValue(0) == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(ay1.getADValue(1) == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(ay2.getValue() == y2, tt::tolerance(tol));
  BOOST_TEST(ay2.getADValue(0) == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(ay2.getADValue(1) == y2x2Derivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(SimpleSum_Traceless) {
  double x1 = -5.25, x2 = 2.52;
  adouble ax1 = x1, ax2 = x2;
  double y1, y2;
  adouble ay1, ay2;

  ax1.setADValue(0, 1.);
  ax2.setADValue(1, 1.);

  ay1 = ax2 + ax2;
  ay2 = ax1 + ax1;

  double y1x1Derivative = 0.;
  double y1x2Derivative = 2.;
  double y2x1Derivative = 2.;
  double y2x2Derivative = 0.;

  y1 = x2 + x2;
  y2 = x1 + x1;

  BOOST_TEST(ay1.getValue() == y1, tt::tolerance(tol));
  BOOST_TEST(ay1.getADValue(0) == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(ay1.getADValue(1) == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(ay2.getValue() == y2, tt::tolerance(tol));
  BOOST_TEST(ay2.getADValue(0) == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(ay2.getADValue(1) == y2x2Derivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(TrigProd_Traceless) {
  double x1 = 5.5, x2 = 0.5;
  adouble ax1 = x1, ax2 = x2;
  double y1, y2;
  adouble ay1, ay2;

  ax1.setADValue(0, 1.);
  ax2.setADValue(1, 1.);

  ay1 = adtl::sin(ax1) * adtl::cos(ax2);
  ay2 = adtl::cos(ax1) * adtl::sin(ax2);

  double y1x1Derivative = std::cos(x1) * std::cos(x2);
  double y1x2Derivative = -std::sin(x1) * std::sin(x2);
  double y2x1Derivative = -std::sin(x1) * std::sin(x2);
  double y2x2Derivative = std::cos(x1) * std::cos(x2);

  y1 = std::sin(x1) * std::cos(x2);
  y2 = std::cos(x1) * std::sin(x2);

  BOOST_TEST(ay1.getValue() == y1, tt::tolerance(tol));
  BOOST_TEST(ay1.getADValue(0) == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(ay1.getADValue(1) == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(ay2.getValue() == y2, tt::tolerance(tol));
  BOOST_TEST(ay2.getADValue(0) == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(ay2.getADValue(1) == y2x2Derivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(MultiHyperb_Traceless) {
  double x1 = 1., x2 = 0.1;
  adouble ax1 = x1, ax2 = x2;
  double y1, y2, y3, y4;
  adouble ay1, ay2, ay3, ay4;

  ax1.setADValue(0, 1.);
  ax2.setADValue(1, 1.);

  ay1 = adtl::sinh(ax1 * ax1) * adtl::cosh(ax2 * ax2 * ax2);
  ay2 = adtl::pow(adtl::cosh(adtl::pow(ax1, 4.)), 2.) -
        adtl::pow(adtl::cosh(adtl::pow(ax1, 4.)), 2.);
  ay3 = -adtl::cosh(adtl::sqrt(ax1) * ax2) * ax2;
  ay4 = adtl::cosh(ax1) / adtl::sinh(ax2);

  double y1x1Derivative =
      2. * x1 * std::cosh(x1 * x1) * std::cosh(x2 * x2 * x2);
  double y1x2Derivative =
      3. * x2 * x2 * std::sinh(x1 * x1) * std::sinh(x2 * x2 * x2);
  double y2x1Derivative = 0.0;
  double y2x2Derivative = 0.0;
  double y3x1Derivative =
      -0.5 * std::sinh(std::sqrt(x1) * x2) * x2 * x2 / std::sqrt(x1);
  double y3x2Derivative = -std::sinh(std::sqrt(x1) * x2) * std::sqrt(x1) * x2 -
                          std::cosh(std::sqrt(x1) * x2);
  double y4x1Derivative = std::sinh(x1) / std::sinh(x2);
  double y4x2Derivative =
      -std::cosh(x1) * std::cosh(x2) / std::pow(std::sinh(x2), 2.);

  y1 = std::sinh(x1 * x1) * std::cosh(x2 * x2 * x2);
  y2 = std::pow(std::cosh(std::pow(x1, 4)), 2) -
       std::pow(std::cosh(std::pow(x1, 4)), 2);
  y3 = -std::cosh(std::sqrt(x1) * x2) * x2;
  y4 = std::cosh(x1) / std::sinh(x2);

  BOOST_TEST(ay1.getValue() == y1, tt::tolerance(tol));
  BOOST_TEST(ay1.getADValue(0) == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(ay1.getADValue(1) == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(ay2.getValue() == y2, tt::tolerance(tol));
  BOOST_TEST(ay2.getADValue(0) == y2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(ay2.getADValue(1) == y2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(ay3.getValue() == y3, tt::tolerance(tol));
  BOOST_TEST(ay3.getADValue(0) == y3x1Derivative, tt::tolerance(tol));
  BOOST_TEST(ay3.getADValue(1) == y3x2Derivative, tt::tolerance(tol));
  BOOST_TEST(ay4.getValue() == y4, tt::tolerance(tol));
  BOOST_TEST(ay4.getADValue(0) == y4x1Derivative, tt::tolerance(tol));
  BOOST_TEST(ay4.getADValue(1) == y4x2Derivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_SUITE_END()

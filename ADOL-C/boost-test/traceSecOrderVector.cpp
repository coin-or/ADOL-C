#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

#include <adolc/adolc.h>

#include "const.h"

BOOST_AUTO_TEST_SUITE(trace_sec_order_vec)

/**************************************/
/* Tests for ADOL-C trace vector mode */
/* drivers hov_forward, hov_reverse   */
/* Author: Philipp Schuette           */
/**************************************/

/* This file contains custom tests for the higher order derivative
 * evaluation drivers hov_forward, hov_reverse.
 *
 * As for the trace/traceless first order derivative tests, the custom
 * functions are described, together with their derivatives, before
 * the actual test implementation.
 *
 * Every test-function is first used to test forward mode and then to
 * test reverse mode.
 */

/* Tested function: 2.*x*x*x
 * First derivative: 2.*3.*x*x
 * Second derivative: 2.*3.*2.*x
 */

BOOST_AUTO_TEST_CASE(CustomCube_HOV_Forward) {

  const auto tapeId = createNewTape();
  setCurrentTape(tapeId);
  double x1 = 3.;
  adouble ax1;
  double y1;
  adouble ay1;

  trace_on(tapeId, 1);
  ax1 <<= x1;

  ay1 = 2. * ax1 * ax1 * ax1;

  ay1 >>= y1;
  trace_off();

  std::vector<double> yprim(1);
  yprim[0] = 2. * x1 * x1 * x1;

  auto yDerivative = Tensor(1, 3, 2);
  yDerivative[0][0][0] = 2. * 3. * x1 * x1;
  yDerivative[0][0][1] = 2. * 3. * x1 * x1 + 0.5 * (2. * 3. * 2. * x1);
  yDerivative[0][1][0] = 2. * 2. * 3. * x1 * x1;
  yDerivative[0][1][1] =
      2. * 2. * 3. * x1 * x1 + 0.5 * (2. * 3. * 2. * x1) * 2. * 2.;
  yDerivative[0][2][0] = 3. * 2. * 3. * x1 * x1;
  yDerivative[0][2][1] =
      3. * 2. * 3. * x1 * x1 + 0.5 * (2. * 3. * 2. * x1) * 3. * 3.;

  std::vector<double> x(1);
  x[0] = 3.;

  auto X = Tensor(1, 3, 2);
  X[0][0][0] = 1.;
  X[0][1][0] = 2.;
  X[0][2][0] = 3.;
  X[0][0][1] = 1.;
  X[0][1][1] = 2.;
  X[0][2][1] = 3.;

  std::vector<double> y(1);

  auto Y = Tensor(1, 3, 2);

  hov_forward(tapeId, 1, 1, 2, 3, x, X, y, Y);

  BOOST_TEST(y[0] == yprim[0], tt::tolerance(tol));
  BOOST_TEST(Y[0][0][0] == yDerivative[0][0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][0][1] == yDerivative[0][0][1], tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] == yDerivative[0][1][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] == yDerivative[0][1][1], tt::tolerance(tol));
  BOOST_TEST(Y[0][2][0] == yDerivative[0][2][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][2][1] == yDerivative[0][2][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(customCube_HOV_Reverse) {
  const auto tapeId = createNewTape();
  setCurrentTape(tapeId);
  double x1 = 3.;
  adouble ax1;
  double y1;
  adouble ay1;

  trace_on(tapeId, 1);
  ax1 <<= x1;

  ay1 = 2. * ax1 * ax1 * ax1;

  ay1 >>= y1;
  trace_off();

  double y1x1Derivative = 6. * x1 * x1;

  double y1x1x1Derivative = 12. * x1;

  std::vector<double> x(1);
  std::vector<double> xd(1);
  std::vector<double> y(1);
  std::vector<double> yd(1);

  x[0] = 3.;
  xd[0] = 1.;

  fos_forward(tapeId, 1, 1, 2, x, xd, y, yd);

  std::vector<double> UCont(2 * 1);
  auto U = MatrixView(UCont, 2, 1);
  auto Z = Tensor(2, 1, 2);

  std::vector<short int> nzCont(2 * 2);
  auto nz = MatrixView(nzCont, 2, 2);

  nz[0][0] = 1;
  nz[1][0] = 1;

  U[0][0] = 1.;
  U[1][0] = 5.;

  hov_reverse(tapeId, 1, 1, 1, 2, U, Z, nz);

  BOOST_TEST(Z[0][0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[0][0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0][0] == 5. * y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0][1] == 5. * y1x1x1Derivative, tt::tolerance(tol));
}

/* Tested function: cos(x1)*sin(x2)
 * First derivatives: (-sin(x1)*sin(x2), cos(x1)*cos(x2))
 * Second derivatives: (-cos(x1)*sin(x2), -sin(x1)*cos(x2),
 *                      -sin(x1)*cos(x2), -cos(x1)*sin(x2))
 */
BOOST_AUTO_TEST_CASE(CustomTrigProd_HOV_Forward) {
  const auto tapeId = createNewTape();
  setCurrentTape(tapeId);
  double x1 = 1.3, x2 = 3.1;
  adouble ax1;
  adouble ax2;
  double y1;
  adouble ay1;

  trace_on(tapeId, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay1 = cos(ax1) * sin(ax2);

  ay1 >>= y1;
  trace_off();

  std::vector<double> yprim(1);
  yprim[0] = std::cos(x1) * std::sin(x2);

  auto yDerivative = Tensor(1, 3, 2);
  yDerivative[0][0][0] = -std::sin(x1) * std::sin(x2);
  yDerivative[0][0][1] =
      -std::sin(x1) * std::sin(x2) - 0.5 * std::cos(x1) * std::sin(x2);
  yDerivative[0][1][0] = std::cos(x1) * std::cos(x2);
  yDerivative[0][1][1] =
      std::cos(x1) * std::cos(x2) - 0.5 * std::cos(x1) * std::sin(x2);
  yDerivative[0][2][0] =
      -5. * std::sin(x1) * std::sin(x2) + 3. * std::cos(x1) * std::cos(x2);
  yDerivative[0][2][1] = -std::sin(x1) * std::sin(x2) -
                         std::cos(x1) * std::cos(x2) +
                         0.5 * (5. * (-5. * std::cos(x1) * std::sin(x2) -
                                      3. * std::sin(x1) * std::cos(x2)) +
                                3. * (-5. * std::sin(x1) * std::cos(x2) +
                                      3. * -std::cos(x1) * std::sin(x2)));

  std::vector<double> x(2);
  x[0] = 1.3;
  x[1] = 3.1;

  auto X = Tensor(2, 3, 2);
  X[0][0][0] = 1.;
  X[0][1][0] = 0.;
  X[0][2][0] = 5.;
  X[0][0][1] = 1.;
  X[0][1][1] = 0.;
  X[0][2][1] = 1.;

  X[1][0][0] = 0.;
  X[1][1][0] = 1.;
  X[1][2][0] = 3.;
  X[1][0][1] = 0.;
  X[1][1][1] = 1.;
  X[1][2][1] = -1.;

  std::vector<double> y(1);

  Tensor<double> Y(1, 3, 2);

  hov_forward(tapeId, 1, 2, 2, 3, x, X, y, Y);

  BOOST_TEST(y[0] == yprim[0], tt::tolerance(tol));
  BOOST_TEST(Y[0][0][0] == yDerivative[0][0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][0][1] == yDerivative[0][0][1], tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] == yDerivative[0][1][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] == yDerivative[0][1][1], tt::tolerance(tol));
  BOOST_TEST(Y[0][2][0] == yDerivative[0][2][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][2][1] == yDerivative[0][2][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(customTrigProd_HOV_Reverse) {
  const auto tapeId = createNewTape();
  setCurrentTape(tapeId);
  double x1 = 1.3, x2 = 3.1;
  adouble ax1;
  adouble ax2;
  double y1;
  adouble ay1;

  trace_on(tapeId, 1);
  ax1 <<= x1;
  ax2 <<= x2;

  ay1 = cos(ax1) * sin(ax2);

  ay1 >>= y1;
  trace_off();

  double y1x1Derivative = -std::sin(x1) * std::sin(x2);
  double y1x2Derivative = std::cos(x1) * std::cos(x2);

  double y1x1x1Derivative = -std::cos(x1) * std::sin(x2);
  double y1x1x2Derivative = -std::sin(x1) * std::cos(x2);
  double y1x2x1Derivative = -std::sin(x1) * cos(x2);
  double y1x2x2Derivative = -std::cos(x1) * std::sin(x2);

  std::vector<double> x(2);
  std::vector<double> xd(2);
  std::vector<double> y(1);
  std::vector<double> yd(1);

  x[0] = 1.3;
  x[1] = 3.1;
  xd[0] = 1.;
  xd[1] = 0.;

  fos_forward(tapeId, 1, 2, 2, x, xd, y, yd);

  std::array<double, 2 * 1> UCont;
  auto U = MatrixView<2, 1, double>(UCont);
  Tensor<double> Z(2, 2, 2);

  std::array<short int, 2 * 2> nzCont;
  auto nz = MatrixView<2, 2, short int>(nzCont);

  nz[0][0] = 4;
  nz[0][1] = 4;
  nz[1][0] = 4;
  nz[1][1] = 4;

  U[0][0] = 1.;
  U[1][0] = 5.;

  hov_reverse(tapeId, 1, 2, 1, 2, U, Z, nz);

  BOOST_TEST(Z[0][0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[0][1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[0][0][1] == y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[0][1][1] == y1x1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0][0] == 5. * y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1][0] == 5. * y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0][1] == 5. * y1x1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1][1] == 5. * y1x1x2Derivative, tt::tolerance(tol));

  xd[0] = 0.;
  xd[1] = 1.;

  fos_forward(tapeId, 1, 2, 2, x, xd, y, yd);

  hov_reverse(tapeId, 1, 2, 1, 2, U, Z, nz);

  BOOST_TEST(Z[0][0][0] == y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[0][1][0] == y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[0][0][1] == y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[0][1][1] == y1x2x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0][0] == 5. * y1x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1][0] == 5. * y1x2Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][0][1] == 5. * y1x2x1Derivative, tt::tolerance(tol));
  BOOST_TEST(Z[1][1][1] == 5. * y1x2x2Derivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_SUITE_END()

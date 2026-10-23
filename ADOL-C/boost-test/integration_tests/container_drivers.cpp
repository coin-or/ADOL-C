#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "../const.h"
#include <adolc/adolc.h>
#include <adolc/drivers/odedrivers.h>
#include <adolc/drivers/taylor.h>
#include <array>
#include <cmath>

namespace tt = boost::test_tools;

BOOST_AUTO_TEST_SUITE(container_drivers)

BOOST_AUTO_TEST_CASE(ODE_Cache_Reused_After_Degree_Growth) {
  const short tag = createNewTape();
  trace_on(tag);
  {
    adouble x, y;
    x <<= 2.0;
    y = x * x;
    double value;
    y >>= value;
  }
  trace_off();

  // x' = x^2, x(0) = 2 has Taylor coefficients 2^(k+1).
  // Grow the retained buffers, then request a smaller degree again.
  for (int degree : {2, 4, 2}) {
    Matrix<double> coefficients(1, degree + 1);
    coefficients[0][0] = 2.0;
    BOOST_REQUIRE(forodec(tag, 1, 1.0, 0, degree, coefficients.data()) >= 0);
    for (int k = 0; k <= degree; ++k)
      BOOST_TEST(coefficients[0][k] == std::pow(2.0, k + 1),
                 tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(Jacobian_Solve_Reuses_And_Refreshes_Factors) {
  const short tag = createNewTape();
  trace_on(tag);
  {
    adouble x0, x1, y0, y1;
    x0 <<= 2.0;
    x1 <<= 1.0;
    y0 = x0 * x0 + x1;
    y1 = 3.0 * x0 + 2.0 * x1 * x1;
    double value;
    y0 >>= value;
    y1 >>= value;
  }
  trace_off();

  // J(x) = [[2*x0, 1], [3, 4*x1]]. Reuse it for a new right-hand side,
  // then change x so the cached factorization must be refreshed.
  const std::array<std::array<double, 2>, 3> points{
      {{2.0, 1.0}, {2.0, 1.0}, {3.0, 2.0}}};
  std::array<std::array<double, 2>, 3> rhs{
      {{2.0, -5.0}, {9.0, 10.0}, {4.0, -13.0}}};
  const std::array<std::array<double, 2>, 3> expected{
      {{1.0, -2.0}, {2.0, 1.0}, {1.0, -2.0}}};
  BOOST_REQUIRE(jac_solv(tag, 2, points[0].data(), rhs[0].data(), 1) >= 0);
  for (size_t i = 0; i < points.size(); ++i) {
    BOOST_REQUIRE(jac_solv(tag, 2, points[i].data(), rhs[i].data(), 2) >= 0);
    for (size_t j = 0; j < 2; ++j)
      BOOST_TEST(rhs[i][j] == expected[i][j], tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(Directional_Gradient_With_No_Switches) {
  const short tag = createNewTape();
  setCurrentTape(tag);
  currentTape().enableMinMaxUsingAbs();
  trace_on(tag);
  {
    adouble x0, x1, y;
    x0 <<= 2.0;
    x1 <<= 3.0;
    y = x0 * x0 + x0 * x1 + x1 * x1;
    double value;
    y >>= value;
  }
  trace_off();

  const std::array<double, 2> point{2.0, 3.0}, direction{1.0, -1.0};
  std::array<double, 2> gradient{};
  BOOST_REQUIRE(directional_active_gradient(tag, 2, point.data(),
                                            direction.data(), gradient.data(),
                                            nullptr) >= 0);
  BOOST_TEST(gradient[0] == 7.0, tt::tolerance(tol));
  BOOST_TEST(gradient[1] == 8.0, tt::tolerance(tol));
}

BOOST_AUTO_TEST_SUITE_END()

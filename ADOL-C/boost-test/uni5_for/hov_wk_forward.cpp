
/*
File for explicit testing functions from uni5_for.cpp file.
*/

#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

#include <adolc/adolc.h>

#include "../const.h"

#include <array>
#include <numeric>
#include <vector>

BOOST_AUTO_TEST_SUITE(test_hov_wk_forward)
BOOST_AUTO_TEST_CASE(FmaxOperator_HOV_WK_Forward) {
  const int16_t tag = 0;
  const size_t dim_out = 1;
  const size_t dim_in = 2;
  const size_t degree = 2;
  const size_t num_dirs = 3;
  const short keep = 1;
  std::vector<double> in{4.0, 3.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  trace_on(tag);
  std::for_each(in.begin(), in.end(), [&, i = 0](int value) mutable {
    indep[i] <<= in[i];
    i++;
  });

  // max(x^2, y^3)
  adouble dep = fmax(pow(indep[0], 2), pow(indep[1], 3));

  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree);
  double ***Y = myalloc3(dim_out, num_dirs, degree);

  X[0][0][0] = 1.0;
  X[1][0][0] = 1.9;

  X[0][1][0] = 2.0;
  X[1][1][0] = 3.0;

  X[0][2][0] = 1.0;
  X[1][2][0] = -1.0;

  X[0][0][1] = -1.0;
  X[1][0][1] = 1.0;

  X[0][1][1] = -2.0;
  X[1][1][1] = -3.0;

  X[0][2][1] = -1.0;
  X[1][2][1] = 1.0;

  /****************************
  TEST A < B
  *****************************/
  std::vector<double> test_in{4.0, 3.2};
  // max(x^2, y^3)
  double test_out = std::fmax(std::pow(test_in[0], 2), std::pow(test_in[1], 3));

  hov_wk_forward(tag, dim_out, dim_in, degree, keep, num_dirs, test_in.data(),
                 X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));
  BOOST_TEST(Y[0][0][0] == 3 * std::pow(test_in[1], 2) * X[1][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] == 3 * std::pow(test_in[1], 2) * X[1][1][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][2][0] == 3 * std::pow(test_in[1], 2) * X[1][2][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][0][1] ==
                 1.0 / 2.0 * 6.0 * test_in[1] * X[1][0][0] * X[1][0][0] +
                     3.0 * std::pow(test_in[1], 2.0) * X[1][0][1],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] ==
                 1.0 / 2.0 * 6.0 * test_in[1] * X[1][1][0] * X[1][1][0] +
                     3.0 * std::pow(test_in[1], 2.0) * X[1][1][1],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][2][1] ==
                 1.0 / 2.0 * 6.0 * test_in[1] * X[1][2][0] * X[1][2][0] +
                     3.0 * std::pow(test_in[1], 2.0) * X[1][2][1],
             tt::tolerance(tol));

  /**************************
  TEST A > B
  ***************************/

  // test_in[0] = 4.0
  test_in[1] = 1.0;
  // max(x^2, y^3)
  test_out = std::fmax(std::pow(test_in[0], 2), std::pow(test_in[1], 3));
  hov_wk_forward(tag, dim_out, dim_in, degree, keep, num_dirs, test_in.data(),
                 X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));
  BOOST_TEST(Y[0][0][0] == 2 * test_in[0] * X[0][0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] == 2 * test_in[0] * X[0][1][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][2][0] == 2 * test_in[0] * X[0][2][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][0][1] == 1.0 / 2.0 * 2.0 * X[0][0][0] * X[0][0][0] +
                               2.0 * test_in[0] * X[0][0][1],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] == 1.0 / 2.0 * 2.0 * X[0][1][0] * X[0][1][0] +
                               2.0 * test_in[0] * X[0][1][1],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][2][1] == 1.0 / 2.0 * 2.0 * X[0][2][0] * X[0][2][0] +
                               2.0 * test_in[0] * X[0][2][1],
             tt::tolerance(tol));

  /**************************
    TEST A == B
  ***************************/

  test_in[0] = 1.0;
  test_in[1] = 1.0;

  // max(x^2, y^3)
  test_out = std::fmax(std::pow(test_in[0], 2), std::pow(test_in[1], 3));

  // A < B
  X[0][0][0] = 1.0;
  X[1][0][0] = 1.9;

  // A > B
  X[0][1][0] = 3.0;
  X[1][1][0] = 2.0;

  // A == B
  X[0][2][0] = 1.0;
  X[1][2][0] = 1.0;

  X[0][0][1] = -1.0;
  X[1][0][1] = 1.0;

  X[0][1][1] = -2.0;
  X[1][1][1] = -3.0;

  // A < B
  X[0][2][1] = 1.0;
  X[1][2][1] = 2.0;

  hov_wk_forward(tag, dim_out, dim_in, degree, keep, num_dirs, test_in.data(),
                 X, out.data(), Y);
  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));

  // A < B
  BOOST_TEST(Y[0][0][0] == 3 * std::pow(test_in[1], 2) * X[1][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][0][1] ==
                 1.0 / 2.0 * 6.0 * test_in[1] * X[1][0][0] * X[1][0][0] +
                     3.0 * std::pow(test_in[1], 2.0) * X[1][0][1],
             tt::tolerance(tol));

  // A < B
  BOOST_TEST(Y[0][1][0] == 2 * test_in[0] * X[0][1][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] == 1.0 / 2.0 * 2.0 * X[0][1][0] * X[0][1][0] +
                               2.0 * test_in[0] * X[0][1][1],
             tt::tolerance(tol));

  // A == B, A < B
  BOOST_TEST(Y[0][2][0] == 3 * std::pow(test_in[1], 2) * X[1][2][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][2][1] ==
                 1.0 / 2.0 * 6.0 * test_in[1] * X[1][2][0] * X[1][2][0] +
                     3.0 * std::pow(test_in[1], 2.0) * X[1][2][1],
             tt::tolerance(tol));

  myfree3(X);
  myfree3(Y);
}

BOOST_AUTO_TEST_CASE(FminOperator_HOV_WK_Forward) {
  const int16_t tag = 0;
  const size_t dim_out = 1;
  const size_t dim_in = 2;
  const size_t degree = 2;
  const size_t num_dirs = 3;
  const short keep = 1;
  std::vector<double> in{4.0, 3.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  trace_on(tag);
  std::for_each(in.begin(), in.end(), [&, i = 0](int value) mutable {
    indep[i] <<= in[i];
    i++;
  });

  // max(x^2, y^3)
  adouble dep = fmin(-pow(indep[0], 2), -pow(indep[1], 3));

  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree);
  double ***Y = myalloc3(dim_out, num_dirs, degree);

  X[0][0][0] = 1.0;
  X[1][0][0] = 1.9;

  X[0][1][0] = 2.0;
  X[1][1][0] = 3.0;

  X[0][2][0] = 1.0;
  X[1][2][0] = -1.0;

  X[0][0][1] = -1.0;
  X[1][0][1] = 1.0;

  X[0][1][1] = -2.0;
  X[1][1][1] = -3.0;

  X[0][2][1] = -1.0;
  X[1][2][1] = 1.0;

  /****************************
  TEST A < B
  *****************************/
  std::vector<double> test_in{4.0, 3.2};
  // max(x^2, y^3)
  double test_out =
      std::fmin(-std::pow(test_in[0], 2), -std::pow(test_in[1], 3));

  hov_wk_forward(tag, dim_out, dim_in, degree, keep, num_dirs, test_in.data(),
                 X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));
  BOOST_TEST(Y[0][0][0] == -3 * std::pow(test_in[1], 2) * X[1][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] == -3 * std::pow(test_in[1], 2) * X[1][1][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][2][0] == -3 * std::pow(test_in[1], 2) * X[1][2][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][0][1] ==
                 -1.0 / 2.0 * 6.0 * test_in[1] * X[1][0][0] * X[1][0][0] -
                     3.0 * std::pow(test_in[1], 2.0) * X[1][0][1],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] ==
                 -1.0 / 2.0 * 6.0 * test_in[1] * X[1][1][0] * X[1][1][0] -
                     3.0 * std::pow(test_in[1], 2.0) * X[1][1][1],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][2][1] ==
                 -1.0 / 2.0 * 6.0 * test_in[1] * X[1][2][0] * X[1][2][0] -
                     3.0 * std::pow(test_in[1], 2.0) * X[1][2][1],
             tt::tolerance(tol));

  /**************************
  TEST A > B
  ***************************/

  // test_in[0] = 4.0
  test_in[1] = 1.0;
  // max(x^2, y^3)
  test_out = std::fmin(-std::pow(test_in[0], 2), -std::pow(test_in[1], 3));
  hov_wk_forward(tag, dim_out, dim_in, degree, keep, num_dirs, test_in.data(),
                 X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));
  BOOST_TEST(Y[0][0][0] == -2 * test_in[0] * X[0][0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] == -2 * test_in[0] * X[0][1][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][2][0] == -2 * test_in[0] * X[0][2][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][0][1] == -1.0 / 2.0 * 2.0 * X[0][0][0] * X[0][0][0] -
                               2.0 * test_in[0] * X[0][0][1],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] == -1.0 / 2.0 * 2.0 * X[0][1][0] * X[0][1][0] -
                               2.0 * test_in[0] * X[0][1][1],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][2][1] == -1.0 / 2.0 * 2.0 * X[0][2][0] * X[0][2][0] -
                               2.0 * test_in[0] * X[0][2][1],
             tt::tolerance(tol));

  /**************************
    TEST A == B
  ***************************/

  test_in[0] = 1.0;
  test_in[1] = 1.0;

  // max(x^2, y^3)
  test_out = std::fmin(-std::pow(test_in[0], 2), -std::pow(test_in[1], 3));

  // A < B
  X[0][0][0] = 1.0;
  X[1][0][0] = 1.9;

  // A > B
  X[0][1][0] = 3.0;
  X[1][1][0] = 2.0;

  // A == B
  X[0][2][0] = 1.0;
  X[1][2][0] = 1.0;

  X[0][0][1] = -1.0;
  X[1][0][1] = 1.0;

  X[0][1][1] = -2.0;
  X[1][1][1] = -3.0;

  // A < B
  X[0][2][1] = 1.0;
  X[1][2][1] = 2.0;

  hov_wk_forward(tag, dim_out, dim_in, degree, keep, num_dirs, test_in.data(),
                 X, out.data(), Y);
  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));

  // A < B
  BOOST_TEST(Y[0][0][0] == -3 * std::pow(test_in[1], 2) * X[1][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][0][1] ==
                 -1.0 / 2.0 * 6.0 * test_in[1] * X[1][0][0] * X[1][0][0] -
                     3.0 * std::pow(test_in[1], 2.0) * X[1][0][1],
             tt::tolerance(tol));

  // A < B
  BOOST_TEST(Y[0][1][0] == -2 * test_in[0] * X[0][1][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] == -1.0 / 2.0 * 2.0 * X[0][1][0] * X[0][1][0] -
                               2.0 * test_in[0] * X[0][1][1],
             tt::tolerance(tol));

  // A == B, A < B
  BOOST_TEST(Y[0][2][0] == -3 * std::pow(test_in[1], 2) * X[1][2][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][2][1] ==
                 -1.0 / 2.0 * 6.0 * test_in[1] * X[1][2][0] * X[1][2][0] -
                     3.0 * std::pow(test_in[1], 2.0) * X[1][2][1],
             tt::tolerance(tol));

  myfree3(X);
  myfree3(Y);
}

BOOST_AUTO_TEST_CASE(ExpOperator_HOV_WK_FORWARD) {
  const int16_t tag = 0;
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{4.0};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // exp(x1^2)
  trace_on(tag);
  indep[0] <<= in[0];
  adouble dep = exp(pow(indep[0], 2));
  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree);
  double ***Y = myalloc3(dim_out, num_dirs, degree);

  X[0][0][0] = 1.0;
  X[0][0][1] = -1.0;
  X[0][0][2] = 1.0;

  X[0][1][0] = 2.0;
  X[0][1][1] = -2.0;

  X[0][1][2] = -2.0;

  std::vector<double> test_in{4.0};

  // exp(x1^2)
  double test_out = std::exp(std::pow(in[0], 2));

  hov_wk_forward(tag, dim_out, dim_in, degree, keep, num_dirs, test_in.data(),
                 X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));

  // first derivative
  BOOST_TEST(Y[0][0][0] == 2.0 * std::exp(std::pow(test_in[0], 2)) *
                               test_in[0] * X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] == 2.0 * std::exp(std::pow(test_in[0], 2)) *
                               test_in[0] * X[0][1][0],
             tt::tolerance(tol));
  // second derivative
  BOOST_TEST(Y[0][0][1] == 2.0 * std::exp(std::pow(test_in[0], 2)) *
                                   test_in[0] * X[0][0][1] +
                               1.0 / 2.0 *
                                   (4 * std::exp(std::pow(test_in[0], 2)) *
                                        std::pow(test_in[0], 2) +
                                    2 * std::exp(std::pow(test_in[0], 2))) *
                                   std::pow(X[0][0][0], 2),
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] == (2.0 * std::exp(std::pow(test_in[0], 2)) *
                                test_in[0] * X[0][1][1] +
                            1.0 / 2.0 *
                                (4 * std::exp(std::pow(test_in[0], 2)) *
                                     std::pow(test_in[0], 2) +
                                 2 * std::exp(std::pow(test_in[0], 2))) *
                                std::pow(X[0][1][0], 2)),
             tt::tolerance(tol));

  // third derivative
  BOOST_TEST(
      Y[0][0][2] ==
          2.0 * std::exp(std::pow(test_in[0], 2)) * test_in[0] * X[0][0][2] +
              (4 * std::exp(std::pow(test_in[0], 2)) * std::pow(test_in[0], 2) +
               2 * std::exp(std::pow(test_in[0], 2))) *
                  X[0][0][1] * X[0][0][0] +
              1.0 / 6.0 *
                  (8 * std::exp(std::pow(test_in[0], 2)) *
                       std::pow(test_in[0], 3) +
                   12 * std::exp(std::pow(test_in[0], 2)) * test_in[0]) *
                  std::pow(X[0][0][0], 3),
      tt::tolerance(tol));

  BOOST_TEST(
      Y[0][1][2] ==
          2.0 * std::exp(std::pow(test_in[0], 2)) * test_in[0] * X[0][1][2] +
              (4 * std::exp(std::pow(test_in[0], 2)) * std::pow(test_in[0], 2) +
               2 * std::exp(std::pow(test_in[0], 2))) *
                  X[0][1][1] * X[0][1][0] +
              1.0 / 6.0 *
                  (8 * std::exp(std::pow(test_in[0], 2)) *
                       std::pow(test_in[0], 3) +
                   12 * std::exp(std::pow(test_in[0], 2)) * test_in[0]) *
                  std::pow(X[0][1][0], 3),
      tt::tolerance(tol));
  myfree3(X);
  myfree3(Y);
}
BOOST_AUTO_TEST_CASE(MultOperator_HOV_Forward) {
  const int16_t tag = 0;
  const size_t dim_out = 1;
  const size_t dim_in = 2;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{-1.0, 1.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // x1^2 * x2^3
  trace_on(tag);
  indep[0] <<= in[0];
  indep[1] <<= in[1];
  adouble dep = pow(indep[0], 2) * pow(indep[1], 3);
  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree);
  double ***Y = myalloc3(dim_out, num_dirs, degree);

  X[0][0][0] = 1.0;
  X[0][0][1] = -1.0;
  X[0][0][2] = 1.0;

  X[1][0][0] = 1.1;
  X[1][0][1] = -1.1;
  X[1][0][2] = 1.1;

  X[0][1][0] = 2.0;
  X[0][1][1] = -2.0;
  X[0][1][2] = -2.0;

  X[1][1][0] = 2.1;
  X[1][1][1] = -2.1;
  X[1][1][2] = -2.1;

  std::vector<double> test_in{-1.0, 1.5};

  // x1^2 * x2^3
  double test_out = std::pow(test_in[0], 2) * std::pow(test_in[1], 3);

  hov_wk_forward(tag, dim_out, dim_in, degree, keep, num_dirs, test_in.data(),
                 X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));

  // first derivative
  BOOST_TEST(Y[0][0][0] ==
                 2.0 * test_in[0] * std::pow(test_in[1], 3) * X[0][0][0] +
                     3.0 * std::pow(test_in[1], 2) * std::pow(test_in[0], 2) *
                         X[1][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] ==
                 2.0 * test_in[0] * std::pow(test_in[1], 3) * X[0][1][0] +
                     3.0 * std::pow(test_in[1], 2) * std::pow(test_in[0], 2) *
                         X[1][1][0],
             tt::tolerance(tol));

  // second derivative
  BOOST_TEST(
      Y[0][0][1] ==
          2.0 * test_in[0] * std::pow(test_in[1], 3) * X[0][0][1] +
              3.0 * std::pow(test_in[1], 2) * std::pow(test_in[0], 2) *
                  X[1][0][1] +
              1.0 / 2.0 *
                  (2.0 * std::pow(test_in[1], 3) * std::pow(X[0][0][1], 2) +
                   12.0 * test_in[0] * std::pow(test_in[1], 2) * X[0][0][1] *
                       X[1][0][1] +
                   6.0 * test_in[1] * std::pow(test_in[0], 2) *
                       std::pow(X[1][0][1], 2)),
      tt::tolerance(tol));
  BOOST_TEST(
      Y[0][1][1] ==
          2.0 * test_in[0] * std::pow(test_in[1], 3) * X[0][1][1] +
              3.0 * std::pow(test_in[1], 2) * std::pow(test_in[0], 2) *
                  X[1][1][1] +
              1.0 / 2.0 *
                  (2.0 * std::pow(test_in[1], 3) * std::pow(X[0][1][0], 2) +
                   12.0 * test_in[0] * std::pow(test_in[1], 2) * X[0][1][0] *
                       X[1][1][0] +
                   6.0 * test_in[1] * std::pow(test_in[0], 2) *
                       std::pow(X[1][1][0], 2)),
      tt::tolerance(tol));

  // third derivative
  BOOST_TEST(Y[0][1][2] ==
                 2.0 * test_in[0] * std::pow(test_in[1], 3) * X[0][1][2] +
                     3.0 * std::pow(test_in[1], 2) * std::pow(test_in[0], 2) *
                         X[1][1][2] +
                     (2.0 * std::pow(test_in[1], 3) * X[0][1][0] +
                      6.0 * test_in[0] * std::pow(test_in[1], 2) * X[1][1][0]) *
                         X[0][1][1] +
                     (6.0 * test_in[0] * std::pow(test_in[1], 2) * X[0][1][0] +
                      6.0 * test_in[1] * std::pow(test_in[0], 2) * X[1][1][0]) *
                         X[1][1][1] +
                     1.0 / 6.0 *
                         (18.0 * std::pow(test_in[1], 2) * X[1][1][0] *
                              X[0][1][0] * X[0][1][0] +
                          36.0 * test_in[0] * test_in[1] * X[1][1][0] *
                              X[1][1][0] * X[0][1][0] +
                          6.0 * std::pow(test_in[0], 2) * X[1][1][0] *
                              X[1][1][0] * X[1][1][0]),
             tt::tolerance(tol));
  BOOST_TEST(Y[0][0][2] ==
                 2.0 * test_in[0] * std::pow(test_in[1], 3) * X[0][0][2] +
                     3.0 * std::pow(test_in[1], 2) * std::pow(test_in[0], 2) *
                         X[1][0][2] +
                     (2.0 * std::pow(test_in[1], 3) * X[0][0][0] +
                      6.0 * test_in[0] * std::pow(test_in[1], 2) * X[1][0][0]) *
                         X[0][0][1] +
                     (6.0 * test_in[0] * std::pow(test_in[1], 2) * X[0][0][0] +
                      6.0 * test_in[1] * std::pow(test_in[0], 2) * X[1][0][0]) *
                         X[1][0][1] +
                     1.0 / 6.0 *
                         (18.0 * std::pow(test_in[1], 2) * X[1][0][0] *
                              X[0][0][0] * X[0][0][0] +
                          36.0 * test_in[0] * test_in[1] * X[1][0][0] *
                              X[1][0][0] * X[0][0][0] +
                          6.0 * std::pow(test_in[0], 2) * X[1][0][0] *
                              X[1][0][0] * X[1][0][0]),
             tt::tolerance(tol));
  myfree3(X);
  myfree3(Y);
}

BOOST_AUTO_TEST_CASE(AddOperator_HOV_Forward) {
  const int16_t tag = 0;
  const size_t dim_out = 1;
  const size_t dim_in = 2;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{-1.0, 1.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // x1^2 + x2^3
  trace_on(tag);
  indep[0] <<= in[0];
  indep[1] <<= in[1];
  adouble dep = pow(indep[0], 2) + pow(indep[1], 3);
  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree);
  double ***Y = myalloc3(dim_out, num_dirs, degree);

  X[0][0][0] = 1.0;
  X[0][0][1] = -1.0;
  X[0][0][2] = 1.0;

  X[1][0][0] = 1.1;
  X[1][0][1] = -1.1;
  X[1][0][2] = 1.1;

  X[0][1][0] = 2.0;
  X[0][1][1] = -2.0;
  X[0][1][2] = -2.0;

  X[1][1][0] = 2.1;
  X[1][1][1] = -2.1;
  X[1][1][2] = -2.1;

  std::vector<double> test_in{-1.0, 1.5};

  // x1^2 + x2^3
  double test_out = std::pow(test_in[0], 2) + std::pow(test_in[1], 3);

  hov_wk_forward(tag, dim_out, dim_in, degree, keep, num_dirs, test_in.data(),
                 X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));

  // first derivative
  BOOST_TEST(Y[0][0][0] == 2.0 * test_in[0] * X[0][0][0] +
                               3.0 * std::pow(test_in[1], 2) * X[1][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] == 2.0 * test_in[0] * X[0][1][0] +
                               3.0 * std::pow(test_in[1], 2) * X[1][1][0],
             tt::tolerance(tol));

  // second derivative
  BOOST_TEST(Y[0][0][1] == 2.0 * test_in[0] * X[0][0][1] +
                               3.0 * std::pow(test_in[1], 2) * X[1][0][1] +
                               1.0 / 2.0 *
                                   (2.0 * X[0][0][0] * X[0][0][0] +
                                    6.0 * test_in[1] * X[1][0][0] * X[1][0][0]),
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] == 2.0 * test_in[0] * X[0][1][1] +
                               3.0 * std::pow(test_in[1], 2) * X[1][1][1] +
                               1.0 / 2.0 *
                                   (2.0 * X[0][1][0] * X[0][1][0] +
                                    6.0 * test_in[1] * X[1][1][0] * X[1][1][0]),
             tt::tolerance(tol));

  // third derivative
  BOOST_TEST(Y[0][0][2] ==
                 2.0 * test_in[0] * X[0][0][2] +
                     3.0 * std::pow(test_in[1], 2) * X[1][0][2] +
                     2.0 * X[0][0][0] * X[0][0][1] +
                     6.0 * test_in[1] * X[1][0][0] * X[1][0][1] +
                     1.0 / 6.0 * 6.0 * X[1][0][0] * X[1][0][0] * X[1][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][2] ==
                 2.0 * test_in[0] * X[0][1][2] +
                     3.0 * std::pow(test_in[1], 2) * X[1][1][2] +
                     2.0 * X[0][1][0] * X[0][1][1] +
                     6.0 * test_in[1] * X[1][1][0] * X[1][1][1] +
                     1.0 / 6.0 * 6.0 * X[1][1][0] * X[1][1][0] * X[1][1][0],
             tt::tolerance(tol));
  myfree3(X);
  myfree3(Y);
}

BOOST_AUTO_TEST_CASE(SubOperator_HOV_Forward) {
  const int16_t tag = 0;
  const size_t dim_out = 1;
  const size_t dim_in = 2;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{-1.0, 1.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // x1^2 - x2^3
  trace_on(tag);
  indep[0] <<= in[0];
  indep[1] <<= in[1];
  adouble dep = pow(indep[0], 2) - pow(indep[1], 3);
  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree);
  double ***Y = myalloc3(dim_out, num_dirs, degree);

  X[0][0][0] = 1.0;
  X[0][0][1] = -1.0;
  X[0][0][2] = 1.0;

  X[1][0][0] = 1.1;
  X[1][0][1] = -1.1;
  X[1][0][2] = 1.1;

  X[0][1][0] = 2.0;
  X[0][1][1] = -2.0;
  X[0][1][2] = -2.0;

  X[1][1][0] = 2.1;
  X[1][1][1] = -2.1;
  X[1][1][2] = -2.1;

  std::vector<double> test_in{-1.0, 1.5};

  // x1^2 - x2^3
  double test_out = std::pow(test_in[0], 2) - std::pow(test_in[1], 3);

  hov_wk_forward(tag, dim_out, dim_in, degree, keep, num_dirs, test_in.data(),
                 X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));

  // first derivative
  BOOST_TEST(Y[0][0][0] == 2.0 * test_in[0] * X[0][0][0] -
                               3.0 * std::pow(test_in[1], 2) * X[1][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] == 2.0 * test_in[0] * X[0][1][0] -
                               3.0 * std::pow(test_in[1], 2) * X[1][1][0],
             tt::tolerance(tol));

  // second derivative
  BOOST_TEST(Y[0][0][1] == 2.0 * test_in[0] * X[0][0][1] -
                               3.0 * std::pow(test_in[1], 2) * X[1][0][1] +
                               1.0 / 2.0 *
                                   (2.0 * X[0][0][0] * X[0][0][0] -
                                    6.0 * test_in[1] * X[1][0][0] * X[1][0][0]),
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] == 2.0 * test_in[0] * X[0][1][1] -
                               3.0 * std::pow(test_in[1], 2) * X[1][1][1] +
                               1.0 / 2.0 *
                                   (2.0 * X[0][1][0] * X[0][1][0] -
                                    6.0 * test_in[1] * X[1][1][0] * X[1][1][0]),
             tt::tolerance(tol));

  // third derivative
  BOOST_TEST(Y[0][0][2] ==
                 2.0 * test_in[0] * X[0][0][2] -
                     3.0 * std::pow(test_in[1], 2) * X[1][0][2] +
                     2.0 * X[0][0][0] * X[0][0][1] -
                     6.0 * test_in[1] * X[1][0][0] * X[1][0][1] -
                     1.0 / 6.0 * 6.0 * X[1][0][0] * X[1][0][0] * X[1][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][2] ==
                 2.0 * test_in[0] * X[0][1][2] -
                     3.0 * std::pow(test_in[1], 2) * X[1][1][2] +
                     2.0 * X[0][1][0] * X[0][1][1] -
                     6.0 * test_in[1] * X[1][1][0] * X[1][1][1] -
                     1.0 / 6.0 * 6.0 * X[1][1][0] * X[1][1][0] * X[1][1][0],
             tt::tolerance(tol));
  myfree3(X);
  myfree3(Y);
}

BOOST_AUTO_TEST_CASE(DivOperator_HOV_Forward) {
  const int16_t tag = 0;
  const size_t dim_out = 1;
  const size_t dim_in = 2;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{-1.0, 1.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // x1^2 / x2^3
  trace_on(tag);
  indep[0] <<= in[0];
  indep[1] <<= in[1];
  adouble dep = pow(indep[0], 2) / pow(indep[1], 3);
  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree);
  double ***Y = myalloc3(dim_out, num_dirs, degree);

  X[0][0][0] = 1.0;
  X[0][0][1] = -1.0;
  X[0][0][2] = 1.0;

  X[1][0][0] = 1.1;
  X[1][0][1] = -1.1;
  X[1][0][2] = 1.1;

  X[0][1][0] = 2.0;
  X[0][1][1] = -2.0;
  X[0][1][2] = -2.0;

  X[1][1][0] = 2.1;
  X[1][1][1] = -2.1;
  X[1][1][2] = -2.1;

  std::vector<double> test_in{-1.0, 1.5};

  // x1^2 - x2^3
  double test_out = std::pow(test_in[0], 2) / std::pow(test_in[1], 3);

  hov_wk_forward(tag, dim_out, dim_in, degree, keep, num_dirs, test_in.data(),
                 X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));

  // first derivative
  BOOST_TEST(Y[0][0][0] ==
                 2.0 * test_in[0] / std::pow(test_in[1], 3) * X[0][0][0] -
                     3.0 * std::pow(test_in[0], 2) / std::pow(test_in[1], 4) *
                         X[1][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] ==
                 2.0 * test_in[0] / std::pow(test_in[1], 3) * X[0][1][0] -
                     3.0 * std::pow(test_in[0], 2) / std::pow(test_in[1], 4) *
                         X[1][1][0],
             tt::tolerance(tol));

  // second derivative
  BOOST_TEST(
      Y[0][0][1] ==
          2.0 * test_in[0] / std::pow(test_in[1], 3) * X[0][0][1] -
              3.0 * std::pow(test_in[0], 2) / std::pow(test_in[1], 4) *
                  X[1][0][1] +
              1.0 / 2.0 *
                  ((2.0 / std::pow(test_in[1], 3) * X[0][0][0] -
                    6 * test_in[0] / std::pow(test_in[1], 4) * X[1][0][0]) *
                       X[0][0][0] +
                   (-6 * test_in[0] / std::pow(test_in[1], 4) * X[0][0][0] +
                    12.0 * std::pow(test_in[0], 2) / std::pow(test_in[1], 5) *
                        X[1][0][0]) *
                       X[1][0][0]),
      tt::tolerance(tol));
  BOOST_TEST(
      Y[0][1][1] ==
          2.0 * test_in[0] / std::pow(test_in[1], 3) * X[0][1][1] -
              3.0 * std::pow(test_in[0], 2) / std::pow(test_in[1], 4) *
                  X[1][1][1] +
              1.0 / 2.0 *
                  ((2.0 / std::pow(test_in[1], 3) * X[0][1][0] -
                    6 * test_in[0] / std::pow(test_in[1], 4) * X[1][1][0]) *
                       X[0][1][0] +
                   (-6 * test_in[0] / std::pow(test_in[1], 4) * X[0][1][0] +
                    12.0 * std::pow(test_in[0], 2) / std::pow(test_in[1], 5) *
                        X[1][1][0]) *
                       X[1][1][0]),
      tt::tolerance(tol));

  // third derivative
  BOOST_TEST(Y[0][0][2] ==
                 2.0 * test_in[0] / std::pow(test_in[1], 3) * X[0][0][2] -
                     3.0 * std::pow(test_in[0], 2) / std::pow(test_in[1], 4) *
                         X[1][0][2] +
                     (2.0 / std::pow(test_in[1], 3) * X[0][0][0] -
                      6.0 * test_in[0] / std::pow(test_in[1], 4) * X[1][0][0]) *
                         X[0][0][1] +
                     (-6.0 * test_in[0] / std::pow(test_in[1], 4) * X[0][0][0] +
                      12.0 * std::pow(test_in[0], 2) / std::pow(test_in[1], 5) *
                          X[1][0][0]) *
                         X[1][0][1] +
                     1.0 / 6.0 *
                         (-18.0 / std::pow(test_in[1], 4) * X[0][0][0] *
                              X[0][0][0] * X[1][0][0] +
                          72.0 * test_in[0] / std::pow(test_in[1], 5) *
                              X[1][0][0] * X[1][0][0] * X[0][0][0] -
                          60.0 * std::pow(test_in[0], 2) /
                              std::pow(test_in[1], 6) * X[1][0][0] *
                              X[1][0][0] * X[1][0][0]),
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][2] ==
                 2.0 * test_in[0] / std::pow(test_in[1], 3) * X[0][1][2] -
                     3.0 * std::pow(test_in[0], 2) / std::pow(test_in[1], 4) *
                         X[1][1][2] +
                     (2.0 / std::pow(test_in[1], 3) * X[0][1][0] -
                      6.0 * test_in[0] / std::pow(test_in[1], 4) * X[1][1][0]) *
                         X[0][1][1] +
                     (-6.0 * test_in[0] / std::pow(test_in[1], 4) * X[0][1][0] +
                      12.0 * std::pow(test_in[0], 2) / std::pow(test_in[1], 5) *
                          X[1][1][0]) *
                         X[1][1][1] +
                     1.0 / 6.0 *
                         (-18.0 / std::pow(test_in[1], 4) * X[0][1][0] *
                              X[0][1][0] * X[1][1][0] +
                          72.0 * test_in[0] / std::pow(test_in[1], 5) *
                              X[1][1][0] * X[1][1][0] * X[0][1][0] -
                          60.0 * std::pow(test_in[0], 2) /
                              std::pow(test_in[1], 6) * X[1][1][0] *
                              X[1][1][0] * X[1][1][0]),
             tt::tolerance(tol));
  myfree3(X);
  myfree3(Y);
}
BOOST_AUTO_TEST_SUITE_END()

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

BOOST_AUTO_TEST_SUITE(test_hov_forward)
BOOST_AUTO_TEST_CASE(FmaxOperator_HOV_Forward) {
  const int16_t tag = 0;
  const size_t dim_out = 1;
  const size_t dim_in = 2;
  const size_t degree = 2;
  const size_t num_dirs = 3;
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

  hov_forward(tag, dim_out, dim_in, degree, num_dirs, test_in.data(), X,
              out.data(), Y);

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
  hov_forward(tag, dim_out, dim_in, degree, num_dirs, test_in.data(), X,
              out.data(), Y);

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

  hov_forward(tag, dim_out, dim_in, degree, num_dirs, test_in.data(), X,
              out.data(), Y);
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

BOOST_AUTO_TEST_CASE(FminOperator_HOV_Forward) {
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

  hov_forward(tag, dim_out, dim_in, degree, num_dirs, test_in.data(), X,
              out.data(), Y);

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
  hov_forward(tag, dim_out, dim_in, degree, num_dirs, test_in.data(), X,
              out.data(), Y);

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

  hov_forward(tag, dim_out, dim_in, degree, num_dirs, test_in.data(), X,
              out.data(), Y);
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
BOOST_AUTO_TEST_SUITE_END()
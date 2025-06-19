
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

const short tapeId10 = 10;
struct TapeInitializer {
  TapeInitializer() { createNewTape(tapeId10); }
};

BOOST_GLOBAL_FIXTURE(TapeInitializer);

BOOST_AUTO_TEST_CASE(FmaxOperator_HOV_WK_Forward) {
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 2;
  const size_t degree = 2;
  const size_t num_dirs = 3;
  const short keep = 1;
  std::vector<double> in{4.0, 3.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  trace_on(tapeId10);
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

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

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
  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

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

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);
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

BOOST_AUTO_TEST_CASE(MaxOperator_HOV_WK_Forward) {
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 2;
  const size_t degree = 2;
  const size_t num_dirs = 3;
  const short keep = 1;
  std::vector<double> in{4.0, 3.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  trace_on(tapeId10);
  std::for_each(in.begin(), in.end(), [&, i = 0](int value) mutable {
    indep[i] <<= in[i];
    i++;
  });

  // max(x^2, y^3)
  adouble dep = max(pow(indep[0], 2), pow(indep[1], 3));

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
  double test_out = std::max(std::pow(test_in[0], 2), std::pow(test_in[1], 3));

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

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
  test_out = std::max(std::pow(test_in[0], 2), std::pow(test_in[1], 3));
  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

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
  test_out = std::max(std::pow(test_in[0], 2), std::pow(test_in[1], 3));

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

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);
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
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 2;
  const size_t degree = 2;
  const size_t num_dirs = 3;
  const short keep = 1;
  std::vector<double> in{4.0, 3.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  trace_on(tapeId10);
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

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

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
  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

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

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);
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

BOOST_AUTO_TEST_CASE(MinOperator_HOV_WK_Forward) {
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 2;
  const size_t degree = 2;
  const size_t num_dirs = 3;
  const short keep = 1;
  std::vector<double> in{4.0, 3.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  trace_on(tapeId10);
  std::for_each(in.begin(), in.end(), [&, i = 0](int value) mutable {
    indep[i] <<= in[i];
    i++;
  });

  // max(x^2, y^3)
  adouble dep = min(-pow(indep[0], 2), -pow(indep[1], 3));

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
      std::min(-std::pow(test_in[0], 2), -std::pow(test_in[1], 3));

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

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
  test_out = std::min(-std::pow(test_in[0], 2), -std::pow(test_in[1], 3));
  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

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
  test_out = std::min(-std::pow(test_in[0], 2), -std::pow(test_in[1], 3));

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

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);
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
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{4.0};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // exp(x1^2)
  trace_on(tapeId10);
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

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

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
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 2;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{-1.0, 1.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // x1^2 * x2^3
  trace_on(tapeId10);
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

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

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
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 2;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{-1.0, 1.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // x1^2 + x2^3
  trace_on(tapeId10);
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

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

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
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 2;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{-1.0, 1.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // x1^2 - x2^3
  trace_on(tapeId10);
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

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

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
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 2;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{-1.0, 1.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // x1^2 / x2^3
  trace_on(tapeId10);
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

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

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

BOOST_AUTO_TEST_CASE(TanOperator_HOV_Forward) {
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{0.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // tan(x1^2)
  trace_on(tapeId10);
  indep[0] <<= in[0];
  adouble dep = tan(pow(indep[0], 2));
  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree);
  double ***Y = myalloc3(dim_out, num_dirs, degree);

  X[0][0][0] = 1.0;
  X[0][0][1] = -1.0;
  X[0][0][2] = 1.1;

  X[0][1][0] = 2.0;
  X[0][1][1] = -2.1;
  X[0][1][2] = -2.0;

  std::vector<double> test_in{0.5};

  // tan(x1^2)
  double test_out = std::tan(std::pow(test_in[0], 2));

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));

  // first derivative
  BOOST_TEST(Y[0][0][0] == 2.0 * test_in[0] /
                               std::pow(std::cos(std::pow(test_in[0], 2)), 2) *
                               X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] == 2.0 * test_in[0] /
                               std::pow(std::cos(std::pow(test_in[0], 2)), 2) *
                               X[0][1][0],
             tt::tolerance(tol));

  // second derivative
  BOOST_TEST(Y[0][0][1] ==
                 2.0 * test_in[0] /
                         std::pow(std::cos(std::pow(test_in[0], 2)), 2) *
                         X[0][0][1] +
                     1.0 / 2.0 *
                         (2.0 * std::pow(std::cos(std::pow(test_in[0], 2)), 2) +
                          8.0 * std::pow(test_in[0], 2) *
                              std::cos(std::pow(test_in[0], 2)) *
                              std::sin(std::pow(test_in[0], 2))) /
                         std::pow(std::cos(std::pow(test_in[0], 2)), 4) *
                         X[0][0][0] * X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] ==
                 2.0 * test_in[0] /
                         std::pow(std::cos(std::pow(test_in[0], 2)), 2) *
                         X[0][1][1] +
                     1.0 / 2.0 *
                         (2.0 * std::pow(std::cos(std::pow(test_in[0], 2)), 2) +
                          8.0 * std::pow(test_in[0], 2) *
                              std::cos(std::pow(test_in[0], 2)) *
                              std::sin(std::pow(test_in[0], 2))) /
                         std::pow(std::cos(std::pow(test_in[0], 2)), 4) *
                         X[0][1][0] * X[0][1][0],
             tt::tolerance(tol));

  // third derivative
  BOOST_TEST(
      Y[0][0][2] ==
          2.0 * test_in[0] / std::pow(std::cos(std::pow(test_in[0], 2)), 2) *
                  X[0][0][2] +
              (2.0 * std::pow(std::cos(std::pow(test_in[0], 2)), 2) +
               8.0 * std::pow(test_in[0], 2) *
                   std::cos(std::pow(test_in[0], 2)) *
                   std::sin(std::pow(test_in[0], 2))) /
                  std::pow(std::cos(std::pow(test_in[0], 2)), 4) * X[0][0][0] *
                  X[0][0][1] +
              1.0 / 6.0 *
                  (96.0 * test_in[0] * std::sin(2.0 * std::pow(test_in[0], 2)) *
                       std::pow(std::cos(std::pow(test_in[0], 2)), 2) /
                       std::pow(std::cos(2.0 * std::pow(test_in[0], 2)) + 1,
                                3) +
                   256.0 * std::pow(test_in[0], 3) *
                       std::pow(std::cos(std::pow(test_in[0], 2)), 4) /
                       std::pow(std::cos(2.0 * std::pow(test_in[0], 2)) + 1,
                                4) +
                   128.0 * std::pow(test_in[0], 3) *
                       std::pow(std::sin(2.0 * std::pow(test_in[0], 2)), 2) *
                       std::pow(std::cos(std::pow(test_in[0], 2)), 2) /
                       std::pow(std::cos(2.0 * std::pow(test_in[0], 2)) + 1,
                                4)) *
                  std::pow(X[0][0][0], 3),
      tt::tolerance(tol));
  BOOST_TEST(
      Y[0][1][2] ==
          2.0 * test_in[0] / std::pow(std::cos(std::pow(test_in[0], 2)), 2) *
                  X[0][1][2] +
              (2.0 * std::pow(std::cos(std::pow(test_in[0], 2)), 2) +
               8.0 * std::pow(test_in[0], 2) *
                   std::cos(std::pow(test_in[0], 2)) *
                   std::sin(std::pow(test_in[0], 2))) /
                  std::pow(std::cos(std::pow(test_in[0], 2)), 4) * X[0][1][0] *
                  X[0][1][1] +
              1.0 / 6.0 *
                  (96.0 * test_in[0] * std::sin(2.0 * std::pow(test_in[0], 2)) *
                       std::pow(std::cos(std::pow(test_in[0], 2)), 2) /
                       std::pow(std::cos(2.0 * std::pow(test_in[0], 2)) + 1,
                                3) +
                   256.0 * std::pow(test_in[0], 3) *
                       std::pow(std::cos(std::pow(test_in[0], 2)), 4) /
                       std::pow(std::cos(2.0 * std::pow(test_in[0], 2)) + 1,
                                4) +
                   128.0 * std::pow(test_in[0], 3) *
                       std::pow(std::sin(2.0 * std::pow(test_in[0], 2)), 2) *
                       std::pow(std::cos(std::pow(test_in[0], 2)), 2) /
                       std::pow(std::cos(2.0 * std::pow(test_in[0], 2)) + 1,
                                4)) *
                  std::pow(X[0][1][0], 3),
      tt::tolerance(tol));
  myfree3(X);
  myfree3(Y);
}

BOOST_AUTO_TEST_CASE(SinOperator_HOV_Forward) {
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{0.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // sin(x1^2)
  trace_on(tapeId10);
  indep[0] <<= in[0];
  adouble dep = sin(pow(indep[0], 2));
  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree);
  double ***Y = myalloc3(dim_out, num_dirs, degree);

  X[0][0][0] = 1.0;
  X[0][0][1] = -1.0;
  X[0][0][2] = 1.1;

  X[0][1][0] = 2.0;
  X[0][1][1] = -2.1;
  X[0][1][2] = -2.0;

  std::vector<double> test_in{0.5};

  // sin(x1^2)
  double test_out = std::sin(std::pow(test_in[0], 2));

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));

  // first derivative
  BOOST_TEST(Y[0][0][0] == 2.0 * std::cos(std::pow(test_in[0], 2)) *
                               test_in[0] * X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] == 2.0 * std::cos(std::pow(test_in[0], 2)) *
                               test_in[0] * X[0][1][0],
             tt::tolerance(tol));

  // second derivative
  BOOST_TEST(Y[0][0][1] == 2.0 * std::cos(std::pow(test_in[0], 2)) *
                                   test_in[0] * X[0][0][1] +
                               1.0 / 2.0 *
                                   (-4.0 * std::sin(std::pow(test_in[0], 2)) *
                                        std::pow(test_in[0], 2) +
                                    2.0 * std::cos(std::pow(test_in[0], 2))) *
                                   X[0][0][0] * X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] == 2.0 * std::cos(std::pow(test_in[0], 2)) *
                                   test_in[0] * X[0][1][1] +
                               1.0 / 2.0 *
                                   (-4.0 * std::sin(std::pow(test_in[0], 2)) *
                                        std::pow(test_in[0], 2) +
                                    2.0 * std::cos(std::pow(test_in[0], 2))) *
                                   X[0][1][0] * X[0][1][0],
             tt::tolerance(tol));

  // third derivative
  BOOST_TEST(
      Y[0][0][2] ==
          2.0 * std::cos(std::pow(test_in[0], 2)) * test_in[0] * X[0][0][2] +
              (-4.0 * std::sin(std::pow(test_in[0], 2)) *
                   std::pow(test_in[0], 2) +
               2.0 * std::cos(std::pow(test_in[0], 2))) *
                  X[0][0][0] * X[0][0][1] +
              1.0 / 6.0 *
                  (-12.0 * test_in[0] * std::sin(std::pow(test_in[0], 2)) -
                   8.0 * std::pow(test_in[0], 3) *
                       std::cos(std::pow(test_in[0], 2))) *
                  std::pow(X[0][0][0], 3),
      tt::tolerance(tol));
  BOOST_TEST(
      Y[0][1][2] ==
          2.0 * std::cos(std::pow(test_in[0], 2)) * test_in[0] * X[0][1][2] +
              (-4.0 * std::sin(std::pow(test_in[0], 2)) *
                   std::pow(test_in[0], 2) +
               2.0 * std::cos(std::pow(test_in[0], 2))) *
                  X[0][1][0] * X[0][1][1] +
              1.0 / 6.0 *
                  (-12.0 * test_in[0] * std::sin(std::pow(test_in[0], 2)) -
                   8.0 * std::pow(test_in[0], 3) *
                       std::cos(std::pow(test_in[0], 2))) *
                  std::pow(X[0][1][0], 3),
      tt::tolerance(tol));
  myfree3(X);
  myfree3(Y);
}

BOOST_AUTO_TEST_CASE(CosOperator_HOV_Forward) {
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{0.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // cos(x1^2)
  trace_on(tapeId10);
  indep[0] <<= in[0];
  adouble dep = cos(pow(indep[0], 2));
  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree);
  double ***Y = myalloc3(dim_out, num_dirs, degree);

  X[0][0][0] = 1.0;
  X[0][0][1] = -1.0;
  X[0][0][2] = 1.1;

  X[0][1][0] = 2.0;
  X[0][1][1] = -2.1;
  X[0][1][2] = -2.0;

  std::vector<double> test_in{0.5};

  // cos(x1^2)
  double test_out = std::cos(std::pow(test_in[0], 2));

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));

  // first derivative
  BOOST_TEST(Y[0][0][0] == -2.0 * std::sin(std::pow(test_in[0], 2)) *
                               test_in[0] * X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] == -2.0 * std::sin(std::pow(test_in[0], 2)) *
                               test_in[0] * X[0][1][0],
             tt::tolerance(tol));

  // second derivative
  BOOST_TEST(Y[0][0][1] == -2.0 * std::sin(std::pow(test_in[0], 2)) *
                                   test_in[0] * X[0][0][1] +
                               1.0 / 2.0 *
                                   (-4.0 * std::cos(std::pow(test_in[0], 2)) *
                                        std::pow(test_in[0], 2) -
                                    2.0 * std::sin(std::pow(test_in[0], 2))) *
                                   X[0][0][0] * X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] == -2.0 * std::sin(std::pow(test_in[0], 2)) *
                                   test_in[0] * X[0][1][1] +
                               1.0 / 2.0 *
                                   (-4.0 * std::cos(std::pow(test_in[0], 2)) *
                                        std::pow(test_in[0], 2) -
                                    2.0 * std::sin(std::pow(test_in[0], 2))) *
                                   X[0][1][0] * X[0][1][0],
             tt::tolerance(tol));

  // third derivative
  BOOST_TEST(
      Y[0][0][2] ==
          -2.0 * std::sin(std::pow(test_in[0], 2)) * test_in[0] * X[0][0][2] +
              (-4.0 * std::cos(std::pow(test_in[0], 2)) *
                   std::pow(test_in[0], 2) -
               2.0 * std::sin(std::pow(test_in[0], 2))) *
                  X[0][0][0] * X[0][0][1] +
              1.0 / 6.0 *
                  (-12.0 * test_in[0] * std::cos(std::pow(test_in[0], 2)) +
                   8.0 * std::pow(test_in[0], 3) *
                       std::sin(std::pow(test_in[0], 2))) *
                  std::pow(X[0][0][0], 3),
      tt::tolerance(tol));
  BOOST_TEST(
      Y[0][1][2] ==
          -2.0 * std::sin(std::pow(test_in[0], 2)) * test_in[0] * X[0][1][2] +
              (-4.0 * std::cos(std::pow(test_in[0], 2)) *
                   std::pow(test_in[0], 2) -
               2.0 * std::sin(std::pow(test_in[0], 2))) *
                  X[0][1][0] * X[0][1][1] +
              1.0 / 6.0 *
                  (-12.0 * test_in[0] * std::cos(std::pow(test_in[0], 2)) +
                   8.0 * std::pow(test_in[0], 3) *
                       std::sin(std::pow(test_in[0], 2))) *
                  std::pow(X[0][1][0], 3),
      tt::tolerance(tol));
  myfree3(X);
  myfree3(Y);
}

BOOST_AUTO_TEST_CASE(SqrtOperator_HOV_Forward) {
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{0.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // sqrt(x)
  trace_on(tapeId10);
  indep[0] <<= in[0];
  adouble dep = sqrt(indep[0]);
  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree);
  double ***Y = myalloc3(dim_out, num_dirs, degree);

  X[0][0][0] = 1.0;
  X[0][0][1] = -1.0;
  X[0][0][2] = 1.1;

  X[0][1][0] = 2.0;
  X[0][1][1] = -2.1;
  X[0][1][2] = -2.0;

  std::vector<double> test_in{0.5};

  // sqrt(x)
  double test_out = std::sqrt(test_in[0]);

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));

  // first derivative
  BOOST_TEST(Y[0][0][0] == 1.0 / (2.0 * std::sqrt(test_in[0])) * X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] == 1.0 / (2.0 * std::sqrt(test_in[0])) * X[0][1][0],
             tt::tolerance(tol));

  // second derivative
  BOOST_TEST(Y[0][0][1] ==
                 1.0 / (2.0 * std::sqrt(test_in[0])) * X[0][0][1] +
                     1.0 / 2.0 *
                         (-1.0 / (4.0 * std::pow(test_in[0], 3.0 / 2.0)) *
                          X[0][0][0] * X[0][0][0]),
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] ==
                 1.0 / (2.0 * std::sqrt(test_in[0])) * X[0][1][1] +
                     1.0 / 2.0 *
                         (-1.0 / (4.0 * std::pow(test_in[0], 3.0 / 2.0)) *
                          X[0][1][0] * X[0][1][0]),
             tt::tolerance(tol));

  // third derivative
  BOOST_TEST(Y[0][0][2] ==
                 1.0 / (2.0 * std::sqrt(test_in[0])) * X[0][0][2] +
                     (-1.0 / (4.0 * std::pow(test_in[0], 3.0 / 2.0)) *
                      X[0][0][0] * X[0][0][1]) +
                     1.0 / 6.0 *
                         (3.0 / (8.0 * std::pow(test_in[0], 5.0 / 2.0)) *
                          std::pow(X[0][0][0], 3)),
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][2] ==
                 1.0 / (2.0 * std::sqrt(test_in[0])) * X[0][1][2] +
                     (-1.0 / (4.0 * std::pow(test_in[0], 3.0 / 2.0)) *
                      X[0][1][0] * X[0][1][1]) +
                     1.0 / 6.0 *
                         (3.0 / (8.0 * std::pow(test_in[0], 5.0 / 2.0)) *
                          std::pow(X[0][1][0], 3)),
             tt::tolerance(tol));
  myfree3(X);
  myfree3(Y);
}

BOOST_AUTO_TEST_CASE(LogOperator_HOV_Forward) {
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{0.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // Log(x)
  trace_on(tapeId10);
  indep[0] <<= in[0];
  adouble dep = log(indep[0]);
  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree);
  double ***Y = myalloc3(dim_out, num_dirs, degree);

  X[0][0][0] = 1.0;
  X[0][0][1] = -1.0;
  X[0][0][2] = 1.1;

  X[0][1][0] = 2.0;
  X[0][1][1] = -2.1;
  X[0][1][2] = -2.0;

  std::vector<double> test_in{0.5};

  // log(x)
  double test_out = std::log(test_in[0]);

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));

  // first derivative
  BOOST_TEST(Y[0][0][0] == 1.0 / test_in[0] * X[0][0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] == 1.0 / test_in[0] * X[0][1][0], tt::tolerance(tol));

  // second derivative
  BOOST_TEST(Y[0][0][1] == 1.0 / test_in[0] * X[0][0][1] +
                               1.0 / 2.0 *
                                   (-1.0 / std::pow(test_in[0], 2.0) *
                                    X[0][0][0] * X[0][0][0]),
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] == 1.0 / test_in[0] * X[0][1][1] +
                               1.0 / 2.0 *
                                   (-1.0 / std::pow(test_in[0], 2.0) *
                                    X[0][1][0] * X[0][1][0]),
             tt::tolerance(tol));

  // third derivative
  BOOST_TEST(Y[0][0][2] == 1.0 / test_in[0] * X[0][0][2] +
                               (-1.0 / std::pow(test_in[0], 2.0) * X[0][0][0] *
                                X[0][0][1]) +
                               1.0 / 6.0 * (2.0 / std::pow(test_in[0], 3.0)) *
                                   std::pow(X[0][0][0], 3),
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][2] == 1.0 / test_in[0] * X[0][1][2] +
                               (-1.0 / std::pow(test_in[0], 2.0) * X[0][1][0] *
                                X[0][1][1]) +
                               1.0 / 6.0 * (2.0 / std::pow(test_in[0], 3.0)) *
                                   std::pow(X[0][1][0], 3),
             tt::tolerance(tol));
  myfree3(X);
  myfree3(Y);
}

BOOST_AUTO_TEST_CASE(SinhOperator_HOV_Forward) {
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{0.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // sinh(x)
  trace_on(tapeId10);
  indep[0] <<= in[0];
  adouble dep = sinh(indep[0]);
  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree);
  double ***Y = myalloc3(dim_out, num_dirs, degree);

  X[0][0][0] = 1.0;
  X[0][0][1] = -1.0;
  X[0][0][2] = 1.1;

  X[0][1][0] = 2.0;
  X[0][1][1] = -2.1;
  X[0][1][2] = -2.0;

  std::vector<double> test_in{0.5};

  // sinh(x)
  double test_out = std::sinh(test_in[0]);

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));

  // first derivative
  BOOST_TEST(Y[0][0][0] == std::cosh(test_in[0]) * X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] == std::cosh(test_in[0]) * X[0][1][0],
             tt::tolerance(tol));

  // second derivative
  BOOST_TEST(Y[0][0][1] ==
                 std::cosh(test_in[0]) * X[0][0][1] +
                     1.0 / 2.0 *
                         (std::sinh(test_in[0]) * X[0][0][0] * X[0][0][0]),
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] ==
                 std::cosh(test_in[0]) * X[0][1][1] +
                     1.0 / 2.0 *
                         (std::sinh(test_in[0]) * X[0][1][0] * X[0][1][0]),
             tt::tolerance(tol));

  // third derivative
  BOOST_TEST(Y[0][0][2] == std::cosh(test_in[0]) * X[0][0][2] +
                               std::sinh(test_in[0]) * X[0][0][0] * X[0][0][1] +
                               1.0 / 6.0 * std::cosh(test_in[0]) *
                                   std::pow(X[0][0][0], 3),
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][2] == std::cosh(test_in[0]) * X[0][1][2] +
                               std::sinh(test_in[0]) * X[0][1][0] * X[0][1][1] +
                               1.0 / 6.0 * std::cosh(test_in[0]) *
                                   std::pow(X[0][1][0], 3),
             tt::tolerance(tol));
  myfree3(X);
  myfree3(Y);
}

BOOST_AUTO_TEST_CASE(CoshOperator_HOV_Forward) {
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{0.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // cosh(x)
  trace_on(tapeId10);
  indep[0] <<= in[0];
  adouble dep = cosh(indep[0]);
  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree);
  double ***Y = myalloc3(dim_out, num_dirs, degree);

  X[0][0][0] = 1.0;
  X[0][0][1] = -1.0;
  X[0][0][2] = 1.1;

  X[0][1][0] = 2.0;
  X[0][1][1] = -2.1;
  X[0][1][2] = -2.0;

  std::vector<double> test_in{0.5};

  // cosh(x)
  double test_out = std::cosh(test_in[0]);

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));

  // first derivative
  BOOST_TEST(Y[0][0][0] == std::sinh(test_in[0]) * X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] == std::sinh(test_in[0]) * X[0][1][0],
             tt::tolerance(tol));

  // second derivative
  BOOST_TEST(Y[0][0][1] ==
                 std::sinh(test_in[0]) * X[0][0][1] +
                     1.0 / 2.0 *
                         (std::cosh(test_in[0]) * X[0][0][0] * X[0][0][0]),
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] ==
                 std::sinh(test_in[0]) * X[0][1][1] +
                     1.0 / 2.0 *
                         (std::cosh(test_in[0]) * X[0][1][0] * X[0][1][0]),
             tt::tolerance(tol));

  // third derivative
  BOOST_TEST(Y[0][0][2] == std::sinh(test_in[0]) * X[0][0][2] +
                               std::cosh(test_in[0]) * X[0][0][0] * X[0][0][1] +
                               1.0 / 6.0 * std::sinh(test_in[0]) *
                                   std::pow(X[0][0][0], 3),
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][2] == std::sinh(test_in[0]) * X[0][1][2] +
                               std::cosh(test_in[0]) * X[0][1][0] * X[0][1][1] +
                               1.0 / 6.0 * std::sinh(test_in[0]) *
                                   std::pow(X[0][1][0], 3),
             tt::tolerance(tol));
  myfree3(X);
  myfree3(Y);
}
BOOST_AUTO_TEST_CASE(TanhOperator_HOV_Forward) {
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{0.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // tanh(x)
  trace_on(tapeId10);
  indep[0] <<= in[0];
  adouble dep = tanh(indep[0]);
  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree);
  double ***Y = myalloc3(dim_out, num_dirs, degree);

  X[0][0][0] = 1.0;
  X[0][0][1] = -1.0;
  X[0][0][2] = 1.1;

  X[0][1][0] = 2.0;
  X[0][1][1] = -2.1;
  X[0][1][2] = -2.0;

  std::vector<double> test_in{0.5};

  // tanh(x)
  double test_out = std::tanh(test_in[0]);

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));

  // first derivative
  BOOST_TEST(Y[0][0][0] ==
                 1.0 / std::pow(std::cosh(test_in[0]), 2) * X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] ==
                 1.0 / std::pow(std::cosh(test_in[0]), 2) * X[0][1][0],
             tt::tolerance(tol));

  // second derivative
  BOOST_TEST(Y[0][0][1] ==
                 1.0 / std::pow(std::cosh(test_in[0]), 2) * X[0][0][1] +
                     1.0 / 2.0 *
                         (-2.0 * std::tanh(test_in[0]) * 1.0 /
                          std::pow(std::cosh(test_in[0]), 2) * X[0][0][0] *
                          X[0][0][0]),
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] ==
                 1.0 / std::pow(std::cosh(test_in[0]), 2) * X[0][1][1] +
                     1.0 / 2.0 *
                         (-2.0 * std::tanh(test_in[0]) * 1.0 /
                          std::pow(std::cosh(test_in[0]), 2) * X[0][1][0] *
                          X[0][1][0]),
             tt::tolerance(tol));

  // third derivative
  BOOST_TEST(Y[0][0][2] ==
                 1.0 / std::pow(std::cosh(test_in[0]), 2) * X[0][0][2] +
                     (-2.0 * std::tanh(test_in[0]) * 1.0 /
                      std::pow(std::cosh(test_in[0]), 2) * X[0][0][0] *
                      X[0][0][1]) +
                     1.0 / 6.0 *
                         (4.0 * std::pow(std::tanh(test_in[0]), 2) * 1.0 /
                              std::pow(std::cosh(test_in[0]), 2) -
                          2.0 / std::pow(std::cosh(test_in[0]), 4)) *
                         std::pow(X[0][0][0], 3),
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][2] ==
                 1.0 / std::pow(std::cosh(test_in[0]), 2) * X[0][1][2] +
                     (-2.0 * std::tanh(test_in[0]) * 1.0 /
                      std::pow(std::cosh(test_in[0]), 2) * X[0][1][0] *
                      X[0][1][1]) +
                     1.0 / 6.0 *
                         (4.0 * std::pow(std::tanh(test_in[0]), 2) * 1.0 /
                              std::pow(std::cosh(test_in[0]), 2) -
                          2.0 / std::pow(std::cosh(test_in[0]), 4)) *
                         std::pow(X[0][1][0], 3),
             tt::tolerance(tol));
  myfree3(X);
  myfree3(Y);
}
BOOST_AUTO_TEST_CASE(AsinOperator_HOV_Forward) {
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{0.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // asin(x)
  trace_on(tapeId10);
  indep[0] <<= in[0];
  adouble dep = asin(indep[0]);
  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree);
  double ***Y = myalloc3(dim_out, num_dirs, degree);

  X[0][0][0] = 1.0;
  X[0][0][1] = -1.0;
  X[0][0][2] = 1.1;

  X[0][1][0] = 2.0;
  X[0][1][1] = -2.1;
  X[0][1][2] = -2.0;

  std::vector<double> test_in{0.5};

  // asin(x)
  double test_out = std::asin(test_in[0]);

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));

  // first derivative
  BOOST_TEST(Y[0][0][0] ==
                 1.0 / std::sqrt(1.0 - std::pow(test_in[0], 2)) * X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] ==
                 1.0 / std::sqrt(1.0 - std::pow(test_in[0], 2)) * X[0][1][0],
             tt::tolerance(tol));

  // second derivative
  BOOST_TEST(Y[0][0][1] ==
                 1.0 / std::sqrt(1.0 - std::pow(test_in[0], 2)) * X[0][0][1] +
                     1.0 / 2.0 *
                         (test_in[0] /
                          std::pow(1.0 - std::pow(test_in[0], 2), 3.0 / 2.0)) *
                         X[0][0][0] * X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] ==
                 1.0 / std::sqrt(1.0 - std::pow(test_in[0], 2)) * X[0][1][1] +
                     1.0 / 2.0 *
                         (test_in[0] /
                          std::pow(1.0 - std::pow(test_in[0], 2), 3.0 / 2.0)) *
                         X[0][1][0] * X[0][1][0],
             tt::tolerance(tol));

  // third derivative
  BOOST_TEST(Y[0][0][2] ==
                 1.0 / std::sqrt(1.0 - std::pow(test_in[0], 2)) * X[0][0][2] +
                     (test_in[0] /
                      std::pow(1.0 - std::pow(test_in[0], 2), 3.0 / 2.0)) *
                         X[0][0][0] * X[0][0][1] +
                     1.0 / 6.0 * (2.0 * std::pow(test_in[0], 2) + 1) /
                         std::pow(1.0 - std::pow(test_in[0], 2), 5.0 / 2.0) *
                         std::pow(X[0][0][0], 3),
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][2] ==
                 1.0 / std::sqrt(1.0 - std::pow(test_in[0], 2)) * X[0][1][2] +
                     (test_in[0] /
                      std::pow(1.0 - std::pow(test_in[0], 2), 3.0 / 2.0)) *
                         X[0][1][0] * X[0][1][1] +
                     1.0 / 6.0 * (2.0 * std::pow(test_in[0], 2) + 1) /
                         std::pow(1.0 - std::pow(test_in[0], 2), 5.0 / 2.0) *
                         std::pow(X[0][1][0], 3),
             tt::tolerance(tol));
  myfree3(X);
  myfree3(Y);
}

BOOST_AUTO_TEST_CASE(AcosOperator_HOV_Forward) {
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{0.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // acos(x)
  trace_on(tapeId10);
  indep[0] <<= in[0];
  adouble dep = acos(indep[0]);
  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree);
  double ***Y = myalloc3(dim_out, num_dirs, degree);

  X[0][0][0] = 1.0;
  X[0][0][1] = -1.0;
  X[0][0][2] = 1.1;

  X[0][1][0] = 2.0;
  X[0][1][1] = -2.1;
  X[0][1][2] = -2.0;

  std::vector<double> test_in{0.5};

  // acos(x)
  double test_out = std::acos(test_in[0]);

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));

  // first derivative
  BOOST_TEST(Y[0][0][0] ==
                 -1.0 / std::sqrt(1.0 - std::pow(test_in[0], 2)) * X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] ==
                 -1.0 / std::sqrt(1.0 - std::pow(test_in[0], 2)) * X[0][1][0],
             tt::tolerance(tol));

  // second derivative
  BOOST_TEST(Y[0][0][1] ==
                 -1.0 / std::sqrt(1.0 - std::pow(test_in[0], 2)) * X[0][0][1] +
                     1.0 / 2.0 *
                         (-test_in[0] /
                          std::pow(1.0 - std::pow(test_in[0], 2), 3.0 / 2.0)) *
                         X[0][0][0] * X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] ==
                 -1.0 / std::sqrt(1.0 - std::pow(test_in[0], 2)) * X[0][1][1] +
                     1.0 / 2.0 *
                         (-test_in[0] /
                          std::pow(1.0 - std::pow(test_in[0], 2), 3.0 / 2.0)) *
                         X[0][1][0] * X[0][1][0],
             tt::tolerance(tol));

  // third derivative
  BOOST_TEST(Y[0][0][2] ==
                 -1.0 / std::sqrt(1.0 - std::pow(test_in[0], 2)) * X[0][0][2] +
                     (-test_in[0] /
                      std::pow(1.0 - std::pow(test_in[0], 2), 3.0 / 2.0)) *
                         X[0][0][0] * X[0][0][1] +
                     -1.0 / 6.0 * (2.0 * std::pow(test_in[0], 2) + 1) /
                         std::pow(1.0 - std::pow(test_in[0], 2), 5.0 / 2.0) *
                         std::pow(X[0][0][0], 3),
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][2] ==
                 -1.0 / std::sqrt(1.0 - std::pow(test_in[0], 2)) * X[0][1][2] +
                     (-test_in[0] /
                      std::pow(1.0 - std::pow(test_in[0], 2), 3.0 / 2.0)) *
                         X[0][1][0] * X[0][1][1] +
                     -1.0 / 6.0 * (2.0 * std::pow(test_in[0], 2) + 1) /
                         std::pow(1.0 - std::pow(test_in[0], 2), 5.0 / 2.0) *
                         std::pow(X[0][1][0], 3),
             tt::tolerance(tol));
  myfree3(X);
  myfree3(Y);
}

BOOST_AUTO_TEST_CASE(AtanOperator_HOV_Forward) {
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{0.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // atan(x)
  trace_on(tapeId10);
  indep[0] <<= in[0];
  adouble dep = atan(indep[0]);
  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree);
  double ***Y = myalloc3(dim_out, num_dirs, degree);

  X[0][0][0] = 1.0;
  X[0][0][1] = -1.0;
  X[0][0][2] = 1.1;

  X[0][1][0] = 2.0;
  X[0][1][1] = -2.1;
  X[0][1][2] = -2.0;

  std::vector<double> test_in{0.5};

  // atan(x)
  double test_out = std::atan(test_in[0]);

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));

  // first derivative
  BOOST_TEST(Y[0][0][0] == 1.0 / (1.0 + std::pow(test_in[0], 2)) * X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] == 1.0 / (1.0 + std::pow(test_in[0], 2)) * X[0][1][0],
             tt::tolerance(tol));

  // second derivative
  BOOST_TEST(Y[0][0][1] == 1.0 / (1.0 + std::pow(test_in[0], 2)) * X[0][0][1] +
                               1.0 / 2.0 * (-2.0 * test_in[0]) /
                                   std::pow(1.0 + std::pow(test_in[0], 2), 2) *
                                   X[0][0][0] * X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] == 1.0 / (1.0 + std::pow(test_in[0], 2)) * X[0][1][1] +
                               1.0 / 2.0 * (-2.0 * test_in[0]) /
                                   std::pow(1.0 + std::pow(test_in[0], 2), 2) *
                                   X[0][1][0] * X[0][1][0],
             tt::tolerance(tol));

  // third derivative
  BOOST_TEST(Y[0][0][2] ==
                 1.0 / (1.0 + std::pow(test_in[0], 2)) * X[0][0][2] +
                     (-2.0 * test_in[0]) /
                         std::pow(1.0 + std::pow(test_in[0], 2), 2) *
                         X[0][0][0] * X[0][0][1] +
                     1.0 / 6.0 *
                         ((6.0 * std::pow(test_in[0], 2) - 2.0) /
                          std::pow(1.0 + std::pow(test_in[0], 2), 3)) *
                         std::pow(X[0][0][0], 3),
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][2] ==
                 1.0 / (1.0 + std::pow(test_in[0], 2)) * X[0][1][2] +
                     (-2.0 * test_in[0]) /
                         std::pow(1.0 + std::pow(test_in[0], 2), 2) *
                         X[0][1][0] * X[0][1][1] +
                     1.0 / 6.0 *
                         ((6.0 * std::pow(test_in[0], 2) - 2.0) /
                          std::pow(1.0 + std::pow(test_in[0], 2), 3)) *
                         std::pow(X[0][1][0], 3),
             tt::tolerance(tol));
  myfree3(X);
  myfree3(Y);
}

BOOST_AUTO_TEST_CASE(Log10Operator_HOV_Forward) {
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{0.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // log10(x)
  trace_on(tapeId10);
  indep[0] <<= in[0];
  adouble dep = log10(indep[0]);
  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree);
  double ***Y = myalloc3(dim_out, num_dirs, degree);

  X[0][0][0] = 1.0;
  X[0][0][1] = -1.0;
  X[0][0][2] = 1.1;

  X[0][1][0] = 2.0;
  X[0][1][1] = -2.1;
  X[0][1][2] = -2.0;

  std::vector<double> test_in{0.5};

  // log10(x)
  double test_out = std::log10(test_in[0]);

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));

  const double log_2_10 = std::log(10);
  // first derivative
  BOOST_TEST(Y[0][0][0] == 1.0 / (test_in[0] * log_2_10) * X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] == 1.0 / (test_in[0] * log_2_10) * X[0][1][0],
             tt::tolerance(tol));

  // second derivative
  BOOST_TEST(Y[0][0][1] ==
                 1.0 / (test_in[0] * log_2_10) * X[0][0][1] +
                     1.0 / 2.0 * (-1.0 / (std::pow(test_in[0], 2) * log_2_10)) *
                         X[0][0][0] * X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] ==
                 1.0 / (test_in[0] * log_2_10) * X[0][1][1] +
                     1.0 / 2.0 * (-1.0 / (std::pow(test_in[0], 2) * log_2_10)) *
                         X[0][1][0] * X[0][1][0],
             tt::tolerance(tol));

  // third derivative
  BOOST_TEST(Y[0][0][2] ==
                 1.0 / (test_in[0] * log_2_10) * X[0][0][2] +
                     (-1.0 / (std::pow(test_in[0], 2) * log_2_10)) *
                         X[0][0][0] * X[0][0][1] +
                     1.0 / 6.0 * (2.0 / (std::pow(test_in[0], 3) * log_2_10)) *
                         std::pow(X[0][0][0], 3),
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][2] ==
                 1.0 / (test_in[0] * log_2_10) * X[0][1][2] +
                     (-1.0 / (std::pow(test_in[0], 2) * log_2_10)) *
                         X[0][1][0] * X[0][1][1] +
                     1.0 / 6.0 * (2.0 / (std::pow(test_in[0], 3) * log_2_10)) *
                         std::pow(X[0][1][0], 3),
             tt::tolerance(tol));
  myfree3(X);
  myfree3(Y);
}

BOOST_AUTO_TEST_CASE(AsinhOperator_HOV_Forward) {
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{0.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // asinh(x)
  trace_on(tapeId10);
  indep[0] <<= in[0];
  adouble dep = asinh(indep[0]);
  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree);
  double ***Y = myalloc3(dim_out, num_dirs, degree);

  X[0][0][0] = 1.0;
  X[0][0][1] = -1.0;
  X[0][0][2] = 1.1;

  X[0][1][0] = 2.0;
  X[0][1][1] = -2.1;
  X[0][1][2] = -2.0;

  std::vector<double> test_in{0.5};

  // asinh(x)
  double test_out = std::asinh(test_in[0]);

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));

  // first derivative
  BOOST_TEST(Y[0][0][0] ==
                 1.0 / std::sqrt(std::pow(test_in[0], 2) + 1) * X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] ==
                 1.0 / std::sqrt(std::pow(test_in[0], 2) + 1) * X[0][1][0],
             tt::tolerance(tol));

  // second derivative
  BOOST_TEST(Y[0][0][1] ==
                 1.0 / std::sqrt(std::pow(test_in[0], 2) + 1) * X[0][0][1] +
                     1.0 / 2.0 *
                         (-test_in[0] /
                          std::pow(std::pow(test_in[0], 2) + 1, 3.0 / 2.0)) *
                         X[0][0][0] * X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] ==
                 1.0 / std::sqrt(std::pow(test_in[0], 2) + 1) * X[0][1][1] +
                     1.0 / 2.0 *
                         (-test_in[0] /
                          std::pow(std::pow(test_in[0], 2) + 1, 3.0 / 2.0)) *
                         X[0][1][0] * X[0][1][0],
             tt::tolerance(tol));

  // third derivative
  BOOST_TEST(
      Y[0][0][2] ==
          1.0 / std::sqrt(std::pow(test_in[0], 2) + 1) * X[0][0][2] +
              (-test_in[0] / std::pow(std::pow(test_in[0], 2) + 1, 3.0 / 2.0)) *
                  X[0][0][0] * X[0][0][1] +
              1.0 / 6.0 * (2.0 * std::pow(test_in[0], 2) - 1.0) /
                  (std::pow(std::pow(test_in[0], 2) + 1, 5.0 / 2.0)) *
                  std::pow(X[0][0][0], 3),
      tt::tolerance(tol));
  BOOST_TEST(
      Y[0][1][2] ==
          1.0 / std::sqrt(std::pow(test_in[0], 2) + 1) * X[0][1][2] +
              (-test_in[0] / std::pow(std::pow(test_in[0], 2) + 1, 3.0 / 2.0)) *
                  X[0][1][0] * X[0][1][1] +
              1.0 / 6.0 * (2.0 * std::pow(test_in[0], 2) - 1.0) /
                  (std::pow(std::pow(test_in[0], 2) + 1, 5.0 / 2.0)) *
                  std::pow(X[0][1][0], 3),
      tt::tolerance(tol));
  myfree3(X);
  myfree3(Y);
}

BOOST_AUTO_TEST_CASE(AcoshOperator_HOV_Forward) {
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{1.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // acosh(x)
  trace_on(tapeId10);
  indep[0] <<= in[0];
  adouble dep = acosh(indep[0]);
  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree);
  double ***Y = myalloc3(dim_out, num_dirs, degree);

  X[0][0][0] = 1.0;
  X[0][0][1] = -1.0;
  X[0][0][2] = 1.1;

  X[0][1][0] = 2.0;
  X[0][1][1] = -2.1;
  X[0][1][2] = -2.0;

  std::vector<double> test_in{1.2};

  // acosh(x)
  double test_out = std::acosh(test_in[0]);

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));

  // first derivative
  BOOST_TEST(Y[0][0][0] ==
                 1.0 / std::sqrt((test_in[0] + 1.0) * (test_in[0] - 1.0)) *
                     X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] ==
                 1.0 / std::sqrt((test_in[0] + 1.0) * (test_in[0] - 1.0)) *
                     X[0][1][0],
             tt::tolerance(tol));

  // second derivative
  BOOST_TEST(Y[0][0][1] ==
                 1.0 / std::sqrt((test_in[0] + 1.0) * (test_in[0] - 1.0)) *
                         X[0][0][1] +
                     1.0 / 2.0 *
                         (-test_in[0] /
                          std::pow((test_in[0] - 1.0) * (test_in[0] + 1.0),
                                   3.0 / 2.0)) *
                         X[0][0][0] * X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] ==
                 1.0 / std::sqrt((test_in[0] + 1.0) * (test_in[0] - 1.0)) *
                         X[0][1][1] +
                     1.0 / 2.0 *
                         (-test_in[0] /
                          std::pow((test_in[0] - 1.0) * (test_in[0] + 1.0),
                                   3.0 / 2.0)) *
                         X[0][1][0] * X[0][1][0],
             tt::tolerance(tol));

  // third derivative
  BOOST_TEST(
      Y[0][0][2] ==
          1.0 / std::sqrt((test_in[0] + 1.0) * (test_in[0] - 1.0)) *
                  X[0][0][2] +
              -test_in[0] /
                  std::pow((test_in[0] - 1.0) * (test_in[0] + 1.0), 3.0 / 2.0) *
                  X[0][0][0] * X[0][0][1] +
              1.0 / 6.0 * (2.0 * std::pow(test_in[0], 2) + 1) /
                  std::pow(std::pow(test_in[0], 2) - 1.0, 5.0 / 2.0) *
                  std::pow(X[0][0][0], 3),
      tt::tolerance(tol));
  BOOST_TEST(
      Y[0][1][2] ==
          1.0 / std::sqrt((test_in[0] + 1.0) * (test_in[0] - 1.0)) *
                  X[0][1][2] +
              -test_in[0] /
                  std::pow((test_in[0] - 1.0) * (test_in[0] + 1.0), 3.0 / 2.0) *
                  X[0][1][0] * X[0][1][1] +
              1.0 / 6.0 * (2.0 * std::pow(test_in[0], 2) + 1) /
                  std::pow(std::pow(test_in[0], 2) - 1.0, 5.0 / 2.0) *
                  std::pow(X[0][1][0], 3),
      tt::tolerance(tol));
  myfree3(X);
  myfree3(Y);
}

BOOST_AUTO_TEST_CASE(AtanhOperator_HOV_Forward) {
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{0.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // atanh(x)
  trace_on(tapeId10);
  indep[0] <<= in[0];
  adouble dep = atanh(indep[0]);
  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree);
  double ***Y = myalloc3(dim_out, num_dirs, degree);

  X[0][0][0] = 1.0;
  X[0][0][1] = -1.0;
  X[0][0][2] = 1.1;

  X[0][1][0] = 2.0;
  X[0][1][1] = -2.1;
  X[0][1][2] = -2.0;

  std::vector<double> test_in{0.2};

  // atanh(x)
  double test_out = std::atanh(test_in[0]);

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));

  // first derivative
  BOOST_TEST(Y[0][0][0] == 1.0 / (1.0 - std::pow(test_in[0], 2)) * X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] == 1.0 / (1.0 - std::pow(test_in[0], 2)) * X[0][1][0],
             tt::tolerance(tol));

  // second derivative
  BOOST_TEST(Y[0][0][1] == 1.0 / (1.0 - std::pow(test_in[0], 2)) * X[0][0][1] +
                               1.0 / 2.0 * 2.0 * test_in[0] /
                                   std::pow(1.0 - std::pow(test_in[0], 2), 2) *
                                   X[0][0][0] * X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] == 1.0 / (1.0 - std::pow(test_in[0], 2)) * X[0][1][1] +
                               1.0 / 2.0 * 2.0 * test_in[0] /
                                   std::pow(1.0 - std::pow(test_in[0], 2), 2) *
                                   X[0][1][0] * X[0][1][0],
             tt::tolerance(tol));

  // third derivative
  BOOST_TEST(Y[0][0][2] == 1.0 / (1.0 - std::pow(test_in[0], 2)) * X[0][0][2] +
                               2.0 * test_in[0] /
                                   std::pow(1.0 - std::pow(test_in[0], 2), 2) *
                                   X[0][0][0] * X[0][0][1] +
                               1.0 / 6.0 * 2.0 *
                                   (3.0 * std::pow(test_in[0], 2) + 1) /
                                   std::pow(1.0 - std::pow(test_in[0], 2), 3) *
                                   std::pow(X[0][0][0], 3),
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][2] == 1.0 / (1.0 - std::pow(test_in[0], 2)) * X[0][1][2] +
                               2.0 * test_in[0] /
                                   std::pow(1.0 - std::pow(test_in[0], 2), 2) *
                                   X[0][1][0] * X[0][1][1] +
                               1.0 / 6.0 * 2.0 *
                                   (3.0 * std::pow(test_in[0], 2) + 1) /
                                   std::pow(1.0 - std::pow(test_in[0], 2), 3) *
                                   std::pow(X[0][1][0], 3),
             tt::tolerance(tol));
  myfree3(X);
  myfree3(Y);
}

BOOST_AUTO_TEST_CASE(InclOperator_HOV_Forward) {
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{0.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);
  adouble dep;

  // x + 1
  trace_on(tapeId10);
  indep[0] <<= in[0];
  dep = ++indep[0];
  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree);
  double ***Y = myalloc3(dim_out, num_dirs, degree);

  X[0][0][0] = 1.0;
  X[0][0][1] = -1.0;
  X[0][0][2] = 1.1;

  X[0][1][0] = 2.0;
  X[0][1][1] = -2.1;
  X[0][1][2] = -2.0;

  std::vector<double> test_in{0.2};

  // x + 1
  double test_out = ++test_in[0];

  // change the value back, since the operator increases test_in[0]
  test_in[0] = 0.2;
  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));

  // first derivative
  BOOST_TEST(Y[0][0][0] == X[0][0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] == X[0][1][0], tt::tolerance(tol));

  // second derivative
  BOOST_TEST(Y[0][0][1] == X[0][0][1], tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] == X[0][1][1], tt::tolerance(tol));

  // third derivative
  BOOST_TEST(Y[0][0][2] == X[0][0][2], tt::tolerance(tol));
  BOOST_TEST(Y[0][1][2] == X[0][1][2], tt::tolerance(tol));
  myfree3(X);
  myfree3(Y);
}

BOOST_AUTO_TEST_CASE(DeclOperator_HOV_Forward) {
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{0.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);
  adouble dep;

  // x - 1
  trace_on(tapeId10);
  indep[0] <<= in[0];
  dep = --indep[0];
  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree);
  double ***Y = myalloc3(dim_out, num_dirs, degree);

  X[0][0][0] = 1.0;
  X[0][0][1] = -1.0;
  X[0][0][2] = 1.1;

  X[0][1][0] = 2.0;
  X[0][1][1] = -2.1;
  X[0][1][2] = -2.0;

  std::vector<double> test_in{0.2};

  // x - 1
  double test_out = --test_in[0];

  // change the value back, since the operator increases test_in[0]
  test_in[0] = 0.2;
  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));

  // first derivative
  BOOST_TEST(Y[0][0][0] == X[0][0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] == X[0][1][0], tt::tolerance(tol));

  // second derivative
  BOOST_TEST(Y[0][0][1] == X[0][0][1], tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] == X[0][1][1], tt::tolerance(tol));

  // third derivative
  BOOST_TEST(Y[0][0][2] == X[0][0][2], tt::tolerance(tol));
  BOOST_TEST(Y[0][1][2] == X[0][1][2], tt::tolerance(tol));
  myfree3(X);
  myfree3(Y);
}

BOOST_AUTO_TEST_CASE(SignPlusOperator_HOV_Forward) {
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{0.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);
  adouble dep;

  // x
  trace_on(tapeId10);
  indep[0] <<= in[0];
  dep = +indep[0];
  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree);
  double ***Y = myalloc3(dim_out, num_dirs, degree);

  X[0][0][0] = 1.0;
  X[0][0][1] = -1.0;
  X[0][0][2] = 1.1;

  X[0][1][0] = 2.0;
  X[0][1][1] = -2.1;
  X[0][1][2] = -2.0;

  std::vector<double> test_in{0.2};

  // x
  double test_out = +test_in[0];

  // change the value back, since the operator increases test_in[0]
  test_in[0] = 0.2;
  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));

  // first derivative
  BOOST_TEST(Y[0][0][0] == X[0][0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] == X[0][1][0], tt::tolerance(tol));

  // second derivative
  BOOST_TEST(Y[0][0][1] == X[0][0][1], tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] == X[0][1][1], tt::tolerance(tol));

  // third derivative
  BOOST_TEST(Y[0][0][2] == X[0][0][2], tt::tolerance(tol));
  BOOST_TEST(Y[0][1][2] == X[0][1][2], tt::tolerance(tol));
  myfree3(X);
  myfree3(Y);
}

BOOST_AUTO_TEST_CASE(SignMinusOperator_HOV_Forward) {
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{0.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);
  adouble dep;

  //-x
  trace_on(tapeId10);
  indep[0] <<= in[0];
  dep = -indep[0];
  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree);
  double ***Y = myalloc3(dim_out, num_dirs, degree);

  X[0][0][0] = 1.0;
  X[0][0][1] = -1.0;
  X[0][0][2] = 1.1;

  X[0][1][0] = 2.0;
  X[0][1][1] = -2.1;
  X[0][1][2] = -2.0;

  std::vector<double> test_in{0.2};

  // -x
  double test_out = -test_in[0];

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));

  // first derivative
  BOOST_TEST(Y[0][0][0] == -X[0][0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] == -X[0][1][0], tt::tolerance(tol));

  // second derivative
  BOOST_TEST(Y[0][0][1] == -X[0][0][1], tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] == -X[0][1][1], tt::tolerance(tol));

  // third derivative
  BOOST_TEST(Y[0][0][2] == -X[0][0][2], tt::tolerance(tol));
  BOOST_TEST(Y[0][1][2] == -X[0][1][2], tt::tolerance(tol));
  myfree3(X);
  myfree3(Y);
}

BOOST_AUTO_TEST_CASE(Atan2Operator_HOV_Forward) {
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 2;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{0.2, 0.4};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // atan2(x, y)
  trace_on(tapeId10);
  indep[0] <<= in[0];
  indep[1] <<= in[1];
  adouble dep = atan2(indep[0], indep[1]);
  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree);
  double ***Y = myalloc3(dim_out, num_dirs, degree);

  X[0][0][0] = 1.0;
  X[0][0][1] = -1.0;
  X[0][0][2] = 1.1;

  X[1][0][0] = 1.1;
  X[1][0][1] = -1.1;
  X[1][0][2] = 1.2;

  X[0][1][0] = 2.0;
  X[0][1][1] = -2.1;
  X[0][1][2] = -2.0;

  X[0][1][0] = 2.1;
  X[0][1][1] = -2.2;
  X[0][1][2] = -2.1;

  std::vector<double> test_in{0.2, 0.4};

  // atan2(x, y)
  double test_out = std::atan2(test_in[0], test_in[1]);

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));

  // first derivative
  BOOST_TEST(
      Y[0][0][0] ==
          test_in[1] / (std::pow(test_in[0], 2) + std::pow(test_in[1], 2)) *
                  X[0][0][0] -
              test_in[0] / (std::pow(test_in[0], 2) + std::pow(test_in[1], 2)) *
                  X[1][0][0],
      tt::tolerance(tol));
  BOOST_TEST(
      Y[0][1][0] ==
          test_in[1] / (std::pow(test_in[0], 2) + std::pow(test_in[1], 2)) *
                  X[0][1][0] -
              test_in[0] / (std::pow(test_in[0], 2) + std::pow(test_in[1], 2)) *
                  X[1][1][0],
      tt::tolerance(tol));

  // second derivative
  BOOST_TEST(
      Y[0][0][1] ==
          test_in[1] / (std::pow(test_in[0], 2) + std::pow(test_in[1], 2)) *
                  X[0][0][1] -
              test_in[0] / (std::pow(test_in[0], 2) + std::pow(test_in[1], 2)) *
                  X[1][0][1] +
              1.0 / 2.0 *
                  (-2.0 * test_in[0] * test_in[1] /
                       std::pow(std::pow(test_in[0], 2) +
                                    std::pow(test_in[1], 2),
                                2) *
                       X[0][0][0] * X[0][0][0] +
                   2.0 * (std::pow(test_in[0], 2) - std::pow(test_in[1], 2)) /
                       std::pow(std::pow(test_in[0], 2) +
                                    std::pow(test_in[1], 2),
                                2) *
                       X[1][0][0] * X[0][0][0] +
                   2.0 * test_in[0] * test_in[1] /
                       std::pow(std::pow(test_in[0], 2) +
                                    std::pow(test_in[1], 2),
                                2) *
                       X[1][0][0] * X[1][0][0]),
      tt::tolerance(tol));
  BOOST_TEST(
      Y[0][1][1] ==
          test_in[1] / (std::pow(test_in[0], 2) + std::pow(test_in[1], 2)) *
                  X[0][1][1] -
              test_in[0] / (std::pow(test_in[0], 2) + std::pow(test_in[1], 2)) *
                  X[1][1][1] +
              1.0 / 2.0 *
                  (-2.0 * test_in[0] * test_in[1] /
                       std::pow(std::pow(test_in[0], 2) +
                                    std::pow(test_in[1], 2),
                                2) *
                       X[0][1][0] * X[0][1][0] +
                   2.0 * (std::pow(test_in[0], 2) - std::pow(test_in[1], 2)) /
                       std::pow(std::pow(test_in[0], 2) +
                                    std::pow(test_in[1], 2),
                                2) *
                       X[1][1][0] * X[0][1][0] +
                   2.0 * test_in[0] * test_in[1] /
                       std::pow(std::pow(test_in[0], 2) +
                                    std::pow(test_in[1], 2),
                                2) *
                       X[1][1][0] * X[1][1][0]),
      tt::tolerance(tol));

  // third derivative
  BOOST_TEST(
      Y[0][0][2] ==
          test_in[1] / (std::pow(test_in[0], 2) + std::pow(test_in[1], 2)) *
                  X[0][0][2] -
              test_in[0] / (std::pow(test_in[0], 2) + std::pow(test_in[1], 2)) *
                  X[1][0][2] +
              (-2.0 * test_in[0] * test_in[1] /
                   std::pow(std::pow(test_in[0], 2) + std::pow(test_in[1], 2),
                            2) *
                   X[0][0][0] * X[0][0][1] +
               (std::pow(test_in[0], 2) - std::pow(test_in[1], 2)) /
                   std::pow(std::pow(test_in[0], 2) + std::pow(test_in[1], 2),
                            2) *
                   X[1][0][0] * X[0][0][1] +
               (std::pow(test_in[0], 2) - std::pow(test_in[1], 2)) /
                   std::pow(std::pow(test_in[0], 2) + std::pow(test_in[1], 2),
                            2) *
                   X[1][0][1] * X[0][0][0] +
               2.0 * test_in[0] * test_in[1] /
                   std::pow(std::pow(test_in[0], 2) + std::pow(test_in[1], 2),
                            2) *
                   X[1][0][0] * X[1][0][1]) +
              1.0 / 6.0 *
                  (-2.0 * test_in[1] *
                       (std::pow(test_in[1], 2) -
                        3.0 * std::pow(test_in[0], 2)) *
                       std::pow(X[0][0][0], 3) -
                   6.0 * test_in[0] *
                       (std::pow(test_in[0], 2) -
                        3.0 * std::pow(test_in[1], 2)) *
                       std::pow(X[0][0][0], 2) * X[1][0][0] +
                   6.0 * test_in[1] *
                       (std::pow(test_in[1], 2) -
                        3.0 * std::pow(test_in[0], 2)) *
                       std::pow(X[1][0][0], 2) * X[0][0][0] +
                   2.0 * test_in[0] *
                       (std::pow(test_in[0], 2) -
                        3.0 * std::pow(test_in[1], 2)) *
                       std::pow(X[1][0][0], 3)) /
                  std::pow(std::pow(test_in[0], 2) + std::pow(test_in[1], 2),
                           3),
      tt::tolerance(tol));
  BOOST_TEST(
      Y[0][1][2] ==
          test_in[1] / (std::pow(test_in[0], 2) + std::pow(test_in[1], 2)) *
                  X[0][1][2] -
              test_in[0] / (std::pow(test_in[0], 2) + std::pow(test_in[1], 2)) *
                  X[1][1][2] +
              (-2.0 * test_in[0] * test_in[1] /
                   std::pow(std::pow(test_in[0], 2) + std::pow(test_in[1], 2),
                            2) *
                   X[0][1][0] * X[0][1][1] +
               (std::pow(test_in[0], 2) - std::pow(test_in[1], 2)) /
                   std::pow(std::pow(test_in[0], 2) + std::pow(test_in[1], 2),
                            2) *
                   X[1][1][0] * X[0][1][1] +
               (std::pow(test_in[0], 2) - std::pow(test_in[1], 2)) /
                   std::pow(std::pow(test_in[0], 2) + std::pow(test_in[1], 2),
                            2) *
                   X[1][1][1] * X[0][1][0] +
               2.0 * test_in[0] * test_in[1] /
                   std::pow(std::pow(test_in[0], 2) + std::pow(test_in[1], 2),
                            2) *
                   X[1][1][0] * X[1][1][1]) +
              1.0 / 6.0 *
                  (-2.0 * test_in[1] *
                       (std::pow(test_in[1], 2) -
                        3.0 * std::pow(test_in[0], 2)) *
                       std::pow(X[0][1][0], 3) -
                   6.0 * test_in[0] *
                       (std::pow(test_in[0], 2) -
                        3.0 * std::pow(test_in[1], 2)) *
                       std::pow(X[0][1][0], 2) * X[1][1][0] +
                   6.0 * test_in[1] *
                       (std::pow(test_in[1], 2) -
                        3.0 * std::pow(test_in[0], 2)) *
                       std::pow(X[1][1][0], 2) * X[0][1][0] +
                   2.0 * test_in[0] *
                       (std::pow(test_in[0], 2) -
                        3.0 * std::pow(test_in[1], 2)) *
                       std::pow(X[1][1][0], 3)) /
                  std::pow(std::pow(test_in[0], 2) + std::pow(test_in[1], 2),
                           3),
      tt::tolerance(tol));
  myfree3(X);
  myfree3(Y);
}

BOOST_AUTO_TEST_CASE(Pow_Operator_HOV_Forward_1) {
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{0.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // x^y
  trace_on(tapeId10);
  indep[0] <<= in[0];
  adouble dep = pow(indep[0], 3.2);
  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree);
  double ***Y = myalloc3(dim_out, num_dirs, degree);

  X[0][0][0] = 1.0;
  X[0][0][1] = -1.0;
  X[0][0][2] = 1.1;

  X[0][1][0] = 2.0;
  X[0][1][1] = -2.1;
  X[0][1][2] = -2.0;

  std::vector<double> test_in{0.2};

  // x^y
  double test_out = std::pow(test_in[0], 3.2);

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));

  // first derivative
  BOOST_TEST(Y[0][0][0] == std::pow(test_in[0], 2.2) * 3.2 * X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] == std::pow(test_in[0], 2.2) * 3.2 * X[0][1][0],
             tt::tolerance(tol));

  // second derivative
  BOOST_TEST(Y[0][0][1] == std::pow(test_in[0], 2.2) * 3.2 * X[0][0][1] +
                               1.0 / 2.0 * std::pow(test_in[0], 1.2) * 7.04 *
                                   X[0][0][0] * X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] == std::pow(test_in[0], 2.2) * 3.2 * X[0][1][1] +
                               1.0 / 2.0 * std::pow(test_in[0], 1.2) * 7.04 *
                                   X[0][1][0] * X[0][1][0],
             tt::tolerance(tol));

  // third derivative
  BOOST_TEST(Y[0][0][2] == std::pow(test_in[0], 2.2) * 3.2 * X[0][0][2] +
                               std::pow(test_in[0], 1.2) * 7.04 * X[0][0][0] *
                                   X[0][0][1] +
                               1.0 / 6.0 * std::pow(test_in[0], 0.2) * 8.448 *
                                   X[0][0][0] * X[0][0][0] * X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][2] == std::pow(test_in[0], 2.2) * 3.2 * X[0][1][2] +
                               std::pow(test_in[0], 1.2) * 7.04 * X[0][1][0] *
                                   X[0][1][1] +
                               1.0 / 6.0 * std::pow(test_in[0], 0.2) * 8.448 *
                                   X[0][1][0] * X[0][1][0] * X[0][1][0],
             tt::tolerance(tol));
  myfree3(X);
  myfree3(Y);
}
BOOST_AUTO_TEST_CASE(PowOperator_HOV_Forward_2) {
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 2;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{0.2, 2.0};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // x^y
  trace_on(tapeId10);
  indep[0] <<= in[0];
  indep[1] <<= in[1];
  adouble dep = pow(indep[0], indep[1]);
  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree);
  double ***Y = myalloc3(dim_out, num_dirs, degree);

  X[0][0][0] = 1.0;
  X[0][0][1] = -1.0;
  X[0][0][2] = 1.1;

  X[0][1][0] = 2.0;
  X[0][1][1] = -2.1;
  X[0][1][2] = -2.0;

  X[1][0][0] = 1.0;
  X[1][0][1] = -1.0;
  X[1][0][2] = 1.1;

  X[1][1][0] = 2.0;
  X[1][1][1] = -2.1;
  X[1][1][2] = -2.0;

  std::vector<double> test_in{0.2, 2.0};

  // x^y
  double test_out = std::pow(test_in[0], test_in[1]);

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));

  // first derivative
  BOOST_TEST(Y[0][0][0] == std::pow(test_in[0], test_in[1] - 1.0) * test_in[1] *
                                   X[0][0][0] +
                               std::pow(test_in[0], test_in[1]) *
                                   std::log(test_in[0]) * X[1][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] == std::pow(test_in[0], test_in[1] - 1.0) * test_in[1] *
                                   X[0][1][0] +
                               std::pow(test_in[0], test_in[1]) *
                                   std::log(test_in[0]) * X[1][1][0],
             tt::tolerance(tol));

  // second derivative
  BOOST_TEST(
      Y[0][0][1] ==
          std::pow(test_in[0], test_in[1] - 1.0) * test_in[1] * X[0][0][1] +
              std::pow(test_in[0], test_in[1]) * std::log(test_in[0]) *
                  X[1][0][1] +
              1.0 / 2.0 *
                  (std::pow(test_in[0], test_in[1] - 2.0) * (test_in[1] - 1.0) *
                       test_in[1] * X[0][0][0] * X[0][0][0] +
                   2.0 * std::pow(test_in[0], test_in[1] - 1.0) *
                       (1.0 + test_in[1] * std::log(test_in[0])) * X[0][0][0] *
                       X[1][0][0] +
                   std::pow(test_in[0], test_in[1]) * std::log(test_in[0]) *
                       std::log(test_in[0]) * X[1][0][0] * X[1][0][0]),
      tt::tolerance(tol));
  BOOST_TEST(
      Y[0][1][1] ==
          std::pow(test_in[0], test_in[1] - 1.0) * test_in[1] * X[0][1][1] +
              std::pow(test_in[0], test_in[1]) * std::log(test_in[0]) *
                  X[1][1][1] +
              1.0 / 2.0 *
                  (std::pow(test_in[0], test_in[1] - 2.0) * (test_in[1] - 1.0) *
                       test_in[1] * X[0][1][0] * X[0][1][0] +
                   2.0 * std::pow(test_in[0], test_in[1] - 1.0) *
                       (1.0 + test_in[1] * std::log(test_in[0])) * X[0][1][0] *
                       X[1][1][0] +
                   std::pow(test_in[0], test_in[1]) * std::log(test_in[0]) *
                       std::log(test_in[0]) * X[1][1][0] * X[1][1][0]),
      tt::tolerance(tol));

  // third derivative
  BOOST_TEST(
      Y[0][0][2] ==
          std::pow(test_in[0], test_in[1] - 1.0) * test_in[1] * X[0][0][2] +
              std::pow(test_in[0], test_in[1]) * std::log(test_in[0]) *
                  X[1][0][2] +
              (std::pow(test_in[0], test_in[1] - 2.0) * (test_in[1] - 1.0) *
                   test_in[1] * X[0][0][0] * X[0][0][1] +
               2.0 * std::pow(test_in[0], test_in[1] - 1.0) *
                   (1.0 + test_in[1] * std::log(test_in[0])) * X[0][0][0] *
                   X[1][0][1] +
               std::pow(test_in[0], test_in[1]) * std::log(test_in[0]) *
                   std::log(test_in[0]) * X[1][0][0] * X[1][0][1]) +
              1.0 / 6.0 *
                  ((test_in[1] - 2.0) * (test_in[1] - 1.0) * test_in[1] *
                       std::pow(test_in[0], test_in[1] - 3.0) *
                       std::pow(X[0][0][0], 3) +
                   3.0 * std::pow(test_in[0], test_in[1] - 2.0) *
                       ((test_in[1] - 1.0) * test_in[1] * std::log(test_in[0]) +
                        2.0 * test_in[1] - 1) *
                       X[0][0][0] * X[0][0][0] * X[1][0][0] +
                   3.0 * std::pow(test_in[0], test_in[1] - 1.0) *
                       std::log(test_in[0]) *
                       (test_in[1] * std::log(test_in[0]) + 2.0) * X[0][0][0] *
                       X[1][0][0] * X[1][0][0] +
                   std::pow(test_in[0], test_in[1]) *
                       std::pow(std::log(test_in[0]), 3) *
                       std::pow(X[1][0][0], 3)),
      tt::tolerance(tol));
  BOOST_TEST(
      Y[0][1][2] ==
          std::pow(test_in[0], test_in[1] - 1.0) * test_in[1] * X[0][1][2] +
              std::pow(test_in[0], test_in[1]) * std::log(test_in[0]) *
                  X[1][1][2] +
              (std::pow(test_in[0], test_in[1] - 2.0) * (test_in[1] - 1.0) *
                   test_in[1] * X[0][1][0] * X[0][1][1] +
               2.0 * std::pow(test_in[0], test_in[1] - 1.0) *
                   (1.0 + test_in[1] * std::log(test_in[0])) * X[0][1][0] *
                   X[1][1][1] +
               std::pow(test_in[0], test_in[1]) * std::log(test_in[0]) *
                   std::log(test_in[0]) * X[1][1][0] * X[1][1][1]) +
              1.0 / 6.0 *
                  ((test_in[1] - 2.0) * (test_in[1] - 1.0) * test_in[1] *
                       std::pow(test_in[0], test_in[1] - 3.0) *
                       std::pow(X[0][1][0], 3) +
                   3.0 * std::pow(test_in[0], test_in[1] - 2.0) *
                       ((test_in[1] - 1.0) * test_in[1] * std::log(test_in[0]) +
                        2.0 * test_in[1] - 1) *
                       X[0][1][0] * X[0][1][0] * X[1][1][0] +
                   3.0 * std::pow(test_in[0], test_in[1] - 1.0) *
                       std::log(test_in[0]) *
                       (test_in[1] * std::log(test_in[0]) + 2.0) * X[0][1][0] *
                       X[1][1][0] * X[1][1][0] +
                   std::pow(test_in[0], test_in[1]) *
                       std::pow(std::log(test_in[0]), 3) *
                       std::pow(X[1][1][0], 3)),
      tt::tolerance(tol));
  myfree3(X);
  myfree3(Y);
}

BOOST_AUTO_TEST_CASE(PowOperator_HOV_Forward_3) {
  setCurrentTape(tapeId10);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{2.1};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // x^y
  trace_on(tapeId10);
  indep[0] <<= in[0];
  adouble dep = pow(1.5, indep[0]);
  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree);
  double ***Y = myalloc3(dim_out, num_dirs, degree);

  X[0][0][0] = 1.0;
  X[0][0][1] = -1.0;
  X[0][0][2] = 1.1;

  X[0][1][0] = 2.0;
  X[0][1][1] = -2.1;
  X[0][1][2] = -2.0;

  std::vector<double> test_in{2.1};

  // x^y
  double test_out = std::pow(1.5, test_in[0]);

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));

  const double log_x = std::log(1.5);
  // first derivative
  BOOST_TEST(Y[0][0][0] == std::pow(1.5, test_in[0]) * log_x * X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] == std::pow(1.5, test_in[0]) * log_x * X[0][1][0],
             tt::tolerance(tol));

  // second derivative
  BOOST_TEST(Y[0][0][1] == std::pow(1.5, test_in[0]) * log_x * X[0][0][1] +
                               1.0 / 2.0 * std::pow(1.5, test_in[0]) * log_x *
                                   log_x * X[0][0][0] * X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] == std::pow(1.5, test_in[0]) * log_x * X[0][1][1] +
                               1.0 / 2.0 * std::pow(1.5, test_in[0]) * log_x *
                                   log_x * X[0][1][0] * X[0][1][0],
             tt::tolerance(tol));

  // third derivative
  BOOST_TEST(Y[0][0][2] == std::pow(1.5, test_in[0]) * log_x * X[0][0][2] +
                               std::pow(1.5, test_in[0]) * log_x * log_x *
                                   X[0][0][0] * X[0][0][1] +
                               1.0 / 6.0 * std::pow(1.5, test_in[0]) * log_x *
                                   log_x * log_x * X[0][0][0] * X[0][0][0] *
                                   X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][2] == std::pow(1.5, test_in[0]) * log_x * X[0][1][2] +
                               std::pow(1.5, test_in[0]) * log_x * log_x *
                                   X[0][1][0] * X[0][1][1] +
                               1.0 / 6.0 * std::pow(1.5, test_in[0]) * log_x *
                                   log_x * log_x * X[0][1][0] * X[0][1][0] *
                                   X[0][1][0],
             tt::tolerance(tol));
  myfree3(X);
  myfree3(Y);
}

/*
BOOST_AUTO_TEST_CASE(CbrtOperator_HOV_Forward) {

  const size_t dim_out = 1;
  const size_t dim_in = 1;
  const size_t degree = 3;
  const size_t num_dirs = 2;
  const short keep = 1;
  std::vector<double> in{0.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // cbrt(x)
  trace_on(tapeId10);
  indep[0] <<= in[0];
  adouble dep = cbrt(indep[0]);
  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree);
  double ***Y = myalloc3(dim_out, num_dirs, degree);

  X[0][0][0] = 1.0;
  X[0][0][1] = -1.0;
  X[0][0][2] = 1.1;

  X[0][1][0] = 2.0;
  X[0][1][1] = -2.1;
  X[0][1][2] = -2.0;

  std::vector<double> test_in{0.5};

  // cbrt(x)
  double test_out = std::cbrt(test_in[0]);

  hov_wk_forward(tapeId10, dim_out, dim_in, degree, keep, num_dirs,
test_in.data(), X, out.data(), Y);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));

  // first derivative
  BOOST_TEST(Y[0][0][0] ==
                 1.0 / (3.0 * std::cbrt(test_in[0]) * std::cbrt(test_in[0])) *
                     X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] ==
                 1.0 / (3.0 * std::cbrt(test_in[0]) * std::cbrt(test_in[0])) *
                     X[0][1][0],
             tt::tolerance(tol));

  // second derivative
  BOOST_TEST(Y[0][0][1] ==
                 1.0 / (3.0 * std::cbrt(test_in[0]) * std::cbrt(test_in[0])) *
                         X[0][0][1] +
                     1.0 / 2.0 *
                         (-2.0 / (9.0 * std::pow(test_in[0], 5.0 / 3.0)) *
                          X[0][0][0] * X[0][0][0]),
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][1] ==
                 1.0 / (3.0 * std::cbrt(test_in[0]) * std::cbrt(test_in[0])) *
                         X[0][1][1] +
                     1.0 / 2.0 *
                         (-2.0 / (9.0 * std::pow(test_in[0], 5.0 / 3.0)) *
                          X[0][1][0] * X[0][1][0]),
             tt::tolerance(tol));

  // third derivative
  BOOST_TEST(Y[0][0][2] ==
                 1.0 / (3.0 * std::cbrt(test_in[0]) * std::cbrt(test_in[0])) *
                         X[0][0][2] +
                     (-2.0 / (9.0 * std::pow(test_in[0], 5.0 / 3.0)) *
                      X[0][0][0] * X[0][0][1]) +
                     1.0 / 6.0 *
                         (10.0 / (27.0 * std::pow(test_in[0], 8.0 / 3.0)) *
                          std::pow(X[0][0][0], 3)),
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][2] ==
                 1.0 / (2.0 * std::cbrt(test_in[0]) * std::cbrt(test_in[0])) *
                         X[0][1][2] +
                     (-2.0 / (9.0 * std::pow(test_in[0], 5.0 / 3.0)) *
                      X[0][1][0] * X[0][1][1]) +
                     1.0 / 6.0 *
                         (10.0 / (27.0 * std::pow(test_in[0], 8.0 / 3.0)) *
                          std::pow(X[0][1][0], 3)),
             tt::tolerance(tol));
  myfree3(X);
  myfree3(Y);
}
*/
BOOST_AUTO_TEST_SUITE_END()
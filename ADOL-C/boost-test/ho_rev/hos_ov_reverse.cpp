
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

#include <adolc/adolc.h>

#include "../const.h"

#include <array>
#include <numeric>
#include <vector>

BOOST_AUTO_TEST_SUITE(test_hos_ov_forward)

const short tapeId165 = 165;
struct TapeInitializer {
  TapeInitializer() { createNewTape(tapeId165); }
};

BOOST_GLOBAL_FIXTURE(TapeInitializer);

BOOST_AUTO_TEST_CASE(PlusOperator_HOS_OV_REVERSE) {
  setCurrentTape(tapeId165);
  const size_t dim_out = 1;
  const size_t dim_in = 2;
  const size_t degree_hov_forward = 1;
  const size_t degree_hos_reverse = 1;
  const size_t num_dirs = 3;
  const short keep = 2;
  std::vector<double> in{4.0, 3.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  trace_on(tapeId165);
  for (auto i = 0; i < in.size(); ++i)
    indep[i] <<= in[i];

  // x^2 + y^3
  adouble dep = pow(indep[0], 2) + pow(indep[1], 3);

  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree_hov_forward);
  double ***Y = myalloc3(dim_out, num_dirs, degree_hov_forward);

  double **U = myalloc2(dim_out, degree_hos_reverse + 1);
  double ***Z = myalloc3(num_dirs, dim_in, degree_hos_reverse + 1);

  X[0][0][0] = 1.2;
  X[1][0][0] = 1.9;

  X[0][1][0] = 2.0;
  X[1][1][0] = 3.0;

  X[0][2][0] = 1.0;
  X[1][2][0] = -1.0;

  U[0][0] = 1.0;
  U[0][1] = -1.3;

  std::vector<double> test_in{2.0, 3.2};
  // x^2 + y^3)
  double test_out = std::pow(test_in[0], 2) + std::pow(test_in[1], 3);
  hov_wk_forward(tapeId165, dim_out, dim_in, degree_hov_forward, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

  hos_ov_reverse(tapeId165, dim_out, dim_in, degree_hos_reverse, num_dirs, U,
                 Z);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));
  BOOST_TEST(Y[0][0][0] == 2 * test_in[0] * X[0][0][0] +
                               3 * std::pow(test_in[1], 2) * X[1][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] == 2 * test_in[0] * X[0][1][0] +
                               3 * std::pow(test_in[1], 2) * X[1][1][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][2][0] == 2 * test_in[0] * X[0][2][0] +
                               3 * std::pow(test_in[1], 2) * X[1][2][0],
             tt::tolerance(tol));

  BOOST_TEST(Z[0][0][0] == U[0][0] * 2 * test_in[0], tt::tolerance(tol));
  BOOST_TEST(Z[0][1][0] == U[0][0] * 3 * std::pow(test_in[1], 2),
             tt::tolerance(tol));

  BOOST_TEST(Z[1][0][0] == U[0][0] * (2.0 * test_in[0]), tt::tolerance(tol));
  BOOST_TEST(Z[1][1][0] == U[0][0] * 3 * std::pow(test_in[1], 2),
             tt::tolerance(tol));

  BOOST_TEST(Z[2][0][0] == U[0][0] * (2.0 * test_in[0]), tt::tolerance(tol));
  BOOST_TEST(Z[2][1][0] == U[0][0] * 3 * std::pow(test_in[1], 2),
             tt::tolerance(tol));

  BOOST_TEST(Z[0][0][1] ==
                 U[0][1] * 2.0 * test_in[0] + U[0][0] * 2.0 * X[0][0][0],
             tt::tolerance(tol));

  BOOST_TEST(Z[0][1][1] == U[0][1] * 3 * std::pow(test_in[1], 2) +
                               U[0][0] * 6 * test_in[1] * X[1][0][0],
             tt::tolerance(tol));

  BOOST_TEST(Z[1][0][1] ==
                 U[0][1] * 2.0 * test_in[0] + U[0][0] * 2.0 * X[0][1][0],
             tt::tolerance(tol));
  BOOST_TEST(Z[1][1][1] == U[0][1] * 3 * std::pow(test_in[1], 2) +
                               U[0][0] * 6 * test_in[1] * X[1][1][0],
             tt::tolerance(tol));

  BOOST_TEST(Z[2][0][1] ==
                 U[0][1] * 2.0 * test_in[0] + U[0][0] * 2.0 * X[0][2][0],
             tt::tolerance(tol));
  BOOST_TEST(Z[2][1][1] == U[0][1] * 3 * std::pow(test_in[1], 2) +
                               U[0][0] * 6 * test_in[1] * X[1][2][0],
             tt::tolerance(tol));
  myfree3(X);
  myfree3(Y);
  myfree2(U);
  myfree3(Z);
}

BOOST_AUTO_TEST_CASE(MinOperator_HOS_OV_REVERSE) {
  setCurrentTape(tapeId165);
  const size_t dim_out = 1;
  const size_t dim_in = 2;
  const size_t degree_hov_forward = 1;
  const size_t degree_hos_reverse = 1;
  const size_t num_dirs = 3;
  const short keep = 2;
  std::vector<double> in{4.0, 3.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  trace_on(tapeId165);
  for (auto i = 0; i < in.size(); ++i)
    indep[i] <<= in[i];

  // min(x^2, y^3)
  adouble dep = min(pow(indep[0], 2), pow(indep[1], 3));

  dep >>= out[0];
  trace_off();

  double ***X = myalloc3(dim_in, num_dirs, degree_hov_forward);
  double ***Y = myalloc3(dim_out, num_dirs, degree_hov_forward);

  double **U = myalloc2(dim_out, degree_hos_reverse + 1);
  double ***Z = myalloc3(num_dirs, dim_in, degree_hos_reverse + 1);

  X[0][0][0] = 1.2;
  X[1][0][0] = 1.9;

  X[0][1][0] = 2.0;
  X[1][1][0] = 3.0;

  X[0][2][0] = 1.0;
  X[1][2][0] = -1.0;

  U[0][0] = 1.0;
  U[0][1] = -1.3;

  /****************************
  TEST X < Y
  *****************************/
  std::vector<double> test_in{2.0, 3.2};
  // min(x^2, y^3)
  double test_out = std::min(std::pow(test_in[0], 2), std::pow(test_in[1], 3));

  hov_wk_forward(tapeId165, dim_out, dim_in, degree_hov_forward, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

  hos_ov_reverse(tapeId165, dim_out, dim_in, degree_hos_reverse, num_dirs, U,
                 Z);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));
  BOOST_TEST(Y[0][0][0] == 2 * test_in[0] * X[0][0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] == 2 * test_in[0] * X[0][1][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][2][0] == 2 * test_in[0] * X[0][2][0], tt::tolerance(tol));

  BOOST_TEST(Z[0][0][0] == U[0][0] * 2 * test_in[0], tt::tolerance(tol));
  BOOST_TEST(Z[0][1][0] == 0.0, tt::tolerance(tol));

  BOOST_TEST(Z[1][0][0] == U[0][0] * (2.0 * test_in[0]), tt::tolerance(tol));
  BOOST_TEST(Z[1][1][0] == 0.0, tt::tolerance(tol));

  BOOST_TEST(Z[2][0][0] == U[0][0] * (2.0 * test_in[0]), tt::tolerance(tol));
  BOOST_TEST(Z[2][1][0] == 0.0, tt::tolerance(tol));

  BOOST_TEST(Z[0][0][1] ==
                 U[0][1] * 2.0 * test_in[0] + U[0][0] * 2.0 * X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Z[0][1][1] == 0.0, tt::tolerance(tol));

  BOOST_TEST(Z[1][0][1] ==
                 U[0][1] * 2.0 * test_in[0] + U[0][0] * 2.0 * X[0][1][0],
             tt::tolerance(tol));
  BOOST_TEST(Z[1][1][1] == 0.0, tt::tolerance(tol));

  BOOST_TEST(Z[2][0][1] ==
                 U[0][1] * 2.0 * test_in[0] + U[0][0] * 2.0 * X[0][2][0],
             tt::tolerance(tol));
  BOOST_TEST(Z[2][1][1] == 0.0, tt::tolerance(tol));

  /**************************
  TEST X > Y
  ***************************/

  // current: test_in[0] = 2.0
  test_in[1] = 1.0;
  // min(x^2, y^3)
  test_out = std::min(std::pow(test_in[0], 2), std::pow(test_in[1], 3));
  hov_wk_forward(tapeId165, dim_out, dim_in, degree_hov_forward, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

  hos_ov_reverse(tapeId165, dim_out, dim_in, degree_hos_reverse, num_dirs, U,
                 Z);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));
  BOOST_TEST(Y[0][0][0] == 3 * std::pow(test_in[1], 2) * X[1][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] == 3 * std::pow(test_in[1], 2) * X[1][1][0],
             tt::tolerance(tol));
  BOOST_TEST(Y[0][2][0] == 3 * std::pow(test_in[1], 2) * X[1][2][0],
             tt::tolerance(tol));

  BOOST_TEST(Z[0][0][0] == 0.0, tt::tolerance(tol));
  BOOST_TEST(Z[0][1][0] == U[0][0] * 3 * std::pow(test_in[1], 2),
             tt::tolerance(tol));

  BOOST_TEST(Z[1][0][0] == 0.0, tt::tolerance(tol));
  BOOST_TEST(Z[1][1][0] == U[0][0] * 3 * std::pow(test_in[1], 2),
             tt::tolerance(tol));

  BOOST_TEST(Z[2][0][0] == 0.0, tt::tolerance(tol));
  BOOST_TEST(Z[2][1][0] == U[0][0] * 3 * std::pow(test_in[1], 2),
             tt::tolerance(tol));

  BOOST_TEST(Z[0][0][1] == 0.0, tt::tolerance(tol));
  BOOST_TEST(Z[0][1][1] == U[0][1] * 3 * std::pow(test_in[1], 2) +
                               U[0][0] * 6 * test_in[1] * X[1][0][0],
             tt::tolerance(tol));

  BOOST_TEST(Z[1][0][1] == 0.0, tt::tolerance(tol));
  BOOST_TEST(Z[1][1][1] == U[0][1] * 3 * std::pow(test_in[1], 2) +
                               U[0][0] * 6 * test_in[1] * X[1][1][0],
             tt::tolerance(tol));

  BOOST_TEST(Z[2][0][1] == 0.0, tt::tolerance(tol));
  BOOST_TEST(Z[2][1][1] == U[0][1] * 3 * std::pow(test_in[1], 2) +
                               U[0][0] * 6 * test_in[1] * X[1][2][0],
             tt::tolerance(tol));

  /**************************
    TEST A == B
  ***************************
  test_in[0] = 1.0;
  test_in[1] = 1.0;
  // min(x^2, y^3)
  test_out = std::min(std::pow(test_in[0], 2), std::pow(test_in[1], 3));
  std::cout << "tie point" << std::endl;
  hov_wk_forward(tapeId165, dim_out, dim_in, degree_hov_forward, keep, num_dirs,
                 test_in.data(), X, out.data(), Y);

  hos_ov_reverse(tapeId165, dim_out, dim_in, degree_hos_reverse, num_dirs, U,
                 Z);

  BOOST_TEST(out[0] == test_out, tt::tolerance(tol));
  BOOST_TEST(Y[0][0][0] == 2 * test_in[0] * X[0][0][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][1][0] == 2 * test_in[0] * X[0][1][0], tt::tolerance(tol));
  BOOST_TEST(Y[0][2][0] == 3 * std::pow(test_in[1], 2) * X[1][2][0],
             tt::tolerance(tol));

  BOOST_TEST(Z[0][0][0] == U[0][0] * 2 * test_in[0], tt::tolerance(tol));
  BOOST_TEST(Z[0][1][0] == 0, tt::tolerance(tol));

  BOOST_TEST(Z[1][0][0] == U[0][0] * 2 * test_in[0], tt::tolerance(tol));
  BOOST_TEST(Z[1][1][0] == 0, tt::tolerance(tol));

  BOOST_TEST(Z[2][0][0] == U[0][0] * 2 * test_in[0], tt::tolerance(tol));
  BOOST_TEST(Z[2][1][0] == 0.0, tt::tolerance(tol));

  BOOST_TEST(Z[0][0][1] ==
                 U[0][1] * 2.0 * test_in[0] + U[0][0] * 2.0 * X[0][0][0],
             tt::tolerance(tol));
  BOOST_TEST(Z[0][1][1] == 0.0, tt::tolerance(tol));

  BOOST_TEST(Z[1][0][1] ==
                 U[0][1] * 2.0 * test_in[0] + U[0][0] * 2.0 * X[0][1][0],
             tt::tolerance(tol));
  BOOST_TEST(Z[1][1][1] == 0.0, tt::tolerance(tol));

  BOOST_TEST(Z[2][0][1] ==
                 U[0][1] * 2.0 * test_in[0] + U[0][0] * 2.0 * X[0][2][0],
             tt::tolerance(tol));
  BOOST_TEST(Z[2][1][1] == 0.0, tt::tolerance(tol));
    */
  myfree3(X);
  myfree3(Y);
  myfree2(U);
  myfree3(Z);
}

BOOST_AUTO_TEST_SUITE_END()

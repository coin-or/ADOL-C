#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

#include "../const.h"
#include <adolc/adolc.h>
#include <array>

BOOST_AUTO_TEST_SUITE(test_tensor_eval)

BOOST_AUTO_TEST_CASE(TensorEval_Repeatable_And_JacobianConsistent) {
  const short tapeId = createNewTape();
  const int n = 2;
  const int m = 2;
  const int p = 2;
  const int d = 1;

  std::array<double, n> x0{1.25, -0.7};
  std::array<double, m> y0{};

  trace_on(tapeId);
  {
    std::array<adouble, n> x;
    std::array<adouble, m> y;
    for (int i = 0; i < n; ++i) {
      x[i] <<= x0[i];
    }

    y[0] = x[0] * x[0] + 2.0 * x[0] * x[1] + 3.0 * x[1];
    y[1] = sin(x[0]) + x[1] * x[1];

    for (int i = 0; i < m; ++i) {
      y[i] >>= y0[i];
    }
  }
  trace_off();

  const size_t dim = binomi(p + d, d);
  double **S = myalloc2(n, p);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < p; ++j) {
      S[i][j] = (i == j) ? 1.0 : 0.0;
    }
  }

  double **tensor1 = myalloc2(m, dim);
  double **tensor2 = myalloc2(m, dim);

  const int rc1 = tensor_eval(tapeId, m, n, d, p, x0.data(), tensor1, S);
  // there was a bug before that caused the second call to get corrupted. This
  // call and corresponding check is to ensure this does not happen again.
  const int rc2 = tensor_eval(tapeId, m, n, d, p, x0.data(), tensor2, S);

  BOOST_TEST(rc1 == rc2);
  BOOST_TEST(rc1 > 0);

  for (int i = 0; i < m; ++i) {
    for (size_t j = 0; j < dim; ++j) {
      BOOST_TEST(tensor1[i][j] == tensor2[i][j], tt::tolerance(tol));
    }
  }

  for (int i = 0; i < m; ++i) {
    BOOST_TEST(tensor1[i][0] == y0[i], tt::tolerance(tol));
  }

  double **J = myalloc2(m, n);
  jacobian(tapeId, m, n, x0.data(), J);

  std::array<int, 1> multi{};
  std::array<double, m> dirDeriv{};

  multi[0] = 1;
  tensor_value(d, m, dirDeriv.data(), tensor1, multi.data());
  for (int i = 0; i < m; ++i) {
    BOOST_TEST(dirDeriv[i] == J[i][0], tt::tolerance(tol));
  }

  multi[0] = 2;
  tensor_value(d, m, dirDeriv.data(), tensor1, multi.data());
  for (int i = 0; i < m; ++i) {
    BOOST_TEST(dirDeriv[i] == J[i][1], tt::tolerance(tol));
  }

  myfree2(J);
  myfree2(tensor1);
  myfree2(tensor2);
  myfree2(S);
}

BOOST_AUTO_TEST_SUITE_END()

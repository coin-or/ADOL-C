#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

#include "../const.h"
#include <adolc/adolc.h> /* use of ALL ADOL-C interfaces */
#include <array>
#include <numeric>

template <typename T>
adouble det(const T &A, size_t row,
            size_t col) // k <= n is the order of the submatrix
{
  if (col == 0)
    return 1.0;
  else {
    adouble t = 0;
    int p = 1;
    int sign;
    if (row % 2)
      sign = 1;
    else
      sign = -1;

    for (auto i = 0; i < A.size(); i++) {
      int p1 = 2 * p;
      if (col % p1 >= p) {
        if (col == p) {
          if (sign > 0)
            t += A[row - 1][i];
          else
            t -= A[row - 1][i];
        } else {
          if (sign > 0)
            t += A[row - 1][i] *
                 det(A, row - 1, col - p); // recursive call to det
          else
            t -= A[row - 1][i] *
                 det(A, row - 1, col - p); // recursive call to det
        }
        sign = -sign;
      }
      p = p1;
    }
    return t;
  }
}

BOOST_AUTO_TEST_SUITE(test_detem_example)
BOOST_AUTO_TEST_CASE(DeterminanteTest) {
  const short tapeId = 1678;
  createNewTape(tapeId);
  setCurrentTape(tapeId);

  const int keep = 1;
  constexpr size_t n = 7;
  int m = 1;

  std::array<std::array<adouble, n>, n> A;

  trace_on(tapeId, keep); // tapeId=1=keep
  double detout = 0.0;
  double diag = 1.0;           // here keep the intermediates for
  for (auto i = 0; i < n; i++) // the subsequent call to reverse
  {
    m *= 2;
    for (auto j = 0; j < n; j++)
      A[i][j] <<= j / (1.0 + i); // make all elements of A independent

    diag += A[i][i].value(); // value(adouble) converts to double
    A[i][i] += 1.0;
  }
  adouble ad = det(A, n, m - 1); // actual function call.

  ad >>= detout;
  BOOST_TEST(detout == diag, tt ::tolerance(tol));
  trace_off();

  std::array<double, 1> u;
  u[0] = 1.0;
  std::array<double, n * n> B;
  reverse(tapeId, 1, n * n, 0, u.data(),
          B.data()); // call reverse to calculate the gradient

  std::array<double, n> res = {5.4071428571428601, 0, 0, 0, 0, 0, 0};
  for (auto i = 0; i < n; i++) {
    adouble sum = 0;
    for (auto j = 0; j < n; j++) // the matrix A times the first n
      sum += A[i][j] * B[j];
    BOOST_TEST(sum.value() == res[i], tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_SUITE_END()
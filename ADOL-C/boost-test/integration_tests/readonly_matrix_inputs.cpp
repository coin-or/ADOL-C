#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "../const.h"
#include <adolc/adolc.h>
#include <adolc/drivers/taylor.h>
#include <array>
#include <stdexcept>
#include <type_traits>

using ADOLC::Identity;
using ADOLC::Matrix;
using ADOLC::Tensor;
using ADOLC::matrixView;

namespace tt = boost::test_tools;

namespace {
template <typename T>
concept WritableForwardInput = requires(T &input, Matrix<double> &output) {
  forward(short{0}, 2, 2, 1, 0, input, output);
};

static_assert(!WritableForwardInput<const Identity<double>>);
static_assert(!WritableForwardInput<const Matrix<double>>);
static_assert(WritableForwardInput<Matrix<double>>);
static_assert(std::is_convertible_v<const Matrix<double> &,
                                    std::span<const double *const>>);
static_assert(std::is_convertible_v<const Matrix<double, 2, 2> &,
                                    std::span<const double *const>>);
static_assert(std::is_convertible_v<const Identity<double> &,
                                    std::span<const double *const>>);
static_assert(
    !std::is_convertible_v<const Identity<double> &, std::span<double *>>);

short recordVectorFunction() {
  const short tag = createNewTape();
  trace_on(tag);
  {
    adouble x0, x1, y0, y1;
    x0 <<= 2.0;
    x1 <<= 3.0;
    y0 = x0 * x0 + x1;
    y1 = x0 + 3.0 * x1 * x1;
    double value;
    y0 >>= value;
    y1 >>= value;
  }
  trace_off();
  return tag;
}

void checkJacobian(const Matrix<double> &J) {
  BOOST_TEST(J[0][0] == 4.0, tt::tolerance(tol));
  BOOST_TEST(J[0][1] == 1.0, tt::tolerance(tol));
  BOOST_TEST(J[1][0] == 1.0, tt::tolerance(tol));
  BOOST_TEST(J[1][1] == 18.0, tt::tolerance(tol));
}

template <typename Seed> void checkForwardAndReverse(const Seed &seed) {
  const short tag = recordVectorFunction();
  const std::array<double, 2> x{2.0, 3.0};
  std::array<double, 2> y{};
  Matrix<double> J(2, 2);
  BOOST_REQUIRE(fov_forward(tag, 2, 2, 2, x, seed, y, J) >= 0);
  checkJacobian(J);
  BOOST_REQUIRE(forward(tag, 2, 2, 2, x, seed, y, J) >= 0);
  checkJacobian(J);
  BOOST_REQUIRE(zos_forward(tag, 2, 2, 1, x, y) >= 0);
  BOOST_REQUIRE(fov_reverse(tag, 2, 2, 2, seed, J) >= 0);
  checkJacobian(J);
  BOOST_REQUIRE(reverse(tag, 2, 2, 0, 2, seed, J) >= 0);
  checkJacobian(J);
  BOOST_REQUIRE(reverse(tag, 2, 2, 2, seed, J) >= 0);
  checkJacobian(J);
}
} // namespace

BOOST_AUTO_TEST_SUITE(readonly_matrix_inputs)

BOOST_AUTO_TEST_CASE(Identity_Rejects_Zero_Dimensions) {
  BOOST_CHECK_THROW(Identity<double>{0}, std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(Identity_And_Const_Dense_Seeds) {
  const Identity<double> identity(2);
  checkForwardAndReverse(identity);
  const Matrix<double> dense = [] {
    Matrix<double> seed(2, 2);
    seed[0][0] = seed[1][1] = 1.0;
    return seed;
  }();
  checkForwardAndReverse(dense);
  const Matrix<double, 2, 2> fixed = [] {
    Matrix<double, 2, 2> seed;
    seed[0][0] = seed[1][1] = 1.0;
    return seed;
  }();
  checkForwardAndReverse(fixed);
  std::array<double, 4> values{1.0, 0.0, 0.0, 1.0};
  const auto view = matrixView<2, 2>(values);
  checkForwardAndReverse(view);
  const std::array<const double *, 2> constRows{values.data(),
                                                values.data() + 2};
  checkForwardAndReverse(constRows);
}

BOOST_AUTO_TEST_CASE(Higher_Order_Matrix_Inputs) {
  const short tag = recordVectorFunction();
  const std::array<double, 2> x{2.0, 3.0};
  std::array<double, 2> y{};
  const Identity<double> identity(2);
  Matrix<double> Y(2, 2), Z(2, 2);
  BOOST_REQUIRE(hos_forward(tag, 2, 2, 2, 0, x, identity, y, Y) >= 0);
  BOOST_TEST(Y[0][0] == 4.0, tt::tolerance(tol));
  BOOST_TEST(Y[0][1] == 2.0, tt::tolerance(tol));
  BOOST_TEST(Y[1][0] == 1.0, tt::tolerance(tol));
  BOOST_TEST(Y[1][1] == 18.0, tt::tolerance(tol));

  BOOST_REQUIRE(hos_forward(tag, 2, 2, 1, 2, x, identity, y, Y) >= 0);
  BOOST_REQUIRE(hos_ti_reverse(tag, 2, 2, 1, identity, Z) >= 0);
  BOOST_TEST(Z[0][0] == 4.0, tt::tolerance(tol));
  BOOST_TEST(Z[0][1] == 3.0, tt::tolerance(tol));
  BOOST_TEST(Z[1][0] == 1.0, tt::tolerance(tol));
  BOOST_TEST(Z[1][1] == 18.0, tt::tolerance(tol));

  Tensor<double> adjoints(2, 2, 2);
  Matrix<short> nonzero(2, 2);
  BOOST_REQUIRE(hov_reverse(tag, 2, 2, 1, 2, identity, adjoints, nonzero) >= 0);
  BOOST_TEST(adjoints[0][0][0] == 4.0, tt::tolerance(tol));
  BOOST_TEST(adjoints[0][0][1] == 2.0, tt::tolerance(tol));
  BOOST_TEST(adjoints[1][1][0] == 18.0, tt::tolerance(tol));
  BOOST_REQUIRE(reverse(tag, 2, 2, 1, 2, identity, adjoints, nonzero) >= 0);
  BOOST_TEST(adjoints[0][0][1] == 2.0, tt::tolerance(tol));

  Tensor<double> directions(2, 2, 1), forwardResult(2, 2, 1);
  directions[0][0][0] = directions[1][1][0] = 1.0;
  BOOST_REQUIRE(
      hov_wk_forward(tag, 2, 2, 1, 2, 2, x, directions, y, forwardResult) >= 0);
  BOOST_REQUIRE(hos_ov_reverse(tag, 2, 2, 1, 2, identity, adjoints) >= 0);
  BOOST_TEST(adjoints[0][0][1] == 3.0, tt::tolerance(tol));
  BOOST_TEST(adjoints[1][0][1] == 1.0, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Hessian_Matrix_Product_With_Identity) {
  const short tag = createNewTape();
  trace_on(tag);
  {
    adouble x0, x1, y;
    x0 <<= 2.0;
    x1 <<= 3.0;
    y = x0 * x0 + x0 * x1 + 3.0 * x1 * x1;
    double value;
    y >>= value;
  }
  trace_off();
  const std::array<double, 2> point{2.0, 3.0};
  const Identity<double> identity(2);
  Matrix<double> H(2, 2);
  BOOST_REQUIRE(hess_mat(tag, 2, 2, point, identity, H) >= 0);
  BOOST_TEST(H[0][0] == 2.0, tt::tolerance(tol));
  BOOST_TEST(H[0][1] == 1.0, tt::tolerance(tol));
  BOOST_TEST(H[1][0] == 1.0, tt::tolerance(tol));
  BOOST_TEST(H[1][1] == 6.0, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Tensor_Drivers_With_Read_Only_Inputs) {
  const short tag = recordVectorFunction();
  std::array<double, 2> point{2.0, 3.0};
  const Identity<double> identity(2);
  for (int degree : {1, 2}) {
    Matrix<double> tensor(2, static_cast<size_t>(binomi(2 + degree, degree)));
    BOOST_REQUIRE(tensor_eval(tag, 2, 2, degree, 2, point, tensor, identity) >=
                  0);
    const Matrix<double> &input = tensor;
    std::array<double, 2> derivative{};
    std::array<int, 1> multi{1};
    tensor_value(1, 2, derivative, input, multi);
    BOOST_TEST(derivative[0] == 4.0, tt::tolerance(tol));
    BOOST_TEST(derivative[1] == 1.0, tt::tolerance(tol));
  }

  const short inverseTag = createNewTape();
  trace_on(inverseTag);
  {
    adouble x, y;
    x <<= 1.5;
    y = 2.0 * x;
    double value;
    y >>= value;
  }
  trace_off();
  std::array<double, 1> inversePoint{1.5};
  const Identity<double> inverseSeed(1);
  Matrix<double> inverseTensor(1, 3);
  BOOST_REQUIRE(inverse_tensor_eval(inverseTag, 1, 2, 1, inversePoint,
                                    inverseTensor, inverseSeed) >= 0);
  BOOST_TEST(inverseTensor[0][1] == 0.5, tt::tolerance(tol));
  const Matrix<double> target = [] {
    Matrix<double> series(1, 3);
    series[0][0] = 3.0;
    series[0][1] = 1.0;
    return series;
  }();
  Matrix<double> inverseSeries(1, 3);
  inverseSeries[0][0] = 1.5;
  BOOST_REQUIRE(inverse_Taylor_prop(inverseTag, 1, 2, target, inverseSeries) >=
                0);
  BOOST_TEST(inverseSeries[0][1] == 0.5, tt::tolerance(tol));
  BOOST_TEST(inverseSeries[0][2] == 0.0, tt::tolerance(tol));
}

BOOST_AUTO_TEST_SUITE_END()

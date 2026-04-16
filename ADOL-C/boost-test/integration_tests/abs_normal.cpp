#include "../const.h"
#include "adolc/tape_interface.h"
#include <adolc/adolc.h>
#include <algorithm>
#include <array>
#include <boost/test/unit_test.hpp>
#include <vector>

namespace tt = boost::test_tools;

BOOST_AUTO_TEST_SUITE(AbsNormalFormTest)

namespace {

template <typename ActualContainer, typename ExpectedContainer>
void checkCloseContainer(const ActualContainer &actual,
                         const ExpectedContainer &expected,
                         const char *containerName) {
  BOOST_REQUIRE_EQUAL(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_TEST_CONTEXT(containerName << "[" << i << "]") {
      BOOST_TEST(actual[i] == expected[i], tt::tolerance(tol));
    }
  }
}

template <size_t Rows, size_t Cols>
void checkCloseMatrix(
    double **actual, const std::array<std::array<double, Cols>, Rows> &expected,
    const char *matrixName) {
  for (size_t row = 0; row < Rows; ++row) {
    for (size_t col = 0; col < Cols; ++col) {
      BOOST_TEST_CONTEXT(matrixName << "(" << row << "," << col << ")") {
        BOOST_TEST(actual[row][col] == expected[row][col], tt::tolerance(tol));
      }
    }
  }
}

std::vector<double *> makeRowPointers(std::vector<double> &storage, size_t rows,
                                      size_t cols) {
  std::vector<double *> matrix(rows);
  for (size_t row = 0; row < rows; ++row) {
    matrix[row] = storage.data() + row * cols;
  }
  return matrix;
}

struct SimpleAbsProblem {
  static constexpr size_t dimIn = 2;
  static constexpr size_t dimOut = 1;

  short tapeId{-1};
  std::array<double, dimIn> x = {1.0, -2.0};
  std::array<double, dimOut> y{};

  size_t numSwitches{0};

  SimpleAbsProblem() : tapeId(createNewTape()) {}
};

void taping(SimpleAbsProblem &problem) {
  setCurrentTape(problem.tapeId);
  currentTape().enableMinMaxUsingAbs();
  trace_on(problem.tapeId);

  std::vector<adouble> ax(SimpleAbsProblem::dimIn);
  std::vector<adouble> ay(SimpleAbsProblem::dimOut);

  for (int i = 0; i < SimpleAbsProblem::dimIn; i++) {
    ax[i] <<= problem.x[i];
  }

  const auto z1 = fabs(ax[0]);
  const auto z2 = fabs(ax[1]);
  ay[0] = ax[0] + ax[1] - z1 - z2;

  ay[0] >>= problem.y[0];
  trace_off();

  problem.numSwitches = get_num_switches(problem.tapeId);
  BOOST_TEST(problem.numSwitches == 2);
}

void checkAbsNormal(SimpleAbsProblem &problem) {
  std::vector<double> z(problem.numSwitches);
  std::vector<double> cz(problem.numSwitches);
  std::vector<double> cy(problem.dimOut);

  std::vector<double> yStorage(problem.dimOut * problem.dimIn);
  auto Y = makeRowPointers(yStorage, problem.dimOut, problem.dimIn);

  std::vector<double> jStorage(problem.dimOut * problem.numSwitches);
  auto J = makeRowPointers(jStorage, problem.dimOut, problem.numSwitches);

  std::vector<double> zStorage(problem.numSwitches * problem.dimIn);
  auto Z = makeRowPointers(zStorage, problem.numSwitches, problem.dimIn);

  std::vector<double> lStorage(problem.numSwitches * problem.numSwitches);
  auto L = makeRowPointers(lStorage, problem.numSwitches, problem.numSwitches);

  const int rc = abs_normal(problem.tapeId, problem.dimOut, problem.dimIn,
                            problem.numSwitches, problem.x.data(),
                            problem.y.data(), z.data(), cz.data(), cy.data(),
                            Y.data(), J.data(), Z.data(), L.data());

  BOOST_TEST(rc == 0);
  checkCloseContainer(z, std::array<double, 2>{1.0, -2.0}, "z");
  checkCloseMatrix(
      L.data(), std::array<std::array<double, 2>, 2>{{{0.0, 0.0}, {0.0, 0.0}}},
      "L");
  checkCloseMatrix(
      Z.data(), std::array<std::array<double, 2>, 2>{{{1.0, 0.0}, {0.0, 1.0}}},
      "Z");
  checkCloseMatrix(Y.data(), std::array<std::array<double, 2>, 1>{{{1.0, 1.0}}},
                   "Y");
  checkCloseMatrix(J.data(),
                   std::array<std::array<double, 2>, 1>{{{-1.0, -1.0}}}, "J");
}

struct NestedAbsProblem {
  static constexpr size_t dimIn = 2;
  static constexpr size_t dimOut = 1;

  short tapeId{-1};
  std::array<double, dimIn> x = {-1.3, 21.3};
  std::array<double, dimOut> y{};

  size_t numSwitches{0};

  NestedAbsProblem() : tapeId(createNewTape()) {}
};

void taping(NestedAbsProblem &problem) {
  setCurrentTape(problem.tapeId);
  currentTape().enableMinMaxUsingAbs();
  trace_on(problem.tapeId);

  adouble a;
  adouble b;
  a <<= problem.x[0];
  b <<= problem.x[1];

  const auto z1 = abs(a);
  const auto z2 = abs(z1 - b);
  auto f = z1 + z2 - abs(z1 - z2);

  f >>= problem.y[0];
  trace_off();

  problem.numSwitches = get_num_switches(problem.tapeId);
  BOOST_TEST(problem.numSwitches == 3);
}

constexpr std::array<std::array<double, 4>, 6> kNestedWeights = {
    {{1.0, 0.0, 0.0, 0.0},
     {0.0, 1.0, 0.0, 0.0},
     {0.0, 0.0, 1.0, 0.0},
     {0.0, 0.0, 0.0, 1.0},
     {0.0, 0.0, 1.0, 1.0},
     {1.0, 1.0, 2.0, 1.0}}};

constexpr std::array<std::array<double, 5>, 6> kNestedExpectedRows = {
    {{0.0, 0.0, 1.0, 1.0, -1.0},
     {1.0, 0.0, 0.0, 0.0, 0.0},
     {0.0, -1.0, 1.0, 0.0, 0.0},
     {0.0, 0.0, 1.0, -1.0, 0.0},
     {0.0, -1.0, 2.0, -1.0, 0.0},
     {1.0, -2.0, 4.0, 0.0, -1.0}}};

} // namespace

BOOST_AUTO_TEST_CASE(AbsNormalForm) {
  SimpleAbsProblem problem{};
  taping(problem);
  checkAbsNormal(problem);
}

BOOST_AUTO_TEST_CASE(AbsNormalForm_NestedAbsForwardAndMatrices) {
  NestedAbsProblem problem{};
  taping(problem);

  std::vector<double> z(problem.numSwitches);
  std::vector<double> cz(problem.numSwitches);
  std::vector<double> cy(problem.dimOut);

  std::vector<double> yStorage(problem.dimOut * problem.dimIn);
  auto Y = makeRowPointers(yStorage, problem.dimOut, problem.dimIn);

  std::vector<double> jStorage(problem.dimOut * problem.numSwitches);
  auto J = makeRowPointers(jStorage, problem.dimOut, problem.numSwitches);

  std::vector<double> zStorage(problem.numSwitches * problem.dimIn);
  auto Z = makeRowPointers(zStorage, problem.numSwitches, problem.dimIn);

  std::vector<double> lStorage(problem.numSwitches * problem.numSwitches);
  auto L = makeRowPointers(lStorage, problem.numSwitches, problem.numSwitches);

  const int rc = abs_normal(problem.tapeId, problem.dimOut, problem.dimIn,
                            problem.numSwitches, problem.x.data(),
                            problem.y.data(), z.data(), cz.data(), cy.data(),
                            Y.data(), J.data(), Z.data(), L.data());

  BOOST_TEST(rc == 0);
  checkCloseContainer(z, std::array<double, 3>{-1.3, -20.0, -18.7}, "z");
  checkCloseMatrix(Z.data(),
                   std::array<std::array<double, 2>, 3>{
                       {{1.0, 0.0}, {0.0, -1.0}, {0.0, 0.0}}},
                   "Z");
  checkCloseMatrix(L.data(),
                   std::array<std::array<double, 3>, 3>{
                       {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, -1.0, 0.0}}},
                   "L");
  checkCloseMatrix(Y.data(), std::array<std::array<double, 2>, 1>{{{0.0, 0.0}}},
                   "Y");
  checkCloseMatrix(
      J.data(), std::array<std::array<double, 3>, 1>{{{1.0, 1.0, -1.0}}}, "J");
}

BOOST_AUTO_TEST_CASE(FosPlReverse_BasisWeightsReturnExtendedJacobianRows) {
  NestedAbsProblem problem{};
  taping(problem);

  std::vector<double> z(problem.numSwitches);
  zos_pl_forward(problem.tapeId, problem.dimOut, problem.dimIn, 1,
                 problem.x.data(), problem.y.data(), z.data());

  std::vector<double> weights(problem.dimOut + problem.numSwitches, 0.0);
  std::vector<double> result(problem.dimIn + problem.numSwitches, 0.0);

  for (size_t row = 0; row < 4; ++row) {
    std::copy(kNestedWeights[row].begin(), kNestedWeights[row].end(),
              weights.begin());
    std::fill(result.begin(), result.end(), 0.0);

    const int rc =
        fos_pl_reverse(problem.tapeId, problem.dimOut, problem.dimIn,
                       problem.numSwitches, weights.data(), weights.data() + problem.dimOut, result.data(), result.data() + problem.dimIn);

    BOOST_TEST(rc >= 0);
    checkCloseContainer(result, kNestedExpectedRows[row], "row");
  }
}

BOOST_AUTO_TEST_CASE(FosPlReverse_CombinesSwitchRowWeightsLinearly) {
  NestedAbsProblem problem{};
  taping(problem);

  std::vector<double> z(problem.numSwitches);
  zos_pl_forward(problem.tapeId, problem.dimOut, problem.dimIn, 1,
                 problem.x.data(), problem.y.data(), z.data());

  std::vector<double> weights(problem.dimOut + problem.numSwitches, 0.0);
  std::vector<double> result(problem.dimIn + problem.numSwitches, 0.0);
  std::copy(kNestedWeights[4].begin(), kNestedWeights[4].end(),
            weights.begin());

  const int rc =
      fos_pl_reverse(problem.tapeId, problem.dimOut, problem.dimIn,
                       problem.numSwitches, weights.data(), weights.data() + problem.dimOut, result.data(), result.data() + problem.dimIn);

  BOOST_TEST(rc >= 0);
  checkCloseContainer(result, kNestedExpectedRows[4], "weighted_result");
}

BOOST_AUTO_TEST_CASE(FovPlReverse_MultipleWeightsReturnExpectedRows) {
  NestedAbsProblem problem{};
  taping(problem);

  std::vector<double> z(problem.numSwitches);
  zos_pl_forward(problem.tapeId, problem.dimOut, problem.dimIn, 1,
                 problem.x.data(), problem.y.data(), z.data());

  std::vector<double> weightsStorage(
      kNestedWeights.size() * (problem.dimOut + problem.numSwitches), 0.0);
  auto weights = makeRowPointers(weightsStorage, kNestedWeights.size(),
                                 problem.dimOut + problem.numSwitches);

  for (size_t row = 0; row < kNestedWeights.size(); ++row) {
    std::copy(kNestedWeights[row].begin(), kNestedWeights[row].end(),
              weights[row]);
  }

  std::vector<double> resultStorage(
      kNestedWeights.size() * (problem.dimIn + problem.numSwitches), 0.0);
  auto results = makeRowPointers(resultStorage, kNestedWeights.size(),
                                 problem.dimIn + problem.numSwitches);
  

  std::vector<double *> weightsSwitch(kNestedWeights.size());
  std::vector<double *> resultsSwitch(kNestedWeights.size()); 

  for (int i = 0; i < kNestedWeights.size(); i++) {
      resultsSwitch[i] = results[i] + problem.dimIn;
      weightsSwitch[i] = weights[i] + problem.dimIn;
  }
  

  const int rc = fov_pl_reverse(
      problem.tapeId, problem.dimOut, problem.dimIn, problem.numSwitches,
      static_cast<int>(kNestedWeights.size()), weights.data(), weightsSwitch.data(), results.data(), resultsSwitch.data());

  BOOST_TEST(rc >= 0);
  checkCloseMatrix(results.data(), kNestedExpectedRows, "fov_pl_reverse");
}

BOOST_AUTO_TEST_SUITE_END()

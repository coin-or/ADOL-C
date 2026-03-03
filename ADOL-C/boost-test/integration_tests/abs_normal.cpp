#include "../const.h"
#include "adolc/tape_interface.h"
#include <adolc/adolc.h>
#include <array>
#include <boost/test/unit_test.hpp>
#include <numeric>

BOOST_AUTO_TEST_SUITE(AbsNormalFormTest)

namespace {
struct ADProblem {
  static constexpr size_t dimIn = 2;
  static constexpr size_t dimOut = 1;

  short tapeId{-1};
  std::array<double, dimIn> x = {1.0, -2.0};
  std::array<double, dimOut> y{};

  size_t numSwitches{0};

  ADProblem() : tapeId(createNewTape()) {}
};

void taping(ADProblem &problem) {
  setCurrentTape(problem.tapeId);
  currentTape().enableMinMaxUsingAbs();
  trace_on(problem.tapeId);

  std::vector<adouble> ax(ADProblem::dimIn);
  std::vector<adouble> ay(ADProblem::dimOut);

  for (int i = 0; i < ADProblem::dimIn; i++)
    ax[i] <<= problem.x[i];

  auto z1 = fabs(ax[0]);
  auto z2 = fabs(ax[1]);
  ay[0] = ax[0] + ax[1] - z1 - z2;

  ay[0] >>= problem.y[0];
  trace_off();

  problem.numSwitches = get_num_switches(problem.tapeId);
  BOOST_TEST(problem.numSwitches == 2);
}

void computerAbsNormal(ADProblem &problem) {
  std::vector<double> z(problem.numSwitches);
  std::vector<double> cz(problem.numSwitches);
  std::vector<double> cy(problem.numSwitches);

  std::vector<double> Y_storage(ADProblem::dimOut * ADProblem::dimIn);
  std::vector<double *> Y(ADProblem::dimOut);
  for (int i = 0; i < ADProblem::dimOut; ++i) {
    Y[i] = Y_storage.data() + i * ADProblem::dimIn;
  }
  std::vector<double> J_storage(ADProblem::dimOut * problem.numSwitches);
  std::vector<double *> J(ADProblem::dimOut);
  for (int i = 0; i < ADProblem::dimOut; ++i) {
    J[i] = J_storage.data() + i * problem.numSwitches;
  }

  std::vector<double> Z_storage(problem.numSwitches * ADProblem::dimIn);
  std::vector<double *> Z(problem.numSwitches);
  for (int i = 0; i < problem.numSwitches; ++i) {
    Z[i] = Z_storage.data() + i * ADProblem::dimIn;
  }

  std::vector<double> L_storage(problem.numSwitches * problem.numSwitches);
  std::vector<double *> L(problem.numSwitches);
  for (int i = 0; i < problem.numSwitches; ++i) {
    L[i] = L_storage.data() + i * problem.numSwitches;
  }

  int rc = abs_normal(problem.tapeId, ADProblem::dimOut, ADProblem::dimIn,
                      problem.numSwitches, problem.x.data(), problem.y.data(),
                      z.data(), cz.data(), cy.data(), Y.data(), J.data(),
                      Z.data(), L.data());

  BOOST_TEST(L[0][0] == 0.0);
  BOOST_TEST(L[1][0] == 0.0);
  BOOST_TEST(L[0][1] == 0.0);
  BOOST_TEST(L[1][1] == 0.0);

  BOOST_TEST(Z[0][0] == 1.0);
  BOOST_TEST(Z[1][0] == 0.0);
  BOOST_TEST(Z[0][1] == 0.0);
  BOOST_TEST(Z[1][1] == 1.0);

  BOOST_TEST(Y[0][0] == 1.0);
  BOOST_TEST(Y[0][1] == 1.0);

  BOOST_TEST(J[0][0] == -1.0);
  BOOST_TEST(J[0][1] == -1.0);
}
} // namespace
BOOST_AUTO_TEST_CASE(AbsNormalForm) {

  ADProblem problem{};
  taping(problem);
  computerAbsNormal(problem);
}

BOOST_AUTO_TEST_SUITE_END()
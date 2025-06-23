#include "../const.h"
#include <adolc/adolc.h>
#include <array>
#include <boost/test/unit_test.hpp>
#include <numeric>

namespace tt = boost::test_tools;

BOOST_AUTO_TEST_SUITE(AccumulateTest)

template <typename T, size_t N> T your_function(const std::array<T, N> &indep) {
  return std::accumulate(indep.begin(), indep.end(), T(0));
}

BOOST_AUTO_TEST_CASE(AccumulateGradientCorrectness) {
  const short tapeId = 1111;
  createNewTape(tapeId);
  setCurrentTape(tapeId);
  constexpr size_t dim = 2;
  std::array<double, dim> inputs;
  inputs.fill(2.0);
  std::array<double, 1> out;

  std::array<adouble, dim> indeps;

  trace_on(tapeId);
  {
    for (size_t i = 0; i < dim; ++i) {
      indeps[i] <<= inputs[i];
    }

    adouble sum = your_function(indeps);
    sum >>= out[0];
  }
  trace_off();

  std::array<double, dim> grad;
  gradient(tapeId, dim, inputs.data(), grad.data());

  // Expected gradient of sum(x_i) is all 1s
  for (size_t i = 0; i < dim; ++i) {
    BOOST_TEST(grad[i] == 1.0, tt::tolerance(tol));
  }

  // Optional: check output value
  BOOST_TEST(out[0] == 4.0, tt::tolerance(tol));
}

BOOST_AUTO_TEST_SUITE_END()
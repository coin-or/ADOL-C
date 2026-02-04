/*
 * In this example we compute a simple function using 200'000 adoubles to
 * simulate a bigger problem. The goal is to force ADOL-C to write data to the
 * disk and check the file handling.
 */
#include "../const.h"
#include <adolc/adolc.h>
#include <boost/test/unit_test.hpp>
#include <vector>

namespace tt = boost::test_tools;

BOOST_AUTO_TEST_SUITE(BigExampleTests)

namespace {
template <size_t dim> void taping() {
  std::vector<adouble> indeps(dim);
  for (auto &indep : indeps) {
    indep <<= 1.3;
  }
  adouble out = 1.0;
  for (auto &v : indeps)
    out = out * v + 1.0;

  double dummyOut;
  out >>= dummyOut;
}
} // namespace

BOOST_AUTO_TEST_CASE(BigExample) {
  constexpr size_t dim = 200'000;
  const short tapeId = 909;
  createNewTape(tapeId);
  trace_on(tapeId, 1);
  taping<dim>();
  trace_off(1);
  std::vector<double> tangent(dim);
  std::vector<double> out(dim);
  std::fill(tangent.begin(), tangent.end(), 1.0);
  // printTapeStats(tapeId);
  gradient(tapeId, dim, tangent.data(), out.data());
  for (auto i = 0; i < out.size(); i++) {
    BOOST_TEST(out[i] == (i + 1), tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_SUITE_END()

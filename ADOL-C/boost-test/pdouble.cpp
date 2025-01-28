
/*
File for explicit testing functions from uni5_for.cpp file.
*/

#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

#include <adolc/adolc.h>
#include <adolc/taping.h>

#include "const.h"

BOOST_AUTO_TEST_CASE(ExpOperator_ZOS_Forward) {
  const int16_t tag = 0;
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{0.5};
  std::vector<double> pd_in{0.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * exp(p)
  trace_on(1);
  indep[0] <<= ad_in[0];

  pdouble pd(pd_in[0]);
  adouble dep = indep[0] * exp(pd);

  dep >>= out[0];
  trace_off();

  BOOST_TEST(out[0] == ad_in[0] * std::exp(pd_in[0]), tt::tolerance(tol));

  zos_forward(1, 1, 1, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::exp(pd_in[0]), tt::tolerance(tol));

  pd_in[0] = 1.2;

  // update pd on tape
  set_param_vec(1, 1, pd_in.data());

  zos_forward(1, 1, 1, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::exp(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(ExpOperator_FOS_Forward) {
  const int16_t tag = 0;
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{0.5};
  std::vector<double> pd_in{0.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * exp(p)
  trace_on(1);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble::mkparam(pd_in[0]);
  adouble dep = indep[0] * exp(pd);

  dep >>= out[0];
  trace_off();

  std::vector<double> X{1.1};
  std::vector<double> Y(dim_out);

  fos_forward(1, 1, 1, 0, ad_in.data(), X.data(), out.data(), Y.data());

  BOOST_TEST(out[0] == ad_in[0] * std::exp(pd_in[0]), tt::tolerance(tol));
  BOOST_TEST(Y[0] == std::exp(pd_in[0]) * X[0], tt::tolerance(tol));

  pd_in[0] = 1.2;

  // update pd on tape
  set_param_vec(1, 1, pd_in.data());

  fos_forward(1, 1, 1, 0, ad_in.data(), X.data(), out.data(), Y.data());

  BOOST_TEST(out[0] == ad_in[0] * std::exp(pd_in[0]), tt::tolerance(tol));
  BOOST_TEST(Y[0] == std::exp(pd_in[0]) * X[0], tt::tolerance(tol));
}
BOOST_AUTO_TEST_SUITE(test_pdouble)
BOOST_AUTO_TEST_SUITE_END()
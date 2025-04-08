
/*
File for explicit testing the pdouble type file.
*/

#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

#include <adolc/adolc.h>

#include "const.h"
BOOST_AUTO_TEST_SUITE(test_pdouble)
BOOST_AUTO_TEST_CASE(ExpOperator_ZOS_Forward) {
  const int16_t tag = 0;
  getTapeBuffer().push_back(std::make_shared<ValueTape>(tag));
  setDefaultTapeId(tag);
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{0.5};
  std::vector<double> pd_in{0.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * exp(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd(pd_in[0]);
  adouble dep = indep[0] * exp(pd);

  dep >>= out[0];
  trace_off(tag);

  BOOST_TEST(out[0] == ad_in[0] * std::exp(pd_in[0]), tt::tolerance(tol));

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::exp(pd_in[0]), tt::tolerance(tol));

  pd_in[0] = 1.2;

  // update pd on tape
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::exp(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(ExpOperator_FOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{0.5};
  std::vector<double> pd_in{0.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * exp(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * exp(pd);

  dep >>= out[0];
  trace_off(tag);

  std::vector<double> X{1.1};
  std::vector<double> Y(dim_out);

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] * std::exp(pd_in[0]), tt::tolerance(tol));
  BOOST_TEST(Y[0] == std::exp(pd_in[0]) * X[0], tt::tolerance(tol));

  pd_in[0] = 1.2;

  // update pd on tape
  tape->set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] * std::exp(pd_in[0]), tt::tolerance(tol));
  BOOST_TEST(Y[0] == std::exp(pd_in[0]) * X[0], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(ExpOperator_FOS_Reverse) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{2.0}; // Input for adouble
  std::vector<double> pd_in{0.5}; // Input for pdouble
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // Tape setup
  trace_on(tag, 1);      // Reverse mode
  indep[0] <<= ad_in[0]; // Bind adouble input

  pdouble pd = pdouble(pd_in[0]);   // Create pdouble parameter
  adouble dep = indep[0] * exp(pd); // Combine adouble and pdouble

  dep >>= out[0]; // Bind output
  trace_off(tag);

  BOOST_TEST(out[0] == ad_in[0] * std::exp(pd_in[0]), tt::tolerance(tol));
  // Expected results
  double aDerivative = std::exp(pd_in[0]);

  // Reverse mode computation
  std::vector<double> u(dim_out, 1.0); // Seed vector
  std::vector<double> z(dim_in);       // Derivative vector

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());
  // Check results
  BOOST_TEST(z[0] == aDerivative, tt::tolerance(tol));

  // Update pdouble parameter and recompute
  pd_in[0] = 1.2;                            // New parameter value
  tape->set_param_vec(tag, 1, pd_in.data()); // Update parameter on tape

  // recompute taylors
  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::exp(pd_in[0]), tt::tolerance(tol));

  fos_reverse(tag, dim_out, dim_in, u.data(),
              z.data()); // Recompute derivatives

  // Check updated results
  double updatedDerivative = std::exp(pd_in[0]);
  BOOST_TEST(z[0] == updatedDerivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(MultOperator_ZOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{2.0};
  std::vector<double> pd_in{3.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // Tape setup: a * p
  trace_on(tag);
  indep[0] <<= ad_in[0];          // Active variable (adouble)
  pdouble pd = pdouble(pd_in[0]); // Parameter (pdouble)
  adouble dep = indep[0] * pd;
  dep >>= out[0];
  trace_off(tag);

  // Test initial parameter value
  std::vector<double> X{2.0}; // Active input
  std::vector<double> Y(dim_out);

  zos_forward(tag, dim_out, dim_in, 0, X.data(), Y.data());
  BOOST_TEST(Y[0] == ad_in[0] * pd_in[0], tt::tolerance(tol));

  // Update parameter and test again
  pd_in[0] = 4.0;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, X.data(), Y.data());
  BOOST_TEST(Y[0] == ad_in[0] * pd_in[0], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(MultOperator_FOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{2.0};
  std::vector<double> pd_in{3.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // Tape setup: a * p
  trace_on(tag);
  indep[0] <<= ad_in[0];
  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * pd;
  dep >>= out[0];
  trace_off(tag);

  // Test derivative w.r.t active variable (a)
  std::vector<double> X{2.0};
  std::vector<double> Xd{1.0}; // Seed for active variable
  std::vector<double> Y(dim_out);
  std::vector<double> Yd(dim_out);

  fos_forward(tag, dim_in, dim_in, 0, X.data(), Xd.data(), Y.data(), Yd.data());
  BOOST_TEST(Y[0] == 2.0 * 3.5, tt::tolerance(tol));
  BOOST_TEST(Yd[0] == 3.5, tt::tolerance(tol)); // dy/da = p

  // Update parameter and test again
  pd_in[0] = 4.0;
  tape->set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_in, dim_in, 0, X.data(), Xd.data(), Y.data(), Yd.data());
  BOOST_TEST(Yd[0] == 4.0, tt::tolerance(tol)); // dy/da = updated p
}
BOOST_AUTO_TEST_CASE(MultOperator_FOS_Reverse) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 2;

  std::vector<double> ad_in{2.0, 3.5}; // Inputs for adouble
  std::vector<double> pd_in{2.3};      // Inputs for pdouble
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  trace_on(tag, 1);
  indep[0] <<= ad_in[0];
  indep[1] <<= ad_in[1];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * pd;

  dep >>= out[0];
  trace_off(tag);

  BOOST_TEST(out[0] == ad_in[0] * pd_in[0], tt::tolerance(tol));

  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == pd_in[0], tt::tolerance(tol));

  // Update pdouble parameter
  pd_in[0] = 4.2;
  tape->set_param_vec(tag, 1, pd_in.data());

  // Recompute Taylor coefficients
  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * pd_in[0], tt::tolerance(tol));

  // Recompute reverse mode derivatives
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == pd_in[0], tt::tolerance(tol));
}
BOOST_AUTO_TEST_CASE(AddOperator_ZOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{2.5};
  std::vector<double> pd_in{3.0};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a + p
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd(pd_in[0]);
  adouble dep = indep[0] + pd;

  dep >>= out[0];
  trace_off(tag);

  BOOST_TEST(out[0] == ad_in[0] + pd_in[0], tt::tolerance(tol));

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] + pd_in[0], tt::tolerance(tol));

  pd_in[0] = 1.2;

  // update pd on tape
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] + pd_in[0], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(AddOperator_FOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{2.5};
  std::vector<double> pd_in{3.0};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a + p
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] + pd;

  dep >>= out[0];
  trace_off(tag);

  std::vector<double> X{1.1};
  std::vector<double> Y(dim_out);

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] + pd_in[0], tt::tolerance(tol));
  BOOST_TEST(Y[0] == X[0], tt::tolerance(tol));

  pd_in[0] = 1.2;

  // update pd on tape
  tape->set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] + pd_in[0], tt::tolerance(tol));
  BOOST_TEST(Y[0] == X[0], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(AddOperator_FOS_Reverse) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{2.5};
  std::vector<double> pd_in{3.0};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // Tape setup
  trace_on(tag, 1);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] + pd;

  dep >>= out[0];
  trace_off(tag);

  BOOST_TEST(out[0] == ad_in[0] + pd_in[0], tt::tolerance(tol));

  // Reverse mode computation
  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == 1.0, tt::tolerance(tol));

  // Update pdouble parameter and recompute
  pd_in[0] = 1.2;
  tape->set_param_vec(tag, 1, pd_in.data());

  // recompute taylors
  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] + pd_in[0], tt::tolerance(tol));

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == 1.0, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(SubOperator_ZOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{3.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a - p
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd(pd_in[0]);
  adouble dep = indep[0] - pd;

  dep >>= out[0];
  trace_off(tag);

  BOOST_TEST(out[0] == ad_in[0] - pd_in[0], tt::tolerance(tol));

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] - pd_in[0], tt::tolerance(tol));

  pd_in[0] = 1.2;

  // update pd on tape
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] - pd_in[0], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(SubOperator_FOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{3.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a - p
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] - pd;

  dep >>= out[0];
  trace_off(tag);

  std::vector<double> X{1.1};
  std::vector<double> Y(dim_out);

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] - pd_in[0], tt::tolerance(tol));
  BOOST_TEST(Y[0] == X[0], tt::tolerance(tol));

  pd_in[0] = 1.2;

  // update pd on tape
  tape->set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] - pd_in[0], tt::tolerance(tol));
  BOOST_TEST(Y[0] == X[0], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(SubOperator_FOS_Reverse) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{3.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // Tape setup
  trace_on(tag, 1);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] - pd;

  dep >>= out[0];
  trace_off(tag);

  BOOST_TEST(out[0] == ad_in[0] - pd_in[0], tt::tolerance(tol));

  // Reverse mode computation
  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == 1.0, tt::tolerance(tol));

  // Update pdouble parameter and recompute
  pd_in[0] = 1.2;
  tape->set_param_vec(tag, 1, pd_in.data());

  // recompute taylors
  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] - pd_in[0], tt::tolerance(tol));

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == 1.0, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(DivOperator_ZOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{0.5};
  std::vector<double> pd_in{4.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a / p
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd(pd_in[0]);
  adouble dep = indep[0] / pd;

  dep >>= out[0];
  trace_off(tag);

  BOOST_TEST(out[0] == ad_in[0] / pd_in[0], tt::tolerance(tol));

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] / pd_in[0], tt::tolerance(tol));

  pd_in[0] = 2.5;

  // update pd on tape
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] / pd_in[0], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(DivOperator_FOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{0.5};
  std::vector<double> pd_in{4.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a / p
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] / pd;

  dep >>= out[0];
  trace_off(tag);

  std::vector<double> X{1.0};
  std::vector<double> Y(dim_out);

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] / pd_in[0], tt::tolerance(tol));
  BOOST_TEST(Y[0] == X[0] / pd_in[0], tt::tolerance(tol));

  pd_in[0] = 2.5;

  // update pd on tape
  tape->set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] / pd_in[0], tt::tolerance(tol));
  BOOST_TEST(Y[0] == X[0] / pd_in[0], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(DivOperator_FOS_Reverse) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{0.5};
  std::vector<double> pd_in{4.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // Tape setup
  trace_on(tag, 1);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] / pd;

  dep >>= out[0];
  trace_off(tag);

  BOOST_TEST(out[0] == ad_in[0] / pd_in[0], tt::tolerance(tol));

  // Reverse mode computation
  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == 1.0 / pd_in[0], tt::tolerance(tol));

  // Update pdouble parameter and recompute
  pd_in[0] = 2.5;
  tape->set_param_vec(tag, 1, pd_in.data());

  // recompute taylors
  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] / pd_in[0], tt::tolerance(tol));

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == 1.0 / pd_in[0], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(TanOperator_ZOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{0.7};
  std::vector<double> pd_in{1.2}; // Example parameter value
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * tan(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * tan(pd);

  dep >>= out[0];
  trace_off(tag);

  BOOST_TEST(out[0] == ad_in[0] * std::tan(pd_in[0]), tt::tolerance(tol));

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::tan(pd_in[0]), tt::tolerance(tol));

  pd_in[0] = 1.5; // Update pdouble parameter
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::tan(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(TanOperator_FOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{0.7};
  std::vector<double> pd_in{1.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * tan(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * tan(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_value = ad_in[0] * std::tan(pd_in[0]);
  double expected_derivative = std::tan(pd_in[0]);

  std::vector<double> X{1.0};
  std::vector<double> Y(dim_out);

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == expected_value, tt::tolerance(tol));
  BOOST_TEST(Y[0] == expected_derivative, tt::tolerance(tol));

  pd_in[0] = 1.5;
  tape->set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] * std::tan(pd_in[0]), tt::tolerance(tol));
  BOOST_TEST(Y[0] == std::tan(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(TanOperator_FOS_Reverse) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{0.7};
  std::vector<double> pd_in{1.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // Tape setup
  trace_on(tag, 1);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * tan(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_value = ad_in[0] * std::tan(pd_in[0]);
  double expected_derivative = std::tan(pd_in[0]);

  // Reverse mode computation
  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == expected_derivative, tt::tolerance(tol));

  pd_in[0] = 1.5;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::tan(pd_in[0]), tt::tolerance(tol));

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());
  BOOST_TEST(z[0] == std::tan(pd_in[0]), tt::tolerance(tol));
}
BOOST_AUTO_TEST_CASE(SinOperator_ZOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.2};
  std::vector<double> pd_in{0.9}; // Example parameter value
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * sin(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * sin(pd);

  dep >>= out[0];
  trace_off(tag);

  BOOST_TEST(out[0] == ad_in[0] * std::sin(pd_in[0]), tt::tolerance(tol));

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::sin(pd_in[0]), tt::tolerance(tol));

  pd_in[0] = 1.1; // Update pdouble parameter
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::sin(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(SinOperator_FOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.2};
  std::vector<double> pd_in{0.9};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * sin(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * sin(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_value = ad_in[0] * std::sin(pd_in[0]);
  double expected_derivative = std::sin(pd_in[0]);

  std::vector<double> X{1.0};
  std::vector<double> Y(dim_out);

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == expected_value, tt::tolerance(tol));
  BOOST_TEST(Y[0] == expected_derivative, tt::tolerance(tol));

  pd_in[0] = 1.1;
  tape->set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] * std::sin(pd_in[0]), tt::tolerance(tol));
  BOOST_TEST(Y[0] == std::sin(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(SinOperator_FOS_Reverse) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.2};
  std::vector<double> pd_in{0.9};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // Tape setup
  trace_on(tag, 1);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * sin(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_value = ad_in[0] * std::sin(pd_in[0]);
  double expected_derivative = std::sin(pd_in[0]);

  // Reverse mode computation
  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == expected_derivative, tt::tolerance(tol));

  pd_in[0] = 1.1;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::sin(pd_in[0]), tt::tolerance(tol));

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());
  BOOST_TEST(z[0] == std::sin(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CosOperator_ZOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.2};
  std::vector<double> pd_in{0.9}; // Example parameter value
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * cos(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * cos(pd);

  dep >>= out[0];
  trace_off(tag);

  BOOST_TEST(out[0] == ad_in[0] * std::cos(pd_in[0]), tt::tolerance(tol));

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::cos(pd_in[0]), tt::tolerance(tol));

  pd_in[0] = 1.1; // Update pdouble parameter
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::cos(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CosOperator_FOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.2};
  std::vector<double> pd_in{0.9};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * cos(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * cos(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_value = ad_in[0] * std::cos(pd_in[0]);
  double expected_derivative = std::cos(pd_in[0]);

  std::vector<double> X{1.0};
  std::vector<double> Y(dim_out);

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == expected_value, tt::tolerance(tol));
  BOOST_TEST(Y[0] == expected_derivative, tt::tolerance(tol));

  pd_in[0] = 1.1;
  tape->set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] * std::cos(pd_in[0]), tt::tolerance(tol));
  BOOST_TEST(Y[0] == std::cos(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CosOperator_FOS_Reverse) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.2};
  std::vector<double> pd_in{0.9};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // Tape setup
  trace_on(tag, 1);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * cos(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_value = ad_in[0] * std::cos(pd_in[0]);
  double expected_derivative = std::cos(pd_in[0]);

  // Reverse mode computation
  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == expected_derivative, tt::tolerance(tol));

  pd_in[0] = 1.1;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::cos(pd_in[0]), tt::tolerance(tol));

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());
  BOOST_TEST(z[0] == std::cos(pd_in[0]), tt::tolerance(tol));
}
BOOST_AUTO_TEST_CASE(SqrtOperator_ZOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{2.2};
  std::vector<double> pd_in{0.9}; // Example parameter value
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * sqrt(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * sqrt(pd);

  dep >>= out[0];
  trace_off(tag);

  BOOST_TEST(out[0] == ad_in[0] * std::sqrt(pd_in[0]), tt::tolerance(tol));

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::sqrt(pd_in[0]), tt::tolerance(tol));

  pd_in[0] = 1.1; // Update pdouble parameter
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::sqrt(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(SqrtOperator_FOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{2.2};
  std::vector<double> pd_in{0.9};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * sqrt(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * sqrt(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_value = ad_in[0] * std::sqrt(pd_in[0]);
  double expected_derivative = std::sqrt(pd_in[0]);

  std::vector<double> X{1.0};
  std::vector<double> Y(dim_out);

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == expected_value, tt::tolerance(tol));
  BOOST_TEST(Y[0] == expected_derivative, tt::tolerance(tol));

  pd_in[0] = 1.1;
  tape->set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] * std::sqrt(pd_in[0]), tt::tolerance(tol));
  BOOST_TEST(Y[0] == std::sqrt(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(SqrtOperator_FOS_Reverse) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{2.2};
  std::vector<double> pd_in{0.9};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // Tape setup
  trace_on(tag, 1);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * sqrt(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_value = ad_in[0] * std::sqrt(pd_in[0]);
  double expected_derivative = std::sqrt(pd_in[0]);

  // Reverse mode computation
  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == expected_derivative, tt::tolerance(tol));

  pd_in[0] = 1.1;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::sqrt(pd_in[0]), tt::tolerance(tol));

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());
  BOOST_TEST(z[0] == std::sqrt(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CbrtOperator_ZOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{2.2};
  std::vector<double> pd_in{0.9}; // Example parameter value
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * cbrt(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * cbrt(pd);

  dep >>= out[0];
  trace_off(tag);

  BOOST_TEST(out[0] == ad_in[0] * std::cbrt(pd_in[0]), tt::tolerance(tol));

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::cbrt(pd_in[0]), tt::tolerance(tol));

  pd_in[0] = 1.1; // Update pdouble parameter
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::cbrt(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CbrtOperator_FOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{2.2};
  std::vector<double> pd_in{0.9};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * cbrt(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * cbrt(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_value = ad_in[0] * std::cbrt(pd_in[0]);
  double expected_derivative = std::cbrt(pd_in[0]);

  std::vector<double> X{1.0};
  std::vector<double> Y(dim_out);

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == expected_value, tt::tolerance(tol));
  BOOST_TEST(Y[0] == expected_derivative, tt::tolerance(tol));

  pd_in[0] = 1.1;
  tape->set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] * std::cbrt(pd_in[0]), tt::tolerance(tol));
  BOOST_TEST(Y[0] == std::cbrt(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CbrtOperator_FOS_Reverse) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{2.2};
  std::vector<double> pd_in{0.9};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // Tape setup
  trace_on(tag, 1);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * cbrt(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_value = ad_in[0] * std::cbrt(pd_in[0]);
  double expected_derivative = std::cbrt(pd_in[0]);

  // Reverse mode computation
  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == expected_derivative, tt::tolerance(tol));

  pd_in[0] = 1.1;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::cbrt(pd_in[0]), tt::tolerance(tol));

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());
  BOOST_TEST(z[0] == std::cbrt(pd_in[0]), tt::tolerance(tol));
}
BOOST_AUTO_TEST_CASE(LogOperator_ZOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{4.9};
  std::vector<double> pd_in{0.9}; // Example parameter value
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * log(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * log(pd);

  dep >>= out[0];
  trace_off(tag);

  BOOST_TEST(out[0] == ad_in[0] * std::log(pd_in[0]), tt::tolerance(tol));

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::log(pd_in[0]), tt::tolerance(tol));

  pd_in[0] = 1.1; // Update pdouble parameter
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::log(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(LogOperator_FOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{4.9};
  std::vector<double> pd_in{0.9};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * log(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * log(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_value = ad_in[0] * std::log(pd_in[0]);
  double expected_derivative = std::log(pd_in[0]);

  std::vector<double> X{1.0};
  std::vector<double> Y(dim_out);

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == expected_value, tt::tolerance(tol));
  BOOST_TEST(Y[0] == expected_derivative, tt::tolerance(tol));

  pd_in[0] = 1.1;
  tape->set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] * std::log(pd_in[0]), tt::tolerance(tol));
  BOOST_TEST(Y[0] == std::log(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(LogOperator_FOS_Reverse) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{4.9};
  std::vector<double> pd_in{0.9};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // Tape setup
  trace_on(tag, 1);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * log(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_value = ad_in[0] * std::log(pd_in[0]);
  double expected_derivative = std::log(pd_in[0]);

  // Reverse mode computation
  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == expected_derivative, tt::tolerance(tol));

  pd_in[0] = 1.1;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::log(pd_in[0]), tt::tolerance(tol));

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());
  BOOST_TEST(z[0] == std::log(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(SinhOperator_ZOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.9}; // Example parameter value
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * sinh(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * sinh(pd);

  dep >>= out[0];
  trace_off(tag);

  BOOST_TEST(out[0] == ad_in[0] * std::sinh(pd_in[0]), tt::tolerance(tol));

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::sinh(pd_in[0]), tt::tolerance(tol));

  pd_in[0] = 1.1;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::sinh(pd_in[0]), tt::tolerance(tol));
}
BOOST_AUTO_TEST_CASE(SinhOperator_FOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.9};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * sinh(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * sinh(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_value = ad_in[0] * std::sinh(pd_in[0]);
  double expected_derivative = std::sinh(pd_in[0]);

  std::vector<double> X{1.0};
  std::vector<double> Y(dim_out);

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == expected_value, tt::tolerance(tol));
  BOOST_TEST(Y[0] == expected_derivative, tt::tolerance(tol));

  // Update pdouble parameter
  pd_in[0] = 1.1;
  tape->set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] * std::sinh(pd_in[0]), tt::tolerance(tol));
  BOOST_TEST(Y[0] == std::sinh(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(SinhOperator_FOS_Reverse) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.9};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // Tape setup
  trace_on(tag, 1);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * sinh(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_value = ad_in[0] * std::sinh(pd_in[0]);
  double expected_derivative = std::sinh(pd_in[0]);

  // Reverse mode computation
  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == expected_derivative, tt::tolerance(tol));

  // Update pdouble parameter
  pd_in[0] = 1.1;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::sinh(pd_in[0]), tt::tolerance(tol));

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());
  BOOST_TEST(z[0] == std::sinh(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CoshOperator_ZOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.9};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * cosh(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * cosh(pd);

  dep >>= out[0];
  trace_off(tag);

  BOOST_TEST(out[0] == ad_in[0] * std::cosh(pd_in[0]), tt::tolerance(tol));

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::cosh(pd_in[0]), tt::tolerance(tol));

  pd_in[0] = 1.1;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::cosh(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CoshOperator_FOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.9};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // Tape setup
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * cosh(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_value = ad_in[0] * std::cosh(pd_in[0]);
  double expected_derivative = std::cosh(pd_in[0]);

  // Forward mode computation
  std::vector<double> X{1.0};
  std::vector<double> Y(dim_out);

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == expected_value, tt::tolerance(tol));
  BOOST_TEST(Y[0] == expected_derivative, tt::tolerance(tol));

  // Update pdouble parameter
  pd_in[0] = 1.1;
  tape->set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] * std::cosh(pd_in[0]), tt::tolerance(tol));
  BOOST_TEST(Y[0] == std::cosh(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CoshOperator_FOS_Reverse) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.9};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // Tape setup
  trace_on(tag, 1);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * cosh(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_derivative = std::cosh(pd_in[0]);

  // Reverse mode computation
  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == expected_derivative, tt::tolerance(tol));

  // Update pdouble parameter
  pd_in[0] = 1.1;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::cosh(pd_in[0]), tt::tolerance(tol));

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());
  BOOST_TEST(z[0] == std::cosh(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(TanhOperator_ZOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.9};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * tanh(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * tanh(pd);

  dep >>= out[0];
  trace_off(tag);

  BOOST_TEST(out[0] == ad_in[0] * std::tanh(pd_in[0]), tt::tolerance(tol));

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::tanh(pd_in[0]), tt::tolerance(tol));

  pd_in[0] = 1.1;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::tanh(pd_in[0]), tt::tolerance(tol));
}
BOOST_AUTO_TEST_CASE(TanhOperator_FOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.9};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // Tape setup
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * tanh(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_value = ad_in[0] * std::tanh(pd_in[0]);
  double expected_derivative = std::tanh(pd_in[0]);

  // Forward mode computation
  std::vector<double> X{1.0};
  std::vector<double> Y(dim_out);

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == expected_value, tt::tolerance(tol));
  BOOST_TEST(Y[0] == expected_derivative, tt::tolerance(tol));

  // Update pdouble parameter
  pd_in[0] = 1.1;
  tape->set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] * std::tanh(pd_in[0]), tt::tolerance(tol));
  BOOST_TEST(Y[0] == std::tanh(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(TanhOperator_FOS_Reverse) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // Tape setup
  trace_on(tag, 1);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * tanh(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_derivative = std::tanh(pd_in[0]);

  // Reverse mode computation
  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == expected_derivative, tt::tolerance(tol));

  // Update pdouble parameter
  pd_in[0] = 0.3;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::tanh(pd_in[0]), tt::tolerance(tol));

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());
  BOOST_TEST(z[0] == std::tanh(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(ASinOperator_ZOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * asin(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * asin(pd);

  dep >>= out[0];
  trace_off(tag);

  BOOST_TEST(out[0] == ad_in[0] * std::asin(pd_in[0]), tt::tolerance(tol));

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::asin(pd_in[0]), tt::tolerance(tol));

  pd_in[0] = 0.3;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::asin(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(AsinOperator_FOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * asin(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * asin(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_value = ad_in[0] * std::asin(pd_in[0]);
  double expected_derivative = std::asin(pd_in[0]);

  // Forward mode computation
  std::vector<double> X{1.0};
  std::vector<double> Y(dim_out);

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == expected_value, tt::tolerance(tol));
  BOOST_TEST(Y[0] == expected_derivative, tt::tolerance(tol));

  // Update pdouble parameter
  pd_in[0] = 0.3;
  tape->set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] * std::asin(pd_in[0]), tt::tolerance(tol));
  BOOST_TEST(Y[0] == std::asin(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(AsinOperator_FOS_Reverse) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.9};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // asin
  trace_on(tag, 1);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * asin(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_derivative = std::asin(pd_in[0]);

  // Reverse mode computation
  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == expected_derivative, tt::tolerance(tol));

  // Update pdouble parameter
  pd_in[0] = 0.2;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::asin(pd_in[0]), tt::tolerance(tol));

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());
  BOOST_TEST(z[0] == std::asin(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(acosOperator_ZOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * acos(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * acos(pd);

  dep >>= out[0];
  trace_off(tag);

  BOOST_TEST(out[0] == ad_in[0] * std::acos(pd_in[0]), tt::tolerance(tol));

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::acos(pd_in[0]), tt::tolerance(tol));

  pd_in[0] = 0.3;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::acos(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(acosOperator_FOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * acos(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * acos(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_value = ad_in[0] * std::acos(pd_in[0]);
  double expected_derivative = std::acos(pd_in[0]);

  // Forward mode computation
  std::vector<double> X{1.0};
  std::vector<double> Y(dim_out);

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == expected_value, tt::tolerance(tol));
  BOOST_TEST(Y[0] == expected_derivative, tt::tolerance(tol));

  // Update pdouble parameter
  pd_in[0] = 0.3;
  tape->set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] * std::acos(pd_in[0]), tt::tolerance(tol));
  BOOST_TEST(Y[0] == std::acos(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(acosOperator_FOS_Reverse) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.9};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * acos
  trace_on(tag, 1);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * acos(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_derivative = std::acos(pd_in[0]);

  // Reverse mode computation
  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == expected_derivative, tt::tolerance(tol));

  // Update pdouble parameter
  pd_in[0] = 0.2;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::acos(pd_in[0]), tt::tolerance(tol));

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());
  BOOST_TEST(z[0] == std::acos(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(atanOperator_ZOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * atan(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * atan(pd);

  dep >>= out[0];
  trace_off(tag);

  BOOST_TEST(out[0] == ad_in[0] * std::atan(pd_in[0]), tt::tolerance(tol));

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::atan(pd_in[0]), tt::tolerance(tol));

  pd_in[0] = 0.3;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::atan(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(atanOperator_FOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * atan(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * atan(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_value = ad_in[0] * std::atan(pd_in[0]);
  double expected_derivative = std::atan(pd_in[0]);

  // Forward mode computation
  std::vector<double> X{1.0};
  std::vector<double> Y(dim_out);

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == expected_value, tt::tolerance(tol));
  BOOST_TEST(Y[0] == expected_derivative, tt::tolerance(tol));

  // Update pdouble parameter
  pd_in[0] = 0.3;
  tape->set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] * std::atan(pd_in[0]), tt::tolerance(tol));
  BOOST_TEST(Y[0] == std::atan(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(atanOperator_FOS_Reverse) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.9};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * atan
  trace_on(tag, 1);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * atan(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_derivative = std::atan(pd_in[0]);

  // Reverse mode computation
  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == expected_derivative, tt::tolerance(tol));

  // Update pdouble parameter
  pd_in[0] = 0.2;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::atan(pd_in[0]), tt::tolerance(tol));

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());
  BOOST_TEST(z[0] == std::atan(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(erfOperator_ZOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * erf(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * erf(pd);

  dep >>= out[0];
  trace_off(tag);

  BOOST_TEST(out[0] == ad_in[0] * std::erf(pd_in[0]), tt::tolerance(tol));

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::erf(pd_in[0]), tt::tolerance(tol));

  pd_in[0] = 0.3;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::erf(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(erfOperator_FOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * erf(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * erf(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_value = ad_in[0] * std::erf(pd_in[0]);
  double expected_derivative = std::erf(pd_in[0]);

  // Forward mode computation
  std::vector<double> X{1.0};
  std::vector<double> Y(dim_out);

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == expected_value, tt::tolerance(tol));
  BOOST_TEST(Y[0] == expected_derivative, tt::tolerance(tol));

  // Update pdouble parameter
  pd_in[0] = 0.3;
  tape->set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] * std::erf(pd_in[0]), tt::tolerance(tol));
  BOOST_TEST(Y[0] == std::erf(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(erfOperator_FOS_Reverse) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.9};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * erf(p)
  trace_on(tag, 1);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * erf(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_derivative = std::erf(pd_in[0]);

  // Reverse mode computation
  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == expected_derivative, tt::tolerance(tol));

  // Update pdouble parameter
  pd_in[0] = 0.2;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::erf(pd_in[0]), tt::tolerance(tol));

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());
  BOOST_TEST(z[0] == std::erf(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(erfcOperator_ZOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * erfc(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * erfc(pd);

  dep >>= out[0];
  trace_off(tag);

  BOOST_TEST(out[0] == ad_in[0] * std::erfc(pd_in[0]), tt::tolerance(tol));

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::erfc(pd_in[0]), tt::tolerance(tol));

  pd_in[0] = 0.3;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::erfc(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(erfcOperator_FOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * erfc(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * erfc(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_value = ad_in[0] * std::erfc(pd_in[0]);
  double expected_derivative = std::erfc(pd_in[0]);

  // Forward mode computation
  std::vector<double> X{1.0};
  std::vector<double> Y(dim_out);

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == expected_value, tt::tolerance(tol));
  BOOST_TEST(Y[0] == expected_derivative, tt::tolerance(tol));

  // Update pdouble parameter
  pd_in[0] = 0.3;
  tape->set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] * std::erfc(pd_in[0]), tt::tolerance(tol));
  BOOST_TEST(Y[0] == std::erfc(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(erfcOperator_FOS_Reverse) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.9};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * erfc(p)
  trace_on(tag, 1);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * erfc(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_derivative = std::erfc(pd_in[0]);

  // Reverse mode computation
  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == expected_derivative, tt::tolerance(tol));

  // Update pdouble parameter
  pd_in[0] = 0.2;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::erfc(pd_in[0]), tt::tolerance(tol));

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());
  BOOST_TEST(z[0] == std::erfc(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(log10Operator_ZOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * log10(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * log10(pd);

  dep >>= out[0];
  trace_off(tag);

  BOOST_TEST(out[0] == ad_in[0] * std::log10(pd_in[0]), tt::tolerance(tol));

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::log10(pd_in[0]), tt::tolerance(tol));

  pd_in[0] = 0.3;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::log10(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(log10Operator_FOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * log10(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * log10(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_value = ad_in[0] * std::log10(pd_in[0]);
  double expected_derivative = std::log10(pd_in[0]);

  // Forward mode computation
  std::vector<double> X{1.0};
  std::vector<double> Y(dim_out);

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == expected_value, tt::tolerance(tol));
  BOOST_TEST(Y[0] == expected_derivative, tt::tolerance(tol));

  // Update pdouble parameter
  pd_in[0] = 0.3;
  tape->set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] * std::log10(pd_in[0]), tt::tolerance(tol));
  BOOST_TEST(Y[0] == std::log10(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(log10Operator_FOS_Reverse) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.9};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * log10(p)
  trace_on(tag, 1);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * log10(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_derivative = std::log10(pd_in[0]);

  // Reverse mode computation
  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == expected_derivative, tt::tolerance(tol));

  // Update pdouble parameter
  pd_in[0] = 0.2;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::log10(pd_in[0]), tt::tolerance(tol));

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());
  BOOST_TEST(z[0] == std::log10(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(fabsOperator_ZOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * fabs(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * fabs(pd);

  dep >>= out[0];
  trace_off(tag);

  BOOST_TEST(out[0] == ad_in[0] * std::fabs(pd_in[0]), tt::tolerance(tol));

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::fabs(pd_in[0]), tt::tolerance(tol));

  pd_in[0] = 0.3;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::fabs(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(fabsOperator_FOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * fabs(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * fabs(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_value = ad_in[0] * std::fabs(pd_in[0]);
  double expected_derivative = std::fabs(pd_in[0]);

  // Forward mode computation
  std::vector<double> X{1.0};
  std::vector<double> Y(dim_out);

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == expected_value, tt::tolerance(tol));
  BOOST_TEST(Y[0] == expected_derivative, tt::tolerance(tol));

  // Update pdouble parameter
  pd_in[0] = 0.3;
  tape->set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] * std::fabs(pd_in[0]), tt::tolerance(tol));
  BOOST_TEST(Y[0] == std::fabs(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(fabsOperator_FOS_Reverse) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.9};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // fabs
  trace_on(tag, 1);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * fabs(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_derivative = std::fabs(pd_in[0]);

  // Reverse mode computation
  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == expected_derivative, tt::tolerance(tol));

  // Update pdouble parameter
  pd_in[0] = 0.2;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::fabs(pd_in[0]), tt::tolerance(tol));

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());
  BOOST_TEST(z[0] == std::fabs(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(ceilOperator_ZOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * ceil(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * ceil(pd);

  dep >>= out[0];
  trace_off(tag);

  BOOST_TEST(out[0] == ad_in[0] * std::ceil(pd_in[0]), tt::tolerance(tol));

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::ceil(pd_in[0]), tt::tolerance(tol));

  pd_in[0] = 0.3;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::ceil(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(ceilOperator_FOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * ceil(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * ceil(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_value = ad_in[0] * std::ceil(pd_in[0]);
  double expected_derivative = std::ceil(pd_in[0]);

  // Forward mode computation
  std::vector<double> X{1.0};
  std::vector<double> Y(dim_out);

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == expected_value, tt::tolerance(tol));
  BOOST_TEST(Y[0] == expected_derivative, tt::tolerance(tol));

  // Update pdouble parameter
  pd_in[0] = 0.3;
  tape->set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] * std::ceil(pd_in[0]), tt::tolerance(tol));
  BOOST_TEST(Y[0] == std::ceil(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(ceilOperator_FOS_Reverse) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.9};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // ceil
  trace_on(tag, 1);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * ceil(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_derivative = std::ceil(pd_in[0]);

  // Reverse mode computation
  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == expected_derivative, tt::tolerance(tol));

  // Update pdouble parameter
  pd_in[0] = 0.2;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::ceil(pd_in[0]), tt::tolerance(tol));

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());
  BOOST_TEST(z[0] == std::ceil(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(floorOperator_ZOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * floor(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * floor(pd);

  dep >>= out[0];
  trace_off(tag);

  BOOST_TEST(out[0] == ad_in[0] * std::floor(pd_in[0]), tt::tolerance(tol));

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::floor(pd_in[0]), tt::tolerance(tol));

  pd_in[0] = 0.3;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::floor(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(floorOperator_FOS_Forward) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * floor(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * floor(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_value = ad_in[0] * std::floor(pd_in[0]);
  double expected_derivative = std::floor(pd_in[0]);

  // Forward mode computation
  std::vector<double> X{1.0};
  std::vector<double> Y(dim_out);

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == expected_value, tt::tolerance(tol));
  BOOST_TEST(Y[0] == expected_derivative, tt::tolerance(tol));

  // Update pdouble parameter
  pd_in[0] = 0.3;
  tape->set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] * std::floor(pd_in[0]), tt::tolerance(tol));
  BOOST_TEST(Y[0] == std::floor(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(floorOperator_FOS_Reverse) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{0.9};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // floor
  trace_on(tag, 1);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = indep[0] * floor(pd);

  dep >>= out[0];
  trace_off(tag);

  double expected_derivative = std::floor(pd_in[0]);

  // Reverse mode computation
  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == expected_derivative, tt::tolerance(tol));

  // Update pdouble parameter
  pd_in[0] = 0.2;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::floor(pd_in[0]), tt::tolerance(tol));

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());
  BOOST_TEST(z[0] == std::floor(pd_in[0]), tt::tolerance(tol));
}
BOOST_AUTO_TEST_CASE(FmaxOperator_ZOS_Forward_1) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> pd_in{4.0, 3.2};
  std::vector<double> ad_in{3.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // fmax(p1, p2)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd1 = pdouble(pd_in[0]);
  pdouble pd2 = pdouble(pd_in[1]);
  adouble dep = indep[0] * fmax(pd1, pd2);

  dep >>= out[0];
  trace_off(tag);

  double expected_value = ad_in[0] * std::fmax(pd_in[0], pd_in[1]);

  std::vector<double> x{3.2};
  std::vector<double> y(dim_out);

  zos_forward(tag, dim_out, dim_in, 0, x.data(), y.data());

  BOOST_TEST(y[0] == expected_value, tt::tolerance(tol));

  // Update parameters
  pd_in[0] = 3.7;
  pd_in[1] = 3.5;
  tape->set_param_vec(tag, 2, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, x.data(), y.data());

  BOOST_TEST(y[0] == ad_in[0] * std::fmax(pd_in[0], pd_in[1]),
             tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(FmaxOperator_FOS_Forward_1) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> pd_in{4.0, 3.2};
  std::vector<double> ad_in{3.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // fmax(p1, p2)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd1 = pdouble(pd_in[0]);
  pdouble pd2 = pdouble(pd_in[1]);
  adouble dep = indep[0] * fmax(pd1, pd2);

  dep >>= out[0];
  trace_off(tag);

  double expected_value = ad_in[0] * std::fmax(pd_in[0], pd_in[1]);
  double expected_derivative = std::fmax(pd_in[0], pd_in[1]);

  std::vector<double> X{3.2};
  std::vector<double> Xd{1.0};
  std::vector<double> Y(dim_out);
  std::vector<double> Yd(dim_out);

  // Test partial derivative w.r.t. first parameter
  fos_forward(tag, dim_out, dim_in, 0, X.data(), Xd.data(), Y.data(),
              Yd.data());

  BOOST_TEST(Y[0] == expected_value, tt::tolerance(tol));
  BOOST_TEST(Yd[0] == expected_derivative, tt::tolerance(tol));

  // Test derivative with updated parameter values
  pd_in[0] = 3.7;
  pd_in[1] = 3.5;
  tape->set_param_vec(tag, 2, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, X.data(), Xd.data(), Y.data(),
              Yd.data());

  expected_value = ad_in[0] * std::fmax(pd_in[0], pd_in[1]);
  expected_derivative = std::fmax(pd_in[0], pd_in[1]);

  BOOST_TEST(Y[0] == expected_value, tt::tolerance(tol));
  BOOST_TEST(Yd[0] == expected_derivative, tt::tolerance(tol));

  // Test when p1 == p2
  std::vector<double> X1{2.5};
  std::vector<double> Xd1{1.3};
  std::vector<double> Y1(dim_out);
  std::vector<double> Yd1(dim_out);

  pd_in[0] = 2.5;
  pd_in[1] = 2.5;
  tape->set_param_vec(tag, 2, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, X1.data(), Xd1.data(), Y1.data(),
              Yd1.data());

  BOOST_TEST(Y1[0] == X1[0] * std::fmax(pd_in[0], pd_in[1]),
             tt::tolerance(tol));
  BOOST_TEST(Yd1[0] == Xd1[0] * std::fmax(pd_in[0], pd_in[1]),
             tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(FmaxOperator_FOS_Reverse_1) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> pd_in{4.0, 3.2};
  std::vector<double> ad_in{3.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * fmax(p1, p2)
  trace_on(tag, 1);
  indep[0] <<= ad_in[0];

  pdouble pd1 = pdouble(pd_in[0]);
  pdouble pd2 = pdouble(pd_in[1]);
  adouble dep = indep[0] * fmax(pd1, pd2);

  dep >>= out[0];
  trace_off(tag);

  double expected_derivative = std::fmax(pd_in[0], pd_in[1]);

  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  // Compute derivatives using reverse mode
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == expected_derivative, tt::tolerance(tol));

  // Update parameter values and recompute derivatives
  pd_in[0] = 2.5;
  pd_in[1] = 2.5;
  tape->set_param_vec(tag, 2, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == std::fmax(pd_in[0], pd_in[1]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(FmaxOperator_ZOS_Forward_2) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> pd_in{4.0};
  std::vector<double> ad_in{3.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // fmax(a, p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd1 = pdouble(pd_in[0]);
  adouble dep = fmax(pd1, indep[0]);

  dep >>= out[0];
  trace_off(tag);

  double expected = std::fmax(pd_in[0], ad_in[0]);

  BOOST_TEST(out[0] == expected, tt::tolerance(tol));

  std::vector<double> y(dim_out);
  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), y.data());

  BOOST_TEST(y[0] == expected, tt::tolerance(tol));

  // Update parameter values and recompute derivatives
  pd_in[0] = 2.5;
  tape->set_param_vec(tag, 1, pd_in.data());
  expected = std::fmax(pd_in[0], ad_in[0]);

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), y.data());

  BOOST_TEST(y[0] == expected, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(FmaxOperator_FOS_Forward_2) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> pd_in{4.0};
  std::vector<double> ad_in{3.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // fmax(a, p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd1 = pdouble(pd_in[0]);
  adouble dep = fmax(pd1, indep[0]);

  dep >>= out[0];
  trace_off(tag);

  double expected = std::fmax(pd_in[0], ad_in[0]);
  double expected_derivative = (pd_in[0] > ad_in[0]) ? 0.0 : 1.0;

  std::vector<double> xd(dim_in, 1.0);
  std::vector<double> yd(dim_out);

  fos_forward(tag, dim_out, dim_in, 1, ad_in.data(), xd.data(), out.data(),
              yd.data());

  BOOST_TEST(out[0] == expected, tt::tolerance(tol));
  BOOST_TEST(yd[0] == expected_derivative, tt::tolerance(tol));

  // Test case where a < b
  pd_in[0] = 2.0;
  ad_in[0] = 2.5;
  tape->set_param_vec(tag, 1, pd_in.data());

  expected = std::fmax(pd_in[0], ad_in[0]);
  expected_derivative = (pd_in[0] > ad_in[0]) ? 0.0 : 1.0;

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());

  fos_forward(tag, dim_out, dim_in, 1, ad_in.data(), xd.data(), out.data(),
              yd.data());

  BOOST_TEST(out[0] == 2.5, tt::tolerance(tol));
  BOOST_TEST(yd[0] == 1.0, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(FmaxOperator_FOS_Reverse_2) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> pd_in{4.0};
  std::vector<double> ad_in{3.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // fmax(a, p)
  trace_on(tag, 1);
  indep[0] <<= ad_in[0];

  pdouble pd1 = pdouble(pd_in[0]);
  adouble dep = fmax(pd1, indep[0]);

  dep >>= out[0];
  trace_off(tag);

  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  // Compute output using ZOS forward mode first
  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());

  // Compute derivatives using reverse mode
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  double expected_derivative = (pd_in[0] > ad_in[0]) ? 0.0 : 1.0;
  BOOST_TEST(z[0] == expected_derivative, tt::tolerance(tol));

  // Update parameters for case where a == b
  pd_in[0] = 0.5;
  ad_in[0] = 2.5;
  tape->set_param_vec(tag, 1, pd_in.data());
  expected_derivative = (pd_in[0] > ad_in[0]) ? 0.0 : 1.0;
  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == 1.0, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(FmaxOperator_ZOS_Forward_3) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> pd_in{3.2};
  std::vector<double> ad_in{4.0};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // fmax(p, b)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd1 = pdouble(pd_in[0]);
  adouble dep = fmax(indep[0], pd1);

  dep >>= out[0];
  trace_off(tag);

  double expected = std::fmax(ad_in[0], pd_in[0]);

  BOOST_TEST(out[0] == expected, tt::tolerance(tol));

  std::vector<double> y(dim_out);
  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), y.data());

  BOOST_TEST(y[0] == expected, tt::tolerance(tol));

  pd_in[0] = 4.1;
  tape->set_param_vec(tag, 1, pd_in.data());
  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), y.data());

  BOOST_TEST(y[0] == pd_in[0], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(FmaxOperator_FOS_Forward_3) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> pd_in{3.2};
  std::vector<double> ad_in{4.0};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // fmax(p, b)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd1 = pdouble(pd_in[0]);
  adouble dep = fmax(indep[0], pd1);

  dep >>= out[0];
  trace_off(tag);

  double expected = std::fmax(ad_in[0], pd_in[0]);
  double expected_derivative = (ad_in[0] > pd_in[0]) ? 1.0 : 0.0;

  std::vector<double> xd(dim_in, 1.0);
  std::vector<double> yd(dim_out);

  fos_forward(tag, dim_out, dim_in, 1, ad_in.data(), xd.data(), out.data(),
              yd.data());

  BOOST_TEST(out[0] == expected, tt::tolerance(tol));
  BOOST_TEST(yd[0] == expected_derivative, tt::tolerance(tol));

  // Test case where a == b
  pd_in[0] = 4.5;
  tape->set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 1, ad_in.data(), xd.data(), out.data(),
              yd.data());

  BOOST_TEST(out[0] == 4.5, tt::tolerance(tol));
  BOOST_TEST(yd[0] == 0.0, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(FmaxOperator_FOS_Reverse_3) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> pd_in{3.2};
  std::vector<double> ad_in{4.0};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // fmax(p, b)
  trace_on(tag, 1);
  indep[0] <<= ad_in[0];

  pdouble pd1 = pdouble(pd_in[0]);
  adouble dep = fmax(indep[0], pd1);

  dep >>= out[0];
  trace_off(tag);

  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  // Compute output using ZOS forward mode first
  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());

  // Compute derivatives using reverse mode
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  double expected_derivative = (ad_in[0] > pd_in[0]) ? 1.0 : 0.0;
  BOOST_TEST(z[0] == expected_derivative, tt::tolerance(tol));

  // Update parameters for case where a < b
  pd_in[0] = 4.5;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == 0.0, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(FminOperator_ZOS_Forward_1) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{4.0};
  std::vector<double> pd_in{3.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // fmin(a, p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = fmin(indep[0], pd);

  dep >>= out[0];
  trace_off(tag);

  double expected = std::fmin(ad_in[0], pd_in[0]);
  BOOST_TEST(out[0] == expected, tt::tolerance(tol));

  std::vector<double> y(dim_out);
  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), y.data());
  BOOST_TEST(y[0] == expected, tt::tolerance(tol));

  // Update parameter value and a < p
  pd_in[0] = 4.1;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), y.data());
  expected = std::fmin(ad_in[0], pd_in[0]);
  BOOST_TEST(y[0] == expected, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(FminOperator_FOS_Forward_1) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{4.0};
  std::vector<double> pd_in{3.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // fmin(a, p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = fmin(indep[0], pd);
  dep >>= out[0];
  trace_off(tag);

  double expected_value = std::fmin(ad_in[0], pd_in[0]);
  double expected_derivative = (ad_in[0] < pd_in[0]) ? 1.0 : 0.0;

  std::vector<double> X{1.0};
  std::vector<double> Y(dim_out);

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());
  BOOST_TEST(out[0] == expected_value, tt::tolerance(tol));
  BOOST_TEST(Y[0] == expected_derivative, tt::tolerance(tol));

  // a < p
  pd_in[0] = 6.1;
  tape->set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());
  expected_value = std::fmin(ad_in[0], pd_in[0]);
  expected_derivative = (ad_in[0] < pd_in[0]) ? 1.0 : 0.0;
  BOOST_TEST(out[0] == expected_value, tt::tolerance(tol));
  BOOST_TEST(Y[0] == expected_derivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(FminOperator_FOS_Reverse_1) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{4.0};
  std::vector<double> pd_in{3.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // fmin(a, p)
  trace_on(tag, 1);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = fmin(indep[0], pd);
  dep >>= out[0];
  trace_off(tag);

  double expected_derivative = (ad_in[0] < pd_in[0]) ? 1.0 : 0.0;

  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());
  BOOST_TEST(z[0] == expected_derivative, tt::tolerance(tol));

  // a < p
  pd_in[0] = 7.1;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == std::fmin(ad_in[0], pd_in[0]), tt::tolerance(tol));

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());
  expected_derivative = (ad_in[0] < pd_in[0]) ? 1.0 : 0.0;
  BOOST_TEST(z[0] == expected_derivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(FminOperator_ZOS_Forward_2) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{4.0};
  std::vector<double> pd_in{3.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble(pd_in[0]);
  adouble dep = fmin(indep[0], pd);
  dep >>= out[0];
  trace_off(tag);

  double expected = std::fmin(ad_in[0], pd_in[0]);
  BOOST_TEST(out[0] == expected, tt::tolerance(tol));

  std::vector<double> y(dim_out);
  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), y.data());
  BOOST_TEST(y[0] == expected, tt::tolerance(tol));

  // a < p
  pd_in[0] = 7.1;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), y.data());
  expected = std::fmin(ad_in[0], pd_in[0]);
  BOOST_TEST(y[0] == expected, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(FminOperator_FOS_Forward_2) {

  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{4.0};
  std::vector<double> pd_in{3.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  trace_on(tag);
  indep[0] <<= ad_in[0];
  pdouble pd = pdouble(pd_in[0]);
  adouble dep = fmin(indep[0], pd);
  dep >>= out[0];
  trace_off(tag);

  double expected = std::fmin(ad_in[0], pd_in[0]);
  double expected_derivative = (ad_in[0] < pd_in[0]) ? 1.0 : 0.0;

  std::vector<double> X{1.0};
  std::vector<double> Y(dim_out);

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == expected, tt::tolerance(tol));
  BOOST_TEST(Y[0] == expected_derivative, tt::tolerance(tol));

  // a < p
  pd_in[0] = 7.1;
  tape->set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());
  expected = std::fmin(ad_in[0], pd_in[0]);
  expected_derivative = (ad_in[0] < pd_in[0]) ? 1.0 : 0.0;
  BOOST_TEST(out[0] == expected, tt::tolerance(tol));
  BOOST_TEST(Y[0] == expected_derivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(FminOperator_FOS_Reverse_2) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{4.0};
  std::vector<double> pd_in{3.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  trace_on(tag);
  indep[0] <<= ad_in[0];
  pdouble pd = pdouble(pd_in[0]);
  adouble dep = fmin(indep[0], pd);
  dep >>= out[0];
  trace_off(tag);

  double expected = std::fmin(ad_in[0], pd_in[0]);
  double expected_derivative = (ad_in[0] < pd_in[0]) ? 1.0 : 0.0;

  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());
  BOOST_TEST(z[0] == expected_derivative, tt::tolerance(tol));

  // a < p
  pd_in[0] = 7.1;
  tape->set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0], tt::tolerance(tol));

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());
  expected_derivative = (ad_in[0] < pd_in[0]) ? 1.0 : 0.0;
  BOOST_TEST(z[0] == expected_derivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(FminOperator_FOS_Forward_3) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  double a = 2.5, b = 2.5;
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  trace_on(tag);
  indep[0] <<= a;

  pdouble pd = pdouble(b);
  adouble dep = fmin(indep[0], pd);
  dep >>= out[0];
  trace_off(tag);

  double expected = std::fmin(a, b); // should be 2.5
  double expected_derivative = 0.0;

  std::vector<double> X{1.0};
  std::vector<double> Y(dim_out);
  fos_forward(tag, dim_out, dim_in, 0, &a, X.data(), out.data(), Y.data());
  BOOST_TEST(out[0] == expected, tt::tolerance(tol));
  BOOST_TEST(Y[0] == expected_derivative, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(FminOperator_FOS_Reverse_3) {
  const int16_t tag = 0;
  std::shared_ptr<ValueTape> tape = getTape(tag);
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  double a = 2.5, b = 2.5;
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  trace_on(tag, 1);
  indep[0] <<= a;

  pdouble pd = pdouble(b);
  adouble dep = fmin(indep[0], pd);
  dep >>= out[0];
  trace_off(tag);

  double expected_derivative = 0.5;

  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);
  zos_forward(tag, dim_out, dim_in, 1, &a, out.data());
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());
  BOOST_TEST(z[0] == expected_derivative, tt::tolerance(tol));
}
BOOST_AUTO_TEST_SUITE_END()
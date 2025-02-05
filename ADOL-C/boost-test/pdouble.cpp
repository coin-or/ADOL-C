
/*
File for explicit testing the pdouble type file.
*/

#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

#include <adolc/adolc.h>
#include <adolc/taping.h>

#include "const.h"
BOOST_AUTO_TEST_SUITE(test_pdouble)
BOOST_AUTO_TEST_CASE(ExpOperator_ZOS_Forward) {
  const int16_t tag = 0;
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
  trace_off();

  BOOST_TEST(out[0] == ad_in[0] * std::exp(pd_in[0]), tt::tolerance(tol));

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::exp(pd_in[0]), tt::tolerance(tol));

  pd_in[0] = 1.2;

  // update pd on tape
  set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
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
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble::mkparam(pd_in[0]);
  adouble dep = indep[0] * exp(pd);

  dep >>= out[0];
  trace_off();

  std::vector<double> X{1.1};
  std::vector<double> Y(dim_out);

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] * std::exp(pd_in[0]), tt::tolerance(tol));
  BOOST_TEST(Y[0] == std::exp(pd_in[0]) * X[0], tt::tolerance(tol));

  pd_in[0] = 1.2;

  // update pd on tape
  set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] * std::exp(pd_in[0]), tt::tolerance(tol));
  BOOST_TEST(Y[0] == std::exp(pd_in[0]) * X[0], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(ExpOperator_FOS_Reverse) {
  const int16_t tag = 1;
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{2.0}; // Input for adouble
  std::vector<double> pd_in{0.5}; // Input for pdouble
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // Tape setup
  trace_on(tag, 1);      // Reverse mode
  indep[0] <<= ad_in[0]; // Bind adouble input

  pdouble pd = pdouble::mkparam(pd_in[0]); // Create pdouble parameter
  adouble dep = indep[0] * exp(pd);        // Combine adouble and pdouble

  dep >>= out[0]; // Bind output
  trace_off();

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
  pd_in[0] = 1.2;                      // New parameter value
  set_param_vec(tag, 1, pd_in.data()); // Update parameter on tape

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
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{2.0};
  std::vector<double> pd_in{3.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // Tape setup: a * p
  trace_on(tag);
  indep[0] <<= ad_in[0];                   // Active variable (adouble)
  pdouble pd = pdouble::mkparam(pd_in[0]); // Parameter (pdouble)
  adouble dep = indep[0] * pd;
  dep >>= out[0];
  trace_off();

  // Test initial parameter value
  std::vector<double> X{2.0}; // Active input
  std::vector<double> Y(dim_out);

  zos_forward(tag, dim_out, dim_in, 0, X.data(), Y.data());
  BOOST_TEST(Y[0] == ad_in[0] * pd_in[0], tt::tolerance(tol));

  // Update parameter and test again
  pd_in[0] = 4.0;
  set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, X.data(), Y.data());
  BOOST_TEST(Y[0] == ad_in[0] * pd_in[0], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(MultOperator_FOS_Forward) {
  const int16_t tag = 0;
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{2.0};
  std::vector<double> pd_in{3.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // Tape setup: a * p
  trace_on(tag);
  indep[0] <<= ad_in[0];
  pdouble pd = pdouble::mkparam(pd_in[0]);
  adouble dep = indep[0] * pd;
  dep >>= out[0];
  trace_off();

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
  set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_in, dim_in, 0, X.data(), Xd.data(), Y.data(), Yd.data());
  BOOST_TEST(Yd[0] == 4.0, tt::tolerance(tol)); // dy/da = updated p
}
BOOST_AUTO_TEST_CASE(MultOperator_FOS_Reverse) {
  const int16_t tag = 1;
  const size_t dim_out = 1;
  const size_t dim_in = 2;

  std::vector<double> ad_in{2.0, 3.5}; // Inputs for adouble
  std::vector<double> pd_in{2.3};      // Inputs for pdouble
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  trace_on(tag, 1);
  indep[0] <<= ad_in[0];
  indep[1] <<= ad_in[1];

  pdouble pd = pdouble::mkparam(pd_in[0]);
  adouble dep = indep[0] * pd;

  dep >>= out[0];
  trace_off();

  BOOST_TEST(out[0] == ad_in[0] * pd_in[0], tt::tolerance(tol));

  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == pd_in[0], tt::tolerance(tol));

  // Update pdouble parameter
  pd_in[0] = 4.2;
  set_param_vec(tag, 1, pd_in.data());

  // Recompute Taylor coefficients
  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * pd_in[0], tt::tolerance(tol));

  // Recompute reverse mode derivatives
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == pd_in[0], tt::tolerance(tol));
}
BOOST_AUTO_TEST_CASE(AddOperator_ZOS_Forward) {
  const int16_t tag = 0;
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
  trace_off();

  BOOST_TEST(out[0] == ad_in[0] + pd_in[0], tt::tolerance(tol));

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] + pd_in[0], tt::tolerance(tol));

  pd_in[0] = 1.2;

  // update pd on tape
  set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] + pd_in[0], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(AddOperator_FOS_Forward) {
  const int16_t tag = 0;
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{2.5};
  std::vector<double> pd_in{3.0};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a + p
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble::mkparam(pd_in[0]);
  adouble dep = indep[0] + pd;

  dep >>= out[0];
  trace_off();

  std::vector<double> X{1.1};
  std::vector<double> Y(dim_out);

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] + pd_in[0], tt::tolerance(tol));
  BOOST_TEST(Y[0] == X[0], tt::tolerance(tol));

  pd_in[0] = 1.2;

  // update pd on tape
  set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] + pd_in[0], tt::tolerance(tol));
  BOOST_TEST(Y[0] == X[0], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(AddOperator_FOS_Reverse) {
  const int16_t tag = 1;
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{2.5};
  std::vector<double> pd_in{3.0};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // Tape setup
  trace_on(tag, 1);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble::mkparam(pd_in[0]);
  adouble dep = indep[0] + pd;

  dep >>= out[0];
  trace_off();

  BOOST_TEST(out[0] == ad_in[0] + pd_in[0], tt::tolerance(tol));

  // Reverse mode computation
  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == 1.0, tt::tolerance(tol));

  // Update pdouble parameter and recompute
  pd_in[0] = 1.2;
  set_param_vec(tag, 1, pd_in.data());

  // recompute taylors
  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] + pd_in[0], tt::tolerance(tol));

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == 1.0, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(SubOperator_ZOS_Forward) {
  const int16_t tag = 0;
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
  trace_off();

  BOOST_TEST(out[0] == ad_in[0] - pd_in[0], tt::tolerance(tol));

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] - pd_in[0], tt::tolerance(tol));

  pd_in[0] = 1.2;

  // update pd on tape
  set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] - pd_in[0], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(SubOperator_FOS_Forward) {
  const int16_t tag = 0;
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{3.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a - p
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble::mkparam(pd_in[0]);
  adouble dep = indep[0] - pd;

  dep >>= out[0];
  trace_off();

  std::vector<double> X{1.1};
  std::vector<double> Y(dim_out);

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] - pd_in[0], tt::tolerance(tol));
  BOOST_TEST(Y[0] == X[0], tt::tolerance(tol));

  pd_in[0] = 1.2;

  // update pd on tape
  set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] - pd_in[0], tt::tolerance(tol));
  BOOST_TEST(Y[0] == X[0], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(SubOperator_FOS_Reverse) {
  const int16_t tag = 1;
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.5};
  std::vector<double> pd_in{3.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // Tape setup
  trace_on(tag, 1);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble::mkparam(pd_in[0]);
  adouble dep = indep[0] - pd;

  dep >>= out[0];
  trace_off();

  BOOST_TEST(out[0] == ad_in[0] - pd_in[0], tt::tolerance(tol));

  // Reverse mode computation
  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == 1.0, tt::tolerance(tol));

  // Update pdouble parameter and recompute
  pd_in[0] = 1.2;
  set_param_vec(tag, 1, pd_in.data());

  // recompute taylors
  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] - pd_in[0], tt::tolerance(tol));

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == 1.0, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(DivOperator_ZOS_Forward) {
  const int16_t tag = 0;
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
  trace_off();

  BOOST_TEST(out[0] == ad_in[0] / pd_in[0], tt::tolerance(tol));

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] / pd_in[0], tt::tolerance(tol));

  pd_in[0] = 2.5;

  // update pd on tape
  set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] / pd_in[0], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(DivOperator_FOS_Forward) {
  const int16_t tag = 0;
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{0.5};
  std::vector<double> pd_in{4.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a / p
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble::mkparam(pd_in[0]);
  adouble dep = indep[0] / pd;

  dep >>= out[0];
  trace_off();

  std::vector<double> X{1.0};
  std::vector<double> Y(dim_out);

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] / pd_in[0], tt::tolerance(tol));
  BOOST_TEST(Y[0] == X[0] / pd_in[0], tt::tolerance(tol));

  pd_in[0] = 2.5;

  // update pd on tape
  set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] / pd_in[0], tt::tolerance(tol));
  BOOST_TEST(Y[0] == X[0] / pd_in[0], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(DivOperator_FOS_Reverse) {
  const int16_t tag = 1;
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{0.5};
  std::vector<double> pd_in{4.5};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // Tape setup
  trace_on(tag, 1);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble::mkparam(pd_in[0]);
  adouble dep = indep[0] / pd;

  dep >>= out[0];
  trace_off();

  BOOST_TEST(out[0] == ad_in[0] / pd_in[0], tt::tolerance(tol));

  // Reverse mode computation
  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == 1.0 / pd_in[0], tt::tolerance(tol));

  // Update pdouble parameter and recompute
  pd_in[0] = 2.5;
  set_param_vec(tag, 1, pd_in.data());

  // recompute taylors
  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] / pd_in[0], tt::tolerance(tol));

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == 1.0 / pd_in[0], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(TanOperator_ZOS_Forward) {
  const int16_t tag = 0;
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{0.7};
  std::vector<double> pd_in{1.2}; // Example parameter value
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * tan(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble::mkparam(pd_in[0]);
  adouble dep = indep[0] * tan(pd);

  dep >>= out[0];
  trace_off();

  BOOST_TEST(out[0] == ad_in[0] * std::tan(pd_in[0]), tt::tolerance(tol));

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::tan(pd_in[0]), tt::tolerance(tol));

  pd_in[0] = 1.5; // Update pdouble parameter
  set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::tan(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(TanOperator_FOS_Forward) {
  const int16_t tag = 0;
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{0.7};
  std::vector<double> pd_in{1.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * tan(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble::mkparam(pd_in[0]);
  adouble dep = indep[0] * tan(pd);

  dep >>= out[0];
  trace_off();

  double expected_value = ad_in[0] * std::tan(pd_in[0]);
  double expected_derivative = std::tan(pd_in[0]);

  std::vector<double> X{1.0};
  std::vector<double> Y(dim_out);

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == expected_value, tt::tolerance(tol));
  BOOST_TEST(Y[0] == expected_derivative, tt::tolerance(tol));

  pd_in[0] = 1.5;
  set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] * std::tan(pd_in[0]), tt::tolerance(tol));
  BOOST_TEST(Y[0] == std::tan(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(TanOperator_FOS_Reverse) {
  const int16_t tag = 1;
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{0.7};
  std::vector<double> pd_in{1.2};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // Tape setup
  trace_on(tag, 1);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble::mkparam(pd_in[0]);
  adouble dep = indep[0] * tan(pd);

  dep >>= out[0];
  trace_off();

  double expected_value = ad_in[0] * std::tan(pd_in[0]);
  double expected_derivative = std::tan(pd_in[0]);

  // Reverse mode computation
  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == expected_derivative, tt::tolerance(tol));

  pd_in[0] = 1.5;
  set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::tan(pd_in[0]), tt::tolerance(tol));

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());
  BOOST_TEST(z[0] == std::tan(pd_in[0]), tt::tolerance(tol));
}
BOOST_AUTO_TEST_CASE(SinOperator_ZOS_Forward) {
  const int16_t tag = 0;
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.2};
  std::vector<double> pd_in{0.9}; // Example parameter value
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * sin(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble::mkparam(pd_in[0]);
  adouble dep = indep[0] * sin(pd);

  dep >>= out[0];
  trace_off();

  BOOST_TEST(out[0] == ad_in[0] * std::sin(pd_in[0]), tt::tolerance(tol));

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::sin(pd_in[0]), tt::tolerance(tol));

  pd_in[0] = 1.1; // Update pdouble parameter
  set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::sin(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(SinOperator_FOS_Forward) {
  const int16_t tag = 0;
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.2};
  std::vector<double> pd_in{0.9};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * sin(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble::mkparam(pd_in[0]);
  adouble dep = indep[0] * sin(pd);

  dep >>= out[0];
  trace_off();

  double expected_value = ad_in[0] * std::sin(pd_in[0]);
  double expected_derivative = std::sin(pd_in[0]);

  std::vector<double> X{1.0};
  std::vector<double> Y(dim_out);

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == expected_value, tt::tolerance(tol));
  BOOST_TEST(Y[0] == expected_derivative, tt::tolerance(tol));

  pd_in[0] = 1.1;
  set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] * std::sin(pd_in[0]), tt::tolerance(tol));
  BOOST_TEST(Y[0] == std::sin(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(SinOperator_FOS_Reverse) {
  const int16_t tag = 1;
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.2};
  std::vector<double> pd_in{0.9};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // Tape setup
  trace_on(tag, 1);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble::mkparam(pd_in[0]);
  adouble dep = indep[0] * sin(pd);

  dep >>= out[0];
  trace_off();

  double expected_value = ad_in[0] * std::sin(pd_in[0]);
  double expected_derivative = std::sin(pd_in[0]);

  // Reverse mode computation
  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == expected_derivative, tt::tolerance(tol));

  pd_in[0] = 1.1;
  set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::sin(pd_in[0]), tt::tolerance(tol));

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());
  BOOST_TEST(z[0] == std::sin(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CosOperator_ZOS_Forward) {
  const int16_t tag = 0;
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.2};
  std::vector<double> pd_in{0.9}; // Example parameter value
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * cos(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble::mkparam(pd_in[0]);
  adouble dep = indep[0] * cos(pd);

  dep >>= out[0];
  trace_off();

  BOOST_TEST(out[0] == ad_in[0] * std::cos(pd_in[0]), tt::tolerance(tol));

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::cos(pd_in[0]), tt::tolerance(tol));

  pd_in[0] = 1.1; // Update pdouble parameter
  set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 0, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::cos(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CosOperator_FOS_Forward) {
  const int16_t tag = 0;
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.2};
  std::vector<double> pd_in{0.9};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // a * cos(p)
  trace_on(tag);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble::mkparam(pd_in[0]);
  adouble dep = indep[0] * cos(pd);

  dep >>= out[0];
  trace_off();

  double expected_value = ad_in[0] * std::cos(pd_in[0]);
  double expected_derivative = std::cos(pd_in[0]);

  std::vector<double> X{1.0};
  std::vector<double> Y(dim_out);

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == expected_value, tt::tolerance(tol));
  BOOST_TEST(Y[0] == expected_derivative, tt::tolerance(tol));

  pd_in[0] = 1.1;
  set_param_vec(tag, 1, pd_in.data());

  fos_forward(tag, dim_out, dim_in, 0, ad_in.data(), X.data(), out.data(),
              Y.data());

  BOOST_TEST(out[0] == ad_in[0] * std::cos(pd_in[0]), tt::tolerance(tol));
  BOOST_TEST(Y[0] == std::cos(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(CosOperator_FOS_Reverse) {
  const int16_t tag = 1;
  const size_t dim_out = 1;
  const size_t dim_in = 1;
  std::vector<double> ad_in{1.2};
  std::vector<double> pd_in{0.9};
  std::vector<adouble> indep(dim_in);
  std::vector<double> out(dim_out);

  // Tape setup
  trace_on(tag, 1);
  indep[0] <<= ad_in[0];

  pdouble pd = pdouble::mkparam(pd_in[0]);
  adouble dep = indep[0] * cos(pd);

  dep >>= out[0];
  trace_off();

  double expected_value = ad_in[0] * std::cos(pd_in[0]);
  double expected_derivative = std::cos(pd_in[0]);

  // Reverse mode computation
  std::vector<double> u(dim_out, 1.0);
  std::vector<double> z(dim_in);

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());

  BOOST_TEST(z[0] == expected_derivative, tt::tolerance(tol));

  pd_in[0] = 1.1;
  set_param_vec(tag, 1, pd_in.data());

  zos_forward(tag, dim_out, dim_in, 1, ad_in.data(), out.data());
  BOOST_TEST(out[0] == ad_in[0] * std::cos(pd_in[0]), tt::tolerance(tol));

  fos_reverse(tag, dim_out, dim_in, u.data(), z.data());
  BOOST_TEST(z[0] == std::cos(pd_in[0]), tt::tolerance(tol));
}

BOOST_AUTO_TEST_SUITE_END()
#include "../const.h"
#include <adolc/adolc.h>
#include <boost/test/unit_test.hpp>
#include <memory>
#include <vector>

namespace tt = boost::test_tools;

BOOST_AUTO_TEST_SUITE(ExternalFunctionTests)

const double h = 0.01;
const int steps = 100;

const short tag_full = 1;
const short tag_part = 2;
const short tag_ext_fct = 3;

// Control parameters
std::vector<double> conp = {1.0, 1.0};
std::vector<double> grad_full(2);
std::vector<double> grad_ext(2);

ext_diff_fct *edf;
std::vector<double> yp = {0};
std::vector<double> ynewp = {0};
std::vector<double> u = {1.0, 1.0};
std::vector<double> z = {0};

void euler_step_act(size_t n, adouble *yin, size_t m, adouble *yout) {
  yout[0] = yin[0] + h * yin[0];
  yout[1] = yin[1] + h * 2 * yin[1];
}

int euler_step(size_t n, double *yin, size_t m, double *yout) {
  yout[0] = yin[0] + h * yin[0];
  yout[1] = yin[1] + h * 2 * yin[1];
  return 1;
}

int zos_for_euler_step(size_t n, double *yin, size_t m, double *yout) {
  int rc;
  set_nested_ctx(tag_ext_fct, true);
  rc = zos_forward(tag_ext_fct, 2, 2, 0, yin, yout);
  set_nested_ctx(tag_ext_fct, false);
  return rc;
}

int fos_rev_euler_step(size_t n, double *u, size_t m, double *z, double *,
                       double *) {
  int rc;
  set_nested_ctx(tag_ext_fct, true);
  zos_forward(tag_ext_fct, 2, 2, 1, edf->dp_x, edf->dp_y);
  rc = fos_reverse(tag_ext_fct, 2, 2, u, z);
  set_nested_ctx(tag_ext_fct, false);
  return rc;
}

void setup_full_taping(const short tag_full, const std::vector<double> &conp) {
  trace_on(tag_full);
  std::vector<adouble> y(2), ynew(2);
  std::vector<adouble> con(2);

  con[0] <<= conp[0];
  con[1] <<= conp[1];
  y[0] = con[0];
  y[1] = con[1];

  for (int i = 0; i < steps; i++) {
    euler_step_act(2, y.data(), 2, ynew.data());
    y[0] = ynew[0];
    y[1] = ynew[1];
  }

  adouble f;
  f = y[0] + y[1];
  double f_out;
  f >>= f_out;
  trace_off();
}

void setup_external_function(const short tag_ext_fct,
                             const std::vector<double> &conp) {
  trace_on(tag_ext_fct);
  std::vector<adouble> y(2), ynew(2);
  y[0] <<= conp[0];
  y[1] <<= conp[1];
  euler_step_act(2, y.data(), 2, ynew.data());

  double f_out;
  ynew[0] >>= f_out; // Dummy output
  ynew[1] >>= f_out; // Dummy output
  trace_off();

  edf = reg_ext_fct(euler_step);
  edf->zos_forward = zos_for_euler_step;
  edf->dp_x = yp.data();
  edf->dp_y = ynewp.data();
  edf->fos_reverse = fos_rev_euler_step;
  edf->dp_U = u.data();
  edf->dp_Z = z.data();
}

void setup_external_taping(size_t tag_part, std::vector<double> conp) {
  trace_on(tag_part);
  ensureContiguousLocations(4);
  std::vector<adouble> y(2), ynew(2);
  std::vector<adouble> con(2);

  con[0] <<= conp[0];
  con[1] <<= conp[1];
  y[0] = con[0];
  y[1] = con[1];

  for (int i = 0; i < steps; i++) {
    call_ext_fct(edf, 2, y.data(), 2, ynew.data());
    y[0] = ynew[0];
    y[1] = ynew[1];
  }

  adouble f;
  f = y[0] + y[1];
  double f_out;
  f >>= f_out;
  trace_off();
}

BOOST_AUTO_TEST_CASE(CompareFullAndExternalGradients) {
  const double expected0 = exp(h * steps);
  const double expected1 = exp(2 * h * steps);

  setup_full_taping(tag_full, conp);
  gradient(tag_full, 2, conp.data(), grad_full.data());

  setup_external_function(tag_ext_fct, conp);
  setup_external_taping(tag_part, conp);

  gradient(tag_part, 2, conp.data(), grad_ext.data());

  // Verify gradients match
  for (int i = 0; i < 2; ++i) {
    BOOST_TEST(grad_full[i] == grad_ext[i], tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_SUITE_END()
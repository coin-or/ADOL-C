#include "../const.h"
#include "adolc/valuetape/valuetape.h"
#include <adolc/adolc.h>
#include <boost/test/unit_test.hpp>
#include <memory>
#include <vector>

namespace tt = boost::test_tools;

BOOST_AUTO_TEST_SUITE(ExternalFunctionTests)

const double h = 0.01;
const int steps = 100;

std::unique_ptr<ValueTape> tapeFullPtr;
std::unique_ptr<ValueTape> tapePartPtr;
std::unique_ptr<ValueTape> tapeExtPtr;

ext_diff_fct *edf;
std::vector<double> yp = {0};
std::vector<double> ynewp = {0};
std::vector<double> u = {1.0, 1.0};
std::vector<double> z = {0};

void euler_step_act(ValueTape &, size_t, adouble *yin, size_t, adouble *yout) {
  yout[0] = yin[0] + h * yin[0];
  yout[1] = yin[1] + h * 2 * yin[1];
}

int euler_step(ValueTape &, size_t, double *yin, size_t, double *yout) {
  yout[0] = yin[0] + h * yin[0];
  yout[1] = yin[1] + h * 2 * yin[1];
  return 1;
}

int zos_for_euler_step(ValueTape &tape, size_t, double *yin, size_t,
                       double *yout) {
  int rc;
  tape.set_nested_ctx(true);
  rc = zos_forward(*edf->innerTapePtr, 2, 2, 0, yin, yout);
  tape.set_nested_ctx(false);
  return rc;
}

int fos_rev_euler_step(ValueTape &tape, size_t, double *u, size_t, double *z,
                       double *, double *) {
  int rc;
  tape.set_nested_ctx(true);
  zos_forward(*edf->innerTapePtr, 2, 2, 1, edf->dp_x, edf->dp_y);
  rc = fos_reverse(*edf->innerTapePtr, 2, 2, u, z);
  tape.set_nested_ctx(false);
  return rc;
}

void setup_full_taping(const std::vector<double> &conp) {
  trace_on(*tapeFullPtr);
  {
    std::vector<adouble> y(2), ynew(2);
    std::vector<adouble> con(2);

    con[0] <<= conp[0];
    con[1] <<= conp[1];
    y[0] = con[0];
    y[1] = con[1];

    for (int i = 0; i < steps; i++) {
      euler_step_act(*tapeFullPtr, 2, y.data(), 2, ynew.data());
      y[0] = ynew[0];
      y[1] = ynew[1];
    }

    adouble f;
    f = y[0] + y[1];
    double f_out;
    f >>= f_out;
  }
  trace_off(*tapeFullPtr);
}

void setup_external_function(ValueTape &tapePart, ValueTape &tapeExt,
                             const std::vector<double> &conp) {
  trace_on(tapeExt);
  {
    std::vector<adouble> y(2), ynew(2);
    y[0] <<= conp[0];
    y[1] <<= conp[1];
    euler_step_act(tapeExt, 2, y.data(), 2, ynew.data());

    double f_out;
    ynew[0] >>= f_out; // Dummy output
    ynew[1] >>= f_out; // Dummy output
  }
  trace_off(tapeExt);

  edf = reg_ext_fct(tapePart, tapeExt, euler_step);
  edf->zos_forward = zos_for_euler_step;
  edf->dp_x = yp.data();
  edf->dp_y = ynewp.data();
  edf->fos_reverse = fos_rev_euler_step;
  edf->dp_U = u.data();
  edf->dp_Z = z.data();
}

void setup_external_taping(ValueTape &tapePart, std::vector<double> conp) {
  trace_on(tapePart);
  {
    currentTapePtr()->ensureContiguousLocations(4);
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
  }
  trace_off(tapePart);
}

BOOST_AUTO_TEST_CASE(CompareFullAndExternalGradients) {

  tapeFullPtr = std::make_unique<ValueTape>();
  tapePartPtr = std::make_unique<ValueTape>();
  tapeExtPtr = std::make_unique<ValueTape>();

  // Control parameters
  std::vector<double> conp = {1.0, 1.0};
  std::vector<double> grad_full(2);
  std::vector<double> grad_ext(2);

  setup_full_taping(conp);
  gradient(*tapeFullPtr, 2, conp.data(), grad_full.data());

  setup_external_function(*tapePartPtr, *tapeExtPtr, conp);
  setup_external_taping(*tapePartPtr, conp);

  gradient(*tapePartPtr, 2, conp.data(), grad_ext.data());

  // Verify gradients match
  for (int i = 0; i < 2; ++i) {
    BOOST_TEST(grad_full[i] == grad_ext[i], tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_SUITE_END()

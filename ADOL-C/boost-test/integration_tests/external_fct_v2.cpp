#include "../const.h"
#include <adolc/adolc.h>
#include <array>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <vector>

namespace tt = boost::test_tools;

BOOST_AUTO_TEST_SUITE(ExternalFunctionTests)

const double h = 0.01;
const int steps = 100;

short tapeIdFullV2;
short tapeIdPartV2;
short tapeIdExtV2;

ext_diff_fct_v2 *edf_v2;

static ext_diff_fct_v2 *current_edf_v2(short tapeId) {
  ValueTape &tape = findTape(tapeId);
  return get_ext_diff_fct_v2(tapeId, tape.ext_diff_fct_index());
}

void grouped_euler_step_act(const adouble *yin, adouble *yout) {
  yout[0] = yin[0] + h * yin[0];
  yout[1] = yin[1] + h * 2 * yin[1];
}

int grouped_euler_step_v2(short, size_t, size_t *, size_t, size_t, size_t *,
                          double **x, size_t *, double **y, void *) {
  y[0][0] = x[0][0] + h * x[0][0];
  y[0][1] = x[0][1] + h * 2 * x[0][1];
  return 0;
}

int grouped_zos_forward_v2(short tapeId, size_t, size_t *, size_t, size_t,
                           size_t *, double **x, size_t *, double **y, void *) {
  ext_diff_fct_v2 *edf = current_edf_v2(tapeId);
  std::array<double, 2> xin{x[0][0], x[0][1]};
  std::array<double, 2> yout{};
  findTape(tapeId).set_nested_ctx(true);
  const int rc =
      zos_forward(edf->ext_tape_id, 2, 2, 0, xin.data(), yout.data());
  findTape(tapeId).set_nested_ctx(false);
  y[0][0] = yout[0];
  y[0][1] = yout[1];
  return rc;
}

int grouped_fos_reverse_v2(short tapeId, size_t, size_t *, size_t, size_t,
                           size_t *, double **up, size_t *, double **zp,
                           double **x, double **y, void *) {
  ext_diff_fct_v2 *edf = current_edf_v2(tapeId);
  std::array<double, 2> xin{x[0][0], x[0][1]};
  std::array<double, 2> yout{y[0][0], y[0][1]};
  std::array<double, 2> uin{up[0][0], up[0][1]};
  std::array<double, 2> zout{};

  findTape(tapeId).set_nested_ctx(true);
  zos_forward(edf->ext_tape_id, 2, 2, 1, xin.data(), yout.data());
  const int rc = fos_reverse(edf->ext_tape_id, 2, 2, uin.data(), zout.data());
  findTape(tapeId).set_nested_ctx(false);

  zp[0][0] = zout[0];
  zp[0][1] = zout[1];
  return rc;
}

void setup_full_taping_v2(const std::array<double, 2> &conp) {
  trace_on(tapeIdFullV2);
  {
    std::array<adouble, 2> y, ynew, con;
    con[0] <<= conp[0];
    con[1] <<= conp[1];
    y[0] = con[0];
    y[1] = con[1];

    for (int i = 0; i < steps; ++i) {
      grouped_euler_step_act(y.data(), ynew.data());
      y[0] = ynew[0];
      y[1] = ynew[1];
    }

    adouble f = y[0] + y[1];
    double f_out;
    f >>= f_out;
  }
  trace_off();
}

void setup_external_function_v2(const std::array<double, 2> &conp) {
  trace_on(tapeIdExtV2);
  {
    std::array<adouble, 2> y, ynew;
    y[0] <<= conp[0];
    y[1] <<= conp[1];
    grouped_euler_step_act(y.data(), ynew.data());

    double dummy;
    ynew[0] >>= dummy;
    ynew[1] >>= dummy;
  }
  trace_off();

  edf_v2 = reg_ext_fct(tapeIdPartV2, tapeIdExtV2, grouped_euler_step_v2);
  edf_v2->zos_forward = grouped_zos_forward_v2;
  edf_v2->fos_reverse = grouped_fos_reverse_v2;
}

void setup_external_taping_v2(const std::array<double, 2> &conp) {
  trace_on(tapeIdPartV2);
  {
    currentTape().ensureContiguousLocations(4);
    std::array<adouble, 2> y, ynew, con;
    adouble *xGroups[1]{y.data()};
    adouble *yGroups[1]{ynew.data()};
    size_t insz[1]{2};
    size_t outsz[1]{2};

    con[0] <<= conp[0];
    con[1] <<= conp[1];
    y[0] = con[0];
    y[1] = con[1];

    for (int i = 0; i < steps; ++i) {
      call_ext_fct(edf_v2, 0, nullptr, 1, 1, insz, xGroups, outsz, yGroups);
      y[0] = ynew[0];
      y[1] = ynew[1];
    }

    adouble f = y[0] + y[1];
    double f_out;
    f >>= f_out;
  }
  trace_off();
}

BOOST_AUTO_TEST_CASE(ExtDiffV2_GradientParity) {
  tapeIdFullV2 = createNewTape();
  tapeIdPartV2 = createNewTape();
  tapeIdExtV2 = createNewTape();

  std::array<double, 2> conp{1.0, 1.0};
  std::array<double, 2> grad_full{};
  std::array<double, 2> grad_ext{};

  setup_full_taping_v2(conp);
  gradient(tapeIdFullV2, 2, conp.data(), grad_full.data());

  setup_external_function_v2(conp);
  setup_external_taping_v2(conp);
  gradient(tapeIdPartV2, 2, conp.data(), grad_ext.data());

  BOOST_TEST(grad_full[0] == grad_ext[0], tt::tolerance(tol));
  BOOST_TEST(grad_full[1] == grad_ext[1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(ExtDiffV2_ChangingLocationsParity) {
  const short tapeIdFull = createNewTape();
  const short tapeIdPart = createNewTape();
  const short tapeIdExt = createNewTape();

  std::array<double, 2> conp{1.0, 1.2};
  std::array<double, 2> grad_full{};
  std::array<double, 2> grad_ext{};

  trace_on(tapeIdExt);
  {
    std::array<adouble, 2> y, ynew;
    y[0] <<= conp[0];
    y[1] <<= conp[1];
    grouped_euler_step_act(y.data(), ynew.data());
    double dummy;
    ynew[0] >>= dummy;
    ynew[1] >>= dummy;
  }
  trace_off();

  trace_on(tapeIdFull);
  {
    std::array<adouble, 2> con;
    std::vector<std::array<adouble, 2>> y(steps + 1);
    std::vector<std::array<adouble, 2>> ynew(steps);
    con[0] <<= conp[0];
    con[1] <<= conp[1];
    y[0][0] = con[0];
    y[0][1] = con[1];

    for (int i = 0; i < steps; ++i) {
      grouped_euler_step_act(y[i].data(), ynew[i].data());
      y[i + 1][0] = ynew[i][0];
      y[i + 1][1] = ynew[i][1];
    }

    adouble f = y[steps][0] + y[steps][1];
    double f_out;
    f >>= f_out;
  }
  trace_off();
  gradient(tapeIdFull, 2, conp.data(), grad_full.data());

  ext_diff_fct_v2 *edf =
      reg_ext_fct(tapeIdPart, tapeIdExt, grouped_euler_step_v2);
  edf->zos_forward = grouped_zos_forward_v2;
  edf->fos_reverse = grouped_fos_reverse_v2;

  trace_on(tapeIdPart);
  {
    currentTape().ensureContiguousLocations(2 * (steps + 1));
    std::array<adouble, 2> con;
    std::vector<std::array<adouble, 2>> y(steps + 1);
    std::vector<std::array<adouble, 2>> ynew(steps);
    adouble *xGroups[1];
    adouble *yGroups[1];
    size_t insz[1]{2};
    size_t outsz[1]{2};

    con[0] <<= conp[0];
    con[1] <<= conp[1];
    y[0][0] = con[0];
    y[0][1] = con[1];

    for (int i = 0; i < steps; ++i) {
      xGroups[0] = y[i].data();
      yGroups[0] = ynew[i].data();
      call_ext_fct(edf, 0, nullptr, 1, 1, insz, xGroups, outsz, yGroups);
      y[i + 1][0] = ynew[i][0];
      y[i + 1][1] = ynew[i][1];
    }

    adouble f = y[steps][0] + y[steps][1];
    double f_out;
    f >>= f_out;
  }
  trace_off();
  gradient(tapeIdPart, 2, conp.data(), grad_ext.data());

  BOOST_TEST(grad_full[0] == grad_ext[0], tt::tolerance(tol));
  BOOST_TEST(grad_full[1] == grad_ext[1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_SUITE_END()

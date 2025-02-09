#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

#include "../const.h"
#include <adolc/adolc.h>
#include <array>
#include <numeric>
#include <vector>

// Euler step (double version)
int euler_step(size_t n, double *y) {
  y[0] = y[0] + 0.01 * y[0];
  y[1] = y[1] + 0.01 * 2 * y[1];
  return 1;
}

// Euler step (adouble version)
int euler_step_act(size_t n, adouble *y) {
  y[0] = y[0] + 0.01 * y[0];
  y[1] = y[1] + 0.01 * 2 * y[1];
  return 1;
}

BOOST_AUTO_TEST_SUITE(test_checkpoint_example)
BOOST_AUTO_TEST_CASE(Checkpointing_Gradient_Comparison) {
  const int16_t tag_full = 1;  // Tag for full taping
  const int16_t tag_part = 2;  // Tag for partial taping with checkpointing
  const int16_t tag_check = 3; // Tag for checkpointing
  const size_t n = 2;          // Number of state variables
  const int steps = 100;       // Number of time steps
  const double t0 = 0.0;       // Initial time
  const double tf = 1.0;       // Final time

  // State variables (double and adouble versions)
  std::vector<double> y_double(n);
  ensureContiguousLocations(2 * n);
  std::vector<adouble> y_adouble_1(n);
  std::vector<adouble> y_adouble_2(n);

  // Control variables (double and adouble versions)
  std::vector<double> conp{1.0, 1.0}; // Initial control values
  std::vector<adouble> con(n);

  // Target value and gradient
  std::vector<double> out(2);
  std::vector<double> grad_full(n); // Gradient from full taping
  std::vector<double> grad_part(n); // Gradient from checkpointing

  // Full taping of the time step loop
  trace_on(tag_full);
  con[0] <<= conp[0];
  con[1] <<= conp[1];
  y_adouble_1[0] = con[0];
  y_adouble_1[1] = con[1];

  for (int i = 0; i < steps; i++) {
    euler_step_act(n, y_adouble_1.data());
  }
  y_adouble_1[0] + y_adouble_1[1] >>= out[0];
  trace_off();

  // Compute gradient using full taping
  gradient(tag_full, n, conp.data(), grad_full.data());

  // Checkpointing setup
  CP_Context cpc(euler_step_act);    // Checkpointing context
  cpc.setDoubleFct(euler_step);      // Double version of the time step function
  cpc.setNumberOfSteps(steps);       // Number of time steps
  cpc.setNumberOfCheckpoints(5);     // Number of checkpoints
  cpc.setDimensionXY(n);             // Dimension of input/output
  cpc.setInput(y_adouble_2.data());  // Input vector
  cpc.setOutput(y_adouble_2.data()); // Output vector
  cpc.setTapeNumber(tag_check);      // Tape number for checkpointing
  cpc.setAlwaysRetaping(false);      // Do not always retape

  // Partial taping with checkpointing
  trace_on(tag_part);
  con[0] <<= conp[0];
  con[1] <<= conp[1];
  y_adouble_2[0] = con[0];
  y_adouble_2[1] = con[1];

  cpc.checkpointing(); // Perform checkpointing

  y_adouble_2[0] + y_adouble_2[1] >>= out[1];
  trace_off();

  // test if both taping results are equal
  BOOST_TEST(out[0] == out[1], tt::tolerance(tol));
  // Compute gradient using checkpointing
  gradient(tag_part, n, conp.data(), grad_part.data());
  // Compare gradients from full taping and checkpointing
  for (size_t i = 0; i < n; i++) {
    BOOST_TEST(grad_full[i] == grad_part[i], tt::tolerance(tol));
  }
}
BOOST_AUTO_TEST_CASE(Checkpointing_fov_reverse) {
  const int16_t tag_full = 1;  // Tag for full taping
  const int16_t tag_part = 2;  // Tag for partial taping with checkpointing
  const int16_t tag_check = 3; // Tag for checkpointing
  const size_t n = 2;          // Number of state variables
  const int steps = 100;       // Number of time steps
  const double t0 = 0.0;       // Initial time
  const double tf = 1.0;       // Final time

  // State variables (double and adouble versions)
  std::vector<double> y_double(n);
  ensureContiguousLocations(2 * n);
  std::vector<adouble> y_adouble_1(n);
  std::vector<adouble> y_adouble_2(n);

  // Control variables (double and adouble versions)
  std::vector<double> conp{1.0, 1.0}; // Initial control values
  std::vector<adouble> con(n);

  // Target value and gradient
  std::vector<double> out(2);
  std::vector<double> grad_full(n); // Gradient from full taping
  std::vector<double> grad_part(n); // Gradient from checkpointing

  // Full taping of the time step loop
  trace_on(tag_full, 1);
  con[0] <<= conp[0];
  con[1] <<= conp[1];
  y_adouble_1[0] = con[0];
  y_adouble_1[1] = con[1];

  for (int i = 0; i < steps; i++) {
    euler_step_act(n, y_adouble_1.data());
  }
  y_adouble_1[0] + y_adouble_1[1] >>= out[0];
  trace_off();

  // weights
  double **U = myalloc2(2, 1);
  U[0][0] = 1.0;
  U[1][0] = -1.0;

  // outputs
  double **Z_full = myalloc2(2, 2);
  double **Z_part = myalloc2(2, 2);

  // Compute vector-mode reverse
  fov_reverse(tag_full, 1, 2, 2, U, Z_full);

  // Checkpointing setup
  CP_Context cpc(euler_step_act);    // Checkpointing context
  cpc.setDoubleFct(euler_step);      // Double version of the time step function
  cpc.setNumberOfSteps(steps);       // Number of time steps
  cpc.setNumberOfCheckpoints(5);     // Number of checkpoints
  cpc.setDimensionXY(n);             // Dimension of input/output
  cpc.setInput(y_adouble_2.data());  // Input vector
  cpc.setOutput(y_adouble_2.data()); // Output vector
  cpc.setTapeNumber(tag_check);      // Tape number for checkpointing
  cpc.setAlwaysRetaping(true);       // Do always retape

  // Partial taping with checkpointing
  trace_on(tag_part, 1);
  con[0] <<= conp[0];
  con[1] <<= conp[1];
  y_adouble_2[0] = con[0];
  y_adouble_2[1] = con[1];

  cpc.checkpointing(); // Perform checkpointing

  y_adouble_2[0] + y_adouble_2[1] >>= out[1];
  trace_off();

  // test if both taping results are equal
  BOOST_TEST(out[0] == out[1], tt::tolerance(tol));

  // Compute gradient using checkpointing
  fov_reverse(tag_part, 1, 2, 2, U, Z_part);
  // Compare gradients from full taping and checkpointing
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j)
      BOOST_TEST(Z_full[i][j] == Z_part[i][j], tt::tolerance(tol));
  }
  myfree2(U);
  myfree2(Z_full);
  myfree2(Z_part);
}
BOOST_AUTO_TEST_SUITE_END()
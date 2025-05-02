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

  const short tapeIdFull = 30;  // Tag for full taping
  const short tapeIdPart = 31;  // Tag for partial taping with checkpointing
  const short tapeIdCheck = 32; // Tag for checkpointing

  createNewTape(tapeIdFull);
  createNewTape(tapeIdPart);
  createNewTape(tapeIdCheck);

  const size_t n = 2;    // Number of state variables
  const int steps = 100; // Number of time steps
  const double t0 = 0.0; // Initial time
  const double tf = 1.0; // Final time

  // State variables (double and adouble versions)
  std::vector<double> y_double(n);

  // Control variables (double and adouble versions)
  std::vector<double> conp{1.0, 1.0}; // Initial control values

  // Target value and gradient
  std::vector<double> out(2);
  std::vector<double> grad_full(n); // Gradient from full taping
  std::vector<double> grad_part(n); // Gradient from checkpointing

  // Full taping of the time step loop
  trace_on(tapeIdFull);
  {
    currentTape().ensureContiguousLocations(n);
    std::vector<adouble> y_adouble_1(n);
    std::vector<adouble> con(n);
    con[0] <<= conp[0];
    con[1] <<= conp[1];
    y_adouble_1[0] = con[0];
    y_adouble_1[1] = con[1];

    for (int i = 0; i < steps; i++) {
      euler_step_act(n, y_adouble_1.data());
    }
    y_adouble_1[0] + y_adouble_1[1] >>= out[0];
  }
  trace_off(); // tapeIdFull

  // Compute gradient using full taping
  gradient(tapeIdFull, n, conp.data(), grad_full.data());
  // Do not always retape

  // Partial taping with checkpointing
  trace_on(tapeIdPart);
  {
    currentTape().ensureContiguousLocations(n);
    std::vector<adouble> y_adouble_2(n);
    std::vector<adouble> con2(n);

    // Checkpointing setup
    CP_Context cpc(tapeIdPart, tapeIdCheck,
                   euler_step_act); // Checkpointing context
    cpc.setDoubleFct(euler_step);   // Double version of the time step function
    cpc.setNumberOfSteps(steps);    // Number of time steps
    cpc.setNumberOfCheckpoints(5);  // Number of checkpoints
    cpc.setDimensionXY(n);          // Dimension of input/output
    cpc.setInput(y_adouble_2.data());  // Input vector
    cpc.setOutput(y_adouble_2.data()); // Output vector
    cpc.setAlwaysRetaping(false);
    con2[0] <<= conp[0];
    con2[1] <<= conp[1];
    y_adouble_2[0] = con2[0];
    y_adouble_2[1] = con2[1];

    cpc.checkpointing(tapeIdPart); // Perform checkpointing

    y_adouble_2[0] + y_adouble_2[1] >>= out[1];
  }
  trace_off();

  // test if both taping results are equal
  BOOST_TEST(out[0] == out[1], tt::tolerance(tol));

  // Compute gradient using checkpointing
  gradient(tapeIdPart, n, conp.data(), grad_part.data());
  // Compare gradients from full taping and checkpointing
  for (size_t i = 0; i < n; i++) {
    BOOST_TEST(grad_full[i] == grad_part[i], tt::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE(Checkpointing_fov_reverse) {

  const short tapeIdFull = 34;  // Tag for full taping
  const short tapeIdPart = 35;  // Tag for partial taping with checkpointing
  const short tapeIdCheck = 36; // Tag for checkpointing

  createNewTape(tapeIdFull);
  createNewTape(tapeIdPart);
  createNewTape(tapeIdCheck);

  const size_t n = 2;    // Number of state variables
  const int steps = 100; // Number of time steps
  const double t0 = 0.0; // Initial time
  const double tf = 1.0; // Final time

  // State variables (double and adouble versions)
  std::vector<double> y_double(n);
  // Control variables (double and adouble versions)
  std::vector<double> conp{1.0, 1.0}; // Initial control values

  // Target value and gradient
  std::vector<double> out(2);
  std::vector<double> grad_full(n); // Gradient from full taping
  std::vector<double> grad_part(n); // Gradient from checkpointing

  // Full taping of the time step loop
  trace_on(tapeIdFull, 1);
  {
    currentTape().ensureContiguousLocations(n);
    std::vector<adouble> y_adouble_1(n);
    std::vector<adouble> con(n);
    con[0] <<= conp[0];
    con[1] <<= conp[1];
    y_adouble_1[0] = con[0];
    y_adouble_1[1] = con[1];

    for (int i = 0; i < steps; i++) {
      euler_step_act(n, y_adouble_1.data());
    }
    y_adouble_1[0] + y_adouble_1[1] >>= out[0];
  }
  trace_off();

  // weights
  double **U = myalloc2(2, 1);
  U[0][0] = 1.0;
  U[1][0] = -1.0;

  // outputs
  double **Z_full = myalloc2(2, 2);
  double **Z_part = myalloc2(2, 2);

  // Compute vector-mode reverse
  fov_reverse(tapeIdFull, 1, 2, 2, U, Z_full);

  // Partial taping with checkpointing
  trace_on(tapeIdPart, 1);
  {
    currentTape().ensureContiguousLocations(n);
    std::vector<adouble> y_adouble_2(n);
    std::vector<adouble> con2(n);

    // Checkpointing setup
    CP_Context cpc(tapeIdPart, tapeIdCheck,
                   euler_step_act); // Checkpointing context
    cpc.setDoubleFct(euler_step);   // Double version of the time step function
    cpc.setNumberOfSteps(steps);    // Number of time steps
    cpc.setNumberOfCheckpoints(5);  // Number of checkpoints
    cpc.setDimensionXY(n);          // Dimension of input/output
    cpc.setInput(y_adouble_2.data());  // Input vector
    cpc.setOutput(y_adouble_2.data()); // Output vector
    cpc.setAlwaysRetaping(true);       // Do always retape
    con2[0] <<= conp[0];
    con2[1] <<= conp[1];
    y_adouble_2[0] = con2[0];
    y_adouble_2[1] = con2[1];

    cpc.checkpointing(tapeIdPart); // Perform checkpointing

    y_adouble_2[0] + y_adouble_2[1] >>= out[1];
  }
  trace_off();

  // test if both taping results are equal
  BOOST_TEST(out[0] == out[1], tt::tolerance(tol));

  // Compute gradient using checkpointing
  fov_reverse(tapeIdPart, 1, 2, 2, U, Z_part);
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
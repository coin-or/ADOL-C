#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

#include "../const.h"
#include <adolc/adolc.h>
#include <array>
#include <numeric>
#include <vector>

// Euler step (double version)
int euler_step(size_t, double *y) {
  y[0] = y[0] + 0.01 * y[0];
  y[1] = y[1] + 0.01 * 2 * y[1];
  return 1;
}

// Euler step (adouble version)
int euler_step_act(size_t, adouble *y) {
  y[0] = y[0] + 0.01 * y[0];
  y[1] = y[1] + 0.01 * 2 * y[1];
  return 1;
}

namespace {

constexpr size_t nonlinearDim = 3;
constexpr int nonlinearSteps = 8;
constexpr int nonlinearCheckpoints = 3;

const std::vector<double> nonlinearBase{0.7, -0.4, 1.2};
const std::vector<double> nonlinearDirection{1.0, -0.25, 0.5};
const std::vector<double> nonlinearDirectionAlt{-0.5, 0.75, 1.25};
const std::vector<double> nonlinearWeights{1.0, -0.75, 0.5};
const std::vector<double> nonlinearWeightsAlt{-0.25, 0.5, 1.0};

int nonlinear_step(size_t, double *y) {
  const double y0 = y[0];
  const double y1 = y[1];
  const double y2 = y[2];
  y[0] = y0 + 0.04 * y0 * y1 + 0.02 * y2;
  y[1] = y1 + 0.03 * y0 * y0 - 0.01 * y1 * y2;
  y[2] = y2 + 0.02 * y0 * y2 + 0.05 * y1;
  return 1;
}

int nonlinear_step_act(size_t, adouble *y) {
  const adouble y0 = y[0];
  const adouble y1 = y[1];
  const adouble y2 = y[2];
  y[0] = y0 + 0.04 * y0 * y1 + 0.02 * y2;
  y[1] = y1 + 0.03 * y0 * y0 - 0.01 * y1 * y2;
  y[2] = y2 + 0.02 * y0 * y2 + 0.05 * y1;
  return 1;
}

void compareVector(const std::vector<double> &lhs,
                   const std::vector<double> &rhs) {
  BOOST_REQUIRE_EQUAL(lhs.size(), rhs.size());
  for (size_t i = 0; i < lhs.size(); ++i)
    BOOST_TEST(lhs[i] == rhs[i], tt::tolerance(tol));
}

void compareMatrix(double **lhs, double **rhs, int rows, int cols) {
  for (int row = 0; row < rows; ++row)
    for (int col = 0; col < cols; ++col)
      BOOST_TEST(lhs[row][col] == rhs[row][col], tt::tolerance(tol));
}

// These dummy adoubles push the real state away from location 0.
// That helps the reverse tests catch bugs in location offsets.
std::vector<adouble> makeDummyAdoubles(size_t count) {
  std::vector<adouble> padding(count);
  for (size_t i = 0; i < count; ++i)
    padding[i] = 0.2 + static_cast<double>(i);
  return padding;
}

void traceNonlinearFull(short tapeId, int keep, const std::vector<double> &base,
                        std::vector<double> &out, size_t locationPadding = 0) {
  trace_on(tapeId, keep);
  {
    auto dummyAdoubles = makeDummyAdoubles(locationPadding);
    currentTape().ensureContiguousLocations(nonlinearDim);
    std::vector<adouble> state(nonlinearDim);
    for (size_t i = 0; i < nonlinearDim; ++i)
      state[i] <<= base[i];

    for (int step = 0; step < nonlinearSteps; ++step)
      nonlinear_step_act(nonlinearDim, state.data());

    for (size_t i = 0; i < nonlinearDim; ++i)
      state[i] >>= out[i];
  }
  trace_off();
}

void traceNonlinearCheckpoint(short tapeId, short cpTapeId, int keep,
                              const std::vector<double> &base,
                              std::vector<double> &out,
                              size_t locationPadding = 0) {
  trace_on(tapeId, keep);
  {
    auto dummyAdoubles = makeDummyAdoubles(locationPadding);
    currentTape().ensureContiguousLocations(nonlinearDim);
    std::vector<adouble> state(nonlinearDim);
    state <<= base;

    ADOLC::CP::Context cpc(tapeId, cpTapeId, nonlinear_step_act);
    cpc.setDoubleFct(nonlinear_step);
    cpc.setNumberOfSteps(nonlinearSteps);
    cpc.setNumberOfCheckpoints(nonlinearCheckpoints);
    cpc.setDimensionXY(static_cast<int>(nonlinearDim));
    cpc.setInput(state.data());
    cpc.setOutput(state.data());
    cpc.setAlwaysRetaping(false);
    cpc.checkpointing(tapeId);

    state >>= out;
  }
  trace_off();
}

} // namespace

BOOST_AUTO_TEST_SUITE(test_checkpoint_example)
BOOST_AUTO_TEST_CASE(Checkpointing_Gradient_Comparison) {
  const auto tapeIdFull = createNewTape();
  const auto tapeIdPart = createNewTape();
  const auto tapeIdCheck = createNewTape();

  const size_t n = 2;    // Number of state variables
  const int steps = 100; // Number of time steps

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
    con <<= conp;
    y_adouble_1 = con;

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
    ADOLC::CP::Context cpc(tapeIdPart, tapeIdCheck,
                           euler_step_act); // Checkpointing context
    cpc.setDoubleFct(euler_step);  // Double version of the time step function
    cpc.setNumberOfSteps(steps);   // Number of time steps
    cpc.setNumberOfCheckpoints(5); // Number of checkpoints
    cpc.setDimensionXY(n);         // Dimension of input/output
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

  const short tapeIdFull = createNewTape();
  const short tapeIdPart = createNewTape();
  const short tapeIdCheck = createNewTape();

  const size_t n = 2;    // Number of state variables
  const int steps = 100; // Number of time steps

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
    ADOLC::CP::Context cpc(tapeIdPart, tapeIdCheck,
                           euler_step_act); // Checkpointing context
    cpc.setDoubleFct(euler_step);  // Double version of the time step function
    cpc.setNumberOfSteps(steps);   // Number of time steps
    cpc.setNumberOfCheckpoints(5); // Number of checkpoints
    cpc.setDimensionXY(n);         // Dimension of input/output
    cpc.setInput(y_adouble_2.data());  // Input vector
    cpc.setOutput(y_adouble_2.data()); // Output vector
    cpc.setAlwaysRetaping(true);       // Do always retape
    con2 <<= conp;
    y_adouble_2 = con2;

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

BOOST_AUTO_TEST_CASE(Checkpointing_ZOS_Forward_Keep0_Nonlinear) {
  const short tapeIdFull = createNewTape();
  const short tapeIdPart = createNewTape();
  const short tapeIdCheck = createNewTape();

  std::vector<double> traceOutFull(nonlinearDim);
  std::vector<double> traceOutPart(nonlinearDim);
  traceNonlinearFull(tapeIdFull, 0, nonlinearBase, traceOutFull);
  traceNonlinearCheckpoint(tapeIdPart, tapeIdCheck, 0, nonlinearBase,
                           traceOutPart);
  compareVector(traceOutFull, traceOutPart);

  std::vector<double> yFull(nonlinearDim);
  std::vector<double> yPart(nonlinearDim);
  zos_forward(tapeIdFull, nonlinearDim, nonlinearDim, 0, nonlinearBase.data(),
              yFull.data());
  zos_forward(tapeIdPart, nonlinearDim, nonlinearDim, 0, nonlinearBase.data(),
              yPart.data());
  compareVector(yFull, yPart);
}

BOOST_AUTO_TEST_CASE(Checkpointing_FOS_Forward_Nonlinear) {
  const short tapeIdFull = createNewTape();
  const short tapeIdPart = createNewTape();
  const short tapeIdCheck = createNewTape();

  std::vector<double> traceOutFull(nonlinearDim);
  std::vector<double> traceOutPart(nonlinearDim);
  traceNonlinearFull(tapeIdFull, 0, nonlinearBase, traceOutFull);
  traceNonlinearCheckpoint(tapeIdPart, tapeIdCheck, 0, nonlinearBase,
                           traceOutPart);

  std::vector<double> yFull(nonlinearDim);
  std::vector<double> yPart(nonlinearDim);
  std::vector<double> YFull(nonlinearDim);
  std::vector<double> YPart(nonlinearDim);
  const int rcFull = fos_forward(
      tapeIdFull, nonlinearDim, nonlinearDim, 0, nonlinearBase.data(),
      nonlinearDirection.data(), yFull.data(), YFull.data());
  const int rcPart = fos_forward(
      tapeIdPart, nonlinearDim, nonlinearDim, 0, nonlinearBase.data(),
      nonlinearDirection.data(), yPart.data(), YPart.data());

  BOOST_REQUIRE_EQUAL(rcFull, rcPart);
  compareVector(yFull, yPart);
  compareVector(YFull, YPart);
}

BOOST_AUTO_TEST_CASE(Checkpointing_FOV_Forward_Nonlinear) {
  const short tapeIdFull = createNewTape();
  const short tapeIdPart = createNewTape();
  const short tapeIdCheck = createNewTape();

  std::vector<double> traceOutFull(nonlinearDim);
  std::vector<double> traceOutPart(nonlinearDim);
  traceNonlinearFull(tapeIdFull, 0, nonlinearBase, traceOutFull);
  traceNonlinearCheckpoint(tapeIdPart, tapeIdCheck, 0, nonlinearBase,
                           traceOutPart);

  const int p = 2;
  double **XpFull = myalloc2(nonlinearDim, p);
  double **XpPart = myalloc2(nonlinearDim, p);
  double **YpFull = myalloc2(nonlinearDim, p);
  double **YpPart = myalloc2(nonlinearDim, p);
  for (size_t row = 0; row < nonlinearDim; ++row) {
    XpFull[row][0] = nonlinearDirection[row];
    XpFull[row][1] = nonlinearDirectionAlt[row];
    XpPart[row][0] = nonlinearDirection[row];
    XpPart[row][1] = nonlinearDirectionAlt[row];
  }

  std::vector<double> yFull(nonlinearDim);
  std::vector<double> yPart(nonlinearDim);
  const int rcFull =
      fov_forward(tapeIdFull, nonlinearDim, nonlinearDim, p,
                  nonlinearBase.data(), XpFull, yFull.data(), YpFull);
  const int rcPart =
      fov_forward(tapeIdPart, nonlinearDim, nonlinearDim, p,
                  nonlinearBase.data(), XpPart, yPart.data(), YpPart);

  BOOST_REQUIRE_EQUAL(rcFull, rcPart);
  compareVector(yFull, yPart);
  compareMatrix(YpFull, YpPart, nonlinearDim, p);
  myfree2(XpFull);
  myfree2(XpPart);
  myfree2(YpFull);
  myfree2(YpPart);
}

BOOST_AUTO_TEST_CASE(Checkpointing_FOS_Reverse_OffsetLocations) {
  const short tapeIdFull = createNewTape();
  const short tapeIdPart = createNewTape();
  const short tapeIdCheck = createNewTape();

  std::vector<double> traceOutFull(nonlinearDim);
  std::vector<double> traceOutPart(nonlinearDim);
  traceNonlinearFull(tapeIdFull, 0, nonlinearBase, traceOutFull, 3);
  traceNonlinearCheckpoint(tapeIdPart, tapeIdCheck, 0, nonlinearBase,
                           traceOutPart, 3);

  std::vector<double> yFull(nonlinearDim);
  std::vector<double> yPart(nonlinearDim);
  zos_forward(tapeIdFull, nonlinearDim, nonlinearDim, 1, nonlinearBase.data(),
              yFull.data());
  zos_forward(tapeIdPart, nonlinearDim, nonlinearDim, 1, nonlinearBase.data(),
              yPart.data());
  compareVector(yFull, yPart);

  std::vector<double> zFull(nonlinearDim);
  std::vector<double> zPart(nonlinearDim);
  fos_reverse(tapeIdFull, nonlinearDim, nonlinearDim, nonlinearWeights.data(),
              zFull.data());
  fos_reverse(tapeIdPart, nonlinearDim, nonlinearDim, nonlinearWeights.data(),
              zPart.data());
  compareVector(zFull, zPart);
}

BOOST_AUTO_TEST_CASE(Checkpointing_FOV_Reverse_OffsetLocations) {
  const short tapeIdFull = createNewTape();
  const short tapeIdPart = createNewTape();
  const short tapeIdCheck = createNewTape();

  std::vector<double> traceOutFull(nonlinearDim);
  std::vector<double> traceOutPart(nonlinearDim);
  traceNonlinearFull(tapeIdFull, 0, nonlinearBase, traceOutFull, 2);
  traceNonlinearCheckpoint(tapeIdPart, tapeIdCheck, 0, nonlinearBase,
                           traceOutPart, 2);

  std::vector<double> yFull(nonlinearDim);
  std::vector<double> yPart(nonlinearDim);
  zos_forward(tapeIdFull, nonlinearDim, nonlinearDim, 1, nonlinearBase.data(),
              yFull.data());
  zos_forward(tapeIdPart, nonlinearDim, nonlinearDim, 1, nonlinearBase.data(),
              yPart.data());
  compareVector(yFull, yPart);

  const int q = 2;
  double **UqFull = myalloc2(q, nonlinearDim);
  double **UqPart = myalloc2(q, nonlinearDim);
  double **ZqFull = myalloc2(q, nonlinearDim);
  double **ZqPart = myalloc2(q, nonlinearDim);
  for (size_t col = 0; col < nonlinearDim; ++col) {
    UqFull[0][col] = nonlinearWeights[col];
    UqFull[1][col] = nonlinearWeightsAlt[col];
    UqPart[0][col] = nonlinearWeights[col];
    UqPart[1][col] = nonlinearWeightsAlt[col];
  }

  fov_reverse(tapeIdFull, nonlinearDim, nonlinearDim, q, UqFull, ZqFull);
  fov_reverse(tapeIdPart, nonlinearDim, nonlinearDim, q, UqPart, ZqPart);
  compareMatrix(ZqFull, ZqPart, q, nonlinearDim);
  myfree2(UqFull);
  myfree2(UqPart);
  myfree2(ZqFull);
  myfree2(ZqPart);
}
BOOST_AUTO_TEST_SUITE_END()

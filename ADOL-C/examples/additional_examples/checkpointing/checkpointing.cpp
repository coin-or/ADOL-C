/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     checkpointing.cpp
 Revision: $Id$
 Contents: example for checkpointing

 Copyright (c) Andrea Walther

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

---------------------------------------------------------------------------*/
#include <adolc/adolc.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

template <class data_type> int euler_step_act(size_t, data_type *y) {
  y[0] = y[0] + 0.01 * y[0];
  y[1] = y[1] + 0.01 * 2 * y[1];

  return 1;
}

int main() {
  constexpr short dim = 2;

  const short tapeIdFull = createNewTape();
  const short tapeIdPart = createNewTape();
  const short tapeIdCheck = createNewTape();

  std::array<double, dim> conp = {1.0, 1.0};

  std::array<double, dim> gradFull{};
  std::array<double, dim> gradCheckpoint{};

  const size_t steps = 100;

  const size_t num_cpts = 5;

  // Record the full loop as a reference.
  trace_on(tapeIdFull);
  {
    std::array<adouble, dim> y;

    std::array<adouble, dim> con;

    for (size_t i = 0; i < con.size(); ++i) {
      con[i] <<= conp[i];
      y[i] = con[i];
    }

    for (size_t i = 0; i < steps; ++i) {
      euler_step_act(dim, y.data());
    }
    double f[] = {0.0};
    y[0] + y[1] >>= f[0];
  }
  trace_off();

  const int fullStatus =
      gradient(tapeIdFull, dim, conp.data(), gradFull.data());

  printf("full taping gradient = [%.6f, %.6f]\n", gradFull[0], gradFull[1]);

  //! [checkpointing-context]
  trace_on(tapeIdPart);
  {
    // ensure that the adoubles stored in y occupy consecutive locations
    currentTape().ensureContiguousLocations(dim);
    std::array<adouble, dim> y;

    std::array<adouble, dim> con;

    for (size_t i = 0; i < con.size(); ++i) {
      con[i] <<= conp[i];
      y[i] = con[i];
    }

    // Define the active variant of the time-step function.
    ADOLC::CP::Context cpc(tapeIdPart, tapeIdCheck, euler_step_act<adouble>);

    // Provide the passive variant of the time-step function.
    cpc.setDoubleFct(euler_step_act<double>);

    cpc.setNumberOfSteps(steps);

    cpc.setNumberOfCheckpoints(num_cpts);

    cpc.setDimensionXY(dim);
    cpc.setInput(y.data());
    cpc.setOutput(y.data());
    // Reuse the recorded time-step tape when possible.
    cpc.setAlwaysRetaping(false);

    cpc.checkpointing(tapeIdPart);

    double f[] = {0.0};
    y[0] + y[1] >>= f[0];
  }
  trace_off();

  const int checkpointStatus =
      gradient(tapeIdPart, dim, conp.data(), gradCheckpoint.data());
  //! [checkpointing-context]

  printf("checkpoint gradient = [%.6f, %.6f]\n", gradCheckpoint[0],
         gradCheckpoint[1]);

  const std::array<double, dim> expected{
      std::pow(1.01, static_cast<double>(steps)),
      std::pow(1.02, static_cast<double>(steps))};
  const auto close = [](double actual, double reference) {
    const double scale = std::max(1.0, std::fabs(reference));
    return std::fabs(actual - reference) <= 1.0e-10 * scale;
  };

  if (fullStatus < 0 || checkpointStatus < 0)
    return 1;

  for (size_t i = 0; i < dim; ++i) {
    if (!close(gradFull[i], expected[i]) ||
        !close(gradCheckpoint[i], expected[i]) ||
        !close(gradCheckpoint[i], gradFull[i])) {
      std::fprintf(stderr, "checkpoint gradient validation failed at %zu\n", i);
      return 1;
    }
  }

  printf("validation passed\n");
}

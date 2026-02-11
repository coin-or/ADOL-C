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
#include "adolc/valuetape/valuetape.h"
#include <adolc/adolc.h>
#include <memory>

// time step function
template <class data_type> int euler_step_act(size_t, data_type *y) {
  // Euler step, adouble version
  y[0] = y[0] + 0.01 * y[0];
  y[1] = y[1] + 0.01 * 2 * y[1];

  return 1;
}

int main() {
  // two input and output variables for checkpointing function
  constexpr short dim = 2;

  auto tapeFullPtr = std::make_unique<ValueTape>();
  auto tapePartPtr = std::make_unique<ValueTape>();
  auto tapeCheckPtr = std::make_unique<ValueTape>();

  // control
  std::array<double, dim> conp = {1.0, 1.0};

  // variables for derivative calculation
  std::array<double, dim> grad;

  // time steps
  const size_t steps = 100;

  // number of checkpoints
  const size_t num_cpts = 5;

  // basis variant: full taping of time step loop
  trace_on(*tapeFullPtr);
  {
    // state, double and adouble version
    std::array<adouble, dim> y;

    // control, double and adouble version
    std::array<adouble, dim> con;

    for (auto i = 0; i < dim; ++i) {
      con[i] <<= conp[i];
      y[i] = con[i];
    }

    for (size_t i = 0; i < steps; ++i) {
      euler_step_act(dim, y.data());
    }
    double f[] = {0.0};
    y[0] + y[1] >>= f[0];
  }
  trace_off(*tapeFullPtr, 1);

  gradient(*tapeFullPtr, dim, conp.data(), grad.data());

  printf(" full taping:\n gradient=( %f, %f)\n\n", grad[0], grad[1]);

  trace_on(*tapePartPtr);
  {
    // ensure that the adoubles stored in y occupy consecutive locations
    currentTapePtr()->ensureContiguousLocations(dim);
    std::array<adouble, dim> y;

    std::array<adouble, dim> con;

    for (auto i = 0; i < dim; ++i) {
      con[i] <<= conp[i];
      y[i] = con[i];
    }

    // Now using checkpointing facilities
    // generate checkpointing context => define active variante of the time step
    CP_Context cpc(*tapePartPtr, *tapeCheckPtr, euler_step_act<adouble>);

    // double variante of the time step function
    cpc.setDoubleFct(euler_step_act<double>);

    // number of time steps to perform
    cpc.setNumberOfSteps(steps);

    // number of checkpoint
    cpc.setNumberOfCheckpoints(num_cpts);

    // dimension of input/output
    cpc.setDimensionXY(dim);
    // input vector
    cpc.setInput(y.data());
    // output vector
    cpc.setOutput(y.data());
    // always retape or not ?
    cpc.setAlwaysRetaping(false);

    cpc.checkpointing(*tapePartPtr);

    double f[] = {0.0};
    y[0] + y[1] >>= f[0];
  }
  trace_off(*tapePartPtr, 1);

  gradient(*tapePartPtr, dim, conp.data(), grad.data());

  printf(" taping with checkpointing facility:\n gradient=( %f, %f)\n\n",
         grad[0], grad[1]);
  return 0;
}

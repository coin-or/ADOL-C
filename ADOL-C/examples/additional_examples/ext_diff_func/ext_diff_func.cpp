/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     ext_diff_func.cpp
 Revision: $Id$
 Contents: example for external differentiated functions

 Copyright (c) Andrea Walther

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

---------------------------------------------------------------------------*/
#include <adolc/adolc.h>

#include <array>
#include <cstdio>

namespace {

constexpr double h = 0.01;
constexpr int steps = 100;

using PassiveState = std::array<double, 2>;
using ActiveState = std::array<adouble, 2>;

void eulerStepAct(const adouble *yin, adouble *yout) {
  yout[0] = yin[0] + h * yin[0];
  yout[1] = yin[1] + h * 2.0 * yin[1];
}

int eulerStep(short, int m, int n, double *yin, double *yout) {
  if (m != 2 || n != 2)
    return -1;

  yout[0] = yin[0] + h * yin[0];
  yout[1] = yin[1] + h * 2.0 * yin[1];
  return 0;
}

// derivatives
constexpr double eulerScale0() { return 1.0 + h; }
constexpr double eulerScale1() { return 1.0 + 2.0 * h; }

ext_diff_fct *registerNestedEulerExternalFunction(short outerTapeId,
                                                  short innerTapeId) {
  ext_diff_fct *edf =
      reg_ext_fct(outerTapeId, innerTapeId, ADOLC_ext_fct(eulerStep));

  edf->zos_forward = [innerTapeId](short, int m, int n, double *x, double *y) {
    return ::zos_forward(innerTapeId, m, n, 0, x, y);
  };

  edf->fos_forward = [innerTapeId](short, int m, int n, double *x, double *X,
                                   double *y, double *Y) {
    return ::fos_forward(innerTapeId, m, n, 0, x, X, y, Y);
  };

  edf->fov_forward = [innerTapeId](short, int m, int n, int p, double *x,
                                   double **Xp, double *y, double **Yp) {
    return ::fov_forward(innerTapeId, m, n, p, x, Xp, y, Yp);
  };

  edf->fos_reverse = [innerTapeId](short, int m, int n, double *u, double *z,
                                   double *x, double *y) {
    ::zos_forward(innerTapeId, m, n, 1, x, y);
    findTape(innerTapeId).nestedReverseEval(true);
    const int rc = ::fos_reverse(innerTapeId, m, n, u, z);
    findTape(innerTapeId).nestedReverseEval(false);
    return rc;
  };

  edf->fov_reverse = [innerTapeId](short, int m, int n, int q, double **Uq,
                                   double **Zq, double *x, double *y) {
    ::zos_forward(innerTapeId, m, n, 1, x, y);
    findTape(innerTapeId).nestedReverseEval(true);
    const int rc = ::fov_reverse(innerTapeId, m, n, q, Uq, Zq);
    findTape(innerTapeId).nestedReverseEval(false);
    return rc;
  };

  return edf;
}

ext_diff_fct *registerManualEulerExternalFunction(short outerTapeId,
                                                  short placeholderTapeId) {
  ext_diff_fct *edf =
      reg_ext_fct(outerTapeId, placeholderTapeId, ADOLC_ext_fct(eulerStep));
  edf->nestedAdolc = 0;

  edf->zos_forward = [](short tapeId, int m, int n, double *x, double *y) {
    return eulerStep(tapeId, m, n, x, y);
  };

  edf->fos_forward = [](short tapeId, int m, int n, double *x, double *X,
                        double *y, double *Y) {
    const int rc = eulerStep(tapeId, m, n, x, y);
    if (rc != 0)
      return rc;

    Y[0] = eulerScale0() * X[0];
    Y[1] = eulerScale1() * X[1];
    return 0;
  };

  edf->fov_forward = [](short tapeId, int m, int n, int p, double *x,
                        double **Xp, double *y, double **Yp) {
    const int rc = eulerStep(tapeId, m, n, x, y);
    if (rc != 0)
      return rc;

    for (int direction = 0; direction < p; ++direction) {
      Yp[0][direction] = eulerScale0() * Xp[0][direction];
      Yp[1][direction] = eulerScale1() * Xp[1][direction];
    }
    return 0;
  };

  edf->fos_reverse = [](short, int m, int n, double *u, double *z, double *,
                        double *) {
    if (m != 2 || n != 2)
      return -1;

    // important to add here. z could contain already computed results.
    z[0] += eulerScale0() * u[0];
    z[1] += eulerScale1() * u[1];
    return 0;
  };

  edf->fov_reverse = [](short, int m, int n, int q, double **Uq, double **Zq,
                        double *, double *) {
    if (m != 2 || n != 2)
      return -1;

    // important to add here. Z could contain already computed results.
    for (int weight = 0; weight < q; ++weight) {
      Zq[weight][0] += eulerScale0() * Uq[weight][0];
      Zq[weight][1] += eulerScale1() * Uq[weight][1];
    }
    return 0;
  };

  return edf;
}

void traceFullTape(short tapeId, const PassiveState &controls) {
  trace_on(tapeId);
  {
    ActiveState control{};
    ActiveState state{};
    ActiveState nextState{};

    control <<= controls;
    state = control;

    for (int i = 0; i < steps; ++i) {
      eulerStepAct(state.data(), nextState.data());
      state = nextState;
    }

    double objective = 0.0;
    adouble(state[0] + state[1]) >>= objective;
  }
  trace_off();
}

void traceEulerInnerTape(short tapeId, const PassiveState &controls) {
  trace_on(tapeId);
  {
    ActiveState y{};
    ActiveState ynew{};
    y <<= controls;
    eulerStepAct(y.data(), ynew.data());

    std::array<double, 2> dummy{};
    ynew >>= dummy;
  }
  trace_off();
}

void traceWithExternal(short tapeId, ext_diff_fct *edf,
                       const PassiveState &controls) {
  trace_on(tapeId);
  {
    currentTape().ensureContiguousLocations(4);

    ActiveState control{};
    ActiveState state{};
    ActiveState nextState{};

    control <<= controls;
    state = control;

    for (int i = 0; i < steps; ++i) {
      call_ext_fct(edf, 2, state.data(), 2, nextState.data());
      state = nextState;
    }

    double objective = 0.0;
    adouble(state[0] + state[1]) >>= objective;
  }
  trace_off();
}

void printGradient(const char *label, const PassiveState &gradient) {
  std::printf("%s:\n gradient=( %f, %f)\n\n", label, gradient[0], gradient[1]);
}

} // namespace

int main() {
  const PassiveState controls{1.0, 1.0};
  PassiveState gradFull{0.0, 0.0};
  PassiveState gradNested{0.0, 0.0};
  PassiveState gradManual{0.0, 0.0};

  const short fullTapeId = createNewTape();
  const short nestedOuterTapeId = createNewTape();
  const short nestedInnerTapeId = createNewTape();
  const short manualOuterTapeId = createNewTape();
  // Manual callbacks do not use an inner tape. A placeholder tape id is only
  // passed here because reg_ext_fct still stores one on the edf object.
  const short manualPlaceholderTapeId = -1;

  traceFullTape(fullTapeId, controls);
  gradient(fullTapeId, 2, controls.data(), gradFull.data());
  printGradient("full taping", gradFull);

  traceEulerInnerTape(nestedInnerTapeId, controls);
  ext_diff_fct *nestedEdf =
      registerNestedEulerExternalFunction(nestedOuterTapeId, nestedInnerTapeId);
  traceWithExternal(nestedOuterTapeId, nestedEdf, controls);
  gradient(nestedOuterTapeId, 2, controls.data(), gradNested.data());
  printGradient("taping with external function facility (nested ADOL-C)",
                gradNested);

  ext_diff_fct *manualEdf = registerManualEulerExternalFunction(
      manualOuterTapeId, manualPlaceholderTapeId);
  traceWithExternal(manualOuterTapeId, manualEdf, controls);
  gradient(manualOuterTapeId, 2, controls.data(), gradManual.data());
  printGradient("taping with external function facility (manual callbacks)",
                gradManual);

  return 0;
}

/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     edfootest.cpp
 Revision: $Id$
 Contents: example for external differentiated functions

 Copyright (c) Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

---------------------------------------------------------------------------*/
#include <adolc/adolc.h>
#include <adolc/edfclasses.h>

#include <array>
#include <cstdio>
#include <memory>

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

class EulerStepNestedEdf final : public EDFobject {
public:
  explicit EulerStepNestedEdf(short outerTapeId, short innerTapeId)
      : EDFobject(outerTapeId, outerTapeId), innerTapeId_(innerTapeId) {}

  int function(short tapeId, int m, int n, double *x, double *y) override {
    return eulerStep(tapeId, m, n, x, y);
  }

  int zos_forward(short, int m, int n, double *x, double *y) override {
    return ::zos_forward(innerTapeId_, m, n, 0, x, y);
  }

  int fos_forward(short, int m, int n, double *x, double *X, double *y,
                  double *Y) override {
    return ::fos_forward(innerTapeId_, m, n, 0, x, X, y, Y);
  }

  int fov_forward(short, int m, int n, int p, double *x, double **Xp, double *y,
                  double **Yp) override {
    return ::fov_forward(innerTapeId_, m, n, p, x, Xp, y, Yp);
  }

  int fos_reverse(short, int m, int n, double *u, double *z, double *x,
                  double *y) override {
    ::zos_forward(innerTapeId_, m, n, 1, x, y);
    findTape(innerTapeId_).nestedReverseEval(true);
    const int rc = ::fos_reverse(innerTapeId_, m, n, u, z);
    findTape(innerTapeId_).nestedReverseEval(false);
    return rc;
  }

  int fov_reverse(short, int m, int n, int q, double **Uq, double **Zq,
                  double *x, double *y) override {
    ::zos_forward(innerTapeId_, m, n, 1, x, y);
    findTape(innerTapeId_).nestedReverseEval(true);
    const int rc = ::fov_reverse(innerTapeId_, m, n, q, Uq, Zq);
    findTape(innerTapeId_).nestedReverseEval(false);
    return rc;
  }

  int hos_forward(short, int, int, int, double *, double **, double *,
                  double **) override {
    return -1;
  }

  int hov_forward(short, int, int, int, int, double *, double ***, double *,
                  double ***) override {
    return -1;
  }

  int hos_reverse(short, int, int, int, double *, double **, double **,
                  double **) override {
    return -1;
  }

  int hov_reverse(short, int, int, int, int, double **, double ***, short **,
                  double **, double **) override {
    return -1;
  }

private:
  short innerTapeId_;
};

class EulerStepManualEdf final : public EDFobject {
public:
  explicit EulerStepManualEdf(short outerTapeId)
      : EDFobject(outerTapeId, outerTapeId) {
    // Manual callbacks evaluate the primal and first-order derivatives
    // directly, so this path does not need an inner tape.
    edf->nestedAdolc = 0;
  }

  int function(short tapeId, int m, int n, double *x, double *y) override {
    return eulerStep(tapeId, m, n, x, y);
  }

  int zos_forward(short tapeId, int m, int n, double *x, double *y) override {
    return eulerStep(tapeId, m, n, x, y);
  }

  int fos_forward(short tapeId, int m, int n, double *x, double *X, double *y,
                  double *Y) override {
    const int rc = eulerStep(tapeId, m, n, x, y);
    if (rc != 0)
      return rc;

    Y[0] = eulerScale0() * X[0];
    Y[1] = eulerScale1() * X[1];
    return 0;
  }

  int fov_forward(short tapeId, int m, int n, int p, double *x, double **Xp,
                  double *y, double **Yp) override {
    const int rc = eulerStep(tapeId, m, n, x, y);
    if (rc != 0)
      return rc;

    for (int direction = 0; direction < p; ++direction) {
      Yp[0][direction] = eulerScale0() * Xp[0][direction];
      Yp[1][direction] = eulerScale1() * Xp[1][direction];
    }
    return 0;
  }

  int fos_reverse(short, int m, int n, double *u, double *z, double *,
                  double *) override {
    if (m != 2 || n != 2)
      return -1;

    // important to add here. z could contain already computed results.
    z[0] += eulerScale0() * u[0];
    z[1] += eulerScale1() * u[1];
    return 0;
  }

  int fov_reverse(short, int m, int n, int q, double **Uq, double **Zq,
                  double *, double *) override {
    if (m != 2 || n != 2)
      return -1;

    // important to add here. Z could contain already computed results.
    for (int weight = 0; weight < q; ++weight) {
      Zq[weight][0] += eulerScale0() * Uq[weight][0];
      Zq[weight][1] += eulerScale1() * Uq[weight][1];
    }
    return 0;
  }

  int hos_forward(short, int, int, int, double *, double **, double *,
                  double **) override {
    return -1;
  }

  int hov_forward(short, int, int, int, int, double *, double ***, double *,
                  double ***) override {
    return -1;
  }

  int hos_reverse(short, int, int, int, double *, double **, double **,
                  double **) override {
    return -1;
  }

  int hov_reverse(short, int, int, int, int, double **, double ***, short **,
                  double **, double **) override {
    return -1;
  }
};

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

void traceWithObject(EDFobject &edf, const PassiveState &controls) {
  currentTape().ensureContiguousLocations(4);

  ActiveState control{};
  ActiveState state{};
  ActiveState nextState{};

  control <<= controls;
  state = control;

  for (int i = 0; i < steps; ++i) {
    edf.call(2, state.data(), 2, nextState.data());
    state = nextState;
  }

  double objective = 0.0;
  adouble(state[0] + state[1]) >>= objective;
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

  traceFullTape(fullTapeId, controls);
  gradient(fullTapeId, 2, controls.data(), gradFull.data());
  printGradient("full taping", gradFull);

  traceEulerInnerTape(nestedInnerTapeId, controls);
  std::unique_ptr<EulerStepNestedEdf> nestedEdf;
  trace_on(nestedOuterTapeId);
  nestedEdf = std::make_unique<EulerStepNestedEdf>(nestedOuterTapeId,
                                                   nestedInnerTapeId);
  traceWithObject(*nestedEdf, controls);
  trace_off();
  gradient(nestedOuterTapeId, 2, controls.data(), gradNested.data());
  printGradient("taping with EDFobject (nested ADOL-C)", gradNested);

  std::unique_ptr<EulerStepManualEdf> manualEdf;
  trace_on(manualOuterTapeId);
  manualEdf = std::make_unique<EulerStepManualEdf>(manualOuterTapeId);
  traceWithObject(*manualEdf, controls);
  trace_off();
  gradient(manualOuterTapeId, 2, controls.data(), gradManual.data());
  printGradient("taping with EDFobject (manual callbacks)", gradManual);

  return 0;
}

/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     checkpointing.cpp
 Revision: $Id$
 Contents: checkpointing algorithms

 Copyright (c) Andreas Kowarz, Jean Utke

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

---------------------------------------------------------------------------*/

#include <adolc/adalloc.h>
#include <adolc/adolcerror.h>
#include <adolc/adtb_types.h>
#include <adolc/checkpointing.h>
#include <adolc/externfcts.h>
#include <adolc/interfaces.h>
#include <adolc/internal/common.h>
#include <adolc/oplate.h>
#include <adolc/revolve.h>
#include <adolc/tape_interface.h>
#include <adolc/valuetape/valuetape.h>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>

namespace ADOLC::CP {

namespace {
// RAII-styler setter for dp_internal_for
// assumes dp_internal_for is nullptr and resets it
struct InternalForwardBuffer {
  detail::Infos *infos;

  InternalForwardBuffer(detail::Infos *infos, double *buffer) : infos(infos) {
    assert(infos->dp_internal_for == nullptr);
    infos->dp_internal_for = buffer;
  }

  ~InternalForwardBuffer() { infos->dp_internal_for = nullptr; }

  InternalForwardBuffer(const InternalForwardBuffer &) = delete;
  InternalForwardBuffer(InternalForwardBuffer &&) = delete;
  InternalForwardBuffer &operator=(const InternalForwardBuffer &) = delete;
  InternalForwardBuffer &operator=(InternalForwardBuffer &&) = delete;
};

detail::Infos *checkpointingInfos(ValueTape &tape, size_t cpIndex) {
  detail::Infos *cpInfos = tape.get_cp_fct(cpIndex);
  if (!cpInfos)
    ADOLCError::fail(ADOLCError::ErrorType::CP_NO_SUCH_IDX, CURRENT_LOCATION,
                     ADOLCError::FailInfo{.info2 = cpIndex});
  return cpInfos;
}

// use this if we don't want to overwrite the state when taping
void tapeStep(detail::Infos *cpInfos, const std::vector<double> &base,
              int keep = 0) {
  std::vector<double> traceState(base);
  InternalForwardBuffer guard(cpInfos, traceState.data());
  cpInfos->tapeStep(keep);
}

void setFovRows(std::vector<double> &values, int rows, int cols,
                std::vector<double *> &rowPointers) {
  for (int row = 0; row < rows; ++row)
    rowPointers[row] = values.data() + static_cast<size_t>(row) * cols;
}

} // namespace

void detail::Infos::tapeStep(int keep) const {
  trace_on(cpTapeId, keep);
  {
    std::vector<adouble> tapingAdoubles(dim);
    for (int i = 0; i < dim; ++i)
      tapingAdoubles[i] <<= dp_internal_for[i];

    function(dim, tapingAdoubles.data());

    for (int i = 0; i < dim; ++i)
      tapingAdoubles[i] >>= dp_internal_for[i];
  }
  trace_off();
}

/****************************************************************************/
/*                                                   revolve error function */
/****************************************************************************/
void detail::Infos::revolveError() {
  switch (info) {
  case 10:
    ADOLCError::fail(ADOLCError::ErrorType::CP_STORED_EXCEEDS_CU,
                     CURRENT_LOCATION);
    break;
  case 11:
    ADOLCError::fail(
        ADOLCError::ErrorType::CP_STORED_EXCEEDS_SNAPS, CURRENT_LOCATION,
        ADOLCError::FailInfo{.info3 = check + 1, .info4 = checkpoints});
    break;
  case 12:
    ADOLCError::fail(ADOLCError::ErrorType::CP_NUMFORW, CURRENT_LOCATION);
    break;
  case 13:
    ADOLCError::fail(ADOLCError::ErrorType::CP_INC_SNAPS, CURRENT_LOCATION);
    break;
  case 14:
    ADOLCError::fail(ADOLCError::ErrorType::CP_SNAPS_EXCEEDS_CU,
                     CURRENT_LOCATION);
    break;
  case 15:
    ADOLCError::fail(ADOLCError::ErrorType::CP_REPS_EXCEEDS_REPSUP,
                     CURRENT_LOCATION);
    break;
  }
}

void detail::Infos::revolveForward(int keep) {
  /* init revolve */
  check = -1;
  capo = 0;
  info = 0;
  fine = steps;

  /* execute all time steps */
  enum revolve_action whattodo;
  do {
    whattodo = revolve(&check, &capo, &fine, checkpoints, &info);

    switch (whattodo) {
    case revolve_takeshot:
      takeSnapshot();
      currentCP = capo;
      break;

    case revolve_advance:
      for (int i = 0; i < capo - currentCP; ++i) {
        function_double(dim, dp_internal_for);
      }
      break;

    case revolve_firsturn:
      tapeStep(keep);
      break;

    case revolve_error:
      revolveError();
      break;

    default:
      ADOLCError::fail(
          ADOLCError::ErrorType::CHECKPOINTING_UNEXPECTED_REVOLVE_ACTION,
          CURRENT_LOCATION);
    }
  } while (whattodo == revolve_takeshot || whattodo == revolve_advance);
}

/* register one time step function (uses buffer template) */
void Context::registerTimeStepFunction(short tapeId, short tapeIdCheck,
                                       TimeStepFunction timeStepFunction) {

  ValueTape &tape = findTape(tapeId);
  cpInfos = tape.cp_append();
  cpInfos->function = timeStepFunction;
  cpInfos->tapeId = tapeId;
  cpInfos->cpTapeId = tapeIdCheck;
}

namespace {

/* we do not really have an ext. diff. function that we want to be called */
int dummyCheckpointingFunction(short, int, int, double *, double *) {
  return 0;
}

void checkInput(short tapeId, detail::Infos *cpInfos) {
  using ADOLCError::fail;
  using ADOLCError::FailInfo;
  using ADOLCError::ErrorType::CHECKPOINTING_CPINFOS_NULLPOINTER;
  using ADOLCError::ErrorType::CHECKPOINTING_NULLPOINTER_ARGUMENT;
  using ADOLCError::ErrorType::CHECKPOINTING_NULLPOINTER_FUNCTION;
  using ADOLCError::ErrorType::CHECKPOINTING_NULLPOINTER_FUNCTION_DOUBLE;
  using ADOLCError::ErrorType::CP_TAPE_MISMATCH;

  if (cpInfos == nullptr)
    fail(CHECKPOINTING_CPINFOS_NULLPOINTER, CURRENT_LOCATION);
  if (tapeId != cpInfos->tapeId)
    fail(CP_TAPE_MISMATCH, CURRENT_LOCATION,
         FailInfo{.info2 = to_size_t(cpInfos->tapeId), .info3 = tapeId});
  if (cpInfos->function == nullptr)
    fail(CHECKPOINTING_NULLPOINTER_FUNCTION, CURRENT_LOCATION);
  if (cpInfos->function_double == nullptr)
    fail(CHECKPOINTING_NULLPOINTER_FUNCTION_DOUBLE, CURRENT_LOCATION);
  if (cpInfos->adp_x == nullptr)
    fail(CHECKPOINTING_NULLPOINTER_ARGUMENT, CURRENT_LOCATION);
}

/****************************************************************************/
/* the following are the main checkpointing functions called by the         */
/* external differentiated function alogrithms                              */
/****************************************************************************/

/* special case: use double version where possible, no taping */
int zos_forward(size_t cpIndex, short tapeId, [[maybe_unused]] int m, int n,
                int keep, const double *x, double *y) {
  ValueTape &tape = findTape(tapeId);

  // get checkpointing information
  detail::Infos *cpInfos = checkpointingInfos(tape, cpIndex);

  assert(cpInfos->dim == m && "CP: Output Dimension mismatch!");
  assert(cpInfos->dim == n && "CP: Input Dimension mismatch!");
  std::vector<double> state(x, x + n);
  InternalForwardBuffer guard(cpInfos, state.data());

  cpInfos->revolveForward(keep);

  // write back
  std::copy(state.begin(), state.end(), y);
  return 0;
}

int fos_forward(size_t cpIndex, short tapeId, int m, int n, int,
                const double *x, const double *X, double *y, double *Y) {
  ValueTape &tape = findTape(tapeId);
  detail::Infos *cpInfos = checkpointingInfos(tape, cpIndex);

  assert(cpInfos->dim == m && "CP: Output Dimension mismatch!");
  assert(cpInfos->dim == n && "CP: Input Dimension mismatch!");

  std::vector<double> state(x, x + n);
  std::vector<double> tangent(X, X + n);
  std::vector<double> nextState(m);
  std::vector<double> nextTangent(m);

  int ret_c = 3;
  for (int step = 0; step < cpInfos->steps; ++step) {
    if (step == 0 || cpInfos->retaping != 0)
      tapeStep(cpInfos, state);

    int rc =
        ::fos_forward(cpInfos->cpTapeId, m, n, 0, state.data(), tangent.data(),
                      nextState.data(), nextTangent.data());

    if (rc < 0 && cpInfos->retaping == 0) {
      tapeStep(cpInfos, state);
      rc = ::fos_forward(cpInfos->cpTapeId, m, n, 0, state.data(),
                         tangent.data(), nextState.data(), nextTangent.data());
    }

    MINDEC(ret_c, rc);
    state.swap(nextState);
    tangent.swap(nextTangent);
  }

  std::copy(state.begin(), state.end(), y);
  std::copy(tangent.begin(), tangent.end(), Y);
  return ret_c;
}

int fov_forward(size_t cpIndex, short tapeId, int m, int n, int p,
                const double *x, double **Xp, double *y, double **Yp) {
  ValueTape &tape = findTape(tapeId);
  detail::Infos *cpInfos = checkpointingInfos(tape, cpIndex);

  assert(cpInfos->dim == m && "CP: Output Dimension mismatch!");
  assert(cpInfos->dim == n && "CP: Input Dimension mismatch!");

  std::vector<double> state(x, x + n);
  std::vector<double> tangent(static_cast<size_t>(n) * p);
  std::vector<double> nextState(m);
  std::vector<double> nextTangent(static_cast<size_t>(m) * p);
  std::vector<double *> tangentRows(n);
  std::vector<double *> nextTangentRows(m);

  for (int row = 0; row < n; ++row)
    for (int col = 0; col < p; ++col)
      tangent[static_cast<size_t>(row) * p + col] = Xp[row][col];

  int ret_c = 3;
  for (int step = 0; step < cpInfos->steps; ++step) {
    if (step == 0 || cpInfos->retaping != 0)
      tapeStep(cpInfos, state);

    setFovRows(tangent, n, p, tangentRows);
    setFovRows(nextTangent, m, p, nextTangentRows);
    int rc = ::fov_forward(cpInfos->cpTapeId, m, n, p, state.data(),
                           tangentRows.data(), nextState.data(),
                           nextTangentRows.data());

    if (rc < 0 && cpInfos->retaping == 0) {
      tapeStep(cpInfos, state);
      setFovRows(tangent, n, p, tangentRows);
      setFovRows(nextTangent, m, p, nextTangentRows);
      rc = ::fov_forward(cpInfos->cpTapeId, m, n, p, state.data(),
                         tangentRows.data(), nextState.data(),
                         nextTangentRows.data());
    }

    MINDEC(ret_c, rc);
    state.swap(nextState);
    tangent.swap(nextTangent);
  }

  std::copy(state.begin(), state.end(), y);
  for (int row = 0; row < m; ++row)
    for (int col = 0; col < p; ++col)
      Yp[row][col] = tangent[static_cast<size_t>(row) * p + col];
  return ret_c;
}

int hos_forward(size_t, short, int, int, int, int, double *, double **,
                double *, double **) {
  printf("WARNING: Checkpointing algorithm not "
         "implemented for the hos_forward mode!\n");
  return 0;
}

int hov_forward(size_t, short, size_t, double *, size_t, size_t, double ***,
                size_t, double *, double ***) {
  printf("WARNING: Checkpointing algorithm not "
         "implemented for the hov_forward mode!\n");
  return 0;
}

int fos_reverse(size_t cpIndex, short tapeId, int m, int n, double *u,
                double *z, double *x, double *) {

  ValueTape &tape = findTape(tapeId);

  detail::Infos *cpInfos = checkpointingInfos(tape, cpIndex);

  assert(cpInfos->dim == m && "CP: Output Dimension mismatch!");
  assert(cpInfos->dim == n && "CP: Input Dimension mismatch!");

  std::vector<double> state(x, x + n);
  std::vector<double> adjoint(u, u + m);
  InternalForwardBuffer guard(cpInfos, state.data());

  // execute second part of revolve_firstturn left from forward sweep
  ::fos_reverse(cpInfos->cpTapeId, m, n, adjoint.data(), adjoint.data());

  const char old_bsw = tape.branchSwitchWarning();
  tape.branchSwitchWarning(0);

  // checkpointing
  enum revolve_action whattodo;
  do {
    whattodo = revolve(&cpInfos->check, &cpInfos->capo, &cpInfos->fine,
                       cpInfos->checkpoints, &cpInfos->info);
    switch (whattodo) {
    case revolve_terminate:
      break;

    case revolve_takeshot:
      cpInfos->takeSnapshot();
      cpInfos->currentCP = cpInfos->capo;
      break;

    case revolve_advance:
      for (int i = 0; i < cpInfos->capo - cpInfos->currentCP; ++i)
        cpInfos->function_double(cpInfos->dim, state.data());
      break;

    case revolve_youturn:
      if (cpInfos->retaping != 0)
        cpInfos->tapeStep(1); // retaping forced
      else {
        // one forward step with keep and retaping if necessary
        const std::vector<double> stepState(state);
        if (::zos_forward(cpInfos->cpTapeId, cpInfos->dim, cpInfos->dim, 1,
                          state.data(), state.data()) < 0) {
          state = stepState;
          cpInfos->tapeStep(1);
        }
      }
      // one reverse step
      ::fos_reverse(cpInfos->cpTapeId, cpInfos->dim, cpInfos->dim,
                    adjoint.data(), adjoint.data());
      break;

    case revolve_restore:
      if (cpInfos->capo != cpInfos->currentCP)
        cpInfos->release();
      cpInfos->currentCP = cpInfos->capo;
      cpInfos->restore();
      break;

    case revolve_error:
      cpInfos->revolveError();
      break;

    default:
      ADOLCError::fail(
          ADOLCError::ErrorType::CHECKPOINTING_UNEXPECTED_REVOLVE_ACTION,
          CURRENT_LOCATION);
      break;
    }
  } while (whattodo != revolve_terminate && whattodo != revolve_error);
  cpInfos->release(); // release first checkpoint if written
  tape.branchSwitchWarning(old_bsw);

  // save results
  std::copy(adjoint.begin(), adjoint.end(), z);

  return 0;
}

int fov_reverse(size_t cpIndex, short tapeId, int m, int n, int q, double **Uq,
                double **Zq, double *x, double *) {

  ValueTape &tape = findTape(tapeId);

  detail::Infos *cpInfos = checkpointingInfos(tape, cpIndex);

  assert(cpInfos->dim == m && "CP: Output Dimension mismatch!");
  assert(cpInfos->dim == n && "CP: Input Dimension mismatch!");

  std::vector<double> state(x, x + n);
  std::vector<double> adjoint(static_cast<size_t>(q) * m);
  std::vector<double *> adjointRows(q);
  InternalForwardBuffer guard(cpInfos, state.data());

  for (int row = 0; row < q; ++row)
    std::copy(Uq[row], Uq[row] + m,
              adjoint.data() + static_cast<size_t>(row) * m);
  setFovRows(adjoint, q, m, adjointRows);

  // execute second part of revolve_firstturn left from forward sweep
  ::fov_reverse(cpInfos->cpTapeId, m, n, q, adjointRows.data(),
                adjointRows.data());

  const char old_bsw = tape.branchSwitchWarning();
  tape.branchSwitchWarning(0);
  // checkpointing
  enum revolve_action whattodo;
  do {
    whattodo = revolve(&cpInfos->check, &cpInfos->capo, &cpInfos->fine,
                       cpInfos->checkpoints, &cpInfos->info);
    switch (whattodo) {
    case revolve_terminate:
      break;

    case revolve_takeshot:
      cpInfos->takeSnapshot();
      cpInfos->currentCP = cpInfos->capo;
      break;

    case revolve_advance:
      for (int i = 0; i < cpInfos->capo - cpInfos->currentCP; ++i)
        cpInfos->function_double(cpInfos->dim, state.data());
      break;

    case revolve_youturn:
      if (cpInfos->retaping != 0)
        cpInfos->tapeStep(1); // retaping forced
      else {
        // one forward step with keep and retaping if necessary
        const std::vector<double> stepState(state);
        if (::zos_forward(cpInfos->cpTapeId, cpInfos->dim, cpInfos->dim, 1,
                          state.data(), state.data()) < 0) {
          state = stepState;
          cpInfos->tapeStep(1);
        }
      }
      // one reverse step
      ::fov_reverse(cpInfos->cpTapeId, cpInfos->dim, cpInfos->dim, q,
                    adjointRows.data(), adjointRows.data());
      break;

    case revolve_restore:
      if (cpInfos->capo != cpInfos->currentCP)
        cpInfos->release();
      cpInfos->currentCP = cpInfos->capo;
      cpInfos->restore();
      break;

    case revolve_error:
      cpInfos->revolveError();
      break;

    default:
      ADOLCError::fail(
          ADOLCError::ErrorType::CHECKPOINTING_UNEXPECTED_REVOLVE_ACTION,
          CURRENT_LOCATION);
      break;
    }
  } while (whattodo != revolve_terminate && whattodo != revolve_error);

  // release first checkpoint if written
  cpInfos->release();
  tape.branchSwitchWarning(old_bsw);

  // save results
  for (int row = 0; row < q; ++row)
    std::copy(adjointRows[row], adjointRows[row] + n, Zq[row]);
  return 0;
}

int hos_reverse(size_t, short, size_t, double *, size_t, size_t, double **) {
  printf("WARNING: Checkpointing algorithm not "
         "implemented for the hos_reverse mode!\n");
  return 0;
}

int hov_reverse(size_t, short, size_t, size_t, double **, size_t, size_t,
                double ***, short **) {
  printf("WARNING: Checkpointing algorithm not "
         "implemented for the hov_reverse mode!\n");
  return 0;
}

} // namespace

/****************************************************************************/
/*                              functions for handling the checkpoint stack */
/****************************************************************************/

void detail::Infos::clearStack() {
  detail::StackElement shot;
  while (!cpStack.empty()) {
    shot = cpStack.top();
    cpStack.pop();
    delete[] shot[0];
    delete[] shot[1];
  }
}

void detail::Infos::takeSnapshot() {
  detail::StackElement shot;
  shot[0] = new double[dim];
  for (int i = 0; i < dim; ++i)
    shot[0][i] = dp_internal_for[i];
  if (saveNonAdoubles != nullptr)
    shot[1] = static_cast<double *>(saveNonAdoubles());
  else
    shot[1] = nullptr;
  cpStack.push(shot);
}

void detail::Infos::restore() {
  using ADOLCError::fail;
  using ADOLCError::FailInfo;
  using ADOLCError::ErrorType::CP_EMPTY_STACK;
  if (cpStack.empty())
    fail(CP_EMPTY_STACK, CURRENT_LOCATION, FailInfo{.info2 = index});

  detail::StackElement shot = cpStack.top();
  for (int i = 0; i < dim; ++i)
    dp_internal_for[i] = shot[0][i];

  if (shot[1] != nullptr)
    restoreNonAdoubles(static_cast<void *>(shot[1]));
}

void detail::Infos::release() {
  if (!cpStack.empty()) {
    detail::StackElement shot = cpStack.top();
    cpStack.pop();
    delete[] shot[0];

    if (shot[1] != nullptr)
      delete[] shot[1];
  }
}

/* initialize the information for the external function in a way that our
 * checkpointing functions are called */
namespace {

void initExternalDiffFunction(ext_diff_fct *edf) {
  edf->function = dummyCheckpointingFunction;

  // ZOS FORWARD
  edf->zos_forward = [edf](short tapeId, int m, int n, int keep, double *x,
                           double *y) {
    return ADOLC::CP::zos_forward(edf->cp_index, tapeId, m, n, keep, x, y);
  };

  // FOS FORWARD
  edf->fos_forward = [edf](short tapeId, int m, int n, int keep, double *x,
                           double *X, double *y, double *Y) {
    return ADOLC::CP::fos_forward(edf->cp_index, tapeId, m, n, keep, x, X, y,
                                  Y);
  };

  // FOV FORWARD
  edf->fov_forward = [edf](short tapeId, int m, int n, int p, double *x,
                           double **Xp, double *y, double **Yp) {
    return ADOLC::CP::fov_forward(edf->cp_index, tapeId, m, n, p, x, Xp, y, Yp);
  };

  // HOS FORWARD
  edf->hos_forward = [edf](short tapeId, int m, int n, int d, int keep,
                           double *x, double **Xd, double *y, double **Yd) {
    return ADOLC::CP::hos_forward(edf->cp_index, tapeId, m, n, d, keep, x, Xd,
                                  y, Yd);
  };

  // HOV FORWARD
  edf->hov_forward = [edf](short tapeId, int m, int n, int d, int p, double *x,
                           double ***Xpd, double *y, double ***Ypd) {
    return ADOLC::CP::hov_forward(edf->cp_index, tapeId, n, x, d, p, Xpd, m, y,
                                  Ypd);
  };

  // FOS REVERSE
  edf->fos_reverse = [edf](short tapeId, int m, int n, double *u, double *z,
                           double *x, double *y) {
    return ADOLC::CP::fos_reverse(edf->cp_index, tapeId, m, n, u, z, x, y);
  };

  // FOV REVERSE
  edf->fov_reverse = [edf](short tapeId, int m, int n, int q, double **Uq,
                           double **Zq, double *x, double *y) {
    return ADOLC::CP::fov_reverse(edf->cp_index, tapeId, m, n, q, Uq, Zq, x, y);
  };

  // HOS REVERSE
  edf->hos_reverse = [edf](short tapeId, int m, int n, int d, double *u,
                           double **Zd, double **, double **) {
    return ADOLC::CP::hos_reverse(edf->cp_index, tapeId, n, u, d, m, Zd);
  };

  // HOV REVERSE
  edf->hov_reverse = [edf](short tapeId, int m, int n, int d, int q,
                           double **Uq, double ***Zqd, short **nz, double **,
                           double **) {
    return ADOLC::CP::hov_reverse(edf->cp_index, tapeId, n, d, Uq, m, q, Zqd,
                                  nz);
  };
}
} // namespace

int Context::checkpointing(short tapeId) {
  // throws if input is invalid
  checkInput(tapeId, cpInfos);

  // register extern function
  ext_diff_fct *edf = reg_ext_fct(cpInfos->tapeId, cpInfos->cpTapeId,
                                  dummyCheckpointingFunction);
  initExternalDiffFunction(edf);
  edf->cp_index = cpInfos->index;
  edf->dp_x_changes = 0;
  edf->dp_y_priorRequired = 0;
  ValueTape &tape = findTape(cpInfos->tapeId);
  // but we do not call it
  // we use direct taping to avoid unnecessary argument copying

  tape.put_op(ext_diff);
  tape.put_loc(edf->index);
  tape.put_loc(cpInfos->dim);
  tape.put_loc(cpInfos->dim);
  edf->firstIndLocation = cpInfos->adp_x[0].loc();
  edf->firstDepLocation = cpInfos->adp_y[0].loc();

  std::vector<double> vals(tape.store(), tape.store() + tape.storeSize());

  std::vector<double> state(cpInfos->dim);
  InternalForwardBuffer guard(cpInfos, state.data());

  // initialize internal arguments
  for (int i = 0; i < cpInfos->dim; ++i)
    state[i] = cpInfos->adp_x[i].value();

  if (tape.keepTaylors()) {
    // perform all time steps, tape the last, take checkpoints
    cpInfos->revolveForward(1);
  } else
    // perform all time steps without taping
    for (int i = 0; i < cpInfos->steps; ++i)
      cpInfos->function_double(cpInfos->dim, state.data());

  std::copy(vals.begin(), vals.end(), tape.store());

  // update taylor stack; same structure as in adouble.cpp +
  // correction in taping.cpp
  tape.add_numTays_Tape(cpInfos->dim);
  if (tape.keepTaylors())
    for (int i = 0; i < cpInfos->dim; ++i)
      tape.write_scaylor(cpInfos->adp_y[i].value());

  // save results
  for (int i = 0; i < cpInfos->dim; ++i) {
    cpInfos->adp_y[i].value(state[i]);
  }

  return 0;
}
} // namespace ADOLC::CP

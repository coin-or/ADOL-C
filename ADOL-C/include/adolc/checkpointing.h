/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     checkpointing.h
 Revision: $Id$
 Contents: Provides all checkointing interfaces.

 Copyright (c) Andreas Kowarz

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#ifndef ADOLC_CHECKPOINTING_H
#define ADOLC_CHECKPOINTING_H

#include <adolc/adolcexport.h>
#include <adolc/internal/common.h>
#include <array>
#include <stack>

class adouble;

namespace ADOLC::CP {

/**
 * @brief Signature of the active one-step function.
 *
 * The function advances the state vector in place during taping.
 */
using TimeStepFunction = int(size_t dimX, adouble *state);

/**
 * @brief Signature of the passive one-step function.
 *
 * The function advances the state vector in place without taping.
 */
using TimeStepFunctionDouble = int(size_t dimX, double *state);

/**
 * @brief Signature of an optional callback that saves extra non-adouble state.
 */
using SaveFunction = void *(void);

/**
 * @brief Signature of an optional callback that restores saved non-adouble
 * state.
 */
using RestoreFunction = void(void *);

namespace detail {

/**
 * @brief Snapshot entry stored on the checkpoint stack.
 *
 * The first pointer stores the active state. The second pointer can store
 * optional user-managed data returned by callbacks.
 */
using StackElement = std::array<double *, 2>;

/**
 * @brief Internal checkpointing state shared between the public context and
 * the recorded external function.
 *
 * User code should configure checkpointing through Context instead of touching
 * this struct directly.
 */
struct Infos {
  ~Infos() { clearStack(); }
  // id of the outer tape, used to get checkpoint in the cp_fos_forward... and
  // reverse methods later
  short tapeId{0};
  TimeStepFunction *function{nullptr};
  TimeStepFunctionDouble *function_double{nullptr};
  SaveFunction *saveNonAdoubles{nullptr};
  RestoreFunction *restoreNonAdoubles{nullptr};
  int steps{0};
  int checkpoints{0};

  // Id of the tape that stores the checkpointing steps. This id
  // should not be confused with the id of the tape that calls
  // the checkpointing process later
  short cpTapeId{0};
  bool retaping{0}; /* != 0 forces retaping before every reverse step */

  int dim{0};              /* number of variables in input and output (n=m) */
  adouble *adp_x{nullptr}; /* input of the first step */
  adouble *adp_y{nullptr}; /* output of the last step; will be set by ADOLC */

  int check{0};
  int capo{0};
  int fine{0};
  int info{0};
  int currentCP{0};
  double *dp_internal_for{nullptr};
  size_t index{0};
  // unsued but required to be stored in our custom buffer type
  char *allmem{nullptr};
  std::stack<StackElement> cpStack;

  /** @brief Save the current checkpoint state on the internal stack. */
  void takeSnapshot();

  /** @brief Restore the current checkpoint state from the top of the stack. */
  void restore();

  /** @brief Drop the most recent checkpoint snapshot. */
  void release();

  /** @brief Free all stored checkpoint snapshots. */
  void clearStack();

  /** @brief Tape one time step at the current internal state. */
  void tapeStep(int keep) const;

  /** @brief Run the forward checkpoint schedule for the current state. */
  void revolveForward(int keep);

  /** @brief Raise an ADOL-C error for the current revolve status code. */
  void revolveError();
};
} // namespace detail
/**
 * @brief Public checkpointing setup and execution context.
 *
 * A Context ties one active time-step function to an outer tape and to a
 * second tape that stores the one-step checkpoint tape used during replay.
 */
class ADOLC_API Context {
public:
  Context() = delete;

  /**
   * @brief Create a checkpointing context for one time-stepping routine.
   *
   * @param tapeId Outer tape that records the full computation.
   * @param tapeIdCheck Tape used for the one-step checkpoint tape.
   * @param timeStepFunction Active one-step function.
   */
  Context(short tapeId, short tapeIdCheck, TimeStepFunction timeStepFunction) {
    registerTimeStepFunction(tapeId, tapeIdCheck, timeStepFunction);
  }

  ~Context() = default;

  /**
   * @brief Register the passive one-step function used for untaped replay.
   *
   * @param timeStepFunction Passive one-step function.
   */
  void setDoubleFct(TimeStepFunctionDouble timeStepFunction) {
    cpInfos->function_double = timeStepFunction;
  }

  /**
   * @brief Register an optional callback that saves extra non-adouble state
   * at each checkpoint.
   *
   * @param saveState Save callback.
   */
  void setSaveFct(SaveFunction saveState) {
    cpInfos->saveNonAdoubles = saveState;
  }

  /**
   * @brief Register an optional callback that restores extra non-adouble state
   * from a checkpoint.
   *
   * @param restoreState Restore callback.
   */
  void setRestoreFct(RestoreFunction restoreState) {
    cpInfos->restoreNonAdoubles = restoreState;
  }

  /**
   * @brief Set the total number of time steps in the checkpointed loop.
   *
   * @param number Number of time steps.
   */
  void setNumberOfSteps(int number) { cpInfos->steps = number; }

  /**
   * @brief Set the number of checkpoints available to revolve.
   *
   * @param number Number of stored checkpoints.
   */
  void setNumberOfCheckpoints(int number) { cpInfos->checkpoints = number; }

  /**
   * @brief Set the dimension of the in-place state vector.
   *
   * @param dim Number of input and output variables per time step.
   */
  void setDimensionXY(int dim) { cpInfos->dim = dim; }

  /**
   * @brief Set the active input state of the checkpointed loop.
   *
   * @param x Pointer to the first active state entry before the loop.
   */
  void setInput(adouble *x) { cpInfos->adp_x = x; }

  /**
   * @brief Set the active output state of the checkpointed loop.
   *
   * @param y Pointer to the first active state entry after the loop.
   */
  void setOutput(adouble *y) { cpInfos->adp_y = y; }

  /**
   * @brief Control whether the one-step tape is rebuilt before every step.
   *
   * @param state If true, always retape. If false, reuse the step tape until a
   * branch change forces retaping.
   */
  void setAlwaysRetaping(bool state) { cpInfos->retaping = state; }

  /**
   * @brief Replace the explicit time loop by a checkpointed external function
   * call during taping.
   *
   * This runs the primal time stepping once and records the information needed
   * for checkpoint-aware forward and reverse drivers on the outer tape.
   *
   * @param tapeId Outer tape id used for the surrounding trace.
   * @return Zero on success.
   */
  int checkpointing(short tapeId);

private:
  /** @brief Register the active time-step function and allocate checkpointing
   * storage on the outer tape. */
  void registerTimeStepFunction(short tapeId, short tapeIdCheck,
                                TimeStepFunction timeStepFunction);
  detail::Infos *cpInfos;
};

} // namespace ADOLC::CP

using ADOLC_TimeStepFuncion = ADOLC::CP::TimeStepFunction;
using ADOLC_TimeStepFuncion_double = ADOLC::CP::TimeStepFunctionDouble;
using ADOLC_saveFct = ADOLC::CP::SaveFunction;
using ADOLC_restoreFct = ADOLC::CP::RestoreFunction;
using CpInfos = ADOLC::CP::detail::Infos;
using CP_Context = ADOLC::CP::Context;

/****************************************************************************/
#endif /* ADOLC_CHECKPOINTING_H */

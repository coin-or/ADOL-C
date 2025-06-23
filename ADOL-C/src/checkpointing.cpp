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
#include <adolc/adtb_types.h>
#include <adolc/checkpointing.h>
#include <adolc/checkpointing_p.h>
#include <adolc/externfcts.h>
#include <adolc/interfaces.h>
#include <adolc/oplate.h>
#include <adolc/revolve.h>
#include <adolc/tape_interface.h>
#include <adolc/valuetape/valuetape.h>

#include <cstring>

/* forward function declarations */
void init_edf(ext_diff_fct *edf);
ADOLC_ext_fct cp_zos_forward;
ADOLC_ext_fct_fos_forward cp_fos_forward;
ADOLC_ext_fct_fov_forward cp_fov_forward;
ADOLC_ext_fct_hos_forward cp_hos_forward;
ADOLC_ext_fct_hov_forward cp_hov_forward;
ADOLC_ext_fct_fos_reverse cp_fos_reverse;
ADOLC_ext_fct_fov_reverse cp_fov_reverse;
ADOLC_ext_fct_hos_reverse cp_hos_reverse;
ADOLC_ext_fct_hov_reverse cp_hov_reverse;

void cp_taping(CpInfos *cpInfos) {
  trace_on(cpInfos->cp_tape_id, 1);
  {
    std::vector<adouble> tapingAdoubles(cpInfos->dim);
    for (size_t i = 0; i < cpInfos->dim; ++i)
      tapingAdoubles[i] <<= cpInfos->dp_internal_for[i];

    cpInfos->function(cpInfos->dim, tapingAdoubles.data());

    for (size_t i = 0; i < cpInfos->dim; ++i)
      tapingAdoubles[i] >>= cpInfos->dp_internal_for[i];
  }
  trace_off();
}

/****************************************************************************/
/*                                                   revolve error function */
/****************************************************************************/
void revolveError(CpInfos *cpInfos) {
  switch (cpInfos->info) {
  case 10:
    ADOLCError::fail(ADOLCError::ErrorType::CP_STORED_EXCEEDS_CU,
                     CURRENT_LOCATION);
  case 11:
    ADOLCError::fail(ADOLCError::ErrorType::CP_STORED_EXCEEDS_SNAPS,
                     CURRENT_LOCATION,
                     ADOLCError::FailInfo{.info3 = cpInfos->check + 1,
                                          .info6 = cpInfos->checkpoints});
  case 12:
    ADOLCError::fail(ADOLCError::ErrorType::CP_NUMFORW, CURRENT_LOCATION);
  case 13:
    ADOLCError::fail(ADOLCError::ErrorType::CP_INC_SNAPS, CURRENT_LOCATION);
  case 14:
    ADOLCError::fail(ADOLCError::ErrorType::CP_SNAPS_EXCEEDS_CU,
                     CURRENT_LOCATION);
  case 15:
    ADOLCError::fail(ADOLCError::ErrorType::CP_REPS_EXCEEDS_REPSUP,
                     CURRENT_LOCATION);
  }
}

void revolve_for(short tapeId, CpInfos *cpInfos) {
  /* init revolve */
  cpInfos->check = -1;
  cpInfos->capo = 0;
  cpInfos->info = 0;
  cpInfos->fine = cpInfos->steps;

  ValueTape &tape = findTape(tapeId);
  /* execute all time steps */
  enum revolve_action whattodo;
  do {
    whattodo = revolve(&cpInfos->check, &cpInfos->capo, &cpInfos->fine,
                       cpInfos->checkpoints, &cpInfos->info);

    switch (whattodo) {
    case revolve_takeshot:
      tape.cp_takeshot(cpInfos);
      cpInfos->currentCP = cpInfos->capo;
      break;

    case revolve_advance:
      for (size_t i = 0; i < cpInfos->capo - cpInfos->currentCP; ++i) {
        cpInfos->function_double(cpInfos->dim, cpInfos->dp_internal_for);
      }
      break;

    case revolve_firsturn:
      cp_taping(cpInfos);
      break;

    case revolve_error:
      revolveError(cpInfos);
      break;

    default:
      ADOLCError::fail(
          ADOLCError::ErrorType::CHECKPOINTING_UNEXPECTED_REVOLVE_ACTION,
          CURRENT_LOCATION);
    }
  } while (whattodo == revolve_takeshot || whattodo == revolve_advance);
}

/* we do not really have an ext. diff. function that we want to be called */
int dummy(short tapeId, size_t dim_x, double *x, size_t dim_y, double *y) {
  return 0;
}

/* register one time step function (uses buffer template) */
CpInfos *reg_timestep_fct(short tapeId, short cp_tape_id,
                          ADOLC_TimeStepFuncion timeStepFunction) {

  ValueTape &tape = findTape(tapeId);
  CpInfos *theCpInfos = tape.cp_append();
  theCpInfos->function = timeStepFunction;
  theCpInfos->tapeId = tapeId;
  theCpInfos->cp_tape_id = cp_tape_id;
  return theCpInfos;
}

void check_input(short tapeId, CpInfos *cpInfos) { // knockout
  if (tapeId != cpInfos->tapeId)
    ADOLCError::fail(
        ADOLCError::ErrorType::CP_TAPE_MISMATCH, CURRENT_LOCATION,
        ADOLCError::FailInfo{.info2 = cpInfos->tapeId, .info3 = tapeId});
  if (cpInfos == nullptr)
    ADOLCError::fail(ADOLCError::ErrorType::CHECKPOINTING_CPINFOS_NULLPOINTER,
                     CURRENT_LOCATION);
  if (cpInfos->function == nullptr)
    ADOLCError::fail(ADOLCError::ErrorType::CHECKPOINTING_NULLPOINTER_FUNCTION,
                     CURRENT_LOCATION);
  if (cpInfos->function_double == nullptr)
    ADOLCError::fail(
        ADOLCError::ErrorType::CHECKPOINTING_NULLPOINTER_FUNCTION_DOUBLE,
        CURRENT_LOCATION);
  if (cpInfos->adp_x == nullptr)
    ADOLCError::fail(ADOLCError::ErrorType::CHECKPOINTING_NULLPOINTER_ARGUMENT,
                     CURRENT_LOCATION);
}
/* This is the main checkpointing function the user calls within the taping
 * process. It performs n time steps with or without taping and registers an
 * external dummy function which calls the actual checkpointing workhorses
 * from within the used drivers. */
int checkpointing(short tapeId, CpInfos *cpInfos) {

  // throws if input is invalid
  check_input(tapeId, cpInfos);

  // register extern function
  ext_diff_fct *edf = reg_ext_fct(cpInfos->tapeId, cpInfos->cp_tape_id, dummy);
  init_edf(edf);

  ValueTape &tape = findTape(cpInfos->tapeId);
  size_t oldTraceFlag = 0;
  // but we do not call it
  // we use direct taping to avoid unnecessary argument copying
  if (tape.traceFlag()) {
    tape.put_op(ext_diff);
    tape.put_loc(edf->index);
    tape.put_loc(0);
    tape.put_loc(0);
    tape.put_loc(cpInfos->adp_x[0].loc());
    tape.put_loc(cpInfos->adp_y[0].loc());
    // this CpInfos id has to be read by the actual checkpointing
    // functions
    tape.put_loc(cpInfos->index);

    oldTraceFlag = tape.traceFlag();
    tape.traceFlag(0);
  }

  std::vector<double> vals(tape.store(), tape.store() + tape.storeSize());

  cpInfos->dp_internal_for = new double[cpInfos->dim];

  // initialize internal arguments
  for (size_t i = 0; i < cpInfos->dim; ++i)
    cpInfos->dp_internal_for[i] = cpInfos->adp_x[i].value();

  if (tape.keepTaylors()) {
    // perform all time steps, tape the last, take checkpoints
    revolve_for(tapeId, cpInfos);
  } else
    // perform all time steps without taping
    for (size_t i = 0; i < cpInfos->steps; ++i)
      cpInfos->function_double(cpInfos->dim, cpInfos->dp_internal_for);

  std::copy(vals.begin(), vals.end(), tape.store());

  // update taylor stack; same structure as in adouble.cpp +
  // correction in taping.cpp
  if (oldTraceFlag) {
    tape.add_numTays_Tape(cpInfos->dim);
    if (tape.keepTaylors())
      for (size_t i = 0; i < cpInfos->dim; ++i)
        tape.write_scaylor(cpInfos->adp_y[i].value());
  }
  // save results
  for (size_t i = 0; i < cpInfos->dim; ++i) {
    cpInfos->adp_y[i].value(cpInfos->dp_internal_for[i]);
  }

  delete[] cpInfos->dp_internal_for;
  cpInfos->dp_internal_for = nullptr;

  // normal taping again
  tape.traceFlag(oldTraceFlag);

  return 0;
}

/* - reinit external function buffer and checkpointing buffer
 * - necessary when using tape within a different program */
void reinit_checkpointing() {}

/* initialize the CpInfos variable (function and index are set within
 * the template code */
void init_CpInfos(CpInfos *cpInfos) {
  char *ptr = nullptr;

  ptr = reinterpret_cast<char *>(cpInfos);
  for (size_t i = 0; i < sizeof(CpInfos); ++i)
    ptr[i] = 0;
  cpInfos->tapeId = -1;
}

/* initialize the information for the external function in a way that our
 * checkpointing functions are called */
void init_edf(ext_diff_fct *edf) {
  edf->function = dummy;
  edf->zos_forward = cp_zos_forward;
  edf->fos_forward = cp_fos_forward;
  edf->fov_forward = cp_fov_forward;
  edf->hos_forward = cp_hos_forward;
  edf->hov_forward = cp_hov_forward;
  edf->fos_reverse = cp_fos_reverse;
  edf->fov_reverse = cp_fov_reverse;
  edf->hos_reverse = cp_hos_reverse;
  edf->hov_reverse = cp_hov_reverse;
}

/****************************************************************************/
/* the following are the main checkpointing functions called by the         */
/* external differentiated function alogrithms                              */
/****************************************************************************/

/* special case: use double version where possible, no taping */
int cp_zos_forward(short tapeId, size_t dim_x, double *dp_x, size_t dim_y,
                   double *dp_y) {

  ValueTape &tape = findTape(tapeId);
  // taping off
  const size_t oldTraceFlag = tape.traceFlag();
  tape.traceFlag(0);

  // get checkpointing information
  CpInfos *cpInfos = tape.get_cp_fct(tape.cp_index());
  if (!cpInfos)
    ADOLCError::fail(ADOLCError::ErrorType::CP_NO_SUCH_IDX, CURRENT_LOCATION,
                     ADOLCError::FailInfo{.info3 = tape.cp_index()});
  // note the mode
  cpInfos->modeForward = TapeInfos::ZOS_FORWARD;
  cpInfos->modeReverse = TapeInfos::ADOLC_NO_MODE;

  // prepare arguments
  cpInfos->dp_internal_for = new double[cpInfos->dim];

  size_t arg = tape.lowestXLoc_for();
  for (size_t i = 0; i < cpInfos->dim; ++i) {
    cpInfos->dp_internal_for[i] = tape.dp_T0()[arg];
    ++arg;
  }

  revolve_for(tapeId, cpInfos);

  // write back
  arg = tape.lowestYLoc_for(); // keep input
  for (size_t i = 0; i < cpInfos->dim; ++i) {
    tape.write_scaylor(tape.dp_T0()[arg]);
    tape.dp_T0()[arg] = cpInfos->dp_internal_for[i];
    ++arg;
  }
  delete[] cpInfos->dp_internal_for;
  cpInfos->dp_internal_for = nullptr;

  // taping "on"
  tape.traceFlag(oldTraceFlag);

  return 0;
}

int cp_fos_forward(short tapeId, size_t dim_x, double *dp_x, double *dp_X,
                   size_t dim_y, double *dp_y, double *dp_Y) {
  printf("WARNING: Checkpointing algorithm not "
         "implemented for the fos_forward mode!\n");
  return 0;
}

int cp_fov_forward(short tapeId, size_t dim_X, double *dp_x, size_t num_dirs,
                   double **dpp_X, size_t dim_y, double *dp_y, double **dpp_Y) {
  printf("WARNING: Checkpointing algorithm not "
         "implemented for the fov_forward mode!\n");
  return 0;
}

int cp_hos_forward(short tapeId, size_t dim_x, double *dp_x, size_t degree,
                   double **dpp_X, size_t dim_y, double *dp_y, double **dpp_Y) {
  printf("WARNING: Checkpointing algorithm not "
         "implemented for the hos_forward mode!\n");
  return 0;
}

int cp_hov_forward(short tapeId, size_t dim_x, double *dp_x, size_t degree,
                   size_t num_dirs, double ***dppp_X, size_t dim_y,
                   double *dp_y, double ***dppp_Y) {
  printf("WARNING: Checkpointing algorithm not "
         "implemented for the hov_forward mode!\n");
  return 0;
}

int cp_fos_reverse(short tapeId, size_t dim_y, double *dp_U, size_t dim_x,
                   double *dp_Z, double *dp_x, double *dp_y) {

  ValueTape &tape = findTape(tapeId);

  CpInfos *cpInfos = tape.get_cp_fct(tape.cp_index());

  // note the mode
  cpInfos->modeReverse = TapeInfos::FOS_REVERSE;

  cpInfos->dp_internal_for = new double[cpInfos->dim];
  cpInfos->dp_internal_rev = new double[cpInfos->dim];

  // taping "off"
  const size_t oldTraceFlag = tape.traceFlag();
  tape.traceFlag(0);

  size_t arg = tape.lowestYLoc_rev();
  for (size_t i = 0; i < cpInfos->dim; ++i) {
    cpInfos->dp_internal_rev[i] = tape.rp_A()[arg];
    ++arg;
  }
  // update taylor buffer
  for (size_t i = 0; i < cpInfos->dim; ++i) {
    --arg;
    tape.get_taylor(arg);
  }
  // execute second part of revolve_firstturn left from forward sweep
  fos_reverse(cpInfos->cp_tape_id, cpInfos->dim, cpInfos->dim,
              cpInfos->dp_internal_rev, cpInfos->dp_internal_rev);

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
      tape.cp_takeshot(cpInfos);
      cpInfos->currentCP = cpInfos->capo;
      break;

    case revolve_advance:
      for (size_t i = 0; i < cpInfos->capo - cpInfos->currentCP; ++i)
        cpInfos->function_double(cpInfos->dim, cpInfos->dp_internal_for);
      break;

    case revolve_youturn:
      if (cpInfos->retaping != 0)
        cp_taping(cpInfos); // retaping forced
      else {
        // one forward step with keep and retaping if necessary
        if (zos_forward(cpInfos->cp_tape_id, cpInfos->dim, cpInfos->dim, 1,
                        cpInfos->dp_internal_for, cpInfos->dp_internal_for) < 0)
          cp_taping(cpInfos);
      }
      // one reverse step
      fos_reverse(cpInfos->cp_tape_id, cpInfos->dim, cpInfos->dim,
                  cpInfos->dp_internal_rev, cpInfos->dp_internal_rev);
      break;

    case revolve_restore:
      if (cpInfos->capo != cpInfos->currentCP)
        tape.cp_release(cpInfos);
      cpInfos->currentCP = cpInfos->capo;
      tape.cp_restore(cpInfos);
      break;

    case revolve_error:
      revolveError(cpInfos);
      break;

    default:
      ADOLCError::fail(
          ADOLCError::ErrorType::CHECKPOINTING_UNEXPECTED_REVOLVE_ACTION,
          CURRENT_LOCATION);
      break;
    }
  } while (whattodo != revolve_terminate && whattodo != revolve_error);
  tape.cp_release(cpInfos); // release first checkpoint if written
  tape.branchSwitchWarning(old_bsw);

  // save results
  size_t start = tape.lowestYLoc_rev();
  std::copy(cpInfos->dp_internal_rev, cpInfos->dp_internal_rev + cpInfos->dim,
            tape.rp_A() + start);

  // clean up
  delete[] cpInfos->dp_internal_for;
  cpInfos->dp_internal_for = nullptr;
  delete[] cpInfos->dp_internal_rev;
  cpInfos->dp_internal_rev = nullptr;

  // taping "on"
  tape.traceFlag(oldTraceFlag);
  return 0;
}

int cp_fov_reverse(short tapeId, size_t dim_y, size_t num_weights,
                   double **dpp_U, size_t dim_x, double **dpp_Z,
                   double * /*unused*/, double * /*unused*/) {

  ValueTape &tape = findTape(tapeId);

  CpInfos *cpInfos = tape.get_cp_fct(tape.cp_index());

  // note the mode
  cpInfos->modeReverse = TapeInfos::FOV_REVERSE;

  const size_t numDirs = tape.numDirs_rev();
  cpInfos->dp_internal_for = new double[cpInfos->dim];
  cpInfos->dpp_internal_rev = myalloc2(numDirs, cpInfos->dim);

  // taping "off"
  const size_t oldTraceFlag = tape.traceFlag();
  tape.traceFlag(0);

  double **rpp_A = tape.rpp_A();
  size_t start = tape.lowestYLoc_rev();

  for (size_t i = start; i < cpInfos->dim + start; ++i) {
    for (size_t j = 0; j < numDirs; ++j) {
      cpInfos->dpp_internal_rev[j][i - start] = rpp_A[i][j];
    }
  }

  // update taylor buffer
  for (size_t i = start + cpInfos->dim; i-- > start;)
    tape.get_taylor(i);

  // execute second part of revolve_firstturn left from forward sweep
  fov_reverse(cpInfos->cp_tape_id, cpInfos->dim, cpInfos->dim, numDirs,
              cpInfos->dpp_internal_rev, cpInfos->dpp_internal_rev);

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
      tape.cp_takeshot(cpInfos);
      cpInfos->currentCP = cpInfos->capo;
      break;

    case revolve_advance:
      for (size_t i = 0; i < cpInfos->capo - cpInfos->currentCP; ++i)
        cpInfos->function_double(cpInfos->dim, cpInfos->dp_internal_for);
      break;

    case revolve_youturn:
      if (cpInfos->retaping != 0)
        cp_taping(cpInfos); // retaping forced
      else {
        // one forward step with keep and retaping if necessary
        if (zos_forward(cpInfos->cp_tape_id, cpInfos->dim, cpInfos->dim, 1,
                        cpInfos->dp_internal_for, cpInfos->dp_internal_for) < 0)
          cp_taping(cpInfos);
      }
      // one reverse step
      fov_reverse(cpInfos->cp_tape_id, cpInfos->dim, cpInfos->dim, numDirs,
                  cpInfos->dpp_internal_rev, cpInfos->dpp_internal_rev);
      break;

    case revolve_restore:
      if (cpInfos->capo != cpInfos->currentCP)
        tape.cp_release(cpInfos);
      cpInfos->currentCP = cpInfos->capo;
      tape.cp_restore(cpInfos);
      break;

    case revolve_error:
      revolveError(cpInfos);
      break;

    default:
      ADOLCError::fail(
          ADOLCError::ErrorType::CHECKPOINTING_UNEXPECTED_REVOLVE_ACTION,
          CURRENT_LOCATION);
      break;
    }
  } while (whattodo != revolve_terminate && whattodo != revolve_error);

  // release first checkpoint if written
  tape.cp_release(cpInfos);
  tape.branchSwitchWarning(old_bsw);

  // save results
  start = tape.lowestYLoc_rev();
  for (size_t i = start; i < cpInfos->dim + start; ++i) {
    for (size_t j = 0; j < numDirs; ++j) {
      rpp_A[i][j] = cpInfos->dpp_internal_rev[j][i];
    }
  }

  // clean up
  delete[] cpInfos->dp_internal_for;
  cpInfos->dp_internal_for = nullptr;
  myfree2(cpInfos->dpp_internal_rev);
  cpInfos->dpp_internal_rev = nullptr;

  // taping "on"
  tape.traceFlag(oldTraceFlag);

  return 0;
}

int cp_hos_reverse(short tapeId, size_t dim_y, double *dp_U, size_t dim_x,
                   size_t degree, double **dpp_Z) {
  printf("WARNING: Checkpointing algorithm not "
         "implemented for the hos_reverse mode!\n");
  return 0;
}

int cp_hov_reverse(short tapeId, size_t dim_y, size_t num_weights,
                   double **dpp_U, size_t dim_x, size_t degree,
                   double ***dppp_Z, short **spp_nz) {
  printf("WARNING: Checkpointing algorithm not "
         "implemented for the hov_reverse mode!\n");
  return 0;
}

/****************************************************************************/
/*                              functions for handling the checkpoint stack */
/****************************************************************************/

void ValueTape::cp_clearStack() {
  StackElement se;

  while (!cp_stack_.empty()) {
    se = cp_stack_.top();
    cp_stack_.pop();
    delete[] se[0];
    delete[] se;
  }
}

void ValueTape::cp_takeshot(CpInfos *cpInfos) {

  StackElement se = new double *[2];
  cp_stack_.push(se);
  se[0] = new double[cpInfos->dim];

  for (size_t i = 0; i < cpInfos->dim; ++i)
    se[0][i] = cpInfos->dp_internal_for[i];

  if (cpInfos->saveNonAdoubles != nullptr)
    se[1] = static_cast<double *>(cpInfos->saveNonAdoubles());
  else
    se[1] = nullptr;
}

void ValueTape::cp_restore(CpInfos *cpInfos) {

  StackElement se = cp_stack_.top();
  for (size_t i = 0; i < cpInfos->dim; ++i)
    cpInfos->dp_internal_for[i] = se[0][i];

  if (se[1] != nullptr)
    cpInfos->restoreNonAdoubles(static_cast<void *>(se[1]));
}

void ValueTape::cp_release(CpInfos *cpInfos) {
  if (!cp_stack_.empty()) {
    StackElement se = cp_stack_.top();
    cp_stack_.pop();
    delete[] se[0];

    if (se[1] != nullptr)
      delete[] se[1];

    delete[] se;
  }
}

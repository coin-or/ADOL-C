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

#include <cstring>

/* forward function declarations */
void CpInfos::taping() {
  trace_on(cp_tape_id, 1);
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
void CpInfos::revolveError() {
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

void CpInfos::revolve_for() {
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
      takeshot();
      currentCP = capo;
      break;

    case revolve_advance:
      for (int i = 0; i < capo - currentCP; ++i) {
        function_double(dim, dp_internal_for);
      }
      break;

    case revolve_firsturn:
      taping();
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

/* we do not really have an ext. diff. function that we want to be called */
int dummy(short, size_t, double *, size_t, double *) { return 0; }

/* register one time step function (uses buffer template) */
void CP_Context::reg_timestep_fct(short tapeId, short cp_tape_id,
                                  ADOLC_TimeStepFuncion timeStepFunction) {

  ValueTape &tape = findTape(tapeId);
  cpInfos = tape.cp_append();
  cpInfos->function = timeStepFunction;
  cpInfos->tapeId = tapeId;
  cpInfos->cp_tape_id = cp_tape_id;
}

void check_input(short tapeId, CpInfos *cpInfos) {
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
int cp_zos_forward(size_t cpIndex, short tapeId, size_t, double *, size_t,
                   double *) {
  ValueTape &tape = findTape(tapeId);

  // get checkpointing information
  CpInfos *cpInfos = tape.get_cp_fct(cpIndex);
  if (!cpInfos)
    ADOLCError::fail(ADOLCError::ErrorType::CP_NO_SUCH_IDX, CURRENT_LOCATION,
                     ADOLCError::FailInfo{.info2 = cpIndex});
  const ext_diff_fct *edfct = tape.ext_diff_getElement(cpInfos->extDiffIndex);

  // prepare arguments
  cpInfos->dp_internal_for = new double[cpInfos->dim];

  size_t arg = edfct->firstIndLocation;
  for (int i = 0; i < cpInfos->dim; ++i) {
    cpInfos->dp_internal_for[i] = tape.dp_T0()[arg];
    ++arg;
  }

  cpInfos->revolve_for();

  // write back
  arg = edfct->firstDepLocation; // keep input
  for (int i = 0; i < cpInfos->dim; ++i) {
    tape.write_scaylor(tape.dp_T0()[arg]);
    tape.dp_T0()[arg] = cpInfos->dp_internal_for[i];
    ++arg;
  }
  delete[] cpInfos->dp_internal_for;
  cpInfos->dp_internal_for = nullptr;
  return 0;
}

int cp_fos_forward(size_t, short, size_t, double *, double *, size_t, double *,
                   double *) {
  printf("WARNING: Checkpointing algorithm not "
         "implemented for the fos_forward mode!\n");
  return 0;
}

int cp_fov_forward(size_t, short, size_t, double *, size_t, double **, size_t,
                   double *, double **) {
  printf("WARNING: Checkpointing algorithm not "
         "implemented for the fov_forward mode!\n");
  return 0;
}

int cp_hos_forward(size_t, short, size_t, double *, size_t, double **, size_t,
                   double *, double **) {
  printf("WARNING: Checkpointing algorithm not "
         "implemented for the hos_forward mode!\n");
  return 0;
}

int cp_hov_forward(size_t, short, size_t, double *, size_t, size_t, double ***,
                   size_t, double *, double ***) {
  printf("WARNING: Checkpointing algorithm not "
         "implemented for the hov_forward mode!\n");
  return 0;
}

int cp_fos_reverse(size_t cpIndex, short tapeId, size_t, double *, size_t,
                   double *, double *, double *) {

  ValueTape &tape = findTape(tapeId);

  CpInfos *cpInfos = tape.get_cp_fct(cpIndex);
  const ext_diff_fct *edfct = tape.ext_diff_getElement(cpInfos->extDiffIndex);

  cpInfos->dp_internal_for = new double[cpInfos->dim];
  cpInfos->dp_internal_rev = new double[cpInfos->dim];

  size_t arg = edfct->firstDepLocation;
  for (int i = 0; i < cpInfos->dim; ++i) {
    cpInfos->dp_internal_rev[i] = tape.rp_A()[arg];
    ++arg;
  }
  // update taylor buffer
  for (int i = 0; i < cpInfos->dim; ++i) {
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
      cpInfos->takeshot();
      cpInfos->currentCP = cpInfos->capo;
      break;

    case revolve_advance:
      for (int i = 0; i < cpInfos->capo - cpInfos->currentCP; ++i)
        cpInfos->function_double(cpInfos->dim, cpInfos->dp_internal_for);
      break;

    case revolve_youturn:
      if (cpInfos->retaping != 0)
        cpInfos->taping(); // retaping forced
      else {
        // one forward step with keep and retaping if necessary
        if (zos_forward(cpInfos->cp_tape_id, cpInfos->dim, cpInfos->dim, 1,
                        cpInfos->dp_internal_for, cpInfos->dp_internal_for) < 0)
          cpInfos->taping();
      }
      // one reverse step
      fos_reverse(cpInfos->cp_tape_id, cpInfos->dim, cpInfos->dim,
                  cpInfos->dp_internal_rev, cpInfos->dp_internal_rev);
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
  size_t start = edfct->firstDepLocation;
  std::copy(cpInfos->dp_internal_rev, cpInfos->dp_internal_rev + cpInfos->dim,
            tape.rp_A() + start);

  // clean up
  delete[] cpInfos->dp_internal_for;
  cpInfos->dp_internal_for = nullptr;
  delete[] cpInfos->dp_internal_rev;
  cpInfos->dp_internal_rev = nullptr;

  return 0;
}

int cp_fov_reverse(size_t cpIndex, short tapeId, size_t, size_t, double **,
                   size_t, double **, double *, double *) {

  ValueTape &tape = findTape(tapeId);

  CpInfos *cpInfos = tape.get_cp_fct(cpIndex);
  const ext_diff_fct *edfct = tape.ext_diff_getElement(cpInfos->extDiffIndex);

  const int numDirs = tape.numDirs_rev();
  cpInfos->dp_internal_for = new double[cpInfos->dim];
  cpInfos->dpp_internal_rev = myalloc2(numDirs, cpInfos->dim);

  double **rpp_A = tape.rpp_A();
  size_t start = edfct->firstDepLocation;

  for (size_t i = start; i < cpInfos->dim + start; ++i) {
    for (int j = 0; j < numDirs; ++j) {
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
      cpInfos->takeshot();
      cpInfos->currentCP = cpInfos->capo;
      break;

    case revolve_advance:
      for (int i = 0; i < cpInfos->capo - cpInfos->currentCP; ++i)
        cpInfos->function_double(cpInfos->dim, cpInfos->dp_internal_for);
      break;

    case revolve_youturn:
      if (cpInfos->retaping != 0)
        cpInfos->taping(); // retaping forced
      else {
        // one forward step with keep and retaping if necessary
        if (zos_forward(cpInfos->cp_tape_id, cpInfos->dim, cpInfos->dim, 1,
                        cpInfos->dp_internal_for, cpInfos->dp_internal_for) < 0)
          cpInfos->taping();
      }
      // one reverse step
      fov_reverse(cpInfos->cp_tape_id, cpInfos->dim, cpInfos->dim, numDirs,
                  cpInfos->dpp_internal_rev, cpInfos->dpp_internal_rev);
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
  start = edfct->firstDepLocation;
  for (size_t i = start; i < cpInfos->dim + start; ++i) {
    for (int j = 0; j < numDirs; ++j) {
      rpp_A[i][j] = cpInfos->dpp_internal_rev[j][i];
    }
  }

  // clean up
  delete[] cpInfos->dp_internal_for;
  cpInfos->dp_internal_for = nullptr;
  myfree2(cpInfos->dpp_internal_rev);
  cpInfos->dpp_internal_rev = nullptr;
  return 0;
}

int cp_hos_reverse(size_t, short, size_t, double *, size_t, size_t, double **) {
  printf("WARNING: Checkpointing algorithm not "
         "implemented for the hos_reverse mode!\n");
  return 0;
}

int cp_hov_reverse(size_t, short, size_t, size_t, double **, size_t, size_t,
                   double ***, short **) {
  printf("WARNING: Checkpointing algorithm not "
         "implemented for the hov_reverse mode!\n");
  return 0;
}

/****************************************************************************/
/*                              functions for handling the checkpoint stack */
/****************************************************************************/

void CpInfos::clearStack() {
  StackElement shot;
  while (!cp_stack.empty()) {
    shot = cp_stack.top();
    cp_stack.pop();
    delete[] shot[0];
    delete[] shot[1];
  }
}

void CpInfos::takeshot() {
  StackElement shot;
  shot[0] = new double[dim];
  for (int i = 0; i < dim; ++i)
    shot[0][i] = dp_internal_for[i];
  if (saveNonAdoubles != nullptr)
    shot[1] = static_cast<double *>(saveNonAdoubles());
  else
    shot[1] = nullptr;
  cp_stack.push(shot);
}

void CpInfos::restore() {
  using ADOLCError::fail;
  using ADOLCError::FailInfo;
  using ADOLCError::ErrorType::CP_EMPTY_STACK;
  if (cp_stack.empty())
    fail(CP_EMPTY_STACK, CURRENT_LOCATION, FailInfo{.info2 = index});

  StackElement shot = cp_stack.top();
  for (int i = 0; i < dim; ++i)
    dp_internal_for[i] = shot[0][i];

  if (shot[1] != nullptr)
    restoreNonAdoubles(static_cast<void *>(shot[1]));
}

void CpInfos::release() {
  if (!cp_stack.empty()) {
    StackElement shot = cp_stack.top();
    cp_stack.pop();
    delete[] shot[0];

    if (shot[1] != nullptr)
      delete[] shot[1];
  }
}

/* initialize the information for the external function in a way that our
 * checkpointing functions are called */
void init_edf(ext_diff_fct *edf) {
  edf->function = dummy;

  // ZOS FORWARD
  edf->zos_forward = [edf](short tapeId, size_t n1, double *x, size_t n2,
                           double *y) {
    return cp_zos_forward(edf->cp_index, tapeId, n1, x, n2, y);
  };

  // FOS FORWARD
  edf->fos_forward = [edf](short tapeId, size_t n, double *x, double *xp,
                           size_t m, double *y, double *yp) {
    return cp_fos_forward(edf->cp_index, tapeId, n, x, xp, m, y, yp);
  };

  // FOV FORWARD
  edf->fov_forward = [edf](short tapeId, size_t n, double *x, size_t p,
                           double **xp, size_t m, double *y, double **yp) {
    return cp_fov_forward(edf->cp_index, tapeId, n, x, p, xp, m, y, yp);
  };

  // HOS FORWARD
  edf->hos_forward = [edf](short tapeId, size_t n, double *x, size_t d,
                           double **xp, size_t m, double *y, double **yp) {
    return cp_hos_forward(edf->cp_index, tapeId, n, x, d, xp, m, y, yp);
  };

  // HOV FORWARD
  edf->hov_forward = [edf](short tapeId, size_t n, double *x, size_t d,
                           size_t p, double ***xp, size_t m, double *y,
                           double ***yp) {
    return cp_hov_forward(edf->cp_index, tapeId, n, x, d, p, xp, m, y, yp);
  };

  // FOS REVERSE
  edf->fos_reverse = [edf](short tapeId, size_t n, double *x, size_t m,
                           double *y, double *u, double *z) {
    return cp_fos_reverse(edf->cp_index, tapeId, n, x, m, y, u, z);
  };

  // FOV REVERSE
  edf->fov_reverse = [edf](short tapeId, size_t n, size_t p, double **x,
                           size_t m, double **y, double *u, double *z) {
    return cp_fov_reverse(edf->cp_index, tapeId, n, p, x, m, y, u, z);
  };

  // HOS REVERSE
  edf->hos_reverse = [edf](short tapeId, size_t n, double *x, size_t d,
                           size_t m, double **y) {
    return cp_hos_reverse(edf->cp_index, tapeId, n, x, d, m, y);
  };

  // HOV REVERSE
  edf->hov_reverse = [edf](short tapeId, size_t n, size_t d, double **x,
                           size_t m, size_t p, double ***y, short **nz) {
    return cp_hov_reverse(edf->cp_index, tapeId, n, d, x, m, p, y, nz);
  };
}
int CP_Context::checkpointing(short tapeId) {
  // throws if input is invalid
  check_input(tapeId, cpInfos);

  // register extern function
  ext_diff_fct *edf = reg_ext_fct(cpInfos->tapeId, cpInfos->cp_tape_id, dummy);
  init_edf(edf);
  edf->cp_index = cpInfos->index;
  cpInfos->extDiffIndex = edf->index;

  ValueTape &tape = findTape(cpInfos->tapeId);
  // but we do not call it
  // we use direct taping to avoid unnecessary argument copying

  tape.put_op(ext_diff);
  tape.put_loc(edf->index);
  // Keep n/m at zero so generic extern-fct interpreter paths skip their
  // argument/taylor handling; checkpointing manages this internally.
  tape.put_loc(0);
  tape.put_loc(0);
  edf->firstIndLocation = cpInfos->adp_x[0].loc();
  edf->firstDepLocation = cpInfos->adp_y[0].loc();

  std::vector<double> vals(tape.store(), tape.store() + tape.storeSize());

  cpInfos->dp_internal_for = new double[cpInfos->dim];

  // initialize internal arguments
  for (int i = 0; i < cpInfos->dim; ++i)
    cpInfos->dp_internal_for[i] = cpInfos->adp_x[i].value();

  if (tape.keepTaylors()) {
    // perform all time steps, tape the last, take checkpoints
    cpInfos->revolve_for();
  } else
    // perform all time steps without taping
    for (int i = 0; i < cpInfos->steps; ++i)
      cpInfos->function_double(cpInfos->dim, cpInfos->dp_internal_for);

  std::copy(vals.begin(), vals.end(), tape.store());

  // update taylor stack; same structure as in adouble.cpp +
  // correction in taping.cpp
  tape.add_numTays_Tape(cpInfos->dim);
  if (tape.keepTaylors())
    for (int i = 0; i < cpInfos->dim; ++i)
      tape.write_scaylor(cpInfos->adp_y[i].value());

  // save results
  for (int i = 0; i < cpInfos->dim; ++i) {
    cpInfos->adp_y[i].value(cpInfos->dp_internal_for[i]);
  }

  delete[] cpInfos->dp_internal_for;
  cpInfos->dp_internal_for = nullptr;
  return 0;
}

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
#include <adolc/checkpointing.h>
#include <adolc/checkpointing_p.h>
#include <adolc/externfcts.h>
#include <adolc/interfaces.h>
#include <adolc/oplate.h>
#include <adolc/revolve.h>
#include <adolc/taping_p.h>

#include <cstring>

#include <stack>
using namespace std;

ADOLC_BUFFER_TYPE ADOLC_EXT_DIFF_FCTS_BUFFER_DECL;

/* field of pointers to the value fields of a checkpoint */
stack<StackElement> ADOLC_CHECKPOINTS_STACK_DECL;

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
void cp_takeshot(CpInfos *cpInfos);
void cp_restore(CpInfos *cpInfos);
void cp_release(CpInfos *cpInfos);
void cp_taping(CpInfos *cpInfos);
void revolve_for(CpInfos *cpInfos);
void revolveError(CpInfos *cpInfos);

/* we do not really have an ext. diff. function that we want to be called */
int dummy(size_t dim_x, double *x, size_t dim_y, double *y) { return 0; }

/* register one time step function (uses buffer template) */
CpInfos *reg_timestep_fct(ADOLC_TimeStepFuncion timeStepFunction) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  CpInfos *theCpInfos = ADOLC_EXT_DIFF_FCTS_BUFFER.append();
  theCpInfos->function = timeStepFunction;
  return theCpInfos;
}

/* This is the main checkpointing function the user calls within the taping
 * process. It performs n time steps with or without taping and registers an
 * external dummy function which calls the actual checkpointing workhorses
 * from within the used drivers. */
int checkpointing(CpInfos *cpInfos) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  // knockout
  if (cpInfos == nullptr)
    fail(ADOLC_CHECKPOINTING_CPINFOS_NULLPOINTER);
  if (cpInfos->function == nullptr)
    fail(ADOLC_CHECKPOINTING_NULLPOINTER_FUNCTION);
  if (cpInfos->function_double == nullptr)
    fail(ADOLC_CHECKPOINTING_NULLPOINTER_FUNCTION_DOUBLE);
  if (cpInfos->adp_x == nullptr)
    fail(ADOLC_CHECKPOINTING_NULLPOINTER_ARGUMENT);

  // register extern function
  ext_diff_fct *edf = reg_ext_fct(dummy);
  init_edf(edf);

  size_t oldTraceFlag = 0;
  // but we do not call it
  // we use direct taping to avoid unnecessary argument copying
  if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
    put_op(ext_diff);
    ADOLC_PUT_LOCINT(edf->index);
    ADOLC_PUT_LOCINT(0);
    ADOLC_PUT_LOCINT(0);
    ADOLC_PUT_LOCINT(cpInfos->adp_x[0].loc());
    ADOLC_PUT_LOCINT(cpInfos->adp_y[0].loc());
    // this CpInfos id has to be read by the actual checkpointing
    // functions
    ADOLC_PUT_LOCINT(cpInfos->index);

    oldTraceFlag = ADOLC_CURRENT_TAPE_INFOS.traceFlag;
    ADOLC_CURRENT_TAPE_INFOS.traceFlag = 0;
  }

  const size_t numVals = ADOLC_GLOBAL_TAPE_VARS.storeSize;
  double *vals = new double[numVals];
  memcpy(vals, ADOLC_GLOBAL_TAPE_VARS.store, numVals * sizeof(double));

  cpInfos->dp_internal_for = new double[cpInfos->dim];

  // initialize internal arguments
  for (size_t i = 0; i < cpInfos->dim; ++i)
    cpInfos->dp_internal_for[i] = cpInfos->adp_x[i].value();

  if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors != 0)
    // perform all time steps, tape the last, take checkpoints
    revolve_for(cpInfos);
  else
    // perform all time steps without taping
    for (size_t i = 0; i < cpInfos->steps; ++i)
      cpInfos->function_double(cpInfos->dim, cpInfos->dp_internal_for);

  memcpy(ADOLC_GLOBAL_TAPE_VARS.store, vals, numVals * sizeof(double));
  delete[] vals;

  // update taylor stack; same structure as in adouble.cpp +
  // correction in taping.cpp
  if (oldTraceFlag != 0) {
    ADOLC_CURRENT_TAPE_INFOS.numTays_Tape += cpInfos->dim;
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors != 0)
      for (size_t i = 0; i < cpInfos->dim; ++i)
        ADOLC_WRITE_SCAYLOR(cpInfos->adp_y[i].value());
  }
  // save results
  for (size_t i = 0; i < cpInfos->dim; ++i) {
    cpInfos->adp_y[i].value(cpInfos->dp_internal_for[i]);
  }

  delete[] cpInfos->dp_internal_for;
  cpInfos->dp_internal_for = nullptr;

  // normal taping again
  ADOLC_CURRENT_TAPE_INFOS.traceFlag = oldTraceFlag;

  return 0;
}

/* - reinit external function buffer and checkpointing buffer
 * - necessary when using tape within a different program */
void reinit_checkpointing() {}

CpInfos *get_cp_fct(int index) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  return ADOLC_EXT_DIFF_FCTS_BUFFER.getElement(index);
}

/* initialize the CpInfos variable (function and index are set within
 * the template code */
void init_CpInfos(CpInfos *cpInfos) {
  char *ptr = nullptr;

  ptr = reinterpret_cast<char *>(cpInfos);
  for (size_t i = 0; i < sizeof(CpInfos); ++i)
    ptr[i] = 0;
  cpInfos->tapeNumber = -1;
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
int cp_zos_forward(size_t dim_x, double *dp_x, size_t dim_y, double *dp_y) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  // taping off
  const size_t oldTraceFlag = ADOLC_CURRENT_TAPE_INFOS.traceFlag;
  ADOLC_CURRENT_TAPE_INFOS.traceFlag = 0;

  // get checkpointing information
  CpInfos *cpInfos = get_cp_fct(ADOLC_CURRENT_TAPE_INFOS.cpIndex);
  double *T0 = ADOLC_CURRENT_TAPE_INFOS.dp_T0;

  // note the mode
  cpInfos->modeForward = ADOLC_ZOS_FORWARD;
  cpInfos->modeReverse = ADOLC_NO_MODE;

  // prepare arguments
  cpInfos->dp_internal_for = new double[cpInfos->dim];

  size_t arg = ADOLC_CURRENT_TAPE_INFOS.lowestXLoc_for;
  for (size_t i = 0; i < cpInfos->dim; ++i) {
    cpInfos->dp_internal_for[i] = T0[arg];
    ++arg;
  }

  revolve_for(cpInfos);

  // write back
  arg = ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_for; // keep input
  for (size_t i = 0; i < cpInfos->dim; ++i) {
    ADOLC_WRITE_SCAYLOR(T0[arg]);
    T0[arg] = cpInfos->dp_internal_for[i];
    ++arg;
  }
  delete[] cpInfos->dp_internal_for;
  cpInfos->dp_internal_for = nullptr;

  // taping "on"
  ADOLC_CURRENT_TAPE_INFOS.traceFlag = oldTraceFlag;

  return 0;
}

void revolve_for(CpInfos *cpInfos) {
  /* init revolve */
  cpInfos->check = -1;
  cpInfos->capo = 0;
  cpInfos->info = 0;
  cpInfos->fine = cpInfos->steps;

  /* execute all time steps */
  enum revolve_action whattodo;
  do {
    whattodo = revolve(&cpInfos->check, &cpInfos->capo, &cpInfos->fine,
                       cpInfos->checkpoints, &cpInfos->info);
    switch (whattodo) {
    case revolve_takeshot:
      cp_takeshot(cpInfos);
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
      fail(ADOLC_CHECKPOINTING_UNEXPECTED_REVOLVE_ACTION);
    }
  } while (whattodo == revolve_takeshot || whattodo == revolve_advance);
}

int cp_fos_forward(size_t dim_x, double *dp_x, double *dp_X, size_t dim_y,
                   double *dp_y, double *dp_Y) {
  printf("WARNING: Checkpointing algorithm not "
         "implemented for the fos_forward mode!\n");
  return 0;
}

int cp_fov_forward(size_t dim_X, double *dp_x, size_t num_dirs, double **dpp_X,
                   size_t dim_y, double *dp_y, double **dpp_Y) {
  printf("WARNING: Checkpointing algorithm not "
         "implemented for the fov_forward mode!\n");
  return 0;
}

int cp_hos_forward(size_t dim_x, double *dp_x, size_t degree, double **dpp_X,
                   size_t dim_y, double *dp_y, double **dpp_Y) {
  printf("WARNING: Checkpointing algorithm not "
         "implemented for the hos_forward mode!\n");
  return 0;
}

int cp_hov_forward(size_t dim_x, double *dp_x, size_t degree, size_t num_dirs,
                   double ***dppp_X, size_t dim_y, double *dp_y,
                   double ***dppp_Y) {
  printf("WARNING: Checkpointing algorithm not "
         "implemented for the hov_forward mode!\n");
  return 0;
}

int cp_fos_reverse(size_t dim_y, double *dp_U, size_t dim_x, double *dp_Z,
                   double *dp_x, double *dp_y) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  revreal *A = ADOLC_CURRENT_TAPE_INFOS.rp_A;
  CpInfos *cpInfos = get_cp_fct(ADOLC_CURRENT_TAPE_INFOS.cpIndex);

  // note the mode
  cpInfos->modeReverse = ADOLC_FOS_REVERSE;

  cpInfos->dp_internal_for = new double[cpInfos->dim];
  cpInfos->dp_internal_rev = new double[cpInfos->dim];

  // taping "off"
  const size_t oldTraceFlag = ADOLC_CURRENT_TAPE_INFOS.traceFlag;
  ADOLC_CURRENT_TAPE_INFOS.traceFlag = 0;

  size_t arg = ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_rev;
  for (size_t i = 0; i < cpInfos->dim; ++i) {
    cpInfos->dp_internal_rev[i] = A[arg];
    ++arg;
  }
  // update taylor buffer
  for (size_t i = 0; i < cpInfos->dim; ++i) {
    --arg;
    ADOLC_GET_TAYLOR(arg);
  }
  // execute second part of revolve_firstturn left from forward sweep
  fos_reverse(cpInfos->tapeNumber, cpInfos->dim, cpInfos->dim,
              cpInfos->dp_internal_rev, cpInfos->dp_internal_rev);

  const char old_bsw = ADOLC_GLOBAL_TAPE_VARS.branchSwitchWarning;
  ADOLC_GLOBAL_TAPE_VARS.branchSwitchWarning = 0;
  // checkpointing
  enum revolve_action whattodo;
  do {
    whattodo = revolve(&cpInfos->check, &cpInfos->capo, &cpInfos->fine,
                       cpInfos->checkpoints, &cpInfos->info);
    switch (whattodo) {
    case revolve_terminate:
      break;

    case revolve_takeshot:
      cp_takeshot(cpInfos);
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
        if (zos_forward(cpInfos->tapeNumber, cpInfos->dim, cpInfos->dim, 1,
                        cpInfos->dp_internal_for, cpInfos->dp_internal_for) < 0)
          cp_taping(cpInfos);
      }
      // one reverse step
      fos_reverse(cpInfos->tapeNumber, cpInfos->dim, cpInfos->dim,
                  cpInfos->dp_internal_rev, cpInfos->dp_internal_rev);
      break;

    case revolve_restore:
      if (cpInfos->capo != cpInfos->currentCP)
        cp_release(cpInfos);
      cpInfos->currentCP = cpInfos->capo;
      cp_restore(cpInfos);
      break;

    case revolve_error:
      revolveError(cpInfos);
      break;

    default:
      fail(ADOLC_CHECKPOINTING_UNEXPECTED_REVOLVE_ACTION);
      break;
    }
  } while (whattodo != revolve_terminate && whattodo != revolve_error);
  cp_release(cpInfos); // release first checkpoint if written
  ADOLC_GLOBAL_TAPE_VARS.branchSwitchWarning = old_bsw;

  // save results
  arg = ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_rev;
  for (size_t i = 0; i < cpInfos->dim; ++i) {
    A[arg] = cpInfos->dp_internal_rev[i];
    ++arg;
  }

  // clean up
  delete[] cpInfos->dp_internal_for;
  cpInfos->dp_internal_for = nullptr;
  delete[] cpInfos->dp_internal_rev;
  cpInfos->dp_internal_rev = nullptr;

  // taping "on"
  ADOLC_CURRENT_TAPE_INFOS.traceFlag = oldTraceFlag;

  return 0;
}

int cp_fov_reverse(size_t dim_y, size_t num_weights, double **dpp_U,
                   size_t dim_x, double **dpp_Z, double * /*unused*/,
                   double * /*unused*/) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  revreal **A = ADOLC_CURRENT_TAPE_INFOS.rpp_A;
  CpInfos *cpInfos = get_cp_fct(ADOLC_CURRENT_TAPE_INFOS.cpIndex);

  // note the mode
  cpInfos->modeReverse = ADOLC_FOV_REVERSE;

  const size_t numDirs = ADOLC_CURRENT_TAPE_INFOS.numDirs_rev;
  cpInfos->dp_internal_for = new double[cpInfos->dim];
  cpInfos->dpp_internal_rev = myalloc2(numDirs, cpInfos->dim);

  // taping "off"
  const size_t oldTraceFlag = ADOLC_CURRENT_TAPE_INFOS.traceFlag;
  ADOLC_CURRENT_TAPE_INFOS.traceFlag = 0;

  size_t arg = ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_rev;
  for (size_t i = 0; i < cpInfos->dim; ++i) {
    for (size_t j = 0; j < numDirs; ++j) {
      cpInfos->dpp_internal_rev[j][i] = A[arg][j];
    }
    ++arg;
  }

  // update taylor buffer
  for (size_t i = 0; i < cpInfos->dim; ++i) {
    --arg;
    ADOLC_GET_TAYLOR(arg);
  }

  // execute second part of revolve_firstturn left from forward sweep
  fov_reverse(cpInfos->tapeNumber, cpInfos->dim, cpInfos->dim, numDirs,
              cpInfos->dpp_internal_rev, cpInfos->dpp_internal_rev);

  const char old_bsw = ADOLC_GLOBAL_TAPE_VARS.branchSwitchWarning;
  ADOLC_GLOBAL_TAPE_VARS.branchSwitchWarning = 0;
  // checkpointing
  enum revolve_action whattodo;
  do {
    whattodo = revolve(&cpInfos->check, &cpInfos->capo, &cpInfos->fine,
                       cpInfos->checkpoints, &cpInfos->info);
    switch (whattodo) {
    case revolve_terminate:
      break;

    case revolve_takeshot:
      cp_takeshot(cpInfos);
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
        if (zos_forward(cpInfos->tapeNumber, cpInfos->dim, cpInfos->dim, 1,
                        cpInfos->dp_internal_for, cpInfos->dp_internal_for) < 0)
          cp_taping(cpInfos);
      }
      // one reverse step
      fov_reverse(cpInfos->tapeNumber, cpInfos->dim, cpInfos->dim, numDirs,
                  cpInfos->dpp_internal_rev, cpInfos->dpp_internal_rev);
      break;

    case revolve_restore:
      if (cpInfos->capo != cpInfos->currentCP)
        cp_release(cpInfos);
      cpInfos->currentCP = cpInfos->capo;
      cp_restore(cpInfos);
      break;

    case revolve_error:
      revolveError(cpInfos);
      break;

    default:
      fail(ADOLC_CHECKPOINTING_UNEXPECTED_REVOLVE_ACTION);
      break;
    }
  } while (whattodo != revolve_terminate && whattodo != revolve_error);

  // release first checkpoint if written
  cp_release(cpInfos);
  ADOLC_GLOBAL_TAPE_VARS.branchSwitchWarning = old_bsw;

  // save results
  arg = ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_rev;
  for (size_t i = 0; i < cpInfos->dim; ++i) {
    for (size_t j = 0; j < numDirs; ++j) {
      A[arg][j] = cpInfos->dpp_internal_rev[j][i];
    }
    ++arg;
  }

  // clean up
  delete[] cpInfos->dp_internal_for;
  cpInfos->dp_internal_for = nullptr;
  myfree2(cpInfos->dpp_internal_rev);
  cpInfos->dpp_internal_rev = nullptr;

  // taping "on"
  ADOLC_CURRENT_TAPE_INFOS.traceFlag = oldTraceFlag;

  return 0;
}

int cp_hos_reverse(size_t dim_y, double *dp_U, size_t dim_x, size_t degree,
                   double **dpp_Z) {
  printf("WARNING: Checkpointing algorithm not "
         "implemented for the hos_reverse mode!\n");
  return 0;
}

int cp_hov_reverse(size_t dim_y, size_t num_weights, double **dpp_U,
                   size_t dim_x, size_t degree, double ***dppp_Z,
                   short **spp_nz) {
  printf("WARNING: Checkpointing algorithm not "
         "implemented for the hov_reverse mode!\n");
  return 0;
}

/****************************************************************************/
/*                              functions for handling the checkpoint stack */
/****************************************************************************/

void cp_clearStack() {
  StackElement se;
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  while (!ADOLC_CHECKPOINTS_STACK.empty()) {
    se = ADOLC_CHECKPOINTS_STACK.top();
    ADOLC_CHECKPOINTS_STACK.pop();
    delete[] se[0];
    delete[] se;
  }
}

void cp_takeshot(CpInfos *cpInfos) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  StackElement se = new double *[2];
  ADOLC_CHECKPOINTS_STACK.push(se);
  se[0] = new double[cpInfos->dim];

  for (size_t i = 0; i < cpInfos->dim; ++i)
    se[0][i] = cpInfos->dp_internal_for[i];

  if (cpInfos->saveNonAdoubles != nullptr)
    se[1] = static_cast<double *>(cpInfos->saveNonAdoubles());
  else
    se[1] = nullptr;
}

void cp_restore(CpInfos *cpInfos) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  StackElement se = ADOLC_CHECKPOINTS_STACK.top();
  for (size_t i = 0; i < cpInfos->dim; ++i)
    cpInfos->dp_internal_for[i] = se[0][i];

  if (se[1] != nullptr)
    cpInfos->restoreNonAdoubles(static_cast<void *>(se[1]));
}

void cp_release(CpInfos *cpInfos) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;

  if (!ADOLC_CHECKPOINTS_STACK.empty()) {
    StackElement se = ADOLC_CHECKPOINTS_STACK.top();
    ADOLC_CHECKPOINTS_STACK.pop();
    delete[] se[0];

    if (se[1] != nullptr)
      delete[] se[1];

    delete[] se;
  }
}

void cp_taping(CpInfos *cpInfos) {
  adouble *tapingAdoubles = new adouble[cpInfos->dim];

  trace_on(cpInfos->tapeNumber, 1);

  for (size_t i = 0; i < cpInfos->dim; ++i)
    tapingAdoubles[i] <<= cpInfos->dp_internal_for[i];

  cpInfos->function(cpInfos->dim, tapingAdoubles);

  for (size_t i = 0; i < cpInfos->dim; ++i)
    tapingAdoubles[i] >>= cpInfos->dp_internal_for[i];

  trace_off();

  delete[] tapingAdoubles;
}

/****************************************************************************/
/*                                                   revolve error function */
/****************************************************************************/
void revolveError(CpInfos *cpInfos) {
  switch (cpInfos->info) {
  case 10:
    printf("   Number of checkpoints stored exceeds "
           "checkup!\n   Increase constant 'checkup' "
           "and recompile!\n");
    break;
  case 11:
    printf("   Number of checkpoints stored = %d exceeds "
           "snaps = %zu!\n   Ensure 'snaps' > 0 and increase "
           "initial 'fine'!\n",
           cpInfos->check + 1, cpInfos->checkpoints);
    break;
  case 12:
    printf("   Error occurred in numforw!\n");
    break;
  case 13:
    printf("   Enhancement of 'fine', 'snaps' checkpoints "
           "stored!\n   Increase 'snaps'!\n");
    break;
  case 14:
    printf("   Number of snaps exceeds checkup!\n   Increase "
           "constant 'checkup' and recompile!\n");
    break;
  case 15:
    printf("   Number of reps exceeds repsup!\n   Increase "
           "constant 'repsup' and recompile!\n");
    break;
  }
  fail(ADOLC_CHECKPOINTING_REVOLVE_IRREGULAR_TERMINATED);
}

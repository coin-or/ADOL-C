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

#if !defined(ADOLC_CHECKPOINTING_H)
#define ADOLC_CHECKPOINTING_H 1

#include <adolc/adtb_types.h>
#include <adolc/internal/common.h>

#if defined(__cplusplus)
/****************************************************************************/
/*                                                          This is all C++ */

typedef int (*ADOLC_TimeStepFuncion)(size_t dim_x, adouble *x);
typedef int (*ADOLC_TimeStepFuncion_double)(size_t dim_x, double *x);
typedef void *(*ADOLC_saveFct)();
typedef void (*ADOLC_restoreFct)(void *);

typedef struct CpInfos {
  ADOLC_TimeStepFuncion function;
  ADOLC_TimeStepFuncion_double function_double;
  ADOLC_saveFct saveNonAdoubles;
  ADOLC_restoreFct restoreNonAdoubles;
  size_t steps;
  size_t checkpoints;
  size_t tapeNumber; /* tape number to be used for checkpointing */
  int retaping;      /* != 0 forces retaping before every reverse step */

  size_t dim;     /* number of variables in input and output (n=m) */
  adouble *adp_x; /* input of the first step */
  adouble *adp_y; /* output of the last step; will be set by ADOLC */

  /* these are internal checkpointing variables => do not use */
  int check;
  int capo;
  int fine;
  int info;
  int currentCP;
  double *dp_internal_for;
  double *dp_internal_rev;
  double **dpp_internal_rev;
  size_t index; /* please do not change */
  char modeForward;
  char modeReverse;
  char *allmem; /* this is dummy to get externfcts and checkpointing both use
                   buffer_temp without a problem */
} CpInfos;

ADOLC_DLL_EXPORT
CpInfos *reg_timestep_fct(ADOLC_TimeStepFuncion timeStepFunction);

ADOLC_DLL_EXPORT int checkpointing(CpInfos *cpInfos);

/* if tape with one program and use the tapes with another program call this
 * function within the latter                                               */
ADOLC_DLL_EXPORT void reinit_checkpointing();

class CP_Context {
public:
  inline CP_Context(ADOLC_TimeStepFuncion tsf);
  inline ~CP_Context() {}

  inline void setDoubleFct(ADOLC_TimeStepFuncion_double tsf);
  inline void setSaveFct(ADOLC_saveFct sf);
  inline void setRestoreFct(ADOLC_restoreFct rf);
  inline void setNumberOfSteps(size_t number);
  inline void setNumberOfCheckpoints(size_t number);
  inline void setTapeNumber(size_t tapeNumber);
  inline void setDimensionXY(size_t dim);
  inline void setInput(adouble *x);
  inline void setOutput(adouble *y);
  inline void setAlwaysRetaping(bool state);

  inline int checkpointing();

private:
  inline CP_Context() {}

  CpInfos *cpInfos;
};

CP_Context::CP_Context(ADOLC_TimeStepFuncion tsf) {
  cpInfos = reg_timestep_fct(tsf);
}

void CP_Context::setDoubleFct(ADOLC_TimeStepFuncion_double tsf) {
  cpInfos->function_double = tsf;
}

void CP_Context::setSaveFct(ADOLC_saveFct sf) { cpInfos->saveNonAdoubles = sf; }

void CP_Context::setRestoreFct(ADOLC_restoreFct rf) {
  cpInfos->restoreNonAdoubles = rf;
}

void CP_Context::setNumberOfSteps(size_t number) { cpInfos->steps = number; }

void CP_Context::setNumberOfCheckpoints(size_t number) {
  cpInfos->checkpoints = number;
}

void CP_Context::setTapeNumber(size_t tapeNumber) {
  cpInfos->tapeNumber = tapeNumber;
}

void CP_Context::setDimensionXY(size_t dim) { cpInfos->dim = dim; }

void CP_Context::setInput(adouble *x) { cpInfos->adp_x = x; }

void CP_Context::setOutput(adouble *y) { cpInfos->adp_y = y; }

void CP_Context::setAlwaysRetaping(bool state) {
  if (state)
    cpInfos->retaping = 1;
  else
    cpInfos->retaping = 0;
}

int CP_Context::checkpointing() { return ::checkpointing(cpInfos); }

#endif /* CPLUSPLUS */

/****************************************************************************/
#endif /* ADOLC_CHECKPOINTING_H */

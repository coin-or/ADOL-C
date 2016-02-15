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

#include "oplate.h"
#include "taping_p.h"
#include <adolc/adalloc.h>
#include <adolc/externfcts.h>
#include <adolc/interfaces.h>
#include <adolc/checkpointing.h>
#include "checkpointing_p.h"
#include <adolc/revolve.h>

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
int dummy(int n, double *x, int m, double *y) {
    return 0;
}

/* register one time step function (uses buffer template) */
CpInfos *reg_timestep_fct(ADOLC_TimeStepFuncion timeStepFunction) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    CpInfos* theCpInfos=ADOLC_EXT_DIFF_FCTS_BUFFER.append();
    theCpInfos->function=timeStepFunction;
    return theCpInfos;
}

/* This is the main checkpointing function the user calls within the taping
 * process. It performs n time steps with or without taping and registers an
 * external dummy function which calls the actual checkpointing workhorses
 * from within the used drivers. */
int checkpointing (CpInfos *cpInfos) {
    int i;
    ext_diff_fct *edf;
    int oldTraceFlag;
    locint numVals;
    double *vals;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    // knockout
    if (cpInfos==NULL)
        fail(ADOLC_CHECKPOINTING_CPINFOS_NULLPOINTER);
    if (cpInfos->function==NULL)
        fail(ADOLC_CHECKPOINTING_NULLPOINTER_FUNCTION);
    if (cpInfos->function_double==NULL)
        fail(ADOLC_CHECKPOINTING_NULLPOINTER_FUNCTION_DOUBLE);
    if (cpInfos->adp_x==NULL)
        fail(ADOLC_CHECKPOINTING_NULLPOINTER_ARGUMENT);

    // register extern function
    edf=reg_ext_fct(dummy);
    init_edf(edf);

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

        oldTraceFlag=ADOLC_CURRENT_TAPE_INFOS.traceFlag;
        ADOLC_CURRENT_TAPE_INFOS.traceFlag=0;
    } else oldTraceFlag=0;

    numVals = ADOLC_GLOBAL_TAPE_VARS.storeSize;
    vals = new double[numVals];
    memcpy(vals, ADOLC_GLOBAL_TAPE_VARS.store,
            numVals * sizeof(double));

    cpInfos->dp_internal_for = new double[cpInfos->n];
    // initialize internal arguments
    for (i=0; i<cpInfos->n; ++i)
        cpInfos->dp_internal_for[i]=cpInfos->adp_x[i].getValue();
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors != 0)
        // perform all time steps, tape the last, take checkpoints
        revolve_for(cpInfos);
    else
        // perform all time steps without taping
        for (i=0; i<cpInfos->steps; ++i)
            cpInfos->function_double(cpInfos->n, cpInfos->dp_internal_for);

    memcpy(ADOLC_GLOBAL_TAPE_VARS.store, vals,
            numVals * sizeof(double));
    delete[] vals;

    // update taylor stack; same structure as in adouble.cpp +
    // correction in taping.c
    if (oldTraceFlag != 0) {
        ADOLC_CURRENT_TAPE_INFOS.numTays_Tape += cpInfos->n;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors != 0)
            for (i = 0; i < cpInfos->n; ++i)
                ADOLC_WRITE_SCAYLOR(cpInfos->adp_y[i].getValue());
    }
    // save results
    for (i=0; i<cpInfos->n; ++i) {
        cpInfos->adp_y[i].setValue(cpInfos->dp_internal_for[i]);
    }
    delete[] cpInfos->dp_internal_for;
    cpInfos->dp_internal_for = NULL;

    // normal taping again
    ADOLC_CURRENT_TAPE_INFOS.traceFlag=oldTraceFlag;

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
    char *ptr;

    ptr = (char *)cpInfos;
    for (unsigned int i = 0; i < sizeof(CpInfos); ++i) ptr[i] = 0;
    cpInfos->tapeNumber = -1;
}

/* initialize the information for the external function in a way that our
 * checkpointing functions are called */
void init_edf(ext_diff_fct *edf) {
    edf->function=dummy;
    edf->zos_forward=cp_zos_forward;
    edf->fos_forward=cp_fos_forward;
    edf->fov_forward=cp_fov_forward;
    edf->hos_forward=cp_hos_forward;
    edf->hov_forward=cp_hov_forward;
    edf->fos_reverse=cp_fos_reverse;
    edf->fov_reverse=cp_fov_reverse;
    edf->hos_reverse=cp_hos_reverse;
    edf->hov_reverse=cp_hov_reverse;
}

/****************************************************************************/
/* the following are the main checkpointing functions called by the         */
/* external differentiated function alogrithms                              */
/****************************************************************************/

/* special case: use double version where possible, no taping */
int cp_zos_forward (int n, double *dp_x, int m, double *dp_y) {
    CpInfos *cpInfos;
    double *T0;
    int i, oldTraceFlag;
    locint arg;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    // taping off
    oldTraceFlag=ADOLC_CURRENT_TAPE_INFOS.traceFlag;
    ADOLC_CURRENT_TAPE_INFOS.traceFlag=0;

    // get checkpointing information
    cpInfos=get_cp_fct(ADOLC_CURRENT_TAPE_INFOS.cpIndex);
    T0 = ADOLC_CURRENT_TAPE_INFOS.dp_T0;

    // note the mode
    cpInfos->modeForward = ADOLC_ZOS_FORWARD;
    cpInfos->modeReverse = ADOLC_NO_MODE;

    // prepare arguments
    cpInfos->dp_internal_for=new double[cpInfos->n];
    arg=ADOLC_CURRENT_TAPE_INFOS.lowestXLoc_for;
    for (i=0; i<cpInfos->n; ++i) {
        cpInfos->dp_internal_for[i]=T0[arg];
        ++arg;
    }

    revolve_for(cpInfos);

    // write back
    arg = ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_for; // keep input
    for (i=0; i<cpInfos->n; ++i) {
        ADOLC_WRITE_SCAYLOR(T0[arg]);
        T0[arg]=cpInfos->dp_internal_for[i];
        ++arg;
    }
    delete[] cpInfos->dp_internal_for;
    cpInfos->dp_internal_for = NULL;

    // taping "on"
    ADOLC_CURRENT_TAPE_INFOS.traceFlag=oldTraceFlag;

    return 0;
}

void revolve_for(CpInfos *cpInfos) {
    /* init revolve */
    cpInfos->check=-1;
    cpInfos->capo=0;
    cpInfos->info = 0;
    cpInfos->fine=cpInfos->steps;

    /* execute all time steps */
    enum revolve_action whattodo;
    do {
        whattodo=revolve(&cpInfos->check, &cpInfos->capo, &cpInfos->fine,
                         cpInfos->checkpoints, &cpInfos->info);
        switch (whattodo) {
            case revolve_takeshot:
                cp_takeshot(cpInfos);
                cpInfos->currentCP=cpInfos->capo;
                break;
            case revolve_advance:
                for (int i=0; i<cpInfos->capo-cpInfos->currentCP; ++i) {
                    cpInfos->function_double(cpInfos->n, cpInfos->dp_internal_for);
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
    } while (whattodo==revolve_takeshot || whattodo==revolve_advance);
}

int cp_fos_forward (int n, double *dp_x, double *dp_X,
                    int m, double *dp_y, double *dp_Y) {
    printf("WARNING: Checkpointing algorithm not "
           "implemented for the fos_forward mode!\n");
    return 0;
}

int cp_fov_forward (int n, double *dp_x, int p, double **dpp_X,
                    int m, double *dp_y, double **dpp_Y) {
    printf("WARNING: Checkpointing algorithm not "
           "implemented for the fov_forward mode!\n");
    return 0;
}

int cp_hos_forward (int n, double *dp_x, int d, double **dpp_X,
                    int m, double *dp_y, double **dpp_Y) {
    printf("WARNING: Checkpointing algorithm not "
           "implemented for the hos_forward mode!\n");
    return 0;
}

int cp_hov_forward (int n, double *dp_x, int d, int p, double ***dppp_X,
                    int m, double *dp_y, double ***dppp_Y) {
    printf("WARNING: Checkpointing algorithm not "
           "implemented for the hov_forward mode!\n");
    return 0;
}

int cp_fos_reverse (int m, double *dp_U, int n, double *dp_Z, double *dp_x, double *dp_y) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    revreal *A = ADOLC_CURRENT_TAPE_INFOS.rp_A;
    int oldTraceFlag;
    locint arg;
    CpInfos *cpInfos=get_cp_fct(ADOLC_CURRENT_TAPE_INFOS.cpIndex);
    char old_bsw;

    // note the mode
    cpInfos->modeReverse = ADOLC_FOS_REVERSE;

    cpInfos->dp_internal_for=new double[cpInfos->n];
    cpInfos->dp_internal_rev=new double[cpInfos->n];

    // taping "off"
    oldTraceFlag=ADOLC_CURRENT_TAPE_INFOS.traceFlag;
    ADOLC_CURRENT_TAPE_INFOS.traceFlag=0;

    arg=ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_rev;
    for (int i=0; i<cpInfos->n; ++i) {
        cpInfos->dp_internal_rev[i]=A[arg];
        ++arg;
    }        
    // update taylor buffer
    for (int i = 0; i < cpInfos->n; ++i) {
        --arg;
        ADOLC_GET_TAYLOR(arg);
    }
    // execute second part of revolve_firstturn left from forward sweep
    fos_reverse(cpInfos->tapeNumber, cpInfos->n, cpInfos->n,
                cpInfos->dp_internal_rev, cpInfos->dp_internal_rev);

    old_bsw = ADOLC_GLOBAL_TAPE_VARS.branchSwitchWarning;
    ADOLC_GLOBAL_TAPE_VARS.branchSwitchWarning = 0;
    // checkpointing
    enum revolve_action whattodo;
    do {
        whattodo=revolve(&cpInfos->check, &cpInfos->capo, &cpInfos->fine,
                         cpInfos->checkpoints, &cpInfos->info);
        switch (whattodo) {
            case revolve_terminate:
                break;
            case revolve_takeshot:
                cp_takeshot(cpInfos);
                cpInfos->currentCP=cpInfos->capo;
                break;
            case revolve_advance:
                for (int i=0; i<cpInfos->capo-cpInfos->currentCP; ++i)
                    cpInfos->function_double(cpInfos->n, cpInfos->dp_internal_for);
                break;
            case revolve_youturn:
                if (cpInfos->retaping != 0) cp_taping(cpInfos); // retaping forced
                else {
                    // one forward step with keep and retaping if necessary
                    if (zos_forward(cpInfos->tapeNumber, cpInfos->n, cpInfos->n, 1,
                                    cpInfos->dp_internal_for, cpInfos->dp_internal_for) < 0)
                        cp_taping(cpInfos);
                }
                // one reverse step
                fos_reverse(cpInfos->tapeNumber, cpInfos->n, cpInfos->n,
                            cpInfos->dp_internal_rev, cpInfos->dp_internal_rev);
                break;
            case revolve_restore:
                if (cpInfos->capo!=cpInfos->currentCP) cp_release(cpInfos);
                cpInfos->currentCP=cpInfos->capo;
                cp_restore(cpInfos);
                break;
            case revolve_error:
                revolveError(cpInfos);
                break;
            default:
                fail(ADOLC_CHECKPOINTING_UNEXPECTED_REVOLVE_ACTION);
                break;
        }
    } while (whattodo!=revolve_terminate && whattodo!=revolve_error);
    cp_release(cpInfos); // release first checkpoint if written
    ADOLC_GLOBAL_TAPE_VARS.branchSwitchWarning = old_bsw;

    // save results
    arg=ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_rev;
    for (int i=0; i<cpInfos->n; ++i) {
        A[arg]=cpInfos->dp_internal_rev[i];
        ++arg;
    }

    // clean up
    delete[] cpInfos->dp_internal_for;
    cpInfos->dp_internal_for = NULL;
    delete[] cpInfos->dp_internal_rev;
    cpInfos->dp_internal_rev = NULL;

    // taping "on"
    ADOLC_CURRENT_TAPE_INFOS.traceFlag=oldTraceFlag;

    return 0;
}

int cp_fov_reverse (int m, int p, double **dpp_U, int n, double **dpp_Z, double */*unused*/, double */*unused*/) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    revreal **A = ADOLC_CURRENT_TAPE_INFOS.rpp_A;
    int oldTraceFlag, numDirs;
    locint arg;
    CpInfos *cpInfos = get_cp_fct(ADOLC_CURRENT_TAPE_INFOS.cpIndex);
    char old_bsw;

    // note the mode
    cpInfos->modeReverse = ADOLC_FOV_REVERSE;

    numDirs = ADOLC_CURRENT_TAPE_INFOS.numDirs_rev;
    cpInfos->dp_internal_for = new double[cpInfos->n];
    cpInfos->dpp_internal_rev = myalloc2(numDirs, cpInfos->n);

    // taping "off"
    oldTraceFlag = ADOLC_CURRENT_TAPE_INFOS.traceFlag;
    ADOLC_CURRENT_TAPE_INFOS.traceFlag = 0;

    arg = ADOLC_CURRENT_TAPE_INFOS.lowestYLoc_rev;
    for (int i = 0; i < cpInfos->n; ++i) {
        for (int j = 0; j < numDirs; ++j) {
            cpInfos->dpp_internal_rev[j][i] = A[arg][j];
        }
        ++arg;
    }
    // update taylor buffer
    for (int i = 0; i < cpInfos->n; ++i) {
        --arg;
        ADOLC_GET_TAYLOR(arg);
    }
    // execute second part of revolve_firstturn left from forward sweep
    fov_reverse(cpInfos->tapeNumber, cpInfos->n, cpInfos->n, numDirs,
                cpInfos->dpp_internal_rev, cpInfos->dpp_internal_rev);

    old_bsw = ADOLC_GLOBAL_TAPE_VARS.branchSwitchWarning;
    ADOLC_GLOBAL_TAPE_VARS.branchSwitchWarning = 0;
    // checkpointing
    enum revolve_action whattodo;
    do {
        whattodo=revolve(&cpInfos->check, &cpInfos->capo, &cpInfos->fine,
                         cpInfos->checkpoints, &cpInfos->info);
        switch (whattodo) {
            case revolve_terminate:
                break;
            case revolve_takeshot:
                cp_takeshot(cpInfos);
                cpInfos->currentCP = cpInfos->capo;
                break;
            case revolve_advance:
                for (int i = 0; i < cpInfos->capo - cpInfos->currentCP; ++i)
                    cpInfos->function_double(cpInfos->n, cpInfos->dp_internal_for);
                break;
            case revolve_youturn:
                if (cpInfos->retaping != 0) cp_taping(cpInfos); // retaping forced
                else {
                    // one forward step with keep and retaping if necessary
                    if (zos_forward(cpInfos->tapeNumber, cpInfos->n, cpInfos->n, 1,
                                    cpInfos->dp_internal_for, cpInfos->dp_internal_for) < 0)
                        cp_taping(cpInfos);
                }
                // one reverse step
                fov_reverse(cpInfos->tapeNumber, cpInfos->n, cpInfos->n, numDirs,
                            cpInfos->dpp_internal_rev, cpInfos->dpp_internal_rev);
                break;
            case revolve_restore:
                if (cpInfos->capo != cpInfos->currentCP) cp_release(cpInfos);
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
    for (int i = 0; i < cpInfos->n; ++i) {
        for (int j = 0; j < numDirs; ++j) {
            A[arg][j] = cpInfos->dpp_internal_rev[j][i];
        }
        ++arg;
    }

    // clean up
    delete[] cpInfos->dp_internal_for;
    cpInfos->dp_internal_for = NULL;
    myfree2(cpInfos->dpp_internal_rev);
    cpInfos->dpp_internal_rev = NULL;

    // taping "on"
    ADOLC_CURRENT_TAPE_INFOS.traceFlag=oldTraceFlag;

    return 0;
}

int cp_hos_reverse (int m, double *dp_U, int n, int d,  double **dpp_Z) {
    printf("WARNING: Checkpointing algorithm not "
           "implemented for the hos_reverse mode!\n");
    return 0;
}

int cp_hov_reverse (int m, int p, double **dpp_U, int n, int d, double ***dppp_Z,
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

void cp_takeshot (CpInfos *cpInfos) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    StackElement se = new double *[2];
    ADOLC_CHECKPOINTS_STACK.push(se);
    se[0] = new double[cpInfos->n];
    for (int i = 0; i < cpInfos->n; ++i)
        se[0][i] = cpInfos->dp_internal_for[i];
    if (cpInfos->saveNonAdoubles != NULL)
        se[1] = static_cast<double *>(cpInfos->saveNonAdoubles());
    else
        se[1] = NULL;
}

void cp_restore (CpInfos *cpInfos) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    StackElement se = ADOLC_CHECKPOINTS_STACK.top();
    for (int i = 0; i < cpInfos->n; ++i)
        cpInfos->dp_internal_for[i] = se[0][i];
    if (se[1] != NULL)
        cpInfos->restoreNonAdoubles(static_cast<void *>(se[1]));
}

void cp_release (CpInfos *cpInfos) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    if (!ADOLC_CHECKPOINTS_STACK.empty()) {
        StackElement se = ADOLC_CHECKPOINTS_STACK.top();
        ADOLC_CHECKPOINTS_STACK.pop();
        delete[] se[0];
        if (se[1] != NULL)
            delete[] se[1];
        delete[] se;
    }
}

void cp_taping(CpInfos *cpInfos) {
    adouble *tapingAdoubles = new adouble[cpInfos->n];

    trace_on(cpInfos->tapeNumber, 1);

    for (int i = 0; i < cpInfos->n; ++i)
        tapingAdoubles[i] <<= cpInfos->dp_internal_for[i];

    cpInfos->function(cpInfos->n, tapingAdoubles);

    for (int i = 0; i < cpInfos->n; ++i)
        tapingAdoubles[i] >>= cpInfos->dp_internal_for[i];

    trace_off();

    delete[] tapingAdoubles;
}

/****************************************************************************/
/*                                                   revolve error function */
/****************************************************************************/
void revolveError (CpInfos *cpInfos) {
    switch(cpInfos->info) {
        case 10:
            printf("   Number of checkpoints stored exceeds "
                   "checkup!\n   Increase constant 'checkup' "
                   "and recompile!\n");
            break;
        case 11:
            printf("   Number of checkpoints stored = %d exceeds "
                   "snaps = %d!\n   Ensure 'snaps' > 0 and increase "
                   "initial 'fine'!\n", cpInfos->check+1,
                   cpInfos->checkpoints);
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


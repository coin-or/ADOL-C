/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     tape_handling.cpp
 Revision: $Id: tape_handling.cpp 37 2009-05-28 12:56:44Z awalther $
 Contents: management of tape infos

 Copyright (c) Andreas Kowarz
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/
#include <taping_p.h>
#include <checkpointing_p.h>
#include <revolve.h>

#include <iostream>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <stack>
#include <errno.h>

using namespace std;

/* vector of tape infos for all tapes in use */
vector<TapeInfos *> ADOLC_TAPE_INFOS_BUFFER_DECL;

/* stack of pointers to tape infos
 * represents the order of tape usage when doing nested taping */
stack<TapeInfos *> ADOLC_TAPE_STACK_DECL;

/* the main tape info buffer and its fallback */
TapeInfos ADOLC_CURRENT_TAPE_INFOS_DECL;
TapeInfos ADOLC_CURRENT_TAPE_INFOS_FALLBACK_DECL;

/* global tapeing variables */
GlobalTapeVars ADOLC_GLOBAL_TAPE_VARS_DECL;

#if defined(_OPENMP)
static vector<TapeInfos *> *tapeInfosBuffer_s;
static stack<TapeInfos *>  *tapeStack_s;
static TapeInfos           *currentTapeInfos_s;
static TapeInfos           *currentTapeInfos_fallBack_s;
static GlobalTapeVars      *globalTapeVars_s;
static ADOLC_BUFFER_TYPE   *ADOLC_extDiffFctsBuffer_s;
static stack<StackElement> *ADOLC_checkpointsStack_s;
static revolve_nums        *revolve_numbers_s;

static vector<TapeInfos *> *tapeInfosBuffer_p;
static stack<TapeInfos *>  *tapeStack_p;
static TapeInfos           *currentTapeInfos_p;
static TapeInfos           *currentTapeInfos_fallBack_p;
static GlobalTapeVars      *globalTapeVars_p;
static ADOLC_BUFFER_TYPE   *ADOLC_extDiffFctsBuffer_p;
static stack<StackElement> *ADOLC_checkpointsStack_p;
static revolve_nums        *revolve_numbers_p;
#endif

/*--------------------------------------------------------------------------*/
/* This function sets the flag "newTape" if either a taylor buffer has been */
/* created or a taping process has been performed. Calling the function is  */
/* also useful to "convince" the linker of including the cleaner part into  */
/* the binary when linking statically!                                      */
/*--------------------------------------------------------------------------*/
void markNewTape() {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    ADOLC_GLOBAL_TAPE_VARS.newTape = 1;
}

/* inits the struct for the new tape */
void initTapeInfos(TapeInfos *newTapeInfos) {
    char *ptr;

    ptr = (char *)newTapeInfos;
    for (unsigned int i = 0; i < sizeof(TapeInfos) -
            sizeof(PersistantTapeInfos); ++i) ptr[i] = 0;
}

/* as above but keep allocated buffers if possible */
void initTapeInfos_keep(TapeInfos *newTapeInfos) {
    unsigned char *opBuffer = newTapeInfos->opBuffer;
    locint *locBuffer = newTapeInfos->locBuffer;
    double *valBuffer = newTapeInfos->valBuffer;
    revreal *tayBuffer = newTapeInfos->tayBuffer;
    FILE *tay_file = newTapeInfos->tay_file;

    initTapeInfos(newTapeInfos);

    newTapeInfos->opBuffer = opBuffer;
    newTapeInfos->locBuffer = locBuffer;
    newTapeInfos->valBuffer = valBuffer;
    newTapeInfos->tayBuffer = tayBuffer;
    newTapeInfos->tay_file = tay_file;
}

/* inits a new tape and updates the tape stack (called from start_trace)
 * - returns 0 without error
 * - returns 1 if tapeID was already/still in use */
int initNewTape(short tapeID) {
    TapeInfos *newTapeInfos = NULL;
    bool newTI = false;
    int retval = 0;

    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    /* check if tape is in use */
    vector<TapeInfos *>::iterator tiIter;
    if (!ADOLC_TAPE_INFOS_BUFFER.empty()) {
        for (tiIter=ADOLC_TAPE_INFOS_BUFFER.begin();
                tiIter!=ADOLC_TAPE_INFOS_BUFFER.end();
                ++tiIter) {
            if ((*tiIter)->tapeID==tapeID) {
                newTapeInfos=*tiIter;
                if ((*tiIter)->inUse != 0) {
                    if ((*tiIter)->tapingComplete == 0)
                        fail(ADOLC_TAPING_TAPE_STILL_IN_USE);
                    if ( (*tiIter)->stats[OP_FILE_ACCESS]  == 0 &&
                            (*tiIter)->stats[LOC_FILE_ACCESS] == 0 &&
                            (*tiIter)->stats[VAL_FILE_ACCESS] == 0  ) {
#              if defined(ADOLC_DEBUG)
                        fprintf(DIAG_OUT, "\nADOL-C warning: Tape %d existed in main memory"
                                " only and gets overwritten!\n\n", tapeID);
#              endif
                        /* free associated resources */
                        retval = 1;
                    }
                }
                if ((*tiIter)->tay_file != NULL)
                    rewind((*tiIter)->tay_file);
                initTapeInfos_keep(*tiIter);
                (*tiIter)->tapeID = tapeID;
                break;
            }
        }
    }

    /* create new info struct and initialize it */
    if (newTapeInfos == NULL) {
        newTapeInfos = new TapeInfos(tapeID);
        newTI = true;
    }
    newTapeInfos->traceFlag=1;
    newTapeInfos->inUse=1;
    newTapeInfos->pTapeInfos.inJacSparseUse=0;
    newTapeInfos->pTapeInfos.inHessSparseUse=0;
    newTapeInfos->stats[OP_BUFFER_SIZE] =
        ADOLC_GLOBAL_TAPE_VARS.operationBufferSize;
    newTapeInfos->stats[LOC_BUFFER_SIZE] =
        ADOLC_GLOBAL_TAPE_VARS.locationBufferSize;
    newTapeInfos->stats[VAL_BUFFER_SIZE] =
        ADOLC_GLOBAL_TAPE_VARS.valueBufferSize;
    newTapeInfos->stats[TAY_BUFFER_SIZE] =
        ADOLC_GLOBAL_TAPE_VARS.taylorBufferSize;

    /* update tapeStack and save tapeInfos */
    if (ADOLC_GLOBAL_TAPE_VARS.currentTapeInfosPtr != NULL) {
        memcpy(ADOLC_GLOBAL_TAPE_VARS.currentTapeInfosPtr,
                &ADOLC_CURRENT_TAPE_INFOS, sizeof(TapeInfos));
        ADOLC_TAPE_STACK.push(ADOLC_GLOBAL_TAPE_VARS.currentTapeInfosPtr);
    } else {
        memcpy(&ADOLC_CURRENT_TAPE_INFOS_FALLBACK,
                &ADOLC_CURRENT_TAPE_INFOS, sizeof(TapeInfos));
        ADOLC_TAPE_STACK.push(&ADOLC_CURRENT_TAPE_INFOS_FALLBACK);
    }
    if (newTI) ADOLC_TAPE_INFOS_BUFFER.push_back(newTapeInfos);

    /* set the new tape infos as current */
    memcpy(&ADOLC_CURRENT_TAPE_INFOS, newTapeInfos, sizeof(TapeInfos));
    ADOLC_GLOBAL_TAPE_VARS.currentTapeInfosPtr = newTapeInfos;

    return retval;
}

/* opens an existing tape or creates a new handle for a tape on hard disk
 * - called from init_for_sweep and init_rev_sweep */
void openTape(short tapeID, char mode) {
    TapeInfos *tempTapeInfos=NULL;

    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    /* check if tape information exist in memory */
    vector<TapeInfos *>::iterator tiIter;
    if (!ADOLC_TAPE_INFOS_BUFFER.empty()) {
        for (tiIter=ADOLC_TAPE_INFOS_BUFFER.begin();
                tiIter!=ADOLC_TAPE_INFOS_BUFFER.end();
                ++tiIter) {
            if ((*tiIter)->tapeID == tapeID) {
                /* tape has been used before (in the current program) */
                if ((*tiIter)->inUse == 0) {
                    /* forward sweep */
                    if ((*tiIter)->tay_file != NULL)
                        rewind((*tiIter)->tay_file);
                    initTapeInfos_keep(*tiIter);
                    (*tiIter)->traceFlag=1;
                    (*tiIter)->tapeID = tapeID;
                    (*tiIter)->tapingComplete = 1;
                    (*tiIter)->inUse = 1;
                    read_tape_stats(*tiIter);
               }
                if (ADOLC_GLOBAL_TAPE_VARS.currentTapeInfosPtr != NULL) {
                    memcpy(ADOLC_GLOBAL_TAPE_VARS.currentTapeInfosPtr,
                            &ADOLC_CURRENT_TAPE_INFOS, sizeof(TapeInfos));
                    ADOLC_TAPE_STACK.push(
                            ADOLC_GLOBAL_TAPE_VARS.currentTapeInfosPtr);
                } else {
                    memcpy(&ADOLC_CURRENT_TAPE_INFOS_FALLBACK,
                            &ADOLC_CURRENT_TAPE_INFOS, sizeof(TapeInfos));
                    ADOLC_TAPE_STACK.push(&ADOLC_CURRENT_TAPE_INFOS_FALLBACK);
                }
                memcpy(&ADOLC_CURRENT_TAPE_INFOS, *tiIter, sizeof(TapeInfos));
                ADOLC_GLOBAL_TAPE_VARS.currentTapeInfosPtr = *tiIter;
                return;
            }
        }
    }

    /* tapeID not used so far */
    if (mode == ADOLC_REVERSE) {
        failAdditionalInfo1 = tapeID;
        fail(ADOLC_REVERSE_NO_TAYLOR_STACK);
    }

    /* create new info struct and initialize it */
    tempTapeInfos = new TapeInfos(tapeID);
    tempTapeInfos->traceFlag=1;
    tempTapeInfos->inUse = 1;
    tempTapeInfos->tapingComplete = 1;
    ADOLC_TAPE_INFOS_BUFFER.push_back(tempTapeInfos);

    read_tape_stats(tempTapeInfos);
    /* update tapeStack and save tapeInfos */
    if (ADOLC_GLOBAL_TAPE_VARS.currentTapeInfosPtr != NULL) {
        memcpy(ADOLC_GLOBAL_TAPE_VARS.currentTapeInfosPtr,
                &ADOLC_CURRENT_TAPE_INFOS, sizeof(TapeInfos));
        ADOLC_TAPE_STACK.push(ADOLC_GLOBAL_TAPE_VARS.currentTapeInfosPtr);
    } else {
        memcpy(&ADOLC_CURRENT_TAPE_INFOS_FALLBACK,
                &ADOLC_CURRENT_TAPE_INFOS, sizeof(TapeInfos));
        ADOLC_TAPE_STACK.push(&ADOLC_CURRENT_TAPE_INFOS_FALLBACK);
    }

    /* set the new tape infos as current */
    memcpy(&ADOLC_CURRENT_TAPE_INFOS, tempTapeInfos, sizeof(TapeInfos));
    ADOLC_GLOBAL_TAPE_VARS.currentTapeInfosPtr = tempTapeInfos;
}

/* release the current tape and give control to the previous one */
void releaseTape() {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    /* if operations, locations and constants tapes have been written and value
     * stack information have not been created tapeInfos are no longer needed*/
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors            == 0 &&
            ADOLC_CURRENT_TAPE_INFOS.stats[OP_FILE_ACCESS]  == 1 &&
            ADOLC_CURRENT_TAPE_INFOS.stats[LOC_FILE_ACCESS] == 1 &&
            ADOLC_CURRENT_TAPE_INFOS.stats[VAL_FILE_ACCESS] == 1 ) {
        ADOLC_CURRENT_TAPE_INFOS.inUse = 0;
    }

    memcpy(ADOLC_GLOBAL_TAPE_VARS.currentTapeInfosPtr,
            &ADOLC_CURRENT_TAPE_INFOS, sizeof(TapeInfos));
    ADOLC_GLOBAL_TAPE_VARS.currentTapeInfosPtr = ADOLC_TAPE_STACK.top();
    memcpy(&ADOLC_CURRENT_TAPE_INFOS,
            ADOLC_GLOBAL_TAPE_VARS.currentTapeInfosPtr, sizeof(TapeInfos));
    ADOLC_TAPE_STACK.pop();
    if (ADOLC_TAPE_STACK.empty())
        ADOLC_GLOBAL_TAPE_VARS.currentTapeInfosPtr = NULL;
}

/* updates the tape infos for the given ID - a tapeInfos struct is created
 * and registered if non is found but its state will remain "not in use" */
TapeInfos *getTapeInfos(short tapeID) {
    TapeInfos *tapeInfos;
    vector<TapeInfos *>::iterator tiIter;

    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    /* check if TapeInfos for tapeID exist */
    if (!ADOLC_TAPE_INFOS_BUFFER.empty()) {
        for (tiIter=ADOLC_TAPE_INFOS_BUFFER.begin();
                tiIter!=ADOLC_TAPE_INFOS_BUFFER.end();
                ++tiIter) {
            if ((*tiIter)->tapeID==tapeID) {
                tapeInfos=*tiIter;
                if (tapeInfos->inUse==0) read_tape_stats(tapeInfos);
                return tapeInfos;
            }
        }
    }
    /* create new TapeInfos, initialize and update tapeInfosBuffer */
    tapeInfos = new TapeInfos(tapeID);
    ADOLC_TAPE_INFOS_BUFFER.push_back(tapeInfos);
    tapeInfos->traceFlag=1;
    tapeInfos->inUse=0;
    tapeInfos->pTapeInfos.inJacSparseUse=0;
    tapeInfos->pTapeInfos.inHessSparseUse=0;
    tapeInfos->tapingComplete = 1;
    read_tape_stats(tapeInfos);
    return tapeInfos;
}

/* updates the tape infos on sparse Jac for the given ID  */
void setTapeInfoJacSparse(short tapeID, SparseJacInfos sJinfos) {
    TapeInfos *tapeInfos;
    vector<TapeInfos *>::iterator tiIter;

    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    /* check if TapeInfos for tapeID exist */
    if (!ADOLC_TAPE_INFOS_BUFFER.empty()) {
        for (tiIter=ADOLC_TAPE_INFOS_BUFFER.begin();
                tiIter!=ADOLC_TAPE_INFOS_BUFFER.end();
                ++tiIter) {
            if ((*tiIter)->tapeID==tapeID) {
                tapeInfos=*tiIter;
		    // memory deallocation is missing !!
		    tapeInfos->pTapeInfos.sJinfos.y=sJinfos.y;
		    tapeInfos->pTapeInfos.sJinfos.Seed=sJinfos.Seed;
		    tapeInfos->pTapeInfos.sJinfos.B=sJinfos.B;
		    tapeInfos->pTapeInfos.sJinfos.JP=sJinfos.JP;
		    tapeInfos->pTapeInfos.sJinfos.nnz_in=sJinfos.nnz_in;
		    tapeInfos->pTapeInfos.sJinfos.p=sJinfos.p;
		    tapeInfos->pTapeInfos.sJinfos.g=sJinfos.g;
            }
        }
    }
}

/* updates the tape infos on sparse Hess for the given ID  */
void setTapeInfoHessSparse(short tapeID, SparseHessInfos sHinfos) {
    TapeInfos *tapeInfos;
    vector<TapeInfos *>::iterator tiIter;

    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    /* check if TapeInfos for tapeID exist */
    if (!ADOLC_TAPE_INFOS_BUFFER.empty()) {
        for (tiIter=ADOLC_TAPE_INFOS_BUFFER.begin();
                tiIter!=ADOLC_TAPE_INFOS_BUFFER.end();
                ++tiIter) {
            if ((*tiIter)->tapeID==tapeID) {
                tapeInfos=*tiIter;
		    // memory deallocation is missing !!
		    tapeInfos->pTapeInfos.sHinfos.Hcomp=sHinfos.Hcomp;
		    tapeInfos->pTapeInfos.sHinfos.Xppp=sHinfos.Xppp;
		    tapeInfos->pTapeInfos.sHinfos.Yppp=sHinfos.Yppp;
		    tapeInfos->pTapeInfos.sHinfos.Zppp=sHinfos.Zppp;
		    tapeInfos->pTapeInfos.sHinfos.Upp=sHinfos.Upp;
		    tapeInfos->pTapeInfos.sHinfos.HP=sHinfos.HP;
		    tapeInfos->pTapeInfos.sHinfos.nnz_in=sHinfos.nnz_in;
		    tapeInfos->pTapeInfos.sHinfos.p=sHinfos.p;
		    tapeInfos->pTapeInfos.sHinfos.g=sHinfos.g;
            }
        }
    }
}

void init() {
    ADOLC_OPENMP_THREAD_NUMBER;
    errno = 0;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

#if defined(_OPENMP)
    tapeInfosBuffer = new vector<TapeInfos *>;
    tapeStack = new stack<TapeInfos *>;
    currentTapeInfos = new TapeInfos;
    currentTapeInfos->tapingComplete = 1;
    currentTapeInfos_fallBack = new TapeInfos;
    globalTapeVars = new GlobalTapeVars;
    ADOLC_extDiffFctsBuffer = new ADOLC_BUFFER_TYPE;
    ADOLC_checkpointsStack = new stack<StackElement>;
    revolve_numbers = new revolve_nums;
#endif /* _OPENMP */

    ADOLC_CURRENT_TAPE_INFOS.traceFlag = 0;
    ADOLC_CURRENT_TAPE_INFOS.keepTaylors = 0;

    ADOLC_GLOBAL_TAPE_VARS.store=NULL;
    ADOLC_GLOBAL_TAPE_VARS.maxLoc=1;
    for (uint i=0; i<sizeof(locint)*8-1; ++i) {
        ADOLC_GLOBAL_TAPE_VARS.maxLoc<<=1;
        ++ADOLC_GLOBAL_TAPE_VARS.maxLoc;
    }
    ADOLC_GLOBAL_TAPE_VARS.locMinUnused = 0;
    ADOLC_GLOBAL_TAPE_VARS.numMaxAlive = 0;
    ADOLC_GLOBAL_TAPE_VARS.storeSize = 0;
    ADOLC_GLOBAL_TAPE_VARS.numToFree = 0;
    ADOLC_GLOBAL_TAPE_VARS.minLocToFree = 0;
    ADOLC_GLOBAL_TAPE_VARS.inParallelRegion = 0;
    ADOLC_GLOBAL_TAPE_VARS.currentTapeInfosPtr = NULL;
    ADOLC_GLOBAL_TAPE_VARS.branchSwitchWarning = 1;

    adolc_id.adolc_ver    = ADOLC_VERSION;
    adolc_id.adolc_sub    = ADOLC_SUBVERSION;
    adolc_id.adolc_lvl    = ADOLC_PATCHLEVEL;
    adolc_id.locint_size  = sizeof(locint);
    adolc_id.revreal_size = sizeof(revreal);

    ADOLC_EXT_DIFF_FCTS_BUFFER.init(init_CpInfos);
}

/* does things like closing/removing temporary files, ... */
void cleanUp() {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    vector<TapeInfos *>::iterator tiIter;
    if (!ADOLC_TAPE_INFOS_BUFFER.empty()) {
        for (tiIter=ADOLC_TAPE_INFOS_BUFFER.begin();
                tiIter!=ADOLC_TAPE_INFOS_BUFFER.end();
                ++tiIter)
        {
            /* close open files though they may be incomplete */
            if ((*tiIter)->op_file!=NULL)
            {
                fclose((*tiIter)->op_file);
                (*tiIter)->op_file = NULL;
            }
            if ((*tiIter)->val_file!=NULL)
            {
                fclose((*tiIter)->val_file);
                (*tiIter)->val_file = NULL;
            }
            if ((*tiIter)->loc_file!=NULL)
            {
                fclose((*tiIter)->loc_file);
                (*tiIter)->loc_file = NULL;
            }
            if ((*tiIter)->tay_file!=NULL) {
                fclose((*tiIter)->tay_file);
                (*tiIter)->tay_file = NULL;
                remove((*tiIter)->pTapeInfos.tay_fileName);
            }
            if ((*tiIter)->opBuffer != NULL)
            {
                free((*tiIter)->opBuffer);
                (*tiIter)->opBuffer = NULL;
            }
            if ((*tiIter)->valBuffer != NULL)
            {
                free((*tiIter)->valBuffer);
                (*tiIter)->valBuffer = NULL;
            }
            if ((*tiIter)->locBuffer != NULL)
            {
                free((*tiIter)->locBuffer);
                (*tiIter)->locBuffer = NULL;
            }
            if ((*tiIter)->tayBuffer != NULL)
            {
                free((*tiIter)->tayBuffer);
                (*tiIter)->tayBuffer = NULL;
            }
            /* remove "main" tape files if not all three have been written */
            int filesWritten = (*tiIter)->stats[OP_FILE_ACCESS] +
                (*tiIter)->stats[LOC_FILE_ACCESS] +
                (*tiIter)->stats[VAL_FILE_ACCESS];
            if ( (filesWritten > 0) && ((*tiIter)->pTapeInfos.keepTape == 0) )
            {
                /* try to remove all tapes (even those not written by this
                 * run) => this ensures that there is no mixture of tapes from
                 * different ADOLC runs */
                if ( (*tiIter)->stats[OP_FILE_ACCESS] == 1 )
                    remove((*tiIter)->pTapeInfos.op_fileName);
                if ( (*tiIter)->stats[LOC_FILE_ACCESS] == 1 )
                    remove((*tiIter)->pTapeInfos.loc_fileName);
                if ( (*tiIter)->stats[VAL_FILE_ACCESS] == 1 )
                    remove((*tiIter)->pTapeInfos.val_fileName);
            }
            if ((*tiIter)->pTapeInfos.op_fileName != NULL)
            {
                free((*tiIter)->pTapeInfos.op_fileName);
                (*tiIter)->pTapeInfos.op_fileName = NULL;
            }
            if ((*tiIter)->pTapeInfos.val_fileName != NULL)
            {
                free((*tiIter)->pTapeInfos.val_fileName);
                (*tiIter)->pTapeInfos.val_fileName = NULL;
            }
            if ((*tiIter)->pTapeInfos.loc_fileName != NULL)
            {
                free((*tiIter)->pTapeInfos.loc_fileName);
                (*tiIter)->pTapeInfos.loc_fileName = NULL;
            }
            if ((*tiIter)->pTapeInfos.tay_fileName != NULL)
            {
                free((*tiIter)->pTapeInfos.tay_fileName);
                (*tiIter)->pTapeInfos.tay_fileName = NULL;
            }

            delete *tiIter;
        }
    }

    cp_clearStack();

    if (ADOLC_GLOBAL_TAPE_VARS.store != NULL) {
        free(ADOLC_GLOBAL_TAPE_VARS.store);
        ADOLC_GLOBAL_TAPE_VARS.store = NULL;
    }

#if defined(_OPENMP)
    if (ADOLC_GLOBAL_TAPE_VARS.inParallelRegion == 0) {
        /* cleanup on program exit */
        delete revolve_numbers;
        delete ADOLC_checkpointsStack;
        delete ADOLC_extDiffFctsBuffer;
        delete globalTapeVars;
        delete currentTapeInfos;
        delete currentTapeInfos_fallBack;
        delete tapeStack;
        delete tapeInfosBuffer;
    }
#endif

    ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
}

int removeTape(short tapeID, short type) {
    TapeInfos *tapeInfos = NULL;
    vector<TapeInfos *>::iterator tiIter;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    /* check if TapeInfos for tapeID exist */
    if (!ADOLC_TAPE_INFOS_BUFFER.empty()) {
        for (tiIter = ADOLC_TAPE_INFOS_BUFFER.begin();
                tiIter != ADOLC_TAPE_INFOS_BUFFER.end();
                ++tiIter)
        {
            if ((*tiIter)->tapeID == tapeID) {
                tapeInfos = *tiIter;
                if (tapeInfos->tapingComplete == 0) return -1;
                ADOLC_TAPE_INFOS_BUFFER.erase(tiIter);
                break;
            }
        }
    }

    if (tapeInfos == NULL) { // might be on disk only
        tapeInfos = new TapeInfos(tapeID);
        tapeInfos->tapingComplete = 1;
    }

    freeTapeResources(tapeInfos);
    ADOLC_OPENMP_RESTORE_THREAD_NUMBER;

    if (type == ADOLC_REMOVE_COMPLETELY) {
        remove(tapeInfos->pTapeInfos.op_fileName);
        remove(tapeInfos->pTapeInfos.loc_fileName);
        remove(tapeInfos->pTapeInfos.val_fileName);
    }

    free(tapeInfos->pTapeInfos.op_fileName);
    free(tapeInfos->pTapeInfos.val_fileName);
    free(tapeInfos->pTapeInfos.loc_fileName);
    if (tapeInfos->pTapeInfos.tay_fileName != NULL)
        free(tapeInfos->pTapeInfos.tay_fileName);

    delete tapeInfos;

    return 0;
}

/****************************************************************************/
/* Initialization for the taping process. Creates buffers for this tape,    */
/* sets files names, and calls appropriate setup routines.                  */
/****************************************************************************/
int trace_on(short tnum, int keepTaylors) {
    int retval = 0;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    /* allocate memory for TapeInfos and update tapeStack */
    retval = initNewTape(tnum);
    ADOLC_CURRENT_TAPE_INFOS.keepTaylors=keepTaylors;
    if (keepTaylors!=0) ADOLC_CURRENT_TAPE_INFOS.deg_save=1;
    start_trace();
    take_stock();               /* record all existing adoubles on the tape */
    return retval;
}

int trace_on(short tnum, int keepTaylors,
        uint obs, uint lbs, uint vbs, uint tbs)
{
    int retval = 0;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    /* allocate memory for TapeInfos and update tapeStack */
    retval = initNewTape(tnum);
    ADOLC_CURRENT_TAPE_INFOS.stats[OP_BUFFER_SIZE] = obs;
    ADOLC_CURRENT_TAPE_INFOS.stats[LOC_BUFFER_SIZE] = lbs;
    ADOLC_CURRENT_TAPE_INFOS.stats[VAL_BUFFER_SIZE] = vbs;
    ADOLC_CURRENT_TAPE_INFOS.stats[TAY_BUFFER_SIZE] = tbs;
    ADOLC_CURRENT_TAPE_INFOS.keepTaylors=keepTaylors;
    if (keepTaylors!=0) ADOLC_CURRENT_TAPE_INFOS.deg_save=1;
    start_trace();
    take_stock();               /* record all existing adoubles on the tape */
    return retval;
}

/****************************************************************************/
/* Stop Tracing. Cleans up, and turns off trace_flag. Flag not equal zero   */
/* enforces writing of the three main tape files (op+loc+val).              */
/****************************************************************************/
void trace_off(int flag) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.keepTape = flag;
    keep_stock();         /* copy remaining live variables + trace_flag = 0 */
    stop_trace(flag);
    cout.flush();
    ADOLC_CURRENT_TAPE_INFOS.tapingComplete = 1;
    releaseTape();
}

bool isTaping() {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    return ADOLC_CURRENT_TAPE_INFOS.traceFlag != 0;
}



/****************************************************************************/
/* A class for initialization/finalization and OpenMP handling              */
/****************************************************************************/
class Keeper {
    public:
        inline Keeper() {
            dummy = 0;
            init();
            readConfigFile();
        }
        inline ~Keeper() {
            cleanUp();
        }

        inline void touch() {
            dummy = 1;
        }

    private:
        int dummy;
};

/* a static instance that does all work */
static Keeper theKeeper;

/**
 * Hope to convince the linker to link the keeper code into the executable. */
void initADOLC() {
    theKeeper.touch();
}

/****************************************************************************/
/****************************************************************************/
/* The following is necessary to provide a separate ADOL-C environment for  */
/* each OpenMP worker.                                                      */
/****************************************************************************/
/****************************************************************************/
#if defined(_OPENMP)
#include "adolc_openmp.h"

ADOLC_OpenMP ADOLC_OpenMP_Handler;
ADOLC_OpenMP_NC ADOLC_OpenMP_Handler_NC;
int ADOLC_parallel_doCopy;

static bool waitForMaster_begin = true;
static bool waitForMaster_end   = true;
static bool firstParallel       = true;

/****************************************************************************/
/* Used by OpenMP to create a separate environment for every worker thread. */
/****************************************************************************/
void beginParallel() {
    ADOLC_OPENMP_THREAD_NUMBER;
#if defined(ADOLC_THREADSAVE_ERRNO)
    errno = omp_get_thread_num();
#endif
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    if (ADOLC_threadNumber == 0) { /* master only */
        int numThreads = omp_get_num_threads();

        tapeInfosBuffer_s           = tapeInfosBuffer;
        tapeStack_s                 = tapeStack;
        currentTapeInfos_s          = currentTapeInfos;
        currentTapeInfos_fallBack_s = currentTapeInfos_fallBack;
        globalTapeVars_s            = globalTapeVars;
        ADOLC_extDiffFctsBuffer_s   = ADOLC_extDiffFctsBuffer;
        ADOLC_checkpointsStack_s    = ADOLC_checkpointsStack;
        revolve_numbers_s           = revolve_numbers;

        if (firstParallel) {
            tapeInfosBuffer           = new vector<TapeInfos *>[numThreads];
            tapeStack                 = new stack<TapeInfos *>[numThreads];
            currentTapeInfos          = new TapeInfos[numThreads];
            currentTapeInfos_fallBack = new TapeInfos[numThreads];
            globalTapeVars            = new GlobalTapeVars[numThreads];
            ADOLC_extDiffFctsBuffer   = new ADOLC_BUFFER_TYPE[numThreads];
            ADOLC_checkpointsStack    = new stack<StackElement>[numThreads];
            revolve_numbers           = new revolve_nums[numThreads];
        } else {
            tapeInfosBuffer           = tapeInfosBuffer_p;
            tapeStack                 = tapeStack_p;
            currentTapeInfos          = currentTapeInfos_p;
            currentTapeInfos_fallBack = currentTapeInfos_fallBack_p;
            globalTapeVars            = globalTapeVars_p;
            ADOLC_extDiffFctsBuffer   = ADOLC_extDiffFctsBuffer_p;
            ADOLC_checkpointsStack    = ADOLC_checkpointsStack_p;
            revolve_numbers         = revolve_numbers_p;
        }

        /* - set inParallelRegion for tmpGlobalTapeVars because it is source
         *   for initializing the parallel globalTapeVars structs
         * - inParallelRegion has to be set to one for all workers by master.
         *   This is necessary, to deter a speedy master from assuming all
         *   workers are done, in endParallel, before they even leaved
         *   beginParallel. */
        globalTapeVars_s[0].inParallelRegion = 1;
        for (int i = 0; i < numThreads; ++i)
            globalTapeVars[i].inParallelRegion = 1;

        waitForMaster_end = true;
        waitForMaster_begin = false;
    } else 
        while (waitForMaster_begin) {
            usleep(1000); /* if anyone knows a better value, ... :-) */
        }

    if (firstParallel) {
        ADOLC_EXT_DIFF_FCTS_BUFFER.init(init_CpInfos);
        memcpy(&ADOLC_GLOBAL_TAPE_VARS, globalTapeVars_s, sizeof(GlobalTapeVars));
        ADOLC_GLOBAL_TAPE_VARS.store = (double *)
            malloc(sizeof(double) * ADOLC_GLOBAL_TAPE_VARS.storeSize);
        memcpy(ADOLC_GLOBAL_TAPE_VARS.store, globalTapeVars_s->store,
                ADOLC_GLOBAL_TAPE_VARS.locMinUnused * sizeof(double));
        ADOLC_GLOBAL_TAPE_VARS.newTape = 0;
        ADOLC_CURRENT_TAPE_INFOS.tapingComplete = 1;
        ADOLC_GLOBAL_TAPE_VARS.currentTapeInfosPtr = NULL;
    } else {
        if (ADOLC_parallel_doCopy) {
            ADOLC_GLOBAL_TAPE_VARS.locMinUnused = globalTapeVars_s->locMinUnused;
            ADOLC_GLOBAL_TAPE_VARS.numMaxAlive = globalTapeVars_s->numMaxAlive;
            ADOLC_GLOBAL_TAPE_VARS.storeSize = globalTapeVars_s->storeSize;
            ADOLC_GLOBAL_TAPE_VARS.numToFree = globalTapeVars_s->numToFree;
            ADOLC_GLOBAL_TAPE_VARS.minLocToFree = globalTapeVars_s->minLocToFree;
            ADOLC_GLOBAL_TAPE_VARS.branchSwitchWarning = globalTapeVars_s->branchSwitchWarning;
            free(ADOLC_GLOBAL_TAPE_VARS.store);
            ADOLC_GLOBAL_TAPE_VARS.store = (double *)
                malloc(sizeof(double) * ADOLC_GLOBAL_TAPE_VARS.storeSize);
            memcpy(ADOLC_GLOBAL_TAPE_VARS.store, globalTapeVars_s->store,
                    ADOLC_GLOBAL_TAPE_VARS.locMinUnused * sizeof(double));
        }
    }
}

/****************************************************************************/
/* Used by OpenMP to destroy the separate environment of every worker.      */
/****************************************************************************/
/* There are n+1 instances of ADOLC_OpenMP => n within the parallel region
 * and one in the serial part! */
void endParallel() {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    /* do nothing if called at program exit (serial part) */
    if (ADOLC_threadNumber == 0 &&
            ADOLC_GLOBAL_TAPE_VARS.inParallelRegion == 0) return;

    ADOLC_GLOBAL_TAPE_VARS.inParallelRegion = 0;

    if (ADOLC_threadNumber == 0) { /* master only */
        int num;
        int numThreads = omp_get_num_threads();
        bool firstIt = true;
        do { /* wait until all slaves have left the parallel part */
            if (firstIt) firstIt = false;
            else usleep(1000); /* no busy waiting */
            num = 1;
            for (int i = 1; i < numThreads; ++i)
                if (globalTapeVars[i].inParallelRegion == 0) ++num;
        } while (num != numThreads);

        firstParallel = false;

        revolve_numbers_p           = revolve_numbers;
        ADOLC_checkpointsStack_p    = ADOLC_checkpointsStack;
        ADOLC_extDiffFctsBuffer_p   = ADOLC_extDiffFctsBuffer;
        globalTapeVars_p            = globalTapeVars;
        currentTapeInfos_p          = currentTapeInfos;
        currentTapeInfos_fallBack_p = currentTapeInfos_fallBack;
        tapeStack_p                 = tapeStack;
        tapeInfosBuffer_p           = tapeInfosBuffer;

        revolve_numbers           = revolve_numbers_s;
        ADOLC_checkpointsStack    = ADOLC_checkpointsStack_s;
        ADOLC_extDiffFctsBuffer   = ADOLC_extDiffFctsBuffer_s;
        globalTapeVars            = globalTapeVars_s;
        currentTapeInfos          = currentTapeInfos_s;
        currentTapeInfos_fallBack = currentTapeInfos_fallBack_s;
        tapeStack                 = tapeStack_s;
        tapeInfosBuffer           = tapeInfosBuffer_s;

        ADOLC_GLOBAL_TAPE_VARS.inParallelRegion = 0;
        waitForMaster_begin = true;
        waitForMaster_end = false;
    } else
        while (waitForMaster_end) {
            usleep(1000); // no busy waiting
        }
}

#endif /* _OPENMP */

TapeInfos::TapeInfos() {
    initTapeInfos(this);
}

TapeInfos::TapeInfos(short _tapeID) {
    initTapeInfos(this);
    tapeID = _tapeID;
    pTapeInfos.op_fileName = createFileName(tapeID, OPERATIONS_TAPE);
    pTapeInfos.loc_fileName = createFileName(tapeID, LOCATIONS_TAPE);
    pTapeInfos.val_fileName = createFileName(tapeID, VALUES_TAPE);
    pTapeInfos.tay_fileName = NULL;
}


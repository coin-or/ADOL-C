/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     tape_handling.cpp
 Revision: $Id$
 Contents: management of tape infos

 Copyright (c) Andreas Kowarz, Andrea Walther, Kshitij Kulshreshtha,
               Benjamin Letschert, Jean Utke

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/
#include "taping_p.h"
#include "checkpointing_p.h"
#include "dvlparms.h"
#include <adolc/revolve.h>
#include <adolc/adalloc.h>

#ifdef ADOLC_MEDIPACK_SUPPORT
#include "medipacksupport_p.h"
#endif

#include <cassert>
#include <limits>
#include <iostream>
#include <string.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <vector>
#include <stack>
#include <errno.h>

using namespace std;

#ifdef SPARSE
BEGIN_C_DECLS
extern void freeSparseJacInfos(double *y, double **B, unsigned int **JP, void *g, 
			       void *jr1d, int seed_rows, int seed_clms, int depen);
extern void freeSparseHessInfos(double **Hcomp, double ***Xppp, double ***Yppp, double ***Zppp, 
				double **Upp, unsigned int **HP,
				void *g, void *hr, int p, int indep);
END_C_DECLS
#endif

GlobalTapeVarsCL::GlobalTapeVarsCL() {
  store = NULL;
#if defined(ADOLC_TRACK_ACTIVITY)
  actStore = NULL;
#endif
  storeSize = 0;
  numLives = 0;
  nominmaxFlag = 0;
  pStore = NULL;
  numparam = 0;
  maxparam = 0;
  initialStoreSize = 0;
#if defined(ADOLC_TRACK_ACTIVITY)
  storeManagerPtr = new StoreManagerLocintBlock(store, actStore, storeSize, numLives);
#else
  storeManagerPtr = new StoreManagerLocintBlock(store, storeSize, numLives);
#endif
  paramStoreMgrPtr = new StoreManagerLocintBlock(pStore, maxparam, numparam);
}

GlobalTapeVarsCL::~GlobalTapeVarsCL() {
  if (storeManagerPtr != NULL) {
    delete storeManagerPtr;
    storeManagerPtr = NULL;
  }
  if (paramStoreMgrPtr != NULL) {
      delete paramStoreMgrPtr;
      paramStoreMgrPtr = NULL;
  }
}

const GlobalTapeVarsCL& GlobalTapeVarsCL::operator=(const GlobalTapeVarsCL& gtv) {
    storeSize = gtv.storeSize;
    numLives = gtv.numLives;
    maxLoc = gtv.maxLoc;
    operationBufferSize = gtv.operationBufferSize;
    locationBufferSize = gtv.locationBufferSize;
    valueBufferSize = gtv.valueBufferSize;
    taylorBufferSize = gtv.taylorBufferSize;
    maxNumberTaylorBuffers = gtv.maxNumberTaylorBuffers;
    inParallelRegion = gtv.inParallelRegion;
    newTape = gtv.newTape;
    branchSwitchWarning = gtv.branchSwitchWarning;
    currentTapeInfosPtr = gtv.currentTapeInfosPtr;
    initialStoreSize = gtv.initialStoreSize;
    store = new double[storeSize];
    memcpy(store, gtv.store, storeSize*sizeof(double));
#if defined(ADOLC_TRACK_ACTIVITY)
    actStore = new char[storeSize];
    memcpy(actStore, gtv.actStore, storeSize*sizeof(char));
#endif
    storeManagerPtr = new
        StoreManagerLocintBlock(
            dynamic_cast<StoreManagerLocintBlock*>(gtv.storeManagerPtr),
            store,
#if defined(ADOLC_TRACK_ACTIVITY)
            actStore,
#endif
            storeSize, numLives);
    paramStoreMgrPtr = new
        StoreManagerLocintBlock(
            dynamic_cast<StoreManagerLocintBlock*>(gtv.paramStoreMgrPtr),
            pStore, maxparam, numparam);
    return *this;
}

#if defined(ADOLC_TRACK_ACTIVITY)

char const* const StoreManagerLocint::nowhere = NULL;

StoreManagerLocint::StoreManagerLocint(double * &storePtr, char* &actStorePtr, size_t &size, size_t &numlives) :
    storePtr(storePtr),
    activityTracking(1),
    actStorePtr(actStorePtr),
    indexFree(0),
    head(0),
    maxsize(size), currentfill(numlives)
{
#ifdef ADOLC_DEBUG
    std::cerr << "StoreManagerInteger::StoreManagerInteger()\n";
#endif
}

StoreManagerLocint::StoreManagerLocint(const StoreManagerLocint *const stm,
				       double * &storePtr, char* &actStorePtr, size_t &size, size_t &numlives) :
    storePtr(storePtr),
    actStorePtr(actStorePtr),
    activityTracking(1),
    maxsize(size), currentfill(numlives)
{
#ifdef ADOLC_DEBUG
    std::cerr << "StoreManagerInteger::StoreManagerInteger()\n";
#endif
    head = stm->head;
    indexFree = new locint[maxsize];
    for (size_t i = 0; i < maxsize; i++)
	indexFree[i] = stm->indexFree[i];
}
#endif

StoreManagerLocint::StoreManagerLocint(double * &storePtr, size_t &size, size_t &numlives) : 
    storePtr(storePtr),
#if defined(ADOLC_TRACK_ACTIVITY)
    activityTracking(0),
    actStorePtr(const_cast<char*&>(nowhere)),
#endif
    indexFree(0),
    head(0),
    maxsize(size), currentfill(numlives)
{
#ifdef ADOLC_DEBUG
    std::cerr << "StoreManagerInteger::StoreManagerInteger()\n";
#endif
}

StoreManagerLocint::~StoreManagerLocint() 
{
#ifdef ADOLC_DEBUG
    std::cerr << "StoreManagerInteger::~StoreManagerInteger()\n";
#endif
    if (storePtr) {
	delete[] storePtr;
	storePtr = 0;
    }
    if (indexFree) {
	delete[] indexFree;
	indexFree = 0;
    }
#if defined(ADOLC_TRACK_ACTIVITY)
    if (activityTracking && actStorePtr) {
	delete[] actStorePtr;
    }
#endif
    maxsize = 0;
    currentfill = 0;
    head = 0;
}

StoreManagerLocint::StoreManagerLocint(const StoreManagerLocint *const stm,
				       double * &storePtr, size_t &size, size_t &numlives) : 
    storePtr(storePtr),
#if defined(ADOLC_TRACK_ACTIVITY)
    activityTracking(0),
    actStorePtr(const_cast<char*&>(nowhere)),
#endif
    maxsize(size), currentfill(numlives)
{
#ifdef ADOLC_DEBUG
    std::cerr << "StoreManagerInteger::StoreManagerInteger()\n";
#endif
    head = stm->head;
    indexFree = new locint[maxsize];
    for (size_t i = 0; i < maxsize; i++)
	indexFree[i] = stm->indexFree[i];
}

locint StoreManagerLocint::next_loc() {
    if (head == 0) {
      grow();
    }
    assert(head);
    locint const result = head;
    head = indexFree[head];
    ++currentfill;
#ifdef ADOLC_DEBUG
    std::cerr << "next_loc: " << result << " fill: " << size() << "max: " << maxSize() << endl;
#endif
    return result;
}

void StoreManagerLocint::free_loc(locint loc) {
    assert(0 < loc && loc < maxsize);
    indexFree[loc] = head;
    head = loc;
    --currentfill;
#ifdef ADOLC_DEBUG
    std::cerr << "free_loc: " << loc << " fill: " << size() << "max: " << maxSize() << endl;
#endif
}

void StoreManagerLocint::ensure_block(size_t n) {
    fprintf(DIAG_OUT,"ADOL-C error: Location block required from singleton location store");
    adolc_exit(-4,"ADOL-C error: Location blocks not alowed",__func__,__FILE__,__LINE__);
}

void StoreManagerLocint::grow(size_t mingrow) {
    if (maxsize == 0) maxsize += initialSize;
    size_t const oldMaxsize = maxsize;
    maxsize *= 2;
    if (maxsize < mingrow) maxsize = mingrow;

    if (maxsize > std::numeric_limits<locint>::max()) {
      // encapsulate this error message
      fprintf(DIAG_OUT,"\nADOL-C error:\n");
      fprintf(DIAG_OUT,"maximal number (%d) of live active variables exceeded\n\n", 
	      std::numeric_limits<locint>::max());
      adolc_exit(-3,"",__func__,__FILE__,__LINE__);
    }

#ifdef ADOLC_DEBUG
    std::cerr << "StoreManagerInteger::grow(): increase size from " << oldMaxsize
	 << " to " << maxsize << " entries (currently " << size() << " entries used)\n";
    assert(oldMaxsize == initialSize or size() == oldMaxsize);
#endif

    double *const oldStore = storePtr;
    locint *const oldIndex = indexFree;
#if defined(ADOLC_TRACK_ACTIVITY)
    char * oldactStore;
    if (activityTracking) {
	oldactStore = actStorePtr;
    }
#endif

#if defined(ADOLC_DEBUG)
    std::cerr << "StoreManagerInteger::grow(): allocate " << maxsize * sizeof(double) << " B doubles " 
	 << "and " << maxsize * sizeof(locint) << " B locints\n";
#endif
    storePtr = new double[maxsize];
    indexFree = new locint[maxsize];
#if defined(ADOLC_TRACK_ACTIVITY)
    if (activityTracking)
	actStorePtr = new char[maxsize];
#endif
    // we use index 0 as end-of-list marker
    size_t i = 1;
    storePtr[0] =  std::numeric_limits<double>::quiet_NaN();

    if (oldMaxsize != initialSize) { // not the first time
#if defined(ADOLC_DEBUG)
      std::cerr << "StoreManagerInteger::grow(): copy values\n";
#endif
      for (size_t j = i; j < oldMaxsize; ++j) {
	indexFree[j] = oldIndex[j];
      }
      for (size_t j = i; j < oldMaxsize; ++j) {
	storePtr[j] = oldStore[j];
      }
#if defined(ADOLC_TRACK_ACTIVITY)
      if (activityTracking) {
	  for (size_t j = i; j < oldMaxsize; ++j) {
	      actStorePtr[j] = oldactStore[j];
	  }
      }
#endif
      // reset i to start of new slots (upper half)
      i = oldMaxsize;

#if defined(ADOLC_DEBUG)
      std::cerr << "StoreManagerInteger::grow(): free " << oldMaxsize * sizeof(double)
		<< " + " << oldMaxsize * sizeof(locint) << " B\n";
#endif
      delete [] oldStore;
      delete [] oldIndex;
#if defined(ADOLC_TRACK_ACTIVITY)
      if (activityTracking)
	  delete [] oldactStore;
#endif
    }

    head = i;
    // create initial linked list for new slots
    for ( ; i < maxsize-1; ++i) {
      indexFree[i] = i + 1;
    }
    indexFree[i] = 0; // end marker
    assert(i == maxsize-1);
}


/****************************************************************************/
/* Returns the next free location in "adouble" memory.                      */
/****************************************************************************/
locint next_loc() {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  return ADOLC_GLOBAL_TAPE_VARS.storeManagerPtr->next_loc();
}

/****************************************************************************/
/* frees the specified location in "adouble" memory                         */
/****************************************************************************/
void free_loc(locint loc) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  ADOLC_GLOBAL_TAPE_VARS.storeManagerPtr->free_loc(loc);
}

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
    char *ptr, *end;

    ptr = (char *)(&newTapeInfos->tapeID);
    end = (char *)(&newTapeInfos->pTapeInfos);
    for ( ; ptr != end ; ptr++ )
        *ptr = 0;
}

/* as above but keep allocated buffers if possible */
void initTapeInfos_keep(TapeInfos *newTapeInfos) {
    unsigned char *opBuffer = newTapeInfos->opBuffer;
    locint *locBuffer = newTapeInfos->locBuffer;
    double *valBuffer = newTapeInfos->valBuffer;
    revreal *tayBuffer = newTapeInfos->tayBuffer;
    double *signature = newTapeInfos->signature;
    FILE *tay_file = newTapeInfos->tay_file;

    initTapeInfos(newTapeInfos);

    newTapeInfos->opBuffer = opBuffer;
    newTapeInfos->locBuffer = locBuffer;
    newTapeInfos->valBuffer = valBuffer;
    newTapeInfos->tayBuffer = tayBuffer;
    newTapeInfos->signature = signature;
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
#ifdef SPARSE
		freeSparseJacInfos(newTapeInfos->pTapeInfos.sJinfos.y,
				   newTapeInfos->pTapeInfos.sJinfos.B,
				   newTapeInfos->pTapeInfos.sJinfos.JP,
				   newTapeInfos->pTapeInfos.sJinfos.g,
				   newTapeInfos->pTapeInfos.sJinfos.jr1d,
				   newTapeInfos->pTapeInfos.sJinfos.seed_rows,
				   newTapeInfos->pTapeInfos.sJinfos.seed_clms,
				   newTapeInfos->pTapeInfos.sJinfos.depen);
		freeSparseHessInfos(newTapeInfos->pTapeInfos.sHinfos.Hcomp, 
				    newTapeInfos->pTapeInfos.sHinfos.Xppp, 
				    newTapeInfos->pTapeInfos.sHinfos.Yppp, 
				    newTapeInfos->pTapeInfos.sHinfos.Zppp, 
				    newTapeInfos->pTapeInfos.sHinfos.Upp, 
				    newTapeInfos->pTapeInfos.sHinfos.HP,
				    newTapeInfos->pTapeInfos.sHinfos.g, 
				    newTapeInfos->pTapeInfos.sHinfos.hr, 
				    newTapeInfos->pTapeInfos.sHinfos.p, 
				    newTapeInfos->pTapeInfos.sHinfos.indep);	
		newTapeInfos->pTapeInfos.sJinfos.B=NULL;
		newTapeInfos->pTapeInfos.sJinfos.y=NULL;
		newTapeInfos->pTapeInfos.sJinfos.g=NULL;
		newTapeInfos->pTapeInfos.sJinfos.jr1d=NULL;
		newTapeInfos->pTapeInfos.sJinfos.Seed=NULL;
		newTapeInfos->pTapeInfos.sJinfos.JP=NULL;
		newTapeInfos->pTapeInfos.sJinfos.depen=0;
		newTapeInfos->pTapeInfos.sJinfos.nnz_in=0;
		newTapeInfos->pTapeInfos.sJinfos.seed_rows=0;
		newTapeInfos->pTapeInfos.sJinfos.seed_clms=0;
		newTapeInfos->pTapeInfos.sHinfos.Zppp=NULL;
		newTapeInfos->pTapeInfos.sHinfos.Yppp=NULL;
		newTapeInfos->pTapeInfos.sHinfos.Xppp=NULL;
		newTapeInfos->pTapeInfos.sHinfos.Upp=NULL;
		newTapeInfos->pTapeInfos.sHinfos.Hcomp=NULL;
		newTapeInfos->pTapeInfos.sHinfos.HP=NULL;
		newTapeInfos->pTapeInfos.sHinfos.g=NULL;
		newTapeInfos->pTapeInfos.sHinfos.hr=NULL;
		newTapeInfos->pTapeInfos.sHinfos.nnz_in=0;
		newTapeInfos->pTapeInfos.sHinfos.indep=0;
		newTapeInfos->pTapeInfos.sHinfos.p=0;
#endif
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
        ADOLC_GLOBAL_TAPE_VARS.currentTapeInfosPtr->copy(
            ADOLC_CURRENT_TAPE_INFOS);
        ADOLC_TAPE_STACK.push(ADOLC_GLOBAL_TAPE_VARS.currentTapeInfosPtr);
    } else {
        ADOLC_CURRENT_TAPE_INFOS_FALLBACK.copy(
                ADOLC_CURRENT_TAPE_INFOS);
        ADOLC_TAPE_STACK.push(&ADOLC_CURRENT_TAPE_INFOS_FALLBACK);
    }
    if (newTI) ADOLC_TAPE_INFOS_BUFFER.push_back(newTapeInfos);

    newTapeInfos->pTapeInfos.skipFileCleanup=0;

    /* set the new tape infos as current */
    ADOLC_CURRENT_TAPE_INFOS.copy(*newTapeInfos);
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
                    ADOLC_GLOBAL_TAPE_VARS.currentTapeInfosPtr->copy(
                            ADOLC_CURRENT_TAPE_INFOS);
                    ADOLC_TAPE_STACK.push(
                            ADOLC_GLOBAL_TAPE_VARS.currentTapeInfosPtr);
                } else {
                    ADOLC_CURRENT_TAPE_INFOS_FALLBACK.copy(
                            ADOLC_CURRENT_TAPE_INFOS);
                    ADOLC_TAPE_STACK.push(&ADOLC_CURRENT_TAPE_INFOS_FALLBACK);
                }
                ADOLC_CURRENT_TAPE_INFOS.copy(**tiIter);
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
        ADOLC_GLOBAL_TAPE_VARS.currentTapeInfosPtr->copy(
                ADOLC_CURRENT_TAPE_INFOS);
        ADOLC_TAPE_STACK.push(ADOLC_GLOBAL_TAPE_VARS.currentTapeInfosPtr);
    } else {
        ADOLC_CURRENT_TAPE_INFOS_FALLBACK.copy(
                ADOLC_CURRENT_TAPE_INFOS);
        ADOLC_TAPE_STACK.push(&ADOLC_CURRENT_TAPE_INFOS_FALLBACK);
    }

    /* set the new tape infos as current */
    ADOLC_CURRENT_TAPE_INFOS.copy(*tempTapeInfos);
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

    ADOLC_GLOBAL_TAPE_VARS.currentTapeInfosPtr->copy(
            ADOLC_CURRENT_TAPE_INFOS);
    ADOLC_GLOBAL_TAPE_VARS.currentTapeInfosPtr = ADOLC_TAPE_STACK.top();
    ADOLC_CURRENT_TAPE_INFOS.copy(
            *ADOLC_GLOBAL_TAPE_VARS.currentTapeInfosPtr);
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
    tapeInfos->tapingComplete = 1;
    read_tape_stats(tapeInfos);
    return tapeInfos;
}

/****************************************************************************/
/* Set a trace to nested_ctx                                                */
/****************************************************************************/
void set_nested_ctx(short tag, char nested) {
    TapeInfos* tiInfos = getTapeInfos(tag);
    tiInfos->in_nested_ctx = nested;
}
/****************************************************************************/
/* Check whether a tape has been set to nested_ctx                          */
/****************************************************************************/
char currently_nested(short tag) {
    TapeInfos* tiInfos = getTapeInfos(tag);
    return tiInfos->in_nested_ctx;
}


void cachedTraceTags(std::vector<short>& result) {
    vector<TapeInfos *>::const_iterator tiIter;
    vector<short>::iterator tIdIter;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    result.resize(ADOLC_TAPE_INFOS_BUFFER.size());
    if (!ADOLC_TAPE_INFOS_BUFFER.empty()) {
        for(tiIter=ADOLC_TAPE_INFOS_BUFFER.begin(), tIdIter=result.begin();
            tiIter!=ADOLC_TAPE_INFOS_BUFFER.end();
            ++tiIter, ++tIdIter) {
            *tIdIter = (*tiIter)->tapeID;
        }
    }
}

#ifdef SPARSE
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
		// free memory of tape entry that had been used previously
		freeSparseJacInfos(tapeInfos->pTapeInfos.sJinfos.y,
                        tapeInfos->pTapeInfos.sJinfos.B,
                        tapeInfos->pTapeInfos.sJinfos.JP,
                        tapeInfos->pTapeInfos.sJinfos.g,
			tapeInfos->pTapeInfos.sJinfos.jr1d,
			tapeInfos->pTapeInfos.sJinfos.seed_rows,
			tapeInfos->pTapeInfos.sJinfos.seed_clms,
			tapeInfos->pTapeInfos.sJinfos.depen);
		tapeInfos->pTapeInfos.sJinfos.y=sJinfos.y;
		tapeInfos->pTapeInfos.sJinfos.Seed=sJinfos.Seed;
		tapeInfos->pTapeInfos.sJinfos.B=sJinfos.B;
		tapeInfos->pTapeInfos.sJinfos.JP=sJinfos.JP;
		tapeInfos->pTapeInfos.sJinfos.depen=sJinfos.depen;
		tapeInfos->pTapeInfos.sJinfos.nnz_in=sJinfos.nnz_in;
		tapeInfos->pTapeInfos.sJinfos.seed_clms=sJinfos.seed_clms;
		tapeInfos->pTapeInfos.sJinfos.seed_rows=sJinfos.seed_rows;
		tapeInfos->pTapeInfos.sJinfos.g=sJinfos.g;
		tapeInfos->pTapeInfos.sJinfos.jr1d=sJinfos.jr1d;
            }
        }
    }
}
#endif

#ifdef SPARSE
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
		// free memory of tape entry that had been used previously
                    freeSparseHessInfos(tapeInfos->pTapeInfos.sHinfos.Hcomp, 
                                        tapeInfos->pTapeInfos.sHinfos.Xppp, 
                                        tapeInfos->pTapeInfos.sHinfos.Yppp, 
                                        tapeInfos->pTapeInfos.sHinfos.Zppp, 
                                        tapeInfos->pTapeInfos.sHinfos.Upp, 
                                        tapeInfos->pTapeInfos.sHinfos.HP,
					tapeInfos->pTapeInfos.sHinfos.g, 
                                        tapeInfos->pTapeInfos.sHinfos.hr, 
                                        tapeInfos->pTapeInfos.sHinfos.p, 
                                        tapeInfos->pTapeInfos.sHinfos.indep);	
		    tapeInfos->pTapeInfos.sHinfos.Hcomp=sHinfos.Hcomp;
		    tapeInfos->pTapeInfos.sHinfos.Xppp=sHinfos.Xppp;
		    tapeInfos->pTapeInfos.sHinfos.Yppp=sHinfos.Yppp;
		    tapeInfos->pTapeInfos.sHinfos.Zppp=sHinfos.Zppp;
		    tapeInfos->pTapeInfos.sHinfos.Upp=sHinfos.Upp;
		    tapeInfos->pTapeInfos.sHinfos.HP=sHinfos.HP;
		    tapeInfos->pTapeInfos.sHinfos.indep=sHinfos.indep;
		    tapeInfos->pTapeInfos.sHinfos.nnz_in=sHinfos.nnz_in;
		    tapeInfos->pTapeInfos.sHinfos.p=sHinfos.p;
		    tapeInfos->pTapeInfos.sHinfos.g=sHinfos.g;
		    tapeInfos->pTapeInfos.sHinfos.hr=sHinfos.hr;
            }
        }
    }
}
#endif

static void init_lib() {
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

    ADOLC_GLOBAL_TAPE_VARS.maxLoc=1;
    for (uint i=0; i<sizeof(locint)*8-1; ++i) {
        ADOLC_GLOBAL_TAPE_VARS.maxLoc<<=1;
        ++ADOLC_GLOBAL_TAPE_VARS.maxLoc;
    }
    ADOLC_GLOBAL_TAPE_VARS.inParallelRegion = 0;
    ADOLC_GLOBAL_TAPE_VARS.currentTapeInfosPtr = NULL;
    ADOLC_GLOBAL_TAPE_VARS.branchSwitchWarning = 1;

    adolc_id.adolc_ver    = ADOLC_VERSION;
    adolc_id.adolc_sub    = ADOLC_SUBVERSION;
    adolc_id.adolc_lvl    = ADOLC_PATCHLEVEL;
    adolc_id.locint_size  = sizeof(locint);
    adolc_id.revreal_size = sizeof(revreal);
    adolc_id.address_size = sizeof(size_t);

    ADOLC_EXT_DIFF_FCTS_BUFFER.init(init_CpInfos);
    readConfigFile();
}

static void clearCurrentTape() {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    TapeInfos* tmpTapeInfos = new TapeInfos;

    ADOLC_CURRENT_TAPE_INFOS.copy(*tmpTapeInfos);
    ADOLC_CURRENT_TAPE_INFOS_FALLBACK.copy(*tmpTapeInfos);
    delete tmpTapeInfos;
}

/* does things like closing/removing temporary files, ... */
void cleanUp() {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    TapeInfos** tiIter;
    clearCurrentTape();
    while (!ADOLC_TAPE_INFOS_BUFFER.empty()) {
        tiIter = &ADOLC_TAPE_INFOS_BUFFER.back();
        ADOLC_TAPE_INFOS_BUFFER.pop_back();
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
            if ((*tiIter)->tay_file!=NULL && (*tiIter)->pTapeInfos.skipFileCleanup==0 ) {
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
	    if ((*tiIter)->signature != NULL)
	    {
		free((*tiIter)->signature);
		(*tiIter)->signature = NULL;
	    }
            if ((*tiIter)->tayBuffer != NULL)
            {
                free((*tiIter)->tayBuffer);
                (*tiIter)->tayBuffer = NULL;
            }

#ifdef SPARSE
	    freeSparseJacInfos((*tiIter)->pTapeInfos.sJinfos.y,
			       (*tiIter)->pTapeInfos.sJinfos.B,
			       (*tiIter)->pTapeInfos.sJinfos.JP,
			       (*tiIter)->pTapeInfos.sJinfos.g,
			       (*tiIter)->pTapeInfos.sJinfos.jr1d,
			       (*tiIter)->pTapeInfos.sJinfos.seed_rows,
			       (*tiIter)->pTapeInfos.sJinfos.seed_clms,
			       (*tiIter)->pTapeInfos.sJinfos.depen);
	    freeSparseHessInfos((*tiIter)->pTapeInfos.sHinfos.Hcomp, 
				(*tiIter)->pTapeInfos.sHinfos.Xppp, 
				(*tiIter)->pTapeInfos.sHinfos.Yppp, 
				(*tiIter)->pTapeInfos.sHinfos.Zppp, 
				(*tiIter)->pTapeInfos.sHinfos.Upp, 
				(*tiIter)->pTapeInfos.sHinfos.HP,
				(*tiIter)->pTapeInfos.sHinfos.g, 
				(*tiIter)->pTapeInfos.sHinfos.hr, 
				(*tiIter)->pTapeInfos.sHinfos.p, 
				(*tiIter)->pTapeInfos.sHinfos.indep);	
#endif

            /* remove "main" tape files if not all three have been written */
            int filesWritten = (*tiIter)->stats[OP_FILE_ACCESS] +
                (*tiIter)->stats[LOC_FILE_ACCESS] +
                (*tiIter)->stats[VAL_FILE_ACCESS];
            if ( (filesWritten > 0) && ((*tiIter)->pTapeInfos.keepTape == 0) && (*tiIter)->pTapeInfos.skipFileCleanup==0 )
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
            *tiIter = NULL;
        }
    }

    cp_clearStack();

    if (ADOLC_GLOBAL_TAPE_VARS.store != NULL) {
        delete[] ADOLC_GLOBAL_TAPE_VARS.store;
        ADOLC_GLOBAL_TAPE_VARS.store = NULL;
    }
    if (ADOLC_GLOBAL_TAPE_VARS.pStore != NULL) {
        delete[] ADOLC_GLOBAL_TAPE_VARS.pStore;
        ADOLC_GLOBAL_TAPE_VARS.pStore = NULL;
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
    clearTapeBaseNames();
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
#ifdef SPARSE
    freeSparseJacInfos(tapeInfos->pTapeInfos.sJinfos.y,
		       tapeInfos->pTapeInfos.sJinfos.B,
		       tapeInfos->pTapeInfos.sJinfos.JP,
		       tapeInfos->pTapeInfos.sJinfos.g,
		       tapeInfos->pTapeInfos.sJinfos.jr1d,
		       tapeInfos->pTapeInfos.sJinfos.seed_rows,
		       tapeInfos->pTapeInfos.sJinfos.seed_clms,
		       tapeInfos->pTapeInfos.sJinfos.depen);
    freeSparseHessInfos(tapeInfos->pTapeInfos.sHinfos.Hcomp, 
			tapeInfos->pTapeInfos.sHinfos.Xppp, 
			tapeInfos->pTapeInfos.sHinfos.Yppp, 
			tapeInfos->pTapeInfos.sHinfos.Zppp, 
			tapeInfos->pTapeInfos.sHinfos.Upp, 
			tapeInfos->pTapeInfos.sHinfos.HP,
			tapeInfos->pTapeInfos.sHinfos.g, 
			tapeInfos->pTapeInfos.sHinfos.hr, 
			tapeInfos->pTapeInfos.sHinfos.p, 
			tapeInfos->pTapeInfos.sHinfos.indep);	
#endif
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
#ifdef ADOLC_MEDIPACK_SUPPORT
    mediInitTape(tnum);
#endif
    ADOLC_CURRENT_TAPE_INFOS.keepTaylors=keepTaylors;
    ADOLC_CURRENT_TAPE_INFOS.stats[NO_MIN_MAX] =
	ADOLC_GLOBAL_TAPE_VARS.nominmaxFlag;
    if (keepTaylors!=0) ADOLC_CURRENT_TAPE_INFOS.deg_save=1;
    start_trace();
    take_stock();               /* record all existing adoubles on the tape */
    return retval;
}

int trace_on(short tnum, int keepTaylors,
        uint obs, uint lbs, uint vbs, uint tbs, int skipFileCleanup)
{
    int retval = 0;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    /* allocate memory for TapeInfos and update tapeStack */
    retval = initNewTape(tnum);
#ifdef ADOLC_MEDIPACK_SUPPORT
    mediInitTape(tnum);
#endif
    freeTapeResources(&ADOLC_CURRENT_TAPE_INFOS);
    ADOLC_CURRENT_TAPE_INFOS.stats[OP_BUFFER_SIZE] = obs;
    ADOLC_CURRENT_TAPE_INFOS.stats[LOC_BUFFER_SIZE] = lbs;
    ADOLC_CURRENT_TAPE_INFOS.stats[VAL_BUFFER_SIZE] = vbs;
    ADOLC_CURRENT_TAPE_INFOS.stats[TAY_BUFFER_SIZE] = tbs;
    ADOLC_CURRENT_TAPE_INFOS.keepTaylors=keepTaylors;
    ADOLC_CURRENT_TAPE_INFOS.stats[NO_MIN_MAX] =
	ADOLC_GLOBAL_TAPE_VARS.nominmaxFlag;
    ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.skipFileCleanup=skipFileCleanup;
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
    if (ADOLC_CURRENT_TAPE_INFOS.workMode != ADOLC_TAPING) {
	failAdditionalInfo1 = ADOLC_CURRENT_TAPE_INFOS.tapeID;
	fail(ADOLC_TAPING_NOT_ACTUALLY_TAPING);
    }
    ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.keepTape = flag;
    keep_stock();         /* copy remaining live variables + trace_flag = 0 */
    stop_trace(flag);
    cout.flush();
    ADOLC_CURRENT_TAPE_INFOS.tapingComplete = 1;
    ADOLC_CURRENT_TAPE_INFOS.workMode = ADOLC_NO_MODE;
    releaseTape();
}

bool isTaping() {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    return ADOLC_CURRENT_TAPE_INFOS.traceFlag != 0;
}

void checkInitialStoreSize(GlobalTapeVars *gtv) {
    if (gtv->initialStoreSize > 
        gtv->storeManagerPtr->initialSize)
        gtv->storeManagerPtr->grow(
            gtv->initialStoreSize);
}

/****************************************************************************/
/* A class for initialization/finalization and OpenMP handling              */
/****************************************************************************/
class Keeper {
    public:
        inline Keeper() {
            dummy = 0;
            init_lib();
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
#include <adolc/adolc_openmp.h>

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

	/* Use assignment operator instead of open coding
	 * this copies the store and the storemanager too
	 */
	ADOLC_GLOBAL_TAPE_VARS = *globalTapeVars_s;

        ADOLC_GLOBAL_TAPE_VARS.newTape = 0;
        ADOLC_CURRENT_TAPE_INFOS.tapingComplete = 1;
        ADOLC_GLOBAL_TAPE_VARS.currentTapeInfosPtr = NULL;
    } else {
        if (ADOLC_parallel_doCopy) {
            ADOLC_GLOBAL_TAPE_VARS.storeSize = globalTapeVars_s->storeSize;
            ADOLC_GLOBAL_TAPE_VARS.numLives = globalTapeVars_s->numLives;
	    
            ADOLC_GLOBAL_TAPE_VARS.branchSwitchWarning = globalTapeVars_s->branchSwitchWarning;

	    /* deleting the storemanager deletes the store too */
	    delete ADOLC_GLOBAL_TAPE_VARS.storeManagerPtr;

            ADOLC_GLOBAL_TAPE_VARS.store = new
                double[ADOLC_GLOBAL_TAPE_VARS.storeSize];
            memcpy(ADOLC_GLOBAL_TAPE_VARS.store, globalTapeVars_s->store,
                    ADOLC_GLOBAL_TAPE_VARS.storeSize * sizeof(double));
	    ADOLC_GLOBAL_TAPE_VARS.storeManagerPtr = new
		StoreManagerLocintBlock(
		    dynamic_cast<StoreManagerLocintBlock*>(globalTapeVars_s->storeManagerPtr),
		    ADOLC_GLOBAL_TAPE_VARS.store,
		    ADOLC_GLOBAL_TAPE_VARS.storeSize,
		    ADOLC_GLOBAL_TAPE_VARS.numLives);
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

TapeInfos::TapeInfos() : pTapeInfos() {
    initTapeInfos(this);
}

TapeInfos::TapeInfos(short _tapeID) : pTapeInfos() {
    initTapeInfos(this);
    tapeID = _tapeID;
    pTapeInfos.op_fileName = createFileName(tapeID, OPERATIONS_TAPE);
    pTapeInfos.loc_fileName = createFileName(tapeID, LOCATIONS_TAPE);
    pTapeInfos.val_fileName = createFileName(tapeID, VALUES_TAPE);
    pTapeInfos.tay_fileName = NULL;
}

void TapeInfos::copy(const TapeInfos& tInfos) {
    char *ptr, *end;
    char const* tIptr = (char const*)(&tInfos.tapeID);

    ptr = (char *)(&this->tapeID);
    end = (char *)(&this->pTapeInfos);
    for ( ; ptr != end ; ptr++, tIptr++ )
        *ptr = *tIptr;
    this->pTapeInfos.copy(tInfos.pTapeInfos);
}

PersistantTapeInfos::PersistantTapeInfos() {
    char *ptr = (char*)(&forodec_nax), *end = (char*)(&paramstore);
    for (; ptr != end ; ptr++ )
        *ptr = 0;
    paramstore = NULL;
}

void PersistantTapeInfos::copy(const PersistantTapeInfos& pTInfos) {
    char *ptr = (char*)(&this->forodec_nax), *end = (char*)(&this->paramstore);
    char const* pTIptr = (char const*)(&pTInfos.forodec_nax);
    for (; ptr != end ; ptr++, pTIptr++ )
        *ptr = *pTIptr;
    paramstore = pTInfos.paramstore;
}

PersistantTapeInfos::~PersistantTapeInfos() {
    if (jacSolv_nax) {
        free(jacSolv_ci);
        free(jacSolv_ri);
        myfree1(jacSolv_xold);
        myfreeI2(jacSolv_nax, jacSolv_I);
        myfree2(jacSolv_J);
        jacSolv_nax = 0;
    }
    if (forodec_nax) {
        myfree1(forodec_y);
        myfree1(forodec_z);
        myfree2(forodec_Z);
        forodec_nax = 0;
    }
    if (paramstore != NULL) {
        free(paramstore);
        paramstore = NULL;
    }
}

#if defined(ADOLC_TRACK_ACTIVITY)

char const* const StoreManagerLocintBlock::nowhere = NULL;

StoreManagerLocintBlock::StoreManagerLocintBlock(double * &storePtr, char* &actStorePtr, size_t &size, size_t &numlives) :
    storePtr(storePtr),
    actStorePtr(actStorePtr),
    activityTracking(1),
    maxsize(size),
    currentfill(numlives)
#ifdef ADOLC_LOCDEBUG
    ,ensure_blockCallsSinceLastConsolidateBlocks(0)
#endif
  {
    indexFree.clear();
#ifdef ADOLC_LOCDEBUG
    std::cerr << "StoreManagerIntegerBlock::StoreManagerIntegerBlock()\n";
#endif
}

StoreManagerLocintBlock::StoreManagerLocintBlock(
    const StoreManagerLocintBlock *const stm,
    double * &storePtr, char * &actStorePtr, size_t &size, size_t &numlives) :
    storePtr(storePtr),
#if defined(ADOLC_TRACK_ACTIVITY)
    actStorePtr(actStorePtr),
    activityTracking(1),
#endif
    maxsize(size),
    currentfill(numlives)
#ifdef ADOLC_LOCDEBUG
    ,ensure_blockCallsSinceLastConsolidateBlocks(0)
#endif
  {
#ifdef ADOLC_LOCDEBUG
    std::cerr << "StoreManagerInteger::StoreManagerInteger()\n";
#endif
    indexFree.clear();
    forward_list<struct FreeBlock>::const_iterator iter = stm->indexFree.begin();
    for (; iter != stm->indexFree.end(); iter++)
	indexFree.emplace_front( *iter );
}
#endif

StoreManagerLocintBlock::StoreManagerLocintBlock(double * &storePtr, size_t &size, size_t &numlives) :
    storePtr(storePtr),
#if defined(ADOLC_TRACK_ACTIVITY)
    activityTracking(0),
    actStorePtr(const_cast<char*&>(nowhere)),
#endif
    maxsize(size),
    currentfill(numlives)
#ifdef ADOLC_LOCDEBUG
    ,ensure_blockCallsSinceLastConsolidateBlocks(0)
#endif
  {
    indexFree.clear();
#ifdef ADOLC_LOCDEBUG
    std::cerr << "StoreManagerIntegerBlock::StoreManagerIntegerBlock()\n";
#endif
}

StoreManagerLocintBlock::~StoreManagerLocintBlock()
{
#ifdef ADOLC_LOCDEBUG
    std::cerr << "StoreManagerIntegerBlock::~StoreManagerIntegerBlock()\n";
#endif
    if (storePtr != NULL) {
     delete[] storePtr;
     storePtr = NULL;
    }
    if (!indexFree.empty() ) {
	indexFree.clear();
    }
#if defined(ADOLC_TRACK_ACTIVITY)
    if (activityTracking && actStorePtr) {
	delete[] actStorePtr;
    }
#endif
    maxsize = 0;
    currentfill = 0;
}

StoreManagerLocintBlock::StoreManagerLocintBlock(
    const StoreManagerLocintBlock *const stm,
    double * &storePtr, size_t &size, size_t &numlives) :
    storePtr(storePtr),
#if defined(ADOLC_TRACK_ACTIVITY)
    activityTracking(0),
    actStorePtr(const_cast<char*&>(nowhere)),
#endif
    maxsize(size),
    currentfill(numlives)
#ifdef ADOLC_LOCDEBUG
    ,ensure_blockCallsSinceLastConsolidateBlocks(0)
#endif
  {
#ifdef ADOLC_LOCDEBUG
    std::cerr << "StoreManagerInteger::StoreManagerInteger()\n";
#endif
    indexFree.clear();
    forward_list<struct FreeBlock>::const_iterator iter = stm->indexFree.begin();
    for (; iter != stm->indexFree.end(); iter++)
	indexFree.emplace_front( *iter );
}


locint StoreManagerLocintBlock::next_loc() {
    if ( indexFree.empty() )
	grow();

    struct FreeBlock &front = indexFree.front();
    locint const result = front.next;
    if (--front.size == 0) {
	if (next(indexFree.cbegin()) == indexFree.cend()) {
            front.next++;
	    grow();
	} else
          indexFree.pop_front();
    } else
        front.next++;

    ++currentfill;

#ifdef ADOLC_LOCDEBUG
    std::cerr << "StoreManagerLocintBlock::next_loc: result: " << result << " fill: " << size() << "max: " << maxSize() << endl;
    forward_list<struct FreeBlock>::iterator iter = indexFree.begin();
    for( ; iter != indexFree.end(); iter++ )
       std::cerr << "INDEXFELD ( " << iter->next << " , " << iter->size << ")" << endl;
#endif

    return result;
}

void StoreManagerLocintBlock::ensure_block(size_t n) {
    bool found = false;
#ifdef ADOLC_LOCDEBUG
    ++ensure_blockCallsSinceLastConsolidateBlocks;
    std::cerr << "StoreManagerLocintBlock::ensure_Block: required " << n << " ... ";
    std::cerr << "searching for big enough block " << endl;
#endif
    if (maxSize()-size()>n) {
      if (indexFree.front().size>=n) found = true;
      if ((!found) && ((double(maxSize())/double(size()))>gcTriggerRatio() || maxSize()>gcTriggerMaxSize())) {
        consolidateBlocks();
#ifdef ADOLC_LOCDEBUG
        std::cerr << "ADOLC: GC called consolidateBlocks because " << maxSize() << "/" << size() << ">" << gcTriggerRatio() << " or " << maxSize() << ">" << gcTriggerMaxSize() << " after " << ensure_blockCallsSinceLastConsolidateBlocks << std::endl;
        ensure_blockCallsSinceLastConsolidateBlocks=0;
#endif
        forward_list<struct FreeBlock>::iterator 
            biter = indexFree.before_begin(), 
            iter = indexFree.begin();
        for (; iter != indexFree.end() ; biter++, iter++ ) {
          if ( iter->size >= n) {
            if (iter != indexFree.begin() ) {
              indexFree.emplace_front(*iter);
              indexFree.erase_after(biter);
            }
            found = true;
            break;
          }
        }
      }
    }
    if (!found) {
#ifdef ADOLC_LOCDEBUG
	std::cerr << "no big enough block...growing " << endl;
#endif
	grow(n);
    }

#ifdef ADOLC_LOCDEBUG
    std::cerr << "StoreManagerLocintBlock::ensure_Block: " << " fill: " << size() << "max: " << maxSize() <<  " ensure_Block (" << n << ")" << endl;
    forward_list<struct FreeBlock>::iterator iter = indexFree.begin();
    for( ; iter != indexFree.end(); iter++ )
	std::cerr << "INDEXFELD ( " << iter->next << " , " << iter->size << ")" << endl;
#endif
}

void StoreManagerLocintBlock::grow(size_t minGrow) {
    // first figure out what eventual size we want
    size_t const oldMaxsize = maxsize;

    if (maxsize == 0){
        maxsize = initialSize;
    } else {
	maxsize *= 2;
    }

    if (minGrow > 0) {
	while (maxsize - oldMaxsize < minGrow) {
	    maxsize *= 2;
	}
    }

    if (maxsize > std::numeric_limits<locint>::max()) {
      // encapsulate this error message
      fprintf(DIAG_OUT,"\nADOL-C error:\n");
      fprintf(DIAG_OUT,"maximal number (%u) of live active variables exceeded\n\n",
           std::numeric_limits<locint>::max());
      adolc_exit(-3,"",__func__,__FILE__,__LINE__);
    }

#ifdef ADOLC_LOCDEBUG
    // index 0 is not used, means one slot less
    std::cerr << "StoreManagerIntegerBlock::grow(): increase size from " << oldMaxsize
      << " to " << maxsize << " entries (currently " << size() << " entries used)\n";
#endif

    double *const oldStore = storePtr;
#if defined(ADOLC_TRACK_ACTIVITY)
    char * oldactStore;
    if (activityTracking)
	oldactStore = actStorePtr;
#endif
#if defined(ADOLC_LOCDEBUG)
    std::cerr << "StoreManagerInteger::grow(): allocate " << maxsize * sizeof(double) << " B doubles\n";
#endif
    storePtr = new double[maxsize];
    assert(storePtr);
    memset(storePtr, 0, maxsize*sizeof(double));
#if defined(ADOLC_TRACK_ACTIVITY)
    if (activityTracking) {
	actStorePtr = new char[maxsize];
	memset(actStorePtr,0,maxsize*sizeof(char));
    }
#endif

    if (oldStore != NULL) { // not the first time
#if defined(ADOLC_LOCDEBUG)
      std::cerr << "StoreManagerInteger::grow(): copy values\n";
#endif

      memcpy(storePtr, oldStore, oldMaxsize*sizeof(double));
#if defined(ADOLC_TRACK_ACTIVITY)
      if (activityTracking) {
	  memcpy(actStorePtr, oldactStore, oldMaxsize*sizeof(char));
      }
#endif

#if defined(ADOLC_LOCDEBUG)
      std::cerr << "StoreManagerInteger::grow(): free " << oldMaxsize * sizeof(double) << "\n";
#endif
      delete [] oldStore;
#if defined(ADOLC_TRACK_ACTIVITY)
      if (activityTracking) {
	  delete[] oldactStore;
      }
#endif

    }

    bool foundTail = false;
    forward_list<struct FreeBlock>::iterator 
        biter = indexFree.before_begin(),
        iter = indexFree.begin();
    for (; iter != indexFree.end() ; biter++,iter++ ) {
         if (iter->next + iter->size == oldMaxsize ) {
	     iter->size += (maxsize - oldMaxsize);
	      indexFree.emplace_front(*iter);
	      indexFree.erase_after(biter);
	      foundTail = true;
	      break;
         }
    }

    if (! foundTail) {
	indexFree.emplace_front(
#if defined(_MSC_VER) && _MSC_VER <= 1800
		FreeBlock(
#endif
		oldMaxsize,(maxsize - oldMaxsize)
#if defined(_MSC_VER) && _MSC_VER <= 1800
		)
#endif
		);
    }

    biter = indexFree.before_begin();
    iter = indexFree.begin();
    while (iter != indexFree.end()) {
         if (iter->size == 0) {
             indexFree.erase_after(biter); // don't leave 0 blocks around
             iter = next(biter);
         }
	 else {
             biter++;
	     iter++;
         }
    }
#ifdef ADOLC_LOCDEBUG
    std::cerr << "Growing:" << endl;
    iter = indexFree.begin();
    for( ; iter != indexFree.end(); iter++ )
       std::cerr << "INDEXFELD ( " << iter->next << " , " << iter->size << ")" << endl;
#endif
}

void StoreManagerLocintBlock::free_loc(locint loc) {
    assert( loc < maxsize);

    struct FreeBlock &front = indexFree.front();
    if ((loc+1 == front.next)
        || (front.next + front.size == loc)) {
	front.size++;
	if (loc + 1 == front.next)
	    front.next = loc;
    }
    else {
         indexFree.emplace_front(
#if defined(_MSC_VER) && _MSC_VER <= 1800
		FreeBlock(
#endif
			 loc,1
#if defined(_MSC_VER) && _MSC_VER <= 1800
			 )
#endif
			 );
    }

    --currentfill;
#ifdef ADOLC_LOCDEBUG
    std::cerr << "free_loc: " << loc << " fill: " << size() << "max: " << maxSize() << endl;
    forward_list<struct FreeBlock>::iterator iter = indexFree.begin();
    for( ; iter != indexFree.end(); iter++ )
       std::cerr << "INDEXFELD ( " << iter->next << " , " << iter->size << ")" << endl;
#endif
}

void ensureContiguousLocations(size_t n) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    ADOLC_GLOBAL_TAPE_VARS.storeManagerPtr->ensure_block(n);
}

void setStoreManagerControl(double gcTriggerRatio, size_t gcTriggerMaxSize) {
  ADOLC_OPENMP_THREAD_NUMBER;
  ADOLC_OPENMP_GET_THREAD_NUMBER;
  ADOLC_GLOBAL_TAPE_VARS.storeManagerPtr->setStoreManagerControl(gcTriggerRatio,gcTriggerMaxSize);
}

void StoreManagerLocintBlock::consolidateBlocks() {
    indexFree.sort();
    forward_list<struct FreeBlock>::iterator 
        iter = indexFree.begin(),
        niter = iter++;
    while (iter != indexFree.end()) {
	if (niter->next + niter->size == iter->next) {
	    niter->size += iter->size;
	    indexFree.erase_after(niter);
	    iter = next(niter);
	} else {
	    niter++;
	    iter++;
	}
    }
#ifdef ADOLC_LOCDEBUG
    std::cerr << "StoreManagerLocintBlock::consolidateBlocks: " << " fill: " << size() << "max: " << maxSize() << endl;
    iter = indexFree.begin();
    for( ; iter != indexFree.end(); iter++ )
	std::cerr << "INDEXFELD ( " << iter->next << " , " << iter->size << ")" << endl;
#endif
}

void enableMinMaxUsingAbs() {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    if (!isTaping())
	ADOLC_GLOBAL_TAPE_VARS.nominmaxFlag = 1;
    else
	fprintf(DIAG_OUT, "ADOL-C warning: "
		"change from native Min/Max to using Abs during tracing "
		"will lead to inconsistent results, not changing behaviour now\n"
		"                "
		"call %s before trace_on(tape_id) for the correct behaviour\n"
		,__FUNCTION__);
}

void disableMinMaxUsingAbs() {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    if (!isTaping())
	ADOLC_GLOBAL_TAPE_VARS.nominmaxFlag = 0;
    else
	fprintf(DIAG_OUT, "ADOL-C warning: "
		"change from native Min/Max to using Abs during tracing "
		"will lead to inconsistent results, not changing behaviour now\n"
		"                "
		"call %s after trace_off() for the correct behaviour\n"
		,__FUNCTION__);
}

#include <adolc/adolc_fatalerror.h>

void adolc_exit(int errorcode, const char *what, const char* function, const char *file, int line) {
    throw FatalError(errorcode, what, function, file, line);
}

/* Only called during stop_trace() via save_params() */
void free_all_taping_params() {
    size_t np;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    np = ADOLC_CURRENT_TAPE_INFOS.stats[NUM_PARAM];
    while ( np > 0 )
        ADOLC_GLOBAL_TAPE_VARS.paramStoreMgrPtr->free_loc(--np);
}

void setStoreManagerType(unsigned char type) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    if (ADOLC_GLOBAL_TAPE_VARS.storeManagerPtr->storeType() != type) {
        if (ADOLC_GLOBAL_TAPE_VARS.numLives == 0) {
            ADOLC_GLOBAL_TAPE_VARS.reallocStore(type);
        } else {
            fprintf(DIAG_OUT,"ADOL-C-warning: called %s after allocating %d active variables\n"
                    "***  WILL NOT CHANGE ***\nto change type deallocate all active variables\n"
                    "continuing ...\n"
                    , __func__, ADOLC_GLOBAL_TAPE_VARS.numLives);
        }
    } else {
            fprintf(DIAG_OUT,"ADOL-C-warning: called %s with same type as before\n"
                    "***  NO CHANGE ***\ncontinuing ...\n",__func__);
    }
}

void GlobalTapeVarsCL::reallocStore(unsigned char type) {
    if (storeManagerPtr != NULL)
        delete storeManagerPtr;

    store = NULL;
#if defined(ADOLC_TRACK_ACTIVITY)
    actStore = NULL;
#endif
    storeSize = 0;
    numLives = 0;
    switch (type) {
        case ADOLC_LOCATION_BLOCKS:
#if defined(ADOLC_TRACK_ACTIVITY)
            storeManagerPtr = new StoreManagerLocintBlock(store, actStore, storeSize, numLives);
#else
            storeManagerPtr = new StoreManagerLocintBlock(store, storeSize, numLives);
#endif
            break;
        case ADOLC_LOCATION_SINGLETONS:
#if defined(ADOLC_TRACK_ACTIVITY)
            storeManagerPtr = new StoreManagerLocint(store, actStore, storeSize, numLives);
#else
            storeManagerPtr = new StoreManagerLocint(store, storeSize, numLives);
#endif
            break;
    }
}

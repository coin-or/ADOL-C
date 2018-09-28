/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     taping_p.h
 Revision: $Id$
 Contents: declarations for used by taping routines
 
 Copyright (c) Andreas Kowarz, Jean Utke

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#if !defined(ADOLC_TAPING_P_H)
#define ADOLC_TAPING_P_H 1

#ifdef __cplusplus
#include "storemanager.h"
#endif
#include <adolc/internal/common.h>
#include <adolc/taping.h>
#include <errno.h>

BEGIN_C_DECLS

enum WORKMODES {
    ADOLC_NO_MODE,

    ADOLC_FORWARD,
    ADOLC_ZOS_FORWARD,
    ADOLC_FOS_FORWARD,
    ADOLC_FOV_FORWARD,
    ADOLC_HOS_FORWARD,
    ADOLC_HOV_FORWARD,

    ADOLC_REVERSE,
    ADOLC_FOS_REVERSE,
    ADOLC_FOV_REVERSE,
    ADOLC_HOS_REVERSE,
    ADOLC_HOV_REVERSE,

    ADOLC_TAPING
};

/****************************************************************************/
/* Tape identification (ADOLC & version check)                              */
/****************************************************************************/
typedef struct {
    short adolc_ver;
    short adolc_sub;
    short adolc_lvl;
    short locint_size;
    short revreal_size;
    short address_size;
}
ADOLC_ID;

extern ADOLC_ID adolc_id;

/****************************************************************************/
/* tape types => used for file name generation                              */
/****************************************************************************/
enum TAPENAMES {
    LOCATIONS_TAPE,
    VALUES_TAPE,
    OPERATIONS_TAPE,
    TAYLORS_TAPE
};

/****************************************************************************/
/* Errors handled by fail(...)                                              */
/****************************************************************************/
enum ADOLC_ERRORS {
    ADOLC_MALLOC_FAILED,
    ADOLC_INTEGER_TAPE_FOPEN_FAILED,
    ADOLC_INTEGER_TAPE_FREAD_FAILED,
    ADOLC_VALUE_TAPE_FOPEN_FAILED,
    ADOLC_VALUE_TAPE_FREAD_FAILED,
    ADOLC_TAPE_TO_OLD,
    ADOLC_WRONG_LOCINT_SIZE,
    ADOLC_MORE_STAT_SPACE_REQUIRED,

    ADOLC_TAPING_BUFFER_ALLOCATION_FAILED,
    ADOLC_TAPING_TBUFFER_ALLOCATION_FAILED,
    ADOLC_TAPING_READ_ERROR_IN_TAYLOR_CLOSE,
    ADOLC_TAPING_TO_MANY_TAYLOR_BUFFERS,
    ADOLC_TAPING_TO_MANY_LOCINTS,
    ADOLC_TAPING_STORE_REALLOC_FAILED,
    ADOLC_TAPING_FATAL_IO_ERROR,
    ADOLC_TAPING_TAPE_STILL_IN_USE,
    ADOLC_TAPING_TAYLOR_OPEN_FAILED,

    ADOLC_EVAL_SEEK_VALUE_STACK,
    ADOLC_EVAL_OP_TAPE_READ_FAILED,
    ADOLC_EVAL_VAL_TAPE_READ_FAILED,
    ADOLC_EVAL_LOC_TAPE_READ_FAILED,
    ADOLC_EVAL_TAY_TAPE_READ_FAILED,

    ADOLC_REVERSE_NO_TAYLOR_STACK,
    ADOLC_REVERSE_COUNTS_MISMATCH,
    ADOLC_REVERSE_TAYLOR_COUNTS_MISMATCH,

    ADOLC_BUFFER_NULLPOINTER_FUNCTION,
    ADOLC_BUFFER_INDEX_TO_LARGE,

    ADOLC_EXT_DIFF_NULLPOINTER_STRUCT,
    ADOLC_EXT_DIFF_WRONG_TAPESTATS,
    ADOLC_EXT_DIFF_NULLPOINTER_FUNCTION,
    ADOLC_EXT_DIFF_NULLPOINTER_DIFFFUNC,
    ADOLC_EXT_DIFF_NULLPOINTER_ARGUMENT,
    ADOLC_EXT_DIFF_WRONG_FUNCTION_INDEX,
    ADOLC_EXT_DIFF_LOCATIONGAP,

    ADOLC_CHECKPOINTING_CPINFOS_NULLPOINTER,
    ADOLC_CHECKPOINTING_NULLPOINTER_ARGUMENT,
    ADOLC_CHECKPOINTING_NULLPOINTER_FUNCTION,
    ADOLC_CHECKPOINTING_NULLPOINTER_FUNCTION_DOUBLE,
    ADOLC_CHECKPOINTING_REVOLVE_IRREGULAR_TERMINATED,
    ADOLC_CHECKPOINTING_UNEXPECTED_REVOLVE_ACTION,
    ADOLC_WRONG_PLATFORM_32,
    ADOLC_WRONG_PLATFORM_64,
    ADOLC_TAPING_NOT_ACTUALLY_TAPING,
    ADOLC_VEC_LOCATIONGAP
};
/* additional infos fail can work with */
extern int failAdditionalInfo1;
extern int failAdditionalInfo2;
extern locint failAdditionalInfo3;
extern locint failAdditionalInfo4;
extern void *failAdditionalInfo5;
extern void *failAdditionalInfo6;

/****************************************************************************/
/* tape information                                                         */
/****************************************************************************/

#ifdef SPARSE
typedef struct SparseJacInfos {
  void *g;
  void *jr1d;

  double *y;
  double **Seed;
  double **B;

  unsigned int **JP;

  int depen, nnz_in, seed_clms, seed_rows;
} SparseJacInfos;

typedef struct SparseHessInfos {
    void *g;
    void *hr;

    double **Hcomp;
    double*** Xppp;
    double*** Yppp;
    double*** Zppp;
    double**  Upp;
  
    unsigned int **HP;

  int nnz_in, indep, p;
} SparseHessInfos;
#endif

typedef struct PersistantTapeInfos { /* survive tape re-usage */
    int forodec_nax, forodec_dax;
    double *forodec_y, *forodec_z, **forodec_Z;
    double **jacSolv_J;
    double **jacSolv_I;
    double *jacSolv_xold;
    int *jacSolv_ri;
    int *jacSolv_ci;
    int jacSolv_nax, jacSolv_modeold, jacSolv_cgd;

#ifdef SPARSE
    /* sparse Jacobian matrices */

    SparseJacInfos sJinfos;

    /* sparse Hessian matrices */

    SparseHessInfos sHinfos;
#endif

    /* file names */

    char *op_fileName;
    char *loc_fileName;
    char *val_fileName;
    char *tay_fileName;

    int keepTape; /* - remember if tapes shall be written out to disk
                     - this information can only be given at taping time and
                       must survive all other actions on the tape */

    /**
     * defaults to 0, if 1 skips file removal (when file operations are costly)
     */
    int skipFileCleanup;

    revreal *paramstore;
#ifdef __cplusplus
    PersistantTapeInfos();
    ~PersistantTapeInfos();
    void copy(const PersistantTapeInfos&);
#endif
} PersistantTapeInfos;

/**
 * maximal number of locations writen per op code 
 */
#if defined(__USE_ISOC99)
extern const int maxLocsPerOp;
#else
#define maxLocsPerOp 10
#endif

typedef struct TapeInfos {
    short tapeID;
    int inUse;
    uint numInds;
    uint numDeps;
    int keepTaylors;             /* == 1 - write taylor stack in taping mode */
    size_t stats[STAT_SIZE];
    int traceFlag;
    char tapingComplete;

    /* operations tape */
    FILE *op_file;              /* file descriptor */
    unsigned char *opBuffer;    /* pointer to the current tape buffer */
    unsigned char *currOp;      /* pointer to the current opcode */
    unsigned char *lastOpP1;    /* pointer to element following the buffer */
    size_t numOps_Tape;           /* overall number of opcodes */
    size_t num_eq_prod;           /* overall number of eq_*_prod for nlf */

    /* values (real) tape */
    FILE *val_file;
    double *valBuffer;
    double *currVal;
    double *lastValP1;
    size_t numVals_Tape;

    /* locations tape */
    FILE *loc_file;
    locint *locBuffer;
    locint *currLoc;
    locint *lastLocP1;
    size_t numLocs_Tape;

    /* taylor stack tape */
    FILE *tay_file;
    revreal *tayBuffer;
    revreal *currTay;
    revreal *lastTayP1;
    size_t numTays_Tape;
    int nextBufferNumber;                   /* the next Buffer to read back */
    char lastTayBlockInCore;      /* == 1 if last taylor buffer is still in
                                            in core (first call of reverse) */
    double **T_for;                          /* derivative buffer - forward */
    uint deg_save;                 /* degree to save and saved respectively */
    uint tay_numInds;             /* # of independents for the taylor stack */
    uint tay_numDeps;               /* # of dependents for the taylor stack */

    /* checkpointing */
    locint lowestXLoc_for;     /* location of the first ind. - forward mode */
    locint lowestYLoc_for;     /* location of the first dep. - forward mode */
    locint lowestXLoc_rev;     /* location of the first ind. - reverse mode */
    locint lowestYLoc_rev;     /* location of the first dep. - reverse mode */
    locint cpIndex;               /* index of the curr. cp function <- tape */
    int numDirs_rev;     /* # of directions for **v_reverse (checkpointing) */

    locint *lowestXLoc_ext_v2;
    locint *lowestYLoc_ext_v2;

    /* evaluation forward */
    double *dp_T0;
    int gDegree, numTay;
    enum WORKMODES workMode;
    /*
     * Taylor coefficient array  allocated like this:
     * dpp_T[ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES][numTay*gDegree]
     */
    double **dpp_T;

    /* evaluation reverse */
    revreal *rp_T;
    revreal **rpp_T;
    revreal *rp_A;
    revreal **rpp_A;
    unsigned long int **upp_A;

    /* extern diff. fcts */
    locint ext_diff_fct_index;    /* set by forward and reverse (from tape) */
    char in_nested_ctx;

    size_t numSwitches;
    locint* switchlocs;
    double* signature;

    PersistantTapeInfos pTapeInfos;

#if defined(__cplusplus)
    TapeInfos();
    TapeInfos(short tapeID);
    ~TapeInfos() {}
    void copy(const TapeInfos&);
#endif
}
TapeInfos;

typedef struct GlobalTapeVarsCL {
    double* store;              /* double store for calc. while taping */
#if defined(ADOLC_TRACK_ACTIVITY)
    char* actStore;              /* activity store for tracking while taping */
#endif
    size_t storeSize;
    size_t numLives;
    locint maxLoc;

    locint operationBufferSize; /* Defaults to the value specified in */
    locint locationBufferSize;  /* usrparms.h. May be overwritten by values */
    locint valueBufferSize;     /* in a local config file .adolcrc. */
    locint taylorBufferSize;
    int maxNumberTaylorBuffers;

    char inParallelRegion;       /* set to 1 if in an OpenMP parallel region */
    char newTape;               /* signals: at least one tape created (0/1) */
    char branchSwitchWarning;
    TapeInfos *currentTapeInfosPtr;
    uint nominmaxFlag;
    size_t numparam;
    size_t maxparam;
    double *pStore;
    size_t initialStoreSize;
#ifdef __cplusplus
    StoreManager *paramStoreMgrPtr;
    StoreManager *storeManagerPtr;
    GlobalTapeVarsCL();
    ~GlobalTapeVarsCL();
    const GlobalTapeVarsCL& operator=(const GlobalTapeVarsCL&);
    void reallocStore(unsigned char type);
#else
    void *paramStoreMgrPtr;
    void *storeManagerPtr;
#endif
}
GlobalTapeVars;

#if defined(_OPENMP)

extern int isParallel();

#define ADOLC_TAPE_INFOS_BUFFER_DECL *tapeInfosBuffer
#define ADOLC_TAPE_STACK_DECL *tapeStack
#define ADOLC_CURRENT_TAPE_INFOS_DECL *currentTapeInfos
#define ADOLC_CURRENT_TAPE_INFOS_FALLBACK_DECL *currentTapeInfos_fallBack
#define ADOLC_GLOBAL_TAPE_VARS_DECL *globalTapeVars
#define ADOLC_EXT_DIFF_FCTS_BUFFER_DECL *ADOLC_extDiffFctsBuffer
#define ADOLC_CHECKPOINTS_STACK_DECL *ADOLC_checkpointsStack

#define ADOLC_OPENMP_THREAD_NUMBER int ADOLC_threadNumber
#if defined(ADOLC_THREADSAVE_ERRNO)
#define ADOLC_OPENMP_GET_THREAD_NUMBER ADOLC_threadNumber = errno
#define ADOLC_OPENMP_RESTORE_THREAD_NUMBER errno = ADOLC_threadNumber
#else
#define ADOLC_OPENMP_GET_THREAD_NUMBER ADOLC_threadNumber = omp_get_thread_num()
#define ADOLC_OPENMP_RESTORE_THREAD_NUMBER
#endif

#define ADOLC_TAPE_INFOS_BUFFER tapeInfosBuffer[ADOLC_threadNumber]
#define ADOLC_TAPE_STACK tapeStack[ADOLC_threadNumber]
#define ADOLC_CURRENT_TAPE_INFOS currentTapeInfos[ADOLC_threadNumber]
#define ADOLC_CURRENT_TAPE_INFOS_FALLBACK currentTapeInfos_fallBack[ADOLC_threadNumber]
#define ADOLC_GLOBAL_TAPE_VARS globalTapeVars[ADOLC_threadNumber]
#define ADOLC_EXT_DIFF_FCTS_BUFFER ADOLC_extDiffFctsBuffer[ADOLC_threadNumber]
#define ADOLC_CHECKPOINTS_STACK ADOLC_checkpointsStack[ADOLC_threadNumber]
#define REVOLVE_NUMBERS revolve_numbers[ADOLC_threadNumber]

#else

#define ADOLC_TAPE_INFOS_BUFFER_DECL tapeInfosBuffer
#define ADOLC_TAPE_STACK_DECL tapeStack
#define ADOLC_CURRENT_TAPE_INFOS_DECL currentTapeInfos
#define ADOLC_CURRENT_TAPE_INFOS_FALLBACK_DECL currentTapeInfos_fallBack
#define ADOLC_GLOBAL_TAPE_VARS_DECL globalTapeVars
#define ADOLC_EXT_DIFF_FCTS_BUFFER_DECL ADOLC_extDiffFctsBuffer
#define ADOLC_CHECKPOINTS_STACK_DECL ADOLC_checkpointsStack

#define ADOLC_OPENMP_THREAD_NUMBER
#define ADOLC_OPENMP_GET_THREAD_NUMBER
#define ADOLC_OPENMP_RESTORE_THREAD_NUMBER

#define ADOLC_TAPE_INFOS_BUFFER tapeInfosBuffer
#define ADOLC_TAPE_STACK tapeStack
#define ADOLC_CURRENT_TAPE_INFOS currentTapeInfos
#define ADOLC_CURRENT_TAPE_INFOS_FALLBACK currentTapeInfos_fallBack
#define ADOLC_GLOBAL_TAPE_VARS globalTapeVars
#define ADOLC_EXT_DIFF_FCTS_BUFFER ADOLC_extDiffFctsBuffer
#define ADOLC_CHECKPOINTS_STACK ADOLC_checkpointsStack
#define REVOLVE_NUMBERS revolve_numbers

#endif /* _OPENMP */

extern TapeInfos ADOLC_CURRENT_TAPE_INFOS_DECL;
extern TapeInfos ADOLC_CURRENT_TAPE_INFOS_FALLBACK_DECL;
extern GlobalTapeVars ADOLC_GLOBAL_TAPE_VARS_DECL;

/****************************************************************************/
/* C Function interfaces                                                    */
/****************************************************************************/

int initNewTape(short tapeID);
/* initializes a new tape
 * - returns 0 on success
 * - returns 1 in case tapeID is already/still in use */

void openTape(short tapeID, char mode);
/* opens an existing tape or creates a new one */

void releaseTape();
/* release the current tape and give control to the previous one
 * if keepVS is not zero (keep value stack for reverse) => belonging TapeInfos
 * are kept marked as being in use */

TapeInfos *getTapeInfos(short tapeID);
/* updates the tape infos for the given ID - a tapeInfos struct is created
 * and registered if non is found but its state will remain "not in use" */

#ifdef SPARSE
void setTapeInfoJacSparse(short tapeID, SparseJacInfos sJinfos);
/* updates the tape infos on sparse Jac for the given ID */

void setTapeInfoHessSparse(short tapeID, SparseHessInfos sHinfos);
/* updates the tape infos n sparse Hess for the given ID */
#endif

void take_stock();
/* record all existing adoubles on the tape
 * - intended to be used in start_trace only */

locint keep_stock();
/* record all remaining live variables on the value stack tape
 * - turns off trace_flag
 * - intended to be used in stop_trace only */

void updateLocs();

locint next_loc();
/* returns the next free location in "adouble" memory */

void free_loc(locint loc);
/* frees the specified location in "adouble" memory */

void taylor_begin(uint bufferSize, int degreeSave);
/* set up statics for writing taylor data */

void taylor_close(uint buffer);
/* close taylor file if necessary and refill buffer if possible */

void taylor_back(short tag, int* dep, int* ind, int* degree);
/* initializes a reverse sweep */

void write_taylor(locint loc, int keep);
/* writes the block of size depth of taylor coefficients from point loc to
 * the taylor buffer, if the buffer is filled, then it is written to the
 * taylor tape */

void write_taylors(locint loc, int keep, int degree, int numDir);
/* writes the block of size depth of taylor coefficients from point loc to
 * the taylor buffer, if the buffer is filled, then it is written to the
 * taylor tape */

#define ADOLC_WRITE_SCAYLOR(X) \
    {\
        if (ADOLC_CURRENT_TAPE_INFOS.currTay == ADOLC_CURRENT_TAPE_INFOS.lastTayP1)\
            put_tay_block(ADOLC_CURRENT_TAPE_INFOS.lastTayP1);\
        *ADOLC_CURRENT_TAPE_INFOS.currTay = (X);\
        ++ADOLC_CURRENT_TAPE_INFOS.currTay;\
    }
/* writes a single element (x) to the taylor buffer and writes the buffer to
 * disk if necessary */

void write_scaylors(revreal *x, uint size);
/* write_scaylors writes # size elements from x to the taylor buffer */

#define ADOLC_OVERWRITE_SCAYLOR(X,Y) \
    {\
        *Y = *(ADOLC_CURRENT_TAPE_INFOS.currTay - 1);\
        *(ADOLC_CURRENT_TAPE_INFOS.currTay - 1) = X;\
    }
/* overwrites the last (single) element (x) of the taylor buffer */

#define ADOLC_DELETE_SCAYLOR(X) \
    {\
        --ADOLC_CURRENT_TAPE_INFOS.currTay;\
        *X = *ADOLC_CURRENT_TAPE_INFOS.currTay;\
    }
/* deletes the last (single) element (x) of the taylor buffer */

void put_tay_block(revreal *lastValP1);
/* writes the taylor stack buffer onto hard disk */

#define ADOLC_GET_TAYLOR(X) \
    {\
        if (ADOLC_CURRENT_TAPE_INFOS.currTay == ADOLC_CURRENT_TAPE_INFOS.tayBuffer)\
            get_tay_block_r();\
        --ADOLC_CURRENT_TAPE_INFOS.currTay;\
        ADOLC_CURRENT_TAPE_INFOS.rp_T[X] = *ADOLC_CURRENT_TAPE_INFOS.currTay;\
    }
/* puts a taylor value from the value stack buffer to the taylor buffer */

void get_taylors(locint loc, int degree);
/* puts a block of taylor coefficients from the value stack buffer to the
 * taylor buffer --- Higher Order Scalar */

void get_taylors_p(locint loc, int degree, int numDir);
/* puts a block of taylor coefficients from the value stack buffer to the
 * taylor buffer --- Higher Order Vector */

void get_tay_block_r();
/* gets the next (previous block) of the value stack */



void initTapeBuffers();
/* free/allocate memory for buffers, initialize pointers */

void start_trace();
/* initialization for the taping process -> buffer allocation, sets
 * files names, and calls appropriate setup routines */

void stop_trace(int flag);
/* stop Tracing, clean up, and turn off trace_flag */

void close_tape(int flag);
/* close open tapes, update stats and clean up */

void freeTapeResources(TapeInfos *tapeInfos);
/* free all resources used by a tape before overwriting the tape */

void read_tape_stats(TapeInfos *tapeInfos);
/* does the actual reading from the hard disk into the stats buffer */

void init_for_sweep(short tag);
/* initialize a forward sweep, get stats, open tapes, fill buffers, ... */

void init_rev_sweep(short tag);
/* initialize a reverse sweep, get stats, open tapes, fill buffers, ... */

void end_sweep();
/* finish a forward or reverse sweep */



void fail(int error);
/* outputs an appropriate error message using DIAG_OUT and exits the running
 * program */

/* print an error message describing the error number */
void printError();

char *createFileName(short tapeID, int tapeType);
/* create file name depending on tape type and number */



/* puts an operation into the operation buffer, ensures that location buffer
 * and constants buffer are prepared to take the belonging stuff */
void put_op_reserve(unsigned char op, unsigned int reserveExtraLocations);
#define put_op(i) put_op_reserve((i),0)

void put_op_block(unsigned char *lastOpP1);
/* writes a block of operations onto hard disk and handles file creation,
 * removal, ... */

void get_op_block_f();
/* reads the next operations block into the internal buffer */

void get_op_block_r();
/* reads the previous block of operations into the internal buffer */

#define ADOLC_PUT_LOCINT(X) \
    {\
        *ADOLC_CURRENT_TAPE_INFOS.currLoc = X;\
        ++ADOLC_CURRENT_TAPE_INFOS.currLoc;\
    }
/* puts a single locations into the location buffer, no disk access */

void put_loc_block(locint *lastLocP1);
/* writes a block of locations onto hard disk and handles file creation,
 * removal, ... */

void get_loc_block_f();
/* reads the next block of locations into the internal buffer */

void get_loc_block_r();
/* reads the previous block of locations into the internal buffer */

#define ADOLC_PUT_VAL(X) \
    {\
        *ADOLC_CURRENT_TAPE_INFOS.currVal = X;\
        ++ADOLC_CURRENT_TAPE_INFOS.currVal;\
    }
/* puts a single constant into the location buffer, no disk access */

void put_vals_writeBlock(double *reals, locint numReals);
/* fill the constants buffer and write it to disk */

void put_vals_notWriteBlock(double *reals, locint numReals);
/* write some constants to the buffer without disk access */

void put_val_block(double *lastValP1);
/* writes a block of constants (real) onto hard disk and handles file
 * creation, removal, ... */

void get_val_block_f();
/* reads the next block of constants into the internal buffer */

void get_val_block_r();
/* reads the previous block of constants into the internal buffer */

locint get_val_space(void);
/* returns the number of free constants in the real tape, ensures that it
 * is at least 5 */

double *get_val_v_f(locint size);
/* return a pointer to the first element of a constants vector
 * -- Forward Mode -- */

double *get_val_v_r(locint size);
/* return a pointer to the first element of a constants vector
 * -- Reverse Mode -- */



/* suspicious function, maybe for vector class - kept for compatibility */
void reset_val_r();

/* updates */
int upd_resloc(locint temp, locint lhs);

int upd_resloc_check(locint temp, locint lhs);

int upd_resloc_inc_prod(locint temp, locint newlhs, unsigned char newop);

/* special IEEE values */
double make_nan();

double make_inf();



#if !defined(ADOLC_HARDDEBUG)
/*--------------------------------------------------------------------------*/
/*                                                        MACRO or FUNCTION */
#define get_op_f() *ADOLC_CURRENT_TAPE_INFOS.currOp++
#define get_op_r() *--ADOLC_CURRENT_TAPE_INFOS.currOp

#define get_locint_f() *ADOLC_CURRENT_TAPE_INFOS.currLoc++
#define get_locint_r() *--ADOLC_CURRENT_TAPE_INFOS.currLoc

#define get_val_f() *ADOLC_CURRENT_TAPE_INFOS.currVal++
#define get_val_r() *--ADOLC_CURRENT_TAPE_INFOS.currVal
#else /* HARDDEBUG */
unsigned char get_op_f();
unsigned char get_op_r();

locint get_locint_f();
locint get_locint_r();

double get_val_f();
double get_val_r();
#endif

/* tries to read a local config file containing, e.g., buffer sizes */
void readConfigFile();

void checkInitialStoreSize(GlobalTapeVars *gtv);

/* clear the tapeBaseNames that were alocated above in readConfigFile() */
void clearTapeBaseNames();

/****************************************************************************/
/* This function sets the flag "newTape" if either a taylor buffer has been */
/* created or a taping process has been performed. Calling the function is  */
/* also useful to "convince" the linker of including the cleaner part into  */
/* the binary when linking statically!                                      */
/****************************************************************************/
void markNewTape();

/****************************************************************************/
/* Allows us to throw an exception instead of calling exit() in case of a   */
/* irrecoverable error                                                      */
/****************************************************************************/
void adolc_exit(int errorcode, const char *what, const char *function, const char* file, int line);

/****************************************************************************/
/* Discards parameters from the end of value tape during reverse mode       */
/****************************************************************************/
void discard_params_r();

/****************************************************************************/
/* Frees parameter indices after taping is complete                         */
/****************************************************************************/
void free_all_taping_params();

END_C_DECLS

/****************************************************************************/
/* That's all                                                               */
/****************************************************************************/

#endif /* ADOLC_TAPING_P_H */



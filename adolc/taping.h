/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     taping.h
 Revision: $Id: taping.h 295 2009-02-25 13:32:25Z awalther $
 Contents: all C functions directly accessing at least one of the four tapes
           (operations, locations, constants, value stack)

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
----------------------------------------------------------------------------*/
#if !defined(ADOLC_TAPING_H)
#define ADOLC_TAPING_H 1

#include <adolc/common.h>

BEGIN_C_DECLS

enum StatEntries {
    NUM_INDEPENDENTS,                          /* # of independent variables */
    NUM_DEPENDENTS,                              /* # of dependent variables */
    NUM_MAX_LIVES,                                /* max # of live variables */
    TAY_STACK_SIZE,               /* # of values in the taylor (value) stack */
    OP_BUFFER_SIZE,   /* # of operations per buffer == OBUFSIZE (usrparms.h) */
    NUM_OPERATIONS,                               /* overall # of operations */
    OP_FILE_ACCESS,                        /* operations file written or not */
    NUM_LOCATIONS,                                 /* overall # of locations */
    LOC_FILE_ACCESS,                        /* locations file written or not */
    NUM_VALUES,                                       /* overall # of values */
    VAL_FILE_ACCESS,                           /* values file written or not */
    LOC_BUFFER_SIZE,   /* # of locations per buffer == LBUFSIZE (usrparms.h) */
    VAL_BUFFER_SIZE,      /* # of values per buffer == CBUFSIZE (usrparms.h) */
    TAY_BUFFER_SIZE,     /* # of taylors per buffer <= TBUFSIZE (usrparms.h) */
    STAT_SIZE                     /* represents the size of the stats vector */
};

enum TapeRemovalType {
    ADOLC_REMOVE_FROM_CORE,
    ADOLC_REMOVE_COMPLETELY
};

/* Returns statistics on the tape "tag". Use enumeration StatEntries for
 * accessing the individual elements of the vector "tape_stats"! */
void tapestats(short tag, int *tape_stats);

/* An all-in-one tape stats printing routine */
void printTapeStats(FILE *stream, short tag);

int removeTape(short tapeID, short type);

void enableBranchSwitchWarnings();
void disableBranchSwitchWarnings();

END_C_DECLS

/**
 * Normally, theKeeper would take care of the initialization and finalization
 * of ADOL-C. However, some compilers do not include the keeper code when
 * linking. "initADOLC" should be called right after main(...), in this case.
 * "initADOLC" will not initialize memory, but is only necessary to reference 
 * "theKeeper", such that this static instance is used at least once. :-(
 */
void initADOLC();

#if defined(__cplusplus)

/* Initialization for the taping process. Creates buffers for this tape, sets
 * files names, and calls appropriate setup routines.
 * This functions return value is different from zero if a tape with with ID
 * tnum is available only in core. The old tape gets overwritten by the new 
 * one in this case. */
int trace_on(short tnum, int keepTaylors = 0);

/* special version including buffersize customization
 *      obs - size of the operation buffer (number of elements)
 *      lbs - size of the location buffer (number of elements)
 *      vbs - size of the value buffer (number of elements)
 *      tbs - size of the taylor buffer (number of elements)
 * trace_on is the last point in time we want to allow the change of buffer
 * sizes for a given tape */
int trace_on(short tnum, int keepTaylors,
        uint obs, uint lbs, uint vbs, uint tbs);

/* Stop Tracing. Cleans up, and turns off trace_flag. Flag not equal zero
 * enforces writing of the three main tape files (op+loc+val). */
void trace_off(int flag = 0);

bool isTaping();

#endif

#endif /* ADOLC_TAPING_H */

/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     taping.c
 Revision: $Id$
 Contents: all C functions directly accessing at least one of the four tapes
           (operations, locations, constants, value stack)

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel,
               Kshitij Kulshreshtha
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
----------------------------------------------------------------------------*/

#include <math.h>
#include <string.h>

#include "oplate.h"
#include "taping_p.h"
#include "dvlparms.h"

#include <sys/types.h>
#include <sys/stat.h>

#ifdef ADOLC_AMPI_SUPPORT
#include "ampi/ampi.h"
#include "ampi/tape/support.h"
#endif

#include <adolc/param.h>

#if defined(_WINDOWS) && !__STDC__
#define stat _stat
#define S_IFDIR _S_IFDIR
#define S_IFMT _S_IFMT
#endif

#ifndef S_ISDIR
#define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)
#endif

/*--------------------------------------------------------------------------*/
/* Tape identification (ADOLC & version check) */
ADOLC_ID adolc_id;
/* first version with new tape structure
 * => to work with older tapes use older ADOL-C version */
#define ADOLC_NEW_TAPE_VERSION 2
#define ADOLC_NEW_TAPE_SUBVERSION 5
#define ADOLC_NEW_TAPE_PATCHLEVEL 3

/****************************************************************************/
/****************************************************************************/
/* HELP FUNCTIONS                                                           */
/****************************************************************************/
/****************************************************************************/

/*--------------------------------------------------------------------------*/
/* additional infos used by fail()                                          */
int failAdditionalInfo1;
int failAdditionalInfo2;
locint failAdditionalInfo3;
locint failAdditionalInfo4;
void *failAdditionalInfo5;
void *failAdditionalInfo6;

/* outputs an appropriate error message using DIAG_OUT and exits the running
 * program */
void fail( int error ) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    switch (error) {
        case ADOLC_MALLOC_FAILED:
            fprintf(DIAG_OUT, "ADOL-C error: Memory allocation failed!\n");
            break;
        case ADOLC_INTEGER_TAPE_FOPEN_FAILED:
        case ADOLC_INTEGER_TAPE_FREAD_FAILED:
            fprintf(DIAG_OUT, "ADOL-C error: "
                    "reading integer tape number %d!\n",
                    failAdditionalInfo1);
            printError();
            break;
        case ADOLC_VALUE_TAPE_FOPEN_FAILED:
        case ADOLC_VALUE_TAPE_FREAD_FAILED:
            fprintf(DIAG_OUT, "ADOL-C error: "
                    "reading value tape number %d!\n",
                    failAdditionalInfo1);
            printError();
            break;
        case ADOLC_TAPE_TO_OLD:
            fprintf(DIAG_OUT, "ADOL-C error: "
                    "Used tape (%d) was written with ADOL-C version "
                    "older than %d.%d.%d\n", failAdditionalInfo1,
                    ADOLC_NEW_TAPE_VERSION, ADOLC_NEW_TAPE_SUBVERSION,
                    ADOLC_NEW_TAPE_PATCHLEVEL);
            fprintf(DIAG_OUT, "              "
                    "This is ADOL-C %d.%d.%d\n", ADOLC_VERSION,
                    ADOLC_SUBVERSION, ADOLC_PATCHLEVEL);
            break;
        case ADOLC_WRONG_LOCINT_SIZE:
            fprintf(DIAG_OUT, "ADOL-C error: Used tape (%d) was written with "
                    "locints of size %d, size %d required.\n",
                    ADOLC_CURRENT_TAPE_INFOS.tapeID, failAdditionalInfo1,
                    failAdditionalInfo2);
            break;
        case ADOLC_MORE_STAT_SPACE_REQUIRED:
            fprintf(DIAG_OUT, "ADOL-C error: Not enough space for stats!\n"
                    "              Please contact the ADOL-C team!\n");
            break;

        case ADOLC_TAPING_BUFFER_ALLOCATION_FAILED:
            fprintf(DIAG_OUT, "ADOL-C error: Cannot allocate tape buffers!\n");
            break;
        case ADOLC_TAPING_TBUFFER_ALLOCATION_FAILED:
            fprintf(DIAG_OUT, "ADOL-C error: Cannot allocate taylor buffer!\n");
            break;
        case ADOLC_TAPING_READ_ERROR_IN_TAYLOR_CLOSE:
            fprintf(DIAG_OUT, "ADOL-C error: Read error in taylor_close n= %d\n",
                    failAdditionalInfo1);
            break;
        case ADOLC_TAPING_TO_MANY_TAYLOR_BUFFERS:
            fprintf(DIAG_OUT, "ADOL-C error: To many taylor buffers!\n"
                    "              Increase ADOLC_GLOBAL_TAPE_VARS.maxNumberTaylorBuffers\n");
            break;
        case ADOLC_TAPING_TO_MANY_LOCINTS:
            fprintf(DIAG_OUT, "ADOL-C error: Maximal number (%d) of live active "
                    "variables exceeded!\n\n"
                    "Possible remedies :\n\n"
                    " 1. Use more automatic local variables and\n"
                    "    allocate/deallocate adoubles on free store\n"
                    "     in a strictly last in first out fashion\n\n"
                    " 2. Extend the range by redefining the type of\n"
                    "    locint (currently %d byte) from unsigned short "
                    "(%d byte) or int\n"
                    "    to int (%d byte) or long (%d byte).\n",
                    failAdditionalInfo3, (int)sizeof(locint),
                    (int)sizeof(unsigned short), (int)sizeof(int),
                    (int)sizeof(long) );
            break;
        case ADOLC_TAPING_STORE_REALLOC_FAILED:
            fprintf(DIAG_OUT, "ADOL-C error: Failure to reallocate storage for "
                    "adouble values!\n\n"
                    "              oldStore     = %p\n"
                    "              newStore     = NULL\n"
                    "              oldStoreSize = %u\n"
                    "              newStoreSize = %u\n\n"
                    "Possible remedies :\n"
                    " 1. Use more automatic local variables and \n"
                    "    allocate/deallocate adoubles on free store\n"
                    "    in a strictly last in first out fashion\n"
                    " 2. Enlarge your system stacksize limit\n"
                    , failAdditionalInfo5, failAdditionalInfo3,
                      failAdditionalInfo4
                    );
            break;
        case ADOLC_TAPING_FATAL_IO_ERROR:
            fprintf(DIAG_OUT, "ADOL-C error: Fatal error-doing a read or "
                    "write!\n");
            printError();
            break;
        case ADOLC_TAPING_TAPE_STILL_IN_USE:
            fprintf(DIAG_OUT, "ADOL-C error: Tape %d is still in use!\n",
                    failAdditionalInfo1);
            break;
        case ADOLC_TAPING_TAYLOR_OPEN_FAILED:
            fprintf(DIAG_OUT, "ADOL-C error: while opening taylor file!\n");
            printError();
            break;

        case ADOLC_EVAL_SEEK_VALUE_STACK:
            fprintf(DIAG_OUT, "ADOL-C error: in seeking value stack file!\n");
            break;
        case ADOLC_EVAL_OP_TAPE_READ_FAILED:
            fprintf(DIAG_OUT, "ADOL-C error: while reading operations tape!\n");
            break;
        case ADOLC_EVAL_VAL_TAPE_READ_FAILED:
            fprintf(DIAG_OUT, "ADOL-C error: while reading values tape!\n");
            break;
        case ADOLC_EVAL_LOC_TAPE_READ_FAILED:
            fprintf(DIAG_OUT, "ADOL-C error: while reading locations tape!\n");
            break;
        case ADOLC_EVAL_TAY_TAPE_READ_FAILED:
            fprintf(DIAG_OUT, "ADOL-C error: while reading value stack tape!\n");
            break;

        case ADOLC_REVERSE_NO_TAYLOR_STACK:
            fprintf(DIAG_OUT, "ADOL-C error: No taylor stack found for tape "
                    "%d! => Check forward sweep!\n", failAdditionalInfo1);
            break;
        case ADOLC_REVERSE_COUNTS_MISMATCH:
            fprintf(DIAG_OUT, "ADOL-C error: Reverse sweep on tape %d aborted!\n"
                    "              Number of dependents(%u) and/or "
                    "independents(%u)\n"
                    "              variables passed to reverse is "
                    "inconsistent\n"
                    "              with number recorded on tape(%zu/%zu)!\n",
                    ADOLC_CURRENT_TAPE_INFOS.tapeID, failAdditionalInfo3,
                    failAdditionalInfo4,
                    ADOLC_CURRENT_TAPE_INFOS.stats[NUM_DEPENDENTS],
                    ADOLC_CURRENT_TAPE_INFOS.stats[NUM_INDEPENDENTS]);
            break;
        case ADOLC_REVERSE_TAYLOR_COUNTS_MISMATCH:
            fprintf(DIAG_OUT, "ADOL-C error: Reverse fails on tape %d because "
                    "the number of independent\n"
                    "              and/or dependent variables given to"
                    " reverse are inconsistent\n"
                    "              with that of the internal taylor "
                    "array!\n",
                    ADOLC_CURRENT_TAPE_INFOS.tapeID);
            break;

        case ADOLC_BUFFER_NULLPOINTER_FUNCTION:
            fprintf(DIAG_OUT, "ADOL-C error: NULL pointer supplied in buffer "
                    "handling.\n");
            break;
        case ADOLC_BUFFER_INDEX_TO_LARGE:
            fprintf(DIAG_OUT, "ADOL-C error: Index for buffer element too "
                    "large.\n");
            break;

        case ADOLC_EXT_DIFF_NULLPOINTER_STRUCT:
            fprintf(DIAG_OUT,
                    "ADOL-C error: Got null pointer as pointer to struct "
                    " containing ext. diff. function information!\n");
            break;
        case ADOLC_EXT_DIFF_WRONG_TAPESTATS:
            fprintf(DIAG_OUT,
                    "ADOL-C error: Number of independents/dependents recorded on"
                    " tape differ from number supplied by user!\n");
            break;
        case ADOLC_EXT_DIFF_NULLPOINTER_FUNCTION:
            fprintf(DIAG_OUT,
                    "ADOL-C error: Got NULL pointer as "
                    "extern function pointer!\n");
            break;
        case ADOLC_EXT_DIFF_NULLPOINTER_DIFFFUNC:
            fprintf(DIAG_OUT,
                    "ADOL-C error: No function for external differentiation found"
                    " to work with (null pointer)\n!");
            break;
        case ADOLC_EXT_DIFF_NULLPOINTER_ARGUMENT:
            fprintf(DIAG_OUT,
                    "ADOL-C error: Got at least one null pointer as argument to"
                    " extern differnetiated function!\n");
            break;
        case ADOLC_EXT_DIFF_WRONG_FUNCTION_INDEX:
            fprintf(DIAG_OUT,
                    "ADOL-C error: Function with specified index not found!\n");
            break;

        case ADOLC_EXT_DIFF_LOCATIONGAP:
          fprintf(DIAG_OUT,
                  "ADOL-C error: active type arguments passed to call_ext_fct do not have contiguous ascending locations; use ensureContiguousLocations(size_t) to reserve  contiguous blocks prior to allocation of the arguments.\n");
          break;

        case ADOLC_CHECKPOINTING_CPINFOS_NULLPOINTER:
            fprintf(DIAG_OUT,
                    "ADOL-C error: Got null pointer as pointer to struct "
                    " containing checkpointing information!\n");
            break;
        case ADOLC_CHECKPOINTING_NULLPOINTER_ARGUMENT:
            fprintf(DIAG_OUT,
                    "ADOL-C error: Got null pointer instead of argument pointer "
                    "within checkpointing infos!\n");
            break;
        case ADOLC_CHECKPOINTING_NULLPOINTER_FUNCTION:
            fprintf(DIAG_OUT,
                    "ADOL-C error: Got null pointer instead of function pointer "
                    "within checkpointing infos!\n");
            break;
        case ADOLC_CHECKPOINTING_NULLPOINTER_FUNCTION_DOUBLE:
            fprintf(DIAG_OUT,
                    "ADOL-C error: Got null pointer instead of function (double "
                    "version) pointer within checkpointing infos!\n");
            break;
        case ADOLC_CHECKPOINTING_REVOLVE_IRREGULAR_TERMINATED:
            fprintf(DIAG_OUT,
                    "ADOL-C error: Irregualar termination of REVOLVE!\n");
            break;
        case ADOLC_CHECKPOINTING_UNEXPECTED_REVOLVE_ACTION:
            fprintf(DIAG_OUT,
                    "ADOL-C error: Unextpected REVOLVE action in forward mode!\n"
                   );
            break;
	case ADOLC_WRONG_PLATFORM_32:
	    fprintf(DIAG_OUT,
		    "ADOL-C error: Trace was created on a 64-bit platform, cannot be opened on 32-bit platform!\n"
		);
	    break;
	case ADOLC_WRONG_PLATFORM_64:
	    fprintf(DIAG_OUT,
		    "ADOL-C error: Trace was created on a 32-bit platform, cannot be opened on 64-bit platform!\n"
		);
	    break;
        case ADOLC_TAPING_NOT_ACTUALLY_TAPING:
	    fprintf(DIAG_OUT,
		    "ADOL-C error: Trace %d is not being currently created!\n",
		    failAdditionalInfo1);
	    break;

        case ADOLC_VEC_LOCATIONGAP:
          fprintf(DIAG_OUT,
                  "ADOL-C error: arrays passed to vector operation do not have contiguous ascending locations;\nuse dynamic_cast<adouble*>(advector&) \nor call ensureContiguousLocations(size_t) to reserve  contiguous blocks prior to allocation of the arrays.\n");
          break;

        default:
            fprintf(DIAG_OUT, "ADOL-C error => unknown error type!\n");
            adolc_exit(-1, "", __func__, __FILE__, __LINE__);
            break;
    }
    adolc_exit(error+1, "", __func__,  __FILE__, __LINE__);
    // exit (error + 1);
}

/* print an error message describing the error number */
void printError() {
    fprintf(DIAG_OUT, "              ");
    switch (errno) {
        case EACCES:
            fprintf(DIAG_OUT, ">>> Access denied! <<<\n");
            break;
        case EFBIG:
            fprintf(DIAG_OUT, ">>> File too big! <<<\n");
            break;
        case EMFILE:
            fprintf(DIAG_OUT, ">>> Too many open files for this process! <<<\n");
            break;
        case ENAMETOOLONG:
            fprintf(DIAG_OUT, ">>> Path/file name too long! <<<\n");
            break;
        case ENFILE:
            fprintf(DIAG_OUT, ">>> Too many open files for this system! <<<\n");
            break;
        case ENOENT:
            fprintf(DIAG_OUT, ">>> File or directory not found! <<<\n");
            break;
        case ENOSPC:
            fprintf(DIAG_OUT, ">>> No space left on device! <<<\n");
            break;
        case EPERM:
            fprintf(DIAG_OUT, ">>> Operation not permitted! <<<\n");
            break;
        case EROFS:
            fprintf(DIAG_OUT, ">>> File system is mounted read only! <<<\n");
            break;
        default:
            fprintf(DIAG_OUT, ">>> ");
            fprintf(DIAG_OUT, "%s", strerror(errno));
            fprintf(DIAG_OUT, " <<<\n");
            break;
    }
}

/* the base names of every tape type */
char *tapeBaseNames[4]={0,0,0,0};

void clearTapeBaseNames() {
    int i;
    for(i=0;i<4;i++) {
	if (tapeBaseNames[i]) {
	    free(tapeBaseNames[i]);
	    tapeBaseNames[i]=0;
	}
    }
}

/****************************************************************************/
/* The subroutine get_fstr appends to the tape base name of type tapeType   */
/* the number fnum and ".tap" and returns a pointer to the resulting string.*/
/* The result string must be freed be the caller!                           */
/****************************************************************************/
char *createFileName(short tapeID, int tapeType) {
    char *numberString, *fileName, *extension = ".tap", *currPos;
#if defined(_OPENMP)
    char *threadName = "thread-", *threadNumberString = NULL;
    int threadNumber, threadNumberStringLength = 0, threadNameLength = 0;
#endif /* _OPENMP */
    int tapeBaseNameLength, numberStringLength, fileNameLength;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    failAdditionalInfo1 = tapeID;
    tapeBaseNameLength = strlen(tapeBaseNames[tapeType]);
    /* determine length of the number string */
    if (tapeID != 0)
        numberStringLength = (int)log10((double)tapeID);
    else numberStringLength = 0;
    ++numberStringLength;
    numberString = malloc(sizeof(char) * (numberStringLength + 1));
    if (numberString == NULL) fail(ADOLC_MALLOC_FAILED);
    sprintf(numberString, "%d", tapeID);
#if defined(_OPENMP)
    /* determine length of the thread number string */
    if (ADOLC_GLOBAL_TAPE_VARS.inParallelRegion == 1) {
        threadNameLength = strlen(threadName);
        threadNumber = omp_get_thread_num();
        if (threadNumber != 0)
            threadNumberStringLength = (int)log10((double)threadNumber);
        else threadNumberStringLength = 0;
        ++threadNumberStringLength;
        threadNumberString =
            malloc(sizeof(char) * (threadNumberStringLength + 2));
        if (threadNumberString == NULL) fail(ADOLC_MALLOC_FAILED);
        sprintf(threadNumberString, "%d", threadNumber);
        threadNumberString[threadNumberStringLength] = '_';
        ++threadNumberStringLength;
        threadNumberString[threadNumberStringLength] = 0;
    }
#endif /* _OPENMP */

    /* malloc and create */
    fileNameLength = tapeBaseNameLength + numberStringLength + 5;
#if defined(_OPENMP)
    if (ADOLC_GLOBAL_TAPE_VARS.inParallelRegion == 1)
        fileNameLength += threadNameLength + threadNumberStringLength;
#endif /* _OPENMP */
    fileName = (char *)malloc(sizeof(char) * fileNameLength);
    if (fileName == NULL) fail(ADOLC_MALLOC_FAILED);
    currPos = fileName;
    strncpy(currPos, tapeBaseNames[tapeType], tapeBaseNameLength);
    currPos += tapeBaseNameLength;
#if defined(_OPENMP)
    if (ADOLC_GLOBAL_TAPE_VARS.inParallelRegion == 1) {
        strncpy(currPos, threadName, threadNameLength);
        currPos += threadNameLength;
        strncpy(currPos, threadNumberString, threadNumberStringLength);
        currPos += threadNumberStringLength;
    }
#endif /* _OPENMP */
    strncpy(currPos, numberString, numberStringLength);
    currPos += numberStringLength;
    strncpy(currPos, extension, 4);
    currPos += 4;
    *currPos = 0;

    free(numberString);
#if defined(_OPENMP)
    if (ADOLC_GLOBAL_TAPE_VARS.inParallelRegion == 1)
        free(threadNumberString);
#endif /* _OPENMP */

    return fileName;
}

/****************************************************************************/
/* Tries to read a local config file containing, e.g., buffer sizes         */
/****************************************************************************/
static char* duplicatestr(const char* instr) {
    size_t len = strlen(instr);
    char *outstr = calloc(len+1,sizeof(char));
    strncpy(outstr,instr,len);
    return outstr;
}

#define ADOLC_LINE_LENGTH 100
void readConfigFile() {
    FILE *configFile = NULL;
    char inputLine[ADOLC_LINE_LENGTH + 1];
    char *pos1 = NULL, *pos2 = NULL, *pos3 = NULL, *pos4 = NULL, *start = NULL, *end = NULL;
    int base;
    unsigned long int number = 0;
    char *path = NULL;
    int defdirsize = strlen(TAPE_DIR PATHSEPARATOR);
    tapeBaseNames[0] = duplicatestr(
	TAPE_DIR PATHSEPARATOR ADOLC_LOCATIONS_NAME);
    tapeBaseNames[1] = duplicatestr(
	TAPE_DIR PATHSEPARATOR ADOLC_VALUES_NAME);
    tapeBaseNames[2] = duplicatestr(
	TAPE_DIR PATHSEPARATOR ADOLC_OPERATIONS_NAME);
    tapeBaseNames[3] = duplicatestr(
	TAPE_DIR PATHSEPARATOR ADOLC_TAYLORS_NAME);

    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    ADOLC_GLOBAL_TAPE_VARS.operationBufferSize = OBUFSIZE;
    ADOLC_GLOBAL_TAPE_VARS.locationBufferSize = LBUFSIZE;
    ADOLC_GLOBAL_TAPE_VARS.valueBufferSize = VBUFSIZE;
    ADOLC_GLOBAL_TAPE_VARS.taylorBufferSize = TBUFSIZE;
    ADOLC_GLOBAL_TAPE_VARS.maxNumberTaylorBuffers = TBUFNUM;
    if ((configFile = fopen(".adolcrc", "r")) != NULL) {
        fprintf(DIAG_OUT, "\nFile .adolcrc found! => Try to parse it!\n");
        fprintf(DIAG_OUT, "****************************************\n");
        while (fgets(inputLine, ADOLC_LINE_LENGTH + 1, configFile) == inputLine) {
            if (strlen(inputLine) == ADOLC_LINE_LENGTH &&
                    inputLine[ADOLC_LINE_LENGTH - 1] != 0xA) {
                fprintf(DIAG_OUT, "ADOL-C warning: Input line in .adolcrc exceeds"
                        " %d characters!\n", ADOLC_LINE_LENGTH);
                fprintf(DIAG_OUT, "                => Parsing aborted!!\n");
                break;
            }
            pos1 = strchr(inputLine, '"');
            pos2 = NULL;
            pos3 = NULL;
            pos4 = NULL;
            if (pos1 != NULL) {
                pos2 = strchr(pos1 + 1, '"');
                if (pos2 != NULL) {
                    pos3 = strchr(pos2 + 1, '"');
                    if (pos3 != NULL) pos4 = strchr(pos3 + 1, '"');
                }
            }
            if (pos4 == NULL) {
                if (pos1 != NULL)
                    fprintf(DIAG_OUT, "ADOL-C warning: Malformed input line "
                            "in .adolcrc ignored!\n");
            } else {
		if (*(pos3 + 1) == '0' && (*(pos3 + 2) == 'x' || *(pos3 + 2) == 'X')) {
		    start = pos3 + 3;
		    base = 16;
		} else if (*(pos3 + 1) == '0') {
		    start = pos3 + 2;
		    base = 8;
		} else {
		    start = pos3 + 1;
		    base = 10;
		}
		number = strtoul(start, &end, base);
                if (end == start) {
		    *pos2 = 0;
		    *pos4 = 0;
		    if (strcmp(pos1 + 1, "TAPE_DIR") == 0) {
			struct stat st;
			int err;
			path = pos3 + 1;
			err = stat(path,&st);
			if (err == 0 && S_ISDIR(st.st_mode)) {
			    int pathlen, pathseplen, namelen[4];
			    int i;
			    pathlen=strlen(path);
			    pathseplen=strlen(PATHSEPARATOR);
			    for(i = 0; i < 4; i++)
				namelen[i] = strlen(tapeBaseNames[i]);
			    clearTapeBaseNames();
			    for(i = 0; i < 4; i++) {
				char *currpos;
				int fnamelen;
				tapeBaseNames[i] = (char*)calloc(namelen[i] - defdirsize + pathlen + pathseplen + 1, sizeof(char));
				currpos = tapeBaseNames[i];
				strncpy(currpos,path,pathlen);
				currpos += pathlen;
				strncpy(currpos,PATHSEPARATOR,pathseplen);
				currpos += pathseplen;
				switch (i) {
				case 0:
				    fnamelen = strlen(ADOLC_LOCATIONS_NAME);
				    strncpy(currpos,ADOLC_LOCATIONS_NAME,fnamelen);
				    break;
				case 1:
				    fnamelen = strlen(ADOLC_VALUES_NAME);
				    strncpy(currpos,ADOLC_VALUES_NAME,fnamelen);
				    break;
				case 2:
				    fnamelen = strlen(ADOLC_OPERATIONS_NAME);
				    strncpy(currpos,ADOLC_OPERATIONS_NAME,fnamelen);
				    break;
				case 3:
				    fnamelen = strlen(ADOLC_TAYLORS_NAME);
				    strncpy(currpos,ADOLC_TAYLORS_NAME,fnamelen);
				    break;
				}
				currpos += fnamelen;
				*currpos = '\0';
			    }
			    fprintf(DIAG_OUT, "ADOL-C info: using TAPE_DIR %s for all disk bound tapes\n",path);
			} else
			    fprintf(DIAG_OUT, "ADOL-C warning: TAPE_DIR %s in .adolcrc is not an existing directory,\n will continue using %s for writing tapes\n", path, TAPE_DIR);
		    }
		    else 
			fprintf(DIAG_OUT, "ADOL-C warning: Unable to parse number in "
				".adolcrc!\n");
                } else {
                    *pos2 = 0;
                    *pos4 = 0;
                    if (strcmp(pos1 + 1, "OBUFSIZE") == 0) {
                        ADOLC_GLOBAL_TAPE_VARS.operationBufferSize = (locint)number;
                        fprintf(DIAG_OUT, "Found operation buffer size: %u\n",
                                (locint)number);
                    } else if (strcmp(pos1 + 1, "LBUFSIZE") == 0) {
                        ADOLC_GLOBAL_TAPE_VARS.locationBufferSize = (locint)number;
                        fprintf(DIAG_OUT, "Found location buffer size: %u\n",
                                (locint)number);
                    } else if (strcmp(pos1 + 1, "VBUFSIZE") == 0) {
                        ADOLC_GLOBAL_TAPE_VARS.valueBufferSize = (locint)number;
                        fprintf(DIAG_OUT, "Found value buffer size: %u\n",
                                (locint)number);
                    } else if (strcmp(pos1 + 1, "TBUFSIZE") == 0) {
                        ADOLC_GLOBAL_TAPE_VARS.taylorBufferSize = (locint)number;
                        fprintf(DIAG_OUT, "Found taylor buffer size: %u\n",
                                (locint)number);
                    } else if (strcmp(pos1 + 1, "TBUFNUM") == 0) {
                        ADOLC_GLOBAL_TAPE_VARS.maxNumberTaylorBuffers = (int)number;
                        fprintf(DIAG_OUT, "Found maximal number of taylor buffers: "
                                "%d\n", (int)number);
                    } else if (strcmp(pos1 + 1, "INITLIVE") == 0) {
                        ADOLC_GLOBAL_TAPE_VARS.initialStoreSize = (locint)number;
                        fprintf(DIAG_OUT, "Found initial live variable store size : %u\n",
                                (locint)number);
                        checkInitialStoreSize(&ADOLC_GLOBAL_TAPE_VARS);
                    } else {
                        fprintf(DIAG_OUT, "ADOL-C warning: Unable to parse "
                                "parameter name in .adolcrc!\n");
                    }
                }
            }
        }
        fprintf(DIAG_OUT, "****************************************\n\n");
        fclose(configFile);
    }
    ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
}

/****************************************************************************/
/****************************************************************************/
/* VALUE STACK FUNCTIONS                                                    */
/****************************************************************************/
/****************************************************************************/

static unsigned int numTBuffersInUse = 0;

/* record all existing adoubles on the tape
 * - intended to be used in start_trace only */
void take_stock() {
    locint space_left, loc = 0;
    double *vals;
    size_t vals_left;
    ADOLC_OPENMP_THREAD_NUMBER;

    ADOLC_OPENMP_GET_THREAD_NUMBER;
    space_left  = get_val_space(); /* remaining space in const. tape buffer */
    vals_left = ADOLC_GLOBAL_TAPE_VARS.storeSize;
    vals      = ADOLC_GLOBAL_TAPE_VARS.store;
    
    /* if we have adoubles in use */
    if (ADOLC_GLOBAL_TAPE_VARS.numLives > 0) {
    /* fill the current values (real) tape buffer and write it to disk
     * - do this as long as buffer can be fully filled */
    while (space_left < vals_left) {
        put_op(take_stock_op);
        ADOLC_PUT_LOCINT(space_left);
        ADOLC_PUT_LOCINT(loc);
        put_vals_writeBlock(vals, space_left);
        vals      += space_left;
        vals_left -= space_left;
        loc       += space_left;
        space_left = get_val_space();
    }
    /* store the remaining adouble values to the values tape buffer
     * -> no write to disk necessary */
    if (vals_left > 0) {
        put_op(take_stock_op);
        ADOLC_PUT_LOCINT(vals_left);
        ADOLC_PUT_LOCINT(loc);
        put_vals_notWriteBlock(vals, vals_left);
    }
    }
    ADOLC_CURRENT_TAPE_INFOS.traceFlag = 1;
}

/****************************************************************************/
/* record all remaining live variables on the value stack tape              */
/* - turns off trace_flag                                                   */
/* - intended to be used in stop_trace only                                 */
/****************************************************************************/
locint keep_stock() {
    locint loc2;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    /* save all the final adoubles when finishing tracing */
        loc2 = ADOLC_GLOBAL_TAPE_VARS.storeSize - 1;

        /* special signal -> all alive adoubles recorded on the end of the
         * value stack -> special handling at the beginning of reverse */
        put_op(death_not);
        ADOLC_PUT_LOCINT(0);    /* lowest loc */
        ADOLC_PUT_LOCINT(loc2); /* highest loc */

        ADOLC_CURRENT_TAPE_INFOS.numTays_Tape += ADOLC_GLOBAL_TAPE_VARS.storeSize;
        /* now really do it if keepTaylors ist set */
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors) {
            do {
                ADOLC_WRITE_SCAYLOR(ADOLC_GLOBAL_TAPE_VARS.store[loc2]);
            } while (loc2-- > 0);
        }
    ADOLC_CURRENT_TAPE_INFOS.traceFlag = 0;
    return ADOLC_GLOBAL_TAPE_VARS.storeSize;
}


/****************************************************************************/
/* Set up statics for writing taylor data                                   */
/****************************************************************************/
void taylor_begin(uint bufferSize, int degreeSave) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (ADOLC_CURRENT_TAPE_INFOS.tayBuffer != NULL) {
        #     if defined(ADOLC_DEBUG)
            fprintf(DIAG_OUT, "\nADOL-C warning: !!! Taylor information for tape %d"
                    " found that will be overwritten !!!\n\n",
                    ADOLC_CURRENT_TAPE_INFOS.tapeID);
        #     endif
        taylor_close(0);
    } else { /* check if new buffer is allowed */
        if (numTBuffersInUse == ADOLC_GLOBAL_TAPE_VARS.maxNumberTaylorBuffers)
            fail(ADOLC_TAPING_TO_MANY_TAYLOR_BUFFERS);
        ++numTBuffersInUse;
        if (ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.tay_fileName == NULL)
            ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.tay_fileName =
                createFileName(ADOLC_CURRENT_TAPE_INFOS.tapeID, TAYLORS_TAPE);
    }

    /* initial setups */
    if (ADOLC_CURRENT_TAPE_INFOS.tayBuffer != NULL)
        free(ADOLC_CURRENT_TAPE_INFOS.tayBuffer);
    ADOLC_CURRENT_TAPE_INFOS.tayBuffer = (revreal *)
            malloc(sizeof(revreal) * bufferSize);
    if (ADOLC_CURRENT_TAPE_INFOS.tayBuffer == NULL)
        fail(ADOLC_TAPING_TBUFFER_ALLOCATION_FAILED);
    ADOLC_CURRENT_TAPE_INFOS.deg_save = degreeSave;
    if (degreeSave >= 0 ) ADOLC_CURRENT_TAPE_INFOS.keepTaylors = 1;
    ADOLC_CURRENT_TAPE_INFOS.currTay = ADOLC_CURRENT_TAPE_INFOS.tayBuffer;
    ADOLC_CURRENT_TAPE_INFOS.lastTayP1 = ADOLC_CURRENT_TAPE_INFOS.currTay + bufferSize;
    ADOLC_CURRENT_TAPE_INFOS.inUse = 1;

    ADOLC_CURRENT_TAPE_INFOS.numTays_Tape = 0;
}

/****************************************************************************/
/* Close the taylor file, reset data.                                       */
/****************************************************************************/
void taylor_close(uint buffer) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    if (buffer == 0) {
        /* enforces failure of reverse => retaping */
        ADOLC_CURRENT_TAPE_INFOS.deg_save = -1;
        if (ADOLC_CURRENT_TAPE_INFOS.tay_file != NULL) {
            fclose(ADOLC_CURRENT_TAPE_INFOS.tay_file);
            remove(ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.tay_fileName);
            ADOLC_CURRENT_TAPE_INFOS.tay_file = NULL;
        }
        return;
    }

    if (ADOLC_CURRENT_TAPE_INFOS.tay_file != NULL) {
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
            put_tay_block(ADOLC_CURRENT_TAPE_INFOS.currTay);
    } else {
        ADOLC_CURRENT_TAPE_INFOS.numTays_Tape =
            ADOLC_CURRENT_TAPE_INFOS.currTay -
            ADOLC_CURRENT_TAPE_INFOS.tayBuffer;
    }
    ADOLC_CURRENT_TAPE_INFOS.lastTayBlockInCore = 1;
    ADOLC_CURRENT_TAPE_INFOS.stats[TAY_STACK_SIZE] =
        ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    /* keep track of the Ind/Dep counts of the taylor stack */
    ADOLC_CURRENT_TAPE_INFOS.tay_numInds =
        ADOLC_CURRENT_TAPE_INFOS.stats[NUM_INDEPENDENTS];
    ADOLC_CURRENT_TAPE_INFOS.tay_numDeps =
        ADOLC_CURRENT_TAPE_INFOS.stats[NUM_DEPENDENTS];

    #if defined(ADOLC_DEBUG)
    if (ADOLC_CURRENT_TAPE_INFOS.tay_file != NULL)
        fprintf(DIAG_OUT, "\n ADOL-C debug: Taylor file of length %d bytes "
                "completed\n",
                (int)(ADOLC_CURRENT_TAPE_INFOS.numTays_Tape*sizeof(revreal)));
    else
        fprintf(DIAG_OUT, "\n ADOL-C debug: Taylor array of length %d bytes "
                "completed\n",
                (int)(ADOLC_CURRENT_TAPE_INFOS.numTays_Tape*sizeof(revreal)));
    #endif
}

/****************************************************************************/
/* Initializes a reverse sweep.                                             */
/****************************************************************************/
void taylor_back (short tag, int* dep, int* ind, int* degree) {
    int i, chunks;
    size_t number, remain, chunkSize;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    /* this should be removed soon since values can be accessed via         */
    /* ADOLC_CURRENT_TAPE_INFOS directly                                    */
    *dep    = ADOLC_CURRENT_TAPE_INFOS.tay_numDeps;
    *ind    = ADOLC_CURRENT_TAPE_INFOS.tay_numInds;
    *degree = ADOLC_CURRENT_TAPE_INFOS.deg_save;

    if (ADOLC_CURRENT_TAPE_INFOS.tayBuffer == NULL)
        fail(ADOLC_REVERSE_NO_TAYLOR_STACK);
    ADOLC_CURRENT_TAPE_INFOS.nextBufferNumber =
        ADOLC_CURRENT_TAPE_INFOS.numTays_Tape /
        ADOLC_CURRENT_TAPE_INFOS.stats[TAY_BUFFER_SIZE];
    number = ADOLC_CURRENT_TAPE_INFOS.numTays_Tape %
           ADOLC_CURRENT_TAPE_INFOS.stats[TAY_BUFFER_SIZE];
    ADOLC_CURRENT_TAPE_INFOS.currTay =
        ADOLC_CURRENT_TAPE_INFOS.tayBuffer + number;
    if (ADOLC_CURRENT_TAPE_INFOS.lastTayBlockInCore != 1) {
        if ( fseek(ADOLC_CURRENT_TAPE_INFOS.tay_file,
                sizeof(revreal) *
                ADOLC_CURRENT_TAPE_INFOS.nextBufferNumber *
                ADOLC_CURRENT_TAPE_INFOS.stats[TAY_BUFFER_SIZE],
                SEEK_SET)
                == -1 ) fail(ADOLC_EVAL_SEEK_VALUE_STACK);
        chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(revreal);
        chunks = number / chunkSize;
        for (i = 0; i < chunks; ++i)
            if ((failAdditionalInfo1 =
                        fread(ADOLC_CURRENT_TAPE_INFOS.tayBuffer +
                            i * chunkSize, chunkSize * sizeof(revreal), 1,
                            ADOLC_CURRENT_TAPE_INFOS.tay_file)) != 1)
                fail(ADOLC_TAPING_FATAL_IO_ERROR);
        remain = number % chunkSize;
        if (remain != 0)
            if ((failAdditionalInfo1 =
                        fread(ADOLC_CURRENT_TAPE_INFOS.tayBuffer +
                            chunks * chunkSize, remain * sizeof(revreal), 1,
                            ADOLC_CURRENT_TAPE_INFOS.tay_file)) != 1)
                fail(ADOLC_TAPING_FATAL_IO_ERROR);
    }
    --ADOLC_CURRENT_TAPE_INFOS.nextBufferNumber;
}

/****************************************************************************/
/* Writes the block of size depth of taylor coefficients from point loc to  */
/* the taylor buffer. If the buffer is filled, then it is written to the    */
/* taylor tape.                                                             */
/****************************************************************************/
void write_taylor(locint loc, int keep) {
    revreal *i;
    double *T;
    ADOLC_OPENMP_THREAD_NUMBER;

    ADOLC_OPENMP_GET_THREAD_NUMBER;
    T = ADOLC_CURRENT_TAPE_INFOS.dpp_T[loc];

    /* write data to buffer and put buffer to disk as long as data remain in
     * the T-buffer => don't create an empty value stack buffer! */
    while (ADOLC_CURRENT_TAPE_INFOS.currTay + keep > ADOLC_CURRENT_TAPE_INFOS.lastTayP1) {
        for (i = ADOLC_CURRENT_TAPE_INFOS.currTay; i < ADOLC_CURRENT_TAPE_INFOS.lastTayP1; ++i) {
            *i = (revreal) * T;
            /* In this assignment the precision will be sacrificed if the type
             * revreal is defined as float. */
            ++T;
        }
        keep -= ADOLC_CURRENT_TAPE_INFOS.lastTayP1 - ADOLC_CURRENT_TAPE_INFOS.currTay;
        put_tay_block(ADOLC_CURRENT_TAPE_INFOS.lastTayP1);
    }

    for (i = ADOLC_CURRENT_TAPE_INFOS.currTay; i < ADOLC_CURRENT_TAPE_INFOS.currTay + keep; ++i) {
        *i = (revreal) * T;
        /* In this assignment the precision will be sacrificed
         * if the type revreal is defined as float. */
        ++T;
    }
    ADOLC_CURRENT_TAPE_INFOS.currTay += keep;
}

/****************************************************************************/
/* Writes the block of size depth of taylor coefficients from point loc to  */
/* the taylor buffer.  If the buffer is filled, then it is written to the   */
/* taylor tape.                                                             */
/*--------------------------------------------------------------------------*/
void write_taylors(locint loc, int keep, int degree, int numDir) {
    int i, j;
    double *T;
    ADOLC_OPENMP_THREAD_NUMBER;

    ADOLC_OPENMP_GET_THREAD_NUMBER;
    T = ADOLC_CURRENT_TAPE_INFOS.dpp_T[loc];

    for (j = 0; j < numDir; ++j) {
        for (i = 0; i < keep; ++i) {
            if (ADOLC_CURRENT_TAPE_INFOS.currTay == ADOLC_CURRENT_TAPE_INFOS.lastTayP1)
                put_tay_block(ADOLC_CURRENT_TAPE_INFOS.lastTayP1);
            *ADOLC_CURRENT_TAPE_INFOS.currTay = (revreal) * T;
            /* The precision will be sacrificed if the type
             * revreal is defined as float. */
            ++ADOLC_CURRENT_TAPE_INFOS.currTay;
            ++T;
        }
/*        for (i = keep; i < degree; ++i) ++T;*/
        if (degree > keep)
            T += degree - keep;
    }
}

/****************************************************************************/
/* Write_scaylors writes # size elements from x to the taylor buffer.       */
/****************************************************************************/
void write_scaylors(revreal *x, uint size) {
    revreal *i;
    uint j = 0;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    /* write data to buffer and put buffer to disk as long as data remain in
     * the x-buffer => don't create an empty value stack buffer! */
    while (ADOLC_CURRENT_TAPE_INFOS.currTay + size > ADOLC_CURRENT_TAPE_INFOS.lastTayP1) {
        for (i = ADOLC_CURRENT_TAPE_INFOS.currTay; i < ADOLC_CURRENT_TAPE_INFOS.lastTayP1; ++i) {
            *i = x[j];
            ++j;
        }
        size -= ADOLC_CURRENT_TAPE_INFOS.lastTayP1 - ADOLC_CURRENT_TAPE_INFOS.currTay;
        put_tay_block(ADOLC_CURRENT_TAPE_INFOS.lastTayP1);
    }

    for (i = ADOLC_CURRENT_TAPE_INFOS.currTay; i < ADOLC_CURRENT_TAPE_INFOS.tayBuffer + size; ++i) {
        *ADOLC_CURRENT_TAPE_INFOS.currTay = x[j];
        ++j;
    }
    ADOLC_CURRENT_TAPE_INFOS.currTay += size;
}

/****************************************************************************/
/* Writes the value stack buffer onto hard disk.                            */
/****************************************************************************/
void put_tay_block(revreal *lastTayP1) {
    int i, chunks;
    size_t number, remain, chunkSize;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    if (ADOLC_CURRENT_TAPE_INFOS.tay_file == NULL) {
        ADOLC_CURRENT_TAPE_INFOS.tay_file =
            fopen(ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.tay_fileName, "w+b");
        if (ADOLC_CURRENT_TAPE_INFOS.tay_file == NULL)
            fail(ADOLC_TAPING_TAYLOR_OPEN_FAILED);
    }
    number = lastTayP1 - ADOLC_CURRENT_TAPE_INFOS.tayBuffer;
    if (number != 0) {
        chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(revreal);
        chunks = number / chunkSize;
        for (i = 0; i < chunks; ++i)
            if ((failAdditionalInfo1 =
                    fwrite(ADOLC_CURRENT_TAPE_INFOS.tayBuffer + i *
                            chunkSize, chunkSize * sizeof(revreal), 1,
                            ADOLC_CURRENT_TAPE_INFOS.tay_file) ) != 1)
                fail(ADOLC_TAPING_FATAL_IO_ERROR);
        remain = number % chunkSize;
        if (remain != 0)
            if ((failAdditionalInfo1 =
                    fwrite(ADOLC_CURRENT_TAPE_INFOS.tayBuffer + chunks *
                            chunkSize, remain * sizeof(revreal), 1,
                            ADOLC_CURRENT_TAPE_INFOS.tay_file) ) != 1)
                fail(ADOLC_TAPING_FATAL_IO_ERROR);
        ADOLC_CURRENT_TAPE_INFOS.numTays_Tape += number;
    }
    ADOLC_CURRENT_TAPE_INFOS.currTay = ADOLC_CURRENT_TAPE_INFOS.tayBuffer;
    ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
}

/****************************************************************************/
/* Puts a block of taylor coefficients from the value stack buffer to the   */
/* taylor buffer. --- Higher Order Scalar                                   */
/****************************************************************************/
void get_taylors(locint loc, int degree) {
    int j;
    revreal *i;
    revreal *T;
    ADOLC_OPENMP_THREAD_NUMBER;

    ADOLC_OPENMP_GET_THREAD_NUMBER;
    T = ADOLC_CURRENT_TAPE_INFOS.rpp_T[loc] + degree;

    /* As long as all values from the taylor stack buffer will be used copy
     * them into the taylor buffer and load the next (previous) buffer. */
    while (ADOLC_CURRENT_TAPE_INFOS.currTay - degree < ADOLC_CURRENT_TAPE_INFOS.tayBuffer) {
        for ( i = ADOLC_CURRENT_TAPE_INFOS.currTay - 1;
                i >= ADOLC_CURRENT_TAPE_INFOS.tayBuffer;
                --i ) {
            --T;
            *T = *i;
        }
        degree -= ADOLC_CURRENT_TAPE_INFOS.currTay - ADOLC_CURRENT_TAPE_INFOS.tayBuffer;
        get_tay_block_r();
    }

    /* Copy the remaining values from the stack into the buffer ... */
    for (j = 0; j < degree; ++j) {
        --ADOLC_CURRENT_TAPE_INFOS.currTay;
        --T;
        *T = *ADOLC_CURRENT_TAPE_INFOS.currTay;
    }
}

/****************************************************************************/
/* Puts a block of taylor coefficients from the value stack buffer to the   */
/* taylor buffer. --- Higher Order Vector                                   */
/****************************************************************************/
void get_taylors_p(locint loc, int degree, int numDir) {
    int i, j;
    revreal *T;
    ADOLC_OPENMP_THREAD_NUMBER;

    ADOLC_OPENMP_GET_THREAD_NUMBER;
    T = ADOLC_CURRENT_TAPE_INFOS.rpp_T[loc] + degree * numDir;

    /* update the directions except the base point parts */
    for (j = 0; j < numDir; ++j) {
        for (i = 1; i < degree; ++i) {
            if (ADOLC_CURRENT_TAPE_INFOS.currTay == ADOLC_CURRENT_TAPE_INFOS.tayBuffer)
                get_tay_block_r();
            --ADOLC_CURRENT_TAPE_INFOS.currTay;
            --T;
            *T = *ADOLC_CURRENT_TAPE_INFOS.currTay;
        }
        --T; /* skip the base point part */
    }
    /* now update the base point parts */
    if (ADOLC_CURRENT_TAPE_INFOS.currTay == ADOLC_CURRENT_TAPE_INFOS.tayBuffer)
	get_tay_block_r();
    --ADOLC_CURRENT_TAPE_INFOS.currTay;
    for (i = 0; i < numDir; ++i) {
        *T = *ADOLC_CURRENT_TAPE_INFOS.currTay;
        T += degree;
    }
}

/****************************************************************************/
/* Gets the next (previous block) of the value stack                        */
/****************************************************************************/
void get_tay_block_r() {
    int i, chunks;
    size_t number, remain, chunkSize;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    ADOLC_CURRENT_TAPE_INFOS.lastTayBlockInCore = 0;
    number = ADOLC_CURRENT_TAPE_INFOS.stats[TAY_BUFFER_SIZE];
    if ( fseek(ADOLC_CURRENT_TAPE_INFOS.tay_file, sizeof(revreal) *
                ADOLC_CURRENT_TAPE_INFOS.nextBufferNumber * number, SEEK_SET)
            == -1 )
        fail(ADOLC_EVAL_SEEK_VALUE_STACK);
    chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(revreal);
    chunks = number / chunkSize;
    for (i = 0; i < chunks; ++i)
        if ((failAdditionalInfo1 = fread(ADOLC_CURRENT_TAPE_INFOS.tayBuffer +
                        i * chunkSize, chunkSize * sizeof(revreal), 1,
                        ADOLC_CURRENT_TAPE_INFOS.tay_file)) != 1)
            fail(ADOLC_TAPING_FATAL_IO_ERROR);
    remain = number % chunkSize;
    if (remain != 0)
        if ((failAdditionalInfo1 = fread(ADOLC_CURRENT_TAPE_INFOS.tayBuffer +
                        chunks * chunkSize, remain * sizeof(revreal), 1,
                        ADOLC_CURRENT_TAPE_INFOS.tay_file)) != 1)
            fail(ADOLC_TAPING_FATAL_IO_ERROR);
    ADOLC_CURRENT_TAPE_INFOS.currTay = ADOLC_CURRENT_TAPE_INFOS.lastTayP1;
    --ADOLC_CURRENT_TAPE_INFOS.nextBufferNumber;
}


/****************************************************************************/
/****************************************************************************/
/* NON-VALUE-STACK FUNCTIONS                                                */
/****************************************************************************/
/****************************************************************************/

void initTapeBuffers() {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    if (ADOLC_CURRENT_TAPE_INFOS.opBuffer == NULL)
        ADOLC_CURRENT_TAPE_INFOS.opBuffer = (unsigned char *)
                malloc(ADOLC_CURRENT_TAPE_INFOS.stats[OP_BUFFER_SIZE] *
                       sizeof(unsigned char));
    if (ADOLC_CURRENT_TAPE_INFOS.locBuffer == NULL)
        ADOLC_CURRENT_TAPE_INFOS.locBuffer = (locint *)
                malloc(ADOLC_CURRENT_TAPE_INFOS.stats[LOC_BUFFER_SIZE] * sizeof(locint));
    if (ADOLC_CURRENT_TAPE_INFOS.valBuffer == NULL)
        ADOLC_CURRENT_TAPE_INFOS.valBuffer = (double *)
                malloc(ADOLC_CURRENT_TAPE_INFOS.stats[VAL_BUFFER_SIZE] * sizeof(double));
    if ( ADOLC_CURRENT_TAPE_INFOS.opBuffer  == NULL ||
            ADOLC_CURRENT_TAPE_INFOS.locBuffer == NULL ||
            ADOLC_CURRENT_TAPE_INFOS.valBuffer == NULL )
        fail(ADOLC_TAPING_BUFFER_ALLOCATION_FAILED);
    ADOLC_CURRENT_TAPE_INFOS.lastOpP1 = ADOLC_CURRENT_TAPE_INFOS.opBuffer +
            ADOLC_CURRENT_TAPE_INFOS.stats[OP_BUFFER_SIZE];
    ADOLC_CURRENT_TAPE_INFOS.lastLocP1 = ADOLC_CURRENT_TAPE_INFOS.locBuffer +
            ADOLC_CURRENT_TAPE_INFOS.stats[LOC_BUFFER_SIZE];
    ADOLC_CURRENT_TAPE_INFOS.lastValP1 = ADOLC_CURRENT_TAPE_INFOS.valBuffer +
            ADOLC_CURRENT_TAPE_INFOS.stats[VAL_BUFFER_SIZE];
}

/****************************************************************************/
/* start_trace: (part of trace_on)                                          */
/* Initialization for the taping process. Does buffer allocation, sets      */
/* files names, and calls appropriate setup routines.                       */
/****************************************************************************/
void start_trace() {
    int i, space;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    initTapeBuffers();
    ADOLC_CURRENT_TAPE_INFOS.currOp  = ADOLC_CURRENT_TAPE_INFOS.opBuffer;
    ADOLC_CURRENT_TAPE_INFOS.currLoc = ADOLC_CURRENT_TAPE_INFOS.locBuffer;
    ADOLC_CURRENT_TAPE_INFOS.currVal = ADOLC_CURRENT_TAPE_INFOS.valBuffer;
    ADOLC_CURRENT_TAPE_INFOS.num_eq_prod = 0;
    ADOLC_CURRENT_TAPE_INFOS.numSwitches = 0;
    ADOLC_CURRENT_TAPE_INFOS.workMode = ADOLC_TAPING;

    /* Put operation denoting the start_of_the tape */
    put_op(start_of_tape);

    /* Leave space for the stats */
    space = STAT_SIZE * sizeof(size_t) + sizeof(ADOLC_ID);
    if (space > statSpace * sizeof(locint))
        fail(ADOLC_MORE_STAT_SPACE_REQUIRED);
    for (i = 0; i < statSpace; ++i) ADOLC_PUT_LOCINT(0);

    /* initialize value stack if necessary */
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
        taylor_begin(ADOLC_CURRENT_TAPE_INFOS.stats[TAY_BUFFER_SIZE], 0);

    /* mark possible (hard disk) tape creation */
    markNewTape();
}

static void save_params() {
    size_t np;
    size_t ip, avail, remain, chunk;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    ADOLC_CURRENT_TAPE_INFOS.stats[NUM_PARAM] =
        ADOLC_GLOBAL_TAPE_VARS.numparam;
    if (ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore != NULL)
	free(ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore);

    ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore =
            malloc(ADOLC_CURRENT_TAPE_INFOS.stats[NUM_PARAM]*sizeof(double));
    memcpy(ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore,
           ADOLC_GLOBAL_TAPE_VARS.pStore,
           ADOLC_CURRENT_TAPE_INFOS.stats[NUM_PARAM]*sizeof(double));
    free_all_taping_params();
    if (ADOLC_CURRENT_TAPE_INFOS.currVal +
        ADOLC_CURRENT_TAPE_INFOS.stats[NUM_PARAM] <
        ADOLC_CURRENT_TAPE_INFOS.lastValP1)
        put_vals_notWriteBlock(ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore,
                               ADOLC_CURRENT_TAPE_INFOS.stats[NUM_PARAM]);
    else {
        np = ADOLC_CURRENT_TAPE_INFOS.stats[NUM_PARAM];
        ip = 0;
        while (ip < np) {
            avail = ADOLC_CURRENT_TAPE_INFOS.lastValP1 - ADOLC_CURRENT_TAPE_INFOS.currVal;
            remain = np - ip;
            chunk = (avail<remain)?avail:remain;
            put_vals_notWriteBlock(ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore + ip, chunk);
            ip += chunk;
            if (ip < np)
                put_val_block(ADOLC_CURRENT_TAPE_INFOS.lastValP1);
        }
    }
}

/****************************************************************************/
/* Stop Tracing.  Clean up, and turn off trace_flag.                        */
/****************************************************************************/
void stop_trace(int flag) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    put_op(end_of_tape);        /* Mark end of tape. */
    save_params();

    ADOLC_CURRENT_TAPE_INFOS.stats[NUM_INDEPENDENTS] =
        ADOLC_CURRENT_TAPE_INFOS.numInds;
    ADOLC_CURRENT_TAPE_INFOS.stats[NUM_DEPENDENTS] =
        ADOLC_CURRENT_TAPE_INFOS.numDeps;
    ADOLC_CURRENT_TAPE_INFOS.stats[NUM_MAX_LIVES] =
        ADOLC_GLOBAL_TAPE_VARS.storeSize;

    ADOLC_CURRENT_TAPE_INFOS.stats[NUM_EQ_PROD] = 
        ADOLC_CURRENT_TAPE_INFOS.num_eq_prod; 

    ADOLC_CURRENT_TAPE_INFOS.stats[NUM_SWITCHES] =
	ADOLC_CURRENT_TAPE_INFOS.numSwitches;

    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors)
	taylor_close(ADOLC_CURRENT_TAPE_INFOS.stats[TAY_BUFFER_SIZE]);

    ADOLC_CURRENT_TAPE_INFOS.stats[TAY_STACK_SIZE] =
        ADOLC_CURRENT_TAPE_INFOS.numTays_Tape;

    /* The taylor stack size base estimation results in a doubled taylor count
     * if we tape with keep (taylors counted in adouble.cpp/avector.cpp and
     * "keep_stock" even if not written and a second time when actually
     * written by "put_tay_block"). Correction follows here. */
    if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors != 0 &&
            ADOLC_CURRENT_TAPE_INFOS.tay_file != NULL)
    {
        ADOLC_CURRENT_TAPE_INFOS.stats[TAY_STACK_SIZE] /= 2;
        ADOLC_CURRENT_TAPE_INFOS.numTays_Tape /= 2;
    }

    close_tape(flag); /* closes the tape, files up stats, and writes the
                         tape stats to the integer tape */
}

/****************************************************************************/
/* Close open tapes, update stats and clean up.                             */
/****************************************************************************/
void close_tape(int flag) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    /* finish operations tape, close it, update stats */
    if (flag != 0 || ADOLC_CURRENT_TAPE_INFOS.op_file != NULL) {
        if (ADOLC_CURRENT_TAPE_INFOS.currOp !=
                ADOLC_CURRENT_TAPE_INFOS.opBuffer)
        {
            put_op_block(ADOLC_CURRENT_TAPE_INFOS.currOp);
        }
        if (ADOLC_CURRENT_TAPE_INFOS.op_file != NULL)
            fclose(ADOLC_CURRENT_TAPE_INFOS.op_file);
        ADOLC_CURRENT_TAPE_INFOS.op_file = NULL;
        ADOLC_CURRENT_TAPE_INFOS.stats[OP_FILE_ACCESS] = 1;
        free(ADOLC_CURRENT_TAPE_INFOS.opBuffer);
        ADOLC_CURRENT_TAPE_INFOS.opBuffer = NULL;
    } else {
	ADOLC_CURRENT_TAPE_INFOS.numOps_Tape =
	    ADOLC_CURRENT_TAPE_INFOS.currOp - ADOLC_CURRENT_TAPE_INFOS.opBuffer;
    }
    ADOLC_CURRENT_TAPE_INFOS.stats[NUM_OPERATIONS] =
        ADOLC_CURRENT_TAPE_INFOS.numOps_Tape;

    /* finish constants tape, close it, update stats */
    if (flag != 0 || ADOLC_CURRENT_TAPE_INFOS.val_file != NULL) {
        if (ADOLC_CURRENT_TAPE_INFOS.currVal !=
                ADOLC_CURRENT_TAPE_INFOS.valBuffer)
        {
            put_val_block(ADOLC_CURRENT_TAPE_INFOS.currVal);
        }
        if (ADOLC_CURRENT_TAPE_INFOS.val_file != NULL) 
            fclose(ADOLC_CURRENT_TAPE_INFOS.val_file);
        ADOLC_CURRENT_TAPE_INFOS.val_file = NULL;
        ADOLC_CURRENT_TAPE_INFOS.stats[VAL_FILE_ACCESS] = 1;
        free(ADOLC_CURRENT_TAPE_INFOS.valBuffer);
        ADOLC_CURRENT_TAPE_INFOS.valBuffer = NULL;
    } else {
	ADOLC_CURRENT_TAPE_INFOS.numVals_Tape =
	    ADOLC_CURRENT_TAPE_INFOS.currVal - ADOLC_CURRENT_TAPE_INFOS.valBuffer;
    }
    ADOLC_CURRENT_TAPE_INFOS.stats[NUM_VALUES] =
        ADOLC_CURRENT_TAPE_INFOS.numVals_Tape;

    /* finish locations tape, update and write tape stats, close tape */
    if (flag != 0 || ADOLC_CURRENT_TAPE_INFOS.loc_file != NULL) {
        if (ADOLC_CURRENT_TAPE_INFOS.currLoc !=
                ADOLC_CURRENT_TAPE_INFOS.locBuffer)
        {
            put_loc_block(ADOLC_CURRENT_TAPE_INFOS.currLoc);
        }
        ADOLC_CURRENT_TAPE_INFOS.stats[NUM_LOCATIONS] =
            ADOLC_CURRENT_TAPE_INFOS.numLocs_Tape;
        ADOLC_CURRENT_TAPE_INFOS.stats[LOC_FILE_ACCESS] = 1;
        /* write tape stats */
        fseek(ADOLC_CURRENT_TAPE_INFOS.loc_file, 0, 0);
        fwrite(&adolc_id, sizeof(ADOLC_ID), 1,
                ADOLC_CURRENT_TAPE_INFOS.loc_file);
        fwrite(ADOLC_CURRENT_TAPE_INFOS.stats, STAT_SIZE * sizeof(size_t), 1,
               ADOLC_CURRENT_TAPE_INFOS.loc_file);
        fclose(ADOLC_CURRENT_TAPE_INFOS.loc_file);
        ADOLC_CURRENT_TAPE_INFOS.loc_file = NULL;
        free(ADOLC_CURRENT_TAPE_INFOS.locBuffer);
        ADOLC_CURRENT_TAPE_INFOS.locBuffer = NULL;
    } else {
	ADOLC_CURRENT_TAPE_INFOS.numLocs_Tape  =
	    ADOLC_CURRENT_TAPE_INFOS.currLoc - ADOLC_CURRENT_TAPE_INFOS.locBuffer;
	ADOLC_CURRENT_TAPE_INFOS.stats[NUM_LOCATIONS] =
	    ADOLC_CURRENT_TAPE_INFOS.numLocs_Tape;
    }
}

/****************************************************************************/
/* Free all resources used by a tape before overwriting the tape.           */
/****************************************************************************/
void freeTapeResources(TapeInfos *tapeInfos) {
    free(tapeInfos->opBuffer);
    tapeInfos->opBuffer = NULL;
    free(tapeInfos->locBuffer);
    tapeInfos->locBuffer = NULL;
    free(tapeInfos->valBuffer);
    tapeInfos->valBuffer = NULL;
    if (tapeInfos->tayBuffer != NULL) {
        free(tapeInfos->tayBuffer);
        tapeInfos->tayBuffer = NULL;
        --numTBuffersInUse;
    }
    if (tapeInfos->op_file != NULL) {
        fclose(tapeInfos->op_file);
        tapeInfos->op_file = NULL;
    }
    if (tapeInfos->loc_file != NULL) {
        fclose(tapeInfos->loc_file);
        tapeInfos->loc_file = NULL;
    }
    if (tapeInfos->val_file != NULL) {
        fclose(tapeInfos->val_file);
        tapeInfos->val_file = NULL;
    }
    if (tapeInfos->tay_file != NULL) {
        fclose(tapeInfos->tay_file);
        tapeInfos->tay_file = NULL;
    }
}

/****************************************************************************/
/* Tapestats:                                                               */
/* Returns statistics on the tape tag with following meaning:               */
/* tape_stat[0] = # of independent variables.                               */
/* tape_stat[1] = # of dependent variables.                                 */
/* tape_stat[2] = max # of live variables.                                  */
/* tape_stat[3] = value stack size.                                         */
/* tape_stat[4] = buffer size (# of chars, # of doubles, # of locints)      */
/* tape_stat[5] = # of operations.                                          */
/* tape_stat[6] = operation file access flag (1 = file in use, 0 otherwise) */
/* tape_stat[7] = # of saved locations.                                     */
/* tape_stat[8] = location file access flag (1 = file in use, 0 otherwise)  */
/* tape_stat[9] = # of saved constant values.                               */
/* tape_stat[10]= value file access flag (1 = file in use, 0 otherwise)     */
/****************************************************************************/
void tapestats(short tag, size_t *tape_stats) {
    int i;
    TapeInfos *tapeInfos;

    /* get the tapeInfos for tag */
    tapeInfos = getTapeInfos(tag);
    /* copy stats to the users field */
    for (i = 0; i < STAT_SIZE; ++i)
        tape_stats[i] = tapeInfos->stats[i];
}

/****************************************************************************/
/* An all-in-one tape stats printing routine.                               */
/****************************************************************************/
void printTapeStats(FILE *stream, short tag) {
    size_t stats[STAT_SIZE];

    tapestats(tag, (size_t *)&stats);
    fprintf(stream, "\n*** TAPE STATS (tape %d) **********\n", (int)tag);
    fprintf(stream, "Number of independents: %10zu\n", stats[NUM_INDEPENDENTS]);
    fprintf(stream, "Number of dependents:   %10zu\n", stats[NUM_DEPENDENTS]);
    fprintf(stream, "\n");
    fprintf(stream, "Max # of live adoubles: %10zu\n", stats[NUM_MAX_LIVES]);
    fprintf(stream, "Taylor stack size:      %10zu\n", stats[TAY_STACK_SIZE]);
    fprintf(stream, "\n");
    fprintf(stream, "Number of operations:   %10zu\n", stats[NUM_OPERATIONS]);
    fprintf(stream, "Number of locations:    %10zu\n", stats[NUM_LOCATIONS]);
    fprintf(stream, "Number of values:       %10zu\n", stats[NUM_VALUES]);
    fprintf(stream, "Number of parameters:   %10zu\n", stats[NUM_PARAM]);
    fprintf(stream, "\n");
    fprintf(stream, "Operation file written: %10zu\n", stats[OP_FILE_ACCESS]);
    fprintf(stream, "Location file written:  %10zu\n", stats[LOC_FILE_ACCESS]);
    fprintf(stream, "Value file written:     %10zu\n", stats[VAL_FILE_ACCESS]);
    fprintf(stream, "\n");
    fprintf(stream, "Operation buffer size:  %10zu\n", stats[OP_BUFFER_SIZE]);
    fprintf(stream, "Location buffer size:   %10zu\n", stats[LOC_BUFFER_SIZE]);
    fprintf(stream, "Value buffer size:      %10zu\n", stats[VAL_BUFFER_SIZE]);
    fprintf(stream, "Taylor buffer size:     %10zu\n", stats[TAY_BUFFER_SIZE]);
    fprintf(stream, "\n");
    fprintf(stream, "Operation type size:    %10zu\n",
            (size_t)sizeof(unsigned char));
    fprintf(stream, "Location type size:     %10zu\n", (size_t)sizeof(locint));
    fprintf(stream, "Value type size:        %10zu\n", (size_t)sizeof(double));
    fprintf(stream, "Taylor type size:       %10zu\n", (size_t)sizeof(revreal));
    fprintf(stream, "**********************************\n\n");
}

/****************************************************************************/
/* Returns the number of parameters recorded on tape                        */
/****************************************************************************/
size_t get_num_param(short tag) {
    TapeInfos *tapeInfos;
    tapeInfos = getTapeInfos(tag);
    return tapeInfos->stats[NUM_PARAM];
}

/****************************************************************************/
/* Reads parameters from the end of value tape for disk based tapes         */
/****************************************************************************/
static void read_params(TapeInfos* tapeInfos) {
    FILE* val_file;
    int i, chunks;
    size_t number, remain, chunkSize, nVT;
    double *valBuffer = NULL, *currVal = NULL, *lastValP1 = NULL;
    size_t np, ip, avail, rsize;
    if (tapeInfos->pTapeInfos.paramstore == NULL)
        tapeInfos->pTapeInfos.paramstore =
            malloc(tapeInfos->stats[NUM_PARAM]*sizeof(double));
    valBuffer = (double*)
        malloc(tapeInfos->stats[VAL_BUFFER_SIZE] *sizeof(double));
    lastValP1 = valBuffer + tapeInfos->stats[VAL_BUFFER_SIZE];
    if ((val_file = fopen(tapeInfos->pTapeInfos.val_fileName, "rb")) == NULL)
        fail(ADOLC_VALUE_TAPE_FOPEN_FAILED);
    number = (tapeInfos->stats[NUM_VALUES] /
              tapeInfos->stats[VAL_BUFFER_SIZE]) *
        tapeInfos->stats[VAL_BUFFER_SIZE];
    fseek(val_file, number * sizeof(double), SEEK_SET);
    number = tapeInfos->stats[NUM_VALUES] % tapeInfos->stats[VAL_BUFFER_SIZE];
    if (number != 0) {
        chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(double);
        chunks = number / chunkSize;
        for (i = 0; i < chunks; ++i)
            if (fread(valBuffer + i * chunkSize, chunkSize * sizeof(double), 1,
                      val_file) != 1 )
                fail(ADOLC_VALUE_TAPE_FREAD_FAILED);
        remain = number % chunkSize;
        if (remain != 0)
            if (fread(valBuffer + chunks * chunkSize, remain * sizeof(double), 1,
                      val_file) != 1)
                fail(ADOLC_VALUE_TAPE_FREAD_FAILED);
    }
    nVT = tapeInfos->stats[NUM_VALUES] - number;
    currVal = valBuffer + number;
    np = tapeInfos->stats[NUM_PARAM];
    ip = np;
    while ( ip > 0) {
        avail = currVal - valBuffer;
        rsize = (avail<ip)?avail:ip;
        for ( i = 0; i < rsize; i++ )
            tapeInfos->pTapeInfos.paramstore[--ip] = *--currVal;
        if (ip > 0) {
            number = tapeInfos->stats[VAL_BUFFER_SIZE];
            fseek(val_file, sizeof(double)*(nVT - number), SEEK_SET);
            chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(double);
            chunks = number / chunkSize;
            for (i = 0; i < chunks; ++i)
                if (fread(valBuffer + i * chunkSize, chunkSize * sizeof(double), 1,
                          val_file) != 1 )
                    fail(ADOLC_VALUE_TAPE_FREAD_FAILED);
            remain = number % chunkSize;
            if (remain != 0)
                if (fread(valBuffer + chunks * chunkSize, remain * sizeof(double), 1,
                          val_file) != 1)
                    fail(ADOLC_VALUE_TAPE_FREAD_FAILED);
            nVT -= number;
            currVal = lastValP1;
        }
    }
    fclose(val_file);
    free(valBuffer);
}

/****************************************************************************/
/* Overrides the parameters for the next evaluations. This will invalidate  */
/* the taylor stack, so next reverse call will fail, if not preceeded by a  */
/* forward call after setting the parameters.                               */
/****************************************************************************/
void set_param_vec(short tag, size_t numparam, revreal* paramvec) {
    size_t i;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    /* mark possible (hard disk) tape creation */
    markNewTape();

    /* make room for tapeInfos and read tape stats if necessary, keep value
     * stack information */
    openTape(tag, ADOLC_FORWARD);
    if (ADOLC_CURRENT_TAPE_INFOS.stats[NUM_PARAM] != numparam) {
        fprintf(DIAG_OUT, "ADOL-C error: Setting parameters on tape %d "
                "aborted!\nNumber of parameters (%zu) passed"
                " is inconsistent with number recorded on tape (%zu)\n",
                tag, numparam, ADOLC_CURRENT_TAPE_INFOS.stats[NUM_PARAM]);
        adolc_exit(-1,"",__func__,__FILE__,__LINE__);
    }
    if (ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore == NULL)
        ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore = (double*)
            malloc(ADOLC_CURRENT_TAPE_INFOS.stats[NUM_PARAM]*sizeof(double));
    for(i = 0; i < ADOLC_CURRENT_TAPE_INFOS.stats[NUM_PARAM]; i++)
        ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.paramstore[i] = paramvec[i];
    taylor_close(0);
    releaseTape();
}

/****************************************************************************/
/* Does the actual reading from the hard disk into the stats buffer         */
/****************************************************************************/
void read_tape_stats(TapeInfos *tapeInfos) {
    FILE *loc_file;
    int tapeVersion, limitVersion;
    ADOLC_ID tape_ADOLC_ID;

    if (tapeInfos->inUse != 0 && tapeInfos->tapingComplete == 0) return;

    limitVersion = 100 * ADOLC_NEW_TAPE_VERSION    +
            10 * ADOLC_NEW_TAPE_SUBVERSION +
            1  * ADOLC_NEW_TAPE_PATCHLEVEL ;

    if ((loc_file = fopen(tapeInfos->pTapeInfos.loc_fileName, "rb")) == NULL)
        fail(ADOLC_INTEGER_TAPE_FOPEN_FAILED);
    if (fread(&tape_ADOLC_ID, sizeof(ADOLC_ID), 1, loc_file) != 1)
        fail(ADOLC_INTEGER_TAPE_FREAD_FAILED);
    if (fread(tapeInfos->stats, STAT_SIZE * sizeof(size_t), 1, loc_file) != 1)
        fail(ADOLC_INTEGER_TAPE_FREAD_FAILED);

    failAdditionalInfo1 = tapeInfos->tapeID;
    tapeVersion = 100 * tape_ADOLC_ID.adolc_ver +
            10 * tape_ADOLC_ID.adolc_sub +
            1  * tape_ADOLC_ID.adolc_lvl ;
    if (tapeVersion < limitVersion) fail(ADOLC_TAPE_TO_OLD);

    if (tape_ADOLC_ID.address_size != adolc_id.address_size) {
	if (tape_ADOLC_ID.address_size < adolc_id.address_size)
	    fail(ADOLC_WRONG_PLATFORM_64);
	else
	    fail(ADOLC_WRONG_PLATFORM_32);
    }

    if (tape_ADOLC_ID.locint_size != adolc_id.locint_size) {
        failAdditionalInfo1 = tape_ADOLC_ID.locint_size;
        failAdditionalInfo2 = adolc_id.locint_size;
        fail(ADOLC_WRONG_LOCINT_SIZE);
    }

    fclose(loc_file);
    tapeInfos->tapingComplete = 1;
    if (tapeInfos->stats[NUM_PARAM] > 0)
        read_params(tapeInfos);
}

void skip_tracefile_cleanup(short tnum) {
    TapeInfos *tinfo = getTapeInfos(tnum);
    tinfo->pTapeInfos.skipFileCleanup = 1;
}

/****************************************************************************/
/* Initialize a forward sweep. Get stats, open tapes, fill buffers, ...     */
/****************************************************************************/
void init_for_sweep(short tag) {
    int i = 0, chunks, numLocsForStats;
    size_t number, remain, chunkSize;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    /* mark possible (hard disk) tape creation */
    markNewTape();

    /* make room for tapeInfos and read tape stats if necessary, keep value
     * stack information */
    openTape(tag, ADOLC_FORWARD);
    initTapeBuffers();

    /* init operations */
    number = 0;
    if (ADOLC_CURRENT_TAPE_INFOS.stats[OP_FILE_ACCESS] == 1) {
        ADOLC_CURRENT_TAPE_INFOS.op_file =
            fopen(ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.op_fileName, "rb");
        /* how much to read ? */
        number = MIN_ADOLC(ADOLC_CURRENT_TAPE_INFOS.stats[OP_BUFFER_SIZE],
                ADOLC_CURRENT_TAPE_INFOS.stats[NUM_OPERATIONS]);
        if (number != 0) {
            chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(unsigned char);
            chunks = number / chunkSize;
            for (i = 0; i < chunks; ++i)
                if (fread(ADOLC_CURRENT_TAPE_INFOS.opBuffer + i * chunkSize,
                            chunkSize * sizeof(unsigned char), 1,
                            ADOLC_CURRENT_TAPE_INFOS.op_file) != 1 )
                    fail(ADOLC_EVAL_OP_TAPE_READ_FAILED);
            remain = number % chunkSize;
            if (remain != 0)
                if (fread(ADOLC_CURRENT_TAPE_INFOS.opBuffer + chunks *
                            chunkSize, remain * sizeof(unsigned char), 1,
                            ADOLC_CURRENT_TAPE_INFOS.op_file) != 1 )
                    fail(ADOLC_EVAL_OP_TAPE_READ_FAILED);
        }
        /* how much remains ? */
        number = ADOLC_CURRENT_TAPE_INFOS.stats[NUM_OPERATIONS] - number;
    }
    ADOLC_CURRENT_TAPE_INFOS.numOps_Tape = number;
    ADOLC_CURRENT_TAPE_INFOS.currOp = ADOLC_CURRENT_TAPE_INFOS.opBuffer;

    /* init locations */
    number = 0;
    if (ADOLC_CURRENT_TAPE_INFOS.stats[LOC_FILE_ACCESS] == 1) {
        ADOLC_CURRENT_TAPE_INFOS.loc_file =
            fopen(ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.loc_fileName, "rb");
        /* how much to read ? */
        number = MIN_ADOLC(ADOLC_CURRENT_TAPE_INFOS.stats[LOC_BUFFER_SIZE],
                ADOLC_CURRENT_TAPE_INFOS.stats[NUM_LOCATIONS]);
        if (number != 0) {
            chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(locint);
            chunks = number / chunkSize;
            for (i = 0; i < chunks; ++i)
                if (fread(ADOLC_CURRENT_TAPE_INFOS.locBuffer + i * chunkSize,
                            chunkSize * sizeof(locint), 1,
                            ADOLC_CURRENT_TAPE_INFOS.loc_file) != 1 )
                    fail(ADOLC_EVAL_LOC_TAPE_READ_FAILED);
            remain = number % chunkSize;
            if (remain != 0)
            if (fread(ADOLC_CURRENT_TAPE_INFOS.locBuffer + chunks * chunkSize,
                        remain * sizeof(locint), 1,
                        ADOLC_CURRENT_TAPE_INFOS.loc_file) != 1 )
                fail(ADOLC_EVAL_LOC_TAPE_READ_FAILED);
        }
        /* how much remains ? */
        number = ADOLC_CURRENT_TAPE_INFOS.stats[NUM_LOCATIONS] - number;
    }
    ADOLC_CURRENT_TAPE_INFOS.numLocs_Tape = number;

    /* skip stats */
    numLocsForStats = statSpace;
    while (numLocsForStats >= ADOLC_CURRENT_TAPE_INFOS.stats[LOC_BUFFER_SIZE])
    {
        get_loc_block_f();
        numLocsForStats -= ADOLC_CURRENT_TAPE_INFOS.stats[LOC_BUFFER_SIZE];
    }
    ADOLC_CURRENT_TAPE_INFOS.currLoc =
        ADOLC_CURRENT_TAPE_INFOS.locBuffer + numLocsForStats;

    /* init constants */
    number = 0;
    if (ADOLC_CURRENT_TAPE_INFOS.stats[VAL_FILE_ACCESS] == 1) {
        ADOLC_CURRENT_TAPE_INFOS.val_file =
            fopen(ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.val_fileName, "rb");
        /* how much to read ? */
        number = MIN_ADOLC(ADOLC_CURRENT_TAPE_INFOS.stats[VAL_BUFFER_SIZE],
                ADOLC_CURRENT_TAPE_INFOS.stats[NUM_VALUES]);
        if (number != 0) {
            chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(double);
            chunks = number / chunkSize;
            for (i = 0; i < chunks; ++i)
                if (fread(ADOLC_CURRENT_TAPE_INFOS.valBuffer + i * chunkSize,
                            chunkSize * sizeof(double), 1,
                            ADOLC_CURRENT_TAPE_INFOS.val_file) != 1 )
                    fail(ADOLC_EVAL_VAL_TAPE_READ_FAILED);
            remain = number % chunkSize;
            if (remain != 0)
                if (fread(ADOLC_CURRENT_TAPE_INFOS.valBuffer + chunks *
                            chunkSize, remain * sizeof(double), 1,
                            ADOLC_CURRENT_TAPE_INFOS.val_file) != 1 )
                    fail(ADOLC_EVAL_VAL_TAPE_READ_FAILED);
        }
        /* how much remains ? */
        number = ADOLC_CURRENT_TAPE_INFOS.stats[NUM_VALUES] - number;
    }
    ADOLC_CURRENT_TAPE_INFOS.numVals_Tape = number;
    ADOLC_CURRENT_TAPE_INFOS.currVal = ADOLC_CURRENT_TAPE_INFOS.valBuffer;
#ifdef ADOLC_AMPI_SUPPORT
    TAPE_AMPI_resetBottom();
#endif
}

/****************************************************************************/
/* Initialize a reverse sweep. Get stats, open tapes, fill buffers, ...     */
/****************************************************************************/
void init_rev_sweep(short tag) {
    int i, chunks;
    size_t number, remain, chunkSize;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    /* mark possible (hard disk) tape creation */
    markNewTape();

    /* make room for tapeInfos and read tape stats if necessary, keep value
     * stack information */
    openTape(tag, ADOLC_REVERSE);
    initTapeBuffers();

    /* init operations */
    number = ADOLC_CURRENT_TAPE_INFOS.stats[NUM_OPERATIONS];
    if (ADOLC_CURRENT_TAPE_INFOS.stats[OP_FILE_ACCESS] == 1) {
        ADOLC_CURRENT_TAPE_INFOS.op_file =
            fopen(ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.op_fileName, "rb");
        number = (ADOLC_CURRENT_TAPE_INFOS.stats[NUM_OPERATIONS] /
                ADOLC_CURRENT_TAPE_INFOS.stats[OP_BUFFER_SIZE]) *
                ADOLC_CURRENT_TAPE_INFOS.stats[OP_BUFFER_SIZE];
        fseek(ADOLC_CURRENT_TAPE_INFOS.op_file,
                number * sizeof(unsigned char), SEEK_SET);
        number = ADOLC_CURRENT_TAPE_INFOS.stats[NUM_OPERATIONS] %
                ADOLC_CURRENT_TAPE_INFOS.stats[OP_BUFFER_SIZE] ;
        if (number != 0) {
            chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(unsigned char);
            chunks = number / chunkSize;
            for (i = 0; i < chunks; ++i)
                if (fread(ADOLC_CURRENT_TAPE_INFOS.opBuffer + i * chunkSize,
                            chunkSize * sizeof(unsigned char), 1,
                            ADOLC_CURRENT_TAPE_INFOS.op_file) != 1 )
                    fail(ADOLC_EVAL_OP_TAPE_READ_FAILED);
            remain = number % chunkSize;
            if (remain != 0)
                if (fread(ADOLC_CURRENT_TAPE_INFOS.opBuffer + chunks *
                            chunkSize, remain * sizeof(unsigned char), 1,
                            ADOLC_CURRENT_TAPE_INFOS.op_file) != 1 )
                    fail(ADOLC_EVAL_OP_TAPE_READ_FAILED);
        }
    }
    ADOLC_CURRENT_TAPE_INFOS.numOps_Tape =
        ADOLC_CURRENT_TAPE_INFOS.stats[NUM_OPERATIONS] - number;
    ADOLC_CURRENT_TAPE_INFOS.currOp =
        ADOLC_CURRENT_TAPE_INFOS.opBuffer + number;

    /* init locations */
    number = ADOLC_CURRENT_TAPE_INFOS.stats[NUM_LOCATIONS];
    if (ADOLC_CURRENT_TAPE_INFOS.stats[LOC_FILE_ACCESS] == 1) {
        ADOLC_CURRENT_TAPE_INFOS.loc_file =
            fopen(ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.loc_fileName, "rb");
        number = (ADOLC_CURRENT_TAPE_INFOS.stats[NUM_LOCATIONS] /
                ADOLC_CURRENT_TAPE_INFOS.stats[LOC_BUFFER_SIZE]) *
                ADOLC_CURRENT_TAPE_INFOS.stats[LOC_BUFFER_SIZE];
        fseek(ADOLC_CURRENT_TAPE_INFOS.loc_file,
                number * sizeof(locint), SEEK_SET);
        number = ADOLC_CURRENT_TAPE_INFOS.stats[NUM_LOCATIONS] %
                ADOLC_CURRENT_TAPE_INFOS.stats[LOC_BUFFER_SIZE];
        if (number != 0) {
            chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(locint);
            chunks = number / chunkSize;
            for (i = 0; i < chunks; ++i)
                if (fread(ADOLC_CURRENT_TAPE_INFOS.locBuffer + i * chunkSize,
                            chunkSize * sizeof(locint), 1,
                            ADOLC_CURRENT_TAPE_INFOS.loc_file) != 1 )
                    fail(ADOLC_EVAL_LOC_TAPE_READ_FAILED);
            remain = number % chunkSize;
            if (remain != 0)
                if (fread(ADOLC_CURRENT_TAPE_INFOS.locBuffer + chunks *
                            chunkSize, remain * sizeof(locint), 1,
                            ADOLC_CURRENT_TAPE_INFOS.loc_file) != 1 )
                    fail(ADOLC_EVAL_LOC_TAPE_READ_FAILED);
        }
    }
    ADOLC_CURRENT_TAPE_INFOS.numLocs_Tape =
        ADOLC_CURRENT_TAPE_INFOS.stats[NUM_LOCATIONS] - number;
    ADOLC_CURRENT_TAPE_INFOS.currLoc =
        ADOLC_CURRENT_TAPE_INFOS.locBuffer + number;

    /* init constants */
    number = ADOLC_CURRENT_TAPE_INFOS.stats[NUM_VALUES];
    if (ADOLC_CURRENT_TAPE_INFOS.stats[VAL_FILE_ACCESS] == 1) {
        ADOLC_CURRENT_TAPE_INFOS.val_file =
            fopen(ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.val_fileName, "rb");
        number = (ADOLC_CURRENT_TAPE_INFOS.stats[NUM_VALUES] /
                ADOLC_CURRENT_TAPE_INFOS.stats[VAL_BUFFER_SIZE]) *
                ADOLC_CURRENT_TAPE_INFOS.stats[VAL_BUFFER_SIZE];
        fseek(ADOLC_CURRENT_TAPE_INFOS.val_file,
                number * sizeof(double), SEEK_SET);
        number = ADOLC_CURRENT_TAPE_INFOS.stats[NUM_VALUES] %
                ADOLC_CURRENT_TAPE_INFOS.stats[VAL_BUFFER_SIZE];
        if (number != 0) {
            chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(double);
            chunks = number / chunkSize;
            for (i = 0; i < chunks; ++i)
                if (fread(ADOLC_CURRENT_TAPE_INFOS.valBuffer + i * chunkSize,
                            chunkSize * sizeof(double), 1,
                            ADOLC_CURRENT_TAPE_INFOS.val_file) != 1 )
                    fail(ADOLC_EVAL_VAL_TAPE_READ_FAILED);
            remain = number % chunkSize;
            if (remain != 0)
                if (fread(ADOLC_CURRENT_TAPE_INFOS.valBuffer + chunks *
                            chunkSize, remain * sizeof(double), 1,
                            ADOLC_CURRENT_TAPE_INFOS.val_file) != 1 )
                    fail(ADOLC_EVAL_VAL_TAPE_READ_FAILED);
        }
    }
    ADOLC_CURRENT_TAPE_INFOS.numVals_Tape =
        ADOLC_CURRENT_TAPE_INFOS.stats[NUM_VALUES] - number;
    ADOLC_CURRENT_TAPE_INFOS.currVal =
        ADOLC_CURRENT_TAPE_INFOS.valBuffer + number;
#ifdef ADOLC_AMPI_SUPPORT
    TAPE_AMPI_resetTop();
#endif
}

/****************************************************************************/
/* Finish a forward or reverse sweep.                                       */
/****************************************************************************/
void end_sweep() {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (ADOLC_CURRENT_TAPE_INFOS.op_file != NULL) {
        fclose(ADOLC_CURRENT_TAPE_INFOS.op_file);
        ADOLC_CURRENT_TAPE_INFOS.op_file = NULL;
    }
    if (ADOLC_CURRENT_TAPE_INFOS.loc_file != NULL) {
        fclose(ADOLC_CURRENT_TAPE_INFOS.loc_file);
        ADOLC_CURRENT_TAPE_INFOS.loc_file = NULL;
    }
    if (ADOLC_CURRENT_TAPE_INFOS.val_file != NULL) {
        fclose(ADOLC_CURRENT_TAPE_INFOS.val_file);
        ADOLC_CURRENT_TAPE_INFOS.val_file = NULL;
    }
    if (ADOLC_CURRENT_TAPE_INFOS.deg_save > 0) releaseTape(); /* keep value stack */
    else releaseTape(); /* no value stack */
}

/* --- Operations --- */

#if defined(__USE_ISOC99)
const int maxLocsPerOp=10;
#endif

/****************************************************************************/
/* Puts an operation into the operation buffer. Ensures that location buffer*/
/* and constants buffer are prepared to take the belonging stuff.           */
/****************************************************************************/
void put_op_reserve(unsigned char op, unsigned int reserveExtraLocations) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    /* make sure we have enough slots to write the locs */
    if (ADOLC_CURRENT_TAPE_INFOS.currLoc + maxLocsPerOp + reserveExtraLocations > ADOLC_CURRENT_TAPE_INFOS.lastLocP1) {
        size_t remainder = ADOLC_CURRENT_TAPE_INFOS.lastLocP1 - ADOLC_CURRENT_TAPE_INFOS.currLoc;
        if (remainder>0) memset(ADOLC_CURRENT_TAPE_INFOS.currLoc,0,(remainder-1)*sizeof(locint));
        *(ADOLC_CURRENT_TAPE_INFOS.lastLocP1 - 1) = remainder;
        put_loc_block(ADOLC_CURRENT_TAPE_INFOS.lastLocP1);
        /* every operation writes 1 opcode */
        if (ADOLC_CURRENT_TAPE_INFOS.currOp + 1 == ADOLC_CURRENT_TAPE_INFOS.lastOpP1) {
            *ADOLC_CURRENT_TAPE_INFOS.currOp = end_of_op;
            put_op_block(ADOLC_CURRENT_TAPE_INFOS.lastOpP1);
            *ADOLC_CURRENT_TAPE_INFOS.currOp = end_of_op;
            ++ADOLC_CURRENT_TAPE_INFOS.currOp;
        }
        *ADOLC_CURRENT_TAPE_INFOS.currOp = end_of_int;
        ++ADOLC_CURRENT_TAPE_INFOS.currOp;
    }
    /* every operation writes <5 values --- 3 should be sufficient */
    if (ADOLC_CURRENT_TAPE_INFOS.currVal + 5 > ADOLC_CURRENT_TAPE_INFOS.lastValP1) {
        locint valRemainder=ADOLC_CURRENT_TAPE_INFOS.lastValP1 - ADOLC_CURRENT_TAPE_INFOS.currVal;
        ADOLC_PUT_LOCINT(valRemainder);
        /* avoid writing uninitialized memory to the file and get valgrind upset */
        memset(ADOLC_CURRENT_TAPE_INFOS.currVal,0,valRemainder*sizeof(double));
        put_val_block(ADOLC_CURRENT_TAPE_INFOS.lastValP1);
        /* every operation writes 1 opcode */
        if (ADOLC_CURRENT_TAPE_INFOS.currOp + 1 == ADOLC_CURRENT_TAPE_INFOS.lastOpP1) {
            *ADOLC_CURRENT_TAPE_INFOS.currOp = end_of_op;
            put_op_block(ADOLC_CURRENT_TAPE_INFOS.lastOpP1);
            *ADOLC_CURRENT_TAPE_INFOS.currOp = end_of_op;
            ++ADOLC_CURRENT_TAPE_INFOS.currOp;
        }
        *ADOLC_CURRENT_TAPE_INFOS.currOp = end_of_val;
        ++ADOLC_CURRENT_TAPE_INFOS.currOp;
    }
    /* every operation writes 1 opcode */
    if (ADOLC_CURRENT_TAPE_INFOS.currOp + 1 == ADOLC_CURRENT_TAPE_INFOS.lastOpP1) {
        *ADOLC_CURRENT_TAPE_INFOS.currOp = end_of_op;
        put_op_block(ADOLC_CURRENT_TAPE_INFOS.lastOpP1);
        *ADOLC_CURRENT_TAPE_INFOS.currOp = end_of_op;
        ++ADOLC_CURRENT_TAPE_INFOS.currOp;
    }
    *ADOLC_CURRENT_TAPE_INFOS.currOp = op;
    ++ADOLC_CURRENT_TAPE_INFOS.currOp;
}

/****************************************************************************/
/* Writes a block of operations onto hard disk and handles file creation,   */
/* removal, ...                                                             */
/****************************************************************************/
void put_op_block(unsigned char *lastOpP1) {
    size_t i, chunks;
    size_t number, remain, chunkSize;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    if (ADOLC_CURRENT_TAPE_INFOS.op_file == NULL) {
        ADOLC_CURRENT_TAPE_INFOS.op_file =
            fopen(ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.op_fileName, "rb");
        if (ADOLC_CURRENT_TAPE_INFOS.op_file != NULL) {
#if defined(ADOLC_DEBUG)
            fprintf(DIAG_OUT, "ADOL-C debug: Old tapefile %s gets removed!\n",
                    ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.op_fileName);
#endif
            fclose(ADOLC_CURRENT_TAPE_INFOS.op_file);
            ADOLC_CURRENT_TAPE_INFOS.op_file = NULL;
            if (remove(ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.op_fileName))
                fprintf(DIAG_OUT, "ADOL-C warning: "
                        "Unable to remove old tapefile\n");
            ADOLC_CURRENT_TAPE_INFOS.op_file =
                fopen(ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.op_fileName, "wb");
        } else {
            ADOLC_CURRENT_TAPE_INFOS.op_file =
                fopen(ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.op_fileName, "wb");
        }
    }

    number = lastOpP1 - ADOLC_CURRENT_TAPE_INFOS.opBuffer;
    chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(unsigned char);
    chunks = number / chunkSize;
    for (i = 0; i < chunks; ++i)
        if ((failAdditionalInfo1 = fwrite(ADOLC_CURRENT_TAPE_INFOS.opBuffer +
                        i * chunkSize, chunkSize *
                        sizeof(unsigned char), 1,
                        ADOLC_CURRENT_TAPE_INFOS.op_file) ) != 1 )
            fail(ADOLC_TAPING_FATAL_IO_ERROR);
    remain = number % chunkSize;
    if (remain != 0)
        if ((failAdditionalInfo1 = fwrite(ADOLC_CURRENT_TAPE_INFOS.opBuffer +
                        chunks * chunkSize, remain *
                        sizeof(unsigned char), 1,
                        ADOLC_CURRENT_TAPE_INFOS.op_file) ) != 1 )
            fail(ADOLC_TAPING_FATAL_IO_ERROR);
    ADOLC_CURRENT_TAPE_INFOS.numOps_Tape += number;
    ADOLC_CURRENT_TAPE_INFOS.currOp = ADOLC_CURRENT_TAPE_INFOS.opBuffer;
    ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
}

/****************************************************************************/
/* Reads the next operations block into the internal buffer.                */
/****************************************************************************/
void get_op_block_f() {
    size_t i, chunks;
    size_t number, remain, chunkSize;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    number = MIN_ADOLC(ADOLC_CURRENT_TAPE_INFOS.stats[OP_BUFFER_SIZE],
            ADOLC_CURRENT_TAPE_INFOS.numOps_Tape);
    chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(unsigned char);
    chunks = number / chunkSize;
    for (i = 0; i < chunks; ++i)
        if (fread(ADOLC_CURRENT_TAPE_INFOS.opBuffer + i * chunkSize,
                    chunkSize * sizeof(unsigned char), 1,
                    ADOLC_CURRENT_TAPE_INFOS.op_file) != 1)
            fail(ADOLC_EVAL_OP_TAPE_READ_FAILED);
    remain = number % chunkSize;
    if (remain != 0)
        if (fread(ADOLC_CURRENT_TAPE_INFOS.opBuffer + chunks * chunkSize,
                    remain * sizeof(unsigned char), 1,
                    ADOLC_CURRENT_TAPE_INFOS.op_file) != 1)
            fail(ADOLC_EVAL_OP_TAPE_READ_FAILED);
    ADOLC_CURRENT_TAPE_INFOS.numOps_Tape -= remain;
    ADOLC_CURRENT_TAPE_INFOS.currOp = ADOLC_CURRENT_TAPE_INFOS.opBuffer;
}

/****************************************************************************/
/* Reads the previous block of operations into the internal buffer.         */
/****************************************************************************/
void get_op_block_r() {
    size_t i, chunks;
    size_t number, remain, chunkSize;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    number = ADOLC_CURRENT_TAPE_INFOS.stats[OP_BUFFER_SIZE];
    fseek(ADOLC_CURRENT_TAPE_INFOS.op_file, sizeof(unsigned char) *
            (ADOLC_CURRENT_TAPE_INFOS.numOps_Tape - number), SEEK_SET);
    chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(unsigned char);
    chunks = number / chunkSize;
    for (i = 0; i < chunks; ++i)
        if (fread(ADOLC_CURRENT_TAPE_INFOS.opBuffer + i * chunkSize,
                    chunkSize * sizeof(unsigned char), 1,
                    ADOLC_CURRENT_TAPE_INFOS.op_file) != 1)
            fail(ADOLC_EVAL_OP_TAPE_READ_FAILED);
    remain = number % chunkSize;
    if (remain != 0)
        if (fread(ADOLC_CURRENT_TAPE_INFOS.opBuffer + chunks * chunkSize,
                    remain * sizeof(unsigned char), 1,
                    ADOLC_CURRENT_TAPE_INFOS.op_file) != 1)
            fail(ADOLC_EVAL_OP_TAPE_READ_FAILED);
    ADOLC_CURRENT_TAPE_INFOS.numOps_Tape -= number;
    ADOLC_CURRENT_TAPE_INFOS.currOp =
        ADOLC_CURRENT_TAPE_INFOS.opBuffer + number;
}

/* --- Locations --- */

/****************************************************************************/
/* Writes a block of locations onto hard disk and handles file creation,   */
/* removal, ...                                                             */
/****************************************************************************/
void put_loc_block(locint *lastLocP1) {
    size_t i, chunks;
    size_t number, remain, chunkSize;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    if (ADOLC_CURRENT_TAPE_INFOS.loc_file == NULL) {
        ADOLC_CURRENT_TAPE_INFOS.loc_file =
            fopen(ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.loc_fileName, "rb");
        if (ADOLC_CURRENT_TAPE_INFOS.loc_file != NULL) {
            #if defined(ADOLC_DEBUG)
            fprintf(DIAG_OUT, "ADOL-C debug: Old tapefile %s gets removed!\n",
                    ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.loc_fileName);
            #endif
            fclose(ADOLC_CURRENT_TAPE_INFOS.loc_file);
            ADOLC_CURRENT_TAPE_INFOS.loc_file = NULL;
            if (remove(ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.loc_fileName))
                fprintf(DIAG_OUT, "ADOL-C warning: "
                        "Unable to remove old tapefile!\n");
            ADOLC_CURRENT_TAPE_INFOS.loc_file =
                fopen(ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.loc_fileName, "wb");
        } else {
            ADOLC_CURRENT_TAPE_INFOS.loc_file =
                fopen(ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.loc_fileName, "wb");
        }
    }

    number = lastLocP1 - ADOLC_CURRENT_TAPE_INFOS.locBuffer;
    chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(locint);
    chunks = number / chunkSize;
    for (i = 0; i < chunks; ++i)
        if ((failAdditionalInfo1 = fwrite(ADOLC_CURRENT_TAPE_INFOS.locBuffer +
                        i * chunkSize, chunkSize * sizeof(locint), 1,
                        ADOLC_CURRENT_TAPE_INFOS.loc_file) ) != 1)
            fail(ADOLC_TAPING_FATAL_IO_ERROR);
    remain = number % chunkSize;
    if (remain != 0)
        if ((failAdditionalInfo1 = fwrite(ADOLC_CURRENT_TAPE_INFOS.locBuffer +
                        chunks * chunkSize, remain * sizeof(locint), 1,
                        ADOLC_CURRENT_TAPE_INFOS.loc_file) ) != 1)
            fail(ADOLC_TAPING_FATAL_IO_ERROR);
    ADOLC_CURRENT_TAPE_INFOS.numLocs_Tape += number;
    ADOLC_CURRENT_TAPE_INFOS.currLoc = ADOLC_CURRENT_TAPE_INFOS.locBuffer;
    ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
}

/****************************************************************************/
/* Reads the next block of locations into the internal buffer.              */
/****************************************************************************/
void get_loc_block_f() {
    size_t i, chunks;
    size_t number, remain, chunkSize;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    number = MIN_ADOLC(ADOLC_CURRENT_TAPE_INFOS.stats[LOC_BUFFER_SIZE],
            ADOLC_CURRENT_TAPE_INFOS.numLocs_Tape);
    chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof (locint);
    chunks = number / chunkSize;
    for (i = 0; i < chunks; ++i)
        if (fread(ADOLC_CURRENT_TAPE_INFOS.locBuffer + i * chunkSize,
                    chunkSize * sizeof(locint), 1,
                    ADOLC_CURRENT_TAPE_INFOS.loc_file) != 1)
            fail(ADOLC_EVAL_LOC_TAPE_READ_FAILED);
    remain = number % chunkSize;
    if (remain != 0)
        if (fread(ADOLC_CURRENT_TAPE_INFOS.locBuffer + chunks * chunkSize,
                    remain * sizeof(locint), 1,
                    ADOLC_CURRENT_TAPE_INFOS.loc_file) != 1)
            fail(ADOLC_EVAL_LOC_TAPE_READ_FAILED);
    ADOLC_CURRENT_TAPE_INFOS.numLocs_Tape -= number;
    ADOLC_CURRENT_TAPE_INFOS.currLoc = ADOLC_CURRENT_TAPE_INFOS.locBuffer;
}

/****************************************************************************/
/* Reads the previous block of locations into the internal buffer.          */
/****************************************************************************/
void get_loc_block_r() {
    size_t i, chunks;
    size_t number, remain, chunkSize;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    number = ADOLC_CURRENT_TAPE_INFOS.stats[LOC_BUFFER_SIZE];
    fseek(ADOLC_CURRENT_TAPE_INFOS.loc_file, sizeof(locint) *
            (ADOLC_CURRENT_TAPE_INFOS.numLocs_Tape - number), SEEK_SET);
    chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(locint);
    chunks = number / chunkSize;
    for (i = 0; i < chunks; ++i)
        if (fread(ADOLC_CURRENT_TAPE_INFOS.locBuffer + i * chunkSize,
                   chunkSize * sizeof(locint), 1,
                   ADOLC_CURRENT_TAPE_INFOS.loc_file) != 1)
            fail(ADOLC_EVAL_LOC_TAPE_READ_FAILED);
    remain = number % chunkSize;
    if (remain != 0)
        if (fread(ADOLC_CURRENT_TAPE_INFOS.locBuffer + chunks * chunkSize,
                   remain * sizeof(locint), 1,
                   ADOLC_CURRENT_TAPE_INFOS.loc_file) != 1)
            fail(ADOLC_EVAL_LOC_TAPE_READ_FAILED);
    ADOLC_CURRENT_TAPE_INFOS.numLocs_Tape -=
        ADOLC_CURRENT_TAPE_INFOS.stats[LOC_BUFFER_SIZE];
    ADOLC_CURRENT_TAPE_INFOS.currLoc = ADOLC_CURRENT_TAPE_INFOS.lastLocP1 -
            *(ADOLC_CURRENT_TAPE_INFOS.lastLocP1 - 1);
}

/* --- Values (Constants -- Real) --- */

/****************************************************************************/
/* Writes a block of constants (real) onto hard disk and handles file       */
/* creation, removal, ...                                                   */
/****************************************************************************/
void put_vals_writeBlock(double *vals, locint numVals) {
    int i;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    for (i = 0; i < numVals; ++i) {
        *ADOLC_CURRENT_TAPE_INFOS.currVal = vals[i];
        ++ADOLC_CURRENT_TAPE_INFOS.currVal;
    }
    ADOLC_PUT_LOCINT(ADOLC_CURRENT_TAPE_INFOS.lastValP1 - ADOLC_CURRENT_TAPE_INFOS.currVal);
    put_val_block(ADOLC_CURRENT_TAPE_INFOS.lastValP1);
    /* every operation writes 1 opcode */
    if (ADOLC_CURRENT_TAPE_INFOS.currOp + 1 == ADOLC_CURRENT_TAPE_INFOS.lastOpP1) {
        *ADOLC_CURRENT_TAPE_INFOS.currOp = end_of_op;
        put_op_block(ADOLC_CURRENT_TAPE_INFOS.lastOpP1);
        *ADOLC_CURRENT_TAPE_INFOS.currOp = end_of_op;
        ++ADOLC_CURRENT_TAPE_INFOS.currOp;
    }
    *ADOLC_CURRENT_TAPE_INFOS.currOp = end_of_val;
    ++ADOLC_CURRENT_TAPE_INFOS.currOp;
}

/****************************************************************************/
/* Write some constants to the buffer without disk access                   */
/****************************************************************************/
void put_vals_notWriteBlock(double *vals, locint numVals) {
    int i;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    for (i = 0; i < numVals; ++i) {
        *ADOLC_CURRENT_TAPE_INFOS.currVal = vals[i];
        ++ADOLC_CURRENT_TAPE_INFOS.currVal;
    }
}

/****************************************************************************/
/* Writes a block of constants (real) onto tape and handles file creation   */
/* removal, ...                                                             */
/****************************************************************************/
void put_val_block(double *lastValP1) {
    size_t i, chunks;
    size_t number, remain, chunkSize;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    if (ADOLC_CURRENT_TAPE_INFOS.val_file == NULL) {
        ADOLC_CURRENT_TAPE_INFOS.val_file =
            fopen(ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.val_fileName, "rb");
        if (ADOLC_CURRENT_TAPE_INFOS.val_file != NULL) {
            #if defined(ADOLC_DEBUG)
            fprintf(DIAG_OUT, "ADOL-C debug: Old tapefile %s gets removed!\n",
                    ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.val_fileName);
            #endif
            fclose(ADOLC_CURRENT_TAPE_INFOS.val_file);
            ADOLC_CURRENT_TAPE_INFOS.val_file = NULL;
            if (remove(ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.val_fileName))
                fprintf(DIAG_OUT, "ADOL-C warning: "
                        "Unable to remove old tapefile\n");
            ADOLC_CURRENT_TAPE_INFOS.val_file =
                fopen(ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.val_fileName, "wb");
        } else {
            ADOLC_CURRENT_TAPE_INFOS.val_file =
                fopen(ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.val_fileName, "wb");
        }
    }

    number = lastValP1 - ADOLC_CURRENT_TAPE_INFOS.valBuffer;
    chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(double);
    chunks = number / chunkSize;
    for (i = 0; i < chunks; ++i)
        if ((failAdditionalInfo1 = fwrite(ADOLC_CURRENT_TAPE_INFOS.valBuffer +
                        i * chunkSize, chunkSize * sizeof(double), 1,
                        ADOLC_CURRENT_TAPE_INFOS.val_file) ) != 1)
            fail(ADOLC_TAPING_FATAL_IO_ERROR);
    remain = number % chunkSize;
    if (remain != 0)
        if ((failAdditionalInfo1 = fwrite(ADOLC_CURRENT_TAPE_INFOS.valBuffer +
                        chunks * chunkSize, remain * sizeof(double), 1,
                        ADOLC_CURRENT_TAPE_INFOS.val_file) ) != 1)
            fail(ADOLC_TAPING_FATAL_IO_ERROR);
    ADOLC_CURRENT_TAPE_INFOS.numVals_Tape += number;
    ADOLC_CURRENT_TAPE_INFOS.currVal = ADOLC_CURRENT_TAPE_INFOS.valBuffer;
    ADOLC_OPENMP_RESTORE_THREAD_NUMBER;
}

/****************************************************************************/
/* Reads the next block of constants into the internal buffer.              */
/****************************************************************************/
void get_val_block_f() {
    size_t i, chunks;
    size_t number, remain, chunkSize;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    number = MIN_ADOLC(ADOLC_CURRENT_TAPE_INFOS.stats[VAL_BUFFER_SIZE],
            ADOLC_CURRENT_TAPE_INFOS.numVals_Tape);
    chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof (double);
    chunks = number / chunkSize;
    for (i = 0; i < chunks; ++i)
        if (fread(ADOLC_CURRENT_TAPE_INFOS.valBuffer + i * chunkSize,
                    chunkSize * sizeof(double), 1,
                    ADOLC_CURRENT_TAPE_INFOS.val_file) != 1)
            fail(ADOLC_EVAL_VAL_TAPE_READ_FAILED);
    remain = number % chunkSize;
    if (remain != 0)
        if (fread(ADOLC_CURRENT_TAPE_INFOS.valBuffer + chunks * chunkSize,
                    remain * sizeof(double), 1,
                    ADOLC_CURRENT_TAPE_INFOS.val_file) != 1)
            fail(ADOLC_EVAL_VAL_TAPE_READ_FAILED);
    ADOLC_CURRENT_TAPE_INFOS.numVals_Tape -= number;
    ADOLC_CURRENT_TAPE_INFOS.currVal = ADOLC_CURRENT_TAPE_INFOS.valBuffer;
    /* get_locint_f(); value used in reverse only */
    ++ADOLC_CURRENT_TAPE_INFOS.currLoc;
}

/****************************************************************************/
/* Reads the previous block of values into the internal buffer.             */
/****************************************************************************/
void get_val_block_r() {
    size_t i, chunks;
    size_t number, remain, chunkSize;
    locint temp;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    number = ADOLC_CURRENT_TAPE_INFOS.stats[VAL_BUFFER_SIZE];
    fseek(ADOLC_CURRENT_TAPE_INFOS.val_file, sizeof(double) *
            (ADOLC_CURRENT_TAPE_INFOS.numVals_Tape - number), SEEK_SET);
    chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(double);
    chunks = number / chunkSize;
    for (i = 0; i < chunks; ++i)
        if (fread(ADOLC_CURRENT_TAPE_INFOS.valBuffer + i * chunkSize,
                   chunkSize * sizeof(double), 1,
                   ADOLC_CURRENT_TAPE_INFOS.val_file) != 1)
            fail(ADOLC_EVAL_VAL_TAPE_READ_FAILED);
    remain = number % chunkSize;
    if (remain != 0)
        if (fread(ADOLC_CURRENT_TAPE_INFOS.valBuffer + chunks * chunkSize,
                    remain * sizeof(double), 1,
                    ADOLC_CURRENT_TAPE_INFOS.val_file) != 1)
            fail(ADOLC_EVAL_VAL_TAPE_READ_FAILED);
    ADOLC_CURRENT_TAPE_INFOS.numVals_Tape -= number;
    --ADOLC_CURRENT_TAPE_INFOS.currLoc;
    temp = *ADOLC_CURRENT_TAPE_INFOS.currLoc;
    ADOLC_CURRENT_TAPE_INFOS.currVal =
        ADOLC_CURRENT_TAPE_INFOS.lastValP1 - temp;
}

/****************************************************************************/
/* Returns the number of free constants in the real tape. Ensures that it   */
/* is at least 5.                                                           */
/****************************************************************************/
locint get_val_space(void) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (ADOLC_CURRENT_TAPE_INFOS.lastValP1 - 5 < ADOLC_CURRENT_TAPE_INFOS.currVal) {
        ADOLC_PUT_LOCINT(ADOLC_CURRENT_TAPE_INFOS.lastValP1 - ADOLC_CURRENT_TAPE_INFOS.currVal);
        put_val_block(ADOLC_CURRENT_TAPE_INFOS.lastValP1);
        /* every operation writes 1 opcode */
        if (ADOLC_CURRENT_TAPE_INFOS.currOp + 1 == ADOLC_CURRENT_TAPE_INFOS.lastOpP1) {
            *ADOLC_CURRENT_TAPE_INFOS.currOp = end_of_op;
            put_op_block(ADOLC_CURRENT_TAPE_INFOS.lastOpP1);
            *ADOLC_CURRENT_TAPE_INFOS.currOp = end_of_op;
            ++ADOLC_CURRENT_TAPE_INFOS.currOp;
        }
        *ADOLC_CURRENT_TAPE_INFOS.currOp = end_of_val;
        ++ADOLC_CURRENT_TAPE_INFOS.currOp;
    }
    return (ADOLC_CURRENT_TAPE_INFOS.lastValP1 - ADOLC_CURRENT_TAPE_INFOS.currVal);
}

/****************************************************************************/
/* Discards parameters from the end of value tape during reverse mode       */
/****************************************************************************/
void discard_params_r(void) {
    size_t i, np, ip, avail, rsize, chunks;
    size_t number, remain, chunkSize;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    np = ADOLC_CURRENT_TAPE_INFOS.stats[NUM_PARAM];
    ip = np;
    while ( ip > 0 ) {
	avail = ADOLC_CURRENT_TAPE_INFOS.currVal - ADOLC_CURRENT_TAPE_INFOS.valBuffer;
	rsize = (avail<ip)?avail:ip;
	ip -= rsize;
	ADOLC_CURRENT_TAPE_INFOS.currVal -= rsize;
	if ( ip > 0 ) {
	    number = ADOLC_CURRENT_TAPE_INFOS.stats[VAL_BUFFER_SIZE];
	    fseek(ADOLC_CURRENT_TAPE_INFOS.val_file, sizeof(double) *
		(ADOLC_CURRENT_TAPE_INFOS.numVals_Tape - number), SEEK_SET);
	    chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(double);
	    chunks = number / chunkSize;
	    for (i = 0; i < chunks; ++i)
		if (fread(ADOLC_CURRENT_TAPE_INFOS.valBuffer +
		i * chunkSize, chunkSize * sizeof(double), 1,
		ADOLC_CURRENT_TAPE_INFOS.val_file) != 1)
		    fail(ADOLC_EVAL_VAL_TAPE_READ_FAILED);
	    remain = number % chunkSize;
	    if (remain != 0)
		if (fread(ADOLC_CURRENT_TAPE_INFOS.valBuffer +
		chunks * chunkSize, remain * sizeof(double), 1,
		ADOLC_CURRENT_TAPE_INFOS.val_file) != 1)
		    fail(ADOLC_EVAL_VAL_TAPE_READ_FAILED);
	    ADOLC_CURRENT_TAPE_INFOS.numVals_Tape -= number;
	    ADOLC_CURRENT_TAPE_INFOS.currVal =
		ADOLC_CURRENT_TAPE_INFOS.lastValP1;
	}
    }
}

/****************************************************************************/
/* Returns a pointer to the first element of a values vector and skips the  */
/* vector. -- Forward Mode --                                               */
/****************************************************************************/
double *get_val_v_f(locint size) {
    double *temp;
    ADOLC_OPENMP_THREAD_NUMBER;

    ADOLC_OPENMP_GET_THREAD_NUMBER;
    temp = ADOLC_CURRENT_TAPE_INFOS.currVal;
    ADOLC_CURRENT_TAPE_INFOS.currVal += size;
    return temp;
}

/****************************************************************************/
/* Returns a pointer to the first element of a values vector and skips the  */
/* vector. -- Reverse Mode --                                               */
/****************************************************************************/
double *get_val_v_r(locint size) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    ADOLC_CURRENT_TAPE_INFOS.currVal -= size;
    return ADOLC_CURRENT_TAPE_INFOS.currVal;
}

/* --- Updates / Corrections --- */

/****************************************************************************/
/* Not sure what's going on here! -> vector class ?  --- kowarz             */
/****************************************************************************/
void reset_val_r(void) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (ADOLC_CURRENT_TAPE_INFOS.currVal == ADOLC_CURRENT_TAPE_INFOS.valBuffer)
        get_val_block_r();
}

/****************************************************************************/
/* Update locations tape to remove assignments involving temp. variables.   */
/* e.g.  t = a + b ; y = t  =>  y = a + b                                   */
/****************************************************************************/
int upd_resloc(locint temp, locint lhs) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (ADOLC_CURRENT_TAPE_INFOS.currLoc - ADOLC_CURRENT_TAPE_INFOS.locBuffer < 1) return 0;
    if (temp == *(ADOLC_CURRENT_TAPE_INFOS.currLoc - 1)) {
        *(ADOLC_CURRENT_TAPE_INFOS.currLoc - 1) = lhs;
        return 1;
    }
    return 0;
}

int upd_resloc_check(locint temp, locint lhs) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (ADOLC_CURRENT_TAPE_INFOS.currLoc - ADOLC_CURRENT_TAPE_INFOS.locBuffer < 1) return 0;
    if (temp == *(ADOLC_CURRENT_TAPE_INFOS.currLoc - 1)) {
        return 1;
    }
    return 0;
}
/****************************************************************************/
/* Update locations and operations tape to remove special operations inv.   */
/* temporary variables. e.g.  t = a * b ; y += t  =>  y += a * b            */
/****************************************************************************/
int upd_resloc_inc_prod(locint temp, locint newlhs, unsigned char newop) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (ADOLC_CURRENT_TAPE_INFOS.currLoc - ADOLC_CURRENT_TAPE_INFOS.locBuffer < 3) return 0;
    if (ADOLC_CURRENT_TAPE_INFOS.currOp - ADOLC_CURRENT_TAPE_INFOS.opBuffer < 1) return 0;
    if (temp == *(ADOLC_CURRENT_TAPE_INFOS.currLoc - 1)    &&
            mult_a_a == *(ADOLC_CURRENT_TAPE_INFOS.currOp - 1) &&
            /* skipping recursive case */
            newlhs != *(ADOLC_CURRENT_TAPE_INFOS.currLoc - 2)  &&
            newlhs != *(ADOLC_CURRENT_TAPE_INFOS.currLoc - 3)    ) {
        *(ADOLC_CURRENT_TAPE_INFOS.currLoc - 1) = newlhs;
        *(ADOLC_CURRENT_TAPE_INFOS.currOp - 1) = newop;
        return 1;
    }
    return 0;
}

void enableBranchSwitchWarnings() {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    ADOLC_GLOBAL_TAPE_VARS.branchSwitchWarning = 1;
}

void disableBranchSwitchWarnings() {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    ADOLC_GLOBAL_TAPE_VARS.branchSwitchWarning = 0;
}

/****************************************************************************/
/*                                                                    UTILs */
/****************************************************************************/
double make_nan() {
    double a, b;
    #ifdef inf_num
    a = non_num;
    b = non_den;
    #endif
    return a / b;
}

double make_inf() {
    double a, b;
    #ifdef inf_num
    a = inf_num;
    b = inf_den;
    #endif
    return a / b;
}

/****************************************************************************/
/*                                                          DEBUG FUNCTIONS */
#if defined(ADOLC_HARDDEBUG)

/*--------------------------------------------------------------------------*/
unsigned char get_op_f() {
    unsigned char temp;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    temp = *ADOLC_CURRENT_TAPE_INFOS.currOp;
    ++ADOLC_CURRENT_TAPE_INFOS.currOp;
    fprintf(DIAG_OUT, "f_op: %i\n", temp - '\0'); /* why -'\0' ??? kowarz */
    return temp;
}

/*--------------------------------------------------------------------------*/
unsigned char get_op_r() {
    unsigned char temp;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    --ADOLC_CURRENT_TAPE_INFOS.currOp;
    temp = *ADOLC_CURRENT_TAPE_INFOS.currOp;
    fprintf(DIAG_OUT, "r_op: %i\n", temp - '\0');
    return temp;
}

/*--------------------------------------------------------------------------*/
locint get_locint_f() {
    locint temp;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    temp = *ADOLC_CURRENT_TAPE_INFOS.currLoc;
    ++ADOLC_CURRENT_TAPE_INFOS.currLoc;
    fprintf(DIAG_OUT, "f_loc: %i\n", temp);
    return temp;
}

/*--------------------------------------------------------------------------*/
locint get_locint_r() {
    unsigned char temp;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    --ADOLC_CURRENT_TAPE_INFOS.currLoc;
    temp = *ADOLC_CURRENT_TAPE_INFOS.currLoc;
    fprintf(DIAG_OUT, "r_loc: %i\n", temp);
    return temp;
}

/*--------------------------------------------------------------------------*/
double get_val_f() {
    double temp;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    temp = *ADOLC_CURRENT_TAPE_INFOS.currVal;
    ++ADOLC_CURRENT_TAPE_INFOS.currVal;
    fprintf(DIAG_OUT, "f_val: %e\n", temp);
    return temp;
}

/*--------------------------------------------------------------------------*/
double get_val_r() {
    double temp;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    --ADOLC_CURRENT_TAPE_INFOS.currVal;
    temp = *ADOLC_CURRENT_TAPE_INFOS.currVal;
    fprintf(DIAG_OUT, "r_val: %e\n", temp);
    return temp;
}

#endif


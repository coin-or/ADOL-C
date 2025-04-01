
#include <adolc/adolcerror.h>
#include <adolc/dvlparms.h> // for ADOLC version infos
#include <cstring>
#include <iostream>
/*--------------------------------------------------------------------------*/
/* print an error message describing the error number */
void printError() {
  std::string message("              \n");
  switch (errno) {
  case EACCES:
    message += ">>> Access denied! <<<\n";
    break;
  case EFBIG:
    message += ">>> File too big! <<<\n";
    break;
  case EMFILE:
    message += ">>> Too many open files for this process! <<<\n";
    break;
  case ENAMETOOLONG:
    message += ">>> Path/file name too long! <<<\n";
    break;
  case ENFILE:
    message += ">>> Too many open files for this system! <<<\n";
    break;
  case ENOENT:
    message += ">>> File or directory not found! <<<\n";
    break;
  case ENOSPC:
    message += ">>> No space left on device! <<<\n";
    break;
  case EPERM:
    message += ">>> Operation not permitted! <<<\n";
    break;
  case EROFS:
    message += ">>> File system is mounted read only! <<<\n";
    break;
  default:
    message += std::format(">>> {} <<<\n", std::strerror(errno));
    break;
  }
}

// outputs an appropriate error message using DIAG_OUT and exits the running
// program
void fail(ADOLC_ERRORS error, const std::source_location LocInfo,
          const FailInfo &failinfo) {
  using std::cerr;
  using std::format;

  switch (to_underlying(error)) {
  case to_underlying(ADOLC_ERRORS::ADOLC_MALLOC_FAILED):
    throw ADOLCError("ADOL-C error: Memory allocation failed!\n", LocInfo);
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_INTEGER_TAPE_FOPEN_FAILED):
  case to_underlying(ADOLC_ERRORS::ADOLC_INTEGER_TAPE_FREAD_FAILED):
    cerr << format("ADOL-C error: reading integer tape number {}!\n",
                   failinfo.info1);
    printError();
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_VALUE_TAPE_FOPEN_FAILED):
  case to_underlying(ADOLC_ERRORS::ADOLC_VALUE_TAPE_FREAD_FAILED):
    cerr << format("ADOL-C error: reading value tape number {}!\n",
                   failinfo.info1);
    printError();
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_TAPE_TO_OLD):
    throw ADOLCError(
        format("ADOL-C error: Used tape ({}) was written with ADOL-C version "
               "older than {}.{}.{}\n This is ADOL-C {}.{}.{}\n",
               failinfo.info1, ADOLC_NEW_TAPE_VERSION,
               ADOLC_NEW_TAPE_SUBVERSION, ADOLC_NEW_TAPE_PATCHLEVEL,
               ADOLC_VERSION, ADOLC_SUBVERSION, ADOLC_PATCHLEVEL),
        LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_WRONG_LOCINT_SIZE):
    throw ADOLCError(format("ADOL-C error: Used tape ({}) was written with "
                            "locints of size {}, size {} required.\n",
                            failinfo.info1, failinfo.info1, failinfo.info2),
                     LocInfo);
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_MORE_STAT_SPACE_REQUIRED):
    throw ADOLCError("ADOL-C error: Not enough space for stats!\n Please "
                     "contact the ADOL-C team!\n",
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_TAPING_BUFFER_ALLOCATION_FAILED):
    throw ADOLCError("ADOL-C error: Cannot allocate tape buffers!\n", LocInfo);
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_TAPING_TBUFFER_ALLOCATION_FAILED):
    throw ADOLCError("ADOL-C error: Cannot allocate taylor buffer!\n", LocInfo);
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_TAPING_READ_ERROR_IN_TAYLOR_CLOSE):
    throw ADOLCError(format("ADOL-C error: Read error in taylor_close n= {}\n",
                            failinfo.info1),
                     LocInfo);
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_TAPING_TO_MANY_TAYLOR_BUFFERS):
    throw ADOLCError("ADOL-C error: To many taylor buffers!\n Increase "
                     "maxNumberTaylorBuffers\n",
                     LocInfo);
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_TAPING_TO_MANY_LOCINTS):
    throw ADOLCError(
        format(
            "ADOL-C error: Maximal number ({}) of live active variables "
            "exceeded!\n\n Possible remedies :\n\n 1. Use more automatic local "
            "variables and\n allocate/deallocate adoubles on free store\n in a "
            "strictly last in first out fashion\n\n 2. Extend the range by "
            "redefining the type of\n locint(currently {} byte) from unsigned "
            "short ({} byte) or int\n to int ({} byte) or long ({} byte).\n",
            failinfo.info3, sizeof(size_t), sizeof(unsigned short), sizeof(int),
            sizeof(long)),
        LocInfo);
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_TAPING_STORE_REALLOC_FAILED):
    throw ADOLCError(
        format(
            "ADOL-C error: Failure to reallocate storage for adouble "
            "values!\n\n oldStore = {}\n newStore = nullptr\n oldStoreSize = "
            "{}\n newStoreSize = {}\n\n Possible remedies :\n 1. Use more "
            "automatic local variables and \n allocate / deallocate adoubles "
            "on free store\n in a strictly last in first out fashion\n 2. "
            "Enlarge your system stacksize limit\n",
            failinfo.info5, failinfo.info3, failinfo.info4),
        LocInfo);
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_TAPING_FATAL_IO_ERROR):
    cerr << "ADOL-C error: Fatal error-doing a read or "
            "write!\n";
    printError();
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_TAPING_TAPE_STILL_IN_USE):
    throw ADOLCError(
        format("ADOL-C error: Tape {} is still in use!\n", failinfo.info1),
        LocInfo);
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_TAPING_TAYLOR_OPEN_FAILED):
    cerr << "ADOL-C error: while opening taylor file!\n";
    printError();
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_EVAL_SEEK_VALUE_STACK):
    throw ADOLCError("ADOL-C error: in seeking value stack file!\n", LocInfo);
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_EVAL_OP_TAPE_READ_FAILED):
    throw ADOLCError("ADOL-C error: while reading operations tape!\n", LocInfo);
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_EVAL_VAL_TAPE_READ_FAILED):
    throw ADOLCError("ADOL-C error: while reading values tape!\n", LocInfo);
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_EVAL_LOC_TAPE_READ_FAILED):
    throw ADOLCError("ADOL-C error: while reading locations tape!\n", LocInfo);
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_EVAL_TAY_TAPE_READ_FAILED):
    throw ADOLCError("ADOL-C error: while reading value stack tape!\n",
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_REVERSE_NO_TAYLOR_STACK):
    throw ADOLCError(
        format(
            "ADOL-C error: No taylor stack found for tape {}! => Check forward "
            "sweep!\n",
            failinfo.info1),
        LocInfo);
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_REVERSE_COUNTS_MISMATCH):
    throw ADOLCError(
        format("ADOL-C error: Reverse sweep on tape {} aborted!\n Number of "
               "dependents({}) and/or independents({})\n variables passed to "
               "reverse "
               "is inconsistent\n with number recorded on tape : ({} / {}) +) "
               "!\n ",
               failinfo.info1, failinfo.info3, failinfo.info4, failinfo.info5,
               failinfo.info6),
        LocInfo);
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_REVERSE_TAYLOR_COUNTS_MISMATCH):
    throw ADOLCError(
        format("ADOL-C error: Reverse fails on tape {} because the number of "
               "independent\n and/or dependent variables given to reverse are "
               "inconsistent\n with that of the internal taylor array!\n",
               failinfo.info1),
        LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_PARAM_COUNTS_MISMATCH):
    throw ADOLCError(
        format("ADOL-C error: Setting parameters on tape {} "
               "aborted!\nNumber of parameters ({}) passed"
               " is inconsistent with number recorded on tape ({})\n",
               failinfo.info1, failinfo.info5, failinfo.info6),
        LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_BUFFER_NULLPOINTER_FUNCTION):
    throw ADOLCError("ADOL-C error: nullptr supplied in buffer "
                     "handling.\n",
                     LocInfo);
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_BUFFER_INDEX_TO_LARGE):
    throw ADOLCError("ADOL-C error: Index for buffer element too "
                     "large.\n",
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_EXT_DIFF_NULLPOINTER_STRUCT):
    throw ADOLCError("ADOL-C error: Got nullptr as pointer to struct "
                     " containing ext. diff. function information!\n",
                     LocInfo);
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_EXT_DIFF_WRONG_TAPESTATS):
    throw ADOLCError(
        "ADOL-C error: Number of independents/dependents recorded on"
        " tape differ from number supplied by user!\n",
        LocInfo);
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_EXT_DIFF_NULLPOINTER_FUNCTION):
    throw ADOLCError("ADOL-C error: Got nullptr as "
                     "extern function pointer!\n",
                     LocInfo);
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_EXT_DIFF_NULLPOINTER_DIFFFUNC):
    throw ADOLCError(
        "ADOL-C error: No function for external differentiation found"
        " to work with (nullptr)\n!",
        LocInfo);
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_EXT_DIFF_NULLPOINTER_ARGUMENT):
    throw ADOLCError("ADOL-C error: Got at least one nullptr as argument to"
                     " extern differentiated function!\n",
                     LocInfo);
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_EXT_DIFF_WRONG_FUNCTION_INDEX):
    throw ADOLCError("ADOL-C error: Function with specified index not found!\n",
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_EXT_DIFF_LOCATIONGAP):
    throw ADOLCError(
        "ADOL-C error: active type arguments passed to call_ext_fct do not "
        "have contiguous ascending locations; use "
        "ensureContiguousLocations(size_t) to reserve  contiguous blocks "
        "prior to allocation of the arguments.\n",
        LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_CHECKPOINTING_CPINFOS_NULLPOINTER):
    throw ADOLCError("ADOL-C error: Got nullptr as pointer to struct "
                     " containing checkpointing information!\n",
                     LocInfo);
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_CHECKPOINTING_NULLPOINTER_ARGUMENT):
    throw ADOLCError("ADOL-C error: Got nullptr instead of argument pointer "
                     "within checkpointing infos!\n",
                     LocInfo);
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_CHECKPOINTING_NULLPOINTER_FUNCTION):
    throw ADOLCError("ADOL-C error: Got nullptr instead of function pointer "
                     "within checkpointing infos!\n",
                     LocInfo);
    break;
  case to_underlying(
      ADOLC_ERRORS::ADOLC_CHECKPOINTING_NULLPOINTER_FUNCTION_DOUBLE):
    throw ADOLCError("ADOL-C error: Got nullptr instead of function (double "
                     "version) pointer within checkpointing infos!\n",
                     LocInfo);
    break;
  case to_underlying(
      ADOLC_ERRORS::ADOLC_CHECKPOINTING_REVOLVE_IRREGULAR_TERMINATED):
    throw ADOLCError("ADOL-C error: Irregualar termination of REVOLVE!\n",
                     LocInfo);
    break;
  case to_underlying(
      ADOLC_ERRORS::ADOLC_CHECKPOINTING_UNEXPECTED_REVOLVE_ACTION):
    throw ADOLCError(
        "ADOL-C error: Unextpected REVOLVE action in forward mode!\n", LocInfo);
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_WRONG_PLATFORM_32):
    throw ADOLCError("ADOL-C error: Trace was created on a 64-bit platform, "
                     "cannot be opened on 32-bit platform!\n",
                     LocInfo);
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_WRONG_PLATFORM_64):
    throw ADOLCError("ADOL-C error: Trace was created on a 32-bit platform, "
                     "cannot be opened on 64-bit platform!\n",
                     LocInfo);
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_TAPING_NOT_ACTUALLY_TAPING):
    throw ADOLCError(
        format("ADOL-C error: Trace {} is not being currently created!\n",
               failinfo.info1),
        LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_VEC_LOCATIONGAP):
    throw ADOLCError("ADOL-C error: arrays passed to vector operation do not "
                     "have contiguous ascending locations;\nuse "
                     "dynamic_cast<adouble*>(advector&) \nor call "
                     "ensureContiguousLocations(size_t) to reserve  "
                     "contiguous blocks prior to allocation of the arrays.\n",
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_ENABLE_MINMAX_USING_ABS):
    throw ADOLCError(
        "ADOL-C warning: change from native Min/Max to using Abs during "
        "tracing will lead to inconsistent results, not changing behaviour "
        "now\n call enableMinMaxUsingAbs before trace_on(tapeId) for the "
        "correct behaviour\n",
        LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_DISABLE_MINMAX_USING_ABS):
    throw ADOLCError("ADOL-C warning: change from native Min/Max to using Abs "
                     "during tracing will lead to inconsistent results, not "
                     "changing behaviour now\n call disableMinMaxUsingAbs "
                     "after trace_off() for the correct behaviour\n",
                     LocInfo);
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_NONPOSITIVE_BASIS):
    throw ADOLCError("ADOL-C message: exponent at zero/negative constant "
                     "basis deactivated",
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_MYALLOC1):
    throw ADOLCError(format("ADOL-C error: myalloc1 cannot allocate {} bytes\n",
                            failinfo.info5),
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_MYALLOC2):
    throw ADOLCError(format("ADOL-C error: myalloc2 cannot allocate {} bytes\n",
                            failinfo.info5),
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_MYALLOC3):
    throw ADOLCError(format("ADOL-C error: myalloc3 cannot allocate {} bytes\n",
                            failinfo.info5),
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_MYALLOCI2):
    throw ADOLCError(
        format("ADOL-C error: myallocI2 cannot allocate {} bytes\n",
               failinfo.info5),
        LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_MYALLOC1_UINT):
    throw ADOLCError(
        format("ADOL-C error: myalloc1_uint cannot allocate {} bytes\n",
               failinfo.info5),
        LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_MYALLOC1_ULONG):
    throw ADOLCError(
        format("ADOL-C error: myalloc1_ulong cannot allocate {} bytes\n",
               failinfo.info5),
        LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_MYALLOC2_ULONG):
    throw ADOLCError(
        format("ADOL-C error: myalloc2_ulong cannot allocate {} bytes\n",
               failinfo.info5),
        LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_SM_ACTIVE_VARS):
    throw ADOLCError(
        format("ADOL-C Error: Can not set StorageManagerType, because of "
               "#{} active Variables!",
               failinfo.info5),
        LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_SM_SAME_TYPE):
    throw ADOLCError(format("ADOL-C Error: Given type is the same as the "
                            "current StorageManagerType",
                            failinfo.info5),
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_ADUBREF_CONSTRUCTOR):
    throw ADOLCError(
        format(
            "ADOL-C error: strange construction of an active vector subscript "
            "reference\n(passed ref = {}, stored refloc_ = {})\n",
            failinfo.info5, failinfo.info6),
        LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_ADUBREF_OOB):
    throw ADOLCError(format("ADOL-C warning: index out of bounds while "
                            "subscripting n={}, idx={}\n",
                            failinfo.info5, failinfo.info6),
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_ADVECTOR_NON_DECREASING):
    throw ADOLCError("ADOL-C error: can only call lookup index if advector "
                     "is nondecreasing\n",
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_ADVECTOR_NON_NEGATIVE):
    throw ADOLCError(
        "ADOL-C error: index lookup needs a nonnegative denominator\n",
        LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_FP_NO_EDF):
    throw ADOLCError("ADOL-C Error! No edf found for fixpoint iteration.\n",
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_CP_STORED_EXCEEDS_CU):
    throw ADOLCError("Number of checkpoints stored exceeds checkup! Increase "
                     "constant 'checkup' and recompile!\n",
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_CP_STORED_EXCEEDS_SNAPS):
    throw ADOLCError(
        format("Number of checkpoints stored = {} exceeds snaps = {}! "
               "Ensure 'snaps' > 0 and increase initial 'fine'!\n",
               failinfo.info3, failinfo.info6),
        LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_CP_NUMFORW):
    throw ADOLCError("Error occurred in numforw!\n", LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_CP_INC_SNAPS):
    throw ADOLCError("Enhancement of 'fine', 'snaps' checkpoints stored!\n "
                     "Increase 'snaps'!\n",
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_CP_SNAPS_EXCEEDS_CU):
    throw ADOLCError("Number of snaps exceeds checkup!\n Increase constant "
                     "'checkup' and recompile!\n",
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_CP_REPS_EXCEEDS_REPSUP):
    throw ADOLCError("Number of reps exceeds repsup!\n Increase "
                     "constant 'repsup' and recompile!\n",
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_CP_TAPE_MISMATCH):
    throw ADOLCError(format("CPInfos was constructor for tape with id: {} but "
                            "checkpointing was called with tape id: {}",
                            failinfo.info2, failinfo.info3),
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_CP_NO_SUCH_IDX):
    throw ADOLCError(
        format("There is no CPInfo with index: {}", failinfo.info3), LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_NO_MINMAX):
    throw ADOLCError(format("ADOL-C error: Tape {} is not created compatible "
                            "for abs norm\n Please "
                            "call enableMinMaxUsingAbs() before tracing\n",
                            failinfo.info1),
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_SWITCHES_MISMATCH):
    throw ADOLCError(
        format("ADOL-C error: Number of switches passed {} does not match "
               "with the one recorded on tape {} ({})\n",
               failinfo.info3, failinfo.info1, failinfo.info6),
        LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_REVERSE_NO_FOWARD):
    throw ADOLCError(
        format("ADOL-C error: reverse fails because it was not "
               "preceded by a forward sweep with degree>{}, keep={}!\n",
               failinfo.info3, failinfo.info4),
        LocInfo);
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_ACTIVE_SUBSCRIPTING):
    throw ADOLCError("ADOL-C error: active subscripting does not work in "
                     "safe mode, please use tight mode\n",
                     LocInfo);
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_ADUBREF_SAFE_MODE):
    throw ADOLCError(
        format("ADOL-C error: indexed active position does not match "
               "referenced position\n indexed = {}, referenced = {}\n",
               failinfo.info5, failinfo.info6),
        LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_ADUBREF_VE_REF):
    throw ADOLCError("ADOL-C error: active vector element referencing does "
                     "not work in safe mode, please use tight mode\n",
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_NO_SUCH_OP):
    throw ADOLCError(
        format("ADOL-C fatal error no such operation {}\n", failinfo.info7),
        LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_HO_OP_NOT_IMPLEMENTED):
    throw ADOLCError(
        format("ADOL-C error: higher order mode of op {} not implemented yet\n",
               failinfo.info7),
        LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_WRONG_DIM_Y):
    throw ADOLCError("ADOL-C error: wrong Y dimension in forward \n", LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_WRONG_DIM_XY):
    throw ADOLCError("ADOL-C error:  wrong X and Y dimensions in forward \n",
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_WRONG_DIM_U):
    throw ADOLCError("ADOL-C error:  wrong U dimension in vector-reverse \n",
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_WRONG_DIM_D):
    throw ADOLCError("ADOL-C error:  wrong degree in vector-reverse \n",
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_WRONG_DIM_uZ):
    throw ADOLCError(
        "ADOL-C error:  wrong u or Z dimension in scalar-reverse \n", LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_WRONG_DIM_Z):
    throw ADOLCError("ADOL-C error:  wrong Z dimension in scalar-reverse \n",
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_WRONG_DIM_u):
    throw ADOLCError("ADOL-C error:  wrong u dimension in scalar-reverse \n",
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_PARAM_OOB):
    throw ADOLCError(format("ADOL-C error: Parameter index {} out of bounds, "
                            "# existing parameters = {}\n",
                            failinfo.info5, failinfo.info6),
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::SM_LOCINT_BLOCK):
    throw ADOLCError("ADOL-C error: Location blocks not allowed", LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::SM_MAX_LIVES):
    throw ADOLCError(
        format("maximal number ({}) of live active variables exceeded\n\n",
               failinfo.info5),
        LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_FWD_COUNTS_MISMATCH):
    throw ADOLCError(
        format(
            "ADOL-C error: Forward sweep on tape {} aborted!\n Number of "
            "dependents({}) and/or independents({})\n variables passed to "
            "forward"
            "is inconsistent\n with number recorded on tape : ({} / {})) !\n ",
            failinfo.info1, failinfo.info3, failinfo.info4, failinfo.info5,
            failinfo.info6),
        LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_FWD_FO_KEEP):
    throw ADOLCError("ADOL-C error: first order scalar forward cannot save"
                     " more \nthan first order taylor coefficients!\n",
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_FWD_ZO_KEEP):
    throw ADOLCError("ADOL-C error: zero order scalar forward cannot save"
                     " more \nthan zero order taylor coefficients!\n",
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::DIRGRAD_NOT_ENOUGH_DIRS):
    throw ADOLCError(" NOT ENOUGH DIRECTIONS !!!!\n", LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::SPARSE_BAD_MODE):
    throw ADOLCError("ADOL-C error: bad mode parameter to bit pattern.\n",
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::SPARSE_NO_BP):
    throw ADOLCError("ADOL-C error: no basepoint for bit"
                     " pattern tight.\n",
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_TAPE_DOC_COUNTS_MISMATCH):
    throw ADOLCError(
        format("ADOL-C error: Tape_doc sweep on tape {} aborted!\n Number of "
               "dependents({}) and/or independents({})\n variables passed to "
               "Tape_doc"
               "is inconsistent\n with number recorded on tape : ({} / {}) +) "
               "!\n ",
               failinfo.info1, failinfo.info3, failinfo.info4, failinfo.info5,
               failinfo.info6),
        LocInfo);
    break;
  case to_underlying(ADOLC_ERRORS::ADOLC_CANNOT_OPEN_FILE):
    throw ADOLCError("cannot open file !\n", LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_NO_COLPACK):
    throw ADOLCError("ADOL-C error: No ColPack found!\n", LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_SPARSE_HESS_IND):
    throw ADOLCError("ADOL-C Error: wrong number of independents stored "
                     "in hessian pattern.\n",
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_SPARSE_CRS):
    throw ADOLCError("ADOL-C input parameter crs may not be nullptr!\n",
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_SPARSE_JAC_MALLOC):
    throw ADOLCError(
        format("ADOL-C error: jac_pat(...) unable to allocate {} bytes !\n",
               failinfo.info2),
        LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_SPARSE_JAC_NO_BP):
    throw ADOLCError("ADOL-C no basepoint x for tight mode supplied.\n",
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_NO_TAPE_ID):
    throw ADOLCError(format("No Tape with ID {}!", failinfo.info1), LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_TAPE_ALREADY_EXIST):
    throw ADOLCError(format("Tape with ID {} already exists!", failinfo.info1),
                     LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_TAY_NULLPTR):
    throw ADOLCError("tay_file is nullptr. This should not happen!", LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_SIZE_MISMATCH):
    throw ADOLCError(
        format("ADOL-C error: Used tape ({}) was written "
               "with locations of size {}, but current defined size is {}.\n ",
               failinfo.info1, failinfo.info5, failinfo.info6),
        LocInfo);
    break;

  case to_underlying(ADOLC_ERRORS::ADOLC_TO_MANY_DIRECTIONS):
    throw ADOLCError("To many input directions", LocInfo);
    break;

  default:
    throw ADOLCError("ADOL-C error => unknown error type!\n", LocInfo);
  }
}
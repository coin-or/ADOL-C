
#include <adolc/adolcerror.h>
#include <adolc/dvlparms.h> // for ADOLC version infos
#include <cstring>
#include <iostream>
#include <sstream>

namespace ADOLCError {

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
    message += ">>> " + std::string(std::strerror(errno)) + " <<<\n";
    break;
  }
  std::cout << message << std::endl;
}

// outputs an appropriate error message using ADOLCError and exits the running
// program
void fail(ErrorType error, source_location LocInfo, const FailInfo &failinfo) {
  using std::cerr;
  std::ostringstream oss;

  switch (to_underlying(error)) {
  case to_underlying(ErrorType::MALLOC_FAILED):
    throw ADOLCError("ADOL-C error: Memory allocation failed!\n", LocInfo);
    break;
  case to_underlying(ErrorType::INTEGER_TAPE_FOPEN_FAILED):
  case to_underlying(ErrorType::INTEGER_TAPE_FREAD_FAILED):
    printError();
    oss << "ADOL-C error: reading integer tape number " << failinfo.info1
        << "!\n";
    throw ADOLCError(oss.str(), LocInfo);
    break;
  case to_underlying(ErrorType::VALUE_TAPE_FOPEN_FAILED):
  case to_underlying(ErrorType::VALUE_TAPE_FREAD_FAILED):
    printError();
    oss << "ADOL-C error: reading value tape number " << failinfo.info1
        << "!\n";
    throw ADOLCError(oss.str(), LocInfo);
    break;
  case to_underlying(ErrorType::TAPE_TO_OLD):
    oss << "ADOL-C error: Used tape (" << failinfo.info1
        << ") was written with ADOL-C version older than "
        << ADOLC_NEW_TAPE_VERSION << "." << ADOLC_NEW_TAPE_SUBVERSION << "."
        << ADOLC_NEW_TAPE_PATCHLEVEL << "\n This is ADOL-C " << ADOLC_VERSION
        << "." << ADOLC_SUBVERSION << "." << ADOLC_PATCHLEVEL << "!\n",
        throw ADOLCError(oss.str(), LocInfo);
    break;

  case to_underlying(ErrorType::MORE_STAT_SPACE_REQUIRED):
    throw ADOLCError("ADOL-C error: Not enough space for stats!\n Please "
                     "contact the ADOL-C team!\n",
                     LocInfo);
    break;

  case to_underlying(ErrorType::TAPING_BUFFER_ALLOCATION_FAILED):
    throw ADOLCError("ADOL-C error: Cannot allocate tape buffers!\n", LocInfo);
    break;
  case to_underlying(ErrorType::TAPING_TBUFFER_ALLOCATION_FAILED):
    throw ADOLCError("ADOL-C error: Cannot allocate taylor buffer!\n", LocInfo);
    break;
  case to_underlying(ErrorType::TAPING_READ_ERROR_IN_TAYLOR_CLOSE):
    oss << "ADOL-C error: Read error in taylor_close n= " << failinfo.info1
        << "!\n";
    throw ADOLCError(oss.str(), LocInfo);
    break;
  case to_underlying(ErrorType::TAPING_TO_MANY_TAYLOR_BUFFERS):
    throw ADOLCError("ADOL-C error: To many taylor buffers!\n Increase "
                     "maxNumberTaylorBuffers\n",
                     LocInfo);
    break;
  case to_underlying(ErrorType::TAPING_TO_MANY_LOCINTS):

    oss << "ADOL-C error: Maximal number (" << failinfo.info3
        << ") of live active variables exceeded!\n\n"
        << "Possible remedies :\n\n"
        << "1. Use more automatic local variables and\n"
        << "   allocate/deallocate adoubles on free store\n"
        << "   in a strictly last in first out fashion\n\n"
        << "2. Extend the range by redefining the type of\n"
        << "   locint(currently " << sizeof(size_t)
        << " byte) from unsigned short (" << sizeof(unsigned short)
        << " byte) or int\n"
        << "   to int (" << sizeof(int) << " byte) or long (" << sizeof(long)
        << " byte).\n";
    throw ADOLCError(oss.str(), LocInfo);
    break;
  case to_underlying(ErrorType::TAPING_STORE_REALLOC_FAILED):
    oss << "ADOL-C error: Failure to reallocate storage for adouble values!\n\n"
        << "oldStore = " << failinfo.info5 << "\n"
        << "newStore = nullptr\n"
        << "oldStoreSize = " << failinfo.info3 << "\n"
        << "newStoreSize = " << failinfo.info4 << "\n\n"
        << "Possible remedies :\n"
        << "1. Use more automatic local variables and\n"
        << "   allocate / deallocate adoubles on free store\n"
        << "   in a strictly last in first out fashion\n"
        << "2. Enlarge your system stacksize limit\n";

    throw ADOLCError(oss.str(), LocInfo);
    break;
  case to_underlying(ErrorType::TAPING_FATAL_IO_ERROR):
    printError();
    throw ADOLCError("ADOL-C error: Fatal error-doing a read or write!\n",
                     LocInfo);
    break;
  case to_underlying(ErrorType::TAPING_TAPE_STILL_IN_USE):
    oss << "ADOL-C error: Tape " << failinfo.info1 << " is still in use!\n";
    throw ADOLCError(oss.str(), LocInfo);
    break;
  case to_underlying(ErrorType::TAPING_TAYLOR_OPEN_FAILED):
    printError();
    throw ADOLCError("ADOL-C error: while opening taylor file!\n", LocInfo);
    break;

  case to_underlying(ErrorType::EVAL_SEEK_VALUE_STACK):
    throw ADOLCError("ADOL-C error: in seeking value stack file!\n", LocInfo);
    break;
  case to_underlying(ErrorType::EVAL_OP_TAPE_READ_FAILED):
    throw ADOLCError("ADOL-C error: while reading operations tape!\n", LocInfo);
    break;
  case to_underlying(ErrorType::EVAL_VAL_TAPE_READ_FAILED):
    throw ADOLCError("ADOL-C error: while reading values tape!\n", LocInfo);
    break;
  case to_underlying(ErrorType::EVAL_LOC_TAPE_READ_FAILED):
    throw ADOLCError("ADOL-C error: while reading locations tape!\n", LocInfo);
    break;
  case to_underlying(ErrorType::EVAL_TAY_TAPE_READ_FAILED):
    throw ADOLCError("ADOL-C error: while reading value stack tape!\n",
                     LocInfo);
    break;

  case to_underlying(ErrorType::REVERSE_NO_TAYLOR_STACK):
    oss << "ADOL-C error: No taylor stack found for tape " << failinfo.info1
        << "! => Check forward "
           "sweep!\n",
        throw ADOLCError(oss.str(), LocInfo);
    break;
  case to_underlying(ErrorType::REVERSE_COUNTS_MISMATCH):
    oss << "ADOL-C error: Reverse sweep on tape " << failinfo.info1
        << " aborted!\n"
        << "Number of dependents(" << failinfo.info3 << ") and/or independents("
        << failinfo.info4 << ") variables passed to reverse\n"
        << "is inconsistent with number recorded on tape : (" << failinfo.info5
        << " / " << failinfo.info6 << ") +) !\n";
    throw ADOLCError(oss.str(), LocInfo);
    break;
  case to_underlying(ErrorType::REVERSE_TAYLOR_COUNTS_MISMATCH):
    oss << "ADOL-C error: Reverse fails on tape " << failinfo.info1
        << " because the number of "
           "independent\n and/or dependent variables given to reverse are "
           "inconsistent\n with that of the internal taylor array!\n";
    throw ADOLCError(oss.str(), LocInfo);
    break;

  case to_underlying(ErrorType::PARAM_COUNTS_MISMATCH):
    oss << "ADOL-C error: Setting parameters on tape " << failinfo.info1
        << " aborted!\n"
        << "Number of parameters (" << failinfo.info5 << ") passed "
        << "is inconsistent with number recorded on tape (" << failinfo.info6
        << ")\n";
    throw ADOLCError(oss.str(), LocInfo);
    break;

  case to_underlying(ErrorType::BUFFER_NULLPOINTER_FUNCTION):
    throw ADOLCError("ADOL-C error: nullptr supplied in buffer "
                     "handling.\n",
                     LocInfo);
    break;
  case to_underlying(ErrorType::BUFFER_INDEX_TO_LARGE):
    throw ADOLCError("ADOL-C error: Index for buffer element too "
                     "large.\n",
                     LocInfo);
    break;

  case to_underlying(ErrorType::EXT_DIFF_NULLPOINTER_STRUCT):
    throw ADOLCError("ADOL-C error: Got nullptr as pointer to struct "
                     " containing ext. diff. function information!\n",
                     LocInfo);
    break;
  case to_underlying(ErrorType::EXT_DIFF_WRONG_TAPESTATS):
    throw ADOLCError(
        "ADOL-C error: Number of independents/dependents recorded on"
        " tape differ from number supplied by user!\n",
        LocInfo);
    break;
  case to_underlying(ErrorType::EXT_DIFF_NULLPOINTER_FUNCTION):
    throw ADOLCError("ADOL-C error: Got nullptr as "
                     "extern function pointer!\n",
                     LocInfo);
    break;
  case to_underlying(ErrorType::EXT_DIFF_NULLPOINTER_DIFFFUNC):
    throw ADOLCError(
        "ADOL-C error: No function for external differentiation found"
        " to work with (nullptr)\n!",
        LocInfo);
    break;
  case to_underlying(ErrorType::EXT_DIFF_NULLPOINTER_ARGUMENT):
    throw ADOLCError("ADOL-C error: Got at least one nullptr as argument to"
                     " extern differentiated function!\n",
                     LocInfo);
    break;
  case to_underlying(ErrorType::EXT_DIFF_WRONG_FUNCTION_INDEX):
    throw ADOLCError("ADOL-C error: Function with specified index not found!\n",
                     LocInfo);
    break;

  case to_underlying(ErrorType::EXT_DIFF_LOCATIONGAP):
    throw ADOLCError(
        "ADOL-C error: active type arguments passed to call_ext_fct do not "
        "have contiguous ascending locations; use "
        "ensureContiguousLocations(size_t) to reserve  contiguous blocks "
        "prior to allocation of the arguments.\n",
        LocInfo);
    break;

  case to_underlying(ErrorType::CHECKPOINTING_CPINFOS_NULLPOINTER):
    throw ADOLCError("ADOL-C error: Got nullptr as pointer to struct "
                     " containing checkpointing information!\n",
                     LocInfo);
    break;
  case to_underlying(ErrorType::CHECKPOINTING_NULLPOINTER_ARGUMENT):
    throw ADOLCError("ADOL-C error: Got nullptr instead of argument pointer "
                     "within checkpointing infos!\n",
                     LocInfo);
    break;
  case to_underlying(ErrorType::CHECKPOINTING_NULLPOINTER_FUNCTION):
    throw ADOLCError("ADOL-C error: Got nullptr instead of function pointer "
                     "within checkpointing infos!\n",
                     LocInfo);
    break;
  case to_underlying(ErrorType::CHECKPOINTING_NULLPOINTER_FUNCTION_DOUBLE):
    throw ADOLCError("ADOL-C error: Got nullptr instead of function (double "
                     "version) pointer within checkpointing infos!\n",
                     LocInfo);
    break;
  case to_underlying(ErrorType::CHECKPOINTING_REVOLVE_IRREGULAR_TERMINATED):
    throw ADOLCError("ADOL-C error: Irregualar termination of REVOLVE!\n",
                     LocInfo);
    break;
  case to_underlying(ErrorType::CHECKPOINTING_UNEXPECTED_REVOLVE_ACTION):
    throw ADOLCError(
        "ADOL-C error: Unextpected REVOLVE action in forward mode!\n", LocInfo);
    break;
  case to_underlying(ErrorType::WRONG_PLATFORM_32):
    throw ADOLCError("ADOL-C error: Trace was created on a 64-bit platform, "
                     "cannot be opened on 32-bit platform!\n",
                     LocInfo);
    break;
  case to_underlying(ErrorType::WRONG_PLATFORM_64):
    throw ADOLCError("ADOL-C error: Trace was created on a 32-bit platform, "
                     "cannot be opened on 64-bit platform!\n",
                     LocInfo);
    break;
  case to_underlying(ErrorType::TAPING_NOT_ACTUALLY_TAPING):
    oss << "ADOL-C error: Trace " << failinfo.info1
        << " is not being currently created!\n";
    throw ADOLCError(oss.str(), LocInfo);
    break;

  case to_underlying(ErrorType::VEC_LOCATIONGAP):
    throw ADOLCError("ADOL-C error: arrays passed to vector operation do not "
                     "have contiguous ascending locations;\nuse "
                     "dynamic_cast<adouble*>(advector&) \nor call "
                     "ensureContiguousLocations(size_t) to reserve  "
                     "contiguous blocks prior to allocation of the arrays.\n",
                     LocInfo);
    break;

  case to_underlying(ErrorType::ENABLE_MINMAX_USING_ABS):
    throw ADOLCError(
        "ADOL-C warning: change from native Min/Max to using Abs during "
        "tracing will lead to inconsistent results, not changing behaviour "
        "now\n call enableMinMaxUsingAbs before trace_on(tapeId) for the "
        "correct behaviour\n",
        LocInfo);
    break;

  case to_underlying(ErrorType::DISABLE_MINMAX_USING_ABS):
    throw ADOLCError("ADOL-C warning: change from native Min/Max to using Abs "
                     "during tracing will lead to inconsistent results, not "
                     "changing behaviour now\n call disableMinMaxUsingAbs "
                     "after trace_off() for the correct behaviour\n",
                     LocInfo);
    break;
  case to_underlying(ErrorType::NONPOSITIVE_BASIS):
    throw ADOLCError("ADOL-C message: exponent at zero/negative constant "
                     "basis deactivated",
                     LocInfo);
    break;

  case to_underlying(ErrorType::MYALLOC1):
    oss << "ADOL-C error: myalloc1 cannot allocate " << failinfo.info5
        << " bytes\n",
        throw ADOLCError(oss.str(), LocInfo);
    break;

  case to_underlying(ErrorType::MYALLOC2):
    oss << "ADOL-C error: myalloc2 cannot allocate " << failinfo.info5
        << " bytes\n",
        throw ADOLCError(oss.str(), LocInfo);
    break;

  case to_underlying(ErrorType::MYALLOC3):
    oss << "ADOL-C error: myalloc3 cannot allocate " << failinfo.info5
        << " bytes\n";
    throw ADOLCError(oss.str(), LocInfo);
    break;

  case to_underlying(ErrorType::MYALLOCI2):
    oss << "ADOL-C error: myallocI2 cannot allocate " << failinfo.info5
        << " bytes\n";
    throw ADOLCError(oss.str(), LocInfo);
    break;

  case to_underlying(ErrorType::MYALLOC1_UINT):
    oss << "ADOL-C error: myalloc1_uint cannot allocate " << failinfo.info5
        << " bytes\n";
    throw ADOLCError(oss.str(), LocInfo);
    break;

  case to_underlying(ErrorType::MYALLOC1_ULONG):
    oss << "ADOL-C error: myalloc1_ulong cannot allocate " << failinfo.info5
        << " bytes\n";
    throw ADOLCError(oss.str(), LocInfo);
    break;

  case to_underlying(ErrorType::MYALLOC2_ULONG):
    oss << "ADOL-C error: myalloc2_ulong cannot allocate " << failinfo.info5
        << " bytes\n";
    throw ADOLCError(oss.str(), LocInfo);
    break;

  case to_underlying(ErrorType::SM_ACTIVE_VARS):
    oss << "ADOL-C Error: Can not set StorageManagerType, because of "
           "#"
        << failinfo.info5 << " active Variables!";
    throw ADOLCError(oss.str(), LocInfo);
    break;

  case to_underlying(ErrorType::SM_SAME_TYPE):
    oss << "ADOL-C Error: Given type " << failinfo.info5
        << " is the same as the current StorageManagerType";
    throw ADOLCError(oss.str(), LocInfo);
    break;

  case to_underlying(ErrorType::ADUBREF_CONSTRUCTOR):
    oss << "ADOL-C error: strange construction of an active vector subscript "
           "reference\n(passed ref = "
        << failinfo.info5 << ", stored refloc_ = " << failinfo.info6 << ")\n";
    throw ADOLCError(oss.str(), LocInfo);
    break;

  case to_underlying(ErrorType::ADUBREF_OOB):
    oss << "ADOL-C warning: index out of bounds while "
           "subscripting n="
        << failinfo.info5 << ", idx=" << failinfo.info6 << "\n";
    throw ADOLCError(oss.str(), LocInfo);
    break;

  case to_underlying(ErrorType::ADVECTOR_NON_DECREASING):
    throw ADOLCError("ADOL-C error: can only call lookup index if advector "
                     "is nondecreasing\n",
                     LocInfo);
    break;

  case to_underlying(ErrorType::ADVECTOR_NON_NEGATIVE):
    throw ADOLCError(
        "ADOL-C error: index lookup needs a nonnegative denominator\n",
        LocInfo);
    break;

  case to_underlying(ErrorType::FP_NO_EDF):
    throw ADOLCError("ADOL-C Error! No edf found for fixpoint iteration.\n",
                     LocInfo);
    break;

  case to_underlying(ErrorType::CP_STORED_EXCEEDS_CU):
    throw ADOLCError("Number of checkpoints stored exceeds checkup! Increase "
                     "constant 'checkup' and recompile!\n",
                     LocInfo);
    break;

  case to_underlying(ErrorType::CP_STORED_EXCEEDS_SNAPS):
    oss << "Number of checkpoints stored = " << failinfo.info3
        << " exceeds snaps = " << failinfo.info6
        << "! Ensure 'snaps' > 0 and increase initial 'fine'!\n";
    throw ADOLCError(oss.str(), LocInfo);
    break;

  case to_underlying(ErrorType::CP_NUMFORW):
    throw ADOLCError("Error occurred in numforw!\n", LocInfo);
    break;

  case to_underlying(ErrorType::CP_INC_SNAPS):
    throw ADOLCError("Enhancement of 'fine', 'snaps' checkpoints stored!\n "
                     "Increase 'snaps'!\n",
                     LocInfo);
    break;

  case to_underlying(ErrorType::CP_SNAPS_EXCEEDS_CU):
    throw ADOLCError("Number of snaps exceeds checkup!\n Increase constant "
                     "'checkup' and recompile!\n",
                     LocInfo);
    break;

  case to_underlying(ErrorType::CP_REPS_EXCEEDS_REPSUP):
    throw ADOLCError("Number of reps exceeds repsup!\n Increase "
                     "constant 'repsup' and recompile!\n",
                     LocInfo);
    break;

  case to_underlying(ErrorType::CP_TAPE_MISMATCH):
    oss << "CPInfos was constructor for tape with id: " << failinfo.info2
        << " but checkpointing was called with tape id: " << failinfo.info3
        << std::endl;
    throw ADOLCError(oss.str(), LocInfo);
    break;

  case to_underlying(ErrorType::CP_NO_SUCH_IDX):
    oss << "There is no CPInfo with index: " << failinfo.info3 << std::endl;
    throw ADOLCError(oss.str(), LocInfo);
    break;

  case to_underlying(ErrorType::NO_MINMAX):
    oss << "ADOL-C error: Tape " << failinfo.info1
        << " is not created compatible "
           "for abs norm\n Please "
           "call enableMinMaxUsingAbs() before tracing\n";
    throw ADOLCError(oss.str(), LocInfo);
    break;

  case to_underlying(ErrorType::SWITCHES_MISMATCH):
    oss << "ADOL-C error: Number of switches passed " << failinfo.info3
        << " does not match with the one recorded on tape " << failinfo.info1
        << " (" << failinfo.info6 << ")\n";
    throw ADOLCError(oss.str(), LocInfo);
    break;

  case to_underlying(ErrorType::REVERSE_NO_FOWARD):
    oss << "ADOL-C error: reverse fails because it was not "
           "preceded by a forward sweep with degree>"
        << failinfo.info3 << ", keep=" << failinfo.info4 << "!\n";
    throw ADOLCError(oss.str(), LocInfo);
    break;
  case to_underlying(ErrorType::ACTIVE_SUBSCRIPTING):
    throw ADOLCError("ADOL-C error: active subscripting does not work in "
                     "safe mode, please use tight mode\n",
                     LocInfo);
    break;
  case to_underlying(ErrorType::ADUBREF_SAFE_MODE):
    oss << "ADOL-C error: indexed active position does not match "
           "referenced position\n indexed = "
        << failinfo.info5 << ", referenced = " << failinfo.info6 << "\n";
    throw ADOLCError(oss.str(), LocInfo);
    break;

  case to_underlying(ErrorType::ADUBREF_VE_REF):
    throw ADOLCError("ADOL-C error: active vector element referencing does "
                     "not work in safe mode, please use tight mode\n",
                     LocInfo);
    break;

  case to_underlying(ErrorType::NO_SUCH_OP):
    oss << "ADOL-C fatal error no such operation " << failinfo.info7 << " \n";
    throw ADOLCError(oss.str(), LocInfo);
    break;

  case to_underlying(ErrorType::HO_OP_NOT_IMPLEMENTED):
    oss << "ADOL-C error: higher order mode of op " << failinfo.info7
        << " not implemented yet\n";
    throw ADOLCError(oss.str(), LocInfo);
    break;

  case to_underlying(ErrorType::WRONG_DIM_Y):
    throw ADOLCError("ADOL-C error: wrong Y dimension in forward \n", LocInfo);
    break;

  case to_underlying(ErrorType::WRONG_DIM_XY):
    throw ADOLCError("ADOL-C error: wrong X and Y dimensions in forward \n",
                     LocInfo);
    break;

  case to_underlying(ErrorType::WRONG_DIM_U):
    throw ADOLCError("ADOL-C error: wrong U dimension in vector-reverse \n",
                     LocInfo);
    break;

  case to_underlying(ErrorType::WRONG_DIM_D):
    throw ADOLCError("ADOL-C error: wrong degree in vector-reverse \n",
                     LocInfo);
    break;

  case to_underlying(ErrorType::WRONG_DIM_uZ):
    throw ADOLCError(
        "ADOL-C error:  wrong u or Z dimension in scalar-reverse \n", LocInfo);
    break;

  case to_underlying(ErrorType::WRONG_DIM_Z):
    throw ADOLCError("ADOL-C error:  wrong Z dimension in scalar-reverse \n",
                     LocInfo);
    break;

  case to_underlying(ErrorType::WRONG_DIM_u):
    throw ADOLCError("ADOL-C error:  wrong u dimension in scalar-reverse \n",
                     LocInfo);
    break;

  case to_underlying(ErrorType::PARAM_OOB):
    oss << "ADOL-C error: Parameter index " << failinfo.info5
        << " out of bounds, # existing parameters = " << failinfo.info6 << "\n";
    throw ADOLCError(oss.str(), LocInfo);
    break;

  case to_underlying(ErrorType::SM_LOCINT_BLOCK):
    throw ADOLCError("ADOL-C error: Location blocks not allowed", LocInfo);
    break;

  case to_underlying(ErrorType::SM_MAX_LIVES):
    oss << "maximal number (" << failinfo.info5
        << ") of live active variables exceeded\n\n";
    throw ADOLCError(oss.str(), LocInfo);
    break;

  case to_underlying(ErrorType::FWD_COUNTS_MISMATCH):
    oss << "ADOL-C error: Forward sweep on tape " << failinfo.info1
        << " aborted!\n Number of dependents(" << failinfo.info3
        << ") and/or independents(" << failinfo.info4
        << ")\n variables passed to forward is inconsistent\n with number "
           "recorded on tape : ( "
        << failinfo.info5 << " / " << failinfo.info6 << ")) !\n ";
    throw ADOLCError(oss.str(), LocInfo);
    break;

  case to_underlying(ErrorType::FWD_FO_KEEP):
    throw ADOLCError("ADOL-C error: first order scalar forward cannot save"
                     " more \nthan first order taylor coefficients!\n",
                     LocInfo);
    break;

  case to_underlying(ErrorType::FWD_ZO_KEEP):
    throw ADOLCError("ADOL-C error: zero order scalar forward cannot save"
                     " more \nthan zero order taylor coefficients!\n",
                     LocInfo);
    break;

  case to_underlying(ErrorType::DIRGRAD_NOT_ENOUGH_DIRS):
    throw ADOLCError(" NOT ENOUGH DIRECTIONS !!!!\n", LocInfo);
    break;

  case to_underlying(ErrorType::SPARSE_BAD_MODE):
    throw ADOLCError("ADOL-C error: bad mode parameter to bit pattern.\n",
                     LocInfo);
    break;

  case to_underlying(ErrorType::SPARSE_NO_BP):
    throw ADOLCError("ADOL-C error: no basepoint for bit"
                     " pattern tight.\n",
                     LocInfo);
    break;

  case to_underlying(ErrorType::TAPE_DOC_COUNTS_MISMATCH):
    oss << "ADOL-C error: Tape_doc sweep on tape " << failinfo.info1
        << " aborted!\n Number of dependents(" << failinfo.info3
        << ") and/or independents(" << failinfo.info4
        << ")\n variables passed to Tape_doc is inconsistent\n with number "
           "recorded on tape : ("
        << failinfo.info5 << " / " << failinfo.info6 << ") +)!\n ";
    throw ADOLCError(oss.str(), LocInfo);
    break;
  case to_underlying(ErrorType::CANNOT_OPEN_FILE):
    throw ADOLCError("cannot open file !\n", LocInfo);
    break;

  case to_underlying(ErrorType::NO_COLPACK):
    throw ADOLCError("ADOL-C error: No ColPack found!\n", LocInfo);
    break;

  case to_underlying(ErrorType::SPARSE_HESS_IND):
    throw ADOLCError("ADOL-C Error: wrong number of independents stored "
                     "in hessian pattern.\n",
                     LocInfo);
    break;

  case to_underlying(ErrorType::SPARSE_CRS):
    throw ADOLCError("ADOL-C input parameter crs may not be nullptr!\n",
                     LocInfo);
    break;

  case to_underlying(ErrorType::SPARSE_JAC_MALLOC):
    oss << "ADOL-C error: jac_pat(...) unable to allocate " << failinfo.info2
        << " bytes !\n";
    throw ADOLCError(oss.str(), LocInfo);
    break;

  case to_underlying(ErrorType::SPARSE_JAC_NO_BP):
    throw ADOLCError("ADOL-C no basepoint x for tight mode supplied.\n",
                     LocInfo);
    break;

  case to_underlying(ErrorType::NO_TAPE_ID):
    oss << "No Tape with ID " << failinfo.info1 << "!";
    throw ADOLCError(oss.str(), LocInfo);
    break;

  case to_underlying(ErrorType::TAPE_ALREADY_EXIST):
    oss << "Tape with ID " << failinfo.info1 << " already exists!";
    throw ADOLCError(oss.str(), LocInfo);
    break;

  case to_underlying(ErrorType::TAY_NULLPTR):
    throw ADOLCError("tay_file is nullptr. This should not happen!", LocInfo);
    break;

  case to_underlying(ErrorType::SIZE_MISMATCH):
    oss << "ADOL-C error: Used tape (" << failinfo.info1
        << ") was written with locations of size " << failinfo.info5
        << ", but current defined size is " << failinfo.info6 << ".\n ";
    throw ADOLCError(oss.str(), LocInfo);
    break;

  case to_underlying(ErrorType::TO_MANY_DIRECTIONS):
    throw ADOLCError("To many input directions", LocInfo);
    break;

  default:
    throw ADOLCError("ADOL-C error => unknown error type!\n", LocInfo);
  }
}

} // namespace ADOLCError

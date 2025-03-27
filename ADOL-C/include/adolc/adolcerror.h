/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     adolcerror.h
 Revision: $Id$
 Contents: handling of fatal errors

 Copyright (c) Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

---------------------------------------------------------------------------*/

#ifndef ADOLC_ERROR_H
#define ADOLC_ERROR_H

#include <format>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

/**
 * @brief Exception class for ADOL-C errors with source location tracking.
 *
 * This exception captures the error message and the source location (file,
 * line, column) where it was thrown. Integrates with the C++ standard exception
 * hierarchy through std::runtime_error.
 *
 * @example
 * Throw example:
 * @code
 * throw ADOLCError("wrong number of independents");  // Auto-captures source
 * location
 * @endcode
 *
 * Handle example:
 * @code
 * try {
 *   // ADOL-C operations...
 * }
 * catch (ADOLCError& e) {
 *   std::cerr << e.what() << std::endl;         // Formatted message
 *   std::cerr << "Error occurred at: "
 *             << e.where().file_name() << ":"
 *             << e.where().line() << std::endl; // Direct source location
 * access
 * }
 * @endcode
 */
class ADOLCError : public std::runtime_error {
  std::source_location info_;

public:
  /**
   * @brief Constructs an ADOLCError with message and source location
   * @param message Error description (will be stored as std::string)
   * @param info Source location (automatically captured if not provided)
   */
  ADOLCError(std::string_view message,
             std::source_location info = std::source_location::current())
      : std::runtime_error(std::format("{}\n[Source] {}:{}:{}", message,
                                       info.file_name(), info.line(),
                                       info.column())),
        info_(info) // std::source_location is trivially copyable
  {}

  /**
   * @brief Access the captured source location
   * @return Const reference to the source_location object
   */
  const std::source_location &where() const noexcept { return info_; }
};

enum class ADOLC_ERRORS : size_t {
  ADOLC_MALLOC_FAILED,
  ADOLC_INTEGER_TAPE_FOPEN_FAILED,
  ADOLC_INTEGER_TAPE_FREAD_FAILED,
  ADOLC_VALUE_TAPE_FOPEN_FAILED,
  ADOLC_VALUE_TAPE_FREAD_FAILED,
  ADOLC_TAPE_TO_OLD,
  ADOLC_WRONG_LOCINT_SIZE,
  ADOLC_MORE_STAT_SPACE_REQUIRED,
  ADOLC_ACTIVE_SUBSCRIPTING,
  ADOLC_NO_SUCH_OP,
  ADOLC_CANNOT_OPEN_FILE,
  ADOLC_NO_MINMAX,
  ADOLC_SWITCHES_MISMATCH,
  ADOLC_NO_COLPACK,
  ADOLC_NO_TAPE_ID,
  ADOLC_TAPE_ALREADY_EXIST,
  ADOLC_TAY_NULLPTR,
  ADOLC_SIZE_MISMATCH,
  ADOLC_TO_MANY_DIRECTIONS,

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
  ADOLC_REVERSE_NO_FOWARD,

  ADOLC_PARAM_COUNTS_MISMATCH,

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
  ADOLC_CP_STORED_EXCEEDS_CU,
  ADOLC_CP_STORED_EXCEEDS_SNAPS,
  ADOLC_CP_NUMFORW,
  ADOLC_CP_INC_SNAPS,
  ADOLC_CP_SNAPS_EXCEEDS_CU,
  ADOLC_CP_REPS_EXCEEDS_REPSUP,
  ADOLC_CP_TAPE_MISMATCH,
  ADOLC_CP_NO_SUCH_IDX,

  ADOLC_WRONG_PLATFORM_32,
  ADOLC_WRONG_PLATFORM_64,

  ADOLC_TAPING_NOT_ACTUALLY_TAPING,
  ADOLC_VEC_LOCATIONGAP,

  ADOLC_ENABLE_MINMAX_USING_ABS,
  ADOLC_DISABLE_MINMAX_USING_ABS,
  ADOLC_NONPOSITIVE_BASIS,

  ADOLC_MYALLOC1,
  ADOLC_MYALLOC2,
  ADOLC_MYALLOC3,
  ADOLC_MYALLOCI2,
  ADOLC_MYALLOC1_UINT,
  ADOLC_MYALLOC1_ULONG,
  ADOLC_MYALLOC2_ULONG,

  ADOLC_SM_ACTIVE_VARS,
  ADOLC_SM_SAME_TYPE,
  SM_LOCINT_BLOCK,
  SM_MAX_LIVES,

  ADOLC_ADUBREF_CONSTRUCTOR,
  ADOLC_ADUBREF_OOB,
  ADOLC_ADUBREF_SAFE_MODE,
  ADOLC_ADUBREF_VE_REF,
  ADOLC_ADVECTOR_NON_DECREASING,
  ADOLC_ADVECTOR_NON_NEGATIVE,

  ADOLC_FP_NO_EDF,
  ADOLC_HO_OP_NOT_IMPLEMENTED,

  ADOLC_WRONG_DIM_Y,
  ADOLC_WRONG_DIM_XY,
  ADOLC_WRONG_DIM_U,
  ADOLC_WRONG_DIM_D,
  ADOLC_WRONG_DIM_uZ,
  ADOLC_WRONG_DIM_Z,
  ADOLC_WRONG_DIM_u,

  ADOLC_PARAM_OOB,

  ADOLC_FWD_COUNTS_MISMATCH,
  ADOLC_FWD_FO_KEEP,
  ADOLC_FWD_ZO_KEEP,

  DIRGRAD_NOT_ENOUGH_DIRS,

  SPARSE_BAD_MODE,
  SPARSE_NO_BP,

  ADOLC_TAPE_DOC_COUNTS_MISMATCH,

  ADOLC_SPARSE_HESS_IND,
  ADOLC_SPARSE_CRS,
  ADOLC_SPARSE_JAC_MALLOC,
  ADOLC_SPARSE_JAC_NO_BP

};

// wrapper for information of errors
struct FailInfo {
  short info1{0}; // for tapeId
  size_t info2{0};
  int info3{0}; // deps
  int info4{0}; // num indeps
  size_t info5{0};
  size_t info6{0};
  unsigned char info7{0}; // Operation char
};

// like c++23's to_underlying
// help to statically cast an enum class to its parent type
// used e.g. for enum class that inherits from size_t
template <typename T> constexpr auto to_underlying(T t) noexcept {
  return static_cast<std::underlying_type_t<T>>(t);
}

/*--------------------------------------------------------------------------*/
// prints an error message describing the error type coming from a file function
// (e.g. fopen)
void printError();

// would like to use always std::source_location::current() as default but this
// does not work for all compilers sometimes the funciton names are empty.
// Currently you have to call std::source_location::current() when calling fail
// by hand
void fail(ADOLC_ERRORS error,
          const std::source_location LocInfo = std::source_location::current(),
          const FailInfo &failinfo = FailInfo());
#endif // ADOLC_ERROR_H

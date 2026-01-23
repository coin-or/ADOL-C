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

#include <adolc/adolcexport.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

// Macro to obtain location information easily
#define CURRENT_LOCATION                                                       \
  ADOLCError::ADOLCError::makeSourceLocation(__FILE__, __func__, __LINE__)

namespace ADOLCError {

// std::source_location is too new, implement own variant
// used in combination with macro CURRENT_LOCATION
struct source_location {

  std::string_view file_;
  std::string_view func_;
  int line_;

  constexpr source_location(std::string_view file, std::string_view func,
                            int line)
      : file_(file), func_(func), line_(line) {}

  constexpr std::string_view file() const { return file_; }
  constexpr std::string_view func() const { return func_; }
  constexpr int line() const { return line_; }
};
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
 *             << e.where().file() << ":"
 *             << e.where().func() << ":"
 *             << e.where().line() << std::endl; // Direct source location
 * access
 * }
 * @endcode
 */
class ADOLCError : public std::runtime_error {

  source_location info_;

public:
  /**
   * @brief Constructs an ADOLCError with message and source location
   * @param message Error description (will be stored as std::string)
   * @param info Source location
   */
  ADOLCError(std::string_view message, const source_location &info)
      : std::runtime_error(makeMessage(message, info)), info_(info) {}

  constexpr static source_location
  makeSourceLocation(std::string_view file, std::string_view func, int line) {
    return source_location(file, func, line);
  }

  static std::string makeMessage(std::string_view message,
                                 const source_location &info) {
    std::ostringstream oss;
    oss << message << "\n[Source] " << info.file() << ":" << info.func() << ":"
        << info.line();
    return oss.str();
  }

  /**
   * @brief Access the captured source location
   * @return Const reference to the source_location object
   */
  const source_location &where() const noexcept { return info_; }
};

enum class ErrorType : size_t {
  MALLOC_FAILED,
  INTEGER_TAPE_FOPEN_FAILED,
  INTEGER_TAPE_FREAD_FAILED,
  VALUE_TAPE_FOPEN_FAILED,
  VALUE_TAPE_FREAD_FAILED,
  TAPE_TO_OLD,
  WRONG_LOCINT_SIZE,
  MORE_STAT_SPACE_REQUIRED,
  ACTIVE_SUBSCRIPTING,
  NO_SUCH_OP,
  CANNOT_OPEN_FILE,
  CANNOT_REMOVE_FILE,
  NO_MINMAX,
  SWITCHES_MISMATCH,
  NO_COLPACK,
  NO_TAPE_ID,
  TAPE_ALREADY_EXIST,
  TAY_NULLPTR,
  SIZE_MISMATCH,
  TO_MANY_DIRECTIONS,

  TAPING_BUFFER_ALLOCATION_FAILED,
  TAPING_TBUFFER_ALLOCATION_FAILED,
  TAPING_READ_ERROR_IN_TAYLOR_CLOSE,
  TAPING_TO_MANY_LOCINTS,
  TAPING_STORE_REALLOC_FAILED,
  TAPING_FATAL_IO_ERROR,
  TAPING_TAPE_STILL_IN_USE,
  TAPING_TAYLOR_OPEN_FAILED,
  TAPING_NOT_ACTUALLY_TAPING,

  EVAL_SEEK_VALUE_STACK,
  EVAL_OP_TAPE_READ_FAILED,
  EVAL_VAL_TAPE_READ_FAILED,
  EVAL_LOC_TAPE_READ_FAILED,
  EVAL_TAY_TAPE_READ_FAILED,

  REVERSE_NO_TAYLOR_STACK,
  REVERSE_COUNTS_MISMATCH,
  REVERSE_TAYLOR_COUNTS_MISMATCH,
  REVERSE_NO_FOWARD,

  PARAM_COUNTS_MISMATCH,

  BUFFER_NULLPOINTER_FUNCTION,
  BUFFER_INDEX_TO_LARGE,

  EXT_DIFF_NULLPOINTER_STRUCT,
  EXT_DIFF_WRONG_TAPESTATS,
  EXT_DIFF_NULLPOINTER_FUNCTION,
  EXT_DIFF_NULLPOINTER_DIFFFUNC,
  EXT_DIFF_NULLPOINTER_ARGUMENT,
  EXT_DIFF_WRONG_FUNCTION_INDEX,
  EXT_DIFF_LOCATIONGAP,

  CHECKPOINTING_CPINFOS_NULLPOINTER,
  CHECKPOINTING_NULLPOINTER_ARGUMENT,
  CHECKPOINTING_NULLPOINTER_FUNCTION,
  CHECKPOINTING_NULLPOINTER_FUNCTION_DOUBLE,
  CHECKPOINTING_REVOLVE_IRREGULAR_TERMINATED,
  CHECKPOINTING_UNEXPECTED_REVOLVE_ACTION,
  CP_STORED_EXCEEDS_CU,
  CP_STORED_EXCEEDS_SNAPS,
  CP_NUMFORW,
  CP_INC_SNAPS,
  CP_SNAPS_EXCEEDS_CU,
  CP_REPS_EXCEEDS_REPSUP,
  CP_TAPE_MISMATCH,
  CP_NO_SUCH_IDX,

  WRONG_PLATFORM_32,
  WRONG_PLATFORM_64,

  VEC_LOCATIONGAP,

  ENABLE_MINMAX_USING_ABS,
  DISABLE_MINMAX_USING_ABS,
  NONPOSITIVE_BASIS,

  MYALLOC1,
  MYALLOC2,
  MYALLOC3,
  MYALLOCI2,
  MYALLOC1_UINT,
  MYALLOC1_ULONG,
  MYALLOC2_ULONG,

  SM_ACTIVE_VARS,
  SM_SAME_TYPE,
  SM_LOCINT_BLOCK,
  SM_MAX_LIVES,

  ADUBREF_CONSTRUCTOR,
  ADUBREF_OOB,
  ADUBREF_SAFE_MODE,
  ADUBREF_VE_REF,
  ADVECTOR_NON_DECREASING,
  ADVECTOR_NON_NEGATIVE,

  FP_NO_EDF,
  HO_OP_NOT_IMPLEMENTED,

  WRONG_DIM_Y,
  WRONG_DIM_XY,
  WRONG_DIM_U,
  WRONG_DIM_D,
  WRONG_DIM_uZ,
  WRONG_DIM_Z,
  WRONG_DIM_u,

  PARAM_OOB,

  FWD_COUNTS_MISMATCH,
  FWD_FO_KEEP,
  FWD_ZO_KEEP,

  DIRGRAD_NOT_ENOUGH_DIRS,

  SPARSE_BAD_MODE,
  SPARSE_NO_BP,

  TAPE_DOC_COUNTS_MISMATCH,

  SPARSE_HESS_IND,
  SPARSE_CRS,
  SPARSE_JAC_MALLOC,
  SPARSE_JAC_NO_BP,

  NOT_IMPLEMENTED
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
void ADOLC_API printError();

// would like to use always CURRENT_LOCATION as default but this
// does not work for all compilers sometimes the funciton names are empty.
// Currently you have to call CURRENT_LOCATION when calling fail
// by hand
void ADOLC_API fail(ErrorType error, const source_location LocInfo,
                    const FailInfo &failinfo = FailInfo());

} // namespace ADOLCError
#endif // ADOLC_ERROR_H

#ifndef ADOLC_TAPE_INTERFACE_H
#define ADOLC_TAPE_INTERFACE_H

#include <adolc/adolcerror.h>
#include <adolc/adolcexport.h>
#include <adolc/valuetape/valuetape.h>
#include <cassert>
#include <stack>
#include <vector>

using ValueTapeStack = std::stack<ValueTape *, std::vector<ValueTape *>>;
/**
 * @brief Returns a thread-local stack holding pointers to ValueTape instances.
 *
 * This stack is modified in trace_on and trace_off. In trace_on the pointer of
 * a possible previous current tape is pushed onto the stack. Then, the tape
 * to be used is set to be the currentTape. Inside trace_off the pointer of the
 * old tape is popped from the stack, and set as current tape again. This
 * allows the nesting of trace_on ... trace_off calls.
 *
 * @return Reference to the thread-local stack of tape pointers.
 */
inline ValueTapeStack &currentTapeStack() {
  thread_local ValueTapeStack cTStack;
  return cTStack;
}

/**
 * @brief Returns a thread-local reference to the pointer to the current
 * ValueTape.
 *
 * This pointer is specific to the thread and denotes the current tape. The
 * current tape is used for creating new tape_locations, free locations, storing
 * traced operations, etc...
 *
 * @return Reference to the thread-local ValueTape pointer.
 */
inline ValueTape *&currentTapePtr() {
  thread_local ValueTape *currTapePtr = nullptr;
  return currTapePtr;
}

/**
 * @brief Sets the current tape pointer to the given tape.
 *
 * @param tape The tape to set as the current one
 */
ADOLC_API inline void setCurrentTapePtr(ValueTape *tapePtr) noexcept {
  currentTapePtr() = tapePtr;
}

/*
 * @brief Initializes taping for the given tapeId.
 *
 * Prepares a ValueTape for recording operations, storing the current tapeId
 * for later restoration. Initializes the tape’s memory and metadata.
 *
 * @param tapeId ID of the tape to activate for tracing.
 * @param keepTaylors Flag indicating whether to keep Taylor coefficients
 * (non-zero to keep).
 * @return 1 on success, 0 on failure.
 *
 * @throws ADOLCError::ErrorType::NO_TAPE_ID if the specified tape does not
 * exist.
 */
ADOLC_API int trace_on(ValueTape &tape, int keepTaylors = 0);

/**
 * @brief Initializes taping with fine-grained buffer configuration.
 *
 * Prepares a ValueTape with custom buffer sizes for operations, locations,
 * values, and Taylor coefficients. Also allows disabling cleanup of
 * intermediate files.
 *
 * @param tapeId ID of the tape to trace.
 * @param keepTaylors Whether to store Taylor coefficients.
 * @param obs Operation buffer size.
 * @param lbs Location buffer size.
 * @param vbs Value buffer size.
 * @param tbs Taylor buffer size.
 * @param skipFileCleanup Whether to skip cleaning up tape files after tracing.
 * @return 1 on success, -1 on failure.
 *
 * @throws ADOLCError::ErrorType::NO_TAPE_ID if the specified tape does not
 * exist.
 */
ADOLC_API int trace_on(ValueTape &tape, int keepTaylors, size_t obs, size_t lbs,
                       size_t vbs, size_t tbs, int skipFileCleanup);

/**
 * @brief Stops the current tracing session and handles the tape.
 *
 * Cleans up the tape, copies remaining live variables, and updates its status.
 * Optionally writes tape files to disk depending on the value of `flag`.
 *
 * @param flag If non-zero, forces the tape to write op/loc/val files.
 *
 * @throws ADOLCError::ErrorType::TAPING_NOT_ACTUALLY_TAPING if no active
 * tracing is detected.
 *
 * @note assert in debug mode if currentTape is nullptr
 */
ADOLC_API void trace_off(ValueTape &tape, int flag = 0);

/**
 * @brief Sets the tape to nestec_ctx
 *
 * @param tapeId the ID of the tape to set nested
 * @param nested char to set
 *
 * @throws ADOLCError::ErrorType::NO_TAPE_ID if the specified tape does not
 * exist.
 */
ADOLC_API inline void set_nested_ctx(ValueTape &tape, char nested) {
  tape.set_nested_ctx(nested);
}

/**
 * @brief Read the nested state of the tape
 *
 * @param tapeId the ID of the tape to set nested
 * @return char that characterizes the stated of the nesting
 *
 * @throws ADOLCError::ErrorType::NO_TAPE_ID if the specified tape does not
 * exist.
 */
ADOLC_API inline char currently_nested(ValueTape &tape) {
  return tape.currently_nested();
}

/**
 * @brief Writes the tapestats of the tape into an array
 *
 * @param tapeId the ID of the tape to read the tapestats
 * @return array of size TapeInfos::STAT_SIZE that stores all tape's metadata
 *
 * @throws ADOLCError::ErrorType::NO_TAPE_ID if the specified tape does not
 * exist.
 */
ADOLC_API inline std::array<size_t, TapeInfos::STAT_SIZE>
tapestats(ValueTape &tape) {
  std::array<size_t, TapeInfos::STAT_SIZE> stats{};
  tape.tapestats(stats.data());
  return stats;
}

/**
 * @brief Prints the stats of the tape
 *
 * @param tapeId the ID of the tape to print the stats
 *
 * @throws ADOLCError::ErrorType::NO_TAPE_ID if the specified tape does not
 * exist.
 */
ADOLC_API void printTapeStats(ValueTape &tape);

/**
 * @brief Returns the number of parameters recorded on tape
 *
 * @param tapeId ID of the tape to read the number of parameters
 * @return number of parameters
 *
 * @throws ADOLCError::ErrorType::NO_TAPE_ID if the specified tape does not
 * exist.
 */
ADOLC_API inline size_t get_num_param(ValueTape &tape) {
  return tape.get_num_param();
}

#ifdef SPARSE

/**
 * @brief Sets the Sparse Jacbian Information of the tape
 *
 * @param tapeId ID of the tape to store the jacobian information
 * @param sJinfos the information to store
 *
 * @throws ADOLCError::ErrorType::NO_TAPE_ID if the specified tape does not
 * exist.
 */
ADOLC_API void setTapeInfoJacSparse(ValueTape &tape, SparseJacInfos sJinfos) {
  tape.setTapeInfoJacSparse(sJinfos);
}

/**
 * @brief Sets the Sparse Hessian Information of the tape
 *
 * @param tapeId ID of the tape to store the hessian information
 * @param sHInfos the information to store
 *
 * @throws ADOLCError::ErrorType::NO_TAPE_ID if the specified tape does not
 * exist.
 */
ADOLC_API void setTapeInfoHessSparse(ValueTape &tape, SparseHessInfos sHInfos) {
  tape.setTageInfoHessSparse(sHInfos);
}
#endif

#endif // ADOLC_TAPE_INTERFACE_H

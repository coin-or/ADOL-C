#ifndef ADOLC_TAPE_INTERFACE_H
#define ADOLC_TAPE_INTERFACE_H

#include <adolc/adolcerror.h>
#include <adolc/adolcexport.h>
#include <adolc/valuetape/valuetape.h>
#include <algorithm>
#include <cassert>
#include <memory>
#include <vector>

/**
 * @brief Returns a thread-local vector that holds unique pointers to ValueTape
 * instances.
 *
 * This buffer serves as storage for all tapes created in the current thread.
 * Each thread has its own isolated buffer, enabling thread-safe tape
 * management.
 *
 * @return Reference to the thread-local vector of unique pointers to ValueTape.
 */
inline std::vector<std::unique_ptr<ValueTape>> &tapeBuffer() {
  thread_local std::vector<std::unique_ptr<ValueTape>> tBuffer;
  return tBuffer;
}

/**
 * @brief Returns a thread-local vector holding tapeIDs.
 *
 * This buffer is modified in trace_on and trace_off. In trace_on the ID of an
 * old currentTape is pushed into the buffer, then a (potentially) new tape is
 * set to the currentTape. Inside trace_off the ID of the old tape is poped from
 * the buffer and set as currentTape.
 *
 * @return Reference to the thread-local vector of tape IDs.
 */
inline std::vector<short> &tapeIdBuffer() {
  thread_local std::vector<short> tIdBuffer;
  return tIdBuffer;
}

/**
 * @brief Attempts to find a ValueTape pointer by tapeId without triggering an
 * error.
 *
 * @param tapeId The ID of the tape to search for.
 * @return Pointer to the matching ValueTape if found, or nullptr otherwise.
 */
inline ValueTape *findTapePtr_(short tapeId) {
  auto tape_iter =
      std::find_if(tapeBuffer().begin(), tapeBuffer().end(),
                   [&tapeId](auto &&tape) { return tape->tapeId() == tapeId; });
  return (tape_iter != tapeBuffer().end()) ? tape_iter->get() : nullptr;
}

/**
 * @brief Returns a pointer to a ValueTape by tape ID, or throws an error if not
 * found.
 *
 * Performs a lookup for the given tape ID in the thread-local tape buffer.
 * If no matching tape is found, an ADOLCError is thrown.
 *
 * @param tapeId The ID of the tape to locate.
 * @return Pointer to the corresponding ValueTape.
 *
 * @throws ADOLCError::ErrorType::NO_TAPE_ID if the specified tape does not
 * exist.
 */
inline ValueTape *findTapePtr(short tapeId) {
  ValueTape *tape = findTapePtr_(tapeId);
  if (!tape)
    ADOLCError::fail(ADOLCError::ErrorType::NO_TAPE_ID, CURRENT_LOCATION,
                     ADOLCError::FailInfo{.info1 = tapeId});
  return tape;
}

/**
 * @brief Returns a reference to a ValueTape by tapeId, or throws an error if
 * not found.
 *
 * This is a convenience wrapper around findTapePtr that dereferences the
 * pointer.
 *
 * @param tapeId The ID of the tape to retrieve.
 * @return Reference to the corresponding ValueTape.
 *
 * @throws ADOLCError::ErrorType::NO_TAPE_ID if the specified tape does not
 * exist.
 */
ADOLC_API inline ValueTape &findTape(short tapeId) {
  return *findTapePtr(tapeId);
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
 * @brief Returns a reference to the current ValueTape.
 *
 * The current tape is used for creating new tape_locations, free locations,
 * storing traced operations, etc...
 *
 * @return Reference to the current ValueTape.
 *
 * @note Asserts if the current tape pointer is null.
 */
ADOLC_API inline ValueTape &currentTape() {
  assert(currentTapePtr() && "Current Tape is nullptr!");
  return *currentTapePtr();
}
/**
 * @brief Sets the current tape pointer to the pointer of the tape with the
 * specified ID.
 *
 * @param tapeId The ID of the tape to set as current.
 *
 * @throws ADOLCError::ErrorType::NO_TAPE_ID if the specified tape does not
 * exist.
 */
ADOLC_API inline void setCurrentTape(short tapeId) {
  currentTapePtr() = findTapePtr(tapeId);
}

/**
 * @brief Creates a new tape with the specified ID and sets it as current if
 * none exists.
 *
 * Ensures that no duplicate tapeId is used within the thread-local buffer.
 * If the current tape is not yet set, the newly created tape becomes the
 * current tape.
 *
 * @param tapeId The ID of the new tape to create.
 *
 * @throws ADOLCError::ErrorType::TAPE_ALREADY_EXIST if a tape with the same ID
 * already exists.
 */
ADOLC_API inline void createNewTape(short tapeId) {
  // try to find tape
  if (findTapePtr_(tapeId))
    ADOLCError::fail(ADOLCError::ErrorType::TAPE_ALREADY_EXIST,
                     CURRENT_LOCATION, ADOLCError::FailInfo{.info1 = tapeId});

  tapeBuffer().emplace_back(std::make_unique<ValueTape>(tapeId));

  // set the current tape to the newly created one
  if (!currentTapePtr())
    setCurrentTape(tapeId);
}

/**
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
ADOLC_API int trace_on(short tapeId, int keepTaylors = 0);

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
ADOLC_API int trace_on(short tapeId, int keepTaylors, size_t obs, size_t lbs,
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
ADOLC_API void trace_off(int flag = 0);

/**
 * @brief Writes the tapeIds of the tapes stored in the thread-local tapeBuffer
 * in `result`
 *
 * @param result vector to store the IDs
 *
 * @note the vector is resized in this function to the size of the tapeBuffer
 */
ADOLC_API void cachedTraceTags(std::vector<short> &result);

/**
 * @brief Sets the tape to nestec_ctx
 *
 * @param tapeId the ID of the tape to set nested
 * @param nested char to set
 *
 * @throws ADOLCError::ErrorType::NO_TAPE_ID if the specified tape does not
 * exist.
 */
ADOLC_API inline void set_nested_ctx(short tapeId, char nested) {
  findTape(tapeId).set_nested_ctx(nested);
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
ADOLC_API inline char currently_nested(short tapeId) {
  return findTape(tapeId).currently_nested();
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
tapestats(short tapeId) {
  ValueTape &tape = findTape(tapeId);
  std::array<size_t, TapeInfos::STAT_SIZE> stats;
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
ADOLC_API void printTapeStats(int tapeId);

/**
 * @brief Returns the number of parameters recorded on tape
 *
 * @param tapeId ID of the tape to read the number of parameters
 * @return number of parameters
 *
 * @throws ADOLCError::ErrorType::NO_TAPE_ID if the specified tape does not
 * exist.
 */
ADOLC_API inline size_t get_num_param(short tapeId) {
  return findTape(tapeId).get_num_param();
}

#ifdef ADOLC_SPARSE
namespace ADOLC::Sparse {
/**
 * @brief Sets the Sparse Jacbian Information of the tape
 *
 * @param tapeId ID of the tape to store the jacobian information
 * @param sJInfos the information to store
 *
 * @throws ADOLCError::ErrorType::NO_TAPE_ID if the specified tape does not
 * exist.
 */
ADOLC_API inline void setTapeInfoJacSparse(short tapeId,
                                           SparseJacInfos &&sJInfos) {
  findTape(tapeId).setTapeInfoJacSparse(std::move(sJInfos));
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
ADOLC_API inline void setTapeInfoHessSparse(short tapeId,
                                            SparseHessInfos &&sHInfos) {
  findTape(tapeId).setTapeInfoHessSparse(std::move(sHInfos));
}

/**
 * @brief Sets the Hessian sparsity pattern for a given tape.
 *
 * This function stores a user-provided Hessian sparsity pattern
 * (`HPIn`) for the taped function identified by `tapeId`.
 * It allows advanced users to bypass the internal sparsity pattern
 * computation (`hess_pat()`) when the pattern is already known.
 *
 * @param tapeId
 *        Identifier of the taped function.
 * @param indep
 *        Number of independent variables for the taped function.
 * @param HPIn
 *        Pointer to the compressed row storage (CRS) representation
 *        of the upper-triangular part of the Hessian sparsity pattern:
 *        - `HPIn[i][0]` → number of nonzero entries in row `i`
 *        - `HPIn[i][1..]` → column indices (1-based) of the nonzeros.
 *
 * @note
 *  - Only the upper-triangular part of the Hessian needs to be provided
 *    in `HPIn`.
 *  - This function transfers ownership of the CRS structure into the
 *    tape's internal storage.
 *  - Use this function only if you already know the exact sparsity
 *    pattern and want to skip its recomputation.
 *
 * @see getHP
 * @see hess_pat
 */
ADOLC_API inline void setHP(short tapeId, int indep,
                            std::vector<uint *> &&HPIn) {
  SparseHessInfos sHinfos;
  sHinfos.setHP(indep, std::move(HPIn));
  findTape(tapeId).setTapeInfoHessSparse(std::move(sHinfos));
}

/**
 * @brief Retrieves the Hessian sparsity pattern from a given tape.
 *
 * This function returns the Hessian sparsity pattern currently stored
 * in the tape identified by `tapeId`. The sparsity pattern must have
 * been either:
 *  - Computed internally by `hess_pat()`, or
 *  - Set explicitly via `setHP()`.
 *
 * @param tapeId
 *        Identifier of the taped function.
 * @param HPOut
 *        Output: Pointer to the CRS representation of the
 *        upper-triangular Hessian sparsity pattern:
 *        - `(*HPOut)[i][0]` → number of nonzero entries in row `i`
 *        - `(*HPOut)[i][1..]` → column indices (1-based) of the nonzeros.
 *
 * @note
 *  - The returned CRS structure is owned by the tape; do not free it.
 *  - Only the upper-triangular part of the Hessian is stored and
 *    returned.
 *
 * @see setHP
 * @see hess_pat
 */
ADOLC_API inline std::vector<uint *> getHP(short tapeId) {
  return findTape(tapeId).sHInfos().getHP();
}

} // namespace ADOLC::Sparse
#endif // ADOLC_SPARSE

#endif // ADOLC_TAPE_INTERFACE_H
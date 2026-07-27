#ifndef ADOLC_TAPE_REGISTRY_H
#define ADOLC_TAPE_REGISTRY_H

#include <adolc/adolcexport.h>
#include <cassert>
#include <memory>
#include <vector>

// forward declaration to use as pointer
class ValueTape;

/**
 * @brief One entry in the thread-local stack of nested recordings.
 *
 * `trace_on()` stores the previously selected tape and the tape being recorded;
 * `trace_off()` uses the frame to restore the earlier selection. The stack is
 * thread-local, while the registry that owns the tapes is process-wide.
 */
struct RecordingFrame {
  ValueTape *prev{nullptr};
  ValueTape *current{nullptr};
};

/**
 * @brief Returns the process-wide vector that owns all ValueTape instances.
 *
 * Tape creation and lookup synchronize access to this registry internally.
 * Pointers to contained tapes remain stable because the vector stores owning
 * pointers and tapes are not removed.
 *
 * @return Reference to the process-wide vector ValueTape pointers.
 */
ADOLC_API std::vector<std::unique_ptr<ValueTape>> &tapeBuffer();

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
inline std::vector<RecordingFrame> &currentTapeStack() {
  thread_local std::vector<RecordingFrame> cTStack{RecordingFrame{}};
  return cTStack;
}
/**
 * @brief Attempts to find a ValueTape pointer by tapeId without triggering an
 * error.
 *
 * @param tapeId The ID of the tape to search for.
 * @return Pointer to the matching ValueTape if found, or nullptr otherwise.
 */
ADOLC_API ValueTape *findTapePtr_(short tapeId);

/**
 * @brief Returns a pointer to a ValueTape by tape ID, or throws an error if not
 * found.
 *
 * Performs a synchronized lookup in the process-wide tape registry.
 * If no matching tape is found, an ADOLCError is thrown.
 *
 * @param tapeId The ID of the tape to locate.
 * @return Pointer to the corresponding ValueTape.
 *
 * @throws ADOLCError::ErrorType::NO_TAPE_ID if the specified tape does not
 * exist.
 */
ADOLC_API ValueTape *findTapePtr(short tapeId);

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
ADOLC_API ValueTape &findTape(short tapeId);

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
inline ADOLC_API ValueTape *&currentTapePtr() {
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
inline ValueTape &currentTape() {
  assert(currentTapePtr() && "Current Tape is nullptr!");
  return *currentTapePtr();
}

/**
 * @brief Sets the current tape pointer to the given tape.
 *
 * @param tape The tape to set as the current one
 */
inline void setCurrentTape(ValueTape *tape) noexcept {
  currentTapePtr() = tape;
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
ADOLC_API void setCurrentTape(short tapeId);
/**
 * @brief Creates a new tape and returns its ID.
 *
 * Allocates a process-wide unique tape ID and inserts the new tape into the
 * global registry.
 * If the calling thread has no current tape, the newly created tape becomes
 * that thread's current tape.
 *
 * @return The ID of the new created tape.
 */
ADOLC_API short createNewTape();

#endif // ADOLC_TAPE_REGISTRY_H

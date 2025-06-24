#ifndef ADOLC_TAPE_INTERFACE_H
#define ADOLC_TAPE_INTERFACE_H

#include <adolc/adolcerror.h>
#include <adolc/valuetape/valuetape.h>
#include <algorithm>
#include <cassert>
#include <memory>
#include <vector>

inline std::vector<std::unique_ptr<ValueTape>> &tapeBuffer() {
  thread_local std::vector<std::unique_ptr<ValueTape>> tBuffer;
  return tBuffer;
}

inline std::vector<short> &tapeIdBuffer() {
  thread_local std::vector<short> tIdBuffer;
  return tIdBuffer;
}

inline ValueTape *findTapePtr_(short tapeId) {
  auto tape_iter =
      std::find_if(tapeBuffer().begin(), tapeBuffer().end(),
                   [&tapeId](auto &&tape) { return tape->tapeId() == tapeId; });

  return (tape_iter != tapeBuffer().end()) ? tape_iter->get() : nullptr;
}

inline ValueTape *findTapePtr(short tapeId) {
  ValueTape *tape = findTapePtr_(tapeId);
  if (!tape)
    ADOLCError::fail(ADOLCError::ErrorType::NO_TAPE_ID, CURRENT_LOCATION,
                     ADOLCError::FailInfo{.info1 = tapeId});
  return tape;
}

inline ValueTape &findTape(short tapeId) { return *findTapePtr(tapeId); }

// points to the current tape
inline ValueTape *&currentTapePtr() {
  thread_local ValueTape *currTapePtr = nullptr;
  return currTapePtr;
}

inline ValueTape &currentTape() {
  assert(currentTapePtr() && "Current Tape is nullptr!");
  return *currentTapePtr();
}

inline void setCurrentTape(short tapeId) {
  currentTapePtr() = findTapePtr(tapeId);
}

// creates tape and set to default!
inline void createNewTape(short tapeId) {
  // try to find tape
  if (findTapePtr_(tapeId))
    ADOLCError::fail(ADOLCError::ErrorType::TAPE_ALREADY_EXIST,
                     CURRENT_LOCATION, ADOLCError::FailInfo{.info1 = tapeId});

  tapeBuffer().emplace_back(std::make_unique<ValueTape>(tapeId));

  // set the current tape to the newly created one
  if (!currentTapePtr())
    setCurrentTape(tapeId);
}

size_t get_num_param(short ta);
void cachedTraceTags(std::vector<short> &result);

int trace_on(short tapeId, int keepTaylors = 0);
void trace_off(int flag = 0);

/***************************************************************************/
/* Initialization for the taping process. Creates buffers for this tape,    */
/* sets files names, and calls appropriate setup routines.                  */
/****************************************************************************/
int trace_on(short tapeId, int keepTaylors);

int trace_on(short tapeId, int keepTaylors, size_t obs, size_t lbs, size_t vbs,
             size_t tbs, int skipFileCleanup);

/****************************************************************************/
/* Stop Tracing. Cleans up, and turns off trace_flag. Flag not equal zero   */
/* enforces writing of the three main tape files (op+loc+val).              */
/****************************************************************************/
void trace_off(int flag);

void cachedTraceTags(std::vector<short> &result);

/****************************************************************************/
/* Set a trace to nested_ctx                                                */
/****************************************************************************/
inline void set_nested_ctx(short tapeId, char nested) {
  findTape(tapeId).set_nested_ctx(nested);
}

/****************************************************************************/
/* Check whether a tape has been set to nested_ctx                          */
/****************************************************************************/
inline char currently_nested(short tapeId) {
  return findTape(tapeId).currently_nested();
}

inline std::array<size_t, TapeInfos::STAT_SIZE> tapestats(short tapeId) {
  /* get the tapeInfos for tapeId */
  ValueTape &tape = findTape(tapeId);
  std::array<size_t, TapeInfos::STAT_SIZE> stats;
  tape.tapestats(stats.data());
  return stats;
}

/****************************************************************************/
/* An all-in-one tape stats printing routine.                               */
/****************************************************************************/

void printTapeStats(int tapeId);

/****************************************************************************/
/* Returns the number of parameters recorded on tape                        */
/****************************************************************************/
inline size_t get_num_param(short tapeId) {
  return findTape(tapeId).get_num_param();
}

#ifdef SPARSE
void setTapeInfoJacSparse(short tapeId, SparseJacInfos sJinfos) {
  findTape(tapeId).setTapeInfoJacSparse(sJinfos);
}
void setTapeInfoHessSparse(short tapeId, SparseHessInfos sHInfos) {
  findTape(tapeId).setTageInfoHessSparse(sHInfos);
}
#endif

#endif // ADOLC_TAPE_INTERFACE_H
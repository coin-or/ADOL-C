#ifndef ADOLC_TAPE_INTERFACE_H
#define ADOLC_TAPE_INTERFACE_H

#include <adolc/adolcerror.h>
#include <adolc/valuetape/valuetape.h>
#include <cassert>
#include <memory>
#include <vector>

inline std::vector<std::unique_ptr<ValueTape>> &tapeBuffer() {
  static std::vector<std::unique_ptr<ValueTape>> tBuffer;
  return tBuffer;
}

inline std::vector<short> &tapeIdBuffer() {
  static std::vector<short> tIdBuffer;
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
    fail(ADOLC_ERRORS::ADOLC_NO_TAPE_ID, std::source_location::current(),
         FailInfo{.info1 = tapeId});
  return tape;
}

inline ValueTape &findTape(short tapeId) { return *findTapePtr(tapeId); }

// points to the current tape
inline ValueTape *&currentTapePtr() {
  static ValueTape *currTapePtr = nullptr;
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
    fail(ADOLC_ERRORS::ADOLC_TAPE_ALREADY_EXIST,
         std::source_location::current(), FailInfo{.info1 = tapeId});

  tapeBuffer().emplace_back(std::make_unique<ValueTape>(tapeId));

  // set the current tape to the newly created one
  if (!currentTapePtr()) {
    setCurrentTape(tapeId);
    std::cout << "ADOLC: Set current Tape to Tape with Id: " << tapeId
              << std::endl;
  }
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
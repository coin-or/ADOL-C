#ifndef ADOLC_TAPE_INTERFACE_H
#define ADOLC_TAPE_INTERFACE_H

#include <adolc/valuetape/valuetape.h>
#include <memory>
#include <vector>

std::vector<std::shared_ptr<ValueTape>> &getTapeBuffer();
short &getDefaultTapeId();
const short &getDefaultTapeIdConst();
void setDefaultTapeId(short default_tape_id);
void setDefaultTape(std::shared_ptr<ValueTape> tape);

std::shared_ptr<ValueTape> getDefaultTape();
std::shared_ptr<ValueTape> getTape(short tapeId);

void throw_if_exist(short tapeId);
size_t get_num_param(short tag);
std::shared_ptr<ValueTape> findTape(short tapeId);
void cachedTraceTags(std::vector<short> &result);
std::shared_ptr<ValueTape> getOrMakeTape(short tapeId);

int trace_on(short tapeId, int keepTaylors = 0);
void trace_off(short tapeId, int flag = 0);

/***************************************************************************/
/* Initialization for the taping process. Creates buffers for this tape,    */
/* sets files names, and calls appropriate setup routines.                  */
/****************************************************************************/
int trace_on(short tapeId, int keepTaylors);

int trace_on(short tnum, int keepTaylors, size_t obs, size_t lbs, size_t vbs,
             size_t tbs, int skipFileCleanup);

/****************************************************************************/
/* Stop Tracing. Cleans up, and turns off trace_flag. Flag not equal zero   */
/* enforces writing of the three main tape files (op+loc+val).              */
/****************************************************************************/
void trace_off(short tapeId, int flag);

void cachedTraceTags(std::vector<short> &result);

/****************************************************************************/
/* Set a trace to nested_ctx                                                */
/****************************************************************************/
inline void set_nested_ctx(short tag, char nested) {
  std::shared_ptr<ValueTape> tape = getTape(tag);
  tape->set_nested_ctx(nested);
}

/****************************************************************************/
/* Check whether a tape has been set to nested_ctx                          */
/****************************************************************************/
inline char currently_nested(short tag) {
  std::shared_ptr<ValueTape> tape = getTape(tag);
  return tape->currently_nested();
}

inline std::array<size_t, TapeInfos::STAT_SIZEE> tapestats(short tapeId) {
  /* get the tapeInfos for tag */
  std::shared_ptr<ValueTape> tape = getTape(tapeId);
  std::array<size_t, TapeInfos::STAT_SIZEE> stats;
  tape->tapestats(stats.data());
  return stats;
}

/****************************************************************************/
/* An all-in-one tape stats printing routine.                               */
/****************************************************************************/

void printTapeStats(int tag);

/****************************************************************************/
/* Returns the number of parameters recorded on tape                        */
/****************************************************************************/
inline size_t get_num_param(short tag) { return getTape(tag)->get_num_param(); }

#ifdef SPARSE
void setTapeInfoJacSparse(short tapeId, SparseJacInfos sJinfos);
void setTapeInfoHessSparse(short tapeId, SparseHessInfos sHInfos);
#endif

#endif // ADOLC_TAPE_INTERFACE_H
/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     taping.c
 Revision: $Id$
 Contents: all C functions directly accessing at least one of the four tapes
           (operations, locations, constants, value stack)

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz,
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel,
               Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#include <adolc/adolcerror.h>
#include <adolc/tape_interface.h>
#include <adolc/valuetape/tapeinfos.h>
#include <adolc/valuetape/valuetape.h>
#include <array>
#include <cassert>
#include <format>
#include <iostream>
#include <memory>
#include <vector>

#ifdef ADOLC_MEDIPACK_SUPPORT
#include <adolc/medipacksupport_p.h>
#endif // ADOLC_MEDIPACK_SUPPORT

#ifdef ADOLC_AMPI_SUPPORT
#include "ampi/ampi.h"
#include "ampi/tape/support.h"
#endif // ADOLC_AMPI_SUPPORT

std::vector<std::shared_ptr<ValueTape>> &getTapeBuffer() {
  static std::vector<std::shared_ptr<ValueTape>> tape_buffer;
  return tape_buffer;
}

short &getDefaultTapeId() {
  static short default_tape_id = 0;
  return default_tape_id;
}
const short &getDefaultTapeIdConst() { return getDefaultTapeId(); }
void setDefaultTapeId(short default_id) { getDefaultTapeId() = default_id; }

void setDefaultTape(std::shared_ptr<ValueTape> tape) {
  getDefaultTapeId() = tape->tapeId();
}

std::shared_ptr<ValueTape> getDefaultTape() {
  return findTape(getDefaultTapeIdConst());
}
std::shared_ptr<ValueTape> getTape(short tapeId) { return findTape(tapeId); }

// pulls the tape with tapeId if exists or creates a new one at the end of
// getTapeBuffer()
std::shared_ptr<ValueTape> getOrMakeTape(short tapeId) {
  try {
    return getTape(tapeId);
  } catch (const ADOLCError &e) {
    getTapeBuffer().emplace_back(std::make_shared<ValueTape>(tapeId));
    return getTapeBuffer().back();
  }
}

std::shared_ptr<ValueTape> findTape_(short tapeId) {
  auto tape_iter = std::find_if(
      getTapeBuffer().begin(), getTapeBuffer().end(),
      [&tapeId](auto &&tapePtr) { return tapePtr->tapeId() == tapeId; });

  return (tape_iter != getTapeBuffer().end()) ? *tape_iter : nullptr;
}

std::shared_ptr<ValueTape> findTape(short tapeId) {
  std::shared_ptr<ValueTape> tape = findTape_(tapeId);
  if (!tape)
    fail(ADOLC_ERRORS::ADOLC_NO_TAPE_ID, std::source_location::current(),
         FailInfo{.info1 = tapeId});
  return tape;
}

void throw_if_exist(short tapeId) {
  std::shared_ptr<ValueTape> tape = findTape_(tapeId);
  if (!tape)
    fail(ADOLC_ERRORS::ADOLC_TAPE_ALREADY_EXIST,
         std::source_location::current(), FailInfo{.info1 = tapeId});
}

/****************************************************************************/
/* Initialization for the taping process. Creates buffers for this tape,    */
/* sets files names, and calls appropriate setup routines.                  */
/****************************************************************************/
int trace_on(short tapeId, int keepTaylors) {
  std::shared_ptr<ValueTape> tape = getOrMakeTape(tapeId);
  int retval = tape->initNewTape();

#ifdef ADOLC_MEDIPACK_SUPPORT
  tape->mediInitTape(tnum);
#endif

  tape->keepTaylors(keepTaylors);
  tape->tapestats(TapeInfos::NO_MIN_MAX, tape->nominmaxFlag());

  if (keepTaylors)
    tape->deg_save(1);
  tape->start_trace();
  tape->take_stock(); /* record all existing adoubles on the tape */

  return retval;
}

int trace_on(short tnum, int keepTaylors, size_t obs, size_t lbs, size_t vbs,
             size_t tbs, int skipFileCleanup) {
  int retval = 0;

  std::shared_ptr<ValueTape> tape = getOrMakeTape(tnum);
  retval = tape->initNewTape();
  if (retval) {
#ifdef ADOLC_MEDIPACK_SUPPORT
    tape->mediInitTape(tnum);
#endif
    // reset the tape buffers
    tape->freeTapeResources();
    tape->tapestats(TapeInfos::OP_BUFFER_SIZE, obs);
    tape->tapestats(TapeInfos::LOC_BUFFER_SIZE, lbs);
    tape->tapestats(TapeInfos::TAY_BUFFER_SIZE, tbs);
    tape->keepTaylors(keepTaylors);
    tape->tapestats(TapeInfos::NO_MIN_MAX, tape->nominmaxFlag());
    tape->skipFileCleanup(skipFileCleanup);
    if (keepTaylors != 0)
      tape->deg_save(1);
    tape->start_trace();
    tape->take_stock(); /* record all existing adoubles on the tape */
    return retval;
  } else
    return -1;
}

/****************************************************************************/
/* Stop Tracing. Cleans up, and turns off trace_flag. Flag not equal zero   */
/* enforces writing of the three main tape files (op+loc+val).              */
/****************************************************************************/
void trace_off(short tapeId, int flag) {
  std::shared_ptr<ValueTape> tape = getTape(tapeId);
  if (tape->workMode() != TapeInfos::ADOLC_TAPING)
    fail(ADOLC_ERRORS::ADOLC_TAPING_NOT_ACTUALLY_TAPING,
         std::source_location::current(), FailInfo{.info1 = tape->tapeId()});
  tape->keepTape(flag);
  tape->keep_stock(); /* copy remaining live variables + trace_flag = 0 */
  tape->stop_trace(flag);
  tape->tapingComplete(1);
  tape->workMode(TapeInfos::ADOLC_NO_MODE);
  tape->releaseTape();
}

void cachedTraceTags(std::vector<short> &result) {
  if (!getTapeBuffer().empty()) {
    result.resize(getTapeBuffer().size());
    for (size_t i = 0; i < getTapeBuffer().size(); ++i) {
      result[i] = getTapeBuffer()[i]->tapeId();
    }
  }
}

/****************************************************************************/
/* An all-in-one tape stats printing routine.                               */
/****************************************************************************/

void printTapeStats(int tag) {
  using std::cout;
  using std::endl;
  using std::format;

  auto stats = tapestats(tag);

  cout << format("\n*** TAPE STATS (tape {}) **********\n", tag)
       << format("Number of independents: {:>10}\n",
                 stats[TapeInfos::NUM_INDEPENDENTS])
       << format("Number of dependents:   {:>10}\n",
                 stats[TapeInfos::NUM_DEPENDENTS])
       << endl;
  cout << format("Max # of live adoubles: {:>10}\n",
                 stats[TapeInfos::NUM_MAX_LIVES])
       << format("Taylor stack size:      {:>10}\n",
                 stats[TapeInfos::TAY_STACK_SIZE])
       << endl;
  cout << format("Number of operations:   {:>10}\n",
                 stats[TapeInfos::NUM_OPERATIONS])
       << format("Number of locations:    {:>10}\n",
                 stats[TapeInfos::NUM_LOCATIONS])
       << format("Number of values:       {:>10}\n",
                 stats[TapeInfos::NUM_VALUES])
       << format("Number of parameters:   {:>10}\n",
                 stats[TapeInfos::NUM_PARAM])
       << endl;

  cout << format("Operation file written: {:>10}\n",
                 stats[TapeInfos::OP_FILE_ACCESS])
       << format("Location file written:  {:>10}\n",
                 stats[TapeInfos::LOC_FILE_ACCESS])
       << format("Value file written:     {:>10}\n",
                 stats[TapeInfos::VAL_FILE_ACCESS])
       << endl;
  cout << format("Operation buffer size:  {:>10}\n",
                 stats[TapeInfos::OP_BUFFER_SIZE])
       << format("Location buffer size:   {:>10}\n",
                 stats[TapeInfos::LOC_BUFFER_SIZE])
       << format("Value buffer size:      {:>10}\n",
                 stats[TapeInfos::VAL_BUFFER_SIZE])
       << format("Taylor buffer size:     {:>10}\n",
                 stats[TapeInfos::TAY_BUFFER_SIZE])
       << endl;
  cout << format("Operation type size:    {:>10}\n", sizeof(unsigned char))
       << format("Location type size:     {:>10}\n", sizeof(size_t))
       << format("Value type size:        {:>10}\n", sizeof(double))
       << format("Taylor type size:       {:>10}\n", sizeof(double))
       << "**********************************\n"
       << endl;
}

#ifdef SPARSE
void setTapeInfoJacSparse(short tapeId, SparseJacInfos sJinfos) {
  std::vector<std::shared_ptr<ValueTape>>::iterator tape_iter =
      std::find_if(getTapeBuffer().begin(), getTapeBuffer().end(),
                   [&tapeId](auto &&ti) { return ti->tapeId() == tapeId; });
  std::shared_ptr<ValueTape> tape =
      (tape_iter != getTapeBuffer().end()) ? *tape_iter : nullptr;

  // Return the tapeInfos pointer if it has been found.
  if (tape) {
    tape->setTapeInfoJacSparse(sJinfos);
  }
}
void setTapeInfoHessSparse(short tapeId, SparseHessInfos sHInfos) {
  std::vector<std::shared_ptr<ValueTape>>::iterator tape_iter =
      std::find_if(getTapeBuffer().begin(), getTapeBuffer().end(),
                   [&tapeId](auto &&ti) { return ti->tapeId() == tapeId; });
  std::shared_ptr<ValueTape> tape =
      (tape_iter != getTapeBuffer().end()) ? *tape_iter : nullptr;

  // Return the tapeInfos pointer if it has been found.
  if (tape) {
    tape->setTageInfoHessSparse(sHInfos);
  }
}
#endif
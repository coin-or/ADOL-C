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
#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#ifdef ADOLC_MEDIPACK_SUPPORT
#include <adolc/medipacksupport_p.h>
#endif // ADOLC_MEDIPACK_SUPPORT

#ifdef ADOLC_AMPI_SUPPORT
#include "ampi/ampi.h"
#include "ampi/tape/support.h"
#endif // ADOLC_AMPI_SUPPORT

int trace_on(short tapeId, int keepTaylors) {

  ValueTape &tape = findTape(tapeId);
  // store Id to restore it after trace_off
  tapeIdBuffer().push_back(currentTape().tapeId());
  setCurrentTape(tapeId);

  int retval = tape.initNewTape();
#ifdef ADOLC_MEDIPACK_SUPPORT
  tape.mediInitTape(tapeId);
#endif

  tape.keepTaylors(keepTaylors);
  tape.tapestats(TapeInfos::NO_MIN_MAX, tape.nominmaxFlag());

  if (keepTaylors)
    tape.deg_save(1);
  tape.start_trace();
  tape.take_stock(); /* record all existing adoubles on the tape */

  return retval;
}

int trace_on(short tapeId, int keepTaylors, size_t obs, size_t lbs, size_t vbs,
             size_t tbs, int skipFileCleanup) {

  ValueTape &tape = findTape(tapeId);
  // store Id to restore it after trace_off
  tapeIdBuffer().push_back(currentTape().tapeId());
  setCurrentTape(tapeId);

  int retval = tape.initNewTape();
  if (retval) {
#ifdef ADOLC_MEDIPACK_SUPPORT
    tape.mediInitTape(tapeId);
#endif
    // reset the tape buffers
    tape.freeTapeResources();
    tape.tapestats(TapeInfos::OP_BUFFER_SIZE, obs);
    tape.tapestats(TapeInfos::LOC_BUFFER_SIZE, lbs);
    tape.tapestats(TapeInfos::TAY_BUFFER_SIZE, tbs);
    tape.keepTaylors(keepTaylors);
    tape.tapestats(TapeInfos::NO_MIN_MAX, tape.nominmaxFlag());
    tape.skipFileCleanup(skipFileCleanup);
    if (keepTaylors != 0)
      tape.deg_save(1);
    tape.start_trace();
    tape.take_stock(); /* record all existing adoubles on the tape */
    return retval;
  } else
    return -1;
}

void trace_off(int flag) {

  ValueTape &tape = currentTape();
  if (tape.workMode() != TapeInfos::TAPING)
    ADOLCError::fail(ADOLCError::ErrorType::TAPING_NOT_ACTUALLY_TAPING,
                     CURRENT_LOCATION,
                     ADOLCError::FailInfo{.info1 = tape.tapeId()});
  tape.keepTape(flag);
  tape.keep_stock(); /* copy remaining live variables + trace_flag = 0 */
  tape.stop_trace(flag);
  tape.tapingComplete(1);
  tape.workMode(TapeInfos::ADOLC_NO_MODE);
  tape.releaseTape();

  // restore previous tapeId and delete it
  setCurrentTape(tapeIdBuffer().back());
  tapeIdBuffer().pop_back();
}

void cachedTraceTags(std::vector<short> &result) {
  if (!tapeBuffer().empty()) {
    result.resize(tapeBuffer().size());
    for (size_t i = 0; i < tapeBuffer().size(); ++i) {
      result[i] = tapeBuffer()[i]->tapeId();
    }
  }
}

void printTapeStats(int tapeId) {
  using std::cout;
  using std::endl;

  auto stats = tapestats(tapeId);

  std::ostringstream oss;
  oss << "\n*** TAPE STATS (tape" << tapeId << ") **********\n"
      << "Number of independents: " << stats[TapeInfos::NUM_INDEPENDENTS]
      << "\n"
      << "Number of dependents: " << stats[TapeInfos::NUM_DEPENDENTS] << "\n"
      << "Max # of live adoubles: " << stats[TapeInfos::NUM_MAX_LIVES] << "\n"
      << "Taylor stack size: " << stats[TapeInfos::TAY_STACK_SIZE] << "\n"
      << "Number of operations: " << stats[TapeInfos::NUM_OPERATIONS] << "\n"
      << "Number of locations: " << stats[TapeInfos::NUM_LOCATIONS] << "\n"
      << "Number of values: " << stats[TapeInfos::NUM_VALUES] << "\n"
      << "Number of parameters: " << stats[TapeInfos::NUM_PARAM] << "\n"
      << "Operation file written: " << stats[TapeInfos::OP_FILE_ACCESS] << "\n"
      << "Location file written: " << stats[TapeInfos::LOC_FILE_ACCESS] << "\n"
      << "Value file written: " << stats[TapeInfos::VAL_FILE_ACCESS] << "\n"
      << "Operation buffer size: " << stats[TapeInfos::OP_BUFFER_SIZE] << "\n"
      << "Location buffer size: " << stats[TapeInfos::LOC_BUFFER_SIZE] << "\n"
      << "Value buffer size: " << stats[TapeInfos::VAL_BUFFER_SIZE] << "\n"
      << "Taylor buffer size: " << stats[TapeInfos::TAY_BUFFER_SIZE] << "\n"
      << "Operation type size: " << sizeof(unsigned char) << "\n"
      << "Location type size: " << sizeof(size_t) << "\n"
      << "Value type size: " << sizeof(double) << "\n"
      << "Taylor type size: " << sizeof(double) << "\n"
      << "**********************************\n"
      << endl;

  cout << oss.str();
}

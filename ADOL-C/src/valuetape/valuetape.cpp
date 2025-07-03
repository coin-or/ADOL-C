#include <adolc/adolcerror.h>
#include <adolc/dvlparms.h>
#include <adolc/tape_interface.h>
#include <adolc/valuetape/valuetape.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <ranges>
#include <span>
#include <string>
#include <sys/stat.h> // used in readconfigFile

ValueTape::~ValueTape() {
  // ensure that we dont delete the valuetape before all adouble or pdouble are
  // deleted!
  assert(numLives() == 0 &&
         "Can not destroy ValueTape there are still active variables!");
  cp_clearStack();
}

void ValueTape::initTapeInfos_keep() {
  // we want to keep the buffers
  unsigned char *opBuffer = tapeInfos_.opBuffer;
  size_t *locBuffer = tapeInfos_.locBuffer;
  double *valBuffer = tapeInfos_.valBuffer;
  double *tayBuffer = tapeInfos_.tayBuffer;
  double *signature = tapeInfos_.signature;
  FILE *tay_file = tapeInfos_.tay_file;
  short tapeId = tapeInfos_.tapeId_;

  // keep the stats to later know the number of indeps, etc...
  auto tmp_stats = std::move(tapeInfos_.stats);

  // make sure the destructor will not destroy them
  tapeInfos_.opBuffer = nullptr;
  tapeInfos_.locBuffer = nullptr;
  tapeInfos_.valBuffer = nullptr;
  tapeInfos_.tayBuffer = nullptr;
  tapeInfos_.signature = nullptr;
  tapeInfos_.tay_file = nullptr;

  tapeInfos_ = TapeInfos();
  tapeInfos_.stats = std::move(tmp_stats);

  tapeInfos_.opBuffer = opBuffer;
  tapeInfos_.locBuffer = locBuffer;
  tapeInfos_.valBuffer = valBuffer;
  tapeInfos_.tayBuffer = tayBuffer;
  tapeInfos_.signature = signature;
  tapeInfos_.tay_file = tay_file;
  tapeInfos_.tapeId_ = tapeId;
}

/* inits a new tape and updates the tape stack (called from start_trace)
 * - returns 0 without error
 * - returns 1 if tapeId was already/still in use */
int ValueTape::initNewTape() {
  int retval = 0;

  if (inUse()) {
    if (!tapingComplete())
      ADOLCError::fail(ADOLCError::ErrorType::TAPING_TAPE_STILL_IN_USE,
                       CURRENT_LOCATION,
                       ADOLCError::FailInfo{.info1 = tapeId()});

    if (!tapestats(TapeInfos::OP_FILE_ACCESS) &&
        !tapestats(TapeInfos::LOC_FILE_ACCESS) &&
        !tapestats(TapeInfos::VAL_FILE_ACCESS)) {
#if defined(ADOLC_DEBUG)
      fprintf(DIAG_OUT,
              "\nADOL-C warning: Tape %d existed in main memory"
              " only and gets overwritten!\n\n",
              tapeId());
#endif
      /* free associated resources */
      retval = 1;
    }
  }
  if (tay_file())
    rewind(tay_file());

  // creates new tapeInfos object with old buffers
  // thus, we dont allocate the buffers again if they are already existent
  initTapeInfos_keep();
#ifdef SPARSE
  initSparse();
#endif

  traceFlag(1);
  inUse(1);

  // those are the old values from the globaltapevars
  // require to init the tape-buffers with correct size
  // or set the last location pointers.
  tapestats(TapeInfos::OP_BUFFER_SIZE, operationBufferSize());
  tapestats(TapeInfos::LOC_BUFFER_SIZE, locationBufferSize());
  tapestats(TapeInfos::VAL_BUFFER_SIZE, valueBufferSize());
  tapestats(TapeInfos::TAY_BUFFER_SIZE, taylorBufferSize());
  skipFileCleanup(0);
  return retval;
}

/* opens an existing tape or creates a new handle for a tape on hard disk
 * - called from init_for_sweep and init_rev_sweep */
void ValueTape::openTape(char mode) {
  /* tape has been used before (in the current program) */
  if (!inUse()) {
    /* forward sweep */
    if (tay_file())
      rewind(tay_file());
    initTapeInfos_keep();
    traceFlag(1);
    tapingComplete(1);
    inUse(1);
    read_tape_stats();
  }
}

/* release the current tape and give control to the previous one */
void ValueTape::releaseTape() {
  /* if operations, locations and constants tapes have been written and value
   * stack information have not been created tapeInfos are no longer needed*/
  if (!keepTaylors() && tapestats(TapeInfos::OP_FILE_ACCESS) == 1 &&
      tapestats(TapeInfos::LOC_FILE_ACCESS) == 1 &&
      tapestats(TapeInfos::VAL_FILE_ACCESS) == 1) {
    inUse(0);
  }
}

/* record all existing adoubles on the tape
 * - intended to be used in start_trace only */
void ValueTape::take_stock() {
  size_t space_left, loc = 0;
  double *vals;
  size_t vals_left;

  space_left = get_val_space(); /* remaining space in const. tape buffer */
  vals_left = globalTapeVars_.storeSize;
  vals = globalTapeVars_.store;

  /* if we have adoubles in use */
  if (globalTapeVars_.numLives > 0) {
    /* fill the current values (real) tape buffer and write it to disk
     * - do this as long as buffer can be fully filled */
    while (space_left < vals_left) {
      put_op(take_stock_op);
      put_loc(space_left);
      put_loc(loc);
      put_vals_writeBlock(vals, space_left);
      vals += space_left;
      vals_left -= space_left;
      loc += space_left;
      space_left = get_val_space();
    }
    /* store the remaining adouble values to the values tape buffer
     * -> no write to disk necessary */
    if (vals_left > 0) {
      put_op(take_stock_op);
      put_loc(vals_left);
      put_loc(loc);
      put_vals_notWriteBlock(vals, vals_left);
    }
  }
  tapeInfos_.traceFlag = 1;
}

/****************************************************************************/
/* record all remaining live variables on the value stack tape              */
/* - turns off trace_flag                                                   */
/* - intended to be used in stop_trace only                                 */
/****************************************************************************/
size_t ValueTape::keep_stock() {
  size_t loc2;

  /* save all the final adoubles when finishing tracing */
  loc2 = globalTapeVars_.storeSize - 1;

  /* special signal -> all alive adoubles recorded on the end of the
   * value stack -> special handling at the beginning of reverse */
  put_op(death_not);
  put_loc(0);    /* lowest loc */
  put_loc(loc2); /* highest loc */

  tapeInfos_.numTays_Tape += globalTapeVars_.storeSize;
  /* now really do it if keepTaylors is set */
  if (tapeInfos_.keepTaylors) {
    do {
      write_scaylor(globalTapeVars_.store[loc2]);
    } while (loc2-- > 0);
  }
  tapeInfos_.traceFlag = 0;
  return globalTapeVars_.storeSize;
}

/****************************************************************************/
/* Set up statics for writing taylor data                                   */
/****************************************************************************/
void ValueTape::taylor_begin(uint bufferSize, int degreeSave) {
  if (tayBuffer()) {
#if defined(ADOLC_DEBUG)
    fprintf(DIAG_OUT,
            "\nADOL-C warning: !!! Taylor information for tape %d"
            " found that will be overwritten !!!\n\n",
            tapeId());
#endif
    taylor_close(false);
  } else { /* check if new buffer is allowed */
    if (numTBuffersInUse() == maxNumberTaylorBuffers())
      ADOLCError::fail(ADOLCError::ErrorType::TAPING_TO_MANY_TAYLOR_BUFFERS,
                       CURRENT_LOCATION);

    increment_numTBuffersInUse();
    if (tay_fileName() == nullptr)
      tay_fileName();
  }

  /* initial setups */
  if (tayBuffer() == nullptr)
    tayBuffer(new double[bufferSize]);

  if (tayBuffer() == nullptr)
    ADOLCError::fail(ADOLCError::ErrorType::TAPING_TBUFFER_ALLOCATION_FAILED,
                     CURRENT_LOCATION);

  deg_save(degreeSave);
  if (degreeSave >= 0)
    keepTaylors(1);
  currTay(tayBuffer());
  lastTayP1(currTay() + bufferSize);
  inUse(1);

  numTays_Tape(0);
}

/****************************************************************************/
/* Close the taylor file, reset data.                                       */
/****************************************************************************/
void ValueTape::taylor_close(bool resetData) {
  if (resetData == false) {
    /* enforces failure of reverse => retaping */
    deg_save(-1);
    if (tay_file()) {
      fclose(tay_file());
      remove(tay_fileName());
      tay_file(nullptr);
    }
    return;
  }

  if (tay_file()) {
    if (keepTaylors())
      put_tay_block(currTay());
  } else {
    numTays_Tape(currTay() - tayBuffer());
  }
  lastTayBlockInCore(1);
  tapestats(TapeInfos::TAY_STACK_SIZE, numTays_Tape());

  /* keep track of the Ind/Dep counts of the taylor stack */
  tay_numInds(tapestats(TapeInfos::NUM_INDEPENDENTS));
  tay_numDeps(tapestats(TapeInfos::NUM_DEPENDENTS));

#if defined(ADOLC_DEBUG)
  if (tapeInfos_.tay_file != nullptr)
    fprintf(DIAG_OUT,
            "\n ADOL-C debug: Taylor file of length %d bytes "
            "completed\n",
            (int)(tapeInfos_.numTays_Tape * sizeof(double)));
  else
    fprintf(DIAG_OUT,
            "\n ADOL-C debug: Taylor array of length %d bytes "
            "completed\n",
            (int)(tapeInfos_.numTays_Tape * sizeof(double)));
#endif
}

/****************************************************************************/
/* Initializes a reverse sweep.                                             */
/****************************************************************************/
void ValueTape::taylor_back(short tag, int *dep, int *ind, int *degree) {
  /* this should be removed soon since values can be accessed via         */
  /* tapeInfos_ directly                                    */
  *dep = tay_numDeps();
  *ind = tay_numInds();
  *degree = deg_save();

  if (tayBuffer() == nullptr)
    ADOLCError::fail(ADOLCError::ErrorType::REVERSE_NO_TAYLOR_STACK,
                     CURRENT_LOCATION, ADOLCError::FailInfo{.info1 = tapeId()});

  nextBufferNumber(numTays_Tape() / tapestats(TapeInfos::TAY_BUFFER_SIZE));
  const size_t number = numTays_Tape() % tapestats(TapeInfos::TAY_BUFFER_SIZE);
  currTay(tayBuffer() + number);

  if (lastTayBlockInCore() != 1) {
    if (!tay_file())
      ADOLCError::fail(ADOLCError::ErrorType::TAY_NULLPTR, CURRENT_LOCATION);

    if (fseek(tay_file(),
              sizeof(double) * nextBufferNumber() *
                  tapestats(TapeInfos::TAY_BUFFER_SIZE),
              SEEK_SET) == -1)
      ADOLCError::fail(ADOLCError::ErrorType::EVAL_SEEK_VALUE_STACK,
                       CURRENT_LOCATION);

    const size_t chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(double);
    const size_t chunks = number / chunkSize;

    for (size_t i = 0; i < chunks; ++i)
      if (fread(tayBuffer() + i * chunkSize, chunkSize * sizeof(double), 1,
                tay_file()) != 1)
        ADOLCError::fail(ADOLCError::ErrorType::TAPING_FATAL_IO_ERROR,
                         CURRENT_LOCATION);

    const size_t remain = number % chunkSize;

    if (remain != 0)
      if (fread(tayBuffer() + chunks * chunkSize, remain * sizeof(double), 1,
                tay_file()) != 1)
        ADOLCError::fail(ADOLCError::ErrorType::TAPING_FATAL_IO_ERROR,
                         CURRENT_LOCATION);
  }
  decrement_nextBufferNumber();
}

/****************************************************************************/
/* Writes the block of size depth of taylor coefficients from point loc to  */
/* the taylor buffer.  If the buffer is filled, then it is written to the   */
/* taylor tape.                                                             */
/*--------------------------------------------------------------------------*/
void ValueTape::write_taylors(size_t loc, int keep, int degree, int numDir) {
  double *T = dpp_T(loc);

  for (size_t j = 0; j < numDir; ++j) {
    for (size_t i = 0; i < keep; ++i) {
      if (currTay() == lastTayP1())
        put_tay_block(lastTayP1());

      currTay(*T);
      increment_currTay();
      ++T;
    }
    /*        for (i = keep; i < degree; ++i) ++T;*/
    if (degree > keep)
      T += degree - keep;
  }
}

/****************************************************************************/
/* Write_scaylors writes # size elements from x to the taylor buffer.       */
/****************************************************************************/
void ValueTape::write_scaylors(double *x, uint size) {
  size_t j = 0;
  std::span<double> taySpan(currTay(), lastTayP1());
  /* write data to buffer and put buffer to disk as long as data remain in
   * the x-buffer => don't create an empty value stack buffer! */
  while (currTay() + size > lastTayP1()) {
    for (double &tay : taySpan) {
      tay = x[j++];
    }
    size -= lastTayP1() - currTay();
    put_tay_block(lastTayP1());
  }

  std::span<double> tayBufferSpan(currTay(), tayBuffer() + size);
  for (double &tay : tayBufferSpan) {
    tay = x[j++];
  }
  currTay(currTay() + size);
}

/****************************************************************************/
/* Puts a block of taylor coefficients from the value stack buffer to the   */
/* taylor buffer. --- Higher Order Scalar                                   */
/****************************************************************************/
void ValueTape::get_taylors(size_t loc, int degree) {
  double *T = rpp_T(loc) + degree;
  std::span<double> taySpan(currTay(), tayBuffer());
  /* As long as all values from the taylor stack buffer will be used copy
   * them into the taylor buffer and load the next (previous) buffer. */
  while (currTay() - degree < tayBuffer()) {
    for (auto tay = taySpan.rbegin(); tay != taySpan.rend(); tay++) {
      *(T--) = *tay;
    }
    degree -= currTay() - tayBuffer();
    get_tay_block_r();
  }

  /* Copy the remaining values from the stack into the buffer ... */
  for (size_t j = 0; j < degree; ++j) {
    decrement_currTay();
    *(--T) = *currTay();
  }
}

/****************************************************************************/
/* Puts a block of taylor coefficients from the value stack buffer to the   */
/* taylor buffer. --- Higher Order Vector                                   */
/****************************************************************************/
void ValueTape::get_taylors_p(size_t loc, int degree, int numDir) {
  double *T = rpp_T(loc) + degree * numDir;

  /* update the directions except the base point parts */
  for (size_t j = 0; j < numDir; ++j) {
    for (size_t i = 1; i < degree; ++i) {
      if (currTay() == tayBuffer())
        get_tay_block_r();

      decrement_currTay();
      --T;
      *T = *currTay();
    }
    --T; /* skip the base point part */
  }
  /* now update the base point parts */
  if (currTay() == tayBuffer())
    get_tay_block_r();

  decrement_currTay();
  for (size_t i = 0; i < numDir; ++i) {
    *T = *currTay();
    T += degree;
  }
}

/****************************************************************************/
/****************************************************************************/
/* NON-VALUE-STACK FUNCTIONS                                                */
/****************************************************************************/
/****************************************************************************/

void ValueTape::initTapeBuffers() {
  if (!opBuffer())
    opBuffer(new unsigned char[tapestats(TapeInfos::OP_BUFFER_SIZE)]);

  if (!locBuffer())
    locBuffer(new size_t[tapestats(TapeInfos::LOC_BUFFER_SIZE)]);

  if (!valBuffer())
    valBuffer(new double[tapestats(TapeInfos::VAL_BUFFER_SIZE)]);

  if (!opBuffer() || !locBuffer() || !valBuffer())
    ADOLCError::fail(ADOLCError::ErrorType::TAPING_BUFFER_ALLOCATION_FAILED,
                     CURRENT_LOCATION);

  lastOpP1(opBuffer() + tapestats(TapeInfos::OP_BUFFER_SIZE));
  lastLocP1(locBuffer() + tapestats(TapeInfos::LOC_BUFFER_SIZE));
  lastValP1(valBuffer() + tapestats(TapeInfos::VAL_BUFFER_SIZE));
}

/****************************************************************************/
/* start_trace: (part of trace_on)                                          */
/* Initialization for the taping process. Does buffer allocation, sets      */
/* files names, and calls appropriate setup routines.                       */
/****************************************************************************/
void ValueTape::start_trace() {
  initTapeBuffers();
  // reset the position pointer to first entry
  currOp(opBuffer());
  currLoc(locBuffer());
  currVal(valBuffer());

  num_eq_prod(0);
  numSwitches(0);
  workMode(TapeInfos::TAPING);

  /* Put operation denoting the start_of_the tape */
  put_op(start_of_tape);

  /* Leave space for the stats */
  const int space = TapeInfos::STAT_SIZE * sizeof(size_t) + sizeof(ADOLC_ID);
  if (space > statSpace * sizeof(size_t))
    ADOLCError::fail(ADOLCError::ErrorType::MORE_STAT_SPACE_REQUIRED,
                     CURRENT_LOCATION);

  for (size_t i = 0; i < statSpace; ++i)
    put_loc(0);

  /* initialize value stack if necessary */
  if (keepTaylors())
    taylor_begin(tapestats(TapeInfos::TAY_BUFFER_SIZE), 0);

  /* mark possible (hard disk) tape creation */
  markNewTape();
}

void ValueTape::save_params() {
  tapestats(TapeInfos::NUM_PARAM, numparam());
  if (paramstore())
    delete[] paramstore();

  paramstore(new double[tapestats(TapeInfos::NUM_PARAM)]);

  // Sometimes we have pStore == nullptr and stats[TapeInfos::NUM_PARAM] == 0.
  // Calling memcpy with that is undefined behavior, and sanitizers will issue a
  // warning.
  if (tapestats(TapeInfos::NUM_PARAM) > 0)
    memcpy(paramstore(), pStore(),
           tapestats(TapeInfos::NUM_PARAM) * sizeof(double));

  free_all_taping_params();
  if (currVal() + tapestats(TapeInfos::NUM_PARAM) < lastValP1())
    put_vals_notWriteBlock(paramstore(), tapestats(TapeInfos::NUM_PARAM));

  else {
    size_t np = tapestats(TapeInfos::NUM_PARAM);
    size_t ip = 0;
    size_t remain = tapestats(TapeInfos::NUM_PARAM);
    while (tapestats(TapeInfos::NUM_PARAM) > ip) {
      remain = tapestats(TapeInfos::NUM_PARAM) - ip;
      const size_t avail = lastValP1() - currVal();
      const size_t chunk = (avail < remain) ? avail : remain;
      put_vals_notWriteBlock(paramstore() + ip, chunk);
      ip += chunk;
      if (ip < np)
        put_val_block(lastValP1());
    }
  }
}

/****************************************************************************/
/* Stop Tracing.  Clean up, and turn off trace_flag.                        */
/****************************************************************************/
void ValueTape::stop_trace(int flag) {
  put_op(end_of_tape); /* Mark end of tape. */
  save_params();

  tapestats(TapeInfos::NUM_INDEPENDENTS, numInds());
  tapestats(TapeInfos::NUM_DEPENDENTS, numDeps());
  tapestats(TapeInfos::NUM_MAX_LIVES, storeSize());
  tapestats(TapeInfos::NUM_EQ_PROD, num_eq_prod());
  tapestats(TapeInfos::NUM_SWITCHES, numSwitches());

  if (keepTaylors())
    taylor_close(true);

  tapestats(TapeInfos::TAY_STACK_SIZE, numTays_Tape());

  /* The taylor stack size base estimation results in a doubled taylor count
   * if we tape with keep (taylors counted in adouble.cpp/avector.cpp and
   * "keep_stock" even if not written and a second time when actually
   * written by "put_tay_block"). Correction follows here. */
  if (keepTaylors() != 0 && tay_file()) {
    tapestats(TapeInfos::TAY_STACK_SIZE,
              tapestats(TapeInfos::TAY_STACK_SIZE) / 2);
    numTays_Tape(numTays_Tape() / 2);
  }

  close_tape(flag); /* closes the tape, files up stats, and writes the
                       tape stats to the integer tape */
}

/****************************************************************************/
/* Close open tapes, update stats and clean up.                             */
/****************************************************************************/
void ValueTape::close_tape(int flag) {
  /* finish operations tape, close it, update stats */
  if (flag != 0 || op_file()) {
    if (currOp() != opBuffer()) {
      put_op_block(currOp());
    }
    if (op_file()) {
      fclose(op_file());
      op_file(nullptr);
    }
    tapestats(TapeInfos::OP_FILE_ACCESS, 1);
    delete[] opBuffer();
    opBuffer(nullptr);
  } else {
    numOps_Tape(currOp() - opBuffer());
  }
  tapestats(TapeInfos::NUM_OPERATIONS, numOps_Tape());

  /* finish constants tape, close it, update stats */
  if (flag != 0 || val_file()) {
    if (currVal() != valBuffer()) {
      put_val_block(currVal());
    }
    if (val_file()) {
      fclose(val_file());
      val_file(nullptr);
    }
    tapestats(TapeInfos::VAL_FILE_ACCESS, 1);
    delete[] valBuffer();
    valBuffer(nullptr);
  } else {
    numVals_Tape(currVal() - valBuffer());
  }
  tapestats(TapeInfos::NUM_VALUES, numVals_Tape());

  /* finish locations tape, update and write tape stats, close tape */
  if (flag != 0 || loc_file()) {
    if (currLoc() != locBuffer()) {
      put_loc_block(currLoc());
    }
    tapestats(TapeInfos::NUM_LOCATIONS, numLocs_Tape());
    tapestats(TapeInfos::LOC_FILE_ACCESS, 1);
    /* write tape stats */
    fseek(loc_file(), 0, 0);
    fwrite(&get_adolc_id(), sizeof(ADOLC_ID), 1, loc_file());
    fwrite(tapestats().data(), TapeInfos::STAT_SIZE * sizeof(size_t), 1,
           loc_file());
    fclose(loc_file());
    loc_file(nullptr);
    delete[] locBuffer();
    locBuffer(nullptr);
  } else {
    numLocs_Tape(currLoc() - locBuffer());
    tapestats(TapeInfos::NUM_LOCATIONS, numLocs_Tape());
  }
}

/****************************************************************************/
/* Reads parameters from the end of value tape for disk based tapes         */
/****************************************************************************/
void ValueTape::read_params() {
  constexpr size_t chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(double);

  if (!paramstore())
    paramstore(new double[tapestats(TapeInfos::NUM_PARAM)]);

  double *valBuffer = new double[tapestats(TapeInfos::VAL_BUFFER_SIZE)];
  double *lastValP1 = valBuffer + tapestats(TapeInfos::VAL_BUFFER_SIZE);

  FILE *val_file = nullptr;
  if ((val_file = fopen(val_fileName(), "rb")) == nullptr)
    ADOLCError::fail(ADOLCError::ErrorType::VALUE_TAPE_FREAD_FAILED,
                     CURRENT_LOCATION, ADOLCError::FailInfo{.info1 = tapeId()});

  size_t number = (tapestats(TapeInfos::NUM_VALUES) /
                   tapestats(TapeInfos::VAL_BUFFER_SIZE)) *
                  tapestats(TapeInfos::VAL_BUFFER_SIZE);

  fseek(val_file, number * sizeof(double), SEEK_SET);
  number =
      tapestats(TapeInfos::NUM_VALUES) % tapestats(TapeInfos::VAL_BUFFER_SIZE);

  if (number != 0) {
    const size_t chunks = number / chunkSize;

    for (size_t i = 0; i < chunks; ++i)
      if (fread(valBuffer + i * chunkSize, chunkSize * sizeof(double), 1,
                val_file) != 1)
        ADOLCError::fail(ADOLCError::ErrorType::VALUE_TAPE_FREAD_FAILED,
                         CURRENT_LOCATION,
                         ADOLCError::FailInfo{.info1 = tapeId()});
    const size_t remain = number % chunkSize;
    if (remain != 0)
      if (fread(valBuffer + chunks * chunkSize, remain * sizeof(double), 1,
                val_file) != 1)
        ADOLCError::fail(ADOLCError::ErrorType::VALUE_TAPE_FREAD_FAILED,
                         CURRENT_LOCATION,
                         ADOLCError::FailInfo{.info1 = tapeId()});
  }
  size_t nVT = tapestats(TapeInfos::NUM_VALUES) - number;
  const double *currVal = valBuffer + number;
  const size_t np = tapestats(TapeInfos::NUM_PARAM);
  size_t ip = np;
  while (ip > 0) {
    size_t avail = currVal - valBuffer;
    size_t rsize = (avail < ip) ? avail : ip;
    double *paramstore_view = paramstore();
    for (size_t i = 0; i < rsize; ++i)
      paramstore_view[--ip] = *--currVal;

    if (ip > 0) {
      const size_t number = tapestats(TapeInfos::VAL_BUFFER_SIZE);
      fseek(val_file, sizeof(double) * (nVT - number), SEEK_SET);
      const size_t chunks = number / chunkSize;
      for (size_t i = 0; i < chunks; ++i)
        if (fread(valBuffer + i * chunkSize, chunkSize * sizeof(double), 1,
                  val_file) != 1)
          ADOLCError::fail(ADOLCError::ErrorType::VALUE_TAPE_FREAD_FAILED,
                           CURRENT_LOCATION,
                           ADOLCError::FailInfo{.info1 = tapeId()});

      const size_t remain = number % chunkSize;
      if (remain != 0)
        if (fread(valBuffer + chunks * chunkSize, remain * sizeof(double), 1,
                  val_file) != 1)
          ADOLCError::fail(ADOLCError::ErrorType::VALUE_TAPE_FREAD_FAILED,
                           CURRENT_LOCATION,
                           ADOLCError::FailInfo{.info1 = tapeId()});

      nVT -= number;
      currVal = lastValP1;
    }
  }
  fclose(val_file);
  delete[] valBuffer;
}

/****************************************************************************/
/* Overrides the parameters for the next evaluations. This will invalidate  */
/* the taylor stack, so next reverse call will fail, if not preceded by a   */
/* forward call after setting the parameters.                               */
/****************************************************************************/
void ValueTape::set_param_vec(short tag, size_t numparam, double *paramvec) {
  /* mark possible (hard disk) tape creation */
  markNewTape();

  /* make room for tapeInfos and read tapestats if necessary, keep value
   * stack information */
  openTape(TapeInfos::FORWARD);
  if (tapestats(TapeInfos::NUM_PARAM) != numparam)
    ADOLCError::fail(
        ADOLCError::ErrorType::PARAM_COUNTS_MISMATCH, CURRENT_LOCATION,
        ADOLCError::FailInfo{.info1 = tag,
                             .info5 = numparam,
                             .info6 = tapeInfos_.stats[TapeInfos::NUM_PARAM]});

  if (!paramstore())
    paramstore(new double[tapestats(TapeInfos::NUM_PARAM)]);

  double *paramstore_view = paramstore();
  for (size_t i = 0; i < tapestats(TapeInfos::NUM_PARAM); ++i)
    paramstore_view[i] = paramvec[i];

  taylor_close(false);
  releaseTape();
}

/**
 * @brief Compares two given ADOLC_IDs
 * @throws runtime_error if the ids are not equal
 */

void ValueTape::compare_adolc_ids(const ADOLC_ID &id1, const ADOLC_ID &id2) {
  constexpr size_t t1Version = 100 * ADOLC_NEW_TAPE_VERSION +
                               10 * ADOLC_NEW_TAPE_SUBVERSION +
                               1 * ADOLC_NEW_TAPE_PATCHLEVEL;

  const size_t t2Version =
      100 * id1.adolc_ver + 10 * id1.adolc_sub + 1 * id1.adolc_lvl;

  if (t1Version > t2Version)
    ADOLCError::fail(ADOLCError::ErrorType::TAPE_TO_OLD, CURRENT_LOCATION,
                     ADOLCError::FailInfo{.info1 = tapeId()});

  if (id1.address_size != id2.address_size) {
    if (id1.address_size > id2.address_size)
      ADOLCError::fail(ADOLCError::ErrorType::WRONG_PLATFORM_64,
                       CURRENT_LOCATION);
    else
      ADOLCError::fail(ADOLCError::ErrorType::WRONG_PLATFORM_32,
                       CURRENT_LOCATION);
  }

  if (id1.locint_size != id2.locint_size)
    ADOLCError::fail(ADOLCError::ErrorType::SIZE_MISMATCH, CURRENT_LOCATION,
                     ADOLCError::FailInfo{.info1 = tapeId(),
                                          .info5 = id1.locint_size,
                                          .info6 = id2.locint_size});
}

/****************************************************************************/
/* Does the actual reading from the hard disk into the stats buffer */
/****************************************************************************/
void ValueTape::read_tape_stats() {
  if (inUse() && !tapingComplete())
    return;

  FILE *loc_file = nullptr;
  ADOLC_ID tape_ADOLC_ID;
  if ((loc_file = fopen(loc_fileName(), "rb")) == nullptr ||
      (fread(&tape_ADOLC_ID, sizeof(ADOLC_ID), 1, loc_file) != 1) ||
      (fread(tapestats().data(), TapeInfos::STAT_SIZE * sizeof(size_t), 1,
             loc_file) != 1)) {
    ADOLCError::fail(ADOLCError::ErrorType::INTEGER_TAPE_FOPEN_FAILED,
                     CURRENT_LOCATION, ADOLCError::FailInfo{.info1 = tapeId()});
  }

  compare_adolc_ids(get_adolc_id(), tape_ADOLC_ID);
  fclose(loc_file);
  tapingComplete(1);
  if (tapestats(TapeInfos::NUM_PARAM) > 0)
    read_params();
}

/****************************************************************************/
/* Initialize a forward sweep. Get stats, open tapes, fill buffers, ... */
/****************************************************************************/
void ValueTape::init_for_sweep(short tag) {
  constexpr size_t chunkSize_uchar =
      ADOLC_IO_CHUNK_SIZE / sizeof(unsigned char);
  constexpr size_t chunkSize_size_t = ADOLC_IO_CHUNK_SIZE / sizeof(size_t);
  constexpr size_t chunkSize_double = ADOLC_IO_CHUNK_SIZE / sizeof(double);

  /* mark possible (hard disk) tape creation */
  markNewTape();

  /* make room for tapeInfos and read tape stats if necessary, keep value
   * stack information */
  openTape(TapeInfos::FORWARD);
  initTapeBuffers();

  /* init operations */
  size_t number = 0;
  if (tapestats(TapeInfos::OP_FILE_ACCESS) == 1) {
    op_file(fopen(op_fileName(), "rb"));
    /* how much to read ? */
    number = MIN_ADOLC(tapestats(TapeInfos::OP_BUFFER_SIZE),
                       tapestats(TapeInfos::NUM_OPERATIONS));

    if (number != 0) {
      const size_t chunks = number / chunkSize_uchar;
      for (size_t i = 0; i < chunks; ++i)
        if (fread(opBuffer() + i * chunkSize_uchar,
                  chunkSize_uchar * sizeof(unsigned char), 1, op_file()) != 1)
          ADOLCError::fail(ADOLCError::ErrorType::EVAL_OP_TAPE_READ_FAILED,
                           CURRENT_LOCATION);

      const size_t remain = number % chunkSize_uchar;
      if (remain != 0)
        if (fread(opBuffer() + chunks * chunkSize_uchar,
                  remain * sizeof(unsigned char), 1, op_file()) != 1)
          ADOLCError::fail(ADOLCError::ErrorType::EVAL_OP_TAPE_READ_FAILED,
                           CURRENT_LOCATION);
    }
    /* how much remains ? */
    number = tapestats(TapeInfos::NUM_OPERATIONS) - number;
  }
  numOps_Tape(number);
  currOp(opBuffer());

  /* init locations */
  number = 0;
  if (tapestats(TapeInfos::LOC_FILE_ACCESS) == 1) {
    loc_file(fopen(loc_fileName(), "rb"));
    /* how much to read ? */
    number = MIN_ADOLC(tapestats(TapeInfos::LOC_BUFFER_SIZE),
                       tapestats(TapeInfos::NUM_LOCATIONS));
    if (number != 0) {

      const size_t chunks = number / chunkSize_size_t;
      for (size_t i = 0; i < chunks; ++i)
        if (fread(locBuffer() + i * chunkSize_size_t,
                  chunkSize_size_t * sizeof(size_t), 1, loc_file()) != 1)
          ADOLCError::fail(ADOLCError::ErrorType::EVAL_LOC_TAPE_READ_FAILED,
                           CURRENT_LOCATION);

      const size_t remain = number % chunkSize_size_t;
      if (remain != 0)
        if (fread(locBuffer() + chunks * chunkSize_size_t,
                  remain * sizeof(size_t), 1, loc_file()) != 1)
          ADOLCError::fail(ADOLCError::ErrorType::EVAL_LOC_TAPE_READ_FAILED,
                           CURRENT_LOCATION);
    }
    /* how much remains ? */
    number = tapestats(TapeInfos::NUM_LOCATIONS) - number;
  }
  numLocs_Tape(number);

  /* skip stats */
  size_t numLocsForStats = statSpace;
  while (numLocsForStats >= tapestats(TapeInfos::LOC_BUFFER_SIZE)) {
    get_loc_block_f();
    numLocsForStats -= tapestats(TapeInfos::LOC_BUFFER_SIZE);
  }
  currLoc(locBuffer() + numLocsForStats);

  /* init constants */
  number = 0;
  if (tapestats(TapeInfos::VAL_FILE_ACCESS) == 1) {
    val_file(fopen(val_fileName(), "rb"));
    /* how much to read ? */
    number = MIN_ADOLC(tapestats(TapeInfos::VAL_BUFFER_SIZE),
                       tapestats(TapeInfos::NUM_VALUES));
    if (number != 0) {
      const size_t chunks = number / chunkSize_double;
      for (size_t i = 0; i < chunks; ++i)
        if (fread(valBuffer() + i * chunkSize_double,
                  chunkSize_double * sizeof(double), 1, val_file()) != 1)
          ADOLCError::fail(ADOLCError::ErrorType::EVAL_VAL_TAPE_READ_FAILED,
                           CURRENT_LOCATION);

      const size_t remain = number % chunkSize_double;
      if (remain != 0)
        if (fread(valBuffer() + chunks * chunkSize_double,
                  remain * sizeof(double), 1, val_file()) != 1)
          ADOLCError::fail(ADOLCError::ErrorType::EVAL_VAL_TAPE_READ_FAILED,
                           CURRENT_LOCATION);
    }
    /* how much remains ? */
    number = tapestats(TapeInfos::NUM_VALUES) - number;
  }
  numVals_Tape(number);
  currVal(valBuffer());
#ifdef ADOLC_AMPI_SUPPORT
  TAPE_AMPI_resetBottom();
#endif
}

/****************************************************************************/
/* Initialize a reverse sweep. Get stats, open tapes, fill buffers, ... */
/****************************************************************************/
void ValueTape::init_rev_sweep(short tag) {
  constexpr size_t chunkSize_uchar =
      ADOLC_IO_CHUNK_SIZE / sizeof(unsigned char);
  constexpr size_t chunkSize_size_t = ADOLC_IO_CHUNK_SIZE / sizeof(size_t);
  constexpr size_t chunkSize_double = ADOLC_IO_CHUNK_SIZE / sizeof(double);

  /* mark possible (hard disk) tape creation */
  markNewTape();

  /* make room for tapeInfos and read tape stats if necessary, keep value
   * stack information */
  openTape(TapeInfos::REVERSE);
  initTapeBuffers();

  /* init operations */
  size_t number = tapestats(TapeInfos::NUM_OPERATIONS);
  if (tapestats(TapeInfos::OP_FILE_ACCESS) == 1) {
    op_file(fopen(perTapeInfos_.op_fileName, "rb"));
    number = (tapestats(TapeInfos::NUM_OPERATIONS) /
              tapestats(TapeInfos::OP_BUFFER_SIZE)) *
             tapestats(TapeInfos::OP_BUFFER_SIZE);
    fseek(op_file(), number * sizeof(unsigned char), SEEK_SET);
    number = tapestats(TapeInfos::NUM_OPERATIONS) %
             tapestats(TapeInfos::OP_BUFFER_SIZE);
    if (number != 0) {
      const size_t chunks = number / chunkSize_uchar;
      for (size_t i = 0; i < chunks; ++i)
        if (fread(opBuffer() + i * chunkSize_uchar,
                  chunkSize_uchar * sizeof(unsigned char), 1, op_file()) != 1)
          ADOLCError::fail(ADOLCError::ErrorType::EVAL_OP_TAPE_READ_FAILED,
                           CURRENT_LOCATION);

      const size_t remain = number % chunkSize_uchar;
      if (remain != 0)
        if (fread(opBuffer() + chunks * chunkSize_uchar,
                  remain * sizeof(unsigned char), 1, op_file()) != 1)
          ADOLCError::fail(ADOLCError::ErrorType::EVAL_OP_TAPE_READ_FAILED,
                           CURRENT_LOCATION);
    }
  }
  numOps_Tape(tapestats(TapeInfos::NUM_OPERATIONS) - number);
  currOp(opBuffer() + number);

  /* init locations */
  number = tapestats(TapeInfos::NUM_LOCATIONS);
  if (tapestats(TapeInfos::LOC_FILE_ACCESS) == 1) {
    loc_file(fopen(loc_fileName(), "rb"));
    number = (tapestats(TapeInfos::NUM_LOCATIONS) /
              tapestats(TapeInfos::LOC_BUFFER_SIZE)) *
             tapestats(TapeInfos::LOC_BUFFER_SIZE);
    fseek(loc_file(), number * sizeof(size_t), SEEK_SET);
    number = tapestats(TapeInfos::NUM_LOCATIONS) %
             tapestats(TapeInfos::LOC_BUFFER_SIZE);
    if (number != 0) {
      const size_t chunks = number / chunkSize_size_t;
      for (size_t i = 0; i < chunks; ++i)
        if (fread(locBuffer() + i * chunkSize_size_t,
                  chunkSize_size_t * sizeof(size_t), 1, loc_file()) != 1)
          ADOLCError::fail(ADOLCError::ErrorType::EVAL_LOC_TAPE_READ_FAILED,
                           CURRENT_LOCATION);

      const size_t remain = number % chunkSize_size_t;
      if (remain != 0)
        if (fread(locBuffer() + chunks * chunkSize_size_t,
                  remain * sizeof(size_t), 1, loc_file()) != 1)
          ADOLCError::fail(ADOLCError::ErrorType::EVAL_LOC_TAPE_READ_FAILED,
                           CURRENT_LOCATION);
    }
  }
  numLocs_Tape(tapestats(TapeInfos::NUM_LOCATIONS) - number);
  currLoc(locBuffer() + number);

  /* init constants */
  number = tapestats(TapeInfos::NUM_VALUES);
  if (tapestats(TapeInfos::VAL_FILE_ACCESS) == 1) {
    val_file(fopen(val_fileName(), "rb"));
    number = (tapestats(TapeInfos::NUM_VALUES) /
              tapestats(TapeInfos::VAL_BUFFER_SIZE)) *
             tapestats(TapeInfos::VAL_BUFFER_SIZE);
    fseek(val_file(), number * sizeof(double), SEEK_SET);
    number = tapestats(TapeInfos::NUM_VALUES) %
             tapestats(TapeInfos::VAL_BUFFER_SIZE);
    if (number != 0) {
      const size_t chunks = number / chunkSize_double;
      for (size_t i = 0; i < chunks; ++i)
        if (fread(valBuffer() + i * chunkSize_double,
                  chunkSize_double * sizeof(double), 1, val_file()) != 1)
          ADOLCError::fail(ADOLCError::ErrorType::EVAL_VAL_TAPE_READ_FAILED,
                           CURRENT_LOCATION);

      const size_t remain = number % chunkSize_double;
      if (remain != 0)
        if (fread(valBuffer() + chunks * chunkSize_double,
                  remain * sizeof(double), 1, val_file()) != 1)
          ADOLCError::fail(ADOLCError::ErrorType::EVAL_VAL_TAPE_READ_FAILED,
                           CURRENT_LOCATION);
    }
  }
  numVals_Tape(tapestats(TapeInfos::NUM_VALUES) - number);
  currVal(valBuffer() + number);
#ifdef ADOLC_AMPI_SUPPORT
  TAPE_AMPI_resetTop();
#endif
}

/****************************************************************************/
/* Finish a forward or reverse sweep. */
/****************************************************************************/
void ValueTape::end_sweep() {
  if (op_file()) {
    fclose(op_file());
    op_file(nullptr);
  }
  if (loc_file()) {
    fclose(loc_file());
    loc_file(nullptr);
  }
  if (val_file()) {
    fclose(val_file());
    val_file(nullptr);
  }
  if (deg_save() > 0)
    releaseTape(); /* keep value stack */
  else
    releaseTape(); /* no value stack */
}

/****************************************************************************/
/* Discards parameters from the end of value tape during reverse mode */
/****************************************************************************/
void ValueTape::discard_params_r(void) {
  constexpr size_t chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(double);
  int ip = tapestats(TapeInfos::NUM_PARAM);
  size_t rsize = 0;
  size_t remain = 0;
  while (ip > 0) {
    rsize = currVal() - valBuffer();
    rsize = (rsize < ip) ? rsize : ip;
    ip -= rsize;
    currVal(currVal() - rsize);
    if (ip > 0) {
      fseek(val_file(),
            sizeof(double) * (numVals_Tape() - TapeInfos::VAL_BUFFER_SIZE),
            SEEK_SET);

      for (size_t i = 0; i < TapeInfos::VAL_BUFFER_SIZE / chunkSize; ++i)
        if (fread(valBuffer() + i * chunkSize, chunkSize * sizeof(double), 1,
                  val_file()) != 1)
          ADOLCError::fail(ADOLCError::ErrorType::EVAL_VAL_TAPE_READ_FAILED,
                           CURRENT_LOCATION);

      remain = TapeInfos::VAL_BUFFER_SIZE % chunkSize;
      if (remain != 0)
        if (fread(valBuffer() + TapeInfos::VAL_BUFFER_SIZE,
                  remain * sizeof(double), 1, val_file()) != 1)
          ADOLCError::fail(ADOLCError::ErrorType::EVAL_VAL_TAPE_READ_FAILED,
                           CURRENT_LOCATION);

      numVals_Tape(numVals_Tape() - TapeInfos::VAL_BUFFER_SIZE);
      currVal(lastValP1());
    }
  }
}

// the following macros are used in readConfigFile()
#if defined(_WINDOWS) && !__STDC__
#define stat _stat
#define S_IFDIR _S_IFDIR
#define S_IFMT _S_IFMT
#endif // _WINDOWS && !__STDC__

#ifndef S_ISDIR
#define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)
#endif // S_ISDIR

#define ADOLC_LINE_LENGTH 100
std::array<std::string, 4> ValueTape::readConfigFile() {
  FILE *configFile = nullptr;
  char inputLine[ADOLC_LINE_LENGTH + 1];
  char *pos1 = nullptr, *pos2 = nullptr, *pos3 = nullptr, *pos4 = nullptr,
       *start = nullptr, *end = nullptr;
  int base;
  size_t number = 0;
  char *path = nullptr;

  std::array<std::string, 4> tapeBaseNames;
  tapeBaseNames[0] =
      std::string(TAPE_DIR) + PATHSEPARATOR + ADOLC_LOCATIONS_NAME;
  tapeBaseNames[1] = std::string(TAPE_DIR) + PATHSEPARATOR + ADOLC_VALUES_NAME;
  tapeBaseNames[2] =
      std::string(TAPE_DIR) + PATHSEPARATOR + ADOLC_OPERATIONS_NAME;
  tapeBaseNames[3] = std::string(TAPE_DIR) + PATHSEPARATOR + ADOLC_TAYLORS_NAME;

  operationBufferSize(OBUFSIZE);
  locationBufferSize(LBUFSIZE);
  valueBufferSize(VBUFSIZE);
  taylorBufferSize(TBUFSIZE);
  maxNumberTaylorBuffers(TBUFNUM);
  if ((configFile = fopen(".adolcrc", "r")) != nullptr) {
    fprintf(DIAG_OUT, "\nFile .adolcrc found! => Try to parse it!\n");
    fprintf(DIAG_OUT, "****************************************\n");
    while (fgets(inputLine, ADOLC_LINE_LENGTH + 1, configFile) == inputLine) {
      if (std::strlen(inputLine) == ADOLC_LINE_LENGTH &&
          inputLine[ADOLC_LINE_LENGTH - 1] != 0xA) {
        fprintf(DIAG_OUT,
                "ADOL-C warning: Input line in .adolcrc exceeds"
                " %d characters!\n",
                ADOLC_LINE_LENGTH);
        fprintf(DIAG_OUT, "                => Parsing aborted!!\n");
        break;
      }
      pos1 = std::strchr(inputLine, '"');
      pos2 = nullptr;
      pos3 = nullptr;
      pos4 = nullptr;
      if (pos1 != nullptr) {
        pos2 = std::strchr(pos1 + 1, '"');
        if (pos2 != nullptr) {
          pos3 = std::strchr(pos2 + 1, '"');
          if (pos3 != nullptr)
            pos4 = std::strchr(pos3 + 1, '"');
        }
      }
      if (pos4 == nullptr) {
        if (pos1 != nullptr)
          fprintf(DIAG_OUT, "ADOL-C warning: Malformed input line "
                            "in .adolcrc ignored!\n");
      } else {
        if (*(pos3 + 1) == '0' && (*(pos3 + 2) == 'x' || *(pos3 + 2) == 'X')) {
          start = pos3 + 3;
          base = 16;
        } else if (*(pos3 + 1) == '0') {
          start = pos3 + 2;
          base = 8;
        } else {
          start = pos3 + 1;
          base = 10;
        }
        number = strtoul(start, &end, base);
        if (end == start) {
          *pos2 = 0;
          *pos4 = 0;
          if (std::strcmp(pos1 + 1, "TAPE_DIR") == 0) {
            struct stat st;
            int err;
            path = pos3 + 1;
            err = stat(path, &st);
            if (err == 0 && S_ISDIR(st.st_mode)) {

              tapeBaseNames[0] =
                  std::string(path) + PATHSEPARATOR + ADOLC_LOCATIONS_NAME;
              tapeBaseNames[1] =
                  std::string(path) + PATHSEPARATOR + ADOLC_VALUES_NAME;
              tapeBaseNames[2] =
                  std::string(path) + PATHSEPARATOR + ADOLC_OPERATIONS_NAME;
              tapeBaseNames[3] =
                  std::string(path) + PATHSEPARATOR + ADOLC_TAYLORS_NAME;

              fprintf(
                  DIAG_OUT,
                  "ADOL-C info: using TAPE_DIR %s for all disk bound tapes\n",
                  path);
            } else
              fprintf(
                  DIAG_OUT,
                  "ADOL-C warning: TAPE_DIR %s in .adolcrc is not an existing "
                  "directory,\n will continue using %s for writing tapes\n",
                  path, TAPE_DIR);
          } else
            fprintf(DIAG_OUT, "ADOL-C warning: Unable to parse number in "
                              ".adolcrc!\n");
        } else {
          *pos2 = 0;
          *pos4 = 0;
          if (std::strcmp(pos1 + 1, "OBUFSIZE") == 0) {
            operationBufferSize(number);
            fprintf(DIAG_OUT, "Found operation buffer size: %zu\n", number);
          } else if (std::strcmp(pos1 + 1, "LBUFSIZE") == 0) {
            locationBufferSize(number);
            fprintf(DIAG_OUT, "Found location buffer size: %zu\n", number);
          } else if (std::strcmp(pos1 + 1, "VBUFSIZE") == 0) {
            valueBufferSize(number);
            fprintf(DIAG_OUT, "Found value buffer size: %zu\n", number);
          } else if (std::strcmp(pos1 + 1, "TBUFSIZE") == 0) {
            taylorBufferSize(number);
            fprintf(DIAG_OUT, "Found taylor buffer size: %zu\n", number);
          } else if (std::strcmp(pos1 + 1, "TBUFNUM") == 0) {
            maxNumberTaylorBuffers(number);
            fprintf(DIAG_OUT,
                    "Found maximal number of taylor buffers: "
                    "%zu\n",
                    number);
          } else if (std::strcmp(pos1 + 1, "INITLIVE") == 0) {
            initialStoreSize(number);
            fprintf(DIAG_OUT, "Found initial live variable store size : %zu\n",
                    number);
            checkInitialStoreSize();
          } else {
            fprintf(DIAG_OUT, "ADOL-C warning: Unable to parse "
                              "parameter name in .adolcrc!\n");
          }
        }
      }
    }
    fprintf(DIAG_OUT, "****************************************\n\n");
    if (configFile != nullptr)
      fclose(configFile);
  }
  return tapeBaseNames;
}

#ifdef SPARSE
/* updates the tape infos on sparse Jac for the given ID  */
void ValueTape::setTapeInfoJacSparse(dSparseJacInfos sJinfos) {
  // free memory of tape entry that had been used previously
  freeSparseJacInfos(
      tapeInfos_.ptapeInfos_.sJinfos.y, tapeInfos_.ptapeInfos_.sJinfos.B,
      tapeInfos_.ptapeInfos_.sJinfos.JP, tapeInfos_.ptapeInfos_.sJinfos.g,
      tapeInfos_.ptapeInfos_.sJinfos.jr1d,
      tapeInfos_.ptapeInfos_.sJinfos.seed_rows,
      tapeInfos_.ptapeInfos_.sJinfos.seed_clms,
      tapeInfos_.ptapeInfos_.sJinfos.depen);
  tapeInfos_.ptapeInfos_.sJinfos.y = sJinfos.y;
  tapeInfos_.ptapeInfos_.sJinfos.Seed = sJinfos.Seed;
  tapeInfos_.ptapeInfos_.sJinfos.B = sJinfos.B;
  tapeInfos_.ptapeInfos_.sJinfos.JP = sJinfos.JP;
  tapeInfos_.ptapeInfos_.sJinfos.depen = sJinfos.depen;
  tapeInfos_.ptapeInfos_.sJinfos.nnz_in = sJinfos.nnz_in;
  tapeInfos_.ptapeInfos_.sJinfos.seed_clms = sJinfos.seed_clms;
  tapeInfos_.ptapeInfos_.sJinfos.seed_rows = sJinfos.seed_rows;
  tapeInfos_.ptapeInfos_.sJinfos.g = sJinfos.g;
  tapeInfos_.ptapeInfos_.sJinfos.jr1d = sJinfos.jr1d;
}

/* updates the tape infos on sparse Hess for the given ID  */
void ValueTape::setTapeInfoHessSparse(SparseHessInfos sHinfos) {
  // free memory of tape entry that had been used previously
  freeSparseHessInfos(
      tapeInfos_.ptapeInfos_.sHinfos.Hcomp, tapeInfos_.ptapeInfos_.sHinfos.Xppp,
      tapeInfos_.ptapeInfos_.sHinfos.Yppp, tapeInfos_.ptapeInfos_.sHinfos.Zppp,
      tapeInfos_.ptapeInfos_.sHinfos.Upp, tapeInfos_.ptapeInfos_.sHinfos.HP,
      tapeInfos_.ptapeInfos_.sHinfos.g, tapeInfos_.ptapeInfos_.sHinfos.hr,
      tapeInfos_.ptapeInfos_.sHinfos.p, tapeInfos_.ptapeInfos_.sHinfos.indep);
  tapeInfos_.ptapeInfos_.sHinfos.Hcomp = sHinfos.Hcomp;
  tapeInfos_.ptapeInfos_.sHinfos.Xppp = sHinfos.Xppp;
  tapeInfos_.ptapeInfos_.sHinfos.Yppp = sHinfos.Yppp;
  tapeInfos_.ptapeInfos_.sHinfos.Zppp = sHinfos.Zppp;
  tapeInfos_.ptapeInfos_.sHinfos.Upp = sHinfos.Upp;
  tapeInfos_.ptapeInfos_.sHinfos.HP = sHinfos.HP;
  tapeInfos_.ptapeInfos_.sHinfos.indep = sHinfos.indep;
  tapeInfos_.ptapeInfos_.sHinfos.nnz_in = sHinfos.nnz_in;
  tapeInfos_.ptapeInfos_.sHinfos.p = sHinfos.p;
  tapeInfos_.ptapeInfos_.sHinfos.g = sHinfos.g;
  tapeInfos_.ptapeInfos_.sHinfos.hr = sHinfos.hr;
}
#endif

#include <adolc/adolcerror.h>
#include <adolc/valuetape/tapeinfos.h>
#include <cstring> // for memset
#include <iostream>

TapeInfos::TapeInfos(short tapeId) { tapeId_ = tapeId; }

TapeInfos::~TapeInfos() {
  if (op_file) {
    fclose(op_file);
    op_file = nullptr;
  }
  delete[] opBuffer;
  opBuffer = nullptr;

  if (val_file) {
    fclose(val_file);
    val_file = nullptr;
  }

  delete[] valBuffer;
  valBuffer = nullptr;

  if (loc_file) {
    fclose(loc_file);
    loc_file = nullptr;
  }
  delete[] locBuffer;
  locBuffer = nullptr;

  if (tay_file) {
    fclose(tay_file);
    tay_file = nullptr;
  }

  delete[] tayBuffer;
  --numTBuffersInUse;
  tayBuffer = nullptr;

  delete[] switchlocs;
  switchlocs = nullptr;

  delete[] signature;
  signature = nullptr;
}

TapeInfos::TapeInfos(TapeInfos &&other) noexcept
    : tapeId_(other.tapeId_), inUse(other.inUse), numInds(other.numInds),
      numDeps(other.numDeps), keepTaylors(other.keepTaylors),
      traceFlag(other.traceFlag), tapingComplete(other.tapingComplete),
      op_file(other.op_file), opBuffer(other.opBuffer), currOp(other.currOp),
      lastOpP1(other.lastOpP1), numOps_Tape(other.numOps_Tape),
      num_eq_prod(other.num_eq_prod), val_file(other.val_file),
      valBuffer(other.valBuffer), currVal(other.currVal),
      lastValP1(other.lastValP1), numVals_Tape(other.numVals_Tape),
      loc_file(other.loc_file), locBuffer(other.locBuffer),
      currLoc(other.currLoc), lastLocP1(other.lastLocP1),
      numLocs_Tape(other.numLocs_Tape), tay_file(other.tay_file),
      tayBuffer(other.tayBuffer), currTay(other.currTay),
      lastTayP1(other.lastTayP1), numTays_Tape(other.numTays_Tape),
      numTBuffersInUse(other.numTBuffersInUse),
      nextBufferNumber(other.nextBufferNumber),
      lastTayBlockInCore(other.lastTayBlockInCore), T_for(other.T_for),
      deg_save(other.deg_save), tay_numInds(other.tay_numInds),
      tay_numDeps(other.tay_numDeps), lowestXLoc_for(other.lowestXLoc_for),
      lowestYLoc_for(other.lowestYLoc_for),
      lowestXLoc_rev(other.lowestXLoc_rev),
      lowestYLoc_rev(other.lowestYLoc_rev), cpIndex(other.cpIndex),
      numDirs_rev(other.numDirs_rev),
      lowestXLoc_ext_v2(other.lowestXLoc_ext_v2),
      lowestYLoc_ext_v2(other.lowestYLoc_ext_v2), dp_T0(other.dp_T0),
      gDegree(other.gDegree), numTay(other.numTay), workMode(other.workMode),
      dpp_T(other.dpp_T), rp_T(other.rp_T), rpp_T(other.rpp_T),
      rp_A(other.rp_A), rpp_A(other.rpp_A), upp_A(other.upp_A),
      ext_diff_fct_index(other.ext_diff_fct_index),
      in_nested_ctx(other.in_nested_ctx), numSwitches(other.numSwitches),
      switchlocs(other.switchlocs), signature(other.signature) {
  std::copy(std::begin(other.stats), std::end(other.stats), std::begin(stats));

  // Null out source object's pointers
  other.op_file = nullptr;
  other.opBuffer = nullptr;
  other.currOp = nullptr;
  other.lastOpP1 = nullptr;

  other.val_file = nullptr;
  other.valBuffer = nullptr;
  other.currVal = nullptr;
  other.lastValP1 = nullptr;

  other.loc_file = nullptr;
  other.locBuffer = nullptr;
  other.currLoc = nullptr;
  other.lastLocP1 = nullptr;

  other.tay_file = nullptr;
  other.tayBuffer = nullptr;
  other.currTay = nullptr;
  other.lastTayP1 = nullptr;

  other.T_for = nullptr;
  other.lowestXLoc_ext_v2 = nullptr;
  other.lowestYLoc_ext_v2 = nullptr;

  other.dp_T0 = nullptr;
  other.dpp_T = nullptr;

  other.rp_T = nullptr;
  other.rpp_T = nullptr;
  other.rp_A = nullptr;
  other.rpp_A = nullptr;
  other.upp_A = nullptr;

  other.switchlocs = nullptr;
  other.signature = nullptr;
}

TapeInfos &TapeInfos::operator=(TapeInfos &&other) noexcept {
  if (this != &other) {
    // Free existing resources to avoid leaks
    delete[] opBuffer;
    delete[] valBuffer;
    delete[] locBuffer;
    delete[] tayBuffer;

    delete[] switchlocs;
    delete[] signature;

    // **2. Move data members**
    tapeId_ = other.tapeId_;
    inUse = other.inUse;
    numInds = other.numInds;
    numDeps = other.numDeps;
    keepTaylors = other.keepTaylors;
    traceFlag = other.traceFlag;
    tapingComplete = other.tapingComplete;
    std::copy(std::begin(other.stats), std::end(other.stats),
              std::begin(stats));

    op_file = other.op_file;
    opBuffer = other.opBuffer;
    currOp = other.currOp;
    lastOpP1 = other.lastOpP1;
    numOps_Tape = other.numOps_Tape;
    num_eq_prod = other.num_eq_prod;

    val_file = other.val_file;
    valBuffer = other.valBuffer;
    currVal = other.currVal;
    lastValP1 = other.lastValP1;
    numVals_Tape = other.numVals_Tape;

    loc_file = other.loc_file;
    locBuffer = other.locBuffer;
    currLoc = other.currLoc;
    lastLocP1 = other.lastLocP1;
    numLocs_Tape = other.numLocs_Tape;

    tay_file = other.tay_file;
    tayBuffer = other.tayBuffer;
    currTay = other.currTay;
    lastTayP1 = other.lastTayP1;
    numTays_Tape = other.numTays_Tape;
    numTBuffersInUse = other.numTBuffersInUse;
    nextBufferNumber = other.nextBufferNumber;
    lastTayBlockInCore = other.lastTayBlockInCore;

    T_for = other.T_for;
    deg_save = other.deg_save;
    tay_numInds = other.tay_numInds;
    tay_numDeps = other.tay_numDeps;

    lowestXLoc_for = other.lowestXLoc_for;
    lowestYLoc_for = other.lowestYLoc_for;
    lowestXLoc_rev = other.lowestXLoc_rev;
    lowestYLoc_rev = other.lowestYLoc_rev;
    cpIndex = other.cpIndex;
    numDirs_rev = other.numDirs_rev;

    lowestXLoc_ext_v2 = other.lowestXLoc_ext_v2;
    lowestYLoc_ext_v2 = other.lowestYLoc_ext_v2;

    dp_T0 = other.dp_T0;
    gDegree = other.gDegree;
    numTay = other.numTay;
    workMode = other.workMode;
    dpp_T = other.dpp_T;

    rp_T = other.rp_T;
    rpp_T = other.rpp_T;
    rp_A = other.rp_A;
    rpp_A = other.rpp_A;
    upp_A = other.upp_A;

    ext_diff_fct_index = other.ext_diff_fct_index;
    in_nested_ctx = other.in_nested_ctx;
    numSwitches = other.numSwitches;
    switchlocs = other.switchlocs;
    signature = other.signature;

    // **3. Null out source object’s pointers to prevent double deletion**
    other.op_file = nullptr;
    other.opBuffer = nullptr;
    other.currOp = nullptr;
    other.lastOpP1 = nullptr;
    other.val_file = nullptr;
    other.valBuffer = nullptr;
    other.currVal = nullptr;
    other.lastValP1 = nullptr;
    other.loc_file = nullptr;
    other.locBuffer = nullptr;
    other.currLoc = nullptr;
    other.lastLocP1 = nullptr;
    other.tay_file = nullptr;
    other.tayBuffer = nullptr;
    other.currTay = nullptr;
    other.lastTayP1 = nullptr;
    other.T_for = nullptr;
    other.lowestXLoc_ext_v2 = nullptr;
    other.lowestYLoc_ext_v2 = nullptr;
    other.dp_T0 = nullptr;
    other.dpp_T = nullptr;
    other.rp_T = nullptr;
    other.rpp_T = nullptr;
    other.rp_A = nullptr;
    other.rpp_A = nullptr;
    other.upp_A = nullptr;
    other.switchlocs = nullptr;
    other.signature = nullptr;
  }
  return *this;
}

void TapeInfos::freeTapeResources() {
  delete[] opBuffer;
  opBuffer = nullptr;

  delete[] locBuffer;
  locBuffer = nullptr;

  delete[] valBuffer;
  valBuffer = nullptr;

  if (tayBuffer) {
    delete[] tayBuffer;
    tayBuffer = nullptr;
    --numTBuffersInUse;
  }
  if (op_file) {
    fclose(op_file);
    op_file = nullptr;
  }
  if (loc_file) {
    fclose(loc_file);
    loc_file = nullptr;
  }
  if (val_file) {
    fclose(val_file);
    val_file = nullptr;
  }
  if (tay_file) {
    fclose(tay_file);
    tay_file = nullptr;
  }
  if (signature) {
    delete[] signature;
    signature = nullptr;
  }
}

#ifdef SPARSE
SparseJacInfos::~SparseJacInfos() {
  myfree1(y);
  y = nullptr;
  myfree2(B);
  B = nullptr;

  for (int i = 0; i < depen; i++) {
    free(JP[i]);
  }
  free(JP);
  JP = nullptr;

#ifdef HAVE_LIBCOLPACK
  if (g)
    delete (BipartiteGraphPartialColoringInterface *)g;
  if (jr1d)
    delete (JacobianRecovery1D *)jr1d;
#endif
}

SparseHessInfos::~SparseHessInfos() {
  myfree2(Hcomp);
  Hcomp = nullptr;

  myfree3(Xppp);
  Xppp = nullptr;

  myfree3(Yppp);
  Yppp = nullptr;

  myfree3(Zppp);
  Zppp = nullptr;

  myfree2(Upp);
  Upp = nullptr;

  if (HP) {
    for (int i = 0; i < indep; i++)
      free(HP[i]);
    free(HP);
    HP = nullptr;
  }

#ifdef HAVE_LIBCOLPACK
  delete (GraphColoringInterface *)g;
  g = nullptr;
  delete (HessianRecovery *)hr;
  hr = nullptr;
#endif
}

void freeSparseJacInfos(double *y, double **B, unsigned int **JP, void *g,
                        void *jr1d, int seed_rows, int seed_clms, int depen) {
  if (y)
    myfree1(y);

  if (B)
    myfree2(B);

  for (int i = 0; i < depen; i++) {
    free(JP[i]);
  }

  free(JP);

#ifdef HAVE_LIBCOLPACK
  delete (BipartiteGraphPartialColoringInterface *)g;
  delete (JacobianRecovery1D *)jr1d;
#endif
}

void freeSparseHessInfos(double **Hcomp, double ***Xppp, double ***Yppp,
                         double ***Zppp, double **Upp, unsigned int **HP,
                         void *g, void *hr, int p, int indep) {
  if (Hcomp)
    myfree2(Hcomp);

  if (Xppp)
    myfree3(Xppp);

  if (Yppp)
    myfree3(Yppp);

  if (Zppp)
    myfree3(Zppp);

  if (Upp)
    myfree2(Upp);

  if (HP) {
    for (int i = 0; i < indep; i++)
      free(HP[i]);
    free(HP);
  }

#ifdef HAVE_LIBCOLPACK
  delete (GraphColoringInterface *)g;
  delete (HessianRecovery *)hr;
#endif
}

#endif

/****************************************************************************/
/* Writes the block of size depth of taylor coefficients from point loc to  */
/* the taylor buffer. If the buffer is filled, then it is written to the    */
/* taylor tape.                                                             */
/****************************************************************************/
void TapeInfos::write_taylor(size_t loc, int keep, const char *tay_fileName) {
  double *i;
  double *T;

  T = dpp_T[loc];

  /* write data to buffer and put buffer to disk as long as data remain in
   * the T-buffer => don't create an empty value stack buffer! */
  while (currTay + keep > lastTayP1) {
    for (i = currTay; i < lastTayP1; ++i) {
      *i = (double)*T;
      /* In this assignment the precision will be sacrificed if the type
       * double is defined as float. */
      ++T;
    }
    keep -= lastTayP1 - currTay;
    put_tay_block(lastTayP1, tay_fileName);
  }

  for (i = currTay; i < currTay + keep; ++i) {
    *i = (double)*T;
    /* In this assignment the precision will be sacrificed
     * if the type double is defined as float. */
    ++T;
  }
  currTay += keep;
}

/**
 * Functions to handle the taylor tape
 */

/****************************************************************************/
/* Writes the value stack buffer onto hard disk.                            */
/****************************************************************************/
void TapeInfos::put_tay_block(double *lastTayP1, const char *tay_fileName) {
  int i, chunks;
  size_t number, remain, chunkSize = 0;

  if (tay_file == nullptr) {
    tay_file = fopen(tay_fileName, "w+b");
    if (tay_file == nullptr)
      ADOLCError::fail(ADOLCError::ErrorType::TAPING_TAYLOR_OPEN_FAILED,
                       CURRENT_LOCATION);

    number = lastTayP1 - tayBuffer;
    if (number != 0) {
      chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(double);
      chunks = number / chunkSize;
      for (i = 0; i < chunks; ++i)
        if (fwrite(tayBuffer + i * chunkSize, chunkSize * sizeof(double), 1,
                   tay_file) != 1)
          ADOLCError::fail(ADOLCError::ErrorType::TAPING_FATAL_IO_ERROR,
                           CURRENT_LOCATION);
    }
    remain = number % chunkSize;
    if (remain != 0)
      if (fwrite(tayBuffer + chunks * chunkSize, remain * sizeof(double), 1,
                 tay_file) != 1) {
        ADOLCError::fail(ADOLCError::ErrorType::TAPING_FATAL_IO_ERROR,
                         CURRENT_LOCATION);
      }
    numTays_Tape += number;
  }
  currTay = tayBuffer;
}

/****************************************************************************/
/* Gets the next (previous block) of the value stack                        */
/****************************************************************************/
void TapeInfos::get_tay_block_r() {

  lastTayBlockInCore = 0;
  const int number = stats[TapeInfos::TAY_BUFFER_SIZE];
  if (nextBufferNumber > 0) {
    if (fseek(tay_file, sizeof(double) * nextBufferNumber * number, SEEK_SET) ==
        -1)
      ADOLCError::fail(ADOLCError::ErrorType::EVAL_SEEK_VALUE_STACK,
                       CURRENT_LOCATION);

    const int chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(double);
    const int chunks = number / chunkSize;

    for (size_t i = 0; i < chunks; ++i)
      if (fread(tayBuffer + i * chunkSize, chunkSize * sizeof(double), 1,
                tay_file) != 1) {
        ADOLCError::fail(ADOLCError::ErrorType::TAPING_FATAL_IO_ERROR,
                         CURRENT_LOCATION);
      }
    const int remain = number % chunkSize;
    if (remain != 0)
      if (fread(tayBuffer + chunks * chunkSize, remain * sizeof(double), 1,
                tay_file) != 1) {
        ADOLCError::fail(ADOLCError::ErrorType::TAPING_FATAL_IO_ERROR,
                         CURRENT_LOCATION);
      }

    currTay = lastTayP1;
    --nextBufferNumber;
  }
}
/**
 * Functions for handling locations tape
 */

/****************************************************************************/
/* Writes a block of locations onto hard disk and handles file creation,   */
/* removal, ...                                                             */
/****************************************************************************/
void TapeInfos::put_loc_block(size_t *lastLocP1, const char *loc_fileName) {
  constexpr size_t chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(size_t);
  const size_t number = lastLocP1 - locBuffer;
  const size_t chunks = number / chunkSize;
  const size_t remain = number % chunkSize;

  if (loc_file == nullptr) {
    if ((loc_file = fopen(loc_fileName, "rb"))) {
#if defined(ADOLC_DEBUG)
      fprintf(DIAG_OUT, "ADOL-C debug: Old tapefile %s gets removed!\n",
              loc_fileName);
#endif
      fclose(loc_file);
      loc_file = nullptr;
      if (remove(loc_fileName))
        fprintf(DIAG_OUT, "ADOL-C warning: "
                          "Unable to remove old tapefile!\n");
      loc_file = fopen(loc_fileName, "wb");
    } else {
      loc_file = fopen(loc_fileName, "wb");
    }
  }
  for (size_t i = 0; i < chunks; ++i)
    if (fwrite(locBuffer + i * chunkSize, chunkSize * sizeof(size_t), 1,
               loc_file) != 1)
      ADOLCError::fail(ADOLCError::ErrorType::TAPING_FATAL_IO_ERROR,
                       CURRENT_LOCATION);

  if (remain != 0) {
    if (fwrite(locBuffer + chunks * chunkSize, remain * sizeof(size_t), 1,
               loc_file) != 1)
      ADOLCError::fail(ADOLCError::ErrorType::TAPING_FATAL_IO_ERROR,
                       CURRENT_LOCATION);
  }
  numLocs_Tape += number;
  currLoc = locBuffer;
}

/****************************************************************************/
/* Reads the next block of locations into the internal buffer.              */
/****************************************************************************/
void TapeInfos::get_loc_block_f() {
  size_t i, chunks;
  size_t number, remain, chunkSize;

  number = MIN_ADOLC(stats[TapeInfos::LOC_BUFFER_SIZE], numLocs_Tape);
  chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(size_t);
  chunks = number / chunkSize;
  for (i = 0; i < chunks; ++i)
    if (fread(locBuffer + i * chunkSize, chunkSize * sizeof(size_t), 1,
              loc_file) != 1)
      ADOLCError::fail(ADOLCError::ErrorType::EVAL_LOC_TAPE_READ_FAILED,
                       CURRENT_LOCATION);
  remain = number % chunkSize;
  if (remain != 0)
    if (fread(locBuffer + chunks * chunkSize, remain * sizeof(size_t), 1,
              loc_file) != 1)
      ADOLCError::fail(ADOLCError::ErrorType::EVAL_LOC_TAPE_READ_FAILED,
                       CURRENT_LOCATION);
  numLocs_Tape -= number;
  currLoc = locBuffer;
}

/****************************************************************************/
/* Reads the previous block of locations into the internal buffer.          */
/****************************************************************************/
void TapeInfos::get_loc_block_r() {
  constexpr size_t chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(size_t);
  const size_t number = stats[TapeInfos::LOC_BUFFER_SIZE];
  const size_t chunks = number / chunkSize;
  const size_t remain = number % chunkSize;

  fseek(loc_file, sizeof(size_t) * (numLocs_Tape - number), SEEK_SET);
  for (size_t i = 0; i < chunks; ++i) {
    if (fread(locBuffer + i * chunkSize, chunkSize * sizeof(size_t), 1,
              loc_file) != 1)
      ADOLCError::fail(ADOLCError::ErrorType::EVAL_LOC_TAPE_READ_FAILED,
                       CURRENT_LOCATION);
  }

  if (remain != 0) {
    if (fread(locBuffer + chunks * chunkSize, remain * sizeof(size_t), 1,
              loc_file) != 1) {
      ADOLCError::fail(ADOLCError::ErrorType::EVAL_LOC_TAPE_READ_FAILED,
                       CURRENT_LOCATION);
    }
  }
  numLocs_Tape -= stats[TapeInfos::LOC_BUFFER_SIZE];
  currLoc = lastLocP1 - *(lastLocP1 - 1);
}

/**
 * Functions for handling operations tape
 */
/****************************************************************************/
/* Puts an operation into the operation buffer. Ensures that location buffer*/
/* and constants buffer are prepared to take the belonging stuff.           */
/****************************************************************************/
void TapeInfos::put_op(OPCODES op, const char *loc_fileName,
                       const char *op_fileName, const char *val_fileName,
                       size_t reserveExtraLocations, size_t maxLocsPerOp) {

  /* make sure we have enough slots to write the locs */
  if (currLoc + maxLocsPerOp + reserveExtraLocations > lastLocP1) {
    size_t remainder = lastLocP1 - currLoc;
    if (remainder > 0)
      std::memset(currLoc, 0, (remainder - 1) * sizeof(size_t));
    *(lastLocP1 - 1) = remainder;
    put_loc_block(lastLocP1, loc_fileName);
    /* every operation writes 1 opcode */
    if (currOp + 1 == lastOpP1) {
      *currOp = end_of_op;
      put_op_block(lastOpP1, op_fileName);
      *currOp = end_of_op;
      ++currOp;
    }
    *currOp = end_of_int;
    ++currOp;
  }
  /* every operation writes <5 values --- 3 should be sufficient */
  if (currVal + 5 > lastValP1) {
    size_t valRemainder = lastValP1 - currVal;
    put_loc(valRemainder);
    /* avoid writing uninitialized memory to the file and get valgrind upset
     */
    std::memset(currVal, 0, valRemainder * sizeof(double));
    put_val_block(lastValP1, val_fileName);
    /* every operation writes 1 opcode */
    if (currOp + 1 == lastOpP1) {
      *currOp = end_of_op;
      put_op_block(lastOpP1, op_fileName);
      *currOp = end_of_op;
      ++currOp;
    }
    *currOp = end_of_val;
    ++currOp;
  }
  /* every operation writes 1 opcode */
  if (currOp + 1 == lastOpP1) {
    *currOp = end_of_op;
    put_op_block(lastOpP1, op_fileName);
    *currOp = end_of_op;
    ++currOp;
  }
  *currOp = op;
  ++currOp;
}

/****************************************************************************/
/* Writes a block of operations onto hard disk and handles file creation,   */
/* removal, ...                                                             */
/****************************************************************************/
void TapeInfos::put_op_block(unsigned char *lastOpP1, const char *op_fileName) {
  size_t i, chunks;
  size_t number, remain, chunkSize;

  if (op_file == nullptr) {
    op_file = fopen(op_fileName, "rb");
    if (op_file != nullptr) {
#if defined(ADOLC_DEBUG)
      fprintf(DIAG_OUT, "ADOL-C debug: Old tapefile %s gets removed!\n",
              op_fileName);
#endif
      fclose(op_file);
      op_file = nullptr;
      if (remove(op_fileName))
        fprintf(DIAG_OUT, "ADOL-C warning: "
                          "Unable to remove old tapefile\n");
      op_file = fopen(op_fileName, "wb");
    } else {
      op_file = fopen(op_fileName, "wb");
    }
  }

  number = lastOpP1 - opBuffer;
  chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(unsigned char);
  chunks = number / chunkSize;
  for (i = 0; i < chunks; ++i)
    if (fwrite(opBuffer + i * chunkSize, chunkSize * sizeof(unsigned char), 1,
               op_file) != 1)
      ADOLCError::fail(ADOLCError::ErrorType::TAPING_FATAL_IO_ERROR,
                       CURRENT_LOCATION);

  remain = number % chunkSize;
  if (remain != 0)
    if (fwrite(opBuffer + chunks * chunkSize, remain * sizeof(unsigned char), 1,
               op_file) != 1)
      ADOLCError::fail(ADOLCError::ErrorType::TAPING_FATAL_IO_ERROR,
                       CURRENT_LOCATION);

  numOps_Tape += number;
  currOp = opBuffer;
}

/****************************************************************************/
/* Reads the next operations block into the internal buffer.                */
/****************************************************************************/
void TapeInfos::get_op_block_f() {
  size_t i, chunks;
  size_t number, remain, chunkSize;

  number = MIN_ADOLC(stats[TapeInfos::OP_BUFFER_SIZE], numOps_Tape);
  chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(unsigned char);
  chunks = number / chunkSize;
  for (i = 0; i < chunks; ++i)
    if (fread(opBuffer + i * chunkSize, chunkSize * sizeof(unsigned char), 1,
              op_file) != 1)
      ADOLCError::fail(ADOLCError::ErrorType::EVAL_OP_TAPE_READ_FAILED,
                       CURRENT_LOCATION);
  remain = number % chunkSize;
  if (remain != 0)
    if (fread(opBuffer + chunks * chunkSize, remain * sizeof(unsigned char), 1,
              op_file) != 1)
      ADOLCError::fail(ADOLCError::ErrorType::EVAL_OP_TAPE_READ_FAILED,
                       CURRENT_LOCATION);
  numOps_Tape -= remain;
  currOp = opBuffer;
}

/****************************************************************************/
/* Reads the previous block of operations into the internal buffer.         */
/****************************************************************************/
void TapeInfos::get_op_block_r() {
  size_t i, chunks;
  size_t number, remain, chunkSize;

  number = stats[TapeInfos::OP_BUFFER_SIZE];
  fseek(op_file, sizeof(unsigned char) * (numOps_Tape - number), SEEK_SET);
  chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(unsigned char);
  chunks = number / chunkSize;
  for (i = 0; i < chunks; ++i)
    if (fread(opBuffer + i * chunkSize, chunkSize * sizeof(unsigned char), 1,
              op_file) != 1)
      ADOLCError::fail(ADOLCError::ErrorType::EVAL_OP_TAPE_READ_FAILED,
                       CURRENT_LOCATION);

  remain = number % chunkSize;
  if (remain != 0)
    if (fread(opBuffer + chunks * chunkSize, remain * sizeof(unsigned char), 1,
              op_file) != 1)
      ADOLCError::fail(ADOLCError::ErrorType::EVAL_OP_TAPE_READ_FAILED,
                       CURRENT_LOCATION);

  numOps_Tape -= number;
  currOp = opBuffer + number;
}

/**
 * functions for handling the value tape
 *
 *
 */

/****************************************************************************/
/* Writes a block of constants (real) onto hard disk and handles file       */
/* creation, removal, ...                                                   */
/****************************************************************************/
void TapeInfos::put_vals_writeBlock(double *vals, size_t numVals,
                                    const char *op_fileName,
                                    const char *val_fileName) {
  int i;

  for (i = 0; i < numVals; ++i) {
    *currVal = vals[i];
    ++currVal;
  }
  put_loc(lastValP1 - currVal);
  put_val_block(lastValP1, val_fileName);
  /* every operation writes 1 opcode */
  if (currOp + 1 == lastOpP1) {
    *currOp = end_of_op;
    put_op_block(lastOpP1, op_fileName);
    *currOp = end_of_op;
    ++currOp;
  }
  *currOp = end_of_val;
  ++currOp;
}

/****************************************************************************/
/* Writes a block of constants (real) onto tape and handles file creation   */
/* removal, ...                                                             */
/****************************************************************************/
void TapeInfos::put_val_block(double *lastValP1, const char *val_fileName) {
  size_t i, chunks;
  size_t number, remain, chunkSize;

  if (val_file == nullptr) {
    val_file = fopen(val_fileName, "rb");
    if (val_file != nullptr) {
#if defined(ADOLC_DEBUG)
      fprintf(DIAG_OUT, "ADOL-C debug: Old tapefile %s gets removed!\n",
              val_fileName);
#endif
      fclose(val_file);
      val_file = nullptr;
      if (remove(val_fileName))
        fprintf(DIAG_OUT, "ADOL-C warning: "
                          "Unable to remove old tapefile\n");
      val_file = fopen(val_fileName, "wb");
    } else {
      val_file = fopen(val_fileName, "wb");
    }
  }

  number = lastValP1 - valBuffer;
  chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(double);
  chunks = number / chunkSize;
  for (i = 0; i < chunks; ++i)
    if (fwrite(valBuffer + i * chunkSize, chunkSize * sizeof(double), 1,
               val_file) != 1)
      ADOLCError::fail(ADOLCError::ErrorType::TAPING_FATAL_IO_ERROR,
                       CURRENT_LOCATION);
  remain = number % chunkSize;
  if (remain != 0)
    if (fwrite(valBuffer + chunks * chunkSize, remain * sizeof(double), 1,
               val_file) != 1)
      ADOLCError::fail(ADOLCError::ErrorType::TAPING_FATAL_IO_ERROR,
                       CURRENT_LOCATION);
  numVals_Tape += number;
  currVal = valBuffer;
}

/****************************************************************************/
/* Reads the next block of constants into the internal buffer.              */
/****************************************************************************/
void TapeInfos::get_val_block_f() {
  size_t i, chunks;
  size_t number, remain, chunkSize;

  number = MIN_ADOLC(stats[TapeInfos::VAL_BUFFER_SIZE], numVals_Tape);
  chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(double);
  chunks = number / chunkSize;
  for (i = 0; i < chunks; ++i)
    if (fread(valBuffer + i * chunkSize, chunkSize * sizeof(double), 1,
              val_file) != 1)
      ADOLCError::fail(ADOLCError::ErrorType::EVAL_VAL_TAPE_READ_FAILED,
                       CURRENT_LOCATION);

  remain = number % chunkSize;
  if (remain != 0)
    if (fread(valBuffer + chunks * chunkSize, remain * sizeof(double), 1,
              val_file) != 1)
      ADOLCError::fail(ADOLCError::ErrorType::EVAL_VAL_TAPE_READ_FAILED,
                       CURRENT_LOCATION);
  numVals_Tape -= number;
  currVal = valBuffer;
  /* get_size_t_f(); value used in reverse only */
  ++currLoc;
}

/****************************************************************************/
/* Reads the previous block of values into the internal buffer.             */
/****************************************************************************/
void TapeInfos::get_val_block_r() {
  size_t i, chunks;
  size_t number, remain, chunkSize;
  size_t temp;

  number = stats[TapeInfos::VAL_BUFFER_SIZE];
  fseek(val_file, sizeof(double) * (numVals_Tape - number), SEEK_SET);
  chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(double);
  chunks = number / chunkSize;
  for (i = 0; i < chunks; ++i)
    if (fread(valBuffer + i * chunkSize, chunkSize * sizeof(double), 1,
              val_file) != 1)
      ADOLCError::fail(ADOLCError::ErrorType::EVAL_VAL_TAPE_READ_FAILED,
                       CURRENT_LOCATION);
  remain = number % chunkSize;
  if (remain != 0)
    if (fread(valBuffer + chunks * chunkSize, remain * sizeof(double), 1,
              val_file) != 1)
      ADOLCError::fail(ADOLCError::ErrorType::EVAL_VAL_TAPE_READ_FAILED,
                       CURRENT_LOCATION);
  numVals_Tape -= number;
  --currLoc;
  temp = *currLoc;
  currVal = lastValP1 - temp;
}

/****************************************************************************/
/* Returns the number of free constants in the real tape. Ensures that it   */
/* is at least 5.                                                           */
/****************************************************************************/
size_t TapeInfos::get_val_space(const char *op_fileName,
                                const char *val_fileName) {

  if (lastValP1 - 5 < currVal) {
    put_loc(lastValP1 - currVal);
    put_val_block(lastValP1, val_fileName);
    /* every operation writes 1 opcode */
    if (currOp + 1 == lastOpP1) {
      *currOp = end_of_op;
      put_op_block(lastOpP1, op_fileName);
      *currOp = end_of_op;
      ++currOp;
    }
    *currOp = end_of_val;
    ++currOp;
  }
  return (lastValP1 - currVal);
}

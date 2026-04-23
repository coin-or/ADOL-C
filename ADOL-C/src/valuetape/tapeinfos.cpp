#include <adolc/adolcerror.h>
#include <adolc/valuetape/infotype.h>
#include <adolc/valuetape/tapeinfos.h>
#include <cstddef>
#include <cstring> // for memset
#include <span>

TapeInfos::TapeInfos(short tapeId) { tapeId_ = tapeId; }

TapeInfos::~TapeInfos() {
  delete[] signature;
  signature = nullptr;
}

TapeInfos::TapeInfos(TapeInfos &&other) noexcept
    : opBuffer_(std::move(other.opBuffer_)),
      valBuffer_(std::move(other.valBuffer_)),
      locBuffer_(std::move(other.locBuffer_)),
      tayBuffer_(std::move(other.tayBuffer_)), stats(other.stats),
      tapeId_(other.tapeId_), numInds(other.numInds), numDeps(other.numDeps),
      keepTaylors(other.keepTaylors), num_eq_prod(other.num_eq_prod),
      nextBufferNumber(other.nextBufferNumber),
      lastTayBlockInCore(other.lastTayBlockInCore), deg_save(other.deg_save),
      tay_numInds(other.tay_numInds), tay_numDeps(other.tay_numDeps),
      workMode(other.workMode), ext_diff_fct_index(other.ext_diff_fct_index),
      nestedReverseEval(other.nestedReverseEval),
      numSwitches(other.numSwitches), signature(other.signature) {
  other.signature = nullptr;
}

TapeInfos &TapeInfos::operator=(TapeInfos &&other) noexcept {
  if (this != &other) {
    // Free existing resources to avoid leaks
    delete[] signature;

    // **2. Move data members**
    tapeId_ = other.tapeId_;
    numInds = other.numInds;
    numDeps = other.numDeps;
    keepTaylors = other.keepTaylors;
    std::copy(std::begin(other.stats), std::end(other.stats),
              std::begin(stats));

    num_eq_prod = other.num_eq_prod;
    opBuffer_ = std::move(other.opBuffer_);
    valBuffer_ = std::move(other.valBuffer_);
    locBuffer_ = std::move(other.locBuffer_);
    tayBuffer_ = std::move(other.tayBuffer_);
    nextBufferNumber = other.nextBufferNumber;
    lastTayBlockInCore = other.lastTayBlockInCore;

    deg_save = other.deg_save;
    tay_numInds = other.tay_numInds;
    tay_numDeps = other.tay_numDeps;

    workMode = other.workMode;
    ext_diff_fct_index = other.ext_diff_fct_index;
    nestedReverseEval = other.nestedReverseEval;
    numSwitches = other.numSwitches;
    signature = other.signature;

    // **3. Null out source object’s pointers to prevent double deletion**
    other.signature = nullptr;
  }
  return *this;
}

void TapeInfos::freeTapeResources() {
  opBuffer_ = {};
  valBuffer_ = {};
  locBuffer_ = {};
  tayBuffer_ = {};
  if (signature) {
    delete[] signature;
    signature = nullptr;
  }
}

/****************************************************************************/
/* Writes the block of size depth of taylor coefficients from point loc to  */
/* the taylor buffer. If the buffer is filled, then it is written to the    */
/* taylor tape.                                                             */
/****************************************************************************/
void TapeInfos::write_taylor(double *taylorCoefficientPos, std::ptrdiff_t keep,
                             const char *tay_fileName) {
  while (tayBuffer_.position() > tayBuffer_.capacity() - keep) {
    for (auto i = tayBuffer_.position(); i < tayBuffer_.capacity(); ++i) {
      tayBuffer_[i] = *taylorCoefficientPos;
      ++taylorCoefficientPos;
    }
    keep -= tayBuffer_.remainingCapacity();
    put_block<TayInfo<TapeInfos, ErrorType>>(tay_fileName,
                                             tayBuffer_.capacity());
  }

  for (int i = 0; i < keep; ++i) {
    tayBuffer_[i + tayBuffer_.position()] = *taylorCoefficientPos;
    ++taylorCoefficientPos;
  }
  tayBuffer_.position(tayBuffer_.position() + keep);
}

void TapeInfos::write_taylors(double *taylorCoefficientPos, int keep,
                              int degree, int numDir,
                              const char *tay_fileName) {
  for (int j = 0; j < numDir; ++j) {
    for (int i = 0; i < keep; ++i) {
      if (tayBuffer_.position() == tayBuffer_.capacity())
        put_block<TayInfo<TapeInfos, ErrorType>>(tay_fileName,
                                                 tayBuffer_.capacity());

      tayBuffer_.writeAndAdvance(*taylorCoefficientPos);
      ++taylorCoefficientPos;
    }
    if (degree > keep)
      taylorCoefficientPos += degree - keep;
  }
}

void TapeInfos::write_scaylors(const double *taylorCoefficientPos,
                               std::ptrdiff_t size, const char *tay_fileName) {
  size_t pos = 0;
  while (tayBuffer_.position() > tayBuffer_.capacity() - size) {
    std::span<double> taySpan(tayBuffer_.current(),
                              tayBuffer_.begin() + tayBuffer_.capacity());
    for (double &tay : taySpan) {
      tay = taylorCoefficientPos[pos++];
    }
    size -= tayBuffer_.remainingCapacity();
    put_block<TayInfo<TapeInfos, ErrorType>>(tay_fileName,
                                             tayBuffer_.capacity());
  }

  std::span<double> tayBufferSpan(tayBuffer_.current(),
                                  tayBuffer_.current() + size);
  for (double &tay : tayBufferSpan) {
    tay = taylorCoefficientPos[pos++];
  }
  tayBuffer_.position(tayBuffer_.position() + size);
}

void TapeInfos::get_taylors(double *taylorCoefficients, std::ptrdiff_t degree) {
  double *T = taylorCoefficients + degree;
  while (tayBuffer_.position() < static_cast<size_t>(degree)) {
    std::span<double> taySpan(tayBuffer_.begin(), tayBuffer_.current());
    for (auto tay = taySpan.rbegin(); tay != taySpan.rend(); tay++) {
      *(T--) = *tay;
    }
    degree -= tayBuffer_.position();
    get_tay_block_r();
  }

  /* Copy the remaining values from the stack into the buffer ... */
  for (int j = 0; j < degree; ++j) {
    *(--T) = tayBuffer_.retreatAndRead();
  }
}

void TapeInfos::get_taylors_p(double *taylorCoefficients, int degree,
                              int numDir) {
  double *T = taylorCoefficients + (static_cast<ptrdiff_t>(degree * numDir));

  /* update the directions except the base point parts */
  for (int j = 0; j < numDir; ++j) {
    for (int i = 1; i < degree; ++i) {
      if (tayBuffer_.position() == 0)
        get_tay_block_r();

      --T;
      *T = tayBuffer_.retreatAndRead();
    }
    --T; /* skip the base point part */
  }
  /* now update the base point parts */
  if (tayBuffer_.position() == 0)
    get_tay_block_r();

  tayBuffer_.retreat();
  for (int i = 0; i < numDir; ++i) {
    *T = *tayBuffer_.current();
    T += degree;
  }
}
/**
 * Functions to handle the taylor tape
 */

/****************************************************************************/
/* Gets the next (previous block) of the value stack                        */
/****************************************************************************/
void TapeInfos::get_tay_block_r() {

  lastTayBlockInCore = 0;
  const size_t number = stats[TapeInfos::TAY_BUFFER_SIZE];
  if (fseek(tayBuffer_.file(),
            static_cast<long>(sizeof(double) * nextBufferNumber * number),
            SEEK_SET) == -1)
    ADOLCError::fail(ADOLCError::ErrorType::EVAL_SEEK_VALUE_STACK,
                     CURRENT_LOCATION);

  const size_t chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(double);
  const size_t chunks = number / chunkSize;

  for (size_t i = 0; i < chunks; ++i)
    if (fread(tayBuffer_.begin() + i * chunkSize, chunkSize * sizeof(double), 1,
              tayBuffer_.file()) != 1) {
      ADOLCError::fail(ADOLCError::ErrorType::TAPING_FATAL_IO_ERROR,
                       CURRENT_LOCATION);
    }
  const int remain = number % chunkSize;
  if (remain != 0)
    if (fread(tayBuffer_.begin() + chunks * chunkSize, remain * sizeof(double),
              1, tayBuffer_.file()) != 1) {
      ADOLCError::fail(ADOLCError::ErrorType::TAPING_FATAL_IO_ERROR,
                       CURRENT_LOCATION);
    }

  tayBuffer_.position(tayBuffer_.capacity());
  --nextBufferNumber;
}
/**
 * Functions for handling locations tape
 */

/****************************************************************************/
/* Reads the next block of locations into the internal buffer.              */
/****************************************************************************/
void TapeInfos::get_loc_block_f() {
  size_t i, chunks;
  size_t number, remain, chunkSize;

  number = MIN_ADOLC(stats[TapeInfos::LOC_BUFFER_SIZE], locBuffer_.numOnTape());
  chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(size_t);
  chunks = number / chunkSize;
  for (i = 0; i < chunks; ++i)
    if (fread(locBuffer_.begin() + i * chunkSize, chunkSize * sizeof(size_t), 1,
              locBuffer_.file()) != 1)
      ADOLCError::fail(ADOLCError::ErrorType::EVAL_LOC_TAPE_READ_FAILED,
                       CURRENT_LOCATION);
  remain = number % chunkSize;
  if (remain != 0)
    if (fread(locBuffer_.begin() + chunks * chunkSize, remain * sizeof(size_t),
              1, locBuffer_.file()) != 1)
      ADOLCError::fail(ADOLCError::ErrorType::EVAL_LOC_TAPE_READ_FAILED,
                       CURRENT_LOCATION);
  locBuffer_.numOnTape(locBuffer_.numOnTape() - number);
  locBuffer_.position(0);
}

/****************************************************************************/
/* Reads the previous block of locations into the internal buffer.          */
/****************************************************************************/
void TapeInfos::get_loc_block_r() {
  constexpr size_t chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(size_t);
  const size_t number = stats[TapeInfos::LOC_BUFFER_SIZE];
  const size_t chunks = number / chunkSize;
  const size_t remain = number % chunkSize;

  fseek(locBuffer_.file(),
        static_cast<long>(sizeof(size_t) * (locBuffer_.numOnTape() - number)),
        SEEK_SET);
  for (size_t i = 0; i < chunks; ++i) {
    if (fread(locBuffer_.begin() + i * chunkSize, chunkSize * sizeof(size_t), 1,
              locBuffer_.file()) != 1)
      ADOLCError::fail(ADOLCError::ErrorType::EVAL_LOC_TAPE_READ_FAILED,
                       CURRENT_LOCATION);
  }

  if (remain != 0) {
    if (fread(locBuffer_.begin() + chunks * chunkSize, remain * sizeof(size_t),
              1, locBuffer_.file()) != 1) {
      ADOLCError::fail(ADOLCError::ErrorType::EVAL_LOC_TAPE_READ_FAILED,
                       CURRENT_LOCATION);
    }
  }
  locBuffer_.numOnTape(locBuffer_.numOnTape() -
                       stats[TapeInfos::LOC_BUFFER_SIZE]);
  locBuffer_.position(locBuffer_.capacity() -
                      locBuffer_[locBuffer_.capacity() - 1]);
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
                       size_t reserveExtraLocations) {
  using ADOLC::detail::LocInfo;
  using ADOLC::detail::OpInfo;
  using ADOLC::detail::ValInfo;
  using ADOLCError::ErrorType;
  /* make sure we have enough slots to write the locs */
  if (locBuffer_.position() >
      locBuffer_.capacity() - maxLocsPerOp - reserveExtraLocations) {
    const size_t remainder = locBuffer_.remainingCapacity();
    if (remainder > 0)
      std::memset(locBuffer_.current(), 0, (remainder - 1) * sizeof(size_t));
    locBuffer_[locBuffer_.capacity() - 1] = remainder;
    put_block<LocInfo<TapeInfos, ErrorType>>(loc_fileName,
                                             locBuffer_.capacity());
    /* every operation writes 1 opcode */
    if (opBuffer_.position() == opBuffer_.capacity() - 1) {
      opBuffer_.writeCurrent(static_cast<unsigned char>(end_of_op));
      put_block<OpInfo<TapeInfos, ErrorType>>(op_fileName,
                                              opBuffer_.capacity());
      opBuffer_.writeAndAdvance(static_cast<unsigned char>(end_of_op));
    }
    opBuffer_.writeAndAdvance(static_cast<unsigned char>(end_of_int));
  }
  /* every operation writes <5 values --- 3 should be sufficient */
  if (valBuffer_.position() > valBuffer_.capacity() - 5) {
    const size_t valRemainder = valBuffer_.remainingCapacity();
    put_loc(valRemainder);
    /* avoid writing uninitialized memory to the file and get valgrind upset
     */
    std::memset(valBuffer_.current(), 0, valRemainder * sizeof(double));
    put_block<ValInfo<TapeInfos, ErrorType>>(val_fileName,
                                             valBuffer_.capacity());
    /* every operation writes 1 opcode */
    if (opBuffer_.position() == opBuffer_.capacity() - 1) {
      opBuffer_.writeCurrent(static_cast<unsigned char>(end_of_op));
      put_block<OpInfo<TapeInfos, ErrorType>>(op_fileName,
                                              opBuffer_.capacity());
      opBuffer_.writeAndAdvance(static_cast<unsigned char>(end_of_op));
    }
    opBuffer_.writeAndAdvance(static_cast<unsigned char>(end_of_val));
  }
  /* every operation writes 1 opcode */
  if (opBuffer_.position() == opBuffer_.capacity() - 1) {
    opBuffer_.writeCurrent(static_cast<unsigned char>(end_of_op));
    put_block<OpInfo<TapeInfos, ErrorType>>(op_fileName, opBuffer_.capacity());
    opBuffer_.writeAndAdvance(static_cast<unsigned char>(end_of_op));
  }
  opBuffer_.writeAndAdvance(static_cast<unsigned char>(op));
}

/****************************************************************************/
/* Reads the next operations block into the internal buffer.                */
/****************************************************************************/
void TapeInfos::get_op_block_f() {
  size_t i, chunks;
  size_t number, remain, chunkSize;

  number = MIN_ADOLC(stats[TapeInfos::OP_BUFFER_SIZE], opBuffer_.numOnTape());
  chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(unsigned char);
  chunks = number / chunkSize;
  for (i = 0; i < chunks; ++i)
    if (fread(opBuffer_.begin() + i * chunkSize,
              chunkSize * sizeof(unsigned char), 1, opBuffer_.file()) != 1)
      ADOLCError::fail(ADOLCError::ErrorType::EVAL_OP_TAPE_READ_FAILED,
                       CURRENT_LOCATION);
  remain = number % chunkSize;
  if (remain != 0)
    if (fread(opBuffer_.begin() + chunks * chunkSize,
              remain * sizeof(unsigned char), 1, opBuffer_.file()) != 1)
      ADOLCError::fail(ADOLCError::ErrorType::EVAL_OP_TAPE_READ_FAILED,
                       CURRENT_LOCATION);
  opBuffer_.numOnTape(opBuffer_.numOnTape() - remain);
  opBuffer_.position(0);
}

/****************************************************************************/
/* Reads the previous block of operations into the internal buffer.         */
/****************************************************************************/
void TapeInfos::get_op_block_r() {
  size_t i, chunks;
  size_t number, remain, chunkSize;

  number = stats[TapeInfos::OP_BUFFER_SIZE];
  fseek(opBuffer_.file(),
        static_cast<long>(sizeof(unsigned char) *
                          (opBuffer_.numOnTape() - number)),
        SEEK_SET);
  chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(unsigned char);
  chunks = number / chunkSize;
  for (i = 0; i < chunks; ++i)
    if (fread(opBuffer_.begin() + i * chunkSize,
              chunkSize * sizeof(unsigned char), 1, opBuffer_.file()) != 1)
      ADOLCError::fail(ADOLCError::ErrorType::EVAL_OP_TAPE_READ_FAILED,
                       CURRENT_LOCATION);

  remain = number % chunkSize;
  if (remain != 0)
    if (fread(opBuffer_.begin() + chunks * chunkSize,
              remain * sizeof(unsigned char), 1, opBuffer_.file()) != 1)
      ADOLCError::fail(ADOLCError::ErrorType::EVAL_OP_TAPE_READ_FAILED,
                       CURRENT_LOCATION);

  opBuffer_.numOnTape(opBuffer_.numOnTape() - number);
  opBuffer_.position(number);
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
  using ADOLC::detail::OpInfo;
  using ADOLC::detail::ValInfo;
  using ADOLCError::ErrorType;
  for (size_t i = 0; i < numVals; ++i) {
    valBuffer_.writeAndAdvance(vals[i]);
  }
  put_loc(valBuffer_.capacity() - valBuffer_.position());
  put_block<ValInfo<TapeInfos, ErrorType>>(val_fileName, valBuffer_.capacity());
  /* every operation writes 1 opcode */
  if (opBuffer_.position() == opBuffer_.capacity() - 1) {
    opBuffer_.writeCurrent(static_cast<unsigned char>(end_of_op));
    put_block<OpInfo<TapeInfos, ErrorType>>(op_fileName, opBuffer_.capacity());
    opBuffer_.writeAndAdvance(static_cast<unsigned char>(end_of_op));
  }
  opBuffer_.writeAndAdvance(static_cast<unsigned char>(end_of_val));
}

/****************************************************************************/
/* Reads the next block of constants into the internal buffer.              */
/****************************************************************************/
void TapeInfos::get_val_block_f() {
  size_t i, chunks;
  size_t number, remain, chunkSize;

  number = MIN_ADOLC(stats[TapeInfos::VAL_BUFFER_SIZE], valBuffer_.numOnTape());
  chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(double);
  chunks = number / chunkSize;
  for (i = 0; i < chunks; ++i)
    if (fread(valBuffer_.begin() + i * chunkSize, chunkSize * sizeof(double), 1,
              valBuffer_.file()) != 1)
      ADOLCError::fail(ADOLCError::ErrorType::EVAL_VAL_TAPE_READ_FAILED,
                       CURRENT_LOCATION);

  remain = number % chunkSize;
  if (remain != 0)
    if (fread(valBuffer_.begin() + chunks * chunkSize, remain * sizeof(double),
              1, valBuffer_.file()) != 1)
      ADOLCError::fail(ADOLCError::ErrorType::EVAL_VAL_TAPE_READ_FAILED,
                       CURRENT_LOCATION);
  valBuffer_.numOnTape(valBuffer_.numOnTape() - number);
  valBuffer_.position(0);
  /* get_size_t_f(); value used in reverse only */
  locBuffer_.advance();
}

/****************************************************************************/
/* Reads the previous block of values into the internal buffer.             */
/****************************************************************************/
void TapeInfos::get_val_block_r() {
  size_t i, chunks;
  size_t number, remain, chunkSize;
  size_t temp;

  number = stats[TapeInfos::VAL_BUFFER_SIZE];
  fseek(valBuffer_.file(),
        static_cast<long>(sizeof(double) * (valBuffer_.numOnTape() - number)),
        SEEK_SET);
  chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(double);
  chunks = number / chunkSize;
  for (i = 0; i < chunks; ++i)
    if (fread(valBuffer_.begin() + i * chunkSize, chunkSize * sizeof(double), 1,
              valBuffer_.file()) != 1)
      ADOLCError::fail(ADOLCError::ErrorType::EVAL_VAL_TAPE_READ_FAILED,
                       CURRENT_LOCATION);
  remain = number % chunkSize;
  if (remain != 0)
    if (fread(valBuffer_.begin() + chunks * chunkSize, remain * sizeof(double),
              1, valBuffer_.file()) != 1)
      ADOLCError::fail(ADOLCError::ErrorType::EVAL_VAL_TAPE_READ_FAILED,
                       CURRENT_LOCATION);
  valBuffer_.numOnTape(valBuffer_.numOnTape() - number);
  temp = locBuffer_.retreatAndRead();
  valBuffer_.position(valBuffer_.capacity() - temp);
}

/****************************************************************************/
/* Returns the number of free constants in the real tape. Ensures that it   */
/* is at least 5.                                                           */
/****************************************************************************/
size_t TapeInfos::get_val_space(const char *op_fileName,
                                const char *val_fileName) {

  using ADOLC::detail::OpInfo;
  using ADOLC::detail::ValInfo;
  using ADOLCError::ErrorType;

  if (valBuffer_.position() > valBuffer_.capacity() - 5) {
    put_loc(valBuffer_.remainingCapacity());
    put_block<ValInfo<TapeInfos, ErrorType>>(val_fileName,
                                             valBuffer_.capacity());
    /* every operation writes 1 opcode */
    if (opBuffer_.position() == opBuffer_.capacity() - 1) {
      opBuffer_.writeCurrent(static_cast<unsigned char>(end_of_op));
      put_block<OpInfo<TapeInfos, ErrorType>>(op_fileName,
                                              opBuffer_.capacity());
      opBuffer_.writeAndAdvance(static_cast<unsigned char>(end_of_op));
    }
    opBuffer_.writeAndAdvance(static_cast<unsigned char>(end_of_val));
  }
  return (valBuffer_.remainingCapacity());
}

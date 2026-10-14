#ifndef ADOLC_TAPE_RECORDING_CONTEXT_H
#define ADOLC_TAPE_RECORDING_CONTEXT_H

#include <adolc/valuetape/infotype.h>
#include <adolc/valuetape/tapeinfos.h>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <utility>

namespace ADOLC::detail {

/**
 * @brief Mutable buffers and bookkeeping used while recording a tape.
 *
 * The context owns the canonical tape buffers. Evaluation contexts may
 * temporarily take ownership in exclusive mode or create non-owning views in
 * shared mode.
 */
struct TapeRecordingContext {
  using StatEntries = TapeInfos::StatEntries;
  static constexpr StatEntries OP_BUFFER_SIZE = TapeInfos::OP_BUFFER_SIZE;
  static constexpr StatEntries NUM_OPERATIONS = TapeInfos::NUM_OPERATIONS;
  static constexpr StatEntries OP_FILE_ACCESS = TapeInfos::OP_FILE_ACCESS;

  static constexpr StatEntries LOC_BUFFER_SIZE = TapeInfos::LOC_BUFFER_SIZE;
  static constexpr StatEntries NUM_LOCATIONS = TapeInfos::NUM_LOCATIONS;
  static constexpr StatEntries LOC_FILE_ACCESS = TapeInfos::LOC_FILE_ACCESS;

  static constexpr StatEntries VAL_BUFFER_SIZE = TapeInfos::VAL_BUFFER_SIZE;
  static constexpr StatEntries NUM_VALUES = TapeInfos::NUM_VALUES;
  static constexpr StatEntries VAL_FILE_ACCESS = TapeInfos::VAL_FILE_ACCESS;

  static constexpr StatEntries TAY_BUFFER_SIZE = TapeInfos::TAY_BUFFER_SIZE;
  static constexpr StatEntries NUM_TAYS = TapeInfos::NUM_TAYS;

  ~TapeRecordingContext() {
    delete[] signature;
    signature = nullptr;
    delete[] paramstore;
    paramstore = nullptr;
  }

  TapeRecordingContext() = default;

  TapeRecordingContext(const TapeRecordingContext &) = delete;
  TapeRecordingContext &operator=(const TapeRecordingContext &) = delete;

  TapeRecordingContext(TapeRecordingContext &&other) noexcept {
    moveData(std::move(other));
  }

  TapeRecordingContext &operator=(TapeRecordingContext &&other) noexcept {
    if (this != &other)
      moveData(std::move(other));
    return *this;
  }
  ADOLC::detail::OpBuffer opBuffer_{};
  ADOLC::detail::ValBuffer valBuffer_{};
  ADOLC::detail::LocBuffer locBuffer_{};
  ADOLC::detail::TayBuffer tayBuffer_{};

  size_t numInds{0};
  size_t numDeps{0};
  // 1 - write taylor stack in taping mode
  int keepTaylors{0};

  size_t num_eq_prod{0};

  // degree to save and saved respectively
  int deg_save{0};
  // # of independents for the taylor stack
  size_t tay_numInds{0};
  // # of dependents for the taylor stack
  size_t tay_numDeps{0};

  size_t numSwitches{0};
  /**
   * Indicates that reverse evaluation of this tape happens inside an outer
   * tape evaluation.
   *
   * First-order reverse uses this flag to accumulate adjoints for independent
   * and dependent variables into the outer tape instead of overwriting them.
   */
  bool nestedReverseEval{false};

  // the next Taylor buffer to read back
  size_t nextBufferNumber{0};

  // == 1 if last taylor buffer is still in
  // in core(first call of reverse)
  char lastTayBlockInCore{0};
  double *signature{nullptr};
  double *paramstore{nullptr};

private:
  void moveData(TapeRecordingContext &&other) noexcept {
    opBuffer_ = std::move(other.opBuffer_);
    valBuffer_ = std::move(other.valBuffer_);
    locBuffer_ = std::move(other.locBuffer_);
    tayBuffer_ = std::move(other.tayBuffer_);

    numInds = other.numInds;
    numDeps = other.numDeps;
    keepTaylors = other.keepTaylors;
    num_eq_prod = other.num_eq_prod;
    deg_save = other.deg_save;
    tay_numInds = other.tay_numInds;
    tay_numDeps = other.tay_numDeps;
    numSwitches = other.numSwitches;
    nestedReverseEval = other.nestedReverseEval;
    nextBufferNumber = other.nextBufferNumber;
    lastTayBlockInCore = other.lastTayBlockInCore;

    delete[] signature;
    signature = std::exchange(other.signature, nullptr);
    delete[] paramstore;
    paramstore = std::exchange(other.paramstore, nullptr);
  }

public:
  // functions for handling loc tape
  void put_loc(size_t loc) { locBuffer_.writeAndAdvance(loc); }

  /**
   * @brief Ensure that the tape file associated with Info exists and is ready
   *        for writing.
   */
  template <InfoType<TapeRecordingContext, ErrorType> Info>
  void openFile(const char *fileName) {
    using ADOLCError::fail;
    using ADOLCError::ErrorType::CANNOT_REMOVE_FILE;
    auto &buffer = Info::getBuffer(*this);
    if (buffer.file() == nullptr) {
      if (Info::removeExistingBeforeWrite) {
        buffer.openFile(fileName, "rb");
        if (buffer.file() != nullptr) {
          buffer.closeFile();
          if (remove(fileName)) {
            fail(CANNOT_REMOVE_FILE, CURRENT_LOCATION);
          }
        }
      }
      buffer.openFile(fileName, Info::openWriteMode);
    }
  }

  /**
   * @brief Flush the current in-memory tape buffer to disk.
   */
  template <InfoType<TapeRecordingContext, ErrorType> Info>
  void put_block(const char *fileName, size_t lengthBlock) {
    using ADOLC::detail::write;
    using ADOLCError::fail;
    using ADOLCError::ErrorType::TAPING_FATAL_IO_ERROR;

    openFile<Info>(fileName);
    const size_t numChunks = lengthBlock / Info::chunkSize;

    for (size_t chunk = 0; chunk < numChunks; chunk++) {
      auto returnCode = write<TapeRecordingContext, ErrorType, Info>(
          *this, chunk, Info::chunkSize);
      if (returnCode != 1)
        fail(TAPING_FATAL_IO_ERROR, CURRENT_LOCATION);
    }

    const size_t remain = lengthBlock % Info::chunkSize;
    if (remain != 0) {
      auto returnCode = write<TapeRecordingContext, ErrorType, Info>(
          *this, numChunks, remain);
      if (returnCode != 1)
        fail(TAPING_FATAL_IO_ERROR, CURRENT_LOCATION);
    }

    auto &buffer = Info::getBuffer(*this);
    buffer.numOnTape(buffer.numOnTape() + lengthBlock);
    buffer.position(0);
  }

  /****************************************************************************/
  /* Write some constants to the buffer without disk access                   */
  /****************************************************************************/
  void put_vals_notWriteBlock(double *vals, size_t numVals) {
    for (size_t i = 0; i < numVals; ++i) {
      valBuffer_.writeAndAdvance(vals[i]);
    }
  }
  void put_op(OPCODES op, const char *loc_fileName, const char *op_fileName,
              const char *val_fileName, size_t reserveExtraLocations = 0);
  void put_vals_writeBlock(double *vals, size_t numVals,
                           const char *op_fileName, const char *val_fileName);
  size_t get_val_space(const char *op_fileName, const char *val_fileName);

  // writes a single element (x) to the taylor buffer and writes the buffer
  // to disk if necessary
  void write_scaylor(double val, const char *tay_fileName) {
    using TayInfoT = ADOLC::detail::TayInfo<TapeRecordingContext, ErrorType>;
    if (tayBuffer_.position() == tayBuffer_.capacity())
      put_block<TayInfoT>(tay_fileName, tayBuffer_.capacity());
    tayBuffer_.writeAndAdvance(val);
  }

  /* Write_scaylors writes # size elements from x to the taylor buffer.       */
  /****************************************************************************/
  void write_scaylors(const double *taylorCoefficientPos, std::ptrdiff_t size,
                      const char *tay_fileName);

  /****************************************************************************/
  /* Update locations tape to remove assignments involving temp. variables.   */
  /* e.g.  t = a + b ; y = t  =>  y = a + b                                   */
  /****************************************************************************/
  int upd_resloc(size_t temp, size_t lhs) {
    // LocBuffer points to the first entry of the Locations and CurrLoc-1 to the
    // last placed location in the buffer. Thus, the check ask if there is no
    // element on the tape.
    if (locBuffer_.position() < 1)
      return 0;
    if (temp == locBuffer_[locBuffer_.position() - 1]) {
      locBuffer_[locBuffer_.position() - 1] = lhs;
      return 1;
    }
    return 0;
  }

  int upd_resloc_check(const size_t temp) {
    // LocBuffer points to the first entry of the Locations and CurrLoc-1 to the
    // last placed location in the buffer. Thus, the check ask if there is no
    // element on the tape.
    if (locBuffer_.position() < 1)
      return 0;
    // checks if tape-element represented by "tmp" is the last created.
    if (temp == locBuffer_[locBuffer_.position() - 1]) {
      return 1;
    }
    return 0;
  }

  /****************************************************************************/
  /* Update locations and operations tape to remove special operations inv.   */
  /* temporary variables. e.g.  t = a * b ; y += t  =>  y += a * b            */
  /****************************************************************************/
  int upd_resloc_inc_prod(size_t temp, size_t newlhs, unsigned char newop) {
    if (locBuffer_.position() < 3)
      return 0;
    if (opBuffer_.position() < 1)
      return 0;
    if (temp == locBuffer_[locBuffer_.position() - 1] &&
        mult_a_a == opBuffer_[opBuffer_.position() - 1] &&
        /* skipping recursive case */
        newlhs != locBuffer_[locBuffer_.position() - 2] &&
        newlhs != locBuffer_[locBuffer_.position() - 3]) {
      locBuffer_[locBuffer_.position() - 1] = newlhs;
      opBuffer_[opBuffer_.position() - 1] = newop;
      return 1;
    }
    return 0;
  }
};

} // namespace ADOLC::detail

#endif // ADOLC_TAPE_RECORDING_CONTEXT_H

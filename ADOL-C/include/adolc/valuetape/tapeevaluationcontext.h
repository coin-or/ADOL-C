#ifndef ADOLC_TAPE_EVALUATION_CONTEXT_H
#define ADOLC_TAPE_EVALUATION_CONTEXT_H

#include <adolc/valuetape/tapeinfos.h>
#include <adolc/valuetape/taperecordingcontext.h>
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <type_traits>
#include <utility>
#include <variant>

namespace ADOLC::detail {

struct TapeEvaluationContext;

template <typename T>
concept EvalOrRecordContextType = std::is_same_v<T, TapeEvaluationContext> ||
                                  std::is_same_v<T, TapeRecordingContext>;

/**
 * @brief Per-sweep state used while evaluating a recorded tape.
 *
 * In exclusive mode the context temporarily owns the recording buffers and
 * returns them when the sweep ends. In shared mode it initially views those
 * buffers and allocates private storage lazily when a sweep must overwrite
 * data.
 */
struct TapeEvaluationContext {
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

  ~TapeEvaluationContext() {
    if (originCtx_ != nullptr) {
      closeSweepFiles();
      releaseTo(*originCtx_);
    }
    delete[] signature;
    delete[] paramstore;
  }

  TapeEvaluationContext() = delete;

  explicit TapeEvaluationContext(
      const TapeRecordingContext &tapeCtx, TapeInfos::StatArray stats,
      std::shared_lock<std::shared_mutex> &&lock,
      std::shared_lock<std::shared_mutex> &&dataAccessLock)
      : lock_(std::move(lock)), dataAccessLock_(std::move(dataAccessLock)) {
    copyData(tapeCtx, stats);
  }

  explicit TapeEvaluationContext(
      TapeRecordingContext &&other, std::unique_lock<std::shared_mutex> &&lock,
      std::shared_lock<std::shared_mutex> &&dataAccessLock) noexcept
      : lock_(std::move(lock)), dataAccessLock_(std::move(dataAccessLock)),
        originCtx_(&other) {
    moveData(std::move(other));
  }

  TapeEvaluationContext(const TapeEvaluationContext &) = delete;
  TapeEvaluationContext &operator=(const TapeEvaluationContext &) = delete;

  TapeEvaluationContext(TapeEvaluationContext &&other) noexcept
      : lock_(std::move(other.lock_)),
        dataAccessLock_(std::move(other.dataAccessLock_)),
        originCtx_(std::exchange(other.originCtx_, nullptr)) {
    moveData(std::move(other));
  }

  TapeEvaluationContext &
  operator=(TapeEvaluationContext &&other) noexcept = delete;

  ADOLC::detail::OpBuffer opBuffer_{};
  ADOLC::detail::ValBuffer valBuffer_{};
  ADOLC::detail::LocBuffer locBuffer_{};
  ADOLC::detail::TayBuffer tayBuffer_{};

  size_t numInds{0};
  size_t numDeps{0};
  int keepTaylors{0};
  size_t num_eq_prod{0};
  int deg_save{0};
  size_t tay_numInds{0};
  size_t tay_numDeps{0};
  size_t numSwitches{0};
  bool nestedReverseEval{false};
  size_t nextBufferNumber{0};
  char lastTayBlockInCore{0};
  double *signature{nullptr};
  double *paramstore{nullptr};

private:
  using Lock = std::variant<std::unique_lock<std::shared_mutex>,
                            std::shared_lock<std::shared_mutex>>;

  Lock lock_;
  std::shared_lock<std::shared_mutex> dataAccessLock_;
  // used to recover push data back to recordCtx if an exception happens.
  TapeRecordingContext *originCtx_{nullptr};

  void copyData(const TapeRecordingContext &other, TapeInfos::StatArray stats) {
    opBuffer_ = OpBuffer{other.opBuffer_, bufferView};
    valBuffer_ = ValBuffer{other.valBuffer_, bufferView};
    locBuffer_ = LocBuffer{other.locBuffer_, bufferView};
    tayBuffer_ = TayBuffer{other.tayBuffer_, bufferView};

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

    if (other.signature != nullptr) {
      signature = new double[stats[TapeInfos::NUM_SWITCHES]];
      std::copy_n(other.signature, stats[TapeInfos::NUM_SWITCHES], signature);
    }
    if (other.paramstore != nullptr) {
      paramstore = new double[stats[TapeInfos::NUM_PARAM]];
      std::copy_n(other.paramstore, stats[TapeInfos::NUM_PARAM], paramstore);
    }
  }

  template <EvalOrRecordContextType Context>
  void moveData(Context &&other) noexcept {
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
    signature = std::exchange(other.signature, nullptr);
    paramstore = std::exchange(other.paramstore, nullptr);
  }

public:
  void closeSweepFiles() {
    opBuffer_.closeFile();
    locBuffer_.closeFile();
    valBuffer_.closeFile();
  }

  void releaseTo(TapeRecordingContext &recordCtx) {
    recordCtx.opBuffer_ = std::move(opBuffer_);
    recordCtx.valBuffer_ = std::move(valBuffer_);
    recordCtx.locBuffer_ = std::move(locBuffer_);
    recordCtx.tayBuffer_ = std::move(tayBuffer_);

    recordCtx.numInds = numInds;
    recordCtx.numDeps = numDeps;
    recordCtx.keepTaylors = keepTaylors;
    recordCtx.num_eq_prod = num_eq_prod;
    recordCtx.deg_save = deg_save;
    recordCtx.tay_numInds = tay_numInds;
    recordCtx.tay_numDeps = tay_numDeps;
    recordCtx.numSwitches = numSwitches;
    recordCtx.nestedReverseEval = nestedReverseEval;
    recordCtx.nextBufferNumber = nextBufferNumber;
    recordCtx.lastTayBlockInCore = lastTayBlockInCore;

    assert(recordCtx.signature == nullptr);
    recordCtx.signature = std::exchange(signature, nullptr);
    assert(recordCtx.paramstore == nullptr);
    recordCtx.paramstore = std::exchange(paramstore, nullptr);

    originCtx_ = nullptr;
  }

  template <InfoType<TapeEvaluationContext, ErrorType> Info>
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

  template <InfoType<TapeEvaluationContext, ErrorType> Info>
  void put_block(const char *fileName, size_t lengthBlock) {
    using ADOLC::detail::write;
    using ADOLCError::fail;
    using ADOLCError::ErrorType::TAPING_FATAL_IO_ERROR;

    openFile<Info>(fileName);
    const size_t numChunks = lengthBlock / Info::chunkSize;

    for (size_t chunk = 0; chunk < numChunks; chunk++) {
      auto returnCode = write<TapeEvaluationContext, ErrorType, Info>(
          *this, chunk, Info::chunkSize);
      if (returnCode != 1)
        fail(TAPING_FATAL_IO_ERROR, CURRENT_LOCATION);
    }

    const size_t remain = lengthBlock % Info::chunkSize;
    if (remain != 0) {
      auto returnCode = write<TapeEvaluationContext, ErrorType, Info>(
          *this, numChunks, remain);
      if (returnCode != 1)
        fail(TAPING_FATAL_IO_ERROR, CURRENT_LOCATION);
    }

    auto &buffer = Info::getBuffer(*this);
    buffer.numOnTape(buffer.numOnTape() + lengthBlock);
    buffer.position(0);
  }

  template <InfoType<TapeEvaluationContext, ErrorType> Info>
  void loadBlockIntoBuffer(size_t blockSize) {
    using ADOLCError::fail;
    auto &buffer = Info::getBuffer(*this);
    if (!buffer.isOwner()) {
      buffer.allocateAndOwn();
    }
    const size_t numChunks = blockSize / Info::chunkSize;
    for (size_t chunk = 0; chunk < numChunks; chunk++) {
      const auto ret =
          fread(buffer.begin() + (chunk * Info::chunkSize),
                Info::chunkSize * sizeof(typename Info::value_type), 1,
                buffer.file());
      if (ret != 1) {
        fail(Info::error, CURRENT_LOCATION);
      }
    }
    const size_t remain = blockSize % Info::chunkSize;
    if (remain != 0) {
      const auto ret =
          fread(buffer.begin() + (numChunks * Info::chunkSize),
                remain * sizeof(typename Info::value_type), 1, buffer.file());
      if (ret != 1) {
        fail(Info::error, CURRENT_LOCATION);
      }
    }
  }

  template <InfoType<TapeEvaluationContext, ErrorType> Info>
  void loadBlockIntoBufferForward(size_t bufferSize) {
    auto &buffer = Info::getBuffer(*this);
    const size_t blockSize = std::min(bufferSize, buffer.numOnTape());
    loadBlockIntoBuffer<Info>(blockSize);
    Info::updateBufferStatsForward(*this, blockSize);
    Info::updateBufferPositionForward(*this);
  }

  template <InfoType<TapeEvaluationContext, ErrorType> Info>
  void loadBlockIntoBufferReverse(size_t blockSize) {
    using ADOLCError::fail;

    auto &buffer = Info::getBuffer(*this);
    const long pos = Info::reverseSeekOffset(*this, blockSize);
    const auto ret = fseek(buffer.file(), pos, SEEK_SET);
    if (ret == -1) {
      fail(Info::error, CURRENT_LOCATION);
    }

    loadBlockIntoBuffer<Info>(blockSize);
    Info::updateBufferStatsReverse(*this, blockSize);
    Info::updateBufferPositionReverse(*this, blockSize);
  }

  template <InfoType<TapeEvaluationContext, ErrorType> Info>
  typename Info::value_type loadNextReverse(size_t blockSize) {
    auto &buffer = Info::getBuffer(*this);
    Info::ensureReverseReadable(*this, blockSize);
    return buffer.retreatAndRead();
  }

  template <InfoType<TapeEvaluationContext, ErrorType> Info>
  typename Info::value_type loadNextForward() {
    auto &buffer = Info::getBuffer(*this);
    return buffer.readAndAdvance();
  }

  double *get_val_v_f(size_t size) {
    double *temp = valBuffer_.current();
    valBuffer_.position(valBuffer_.position() + size);
    return temp;
  }

  double *get_val_v_r(size_t size) {
    valBuffer_.position(valBuffer_.position() - size);
    return valBuffer_.current();
  }

  void reset_val_r(size_t valueBufferSize) {
    using ValInfoT = ADOLC::detail::ValInfo<TapeEvaluationContext, ErrorType>;
    if (valBuffer_.position() == 0) {
      loadBlockIntoBufferReverse<ValInfoT>(valueBufferSize);
    }
  }

  void discard_params_r(size_t valueBufferSize, size_t numParam);

  void finish_tay_file(const char *tay_fileName) {
    deg_save = -1;
    if (tayBuffer_.file() != nullptr)
      tayBuffer_.closeFile();
    remove(tay_fileName);
  }

  void taylor_begin(int degreeSave, size_t taylorBufferSize,
                    const char *tay_fileName) {
    if (tayBuffer_.begin()) {
      finish_tay_file(tay_fileName);
    } else {
      tayBuffer_.allocIfNull(taylorBufferSize);
    }

    deg_save = degreeSave;
    if (degreeSave >= 0)
      keepTaylors = 1;
    tayBuffer_.position(0);
    tayBuffer_.numOnTape(0);
  }

  size_t taylor_close(const char *tay_fileName) {
    using TayInfoT = ADOLC::detail::TayInfo<TapeEvaluationContext, ErrorType>;
    if (tayBuffer_.file() != nullptr) {
      if (keepTaylors != 0) {
        put_block<TayInfoT>(tay_fileName, tayBuffer_.position());
        if (std::fflush(tayBuffer_.file()) != 0)
          ADOLCError::fail(ErrorType::TAPING_FATAL_IO_ERROR, CURRENT_LOCATION);
      }
    } else {
      tayBuffer_.numOnTape(tayBuffer_.position());
    }
    lastTayBlockInCore = 1;
    return tayBuffer_.numOnTape();
  }

  void taylor_back(size_t taylorBufferSize, short tapeId,
                   const char *tay_fileName);

  void write_taylor(double *taylorCoefficientPos, std::ptrdiff_t keep,
                    const char *tay_fileName);

  void write_scaylor(double val, const char *tay_fileName) {
    using TayInfoT = ADOLC::detail::TayInfo<TapeEvaluationContext, ErrorType>;
    if (tayBuffer_.position() == tayBuffer_.capacity())
      put_block<TayInfoT>(tay_fileName, tayBuffer_.capacity());
    tayBuffer_.writeAndAdvance(val);
  }

  void write_taylors(double *taylorCoefficientPos, int keep, int degree,
                     int numDir, const char *tay_fileName) {
    using TayInfoT = ADOLC::detail::TayInfo<TapeEvaluationContext, ErrorType>;
    for (int j = 0; j < numDir; ++j) {
      for (int i = 0; i < keep; ++i) {
        if (tayBuffer_.position() == tayBuffer_.capacity())
          put_block<TayInfoT>(tay_fileName, tayBuffer_.capacity());

        tayBuffer_.writeAndAdvance(*taylorCoefficientPos);
        ++taylorCoefficientPos;
      }
      if (degree > keep)
        taylorCoefficientPos += degree - keep;
    }
  }

  void get_taylors(double *taylorCoefficients, std::ptrdiff_t degree,
                   size_t taylorBufferSize);

  void get_taylors_p(double *taylorCoefficients, int degree, int numDir,
                     size_t taylorBufferSize);
};

} // namespace ADOLC::detail

#endif // ADOLC_TAPE_EVALUATION_CONTEXT_H

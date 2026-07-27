#ifndef ADOLC_INFO_TYPE_H
#define ADOLC_INFO_TYPE_H

#include <adolc/adolcerror.h>
#include <adolc/dvlparms.h>
#include <adolc/internal/usrparms.h> // ADOLC_IO_CHUNK_SIZE
#include <adolc/valuetape/bufferstate.h>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdio>
#include <type_traits>
#include <utility>

/*
  This header defines the small policy interface used by ADOL-C tape I/O code.

  The Info structs (OpInfo, LocInfo, ValInfo, TayInfo) are stateless adapters:
    - they select the concrete buffer stored in a buffer context,
    - they name the metadata entries that higher-level code stores in TapeInfos,
    - they provide tape-specific hooks for forward/reverse sweeps.

  The important split is intentional: Info adapters operate on a buffer context,
  while metadata values are passed as an explicit stats array to the operations
  that need them.
*/

namespace ADOLC::detail {

template <class Info, class BufferContext>
using InfoBufferType = std::remove_cvref_t<decltype(Info::getBuffer(
    std::declval<BufferContext &>()))>;

/**
 * @brief Concept describing the required interface of a tape Info adapter.
 *
 * @tparam T             The adapter type (e.g. OpInfo<Context, EType>).
 * @tparam BufferContext The context type that stores buffers, counters, file
 *                       handles, etc.
 * @tparam ErrorType     The error enum/type used by higher-level code.
 */
template <class T, class BufferContext, class ErrorType>
concept InfoType =
    requires(BufferContext &context, const BufferContext &constContext,
             size_t blockSize) {
      typename T::value_type;
      requires BufferStateType<InfoBufferType<T, BufferContext>,
                               typename T::value_type>;

      { T::fileIndex } -> std::convertible_to<size_t>;
      { T::num } -> std::same_as<const typename BufferContext::StatEntries &>;
      { T::error } -> std::convertible_to<ErrorType>;
      { T::chunkSize } -> std::convertible_to<size_t>;
      { T::removeExistingBeforeWrite } -> std::convertible_to<bool>;
      { T::openWriteMode } -> std::convertible_to<const char *>;

      {
        T::getBuffer(context)
      } -> std::same_as<InfoBufferType<T, BufferContext> &>;
      {
        T::getBuffer(constContext)
      } -> std::same_as<const InfoBufferType<T, BufferContext> &>;

      T::ensureReverseReadable(context, blockSize);
      {
        T::reverseSeekOffset(constContext, blockSize)
      } -> std::convertible_to<long>;
      T::updateBufferStatsForward(context, blockSize);
      T::updateBufferStatsReverse(context, blockSize);
      T::updateBufferPositionForward(context);
      T::updateBufferPositionReverse(context, blockSize);
      T::prepareForwardPosition(context, blockSize);
    };

/**
 * @brief Common compile-time interface for operation, location, value, and
 * Taylor tape buffers.
 *
 * Derived adapters provide the format-specific buffer, counters, and I/O
 * policy while this base supplies the shared cursor operations.
 */
template <class Derived, class BufferContext, typename ValueType>
struct InfoAdapter {
  using value_type = ValueType;
  using StatEntries = typename BufferContext::StatEntries;

  static constexpr bool removeExistingBeforeWrite = true;
  static constexpr const char *openWriteMode = "wb";

  static void ensureReverseReadable(BufferContext & /*context*/,
                                    size_t /*blockSize*/) {}

  static long reverseSeekOffset(const BufferContext &context,
                                size_t blockSize) {
    const auto &buffer = Derived::getBuffer(context);
    return static_cast<long>(sizeof(value_type) *
                             (buffer.numOnTape() - blockSize));
  }

  static void updateBufferStatsForward(BufferContext &context,
                                       size_t blockSize) {
    auto &buffer = Derived::getBuffer(context);
    buffer.numOnTape(buffer.numOnTape() - blockSize);
  }

  static void updateBufferStatsReverse(BufferContext &context,
                                       size_t blockSize) {
    auto &buffer = Derived::getBuffer(context);
    buffer.numOnTape(buffer.numOnTape() - blockSize);
  }

  static void updateBufferPositionForward(BufferContext &context) {
    Derived::getBuffer(context).position(0);
  }

  static void updateBufferPositionReverse(BufferContext &context,
                                          size_t blockSize) {
    Derived::getBuffer(context).position(blockSize);
  }

  static void prepareForwardPosition(BufferContext &context,
                                     size_t /*blockSize*/) {
    Derived::getBuffer(context).position(0);
  }
};

/**
 * @brief Adapter for the operations tape (op tape).
 */
template <class BufferContext, class EType>
struct OpInfo
    : InfoAdapter<OpInfo<BufferContext, EType>, BufferContext, unsigned char> {
  using Self = OpInfo<BufferContext, EType>;
  using Base = InfoAdapter<Self, BufferContext, unsigned char>;
  using value_type = typename Base::value_type;
  using StatEntries = typename Base::StatEntries;

  static constexpr StatEntries num = BufferContext::NUM_OPERATIONS;
  static constexpr StatEntries fileAccess = BufferContext::OP_FILE_ACCESS;
  static constexpr StatEntries bufferSize = BufferContext::OP_BUFFER_SIZE;
  static constexpr size_t fileIndex = 0;

  static constexpr EType error = EType::OP_READ_FAILED;
  static constexpr size_t chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(value_type);

  static OpBuffer &getBuffer(BufferContext &context) {
    return context.opBuffer_;
  }
  static const OpBuffer &getBuffer(const BufferContext &context) {
    return context.opBuffer_;
  }
};

/**
 * @brief Adapter for the locations tape (loc tape).
 */
template <class BufferContext, class EType>
struct LocInfo
    : InfoAdapter<LocInfo<BufferContext, EType>, BufferContext, size_t> {
  using Self = LocInfo<BufferContext, EType>;
  using Base = InfoAdapter<Self, BufferContext, size_t>;
  using value_type = typename Base::value_type;
  using StatEntries = typename Base::StatEntries;

  static constexpr StatEntries num = BufferContext::NUM_LOCATIONS;
  static constexpr StatEntries fileAccess = BufferContext::LOC_FILE_ACCESS;
  static constexpr StatEntries bufferSize = BufferContext::LOC_BUFFER_SIZE;
  static constexpr size_t fileIndex = 1;

  static constexpr EType error = EType::LOC_READ_FAILED;
  static constexpr size_t chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(value_type);

  static LocBuffer &getBuffer(BufferContext &context) {
    return context.locBuffer_;
  }
  static const LocBuffer &getBuffer(const BufferContext &context) {
    return context.locBuffer_;
  }

  static void updateBufferPositionReverse(BufferContext &context,
                                          size_t blockSize) {
    auto &buffer = getBuffer(context);
    const auto loc = blockSize - buffer[blockSize - 1];
    buffer.position(loc);
  }

  static void prepareForwardPosition(BufferContext &context, size_t blockSize) {
    size_t numLocsForStats = statSpace;
    while (numLocsForStats >= blockSize) {
      context.template loadBlockIntoBufferForward<Self>(blockSize);
      numLocsForStats -= blockSize;
    }
    getBuffer(context).position(numLocsForStats);
  }
};

/**
 * @brief Adapter for the values tape (val tape).
 */
template <class BufferContext, class EType>
struct ValInfo
    : InfoAdapter<ValInfo<BufferContext, EType>, BufferContext, double> {
  using Self = ValInfo<BufferContext, EType>;
  using Base = InfoAdapter<Self, BufferContext, double>;
  using value_type = typename Base::value_type;
  using StatEntries = typename Base::StatEntries;

  static constexpr StatEntries num = BufferContext::NUM_VALUES;
  static constexpr StatEntries fileAccess = BufferContext::VAL_FILE_ACCESS;
  static constexpr StatEntries bufferSize = BufferContext::VAL_BUFFER_SIZE;
  static constexpr size_t fileIndex = 2;

  static constexpr EType error = EType::VAL_READ_FAILED;
  static constexpr size_t chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(value_type);

  static ValBuffer &getBuffer(BufferContext &context) {
    return context.valBuffer_;
  }
  static const ValBuffer &getBuffer(const BufferContext &context) {
    return context.valBuffer_;
  }

  static void updateBufferPositionForward(BufferContext &context) {
    getBuffer(context).position(0);
    context.locBuffer_.advance();
  }

  static void updateBufferPositionReverse(BufferContext &context,
                                          size_t blockSize) {
    getBuffer(context).position(blockSize -
                                context.locBuffer_.retreatAndRead());
  }
};

/**
 * @brief Adapter for the Taylor tape (tay tape).
 */
template <class BufferContext, class EType>
struct TayInfo
    : InfoAdapter<TayInfo<BufferContext, EType>, BufferContext, double> {
  using Self = TayInfo<BufferContext, EType>;
  using Base = InfoAdapter<Self, BufferContext, double>;
  using value_type = typename Base::value_type;
  using StatEntries = typename Base::StatEntries;

  static constexpr StatEntries num = BufferContext::NUM_TAYS;
  static constexpr StatEntries bufferSize = BufferContext::TAY_BUFFER_SIZE;
  static constexpr size_t fileIndex = 3;
  static constexpr bool removeExistingBeforeWrite = false;
  static constexpr const char *openWriteMode = "w+b";

  static constexpr EType error = EType::TAY_READ_FAILED;
  static constexpr size_t chunkSize = ADOLC_IO_CHUNK_SIZE / sizeof(value_type);

  static TayBuffer &getBuffer(BufferContext &context) {
    return context.tayBuffer_;
  }
  static const TayBuffer &getBuffer(const BufferContext &context) {
    return context.tayBuffer_;
  }

  static void ensureReverseReadable(BufferContext &context, size_t blockSize) {
    if (getBuffer(context).position() == 0) {
      context.template loadBlockIntoBufferReverse<Self>(blockSize);
    }
  }

  static long reverseSeekOffset(const BufferContext &context,
                                size_t blockSize) {
    return static_cast<long>(sizeof(value_type) *
                             (context.nextBufferNumber * blockSize));
  }

  static void updateBufferStatsReverse(BufferContext &context,
                                       size_t /*blockSize*/) {
    context.lastTayBlockInCore = 0;
    --context.nextBufferNumber;
  }
};

/// Wrapper of fwrite.
template <typename BufferContext, typename ErrorType,
          InfoType<BufferContext, ErrorType> Info>
static size_t write(BufferContext &context, size_t chunk, size_t size) {
  auto &buffer = Info::getBuffer(context);
  return fwrite(buffer.begin() + (chunk * Info::chunkSize),
                size * sizeof(typename Info::value_type), 1, buffer.file());
}

} // namespace ADOLC::detail

#endif // ADOLC_INFO_TYPE_H
